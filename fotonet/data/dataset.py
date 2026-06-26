import os
import hashlib
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import cv2
except Exception:
    cv2 = None

from fotonet.data.augmentations import (
    build_augment_hyp,
    copy_paste,
    finalize_sample,
    letterbox_resize,
    load_mosaic,
    mixup,
    random_affine,
    sanitize_boxes_xywhn,
    to_tensor,
)


class FOTONETDataset(Dataset):
    """
    Dataset for YOLO-format object detection data.
    Uses an on-demand bounded RAM image cache so augmentation runs against
    decoded images instead of repeatedly hitting the filesystem.
    """

    def __init__(
        self,
        img_dir,
        imgsz=640,
        augment=False,
        cache_labels=False,
        cache_to_ram=False,
        ram_cache_images=1024,
        disk_cache_images=False,
        disk_cache_dir=None,
        label_cache_workers=None,
        label_cache_dir=None,
        augment_hyp=None,
        num_classes=None,
    ):
        self.imgsz = imgsz
        self.augment = augment
        self.epoch = 0
        self.max_epochs = None
        self.cache = {}
        self.img_files = []
        self.cache_to_ram = bool(cache_to_ram)
        self.ram_cache_images = max(int(ram_cache_images or 0), 0)
        self.disk_cache_images = bool(disk_cache_images)
        self.disk_cache_dir = disk_cache_dir
        default_label_workers = min(max((os.cpu_count() or 1), 1), 8)
        self.label_cache_workers = max(int(label_cache_workers if label_cache_workers is not None else default_label_workers), 1)
        self.label_cache_dir = label_cache_dir
        self.augment_hyp = build_augment_hyp(augment_hyp if augment else {})
        self.num_classes = int(num_classes) if num_classes is not None else None
        self._image_cache = OrderedDict()
        self._label_dirs_by_image_dir = {}

        if isinstance(img_dir, list):
            if img_dir and os.path.isfile(img_dir[0]):
                self.img_files = [p for p in img_dir if os.path.isfile(p)]
            else:
                for d in img_dir:
                    abs_d = os.path.abspath(d)
                    if os.path.isdir(abs_d):
                        valid = [
                            os.path.join(abs_d, f)
                            for f in os.listdir(abs_d)
                            if f.lower().endswith((".png", ".jpg", ".jpeg"))
                        ]
                        self.img_files.extend(valid)
        elif isinstance(img_dir, str):
            if img_dir.endswith(".txt") and os.path.isfile(img_dir):
                with open(img_dir, "r") as f:
                    self.img_files = [line.strip() for line in f.readlines() if line.strip()]
            else:
                abs_d = os.path.abspath(img_dir) if not os.path.isabs(img_dir) else img_dir
                if os.path.isdir(abs_d):
                    self.img_files = [
                        os.path.join(abs_d, f)
                        for f in os.listdir(abs_d)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))
                    ]

        self.img_files = sorted(list(set(self.img_files)))
        self._label_dirs_by_image_dir = self._detect_label_dirs()
        print(
            f"[INFO] FOTONETDataset: Found {len(self.img_files)} images. "
            f"augment={augment} cache_to_ram={self.cache_to_ram} ram_cache_images={self.ram_cache_images} "
            f"disk_cache_images={self.disk_cache_images}",
            flush=True,
        )

        if cache_labels:
            self._cache_all_labels()

    def _detect_label_dirs(self):
        label_dirs = {}
        for img_path in self.img_files:
            img_dir = os.path.dirname(os.path.abspath(img_path))
            if img_dir in label_dirs:
                continue

            base = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            candidates = []

            parts = img_dir.split(os.sep)
            if "images" in parts:
                image_index = parts.index("images")
                label_parts = parts[:]
                label_parts[image_index] = "labels"
                candidates.append(os.sep.join(label_parts))

            parent = os.path.dirname(img_dir)
            subset = os.path.basename(img_dir)
            candidates.extend(
                [
                    os.path.join(parent, "labels", subset),
                    os.path.join(parent, "coco", "labels", subset),
                    os.path.join(parent, "Annotations", subset),
                ]
            )

            for candidate in candidates:
                if os.path.isdir(candidate):
                    label_dirs[img_dir] = candidate
                    break
        return label_dirs

    def _get_label_path(self, img_path):
        img_dir = os.path.dirname(os.path.abspath(img_path))
        label_dir = self._label_dirs_by_image_dir.get(img_dir)
        if label_dir:
            return os.path.join(label_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")

        from fotonet.utils.config import get_label_path

        return get_label_path(img_path)

    @staticmethod
    def _empty_label_result():
        return {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
        }

    @staticmethod
    def _parse_label_file(label_path, num_classes=None):
        boxes = []
        labels = []
        try:
            exists = os.path.exists(label_path)
        except OSError:
            exists = False

        if exists:
            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except OSError:
                lines = []

            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    cls, x, y, bw, bh = map(float, parts)
                except ValueError:
                    continue
                values = np.asarray([cls, x, y, bw, bh], dtype=np.float32)
                if not np.isfinite(values).all():
                    continue
                if not float(cls).is_integer():
                    continue
                cls_i = int(cls)
                if cls_i < 0 or (num_classes is not None and cls_i >= num_classes):
                    continue
                if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
                    continue
                if bw <= 0.0 or bw > 1.0 or bh <= 0.0 or bh > 1.0:
                    continue
                labels.append(cls_i)
                boxes.append([x, y, bw, bh])

        if not boxes:
            return FOTONETDataset._empty_label_result()
        return {
            "boxes": np.asarray(boxes, dtype=np.float32),
            "labels": np.asarray(labels, dtype=np.int64),
        }

    @staticmethod
    def _parse_label_item(args):
        idx, label_path, num_classes = args
        return idx, FOTONETDataset._parse_label_file(label_path, num_classes)

    @staticmethod
    def _label_signature(label_path):
        try:
            stat = os.stat(label_path)
        except OSError:
            return (0, 0, 0)
        return (1, int(stat.st_size), int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))))

    def _label_cache_root(self):
        if self.label_cache_dir:
            cache_dir = self.label_cache_dir
        elif self.img_files:
            common = os.path.commonpath([os.path.abspath(p) for p in self.img_files])
            if not os.path.isdir(common):
                common = os.path.dirname(common)
            cache_dir = os.path.join(common, ".fotonet_cache", "labels")
        else:
            cache_dir = os.path.join(os.getcwd(), ".fotonet_cache", "labels")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _label_cache_path(self, label_paths):
        h = hashlib.sha1()
        h.update(b"fotonet-label-cache-v2\0")
        h.update(str(self.num_classes).encode("utf-8"))
        h.update(b"\0")
        for img_path, label_path in zip(self.img_files, label_paths):
            h.update(os.path.abspath(img_path).encode("utf-8", errors="ignore"))
            h.update(b"\0")
            h.update(os.path.abspath(label_path).encode("utf-8", errors="ignore"))
            h.update(b"\0")
        return os.path.join(self._label_cache_root(), h.hexdigest() + ".npz")

    def _load_label_cache_file(self, cache_path, label_paths):
        if not os.path.exists(cache_path):
            return False
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                version = int(data["version"][0])
                if version != 2:
                    return False
                saved_nc = int(data["num_classes"][0])
                expected_nc = -1 if self.num_classes is None else int(self.num_classes)
                if saved_nc != expected_nc:
                    return False

                saved_images = data["image_paths"].astype(str).tolist()
                saved_labels = data["label_paths"].astype(str).tolist()
                expected_images = [os.path.abspath(p) for p in self.img_files]
                expected_labels = [os.path.abspath(p) for p in label_paths]
                if saved_images != expected_images or saved_labels != expected_labels:
                    return False

                signatures = data["signatures"].astype(np.int64, copy=False)
                current_signatures = np.asarray(
                    [self._label_signature(p) for p in label_paths],
                    dtype=np.int64,
                )
                if signatures.shape != current_signatures.shape or not np.array_equal(signatures, current_signatures):
                    return False

                counts = data["counts"].astype(np.int64, copy=False)
                boxes_all = data["boxes"].astype(np.float32, copy=False)
                labels_all = data["labels"].astype(np.int64, copy=False)
                offset = 0
                self.cache = {}
                for idx, count in enumerate(counts.tolist()):
                    next_offset = offset + int(count)
                    self.cache[idx] = {
                        "boxes": boxes_all[offset:next_offset].copy(),
                        "labels": labels_all[offset:next_offset].copy(),
                    }
                    offset = next_offset
            return len(self.cache) == len(self.img_files)
        except Exception:
            return False

    def _save_label_cache_file(self, cache_path, label_paths):
        counts = np.asarray([len(self.cache[i]["labels"]) for i in range(len(self.img_files))], dtype=np.int64)
        boxes_parts = [self.cache[i]["boxes"] for i in range(len(self.img_files)) if len(self.cache[i]["boxes"]) > 0]
        labels_parts = [self.cache[i]["labels"] for i in range(len(self.img_files)) if len(self.cache[i]["labels"]) > 0]
        boxes_all = np.concatenate(boxes_parts, axis=0).astype(np.float32, copy=False) if boxes_parts else np.zeros((0, 4), dtype=np.float32)
        labels_all = np.concatenate(labels_parts, axis=0).astype(np.int64, copy=False) if labels_parts else np.zeros((0,), dtype=np.int64)
        signatures = np.asarray([self._label_signature(p) for p in label_paths], dtype=np.int64)

        tmp_path = cache_path + ".tmp"
        try:
            with open(tmp_path, "wb") as f:
                np.savez(
                    f,
                    version=np.asarray([2], dtype=np.int64),
                    num_classes=np.asarray([-1 if self.num_classes is None else int(self.num_classes)], dtype=np.int64),
                    image_paths=np.asarray([os.path.abspath(p) for p in self.img_files]),
                    label_paths=np.asarray([os.path.abspath(p) for p in label_paths]),
                    signatures=signatures,
                    counts=counts,
                    boxes=boxes_all,
                    labels=labels_all,
                )
            os.replace(tmp_path, cache_path)
        except OSError:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

    def _cache_all_labels(self):
        n = len(self.img_files)
        label_paths = [self._get_label_path(p) for p in self.img_files]
        cache_path = self._label_cache_path(label_paths)
        start = time.time()
        if self._load_label_cache_file(cache_path, label_paths):
            elapsed = max(time.time() - start, 1e-9)
            rate = n / elapsed if n else 0.0
            print(f"[INFO] Loaded label cache: {n}/{n} ({rate:.0f}/s)", flush=True)
            return

        workers = min(self.label_cache_workers, max(n, 1))
        print(f"[INFO] Caching labels for {n} images with {workers} workers...", flush=True)
        start = time.time()
        if workers <= 1 or n <= 1:
            iterator = map(
                self._parse_label_item,
                ((idx, label_path, self.num_classes) for idx, label_path in enumerate(label_paths)),
            )
        else:
            executor = ThreadPoolExecutor(max_workers=workers)
            iterator = executor.map(
                self._parse_label_item,
                ((idx, label_path, self.num_classes) for idx, label_path in enumerate(label_paths)),
            )

        try:
            for done, (idx, res) in enumerate(iterator, start=1):
                self.cache[idx] = res
                if done % 10000 == 0 or done == n:
                    elapsed = max(time.time() - start, 1e-9)
                    rate = done / elapsed
                    remaining = (n - done) / max(rate, 1e-9)
                    print(f"[INFO] Label cache: {done}/{n} ({rate:.0f}/s, eta={remaining:.1f}s)", flush=True)
        finally:
            if "executor" in locals():
                executor.shutdown(wait=True)

        self._save_label_cache_file(cache_path, label_paths)

    def _get_label(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        img_path = self.img_files[idx]
        label_path = self._get_label_path(img_path)
        res = self._parse_label_file(label_path, self.num_classes)
        self.cache[idx] = res
        return res

    def _disk_cache_path(self, img_path):
        if not self.disk_cache_dir:
            root = os.path.dirname(os.path.dirname(os.path.abspath(img_path)))
            cache_dir = os.path.join(root, ".fotonet_cache", "decoded_rgb")
        else:
            cache_dir = self.disk_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        stat = os.stat(img_path)
        key = f"{os.path.abspath(img_path)}|{stat.st_size}|{int(stat.st_mtime)}"
        name = hashlib.sha1(key.encode("utf-8")).hexdigest() + ".npy"
        return os.path.join(cache_dir, name)

    def _decode_image(self, img_path):
        if cv2 is not None:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Failed to read image: {img_path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        from PIL import Image

        return np.asarray(Image.open(img_path).convert("RGB"))

    def _read_image(self, idx):
        img_path = self.img_files[idx]
        if not self.disk_cache_images:
            return self._decode_image(img_path)

        cache_path = self._disk_cache_path(img_path)
        if os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

        img = self._decode_image(img_path)
        tmp_path = cache_path + ".tmp"
        try:
            with open(tmp_path, "wb") as f:
                np.save(f, img, allow_pickle=False)
            os.replace(tmp_path, cache_path)
        except OSError:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
        return img

    def _get_cached_image(self, idx):
        if not self.cache_to_ram or self.ram_cache_images <= 0:
            return self._read_image(idx)

        cached = self._image_cache.get(idx)
        if cached is not None:
            self._image_cache.move_to_end(idx)
            return cached

        img = self._read_image(idx)
        self._image_cache[idx] = img
        self._image_cache.move_to_end(idx)
        while len(self._image_cache) > self.ram_cache_images:
            self._image_cache.popitem(last=False)
        return img

    def __len__(self):
        return len(self.img_files)

    def get_raw(self, idx):
        img = self._get_cached_image(idx)
        target = self._get_label(idx)
        return img, {"boxes": target["boxes"].copy(), "labels": target["labels"].copy()}

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_total_epochs(self, max_epochs):
        self.max_epochs = max_epochs

    def set_imgsz(self, imgsz):
        self.imgsz = int(imgsz)

    def _mosaic_enabled(self):
        if not self.augment or self.augment_hyp["mosaic"] <= 0.0:
            return False
        close_mosaic = int(self.augment_hyp.get("close_mosaic", 0) or 0)
        if self.max_epochs is not None and close_mosaic > 0:
            if self.epoch >= max(self.max_epochs - close_mosaic, 0):
                return False
        return True

    def _build_base_sample(self, idx):
        use_mosaic = self._mosaic_enabled() and np.random.random() < self.augment_hyp["mosaic"]
        if use_mosaic:
            img, boxes, labels = load_mosaic(self, idx, self.imgsz)
        else:
            img, target = self.get_raw(idx)
            boxes, labels = target["boxes"], target["labels"]
            img, boxes = letterbox_resize(img, self.imgsz, boxes)

        if self.augment:
            img, boxes, labels = copy_paste(
                img,
                boxes,
                labels,
                p=self.augment_hyp["copy_paste"],
                mode=self.augment_hyp["copy_paste_mode"],
            )
            img, boxes, labels = random_affine(
                img,
                boxes,
                labels,
                degrees=self.augment_hyp["degrees"],
                translate=self.augment_hyp["translate"],
                scale=self.augment_hyp["scale"],
                shear=self.augment_hyp["shear"],
            )
            img, boxes, labels = finalize_sample(
                img,
                boxes,
                labels,
                hsv_h=self.augment_hyp["hsv_h"],
                hsv_s=self.augment_hyp["hsv_s"],
                hsv_v=self.augment_hyp["hsv_v"],
                fliplr=self.augment_hyp["fliplr"],
                flipud=self.augment_hyp["flipud"],
                bgr=self.augment_hyp["bgr"],
            )

        return img, boxes.astype(np.float32, copy=False), labels.astype(np.int64, copy=False)

    def __getitem__(self, idx):
        img, boxes, labels = self._build_base_sample(idx)

        if self.augment and self.augment_hyp["mixup"] > 0.0 and np.random.random() < self.augment_hyp["mixup"] and len(self.img_files) > 1:
            mix_idx = np.random.randint(0, len(self.img_files))
            mix_img, mix_boxes, mix_labels = self._build_base_sample(mix_idx)
            img, boxes, labels = mixup(
                img,
                boxes,
                labels,
                mix_img,
                mix_boxes,
                mix_labels,
                alpha=self.augment_hyp["mixup_alpha"],
            )

        boxes, labels = sanitize_boxes_xywhn(boxes, labels)
        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "image_id": torch.tensor([idx]),
        }
        return to_tensor(img), target
