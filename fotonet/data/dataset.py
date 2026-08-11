import os
import hashlib
import time
from pathlib import Path
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from fotonet.data.audit import AnnotationAudit

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


_IMAGE_SUFFIXES = {".bmp", ".dib", ".jpeg", ".jpg", ".jfif", ".png", ".tif", ".tiff", ".webp"}
_MANIFEST_SUFFIXES = {".list", ".lst", ".txt"}


def _as_path(value):
    try:
        return os.path.abspath(os.path.expanduser(os.fspath(value)))
    except TypeError as exc:
        raise TypeError(f"Dataset source must be a path or a sequence of paths, got {type(value).__name__}") from exc


def _is_image_path(path):
    return Path(path).suffix.lower() in _IMAGE_SUFFIXES


def expand_image_sources(source, recursive=True):
    """Expand mixed directories, image files, and manifests deterministically.

    Manifest entries are resolved relative to the manifest file, which is the
    behavior users expect when moving a dataset directory.  Unlike the old
    first-item heuristic, every member of a list is expanded independently.
    """
    seen_manifests = set()
    expanded = []

    def add(item, base_dir=None):
        if isinstance(item, (list, tuple, set)):
            for child in item:
                add(child, base_dir=base_dir)
            return

        path_text = os.fspath(item)
        if base_dir is not None and not os.path.isabs(path_text):
            path_text = os.path.join(base_dir, path_text)
        path = _as_path(path_text)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset source does not exist: {path}")

        if os.path.isdir(path):
            if recursive:
                for root, dirs, files in os.walk(path):
                    dirs.sort()
                    for name in sorted(files):
                        candidate = os.path.join(root, name)
                        if _is_image_path(candidate):
                            expanded.append(candidate)
            else:
                for name in sorted(os.listdir(path)):
                    candidate = os.path.join(path, name)
                    if os.path.isfile(candidate) and _is_image_path(candidate):
                        expanded.append(candidate)
            return

        suffix = Path(path).suffix.lower()
        if suffix in _MANIFEST_SUFFIXES:
            real_manifest = os.path.realpath(path)
            if real_manifest in seen_manifests:
                raise ValueError(f"Dataset manifests form a cycle at: {path}")
            seen_manifests.add(real_manifest)
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    for line_no, raw_line in enumerate(handle, start=1):
                        line = raw_line.strip()
                        if not line or line.startswith("#"):
                            continue
                        try:
                            add(line, base_dir=os.path.dirname(path))
                        except Exception as exc:
                            raise ValueError(f"Invalid manifest entry {path}:{line_no}: {line!r}") from exc
            finally:
                seen_manifests.remove(real_manifest)
            return

        if _is_image_path(path):
            expanded.append(path)
            return
        raise ValueError(
            f"Unsupported dataset source {path!r}. Expected an image, directory, or manifest "
            f"({', '.join(sorted(_MANIFEST_SUFFIXES))})."
        )

    add(source)
    # Stable de-duplication followed by lexical order makes source handling
    # deterministic across OS/filesystem iteration order.
    return sorted(dict.fromkeys(os.path.abspath(path) for path in expanded))


class DetectionDataset(Dataset):
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
        annotation_policy="fix",
        allow_missing_labels=False,
        source_recursive=True,
    ):
        self.imgsz = self._normalize_imgsz(imgsz)
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
        self.annotation_audit = AnnotationAudit(
            policy=annotation_policy,
            allow_missing_labels=allow_missing_labels,
        )
        # Strict and repair-mode caches persist both sanitized targets and their
        # health summary. Broad warn/ignore policies still recheck every source
        # because they can accept arbitrary malformed rows.
        self._disk_label_cache_safe = self.annotation_audit.policy in {
            "error",
            "clamp",
            "fix",
        }
        self._image_cache = OrderedDict()
        self._label_dirs_by_image_dir = {}
        self._sampling_indices = None
        self._sampling_shared_indices = None
        self._sampling_shared_count = None
        self._sampling_shared_version = None

        self.img_files = expand_image_sources(img_dir, recursive=bool(source_recursive))
        if not self.img_files:
            raise ValueError(f"Dataset source resolved to zero supported images: {img_dir!r}")
        # DataLoader workers are persistent by default.  On Windows those
        # workers own spawned dataset copies, so replacing a normal Python
        # tuple in the parent process does not update mosaic/mixup partner
        # selection.  A fixed-capacity shared table lets epoch-level samplers
        # publish a new active pool without respawning workers.
        self._sampling_shared_indices = torch.empty(
            len(self.img_files), dtype=torch.int64
        ).share_memory_()
        self._sampling_shared_count = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._sampling_shared_version = torch.zeros(1, dtype=torch.int64).share_memory_()
        try:
            self._image_root = os.path.commonpath([os.path.dirname(p) for p in self.img_files])
        except ValueError:
            # Different Windows drives have no meaningful common path.  The
            # Different Windows drives do not have one meaningful common root.
            self._image_root = ""
        stems = [os.path.splitext(os.path.basename(p))[0] for p in self.img_files]
        stem_counts = Counter(stems)
        self._duplicate_image_stems = {stem for stem, count in stem_counts.items() if count > 1}
        self._label_dirs_by_image_dir = self._detect_label_dirs()
        print(
            f"[dataset] images={len(self.img_files)} augment={bool(augment)} "
            f"policy={self.annotation_audit.policy}",
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
                # This is the canonical YOLO mapping even before labels have
                # been created.  Falling through to a sibling ``.txt`` path
                # would otherwise turn a missing-label audit into the wrong
                # location and make newly added labels invisible to this
                # dataset instance.
                label_dirs[img_dir] = os.sep.join(label_parts)
                continue

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
            "eval_boxes": np.zeros((0, 4), dtype=np.float32),
            "eval_labels": np.zeros((0,), dtype=np.int64),
            "eval_iscrowd": np.zeros((0,), dtype=bool),
            "eval_ignore": np.zeros((0,), dtype=bool),
        }

    @staticmethod
    def _parse_label_file(label_path, num_classes=None, audit=None):
        """Parse one YOLO label file without silently converting damage to BG."""
        boxes = []
        labels = []
        audit = audit or AnnotationAudit(policy="error")

        def issue(kind, line=None, detail=None, fatal=True):
            return audit.issue(kind, label_path, line=line, detail=detail, fatal=fatal)

        try:
            exists = os.path.exists(label_path)
        except OSError:
            exists = False

        if not exists:
            issue("missing_label", detail="create an empty .txt file for an intentional negative image", fatal=not audit.allow_missing_labels)
            return DetectionDataset._empty_label_result()

        try:
            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError as exc:
            issue("unreadable_label", detail=str(exc))
            return DetectionDataset._empty_label_result()

        for line_no, line in enumerate(lines, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                issue("malformed_label_row", line=line_no, detail="expected: class x_center y_center width height")
                continue
            try:
                cls, x, y, bw, bh = map(float, parts)
            except ValueError:
                issue("non_numeric_label", line=line_no)
                continue
            values = np.asarray([cls, x, y, bw, bh], dtype=np.float32)
            if not np.isfinite(values).all():
                issue("non_finite_label", line=line_no)
                continue
            if not float(cls).is_integer():
                issue("non_integer_class", line=line_no)
                continue
            cls_i = int(cls)
            if cls_i < 0 or (num_classes is not None and cls_i >= num_classes):
                issue("class_out_of_range", line=line_no, detail=f"class={cls_i}, nc={num_classes}")
                continue
            if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
                issue("center_out_of_range", line=line_no)
                continue
            if bw <= 0.0 or bw > 1.0 or bh <= 0.0 or bh > 1.0:
                issue("size_out_of_range", line=line_no)
                continue
            # YOLO rows are normalized to the image bounds.  Checking only a
            # center and width independently lets a box such as x=.05,w=.5
            # leak outside the image.  Training later clips it during
            # letterboxing while validation used to retain the raw box, making
            # the same annotation mean two different things.  Under the
            # permissive policies retain the visible portion consistently;
            # strict datasets fail loudly through ``issue``.
            x1, y1 = x - bw * 0.5, y - bh * 0.5
            x2, y2 = x + bw * 0.5, y + bh * 0.5
            if x1 < 0.0 or y1 < 0.0 or x2 > 1.0 or y2 > 1.0:
                # Six/seven-decimal YOLO exports can move a mathematically
                # edge-aligned box a few 1e-7 outside [0, 1]. Clip that
                # serialization noise without weakening strict validation for
                # a genuinely out-of-bounds annotation.
                coordinate_epsilon = 1e-6
                materially_outside = (
                    x1 < -coordinate_epsilon
                    or y1 < -coordinate_epsilon
                    or x2 > 1.0 + coordinate_epsilon
                    or y2 > 1.0 + coordinate_epsilon
                )
                if materially_outside or audit.policy != "error":
                    issue("box_extends_outside_image", line=line_no, detail="clipped to normalized image bounds")
                x1, y1 = max(x1, 0.0), max(y1, 0.0)
                x2, y2 = min(x2, 1.0), min(y2, 1.0)
                if x2 <= x1 or y2 <= y1:
                    issue("clipped_empty_label", line=line_no)
                    continue
                x, y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                bw, bh = x2 - x1, y2 - y1
            labels.append(cls_i)
            boxes.append([x, y, bw, bh])

        if not boxes:
            return DetectionDataset._empty_label_result()
        boxes_arr = np.asarray(boxes, dtype=np.float32)
        labels_arr = np.asarray(labels, dtype=np.int64)
        # Exact duplicate labels are never distinct objects in YOLO text and
        # otherwise create contradictory assignment/metric behavior.
        rows = np.concatenate([labels_arr[:, None], boxes_arr], axis=1)
        _, keep = np.unique(rows, axis=0, return_index=True)
        if len(keep) != len(rows):
            audit.issue("duplicate_label", label_path, detail=f"dropped {len(rows) - len(keep)} exact duplicate row(s)")
            keep = np.sort(keep)
            boxes_arr = boxes_arr[keep]
            labels_arr = labels_arr[keep]
        return {
            "boxes": boxes_arr,
            "labels": labels_arr,
            "eval_boxes": boxes_arr.copy(),
            "eval_labels": labels_arr.copy(),
            "eval_iscrowd": np.zeros((len(labels_arr),), dtype=bool),
            "eval_ignore": np.zeros((len(labels_arr),), dtype=bool),
        }

    @staticmethod
    @staticmethod
    def _parse_label_item(args):
        idx, label_path, num_classes, audit = args
        result = DetectionDataset._parse_label_file(
            label_path, num_classes, audit=audit
        )
        return idx, result

    @staticmethod
    def _label_signature(label_path):
        if not label_path:
            return (0, 0, 0)
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
        digest = hashlib.sha1()
        digest.update(b"fotonet-label-cache-v8\0")
        digest.update(str(self.num_classes).encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            (
                f"{self.annotation_audit.policy}|"
                f"{int(self.annotation_audit.allow_missing_labels)}"
            ).encode("utf-8")
        )
        digest.update(b"\0")
        for image_path, label_path in zip(self.img_files, label_paths):
            digest.update(os.path.abspath(image_path).encode("utf-8", errors="ignore"))
            digest.update(b"\0")
            digest.update(os.path.abspath(label_path).encode("utf-8", errors="ignore"))
            digest.update(b"\0")
        return os.path.join(self._label_cache_root(), digest.hexdigest() + ".npz")

    def _load_label_cache_file(self, cache_path, label_paths):
        if not os.path.exists(cache_path):
            return False
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                version = int(data["version"][0])
                if version != 8:
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
                audit_kinds = data["audit_kinds"].astype(str).tolist()
                audit_counts = data["audit_counts"].astype(np.int64, copy=False).tolist()
                audit_examples = data["audit_examples"].astype(str).tolist()
                if len(audit_kinds) != len(audit_counts):
                    return False

                counts = data["counts"].astype(np.int64, copy=False)
                boxes_all = data["boxes"].astype(np.float32, copy=False)
                labels_all = data["labels"].astype(np.int64, copy=False)
                eval_counts = data["eval_counts"].astype(np.int64, copy=False)
                eval_boxes_all = data["eval_boxes"].astype(np.float32, copy=False)
                eval_labels_all = data["eval_labels"].astype(np.int64, copy=False)
                if (
                    counts.ndim != 1
                    or eval_counts.ndim != 1
                    or len(counts) != len(self.img_files)
                    or len(eval_counts) != len(self.img_files)
                    or boxes_all.shape != (int(counts.sum()), 4)
                    or labels_all.shape != (int(counts.sum()),)
                    or eval_boxes_all.shape != (int(eval_counts.sum()), 4)
                    or eval_labels_all.shape != (int(eval_counts.sum()),)
                ):
                    return False
                offset = 0
                eval_offset = 0
                self.cache = {}
                for idx, count in enumerate(counts.tolist()):
                    next_offset = offset + int(count)
                    eval_next_offset = eval_offset + int(eval_counts[idx])
                    cached_boxes = boxes_all[offset:next_offset].copy()
                    cached_labels = labels_all[offset:next_offset].copy()
                    cached_eval_boxes = eval_boxes_all[eval_offset:eval_next_offset].copy()
                    cached_eval_labels = eval_labels_all[eval_offset:eval_next_offset].copy()
                    self.cache[idx] = {
                        "boxes": cached_boxes,
                        "labels": cached_labels,
                        "eval_boxes": cached_eval_boxes,
                        "eval_labels": cached_eval_labels,
                        "eval_iscrowd": np.zeros((len(cached_eval_labels),), dtype=bool),
                        "eval_ignore": np.zeros((len(cached_eval_labels),), dtype=bool),
                    }
                    offset = next_offset
                    eval_offset = eval_next_offset
                self.annotation_audit.restore_summary(
                    {
                        "counts": dict(zip(audit_kinds, audit_counts)),
                        "examples": audit_examples,
                    }
                )
            return len(self.cache) == len(self.img_files)
        except Exception:
            return False

    def _save_label_cache_file(self, cache_path, label_paths):
        counts = np.asarray([len(self.cache[i]["labels"]) for i in range(len(self.img_files))], dtype=np.int64)
        eval_counts = np.asarray([len(self.cache[i]["eval_labels"]) for i in range(len(self.img_files))], dtype=np.int64)
        boxes_parts = [self.cache[i]["boxes"] for i in range(len(self.img_files)) if len(self.cache[i]["boxes"]) > 0]
        labels_parts = [self.cache[i]["labels"] for i in range(len(self.img_files)) if len(self.cache[i]["labels"]) > 0]
        eval_boxes_parts = [self.cache[i]["eval_boxes"] for i in range(len(self.img_files)) if len(self.cache[i]["eval_boxes"]) > 0]
        eval_labels_parts = [self.cache[i]["eval_labels"] for i in range(len(self.img_files)) if len(self.cache[i]["eval_labels"]) > 0]
        boxes_all = np.concatenate(boxes_parts, axis=0).astype(np.float32, copy=False) if boxes_parts else np.zeros((0, 4), dtype=np.float32)
        labels_all = np.concatenate(labels_parts, axis=0).astype(np.int64, copy=False) if labels_parts else np.zeros((0,), dtype=np.int64)
        eval_boxes_all = np.concatenate(eval_boxes_parts, axis=0).astype(np.float32, copy=False) if eval_boxes_parts else np.zeros((0, 4), dtype=np.float32)
        eval_labels_all = np.concatenate(eval_labels_parts, axis=0).astype(np.int64, copy=False) if eval_labels_parts else np.zeros((0,), dtype=np.int64)
        signatures = np.asarray([self._label_signature(p) for p in label_paths], dtype=np.int64)
        audit_summary = self.annotation_audit.summary()
        audit_kinds = sorted(audit_summary["counts"])
        audit_counts = [audit_summary["counts"][kind] for kind in audit_kinds]

        tmp_path = cache_path + ".tmp"
        try:
            with open(tmp_path, "wb") as f:
                np.savez(
                    f,
                    version=np.asarray([8], dtype=np.int64),
                    num_classes=np.asarray([-1 if self.num_classes is None else int(self.num_classes)], dtype=np.int64),
                    image_paths=np.asarray([os.path.abspath(p) for p in self.img_files]),
                    label_paths=np.asarray([os.path.abspath(p) for p in label_paths]),
                    signatures=signatures,
                    counts=counts,
                    boxes=boxes_all,
                    labels=labels_all,
                    eval_counts=eval_counts,
                    eval_boxes=eval_boxes_all,
                    eval_labels=eval_labels_all,
                    audit_kinds=np.asarray(audit_kinds, dtype=np.str_),
                    audit_counts=np.asarray(audit_counts, dtype=np.int64),
                    audit_examples=np.asarray(audit_summary["examples"], dtype=np.str_),
                )
            os.replace(tmp_path, cache_path)
            return True
        except OSError:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            return False

    def _print_label_health_summary(self):
        counts = self.annotation_audit.summary()["counts"]
        clamped = int(counts.get("box_extends_outside_image", 0))
        missing = int(counts.get("missing_label", 0))
        duplicates = int(counts.get("duplicate_label", 0))
        summary = []
        if clamped:
            summary.append(f"clamped={clamped}")
        if missing:
            summary.append(f"missing_as_background={missing}")
        if duplicates:
            summary.append(f"duplicates_dropped={duplicates}")
        other_counts = {
            kind: count
            for kind, count in counts.items()
            if kind
            not in {
                "box_extends_outside_image",
                "missing_label",
                "duplicate_label",
            }
        }
        if other_counts:
            details = ", ".join(
                f"{kind}={count}" for kind, count in sorted(other_counts.items())
            )
            summary.append(details)
        if summary:
            print(f"[dataset] repaired {' '.join(summary)}", flush=True)

    def _cache_all_labels(self):
        count = len(self.img_files)
        label_paths = [self._get_label_path(path) for path in self.img_files]
        cache_path = self._label_cache_path(label_paths)
        start = time.time()
        if self._disk_label_cache_safe and self._load_label_cache_file(
            cache_path, label_paths
        ):
            elapsed = max(time.time() - start, 1e-9)
            print(
                f"[dataset] labels={count} cache=hit rate={count / elapsed:.0f}/s",
                flush=True,
            )
            if self.annotation_audit.policy in {"clamp", "fix"}:
                self._print_label_health_summary()
            return

        workers = min(self.label_cache_workers, max(count, 1))
        items = (
            (index, label_path, self.num_classes, self.annotation_audit)
            for index, label_path in enumerate(label_paths)
        )
        executor = None
        try:
            if workers <= 1 or count <= 1:
                iterator = map(self._parse_label_item, items)
            else:
                executor = ThreadPoolExecutor(max_workers=workers)
                iterator = executor.map(self._parse_label_item, items)
            for index, result in iterator:
                self.cache[index] = result
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        elapsed = max(time.time() - start, 1e-9)
        print(
            f"[dataset] labels={count} cache=built rate={count / elapsed:.0f}/s",
            flush=True,
        )
        if self.annotation_audit.policy in {"clamp", "fix"}:
            self._print_label_health_summary()
        if self._disk_label_cache_safe:
            self._save_label_cache_file(cache_path, label_paths)

    def _get_label(self, idx):
        if idx not in self.cache:
            label_path = self._get_label_path(self.img_files[idx])
            self.cache[idx] = self._parse_label_file(
                label_path, self.num_classes, audit=self.annotation_audit
            )
        return self.cache[idx]

    def _disk_cache_path(self, img_path):
        if not self.disk_cache_dir:
            root = os.path.dirname(os.path.dirname(os.path.abspath(img_path)))
            cache_dir = os.path.join(root, ".fotonet_cache", "decoded_rgb")
        else:
            cache_dir = self.disk_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        stat = os.stat(img_path)
        mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))
        key = f"{os.path.abspath(img_path)}|{stat.st_size}|{mtime_ns}"
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

    def _ram_cache_capacity(self):
        worker = get_worker_info()
        if worker is not None and worker.num_workers > 1:
            # Every process owns a separate dataset instance on Windows. Split
            # the configured capacity across workers so ``ram_cache_images``
            # remains a run-wide budget instead of multiplying by worker count.
            per_worker, remainder = divmod(self.ram_cache_images, worker.num_workers)
            return per_worker + int(worker.id < remainder)
        return self.ram_cache_images

    def _get_cached_image(self, idx):
        cache_capacity = self._ram_cache_capacity()
        if not self.cache_to_ram or cache_capacity <= 0:
            return self._read_image(idx)

        cached = self._image_cache.get(idx)
        if cached is not None:
            self._image_cache.move_to_end(idx)
            return cached

        img = self._read_image(idx)
        self._image_cache[idx] = img
        self._image_cache.move_to_end(idx)
        while len(self._image_cache) > cache_capacity:
            self._image_cache.popitem(last=False)
        return img

    def __len__(self):
        return len(self.img_files)

    @staticmethod
    def _normalize_imgsz(imgsz):
        """Keep the established square scalar form while allowing validation rectangles."""
        if isinstance(imgsz, (tuple, list, np.ndarray)):
            if len(imgsz) != 2:
                raise TypeError("imgsz must be a positive integer or an (height, width) pair")
            height, width = (int(value) for value in imgsz)
            if height <= 0 or width <= 0:
                raise ValueError("imgsz dimensions must be positive")
            return (height, width)
        size = int(imgsz)
        if size <= 0:
            raise ValueError("imgsz must be positive")
        return size

    def get_raw(self, idx):
        img = self._get_cached_image(idx)
        target = self._get_label(idx)
        copied = {}
        for key, value in target.items():
            copied[key] = value.copy() if isinstance(value, np.ndarray) else value
        return img, copied

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_total_epochs(self, max_epochs):
        max_epochs = int(max_epochs) if max_epochs is not None else 0
        self.max_epochs = max_epochs if max_epochs > 0 else None

    def set_sampling_indices(self, indices):
        """Restrict mosaic/mixup partner images to the active training split."""
        valid = [] if indices is None else sorted(
            {int(i) for i in indices if 0 <= int(i) < len(self.img_files)}
        )
        self._sampling_indices = tuple(valid) if valid else None
        shared_indices = self._sampling_shared_indices
        shared_count = self._sampling_shared_count
        shared_version = self._sampling_shared_version
        if shared_indices is None or shared_count is None or shared_version is None:
            return
        # Odd versions are being written; even versions are stable.  The
        # version is published last so a worker never observes a new count
        # paired with partially copied indices.
        shared_version.add_(1)
        if valid:
            shared_indices[:len(valid)].copy_(
                torch.as_tensor(valid, dtype=torch.int64)
            )
        shared_count[0] = len(valid)
        shared_version.add_(1)

    def sample_index(self):
        shared_indices = self._sampling_shared_indices
        shared_count = self._sampling_shared_count
        shared_version = self._sampling_shared_version
        if shared_indices is not None and shared_count is not None and shared_version is not None:
            for _ in range(3):
                version_before = int(shared_version[0])
                if version_before % 2:
                    continue
                count = int(shared_count[0])
                if count <= 0:
                    break
                offset = int(np.random.randint(0, count))
                index = int(shared_indices[offset])
                version_after = int(shared_version[0])
                if version_before == version_after and version_after % 2 == 0:
                    return index
        indices = self._sampling_indices
        if indices:
            return indices[int(np.random.randint(0, len(indices)))]
        return int(np.random.randint(0, len(self.img_files)))

    def set_imgsz(self, imgsz):
        self.imgsz = self._normalize_imgsz(imgsz)

    def _mosaic_enabled(self):
        if not self.augment or self.augment_hyp["mosaic"] <= 0.0:
            return False
        close_mosaic = int(self.augment_hyp.get("close_mosaic", 0) or 0)
        if self.max_epochs is not None and self.max_epochs > close_mosaic > 0:
            if self.epoch >= max(self.max_epochs - close_mosaic, 0):
                return False
        return True

    def _build_base_sample(self, idx):
        use_mosaic = self._mosaic_enabled() and np.random.random() < self.augment_hyp["mosaic"]
        eval_payload = None
        if use_mosaic:
            if not isinstance(self.imgsz, int):
                raise ValueError(
                    "Mosaic augmentation currently requires a square scalar imgsz; "
                    "use a scalar training size or disable mosaic for rectangular preprocessing."
                )
            img, boxes, labels = load_mosaic(self, idx, self.imgsz)
        else:
            img, target = self.get_raw(idx)
            orig_h, orig_w = img.shape[:2]
            boxes, labels = target["boxes"], target["labels"]
            img, boxes, labels = letterbox_resize(img, self.imgsz, boxes, labels=labels)
            # Keep immutable GT in original image coordinates for validation.
            eval_payload = {
                "eval_boxes": np.asarray(target.get("eval_boxes", target["boxes"]), dtype=np.float32).copy(),
                "eval_labels": np.asarray(target.get("eval_labels", target["labels"]), dtype=np.int64).copy(),
                "eval_iscrowd": np.asarray(
                    target.get("eval_iscrowd", np.zeros((len(target.get("eval_labels", target["labels"])),), dtype=bool)),
                    dtype=bool,
                ).copy(),
                "eval_ignore": np.asarray(
                    target.get("eval_ignore", np.zeros((len(target.get("eval_labels", target["labels"])),), dtype=bool)),
                    dtype=bool,
                ).copy(),
                "orig_shape": np.asarray([orig_h, orig_w], dtype=np.int64),
                "image_id": int(target.get("image_id", idx)),
            }
            for key in ("coco_category_ids", "metric_protocol", "coco_annotations_path"):
                if key in target:
                    eval_payload[key] = target[key]

        if self.augment:
            # Evaluation metadata describes the unaugmented source image; do
            # not leak it into a training/mosaic sample where it is no longer
            # aligned with the pixels or targets.
            eval_payload = None
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
                min_label_keep=self.augment_hyp["affine_min_label_keep"],
                max_retries=self.augment_hyp["affine_max_retries"],
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

        return img, boxes.astype(np.float32, copy=False), labels.astype(np.int64, copy=False), eval_payload

    def __getitem__(self, idx):
        img, boxes, labels, eval_payload = self._build_base_sample(idx)

        if self.augment and self.augment_hyp["mixup"] > 0.0 and np.random.random() < self.augment_hyp["mixup"] and len(self.img_files) > 1:
            mix_idx = self.sample_index()
            mix_img, mix_boxes, mix_labels, _ = self._build_base_sample(mix_idx)
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
            "image_id": torch.tensor([idx if eval_payload is None else eval_payload["image_id"]]),
        }
        if eval_payload is not None:
            target.update({
                "eval_boxes": torch.from_numpy(eval_payload["eval_boxes"]),
                "eval_labels": torch.from_numpy(eval_payload["eval_labels"]),
                "eval_iscrowd": torch.from_numpy(eval_payload["eval_iscrowd"]),
                "eval_ignore": torch.from_numpy(eval_payload["eval_ignore"]),
                "orig_shape": torch.from_numpy(eval_payload["orig_shape"]),
            })
            for key in ("coco_category_ids", "metric_protocol", "coco_annotations_path"):
                if key in eval_payload:
                    target[key] = eval_payload[key]
        return to_tensor(img), target


def build_detection_dataset(source, *, imgsz=640, num_classes=None, **kwargs):
    """Build a YOLO-text or direct-COCO dataset from one explicit source.

    COCO sources use ``{"format": "coco", "images": ..., "annotations": ...}``.
    A direct ``.json`` source is also accepted when its image directory is
    supplied as ``coco_images=...``.  Mixed multiple COCO JSON files are
    rejected instead of being silently merged into a non-canonical metric.
    """
    coco_images = kwargs.pop("coco_images", None)
    category_id_to_class = kwargs.pop("category_id_to_class", None)
    if isinstance(source, dict):
        source_format = str(source.get("format", "")).lower()
        annotations = source.get("annotations", source.get("annotation", source.get("json")))
        images = source.get("images", source.get("image_root", source.get("path", coco_images)))
        if source_format == "coco" or annotations:
            if not annotations:
                raise ValueError("COCO dataset source requires an 'annotations' JSON path")
            from fotonet.data.coco import CocoDetectionDataset

            return CocoDetectionDataset(
                annotations=annotations,
                images=images,
                category_id_to_class=source.get("category_id_to_class", category_id_to_class),
                imgsz=imgsz,
                num_classes=num_classes,
                **kwargs,
            )
        source = source.get("images", source.get("path"))
        if source is None:
            raise ValueError("Dataset source dict requires 'images'/'path' or COCO annotations")

    if isinstance(source, (str, os.PathLike)) and Path(os.fspath(source)).suffix.lower() == ".json":
        from fotonet.data.coco import CocoDetectionDataset

        return CocoDetectionDataset(
            annotations=source,
            images=coco_images,
            category_id_to_class=category_id_to_class,
            imgsz=imgsz,
            num_classes=num_classes,
            **kwargs,
        )

    if isinstance(source, (list, tuple)):
        coco_sources = [
            item
            for item in source
            if (isinstance(item, (str, os.PathLike)) and Path(os.fspath(item)).suffix.lower() == ".json")
            or (isinstance(item, dict) and (str(item.get("format", "")).lower() == "coco" or item.get("annotations") or item.get("annotation") or item.get("json")))
        ]
        if coco_sources:
            if len(source) != 1 or len(coco_sources) != 1:
                raise ValueError(
                    "Multiple or mixed COCO JSON validation sources are not a single canonical COCO protocol; "
                    "evaluate each JSON separately."
                )
            return build_detection_dataset(
                coco_sources[0],
                imgsz=imgsz,
                num_classes=num_classes,
                coco_images=coco_images,
                category_id_to_class=category_id_to_class,
                **kwargs,
            )

    return DetectionDataset(source, imgsz=imgsz, num_classes=num_classes, **kwargs)
