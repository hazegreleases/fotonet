"""Sample-level augmentations and final composition."""
import random

import numpy as np

from fotonet.data.augmentations.boxes import (
    bbox_ioa,
    clip_boxes_xyxy,
    sanitize_boxes_xywhn,
    xywhn_to_xyxy,
    xyxy_to_xywhn,
)
from fotonet.data.augmentations.image import (
    augment_hsv_safe,
    random_channel_swap,
    random_flip,
    resize_longest_side,
)
from fotonet.data.augmentations.hyp import finite_float


def _non_overlapping_box(candidate, occupied, ioa_thr=0.3):
    if len(occupied) == 0:
        return True
    return not np.any(bbox_ioa(candidate, occupied) > ioa_thr)


def _find_copy_paste_box(preferred, patch_w, patch_h, img_w, img_h, occupied, attempts=30):
    if patch_w <= 0 or patch_h <= 0 or patch_w > img_w or patch_h > img_h:
        return None

    preferred = preferred.astype(np.float32, copy=True)
    preferred[0] = np.clip(preferred[0], 0, img_w - patch_w)
    preferred[1] = np.clip(preferred[1], 0, img_h - patch_h)
    preferred[2] = preferred[0] + patch_w
    preferred[3] = preferred[1] + patch_h
    if _non_overlapping_box(preferred, occupied):
        return preferred

    max_x = max(int(img_w - patch_w), 0)
    max_y = max(int(img_h - patch_h), 0)
    for _ in range(max(int(attempts), 0)):
        x1 = int(np.random.randint(0, max_x + 1)) if max_x > 0 else 0
        y1 = int(np.random.randint(0, max_y + 1)) if max_y > 0 else 0
        candidate = np.asarray([x1, y1, x1 + patch_w, y1 + patch_h], dtype=np.float32)
        if _non_overlapping_box(candidate, occupied):
            return candidate
    return None


def copy_paste(img, boxes, labels, p=0.0, mode="flip"):
    if p <= 0.0 or len(boxes) == 0 or random.random() >= p:
        return img, boxes, labels

    if mode != "flip":
        return img, boxes, labels

    h, w = img.shape[:2]
    boxes_xyxy = xywhn_to_xyxy(boxes, w, h)
    flipped = np.ascontiguousarray(img[:, ::-1])
    out = img.copy()
    pasted_boxes = []
    pasted_labels = []

    order = np.random.permutation(len(boxes_xyxy))
    for idx in order:
        x1, y1, x2, y2 = boxes_xyxy[idx].astype(np.int32)
        if x2 <= x1 or y2 <= y1:
            continue

        mx1 = max(w - x2, 0)
        mx2 = min(w - x1, w)
        mirrored_box = np.array([mx1, y1, mx2, y2], dtype=np.float32)

        patch = flipped[y1:y2, mx1:mx2]
        if patch.size == 0 or patch.shape[0] != (y2 - y1) or patch.shape[1] != (mx2 - mx1):
            continue

        occupied = boxes_xyxy if not pasted_boxes else np.concatenate([boxes_xyxy, np.stack(pasted_boxes)], axis=0)
        target_box = _find_copy_paste_box(
            mirrored_box,
            patch.shape[1],
            patch.shape[0],
            w,
            h,
            occupied,
        )
        if target_box is None:
            continue

        tx1, ty1, tx2, ty2 = target_box.astype(np.int32)
        out[ty1:ty2, tx1:tx2] = patch
        pasted_boxes.append(target_box)
        pasted_labels.append(labels[idx])

    if pasted_boxes:
        new_boxes = np.concatenate([boxes_xyxy, np.stack(pasted_boxes)], axis=0)
        new_labels = np.concatenate([labels, np.asarray(pasted_labels, dtype=np.int64)], axis=0)
        new_boxes = xyxy_to_xywhn(new_boxes, w, h)
        new_boxes, new_labels = sanitize_boxes_xywhn(new_boxes, new_labels)
        return out, new_boxes, new_labels

    return img, boxes, labels


def mixup(img1, boxes1, labels1, img2, boxes2, labels2, alpha=32.0):
    alpha = finite_float(alpha, 0.0)
    if alpha <= 0:
        return img1, boxes1, labels1
    lam = np.random.beta(alpha, alpha)
    mixed = ((img1.astype(np.float32) * lam) + (img2.astype(np.float32) * (1.0 - lam))).clip(0, 255).astype(np.uint8)
    if len(boxes1) and len(boxes2):
        boxes = np.concatenate([boxes1, boxes2], axis=0)
        labels = np.concatenate([labels1, labels2], axis=0)
    elif len(boxes1):
        boxes, labels = boxes1, labels1
    else:
        boxes, labels = boxes2, labels2
    return mixed, boxes, labels


def load_mosaic(dataset, index, imgsz=None, max_retries=2):
    if imgsz is None:
        imgsz = dataset.imgsz
    imgsz = int(imgsz)
    best = None
    best_count = -1
    for _ in range(max(int(max_retries), 0) + 1):
        img, boxes, labels, source_count = _load_mosaic_once(dataset, index, imgsz)
        if source_count == 0 or len(labels) > 0:
            return img, boxes, labels
        if len(labels) > best_count:
            best = (img, boxes, labels)
            best_count = len(labels)
    return best


def _sample_dataset_index(dataset):
    sample_index = getattr(dataset, "sample_index", None)
    if callable(sample_index):
        return int(sample_index())
    return random.randint(0, len(dataset) - 1)


def _load_mosaic_once(dataset, index, imgsz):
    indices = [index] + [_sample_dataset_index(dataset) for _ in range(3)]
    center_x = int(random.uniform(imgsz * 0.5, imgsz * 1.5))
    center_y = int(random.uniform(imgsz * 0.5, imgsz * 1.5))
    large_size = imgsz * 2
    mosaic_img = None
    mosaic_boxes = []
    mosaic_labels = []
    source_count = 0

    for slot, idx in enumerate(indices):
        img, target = dataset.get_raw(idx)
        boxes = target["boxes"]
        labels = target["labels"]
        source_count += len(labels)
        img = resize_longest_side(img, imgsz)
        h, w = img.shape[:2]
        if mosaic_img is None:
            mosaic_img = np.full((large_size, large_size, img.shape[2]), 114, dtype=np.uint8)

        if slot == 0:
            x1a, y1a, x2a, y2a = max(center_x - w, 0), max(center_y - h, 0), center_x, center_y
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif slot == 1:
            x1a, y1a, x2a, y2a = center_x, max(center_y - h, 0), min(center_x + w, large_size), center_y
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif slot == 2:
            x1a, y1a, x2a, y2a = max(center_x - w, 0), center_y, center_x, min(large_size, center_y + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        else:
            x1a, y1a, x2a, y2a = center_x, center_y, min(center_x + w, large_size), min(large_size, center_y + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        if x2a <= x1a or y2a <= y1a or x2b <= x1b or y2b <= y1b:
            continue

        mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        padw = x1a - x1b
        padh = y1a - y1b

        if len(boxes) > 0:
            boxes_xyxy = xywhn_to_xyxy(boxes, w, h)
            boxes_xyxy[:, [0, 2]] += padw
            boxes_xyxy[:, [1, 3]] += padh
            mosaic_boxes.append(boxes_xyxy)
            mosaic_labels.append(labels)

    if mosaic_img is None:
        mosaic_img = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        return mosaic_img, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64), source_count

    crop_x1 = imgsz // 2
    crop_y1 = imgsz // 2
    crop_x2 = crop_x1 + imgsz
    crop_y2 = crop_y1 + imgsz
    out_img = mosaic_img[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    if mosaic_boxes:
        boxes = np.concatenate(mosaic_boxes, axis=0)
        labels = np.concatenate(mosaic_labels, axis=0)
        boxes[:, [0, 2]] -= crop_x1
        boxes[:, [1, 3]] -= crop_y1
        original_area = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(
            boxes[:, 3] - boxes[:, 1],
            0.0,
        )
        boxes = clip_boxes_xyxy(boxes, imgsz, imgsz)
        box_w = boxes[:, 2] - boxes[:, 0]
        box_h = boxes[:, 3] - boxes[:, 1]
        visible = (box_w * box_h) / np.maximum(original_area, 1e-6)
        keep = (
            (box_w > 2.0)
            & (box_h > 2.0)
            & (visible > 0.1)
            & np.isfinite(boxes).all(axis=1)
        )
        boxes = boxes[keep]
        labels = labels[keep]
        if len(boxes) == 0:
            return out_img, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64), source_count
        boxes = xyxy_to_xywhn(boxes, imgsz, imgsz)
        boxes, labels = sanitize_boxes_xywhn(boxes, labels)
        return out_img, boxes, labels, source_count

    return out_img, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64), source_count


def finalize_sample(img, boxes, labels, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, fliplr=0.0, flipud=0.0, bgr=0.0):
    img, boxes = random_flip(img, boxes, fliplr=fliplr, flipud=flipud)
    img = augment_hsv_safe(img, hgain=hsv_h, sgain=hsv_s, vgain=hsv_v)
    img = random_channel_swap(img, p=bgr)
    boxes, labels = sanitize_boxes_xywhn(boxes, labels)
    return img, boxes, labels
