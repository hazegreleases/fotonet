"""Sample-level augmentations and final composition."""
import random

import numpy as np

from fotonet.data.augmentations.boxes import bbox_ioa, sanitize_boxes_xywhn, xywhn_to_xyxy, xyxy_to_xywhn
from fotonet.data.augmentations.image import augment_hsv_safe, letterbox_resize, random_channel_swap, random_flip
from fotonet.data.augmentations.hyp import finite_float


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
        occupied = boxes_xyxy if not pasted_boxes else np.concatenate([boxes_xyxy, np.stack(pasted_boxes)], axis=0)
        if len(occupied) > 0 and np.any(bbox_ioa(mirrored_box, occupied) > 0.3):
            continue

        patch = flipped[y1:y2, mx1:mx2]
        if patch.size == 0 or patch.shape[0] != (y2 - y1) or patch.shape[1] != (mx2 - mx1):
            continue

        out[y1:y2, mx1:mx2] = patch
        pasted_boxes.append(mirrored_box)
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


def load_mosaic(dataset, index, imgsz=None):
    if imgsz is None:
        imgsz = dataset.imgsz

    half = imgsz // 2
    indices = [index] + [random.randint(0, len(dataset) - 1) for _ in range(3)]
    mosaic_img = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    mosaic_boxes = []
    mosaic_labels = []

    for slot, idx in enumerate(indices):
        img, target = dataset.get_raw(idx)
        boxes = target["boxes"]
        labels = target["labels"]
        img, boxes = letterbox_resize(img, half, boxes)

        off_x = half if slot in (1, 3) else 0
        off_y = half if slot in (2, 3) else 0
        mosaic_img[off_y:off_y + half, off_x:off_x + half] = img

        if len(boxes) > 0:
            boxes = boxes.copy()
            boxes[:, 0] = (boxes[:, 0] * half + off_x) / imgsz
            boxes[:, 1] = (boxes[:, 1] * half + off_y) / imgsz
            boxes[:, 2] = (boxes[:, 2] * half) / imgsz
            boxes[:, 3] = (boxes[:, 3] * half) / imgsz
            mosaic_boxes.append(boxes)
            mosaic_labels.append(labels)

    if mosaic_boxes:
        boxes = np.concatenate(mosaic_boxes, axis=0)
        labels = np.concatenate(mosaic_labels, axis=0)
        return mosaic_img, *sanitize_boxes_xywhn(boxes, labels)

    return mosaic_img, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)


def finalize_sample(img, boxes, labels, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, fliplr=0.0, flipud=0.0, bgr=0.0):
    img, boxes = random_flip(img, boxes, fliplr=fliplr, flipud=flipud)
    img = augment_hsv_safe(img, hgain=hsv_h, sgain=hsv_s, vgain=hsv_v)
    img = random_channel_swap(img, p=bgr)
    boxes, labels = sanitize_boxes_xywhn(boxes, labels)
    return img, boxes, labels
