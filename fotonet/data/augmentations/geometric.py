"""Geometric augmentations."""
import math
import random

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from fotonet.data.augmentations.boxes import clip_boxes_xyxy, sanitize_boxes_xywhn, xywhn_to_xyxy, xyxy_to_xywhn
from fotonet.data.augmentations.hyp import clamp


def random_affine(
    img,
    boxes,
    labels,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    border=0,
    min_label_keep=1.0,
    max_retries=2,
):
    if cv2 is None:
        return img, *sanitize_boxes_xywhn(boxes, labels)

    boxes, labels = sanitize_boxes_xywhn(boxes, labels)
    original_count = len(labels)

    def _apply(strength=1.0):
        h = img.shape[0] + border * 2
        w = img.shape[1] + border * 2

        center = np.eye(3, dtype=np.float32)
        center[0, 2] = -img.shape[1] * 0.5
        center[1, 2] = -img.shape[0] * 0.5

        rotation = np.eye(3, dtype=np.float32)
        angle = random.uniform(-degrees * strength, degrees * strength)
        scale_range = max(scale * strength, 0.0)
        scale_factor = random.uniform(max(1.0 - scale_range, 0.05), 1.0 + scale_range)
        rotation[:2] = cv2.getRotationMatrix2D((0, 0), angle, scale_factor)

        shear_mat = np.eye(3, dtype=np.float32)
        shear_range = min(abs(shear * strength), 89.0)
        shear_mat[0, 1] = math.tan(random.uniform(-shear_range, shear_range) * math.pi / 180.0)
        shear_mat[1, 0] = math.tan(random.uniform(-shear_range, shear_range) * math.pi / 180.0)

        translate_mat = np.eye(3, dtype=np.float32)
        translate_range = max(float(translate) * strength, 0.0)
        translate_mat[0, 2] = random.uniform(0.5 - translate_range, 0.5 + translate_range) * w
        translate_mat[1, 2] = random.uniform(0.5 - translate_range, 0.5 + translate_range) * h

        matrix = translate_mat @ shear_mat @ rotation @ center
        warped = cv2.warpAffine(img, matrix[:2], dsize=(w, h), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

        if original_count == 0:
            return warped, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        boxes_xyxy = xywhn_to_xyxy(boxes, img.shape[1], img.shape[0])
        n = len(boxes_xyxy)
        corners = np.ones((n * 4, 3), dtype=np.float32)
        corners[:, :2] = boxes_xyxy[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
        corners = corners @ matrix.T
        corners = corners[:, :2].reshape(n, 8)

        x = corners[:, [0, 2, 4, 6]]
        y = corners[:, [1, 3, 5, 7]]
        new_boxes = np.stack((x.min(1), y.min(1), x.max(1), y.max(1)), axis=1)
        new_boxes = clip_boxes_xyxy(new_boxes, w, h)

        orig_w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        orig_h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        new_w = new_boxes[:, 2] - new_boxes[:, 0]
        new_h = new_boxes[:, 3] - new_boxes[:, 1]
        area_ratio = (new_w * new_h) / np.maximum(orig_w * orig_h, 1e-6)
        keep = (new_w > 2.0) & (new_h > 2.0) & (area_ratio > 0.1) & np.isfinite(new_boxes).all(axis=1)

        new_boxes = new_boxes[keep]
        new_labels = labels[keep]
        if len(new_boxes) == 0:
            return warped, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        new_boxes = xyxy_to_xywhn(new_boxes, w, h)
        new_boxes, new_labels = sanitize_boxes_xywhn(new_boxes, new_labels)
        return warped, new_boxes, new_labels

    keep_ratio = clamp(min_label_keep, 0.0, 1.0, 1.0)
    min_keep = max(1, int(math.ceil(original_count * keep_ratio))) if original_count else 0
    best = None
    for attempt in range(max(int(max_retries), 0) + 1):
        strength = 1.0 if attempt == 0 else 0.5 ** attempt
        candidate = _apply(strength)
        if best is None or len(candidate[2]) > len(best[2]):
            best = candidate
        if original_count == 0 or len(candidate[2]) >= min_keep:
            return candidate

    return img, boxes, labels
