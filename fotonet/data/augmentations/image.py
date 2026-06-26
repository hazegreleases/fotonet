"""Image-space augmentation helpers."""
import random

import numpy as np
import torch

try:
    import cv2
except ImportError:
    cv2 = None

from fotonet.data.augmentations.boxes import (
    clip_boxes_xyxy,
    sanitize_boxes_xywhn,
    xywhn_to_xyxy,
    xyxy_to_xywhn,
)


def to_tensor(img):
    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    return torch.from_numpy(img).float().div_(255.0)


def letterbox_resize(img, target_size, boxes=None, pad_value=114):
    if int(target_size) <= 0:
        raise ValueError(f"letterbox target_size must be positive, got {target_size}")
    target_size = int(target_size)
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"letterbox needs non-empty image, got shape={img.shape}")
    scale = min(target_size / h, target_size / w)
    new_w = max(int(round(w * scale)), 1)
    new_h = max(int(round(h * scale)), 1)
    if cv2 is not None:
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = np.array(
            torch.nn.functional.interpolate(
                torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float(),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 2, 0)
            .byte()
        )
    padded = np.full((target_size, target_size, 3), pad_value, dtype=np.uint8)
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    if boxes is None or len(boxes) == 0:
        return padded, np.zeros((0, 4), dtype=np.float32)

    boxes_xyxy = xywhn_to_xyxy(boxes, w, h)
    boxes_xyxy[:, [0, 2]] *= scale
    boxes_xyxy[:, [1, 3]] *= scale
    boxes_xyxy[:, [0, 2]] += pad_left
    boxes_xyxy[:, [1, 3]] += pad_top
    boxes_xyxy = clip_boxes_xyxy(boxes_xyxy, target_size, target_size)
    boxes_xywh = xyxy_to_xywhn(boxes_xyxy, target_size, target_size)
    return padded, sanitize_boxes_xywhn(boxes_xywh)


def augment_hsv_safe(img, hgain=0.015, sgain=0.7, vgain=0.4):
    if cv2 is None or (hgain == 0 and sgain == 0 and vgain == 0):
        return img

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(hsv)
    x = np.arange(0, 256, dtype=np.int16)
    gains = np.random.uniform(-1.0, 1.0, 3)
    lut_hue = ((x + gains[0] * hgain * 180) % 180).astype(np.uint8)
    lut_sat = np.clip(x * (1.0 + gains[1] * sgain), 0, 255).astype(np.uint8)
    lut_val = np.clip(x * (1.0 + gains[2] * vgain), 0, 255).astype(np.uint8)
    hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def random_channel_swap(img, p=0.0):
    return img[:, :, ::-1].copy() if p > 0.0 and random.random() < p else img


def random_flip(img, boxes, fliplr=0.5, flipud=0.0):
    if fliplr > 0.0 and random.random() < fliplr:
        img = np.ascontiguousarray(img[:, ::-1])
        if len(boxes) > 0:
            boxes = boxes.copy()
            boxes[:, 0] = 1.0 - boxes[:, 0]
    if flipud > 0.0 and random.random() < flipud:
        img = np.ascontiguousarray(img[::-1, :])
        if len(boxes) > 0:
            boxes = boxes.copy()
            boxes[:, 1] = 1.0 - boxes[:, 1]
    return img, boxes
