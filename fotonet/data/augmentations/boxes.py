"""Bounding-box conversion and validation helpers."""
import numpy as np


def xywhn_to_xyxy(boxes, w, h, padw=0.0, padh=0.0):
    if len(boxes) == 0:
        return boxes.copy()
    out = boxes.astype(np.float32, copy=True)
    out[:, 0] = w * (boxes[:, 0] - boxes[:, 2] * 0.5) + padw
    out[:, 1] = h * (boxes[:, 1] - boxes[:, 3] * 0.5) + padh
    out[:, 2] = w * (boxes[:, 0] + boxes[:, 2] * 0.5) + padw
    out[:, 3] = h * (boxes[:, 1] + boxes[:, 3] * 0.5) + padh
    return out


def xyxy_to_xywhn(boxes, w, h, eps=1e-6):
    if len(boxes) == 0:
        return boxes.copy()
    if w <= 0 or h <= 0:
        raise ValueError(f"xyxy_to_xywhn needs positive image size, got w={w}, h={h}")
    out = boxes.astype(np.float32, copy=True)
    out[:, [0, 2]] = out[:, [0, 2]].clip(0.0, max(w - eps, eps))
    out[:, [1, 3]] = out[:, [1, 3]].clip(0.0, max(h - eps, eps))
    xywh = np.zeros_like(out, dtype=np.float32)
    xywh[:, 0] = ((out[:, 0] + out[:, 2]) * 0.5) / w
    xywh[:, 1] = ((out[:, 1] + out[:, 3]) * 0.5) / h
    xywh[:, 2] = (out[:, 2] - out[:, 0]) / w
    xywh[:, 3] = (out[:, 3] - out[:, 1]) / h
    return xywh


def filter_boxes_xywhn(boxes, labels=None, min_wh=1e-4, min_area=1e-7):
    """Drop clipped/augmented boxes that collapsed to zero or near-zero area."""
    if len(boxes) == 0:
        if labels is None:
            return boxes
        return boxes, labels[:0]

    boxes = boxes.astype(np.float32, copy=False)
    if labels is not None and len(labels) != len(boxes):
        n = min(len(labels), len(boxes))
        boxes = boxes[:n]
        labels = labels[:n]
    keep = (
        (boxes[:, 2] > min_wh)
        & (boxes[:, 3] > min_wh)
        & ((boxes[:, 2] * boxes[:, 3]) > min_area)
        & np.isfinite(boxes).all(axis=1)
    )
    if labels is None:
        return boxes[keep]
    return boxes[keep], labels[keep]


def sanitize_boxes_xywhn(boxes, labels=None, min_wh=1e-4, min_area=1e-7):
    """Clip normalized YOLO boxes by corners and drop invalid rows."""
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        empty_boxes = np.zeros((0, 4), dtype=np.float32)
        if labels is None:
            return empty_boxes
        return empty_boxes, np.asarray(labels)[:0]

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        boxes = np.zeros((0, 4), dtype=np.float32)
        if labels is None:
            return boxes
        return boxes, np.asarray(labels)[:0]

    if labels is not None:
        labels = np.asarray(labels)
        if len(labels) != len(boxes):
            n = min(len(labels), len(boxes))
            boxes = boxes[:n]
            labels = labels[:n]

    keep = np.isfinite(boxes).all(axis=1) & (boxes[:, 2] > 0.0) & (boxes[:, 3] > 0.0)
    boxes = boxes[keep]
    if labels is not None:
        labels = labels[keep]
    if len(boxes) == 0:
        empty_boxes = np.zeros((0, 4), dtype=np.float32)
        if labels is None:
            return empty_boxes
        return empty_boxes, labels[:0]

    x1 = np.clip(boxes[:, 0] - boxes[:, 2] * 0.5, 0.0, 1.0)
    y1 = np.clip(boxes[:, 1] - boxes[:, 3] * 0.5, 0.0, 1.0)
    x2 = np.clip(boxes[:, 0] + boxes[:, 2] * 0.5, 0.0, 1.0)
    y2 = np.clip(boxes[:, 1] + boxes[:, 3] * 0.5, 0.0, 1.0)
    widths = x2 - x1
    heights = y2 - y1
    keep = (
        (widths > min_wh)
        & (heights > min_wh)
        & ((widths * heights) > min_area)
        & np.isfinite(widths)
        & np.isfinite(heights)
    )

    clipped = np.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5, widths, heights), axis=1).astype(
        np.float32,
        copy=False,
    )
    clipped = clipped[keep]
    if labels is None:
        return clipped
    return clipped, labels[keep]


def clip_boxes_xyxy(boxes, w, h):
    if len(boxes) == 0:
        return boxes
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0.0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0.0, h)
    return boxes


def bbox_ioa(box, boxes, eps=1e-7):
    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(inter_x2 - inter_x1, 0.0) * np.maximum(inter_y2 - inter_y1, 0.0)
    area = np.maximum((box[2] - box[0]) * (box[3] - box[1]), eps)
    return inter / area
