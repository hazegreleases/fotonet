"""Shared box geometry helpers for training, matching, metrics, and inference."""
import numpy as np
import torch


def xywh_to_xyxy(boxes):
    """Convert torch boxes from normalized/absolute xywh to xyxy."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5), dim=-1)


def box_iou(boxes1, boxes2, eps=1e-7):
    """Pairwise IoU for torch xyxy boxes. Returns [N, M]."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    inter_x1 = torch.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    return inter / (area1[:, None] + area2[None, :] - inter + eps)


def box_giou(boxes1, boxes2, eps=1e-7):
    """Pairwise generalized IoU for torch xyxy boxes. Returns [N, M]."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    inter_x1 = torch.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp(0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp(0))
    union = area1[:, None] + area2[None, :] - inter + eps
    iou = inter / union

    enc_x1 = torch.minimum(boxes1[:, None, 0], boxes2[None, :, 0])
    enc_y1 = torch.minimum(boxes1[:, None, 1], boxes2[None, :, 1])
    enc_x2 = torch.maximum(boxes1[:, None, 2], boxes2[None, :, 2])
    enc_y2 = torch.maximum(boxes1[:, None, 3], boxes2[None, :, 3])
    enc_area = (enc_x2 - enc_x1).clamp(0) * (enc_y2 - enc_y1).clamp(0) + eps
    return iou - (enc_area - union) / enc_area


def xywh_to_xyxy_np(boxes):
    """Convert numpy boxes from normalized/absolute xywh to xyxy."""
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.asarray(boxes).dtype if np.asarray(boxes).size else np.float32)
    boxes = np.asarray(boxes)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack((cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5), axis=1)


def box_iou_np(boxes1, boxes2, eps=1e-7):
    """Pairwise IoU for numpy xyxy boxes. Returns [N, M]."""
    boxes1 = np.asarray(boxes1)
    boxes2 = np.asarray(boxes2)
    n, m = len(boxes1), len(boxes2)
    if n == 0 or m == 0:
        return np.zeros((n, m))

    inter_x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    return inter / (area1[:, None] + area2[None, :] - inter + eps)
