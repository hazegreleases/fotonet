"""
Batched NMS for FOTONET inference.
Box format: [x_center, y_center, width, height] normalized 0-1.
"""
import torch
from fotonet.utils.boxes import box_iou, xywh_to_xyxy


def _nms_fallback(boxes_xyxy, scores, iou_threshold):
    """Pure PyTorch NMS if torchvision not available."""
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(boxes_xyxy[i].unsqueeze(0), boxes_xyxy[order[1:]])[0]
        mask = ious <= iou_threshold
        order = order[1:][mask]
    return torch.tensor(keep, device=boxes_xyxy.device, dtype=torch.long)


def batched_nms(boxes, scores, classes, iou_threshold=0.45, score_threshold=0.25):
    """
    Per-class NMS: run NMS separately per class so different classes don't suppress each other.
    boxes: [N, 4] in normalized xywh
    scores: [N]
    classes: [N] int
    Returns: (boxes, scores, classes) filtered and sorted by score.
    """
    if boxes.numel() == 0:
        return boxes, scores, classes

    if score_threshold > 0:
        mask = scores >= score_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]
    if boxes.numel() == 0:
        return boxes, scores, classes

    xyxy = xywh_to_xyxy(boxes)
    try:
        from torchvision.ops import nms as tv_nms
        nms_fn = tv_nms
    except Exception:
        nms_fn = lambda b, s, iou: _nms_fallback(b, s, iou)

    keep_all = []
    for c in classes.unique():
        idx = (classes == c).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        k = nms_fn(xyxy[idx], scores[idx], iou_threshold)
        keep_all.append(idx[k])
    if not keep_all:
        return boxes.new_empty(0, 4), scores.new_empty(0), classes.new_empty(0)
    keep = torch.cat(keep_all)
    _, order = scores[keep].sort(descending=True)
    keep = keep[order]
    return boxes[keep], scores[keep], classes[keep]
