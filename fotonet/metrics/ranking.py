"""Diagnostics for localization-aware prediction ranking."""
from __future__ import annotations

import numpy as np
import torch

from fotonet.utils.boxes import box_iou, xywh_to_xyxy


def _as_tensor(value, dtype=None):
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)

    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    for group_idx, count in enumerate(counts):
        if count <= 1:
            continue
        mask = inverse == group_idx
        ranks[mask] = ranks[mask].mean()
    return ranks


def _corrcoef(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return 0.0
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def score_iou_diagnostics(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, topk=100):
    """Measure score/localization alignment without rewarding duplicate boxes.

    Every ground-truth object can contribute to at most one prediction. Images
    with no detections still affect the top-1 aggregate and expose their missed
    ground truth count, avoiding the former flattering "skip empty image"
    behavior.  This remains a ranking diagnostic, not AP.
    """
    paired_scores = []
    paired_ious = []
    top1_ious = []
    missed_ground_truth = 0
    duplicate_or_unmatched_predictions = 0
    evaluated_images = 0

    for p_boxes, p_scores, p_classes, g_boxes, g_classes in zip(
        pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes
    ):
        p_boxes = _as_tensor(p_boxes, dtype=torch.float32)
        p_scores = _as_tensor(p_scores, dtype=torch.float32).reshape(-1)
        p_classes = _as_tensor(p_classes, dtype=torch.long).reshape(-1)
        g_boxes = _as_tensor(g_boxes, dtype=torch.float32)
        g_classes = _as_tensor(g_classes, dtype=torch.long).reshape(-1)
        evaluated_images += 1
        if p_boxes.numel() == 0 or p_scores.numel() == 0:
            top1_ious.append(0.0)
            missed_ground_truth += int(g_boxes.shape[0])
            continue

        keep = torch.argsort(p_scores, descending=True)[: int(topk)]
        p_boxes = p_boxes[keep]
        p_scores = p_scores[keep]
        p_classes = p_classes[keep]

        image_ious = []
        matched_gt = torch.zeros((g_boxes.shape[0],), dtype=torch.bool, device=g_boxes.device)
        if g_boxes.numel() > 0:
            ious = box_iou(xywh_to_xyxy(p_boxes), xywh_to_xyxy(g_boxes))
            for row_idx in range(p_boxes.shape[0]):
                same_class = g_classes == p_classes[row_idx]
                candidates = same_class & ~matched_gt
                if bool(candidates.any()):
                    candidate_indices = torch.nonzero(candidates, as_tuple=False).reshape(-1)
                    candidate_ious = ious[row_idx, candidate_indices]
                    best_local = int(torch.argmax(candidate_ious).item())
                    best_idx = candidate_indices[best_local]
                    best_iou = candidate_ious[best_local]
                    if float(best_iou.item()) > 0.0:
                        matched_gt[best_idx] = True
                    else:
                        duplicate_or_unmatched_predictions += 1
                else:
                    best_iou = ious.new_tensor(0.0)
                    duplicate_or_unmatched_predictions += 1
                image_ious.append(float(best_iou.item()))
        else:
            image_ious = [0.0 for _ in range(p_boxes.shape[0])]
            duplicate_or_unmatched_predictions += int(p_boxes.shape[0])

        paired_scores.extend(float(x) for x in p_scores.detach().cpu().tolist())
        paired_ious.extend(image_ious)
        if image_ious:
            top1_ious.append(float(image_ious[0]))
        missed_ground_truth += int((~matched_gt).sum().item())

    if not paired_scores:
        return {
            "ranking_pairs": 0,
            "score_iou_pearson": 0.0,
            "score_iou_spearman": 0.0,
            "top1_mean_iou": 0.0,
            "topk_mean_iou": 0.0,
            "ranking_images": int(evaluated_images),
            "missed_ground_truth": int(missed_ground_truth),
            "duplicate_or_unmatched_predictions": int(duplicate_or_unmatched_predictions),
        }

    scores_np = np.asarray(paired_scores, dtype=np.float64)
    ious_np = np.asarray(paired_ious, dtype=np.float64)
    return {
        "ranking_pairs": int(scores_np.size),
        "score_iou_pearson": round(_corrcoef(scores_np, ious_np), 4),
        "score_iou_spearman": round(_corrcoef(_rankdata(scores_np), _rankdata(ious_np)), 4),
        "top1_mean_iou": round(float(np.mean(top1_ious)) if top1_ious else 0.0, 4),
        "topk_mean_iou": round(float(np.mean(ious_np)), 4),
        "ranking_images": int(evaluated_images),
        "missed_ground_truth": int(missed_ground_truth),
        "duplicate_or_unmatched_predictions": int(duplicate_or_unmatched_predictions),
    }
