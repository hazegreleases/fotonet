"""Uniform sampling policy identity and per-image metric helpers.

The adaptive sampler was removed. The policy normalizer remains so the frozen
launcher and active Nano checkpoint retain their exact run identity while
accepting the sole supported strategy: uniform sampling.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
SAMPLING_POLICY_VERSION = 1

DEFAULT_SAMPLING_HYP = {
    "strategy": "uniform",
    "warmup_epochs": 120,
    "update_interval": 5,
    "easy_threshold": 0.85,
    "moderate_threshold": 0.55,
    "easy_fraction": 0.02,
    "easy_review_interval": 10,
    "easy_forced_fraction": 0.50,
    "moderate_fraction": 0.40,
    "moderate_coverage_interval": 3,
    "score_conf": 0.25,
    "score_iou": 0.50,
    "score_max_dets": 100,
    "score_model": "ema",
    "score_imgsz": "current",
    "minimum_epoch_images": "auto",
    "seed": 42,
    "restrict_mosaic_partners": True,
    "dataset_mismatch": "error",
}

def normalize_sampling_hyp(sampling_hyp=None, *, seed=None):
    """Return the one canonical policy accepted by training protocol 1.

    The complete mapping is retained because it is part of the active run's
    checkpoint fingerprint. It no longer selects or configures an adaptive
    implementation.
    """
    if sampling_hyp is None:
        supplied = {}
    elif isinstance(sampling_hyp, Mapping):
        supplied = dict(sampling_hyp)
    else:
        raise TypeError(
            f"sampling_hyp must be a mapping, got {type(sampling_hyp).__name__}"
        )
    unknown = sorted(set(supplied) - set(DEFAULT_SAMPLING_HYP))
    if unknown:
        raise ValueError(f"Unknown uniform sampling option(s): {', '.join(unknown)}")
    if seed is not None and "seed" in supplied and supplied["seed"] != seed:
        raise ValueError("Conflicting uniform sampling seed values")
    hyp = {**DEFAULT_SAMPLING_HYP, **supplied}
    if seed is not None:
        hyp["seed"] = seed
    if hyp["strategy"] != "uniform":
        raise ValueError("Only uniform sampling is supported by training protocol 1")
    if not isinstance(hyp["seed"], int) or isinstance(hyp["seed"], bool):
        raise TypeError("seed must be an integer")
    if hyp["dataset_mismatch"] != "error":
        raise ValueError("dataset_mismatch must be 'error' for the frozen policy")
    return hyp


def _json_compatible(value):
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "Fingerprint inputs must contain only paths, mappings, sequences, "
        f"strings, numbers, booleans, or None; got {type(value).__name__}"
    )


def sampling_policy_fingerprint(sampling_hyp):
    """Return a stable digest of every trajectory-affecting sampling setting."""

    normalized = normalize_sampling_hyp(sampling_hyp)
    payload = {
        "policy_version": SAMPLING_POLICY_VERSION,
        "config": _json_compatible(normalized),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_int(value, name, *, minimum=None):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and converted < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {converted}")
    return converted


def _as_fraction(value, name, *, lower=0.0, upper=1.0, lower_open=False):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    lower_valid = converted > lower if lower_open else converted >= lower
    if not lower_valid or converted > upper:
        left = "(" if lower_open else "["
        raise ValueError(f"{name} must be in {left}{lower}, {upper}]")
    return converted


def _as_numpy(value, *, dtype, name):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        value = value.numpy()
    try:
        return np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} cannot be converted to {np.dtype(dtype)}") from exc


def _boxes_xyxy(boxes, box_format, name):
    boxes = _as_numpy(boxes, dtype=np.float64, name=name)
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"{name} must have shape [N, 4], got {boxes.shape}")
    if not np.isfinite(boxes).all():
        raise FloatingPointError(f"{name} contains NaN or infinity")
    normalized_format = str(box_format).strip().lower()
    if normalized_format in {"xywh", "xywhn", "cxcywh"}:
        if np.any(boxes[:, 2:] < 0):
            raise ValueError(f"{name} contains a negative width or height")
        converted = np.empty_like(boxes)
        converted[:, 0] = boxes[:, 0] - boxes[:, 2] * 0.5
        converted[:, 1] = boxes[:, 1] - boxes[:, 3] * 0.5
        converted[:, 2] = boxes[:, 0] + boxes[:, 2] * 0.5
        converted[:, 3] = boxes[:, 1] + boxes[:, 3] * 0.5
        return converted
    if normalized_format == "xyxy":
        if np.any(boxes[:, 2] < boxes[:, 0]) or np.any(boxes[:, 3] < boxes[:, 1]):
            raise ValueError(f"{name} contains inverted xyxy coordinates")
        return boxes.copy()
    raise ValueError("box_format must be 'xywh'/'xywhn' or 'xyxy'")


def _class_array(value, name):
    original = _as_numpy(value, dtype=np.float64, name=name).reshape(-1)
    if not np.isfinite(original).all():
        raise FloatingPointError(f"{name} contains NaN or infinity")
    if not np.equal(original, np.floor(original)).all():
        raise ValueError(f"{name} must contain integer class IDs")
    converted = original.astype(np.int64)
    if np.any(converted < 0):
        raise ValueError(f"{name} cannot contain negative class IDs")
    return converted


def _boolean_array(value, name, length):
    if value is None:
        return np.zeros(length, dtype=bool)
    array = _as_numpy(value, dtype=bool, name=name).reshape(-1)
    if len(array) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(array)}")
    return array


def _intersection(pred_box, gt_boxes):
    left = np.maximum(pred_box[0], gt_boxes[:, 0])
    top = np.maximum(pred_box[1], gt_boxes[:, 1])
    right = np.minimum(pred_box[2], gt_boxes[:, 2])
    bottom = np.minimum(pred_box[3], gt_boxes[:, 3])
    return np.maximum(right - left, 0.0) * np.maximum(bottom - top, 0.0)


def _iou_one_to_many(pred_box, gt_boxes):
    if len(gt_boxes) == 0:
        return np.zeros((0,), dtype=np.float64)
    intersection = _intersection(pred_box, gt_boxes)
    pred_area = max(
        (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1]), 0.0
    )
    gt_area = np.maximum(gt_boxes[:, 2] - gt_boxes[:, 0], 0.0) * np.maximum(
        gt_boxes[:, 3] - gt_boxes[:, 1], 0.0
    )
    union = pred_area + gt_area - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def _ioa_prediction(pred_box, crowd_boxes):
    if len(crowd_boxes) == 0:
        return np.zeros((0,), dtype=np.float64)
    intersection = _intersection(pred_box, crowd_boxes)
    pred_area = max(
        (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1]), 0.0
    )
    if pred_area <= 0:
        return np.zeros_like(intersection)
    return intersection / pred_area


@dataclass(frozen=True)
class ImageSufficiencyScore:
    precision: float
    recall: float
    sufficiency: float
    true_positives: int
    false_positives: int
    false_negatives: int
    ignored_predictions: int
    evaluable_ground_truth: int
    considered_predictions: int


def score_image_sufficiency(
    pred_boxes,
    pred_scores,
    pred_classes,
    gt_boxes=None,
    gt_classes=None,
    *,
    target=None,
    gt_iscrowd=None,
    gt_ignore=None,
    confidence_threshold=0.25,
    iou_threshold=0.50,
    max_detections=100,
    box_format="xywh",
):
    """Compute class-aware per-image precision, recall, and ``min(P, R)``.

    Predictions must already be mapped into the same coordinate system as the
    ground truth.  Ordinary GT is greedily matched one-to-one in stable
    descending-score order.  Unmatched detections on same-class ignore regions
    are removed from the precision denominator; crowd regions use COCO-style
    intersection-over-detection-area and can absorb multiple predictions.
    """

    confidence_threshold = _as_fraction(
        confidence_threshold, "confidence_threshold"
    )
    iou_threshold = _as_fraction(
        iou_threshold,
        "iou_threshold",
        lower=0.0,
        upper=1.0,
        lower_open=True,
    )
    max_detections = _as_int(
        max_detections, "max_detections", minimum=1
    )
    if target is not None:
        if gt_boxes is not None or gt_classes is not None:
            raise ValueError(
                "Pass either target= or explicit gt_boxes/gt_classes, not both"
            )
        if not isinstance(target, Mapping):
            raise TypeError("target must be a mapping")
        gt_boxes = target.get("eval_boxes", target.get("boxes"))
        gt_classes = target.get("eval_labels", target.get("labels"))
        if gt_iscrowd is None:
            gt_iscrowd = target.get("eval_iscrowd")
        if gt_ignore is None:
            gt_ignore = target.get("eval_ignore")
    if gt_boxes is None or gt_classes is None:
        raise ValueError("Ground-truth boxes and classes are required")

    pred_boxes = _boxes_xyxy(pred_boxes, box_format, "pred_boxes")
    gt_boxes = _boxes_xyxy(gt_boxes, box_format, "gt_boxes")
    pred_scores = _as_numpy(
        pred_scores, dtype=np.float64, name="pred_scores"
    ).reshape(-1)
    pred_classes = _class_array(pred_classes, "pred_classes")
    gt_classes = _class_array(gt_classes, "gt_classes")
    if not (
        len(pred_boxes) == len(pred_scores) == len(pred_classes)
    ):
        raise ValueError("Prediction boxes, scores, and classes lengths must match")
    if len(gt_boxes) != len(gt_classes):
        raise ValueError("Ground-truth boxes and classes lengths must match")
    if not np.isfinite(pred_scores).all():
        raise FloatingPointError("pred_scores contains NaN or infinity")
    if np.any((pred_scores < 0) | (pred_scores > 1)):
        raise ValueError("pred_scores must be probabilities in [0, 1]")
    crowd = _boolean_array(gt_iscrowd, "gt_iscrowd", len(gt_boxes))
    ignore = _boolean_array(gt_ignore, "gt_ignore", len(gt_boxes))

    keep = np.flatnonzero(pred_scores >= confidence_threshold)
    if len(keep):
        order = np.argsort(-pred_scores[keep], kind="mergesort")
        keep = keep[order[:max_detections]]
    pred_boxes = pred_boxes[keep]
    pred_classes = pred_classes[keep]

    ordinary = ~(crowd | ignore)
    evaluable_gt = int(ordinary.sum())
    matched = np.zeros(len(gt_boxes), dtype=bool)
    true_positives = 0
    false_positives = 0
    ignored_predictions = 0

    for pred_box, class_id in zip(pred_boxes, pred_classes):
        ordinary_candidates = np.flatnonzero(
            ordinary & ~matched & (gt_classes == class_id)
        )
        if len(ordinary_candidates):
            overlaps = _iou_one_to_many(
                pred_box, gt_boxes[ordinary_candidates]
            )
            best_local = int(np.argmax(overlaps))
            if overlaps[best_local] >= iou_threshold:
                matched[ordinary_candidates[best_local]] = True
                true_positives += 1
                continue

        ignore_candidates = np.flatnonzero(
            ignore & ~crowd & ~matched & (gt_classes == class_id)
        )
        if len(ignore_candidates):
            overlaps = _iou_one_to_many(
                pred_box, gt_boxes[ignore_candidates]
            )
            best_local = int(np.argmax(overlaps))
            if overlaps[best_local] >= iou_threshold:
                # COCO-style ordinary ignore regions absorb one detection;
                # crowd regions below intentionally absorb any number.
                matched[ignore_candidates[best_local]] = True
                ignored_predictions += 1
                continue
        crowd_candidates = np.flatnonzero(
            crowd & (gt_classes == class_id)
        )
        if len(crowd_candidates) and bool(
            (
                _ioa_prediction(pred_box, gt_boxes[crowd_candidates])
                >= iou_threshold
            ).any()
        ):
            ignored_predictions += 1
            continue
        false_positives += 1

    false_negatives = evaluable_gt - true_positives
    precision_denominator = true_positives + false_positives
    if precision_denominator:
        precision = true_positives / precision_denominator
    else:
        precision = 1.0 if evaluable_gt == 0 else 0.0
    recall = true_positives / evaluable_gt if evaluable_gt else 1.0
    return ImageSufficiencyScore(
        precision=float(precision),
        recall=float(recall),
        sufficiency=float(min(precision, recall)),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        ignored_predictions=ignored_predictions,
        evaluable_ground_truth=evaluable_gt,
        considered_predictions=len(pred_boxes),
    )


__all__ = [
    "DEFAULT_SAMPLING_HYP",
    "ImageSufficiencyScore",
    "normalize_sampling_hyp",
    "sampling_policy_fingerprint",
    "score_image_sufficiency",
]
