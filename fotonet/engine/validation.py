"""Shared public/trainer validation path.

Keeping this outside either facade prevents ``FOTONET.val()`` from quietly
using a different data/metric protocol than epoch validation.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from fotonet.data.augmentations.image import undo_letterbox_boxes_xywhn
from fotonet.metrics.map import CocoMapEvaluator
from fotonet.metrics.ranking import score_iou_diagnostics


def _as_numpy(value, dtype=None):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        # NumPy has no bfloat16 dtype. Native BF16 deployment models are valid
        # validation targets, so stage their metric values as float32 rather
        # than failing after a correct forward pass.
        if value.dtype == torch.bfloat16:
            value = value.float()
        value = value.numpy()
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _target_value(target, key, fallback):
    value = target.get(key, fallback)
    return _as_numpy(value)


def _limit_per_class(boxes, scores, classes, max_dets):
    """Exact COCO-safe prefilter: never globally erase another class."""
    max_dets = int(max_dets)
    if len(scores) == 0:
        return boxes, scores, classes
    keep = []
    for class_id in np.unique(classes):
        class_indices = np.flatnonzero(classes == class_id)
        if len(class_indices) > max_dets:
            local = np.argsort(-scores[class_indices], kind="mergesort")[:max_dets]
            class_indices = class_indices[local]
        keep.append(class_indices)
    indices = np.concatenate(keep) if keep else np.zeros((0,), dtype=np.int64)
    # A stable score order makes diagnostics reproducible. COCOeval re-sorts
    # inside each image/category, so this does not change AP semantics.
    if len(indices):
        indices = indices[np.argsort(-scores[indices], kind="mergesort")]
    return boxes[indices], scores[indices], classes[indices]


def _root_dataset(dataset):
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def _normalize_class_names(names):
    if names is None:
        return None
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    raise TypeError("class_names must be a mapping, list, tuple, or None")


def _prediction_tensor(outputs, nc):
    """Normalize native inference output without accepting ambiguous shapes."""
    if isinstance(outputs, dict):
        logits = outputs.get("pred_logits_o2o")
        boxes = outputs.get("pred_boxes_o2o")
        if logits is None or boxes is None:
            raise ValueError("Validation model dict output must contain pred_logits_o2o and pred_boxes_o2o")
        if logits.shape[:-1] != boxes.shape[:-1] or logits.shape[-1] != int(nc) or boxes.shape[-1] != 4:
            raise ValueError("Validation dict output has incompatible logits/boxes shapes")
        return torch.cat((logits, boxes), dim=-1)
    if isinstance(outputs, (tuple, list)):
        if len(outputs) != 1:
            raise ValueError("Validation model must return one prediction tensor, not multiple outputs")
        outputs = outputs[0]
    if not torch.is_tensor(outputs):
        raise TypeError(f"Validation model output must be a tensor/dict, got {type(outputs).__name__}")
    return outputs


def _stage_prediction_rows(output, nc, conf):
    """Transfer one image's metric inputs without a CUDA scalar read.

    Validation ultimately evaluates predictions on the host.  Checking
    ``torch.isfinite(outputs).all()`` in a Python conditional before that
    transfer forces an extra device synchronization for every batch.  Instead
    we carry a per-row finiteness sentinel alongside the metric values.  Any
    invalid row is deliberately retained even if its score is below ``conf``;
    the host then rejects it before it can affect AP.  That is equivalent to
    the former whole-output check while avoiding a separate CUDA-to-host
    scalar transfer.

    Class indices stay in their native integer dtype rather than being packed
    into the floating payload, so arbitrarily large valid class IDs cannot
    lose precision on half-precision validation models.
    """
    scores_t, classes_t = output[:, :nc].sigmoid().max(1)
    boxes_t = output[:, nc : nc + 4]

    # ``all(dim=1)`` stays on the model device.  Do not turn it into a Python
    # bool here: the selected rows below are already transferred for metrics.
    finite_rows_t = torch.isfinite(output).all(dim=1)
    keep = (scores_t >= conf) | ~finite_rows_t

    # Pack boxes, score, and the finite sentinel into one D2H transfer.  The
    # class IDs use a second, integer-preserving transfer only after the
    # floating payload has proved valid.
    payload_t = torch.cat(
        (boxes_t, scores_t.unsqueeze(1), finite_rows_t.to(dtype=boxes_t.dtype).unsqueeze(1)),
        dim=1,
    )[keep]
    payload = _as_numpy(payload_t, np.float64)
    if not np.all(payload[:, 5] == 1.0):
        raise FloatingPointError("Validation model output contains NaN or infinity; metrics would be invalid")

    classes = _as_numpy(classes_t[keep], np.int64)
    return payload[:, :4], payload[:, 4], classes


@torch.inference_mode()
def evaluate_detection_model(
    model,
    loader,
    device,
    nc,
    *,
    conf=0.0,
    coco_max_dets=100,
    operating_conf=0.25,
    operating_iou=0.50,
    amp=False,
    class_names=None,
    progress=False,
    progress_interval=100,
):
    """Evaluate a detection model under one explicit, reproducible protocol."""
    conf = float(conf)
    if not 0.0 <= conf <= 1.0:
        raise ValueError(f"validation confidence must be in [0, 1], got {conf}")
    operating_conf = float(operating_conf)
    if not 0.0 <= operating_conf <= 1.0:
        raise ValueError(f"operating_conf must be in [0, 1], got {operating_conf}")
    if conf > operating_conf:
        raise ValueError(
            "validation confidence floor cannot exceed operating_conf: that would make the reported "
            "operating-point precision/recall use a different effective threshold"
        )
    coco_max_dets = int(coco_max_dets)
    if coco_max_dets < 1:
        raise ValueError(f"coco_max_dets must be positive, got {coco_max_dets}")
    progress_interval = max(int(progress_interval), 1)
    device = torch.device(device)
    source_dataset = _root_dataset(loader.dataset)
    if bool(getattr(source_dataset, "augment", False)):
        raise ValueError(
            "Validation requires a dataset built with augment=False. "
            "Augmented pixels cannot be scored against immutable source-image ground truth."
        )
    was_training = bool(model.training)
    model.eval()
    try:
        parameters = model.parameters()
        model_dtype = next(parameter for parameter in parameters if parameter.is_floating_point()).dtype
    except (AttributeError, StopIteration):
        model_dtype = getattr(model, "input_torch_dtype", torch.float32)

    preds_b, preds_s, preds_c = [], [], []
    gts_b, gts_c, gts_iscrowd, gts_ignore = [], [], [], []
    image_ids, image_sizes = [], []
    n_images = 0
    total_images = len(loader.dataset)
    next_progress = progress_interval
    inference_started = time.monotonic()
    if progress:
        print(
            f"[validate] inference started: 0/{total_images} images",
            flush=True,
        )
    try:
        for imgs, targets in loader:
            imgs = imgs.to(device, dtype=model_dtype, non_blocking=device.type == "cuda")
            n_images += int(imgs.shape[0])
            with torch.amp.autocast(device.type, enabled=bool(amp and device.type == "cuda"), dtype=torch.float16):
                outputs = model(imgs)
            outputs = _prediction_tensor(outputs, nc)
            if outputs.ndim != 3:
                raise ValueError(f"Model validation output must have shape [B, N, nc + 4], got {tuple(outputs.shape)}")
            if outputs.shape[-1] < int(nc) + 4:
                raise ValueError(f"Model output has {outputs.shape[-1]} channels but validation needs nc + 4 = {int(nc) + 4}")
            if outputs.shape[0] != len(targets):
                raise ValueError(f"Model returned {outputs.shape[0]} predictions for a batch of {len(targets)} targets")

            for batch_index, target in enumerate(targets):
                output = outputs[batch_index]
                boxes, scores, classes = _stage_prediction_rows(output, nc, conf)

                orig_shape = _target_value(target, "orig_shape", np.asarray(imgs.shape[-2:], dtype=np.int64)).astype(np.int64)
                boxes = undo_letterbox_boxes_xywhn(boxes, orig_shape, tuple(int(x) for x in imgs.shape[-2:]))
                boxes, scores, classes = _limit_per_class(boxes, scores, classes, coco_max_dets)
                preds_b.append(boxes)
                preds_s.append(scores)
                preds_c.append(classes)

                gt_boxes = _target_value(target, "eval_boxes", target["boxes"]).astype(np.float64)
                gt_classes = _target_value(target, "eval_labels", target["labels"]).astype(np.int64)
                crowd = _target_value(target, "eval_iscrowd", np.zeros((len(gt_boxes),), dtype=bool)).astype(bool)
                ignore = _target_value(target, "eval_ignore", np.zeros((len(gt_boxes),), dtype=bool)).astype(bool)
                if not (len(gt_boxes) == len(gt_classes) == len(crowd) == len(ignore)):
                    raise ValueError("Validation target eval_boxes/eval_labels/crowd/ignore lengths must match")
                gts_b.append(gt_boxes)
                gts_c.append(gt_classes)
                gts_iscrowd.append(crowd)
                gts_ignore.append(ignore)
                image_ids.append(int(_target_value(target, "image_id", [len(image_ids)]).reshape(-1)[0]))
                image_sizes.append(tuple(int(x) for x in orig_shape.reshape(-1)[:2]))
            if progress and (n_images >= next_progress or n_images >= total_images):
                elapsed = max(time.monotonic() - inference_started, 1e-9)
                rate = n_images / elapsed
                remaining = max(total_images - n_images, 0)
                eta = remaining / rate if rate > 0.0 else 0.0
                print(
                    f"[validate] inference {n_images}/{total_images} "
                    f"({100.0 * n_images / max(total_images, 1):.1f}%) "
                    f"{rate:.1f} images/s ETA {eta / 60.0:.1f} min",
                    flush=True,
                )
                while next_progress <= n_images:
                    next_progress += progress_interval
    finally:
        model.train(was_training)

    evaluable_gt = int(sum((~(crowd | ignore)).sum() for crowd, ignore in zip(gts_iscrowd, gts_ignore)))
    if evaluable_gt == 0:
        raise ValueError(
            "Validation contains no non-crowd, non-ignored ground-truth boxes. "
            "AP is undefined for an all-negative validation split; configure an evaluable validation dataset."
        )

    if progress:
        print("[validate] inference complete; preparing COCOeval...", flush=True)
    if getattr(source_dataset, "is_coco", False):
        if hasattr(source_dataset, "get_coco_evaluator"):
            evaluator = source_dataset.get_coco_evaluator(
                image_ids,
                max_dets=coco_max_dets,
                operating_conf=operating_conf,
                operating_iou=operating_iou,
            )
        else:
            evaluator = CocoMapEvaluator.from_coco_json(
                source_dataset.coco_annotations_path,
                image_ids,
                source_dataset.class_to_category_id,
                class_names=getattr(source_dataset, "class_names", None),
                max_dets=coco_max_dets,
                operating_conf=operating_conf,
                operating_iou=operating_iou,
            )
    else:
        evaluator = CocoMapEvaluator(
            gts_b,
            gts_c,
            image_ids=image_ids,
            image_sizes=image_sizes,
            num_classes=int(nc),
            max_dets=coco_max_dets,
            gt_iscrowd=gts_iscrowd,
            gt_ignore=gts_ignore,
            class_names=_normalize_class_names(class_names),
            metric_protocol="yolo_converted_no_crowd",
            operating_conf=operating_conf,
            operating_iou=operating_iou,
        )
    if progress:
        print("[validate] running official pycocotools COCOeval...", flush=True)
    metrics = evaluator.evaluate(preds_b, preds_s, preds_c, verbose=progress)
    # Dense-scene operating-point metrics are intentionally computed from the
    # already staged predictions, so they add no model inference or COCOeval
    # pass. Ten evaluable instances is a transparent, fixed scene definition.
    from fotonet.engine.sampling import score_image_sufficiency

    dense_totals = {
        "images": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
    }
    for pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, crowd, ignore in zip(
        preds_b,
        preds_s,
        preds_c,
        gts_b,
        gts_c,
        gts_iscrowd,
        gts_ignore,
    ):
        if int((~(crowd | ignore)).sum()) < 10:
            continue
        score = score_image_sufficiency(
            pred_boxes,
            pred_scores,
            pred_classes,
            gt_boxes,
            gt_classes,
            gt_iscrowd=crowd,
            gt_ignore=ignore,
            confidence_threshold=operating_conf,
            iou_threshold=operating_iou,
            max_detections=coco_max_dets,
            box_format="xywh",
        )
        dense_totals["images"] += 1
        dense_totals["true_positives"] += score.true_positives
        dense_totals["false_positives"] += score.false_positives
        dense_totals["false_negatives"] += score.false_negatives
    dense_tp = dense_totals["true_positives"]
    dense_fp = dense_totals["false_positives"]
    dense_fn = dense_totals["false_negatives"]
    metrics.update({
        "dense_scene_min_objects": 10,
        "dense_scene_images": dense_totals["images"],
        "dense_scene_precision_at_conf": (
            dense_tp / (dense_tp + dense_fp)
            if dense_tp + dense_fp
            else 0.0
        ),
        "dense_scene_recall_at_conf": (
            dense_tp / (dense_tp + dense_fn)
            if dense_tp + dense_fn
            else None
        ),
    })
    if progress:
        print("[validate] COCOeval complete.", flush=True)
    sanitization_reasons = getattr(evaluator, "sanitization_reasons", ())
    if sanitization_reasons:
        metrics["metric_protocol_notes"] = list(sanitization_reasons)
    # COCO AP ranks all submitted detections and lets maxDets perform the only
    # score-based truncation.  An explicit nonzero floor remains available for
    # constrained diagnostics, but must never masquerade as the unfiltered
    # protocol used by the default path.
    metrics["ap_score_floor"] = conf
    metrics["ap_uses_all_scores"] = bool(conf == 0.0)
    if conf > 0.0:
        metrics["metric_protocol"] = f"{metrics['metric_protocol']}_score_floor{conf:g}"

    # Ranking diagnostics use only ordinary GT: crowd/ignore regions should not
    # make a detector look better or worse in this auxiliary signal.
    rank_boxes = []
    rank_classes = []
    for boxes, classes, crowd, ignore in zip(gts_b, gts_c, gts_iscrowd, gts_ignore):
        keep = ~(crowd | ignore)
        rank_boxes.append(boxes[keep])
        rank_classes.append(classes[keep])
    ranking = score_iou_diagnostics(preds_b, preds_s, preds_c, rank_boxes, rank_classes, topk=coco_max_dets)
    metrics.update(ranking)
    metrics["val_images"] = int(n_images)
    metrics["evaluable_ground_truth"] = evaluable_gt
    audit = getattr(source_dataset, "annotation_audit", None)
    if audit is not None:
        metrics["annotation_audit"] = audit.summary()
    return metrics
