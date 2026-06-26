"""
COCOeval metrics for FOTONET.
Box format: normalized xywh [cx, cy, w, h].
"""
import contextlib
import io

import numpy as np


def _require_pycocotools():
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError(
            "FOTONET validation now uses official pycocotools COCOeval only. "
            "pycocotools is missing from this environment."
        ) from exc
    return COCO, COCOeval


def _xywhn_to_coco_xywh(boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    x = boxes[:, 0] - boxes[:, 2] * 0.5
    y = boxes[:, 1] - boxes[:, 3] * 0.5
    w = boxes[:, 2]
    h = boxes[:, 3]
    return np.stack([x, y, w, h], axis=1)


def _normalize_image_ids(image_ids, n_images):
    if image_ids is None:
        return list(range(n_images))
    return [int(np.asarray(x).reshape(-1)[0]) for x in image_ids]


def _infer_num_classes(pred_classes, gt_classes):
    max_pred = max([int(np.max(c)) for c in pred_classes if len(c) > 0] or [0])
    max_gt = max([int(np.max(c)) for c in gt_classes if len(c) > 0] or [0])
    return max(max_pred, max_gt) + 1


def _empty_metrics():
    return {
        "mAP50": 0.0,
        "mAP50_95": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "coco_AR100": 0.0,
        "per_class": {},
        "metric_backend": "pycocotools",
    }


def _mean_valid(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[values > -1]
    return float(values.mean()) if values.size else 0.0


class CocoMapEvaluator:
    """Reusable COCOeval ground-truth index for one validation set."""

    def __init__(self, gt_boxes, gt_classes, image_ids=None, num_classes=None, max_dets=100):
        COCO, _ = _require_pycocotools()
        self.image_ids = _normalize_image_ids(image_ids, len(gt_boxes))
        self.num_classes = int(num_classes or _infer_num_classes([], gt_classes))
        self.cat_ids = list(range(self.num_classes))
        self.max_dets = int(max_dets)

        dataset = {
            "info": {"description": "FOTO-NET in-memory validation set"},
            "licenses": [],
            "images": [{"id": int(img_id), "width": 1, "height": 1} for img_id in self.image_ids],
            "categories": [{"id": int(c), "name": str(c)} for c in self.cat_ids],
            "annotations": [],
        }

        ann_id = 1
        for img_id, boxes, classes in zip(self.image_ids, gt_boxes, gt_classes):
            boxes_coco = _xywhn_to_coco_xywh(boxes)
            classes = np.asarray(classes, dtype=np.int64)
            for box, cls in zip(boxes_coco, classes):
                w = float(max(box[2], 0.0))
                h = float(max(box[3], 0.0))
                if w <= 0.0 or h <= 0.0:
                    continue
                dataset["annotations"].append({
                    "id": ann_id,
                    "image_id": int(img_id),
                    "category_id": int(cls),
                    "bbox": [float(box[0]), float(box[1]), w, h],
                    "area": w * h,
                    "iscrowd": 0,
                })
                ann_id += 1

        self.has_annotations = bool(dataset["annotations"])
        self.coco_gt = COCO()
        self.coco_gt.dataset = dataset
        with contextlib.redirect_stdout(io.StringIO()):
            self.coco_gt.createIndex()

    def _detections(self, pred_boxes, pred_scores, pred_classes):
        detections = []
        for img_id, boxes, scores, classes in zip(self.image_ids, pred_boxes, pred_scores, pred_classes):
            boxes_coco = _xywhn_to_coco_xywh(boxes)
            scores = np.asarray(scores, dtype=np.float64)
            classes = np.asarray(classes, dtype=np.int64)
            if len(scores) > self.max_dets:
                order = np.argsort(-scores)[:self.max_dets]
                boxes_coco = boxes_coco[order]
                scores = scores[order]
                classes = classes[order]
            for box, score, cls in zip(boxes_coco, scores, classes):
                w = float(max(box[2], 0.0))
                h = float(max(box[3], 0.0))
                if w <= 0.0 or h <= 0.0:
                    continue
                detections.append({
                    "image_id": int(img_id),
                    "category_id": int(cls),
                    "bbox": [float(box[0]), float(box[1]), w, h],
                    "score": float(score),
                })
        return detections

    def evaluate(self, pred_boxes, pred_scores, pred_classes, verbose=False):
        if not self.has_annotations:
            return _empty_metrics()

        _, COCOeval = _require_pycocotools()
        detections = self._detections(pred_boxes, pred_scores, pred_classes)
        if not detections:
            return _empty_metrics()

        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = self.coco_gt.loadRes(detections)
            evaluator = COCOeval(self.coco_gt, coco_dt, "bbox")
            evaluator.params.imgIds = self.image_ids
            evaluator.params.catIds = self.cat_ids
            evaluator.params.maxDets = [1, 10, int(self.max_dets)]
            evaluator.params.areaRng = [[0.0, 1e10]]
            evaluator.params.areaRngLbl = ["all"]
            evaluator.evaluate()
            evaluator.accumulate()

        precision, recall, per_class = self._precision_recall(evaluator)
        coco_precision = evaluator.eval.get("precision")
        coco_recall = evaluator.eval.get("recall")
        ap_all = _mean_valid(coco_precision[:, :, :, 0, -1]) if coco_precision is not None else 0.0
        ap50 = _mean_valid(coco_precision[0, :, :, 0, -1]) if coco_precision is not None else 0.0
        ar100 = _mean_valid(coco_recall[:, :, 0, -1]) if coco_recall is not None else 0.0
        if verbose:
            print(f"COCOeval all-area: AP50={ap50:.4f} AP50-95={ap_all:.4f} AR100={ar100:.4f}")
        return {
            "mAP50": float(ap50),
            "mAP50_95": float(ap_all),
            "precision": float(precision),
            "recall": float(recall),
            "coco_AR100": float(ar100),
            "per_class": per_class,
            "metric_backend": "pycocotools",
        }

    def _precision_recall(self, evaluator):
        precision = evaluator.eval.get("precision")
        if precision is None:
            return 0.0, 0.0, {}

        ap50_precision = precision[0, :, :, 0, -1]
        rec_thrs = np.asarray(evaluator.params.recThrs, dtype=np.float64)
        per_class = {}
        best_p = []
        best_r = []

        for idx, cat_id in enumerate(self.cat_ids):
            vals = ap50_precision[:, idx].astype(np.float64, copy=True)
            valid = vals > -1
            if not valid.any():
                continue
            per_class[int(cat_id)] = float(vals[valid].mean())
            vals[~valid] = np.nan
            f1 = 2.0 * vals * rec_thrs / (vals + rec_thrs + 1e-16)
            if np.all(np.isnan(f1)):
                continue
            best = int(np.nanargmax(f1))
            best_p.append(float(vals[best]))
            best_r.append(float(rec_thrs[best]))

        if not best_p:
            return 0.0, 0.0, per_class
        return float(np.mean(best_p)), float(np.mean(best_r)), per_class


def compute_coco_metrics(
    pred_boxes,
    pred_scores,
    pred_classes,
    gt_boxes,
    gt_classes,
    image_ids=None,
    num_classes=None,
    max_dets=100,
    verbose=False,
    evaluator=None,
):
    coco = evaluator or CocoMapEvaluator(
        gt_boxes,
        gt_classes,
        image_ids=image_ids,
        num_classes=num_classes,
        max_dets=max_dets,
    )
    return coco.evaluate(pred_boxes, pred_scores, pred_classes, verbose=verbose)


def compute_coco_map(
    pred_boxes,
    pred_scores,
    pred_classes,
    gt_boxes,
    gt_classes,
    image_ids=None,
    num_classes=None,
    max_dets=100,
    verbose=False,
):
    """Compatibility wrapper returning the old tuple shape, backed by COCOeval only."""
    stats = compute_coco_metrics(
        pred_boxes,
        pred_scores,
        pred_classes,
        gt_boxes,
        gt_classes,
        image_ids=image_ids,
        num_classes=num_classes,
        max_dets=max_dets,
        verbose=verbose,
    )
    return stats["mAP50"], stats["mAP50_95"], stats["coco_AR100"], stats["per_class"], stats["metric_backend"]
