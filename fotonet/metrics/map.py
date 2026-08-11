"""COCO-compatible detection metrics with explicit protocol metadata.

The previous evaluator rebuilt a synthetic 1x1 COCO dataset from transformed
labels and then reported an oracle best-F1 point as precision/recall.  This
module keeps original image dimensions and crowd flags, supports raw COCO JSON
ground truth, and labels non-canonical protocols honestly.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
import hashlib
import io
import os
import sys

import numpy as np


COCO_STANDARD_AREAS = (
    (0.0**2, 1e5**2),
    (0.0**2, 32.0**2),
    (32.0**2, 96.0**2),
    (96.0**2, 1e5**2),
)
COCOEVAL_CHUNK_DETECTION_LIMIT = 250_000


def _require_pycocotools():
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError(
            "FOTONET validation requires official pycocotools COCOeval. Install pycocotools before validating."
        ) from exc
    return COCO, COCOeval


@dataclass(frozen=True)
class CocoProtocol:
    """Metric settings that materially change an AP result."""

    max_dets: int = 100
    operating_conf: float = 0.25
    operating_iou: float = 0.50

    def __post_init__(self):
        if int(self.max_dets) < 1:
            raise ValueError(f"max_dets must be positive, got {self.max_dets}")
        if not 0.0 <= float(self.operating_conf) <= 1.0:
            raise ValueError(f"operating_conf must be in [0, 1], got {self.operating_conf}")
        if not 0.0 < float(self.operating_iou) <= 1.0:
            raise ValueError(f"operating_iou must be in (0, 1], got {self.operating_iou}")


def _as_boxes(value, name):
    boxes = np.asarray(value, dtype=np.float64)
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"{name} must have shape [N, 4], got {boxes.shape}")
    return boxes


def _as_vector(value, length, name, dtype):
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if len(array) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(array)}")
    return array


def _normalize_image_ids(image_ids, n_images):
    if image_ids is None:
        return list(range(n_images))
    if len(image_ids) != n_images:
        raise ValueError(f"image_ids has {len(image_ids)} items for {n_images} images")
    ids = [int(np.asarray(x).reshape(-1)[0]) for x in image_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("image_ids must be unique within one validation evaluator")
    return ids


def _normalize_image_sizes(image_sizes, n_images):
    if image_sizes is None:
        raise ValueError(
            "image_sizes is required for COCO metrics; fabricating 1x1 images corrupts "
            "area-range AP and is not a valid evaluation protocol"
        )
    if len(image_sizes) != n_images:
        raise ValueError(f"image_sizes has {len(image_sizes)} items for {n_images} images")
    normalized = []
    for size in image_sizes:
        values = np.asarray(size).reshape(-1)
        if len(values) != 2:
            raise ValueError(f"image size must be (height, width), got {size!r}")
        height, width = int(values[0]), int(values[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"image size must be positive, got {height}x{width}")
        normalized.append((height, width))
    return normalized


def _infer_num_classes(pred_classes, gt_classes):
    max_pred = max([int(np.max(c)) for c in pred_classes if len(c) > 0] or [-1])
    max_gt = max([int(np.max(c)) for c in gt_classes if len(c) > 0] or [-1])
    return max(max_pred, max_gt) + 1


def _as_class_vector(value, length, name):
    """Validate categorical IDs before converting them to integers.

    ``astype(np.int64)`` silently turns a malformed value such as ``0.9``
    into class zero.  That is especially dangerous in a metric path because
    it can turn an invalid prediction into a false-looking true positive.
    """
    raw = np.asarray(value).reshape(-1)
    if len(raw) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(raw)}")
    if raw.size == 0:
        return np.zeros((0,), dtype=np.int64)
    try:
        numeric = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite integer class IDs") from exc
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} must contain finite integer class IDs")
    return numeric.astype(np.int64, copy=False)


def _mean_valid(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[values > -1]
    return float(values.mean()) if values.size else 0.0


def _value_or_none(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[values > -1]
    return float(values.mean()) if values.size else None


def _xywhn_to_coco_xywh(boxes, image_size):
    boxes = _as_boxes(boxes, "boxes")
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    height, width = image_size
    x = (boxes[:, 0] - boxes[:, 2] * 0.5) * width
    y = (boxes[:, 1] - boxes[:, 3] * 0.5) * height
    w = boxes[:, 2] * width
    h = boxes[:, 3] * height
    return np.stack([x, y, w, h], axis=1)


def _xywhn_iou(box, boxes):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float64)
    x1 = box[0] - box[2] * 0.5
    y1 = box[1] - box[3] * 0.5
    x2 = box[0] + box[2] * 0.5
    y2 = box[1] + box[3] * 0.5
    bx1 = boxes[:, 0] - boxes[:, 2] * 0.5
    by1 = boxes[:, 1] - boxes[:, 3] * 0.5
    bx2 = boxes[:, 0] + boxes[:, 2] * 0.5
    by2 = boxes[:, 1] + boxes[:, 3] * 0.5
    inter = np.maximum(0.0, np.minimum(x2, bx2) - np.maximum(x1, bx1)) * np.maximum(
        0.0, np.minimum(y2, by2) - np.maximum(y1, by1)
    )
    union = np.maximum((x2 - x1) * (y2 - y1) + (bx2 - bx1) * (by2 - by1) - inter, 1e-12)
    return inter / union


def _max_dets_params(max_dets):
    max_dets = int(max_dets)
    if max_dets == 100:
        return [1, 10, 100]
    if max_dets > 100:
        return [1, 10, 100, max_dets]
    if max_dets >= 10:
        return [1, 10, max_dets]
    # COCOeval accepts repeated entries.  They are needed here because its
    # parameter shape is fixed at three max-detection slots; callers still get
    # an honestly named AR{max_dets}, never a fabricated AR10.
    return [1, max_dets, max_dets]


class CocoMapEvaluator:
    """Reusable immutable ground-truth evaluator for one validation protocol."""

    def __init__(
        self,
        gt_boxes,
        gt_classes,
        image_ids=None,
        num_classes=None,
        max_dets=100,
        *,
        image_sizes=None,
        gt_iscrowd=None,
        gt_ignore=None,
        class_to_category_id=None,
        class_names=None,
        metric_protocol=None,
        operating_conf=0.25,
        operating_iou=0.50,
    ):
        COCO, _ = _require_pycocotools()
        n_images = len(gt_boxes)
        if len(gt_classes) != n_images:
            raise ValueError("gt_boxes and gt_classes must have the same number of images")
        self.protocol = CocoProtocol(max_dets=max_dets, operating_conf=operating_conf, operating_iou=operating_iou)
        self.image_ids = _normalize_image_ids(image_ids, n_images)
        self.image_sizes = _normalize_image_sizes(image_sizes, n_images)
        self.num_classes = int(num_classes if num_classes is not None else _infer_num_classes([], gt_classes))
        if self.num_classes < 1:
            raise ValueError("num_classes must be positive")
        self.class_to_category_id = (
            {int(k): int(v) for k, v in dict(class_to_category_id).items()}
            if class_to_category_id is not None
            else {class_id: class_id for class_id in range(self.num_classes)}
        )
        expected_classes = set(range(self.num_classes))
        configured_classes = set(self.class_to_category_id)
        if configured_classes != expected_classes:
            missing_classes = sorted(expected_classes - configured_classes)
            extra_classes = sorted(configured_classes - expected_classes)
            raise ValueError(
                "class_to_category_id must map exactly the model class IDs "
                f"[0, {self.num_classes}); missing={missing_classes}, extra={extra_classes}"
            )
        self.cat_ids = [self.class_to_category_id[class_id] for class_id in range(self.num_classes)]
        if len(set(self.cat_ids)) != len(self.cat_ids):
            raise ValueError("class_to_category_id must map each model class to a unique category ID")
        self.category_to_class = {category_id: class_id for class_id, category_id in self.class_to_category_id.items()}
        self.class_names = {int(k): str(v) for k, v in (class_names or {}).items()}
        self.metric_protocol = metric_protocol or "yolo_converted_no_crowd"
        self.annotations_path = None
        self._raw_json = False
        self._gt_boxes = []
        self._gt_classes = []
        self._gt_iscrowd = []
        self._gt_ignore = []

        if gt_iscrowd is None:
            gt_iscrowd = [np.zeros((len(boxes),), dtype=bool) for boxes in gt_boxes]
        if gt_ignore is None:
            gt_ignore = [np.zeros((len(boxes),), dtype=bool) for boxes in gt_boxes]
        if len(gt_iscrowd) != n_images or len(gt_ignore) != n_images:
            raise ValueError("gt_iscrowd and gt_ignore must align with gt_boxes")

        dataset = {
            "info": {"description": "FOTONET immutable validation ground truth"},
            "licenses": [],
            "images": [
                {"id": image_id, "width": width, "height": height}
                for image_id, (height, width) in zip(self.image_ids, self.image_sizes)
            ],
            "categories": [
                {"id": category_id, "name": self.class_names.get(class_id, str(class_id))}
                for class_id, category_id in self.class_to_category_id.items()
            ],
            "annotations": [],
        }
        ann_id = 1
        for image_index, (image_id, boxes, classes, iscrowd, ignore, image_size) in enumerate(
            zip(self.image_ids, gt_boxes, gt_classes, gt_iscrowd, gt_ignore, self.image_sizes)
        ):
            boxes = _as_boxes(boxes, f"gt_boxes[{image_index}]")
            classes = _as_class_vector(classes, len(boxes), f"gt_classes[{image_index}]")
            iscrowd = _as_vector(iscrowd, len(boxes), f"gt_iscrowd[{image_index}]", bool)
            ignore = _as_vector(ignore, len(boxes), f"gt_ignore[{image_index}]", bool)
            if np.any(ignore & ~iscrowd):
                raise ValueError(
                    "pycocotools COCOeval discards ignore=1 when iscrowd=0. "
                    "Encode ignored regions as iscrowd=1 or remove them from this evaluation."
                )
            if np.any(classes < 0) or np.any(classes >= self.num_classes):
                raise ValueError(f"gt_classes[{image_index}] contains class outside [0, {self.num_classes})")
            boxes_coco = _xywhn_to_coco_xywh(boxes, image_size)
            finite = np.isfinite(boxes_coco).all(axis=1) & (boxes_coco[:, 2] > 0.0) & (boxes_coco[:, 3] > 0.0)
            if not np.all(finite):
                raise ValueError(f"gt_boxes[{image_index}] contains non-finite or empty boxes")
            self._gt_boxes.append(boxes)
            self._gt_classes.append(classes)
            self._gt_iscrowd.append(iscrowd)
            self._gt_ignore.append(ignore)
            for box, class_id, crowd, ignored in zip(boxes_coco, classes, iscrowd, ignore):
                dataset["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": self.class_to_category_id[int(class_id)],
                        "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        "area": float(box[2] * box[3]),
                        "iscrowd": int(bool(crowd)),
                        "ignore": int(bool(ignored)),
                    }
                )
                ann_id += 1

        self.has_annotations = bool(dataset["annotations"])
        self.coco_gt = COCO()
        self.coco_gt.dataset = dataset
        with contextlib.redirect_stdout(io.StringIO()):
            self.coco_gt.createIndex()
        self.fingerprint = self._fingerprint_in_memory()

    @classmethod
    def from_coco_json(
        cls,
        annotations_path,
        image_ids,
        class_to_category_id,
        *,
        class_names=None,
        max_dets=100,
        operating_conf=0.25,
        operating_iou=0.50,
    ):
        """Use untouched COCO JSON as the authoritative validation GT."""
        COCO, _ = _require_pycocotools()
        annotations_path = os.path.abspath(os.fspath(annotations_path))
        if not os.path.isfile(annotations_path):
            raise FileNotFoundError(f"COCO annotation JSON does not exist: {annotations_path}")
        evaluator = cls.__new__(cls)
        evaluator.protocol = CocoProtocol(max_dets=max_dets, operating_conf=operating_conf, operating_iou=operating_iou)
        with contextlib.redirect_stdout(io.StringIO()):
            evaluator.coco_gt = COCO(annotations_path)
        evaluator.annotations_path = annotations_path
        evaluator._raw_json = True
        evaluator.metric_protocol = "coco" if int(max_dets) == 100 else f"coco_maxDet{int(max_dets)}"
        evaluator.class_to_category_id = {int(k): int(v) for k, v in dict(class_to_category_id).items()}
        evaluator.category_to_class = {category_id: class_id for class_id, category_id in evaluator.class_to_category_id.items()}
        evaluator.num_classes = len(evaluator.class_to_category_id)
        if set(evaluator.class_to_category_id) != set(range(evaluator.num_classes)):
            raise ValueError("class_to_category_id must cover contiguous model class IDs starting at zero")
        evaluator.cat_ids = [evaluator.class_to_category_id[class_id] for class_id in range(evaluator.num_classes)]
        evaluator.class_names = {int(k): str(v) for k, v in (class_names or {}).items()}
        evaluator.image_ids = _normalize_image_ids(image_ids, len(image_ids))
        available_images = evaluator.coco_gt.imgs
        missing = [image_id for image_id in evaluator.image_ids if image_id not in available_images]
        if missing:
            raise ValueError(f"COCO image IDs not present in annotation JSON: {missing[:5]}")
        evaluator.image_sizes = [
            (int(available_images[image_id]["height"]), int(available_images[image_id]["width"]))
            for image_id in evaluator.image_ids
        ]
        category_ids = set(evaluator.coco_gt.cats)
        missing_categories = [category_id for category_id in evaluator.cat_ids if category_id not in category_ids]
        if missing_categories:
            raise ValueError(f"COCO category IDs missing from annotation JSON: {missing_categories}")
        annotation_ids = evaluator.coco_gt.getAnnIds(imgIds=evaluator.image_ids, catIds=evaluator.cat_ids)
        unsupported_ignore = [
            ann_id
            for ann_id in annotation_ids
            if bool(evaluator.coco_gt.anns[ann_id].get("ignore", 0))
            and not bool(evaluator.coco_gt.anns[ann_id].get("iscrowd", 0))
        ]
        if unsupported_ignore:
            raise ValueError(
                "COCO annotation JSON contains ignore=1, iscrowd=0 annotations. "
                "pycocotools COCOeval silently treats those as ordinary GT; encode them as iscrowd=1 instead."
            )
        evaluator._gt_boxes = evaluator._gt_classes = evaluator._gt_iscrowd = evaluator._gt_ignore = None
        evaluator.has_annotations = bool(
            annotation_ids
        )
        evaluator.fingerprint = evaluator._fingerprint_raw_json()
        return evaluator

    @property
    def max_dets(self):
        return int(self.protocol.max_dets)

    def _fingerprint_raw_json(self):
        digest = hashlib.sha256()
        digest.update(b"fotonet-coco-json-v2\0")
        with open(self.annotations_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(repr((self.image_ids, self.cat_ids, self.protocol)).encode("utf-8"))
        return digest.hexdigest()

    def _fingerprint_in_memory(self):
        digest = hashlib.sha256()
        digest.update(b"fotonet-in-memory-coco-v2\0")
        digest.update(repr((self.image_ids, self.image_sizes, self.class_to_category_id, self.metric_protocol, self.protocol)).encode("utf-8"))
        for group in (self._gt_boxes, self._gt_classes, self._gt_iscrowd, self._gt_ignore):
            for value in group:
                array = np.ascontiguousarray(np.asarray(value))
                digest.update(str(array.dtype).encode("ascii"))
                digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
                digest.update(array.tobytes())
        return digest.hexdigest()

    def _detections(self, pred_boxes, pred_scores, pred_classes):
        prepared, invalid, _ = self._prepare_predictions(
            pred_boxes,
            pred_scores,
            pred_classes,
        )
        return self._detections_from_prepared(prepared), invalid

    def _prepare_predictions(self, pred_boxes, pred_scores, pred_classes):
        """Validate predictions without expanding them into Python dictionaries."""
        if not (len(pred_boxes) == len(pred_scores) == len(pred_classes) == len(self.image_ids)):
            raise ValueError("prediction arrays must align with this evaluator's image_ids")
        invalid = {
            "invalid_box": 0,
            "invalid_class": 0,
            "non_finite_score": 0,
            # A finite prediction wholly inside letterbox padding collapses
            # to zero area when mapped back to the source image. COCOeval
            # accepts it as a zero-area FP, so audit it separately instead of
            # misclassifying a valid conservative submission as corrupt.
            "zero_area_box": 0,
        }
        prepared = []
        class_counts = np.zeros((self.num_classes,), dtype=np.int64)
        for image_index, (boxes, scores, classes) in enumerate(
            zip(pred_boxes, pred_scores, pred_classes)
        ):
            boxes = _as_boxes(boxes, f"pred_boxes[{image_index}]")
            scores = _as_vector(scores, len(boxes), f"pred_scores[{image_index}]", np.float64)
            classes = _as_class_vector(classes, len(boxes), f"pred_classes[{image_index}]")
            invalid_classes = (classes < 0) | (classes >= self.num_classes)
            if invalid_classes.any():
                invalid["invalid_class"] = int(invalid_classes.sum())
                bad_class = int(classes[np.flatnonzero(invalid_classes)[0]])
                raise ValueError(
                    f"pred_classes[{image_index}] contains class {bad_class} "
                    "outside the configured model classes"
                )
            non_finite_scores = ~np.isfinite(scores)
            if non_finite_scores.any():
                invalid["non_finite_score"] = int(non_finite_scores.sum())
                raise ValueError(f"pred_scores[{image_index}] contains a non-finite score")
            non_finite_boxes = ~np.isfinite(boxes).all(axis=1)
            if non_finite_boxes.any():
                invalid["invalid_box"] = int(non_finite_boxes.sum())
                raise ValueError(f"pred_boxes[{image_index}] contains a non-finite box")
            invalid["zero_area_box"] += int(
                ((boxes[:, 2] <= 0.0) | (boxes[:, 3] <= 0.0)).sum()
            )
            class_counts += np.bincount(classes, minlength=self.num_classes)[: self.num_classes]
            prepared.append((boxes, scores, classes))
        return prepared, invalid, class_counts

    def _detections_from_prepared(self, prepared, class_ids=None):
        """Materialize only the requested classes as COCO result dictionaries."""
        selected_classes = (
            None
            if class_ids is None
            else sorted({int(class_id) for class_id in class_ids})
        )
        detections = []
        for image_id, image_size, (boxes, scores, classes) in zip(
            self.image_ids,
            self.image_sizes,
            prepared,
        ):
            if selected_classes is None:
                keep = np.ones((len(classes),), dtype=bool)
            elif (
                selected_classes
                and selected_classes
                == list(range(selected_classes[0], selected_classes[-1] + 1))
            ):
                keep = (
                    (classes >= selected_classes[0])
                    & (classes <= selected_classes[-1])
                )
            else:
                keep = np.isin(classes, selected_classes)
            if not keep.any():
                continue
            boxes_coco = _xywhn_to_coco_xywh(boxes[keep], image_size)
            kept_scores = scores[keep]
            kept_classes = classes[keep]
            for box, score, class_id in zip(boxes_coco, kept_scores, kept_classes):
                if box[2] <= 0.0 or box[3] <= 0.0:
                    # Preserve finite degenerate predictions as zero-area FPs.
                    box = box.copy()
                    box[2:] = 0.0
                detections.append(
                    {
                        "image_id": int(image_id),
                        "category_id": self.class_to_category_id[int(class_id)],
                        "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        "score": float(score),
                    }
                )
        return detections

    def _build_detection_coco(self, detections):
        """Build the bbox-only result index without COCO.loadRes expansion.

        ``COCO.loadRes`` adds an eight-coordinate segmentation polygon to every
        bbox detection. Millions of canonical all-score detections can thereby
        consume many gigabytes even though COCOeval's bbox path never reads the
        polygon. This builds the exact bbox fields COCOeval consumes.
        """
        COCO, _ = _require_pycocotools()
        for annotation_id, detection in enumerate(detections, start=1):
            box = detection["bbox"]
            detection["id"] = annotation_id
            detection["area"] = float(box[2] * box[3])
            detection["iscrowd"] = 0
        result = COCO()
        result.dataset = {
            "info": dict(self.coco_gt.dataset.get("info", {})),
            "images": list(self.coco_gt.dataset.get("images", [])),
            "categories": list(self.coco_gt.dataset.get("categories", [])),
            "annotations": detections,
        }
        result.createIndex()
        return result

    def _run_coco_eval(self, detections, cat_ids, COCOeval):
        coco_dt = self._build_detection_coco(detections)
        evaluator = COCOeval(self.coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = list(self.image_ids)
        evaluator.params.catIds = [int(category_id) for category_id in cat_ids]
        evaluator.params.maxDets = _max_dets_params(self.max_dets)
        evaluator.evaluate()
        evaluator.accumulate()
        return evaluator

    def _category_chunks(self, class_counts):
        chunks = []
        current = []
        current_count = 0
        for class_id, count in enumerate(class_counts.tolist()):
            count = int(count)
            if (
                current
                and current_count + count > COCOEVAL_CHUNK_DETECTION_LIMIT
            ):
                chunks.append((current, current_count))
                current = []
                current_count = 0
            current.append(class_id)
            current_count += count
        if current:
            chunks.append((current, current_count))
        return chunks

    @staticmethod
    def _metric_from_precision(precision, iou_index=None, area_index=0, max_index=-1, undefined=0.0):
        if precision is None:
            return undefined
        if iou_index is None:
            values = precision[:, :, :, area_index, max_index]
        else:
            values = precision[iou_index, :, :, area_index, max_index]
        value = _value_or_none(values)
        return undefined if value is None else value

    @staticmethod
    def _metric_from_recall(recall, max_index, area_index=0):
        if recall is None:
            return 0.0
        return _mean_valid(recall[:, :, area_index, max_index])

    def _operating_totals(self, evaluator):
        """Collect exact fixed-threshold TP/FP/GT totals from COCOeval."""
        iou_index = int(np.argmin(np.abs(np.asarray(evaluator.params.iouThrs) - self.protocol.operating_iou)))
        if not np.isclose(evaluator.params.iouThrs[iou_index], self.protocol.operating_iou):
            raise ValueError(f"operating_iou={self.protocol.operating_iou} is not represented by COCOeval thresholds")
        all_area = tuple(float(x) for x in evaluator.params.areaRng[0])
        max_det = int(evaluator.params.maxDets[-1])
        totals = {
            int(category_id): {"tp": 0, "fp": 0, "gt": 0}
            for category_id in evaluator.params.catIds
        }
        for item in evaluator.evalImgs:
            if item is None:
                continue
            if tuple(float(x) for x in item["aRng"]) != all_area or int(item["maxDet"]) != max_det:
                continue
            category_id = int(item["category_id"])
            if category_id not in totals:
                continue
            gt_ignore = np.asarray(item["gtIgnore"], dtype=bool)
            totals[category_id]["gt"] += int((~gt_ignore).sum())
            scores = np.asarray(item["dtScores"], dtype=np.float64)
            if not len(scores):
                continue
            selected = scores >= self.protocol.operating_conf
            matches = np.asarray(item["dtMatches"], dtype=np.int64)[iou_index, selected] > 0
            ignored = np.asarray(item["dtIgnore"], dtype=bool)[iou_index, selected]
            totals[category_id]["tp"] += int((matches & ~ignored).sum())
            totals[category_id]["fp"] += int((~matches & ~ignored).sum())
        return totals, max_det

    def _operating_from_totals(self, totals, max_det):
        total_tp = sum(value["tp"] for value in totals.values())
        total_fp = sum(value["fp"] for value in totals.values())
        total_gt = sum(value["gt"] for value in totals.values())
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        recall = total_tp / total_gt if total_gt else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class = {}
        for category_id, values in totals.items():
            class_id = self.category_to_class[category_id]
            tp, fp, gt = values["tp"], values["fp"], values["gt"]
            class_precision = tp / (tp + fp) if (tp + fp) else 0.0
            class_recall = tp / gt if gt else None
            per_class[class_id] = {
                "num_gt": int(gt),
                "precision_at_conf": float(class_precision),
                "recall_at_conf": None if class_recall is None else float(class_recall),
            }
        return {
            "precision_at_conf": float(precision),
            "recall_at_conf": float(recall),
            "f1_at_conf": float(f1),
            "operating_conf": float(self.protocol.operating_conf),
            "operating_iou": float(self.protocol.operating_iou),
            "operating_max_dets": int(max_det),
            "per_class": per_class,
        }

    def _operating_point(self, evaluator):
        """Exact COCOeval micro P/R at one real score threshold."""
        totals, max_det = self._operating_totals(evaluator)
        return self._operating_from_totals(totals, max_det)

    def _per_class_metrics(self, precision, recall, operating):
        per_class = {}
        # Use evaluator-derived axes where possible; COCO's default sequence
        # contains 0.50 and 0.75 exactly.
        iou_values = np.asarray(getattr(self, "_last_iou_thresholds", np.arange(0.5, 1.0, 0.05)))
        iou50 = int(np.argmin(np.abs(iou_values - 0.50)))
        iou75 = int(np.argmin(np.abs(iou_values - 0.75)))
        for class_id, category_id in self.class_to_category_id.items():
            category_index = self.cat_ids.index(category_id)
            if precision is None:
                ap = ap50 = ap75 = None
            else:
                ap = _value_or_none(precision[:, :, category_index, 0, -1])
                ap50 = _value_or_none(precision[iou50, :, category_index, 0, -1])
                ap75 = _value_or_none(precision[iou75, :, category_index, 0, -1])
            ar = _value_or_none(recall[:, category_index, 0, -1]) if recall is not None else None
            op = operating["per_class"].get(class_id, {})
            per_class[class_id] = {
                "name": self.class_names.get(class_id, str(class_id)),
                "category_id": int(category_id),
                "num_gt": int(op.get("num_gt", 0)),
                "ap50_95": ap,
                "ap50": ap50,
                "ap75": ap75,
                f"ar{self.max_dets}": ar,
                "precision_at_conf": float(op.get("precision_at_conf", 0.0)),
                "recall_at_conf": op.get("recall_at_conf"),
            }
        return per_class

    def _ground_truth_counts(self):
        """Count COCOeval-valid GT for transparent zero-detection output."""
        counts = {class_id: 0 for class_id in range(self.num_classes)}
        if self._raw_json:
            annotation_ids = self.coco_gt.getAnnIds(imgIds=self.image_ids, catIds=self.cat_ids)
            for ann_id in annotation_ids:
                annotation = self.coco_gt.anns[ann_id]
                if bool(annotation.get("iscrowd", 0)):
                    continue
                class_id = self.category_to_class.get(int(annotation["category_id"]))
                if class_id is not None:
                    counts[class_id] += 1
            return counts
        for classes, crowd, ignore in zip(self._gt_classes, self._gt_iscrowd, self._gt_ignore):
            for class_id in classes[~(crowd | ignore)]:
                counts[int(class_id)] += 1
        return counts

    def _area_ground_truth_counts(self):
        """Count evaluable GT in COCO's all/small/medium/large area bins."""
        counts = [0, 0, 0, 0]

        def add_area(area):
            for index, (low, high) in enumerate(COCO_STANDARD_AREAS):
                if low <= area <= high:
                    counts[index] += 1

        if self._raw_json:
            annotation_ids = self.coco_gt.getAnnIds(imgIds=self.image_ids, catIds=self.cat_ids)
            for ann_id in annotation_ids:
                annotation = self.coco_gt.anns[ann_id]
                if bool(annotation.get("iscrowd", 0)):
                    continue
                area = float(annotation.get("area", 0.0))
                if np.isfinite(area) and area >= 0.0:
                    add_area(area)
            return counts
        for boxes, crowd, ignore, image_size in zip(
            self._gt_boxes, self._gt_iscrowd, self._gt_ignore, self.image_sizes
        ):
            height, width = image_size
            valid = ~(crowd | ignore)
            for box in boxes[valid]:
                area = float(box[2] * width * box[3] * height)
                if np.isfinite(area) and area >= 0.0:
                    add_area(area)
        return counts

    def _empty_per_class_metrics(self):
        counts = self._ground_truth_counts()
        per_class = {}
        for class_id, category_id in self.class_to_category_id.items():
            num_gt = int(counts[class_id])
            has_gt = num_gt > 0
            per_class[class_id] = {
                "name": self.class_names.get(class_id, str(class_id)),
                "category_id": int(category_id),
                "num_gt": num_gt,
                "ap50_95": 0.0 if has_gt else None,
                "ap50": 0.0 if has_gt else None,
                "ap75": 0.0 if has_gt else None,
                f"ar{self.max_dets}": 0.0 if has_gt else None,
                "precision_at_conf": 0.0,
                "recall_at_conf": 0.0 if has_gt else None,
            }
        return per_class

    def _empty_metrics(self, invalid_predictions=None):
        max_key = f"coco_AR{self.max_dets}"
        has_evaluable_gt = any(self._ground_truth_counts().values())
        area_counts = self._area_ground_truth_counts()
        return {
            "mAP50": 0.0,
            "mAP50_95": 0.0,
            "mAP75": 0.0,
            "mAP_small": 0.0 if area_counts[1] else None,
            "mAP_medium": 0.0 if area_counts[2] else None,
            "mAP_large": 0.0 if area_counts[3] else None,
            "precision_at_conf": 0.0,
            "recall_at_conf": 0.0,
            "f1_at_conf": 0.0,
            "operating_conf": float(self.protocol.operating_conf),
            "operating_iou": float(self.protocol.operating_iou),
            "operating_max_dets": self.max_dets,
            # Backwards-compatible aliases, now explicitly tied to the same
            # fixed operating point rather than an oracle best-F1 envelope.
            "precision": 0.0,
            "recall": 0.0,
            "coco_AR1": 0.0,
            "coco_AR10": 0.0 if self.max_dets >= 10 else None,
            max_key: 0.0,
            "coco_AR100": 0.0 if self.max_dets >= 100 else None,
            "coco_AR300": 0.0 if self.max_dets >= 300 else None,
            "max_dets": self.max_dets,
            "metric_protocol": self.metric_protocol,
            "metric_backend": "pycocotools",
            "ap_defined": bool(has_evaluable_gt),
            "per_class": self._empty_per_class_metrics(),
            "invalid_predictions": invalid_predictions or {
                "invalid_box": 0,
                "invalid_class": 0,
                "non_finite_score": 0,
                "zero_area_box": 0,
            },
            "evaluator_fingerprint": self.fingerprint,
        }

    def evaluate(self, pred_boxes, pred_scores, pred_classes, verbose=False):
        if not self.has_annotations:
            return self._empty_metrics()
        _, COCOeval = _require_pycocotools()
        prepared, invalid_predictions, class_counts = self._prepare_predictions(
            pred_boxes,
            pred_scores,
            pred_classes,
        )
        detection_count = int(class_counts.sum())
        if detection_count == 0:
            return self._empty_metrics(invalid_predictions)

        with contextlib.redirect_stdout(io.StringIO()):
            if detection_count <= COCOEVAL_CHUNK_DETECTION_LIMIT:
                detections = self._detections_from_prepared(prepared)
                evaluator = self._run_coco_eval(
                    detections,
                    self.cat_ids,
                    COCOeval,
                )
                precision = evaluator.eval.get("precision")
                recall = evaluator.eval.get("recall")
                operating = self._operating_point(evaluator)
                iou_thresholds = evaluator.params.iouThrs
            else:
                chunks = self._category_chunks(class_counts)
                if verbose:
                    print(
                        f"[COCOeval] {detection_count:,} detections split into "
                        f"{len(chunks)} memory-safe category chunk(s).",
                        file=sys.stderr,
                        flush=True,
                    )
                precision_parts = []
                recall_parts = []
                operating_totals = {}
                max_det = self.max_dets
                iou_thresholds = None
                for chunk_index, (class_ids, chunk_count) in enumerate(chunks, start=1):
                    if verbose:
                        print(
                            f"[COCOeval] chunk {chunk_index}/{len(chunks)}: "
                            f"classes={class_ids[0]}-{class_ids[-1]} "
                            f"detections={chunk_count:,}",
                            file=sys.stderr,
                            flush=True,
                        )
                    detections = self._detections_from_prepared(
                        prepared,
                        class_ids=class_ids,
                    )
                    category_ids = [
                        self.class_to_category_id[class_id]
                        for class_id in class_ids
                    ]
                    evaluator = self._run_coco_eval(
                        detections,
                        category_ids,
                        COCOeval,
                    )
                    precision_parts.append(evaluator.eval.get("precision"))
                    recall_parts.append(evaluator.eval.get("recall"))
                    chunk_totals, max_det = self._operating_totals(evaluator)
                    operating_totals.update(chunk_totals)
                    iou_thresholds = evaluator.params.iouThrs
                    del detections, evaluator
                precision = np.concatenate(precision_parts, axis=2)
                recall = np.concatenate(recall_parts, axis=1)
                operating = self._operating_from_totals(
                    operating_totals,
                    max_det,
                )

        self._last_iou_thresholds = np.asarray(iou_thresholds, dtype=np.float64)
        iou50 = int(np.argmin(np.abs(self._last_iou_thresholds - 0.50)))
        iou75 = int(np.argmin(np.abs(self._last_iou_thresholds - 0.75)))
        max_key = f"coco_AR{self.max_dets}"
        metrics = {
            "mAP50": self._metric_from_precision(precision, iou50),
            "mAP50_95": self._metric_from_precision(precision),
            "mAP75": self._metric_from_precision(precision, iou75),
            "mAP_small": self._metric_from_precision(precision, area_index=1, undefined=None),
            "mAP_medium": self._metric_from_precision(precision, area_index=2, undefined=None),
            "mAP_large": self._metric_from_precision(precision, area_index=3, undefined=None),
            "coco_AR1": self._metric_from_recall(recall, 0),
            "coco_AR10": self._metric_from_recall(recall, 1) if self.max_dets >= 10 else None,
            max_key: self._metric_from_recall(recall, -1),
            "coco_AR100": self._metric_from_recall(recall, 2 if self.max_dets > 100 else -1) if self.max_dets >= 100 else None,
            "coco_AR300": self._metric_from_recall(recall, -1) if self.max_dets >= 300 else None,
            "max_dets": self.max_dets,
            "metric_protocol": self.metric_protocol,
            "metric_backend": "pycocotools",
            "ap_defined": bool(any(self._ground_truth_counts().values())),
            "invalid_predictions": invalid_predictions,
            "evaluator_fingerprint": self.fingerprint,
        }
        metrics.update({key: value for key, value in operating.items() if key != "per_class"})
        metrics["precision"] = metrics["precision_at_conf"]
        metrics["recall"] = metrics["recall_at_conf"]
        metrics["per_class"] = self._per_class_metrics(precision, recall, operating)
        if verbose:
            print(
                f"{metrics['metric_protocol']}: AP50={metrics['mAP50']:.4f} "
                f"AP50-95={metrics['mAP50_95']:.4f} {max_key}={metrics[max_key]:.4f}"
            )
        return metrics


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
    *,
    image_sizes=None,
    gt_iscrowd=None,
    gt_ignore=None,
    class_to_category_id=None,
    class_names=None,
    metric_protocol=None,
    operating_conf=0.25,
    operating_iou=0.50,
):
    coco = evaluator or CocoMapEvaluator(
        gt_boxes,
        gt_classes,
        image_ids=image_ids,
        num_classes=num_classes,
        max_dets=max_dets,
        image_sizes=image_sizes,
        gt_iscrowd=gt_iscrowd,
        gt_ignore=gt_ignore,
        class_to_category_id=class_to_category_id,
        class_names=class_names,
        metric_protocol=metric_protocol,
        operating_conf=operating_conf,
        operating_iou=operating_iou,
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
    **kwargs,
):
    """Compatibility wrapper returning the historic tuple shape."""
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
        **kwargs,
    )
    return stats["mAP50"], stats["mAP50_95"], stats[f"coco_AR{max_dets}"], stats["per_class"], stats["metric_backend"]
