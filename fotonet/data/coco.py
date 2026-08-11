"""COCO JSON dataset adapter with immutable evaluation ground truth.

The training tensors use contiguous model class indices.  The original COCO
annotation file remains the authority for validation, including sparse category
IDs, image dimensions, areas, and crowd annotations.
"""
from __future__ import annotations

from collections import defaultdict
import json
import os

import numpy as np
from PIL import Image

from fotonet.data.audit import AnnotationAudit
from fotonet.data.dataset import DetectionDataset


class CocoDetectionDataset(DetectionDataset):
    """A detection image pipeline backed by a COCO annotation JSON file."""

    def __init__(
        self,
        annotations,
        images=None,
        *,
        category_id_to_class=None,
        annotation_policy="fix",
        **kwargs,
    ):
        annotation_path = os.path.abspath(os.fspath(annotations))
        if not os.path.isfile(annotation_path):
            raise FileNotFoundError(f"COCO annotation JSON does not exist: {annotation_path}")
        if images is None:
            images = os.path.dirname(annotation_path)
        image_root = os.path.abspath(os.fspath(images))
        if not os.path.isdir(image_root):
            raise NotADirectoryError(f"COCO image directory does not exist: {image_root}")

        audit = AnnotationAudit(policy=annotation_policy, allow_missing_labels=True)
        # The raw JSON is the official COCO source only while every selected
        # annotation reaches the model unchanged.  Under a permissive audit
        # policy we may need to clip or discard malformed rows; in that case
        # evaluating the untouched JSON would score different ground truth than
        # the model received.  Keep an explicit fallback protocol instead of
        # publishing a deceptively canonical AP number.
        self._raw_coco_evaluation_safe = True
        self._raw_coco_evaluation_reasons = []

        def mark_noncanonical(reason):
            self._raw_coco_evaluation_safe = False
            if reason not in self._raw_coco_evaluation_reasons:
                self._raw_coco_evaluation_reasons.append(reason)

        try:
            with open(annotation_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read COCO annotations {annotation_path}: {exc}") from exc

        images_payload = payload.get("images")
        categories_payload = payload.get("categories")
        annotations_payload = payload.get("annotations")
        if not isinstance(images_payload, list) or not isinstance(categories_payload, list) or not isinstance(annotations_payload, list):
            raise ValueError("COCO JSON must contain list-valued images, categories, and annotations fields")

        try:
            category_id_values = [int(category["id"]) for category in categories_payload]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Every COCO category must contain an integer 'id'") from exc
        category_ids = sorted(set(category_id_values))
        if not category_ids:
            raise ValueError("COCO JSON has no categories")
        if len(category_ids) != len(category_id_values):
            raise ValueError("COCO JSON contains duplicate category IDs")
        if category_id_to_class is None:
            category_id_to_class = {cat_id: idx for idx, cat_id in enumerate(category_ids)}
        else:
            category_id_to_class = {int(key): int(value) for key, value in dict(category_id_to_class).items()}
        if set(category_id_to_class) != set(category_ids):
            raise ValueError("category_id_to_class must map every and only category in the COCO JSON")
        class_to_category_id = {}
        for category_id, class_id in category_id_to_class.items():
            if class_id in class_to_category_id:
                raise ValueError(f"Multiple COCO categories map to model class {class_id}")
            class_to_category_id[class_id] = category_id
        class_ids = sorted(class_to_category_id)
        if class_ids != list(range(len(class_ids))):
            raise ValueError("COCO category_id_to_class values must be contiguous model classes starting at zero")

        requested_nc = kwargs.get("num_classes")
        if requested_nc is not None:
            requested_nc = int(requested_nc)
            if len(class_ids) != requested_nc:
                raise ValueError(
                    f"COCO categories map to {len(class_ids)} classes but num_classes/model nc is {requested_nc}. "
                    "Use a model/config with exactly the same class set."
                )

        records = {}
        ordered_paths = []
        image_by_id = {}
        for image in images_payload:
            try:
                image_id = int(image["id"])
                file_name = os.fspath(image["file_name"])
                width = int(image["width"])
                height = int(image["height"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Malformed COCO image record: {image!r}") from exc
            if width <= 0 or height <= 0:
                audit.issue("invalid_coco_image_size", annotation_path, detail=f"image_id={image_id}, {width}x{height}")
                continue
            if image_id in image_by_id:
                raise ValueError(f"COCO JSON contains duplicate image ID: {image_id}")
            path = os.path.abspath(os.path.join(image_root, file_name))
            try:
                if os.path.commonpath([image_root, path]) != image_root:
                    raise ValueError("file_name escapes the configured image root")
            except ValueError:
                raise ValueError(f"Invalid COCO file_name for image {image_id}: {file_name!r}")
            if not os.path.isfile(path):
                audit.issue("missing_coco_image", path, detail=f"image_id={image_id}")
                continue
            try:
                with Image.open(path) as image_file:
                    actual_width, actual_height = image_file.size
            except (OSError, ValueError) as exc:
                audit.issue("unreadable_coco_image", path, detail=f"image_id={image_id}: {exc}")
                continue
            if (actual_width, actual_height) != (width, height):
                audit.issue(
                    "coco_image_dimension_mismatch",
                    path,
                    detail=(
                        f"image_id={image_id}; JSON={width}x{height}, "
                        f"file={actual_width}x{actual_height}"
                    ),
                )
                continue
            if path in records:
                raise ValueError(f"COCO JSON maps multiple image records to the same file: {path}")
            image_by_id[image_id] = (path, width, height)
            ordered_paths.append(path)
            records[path] = {
                "boxes": [],
                "labels": [],
                "eval_boxes": [],
                "eval_labels": [],
                "eval_iscrowd": [],
                "eval_ignore": [],
                "image_id": image_id,
                "orig_area": [],
            }

        annotations_by_image = defaultdict(list)
        annotation_ids = set()
        for annotation_index, annotation in enumerate(annotations_payload, start=1):
            try:
                annotation_image_id = int(annotation["image_id"])
            except (KeyError, TypeError, ValueError):
                audit.issue("malformed_coco_annotation", annotation_path, detail=repr(annotation))
                mark_noncanonical("malformed annotation image_id")
                continue
            try:
                annotation_id = annotation["id"]
                if isinstance(annotation_id, bool) or int(annotation_id) != annotation_id:
                    raise ValueError
                annotation_id = int(annotation_id)
            except (KeyError, TypeError, ValueError):
                audit.issue(
                    "invalid_coco_annotation_id",
                    annotation_path,
                    detail=f"annotation #{annotation_index} needs a unique integer id",
                )
                # pycocotools indexes the whole JSON before it filters image
                # IDs, so an invalid ID anywhere makes raw evaluation unsafe.
                mark_noncanonical("invalid annotation id")
            else:
                if annotation_id in annotation_ids:
                    audit.issue(
                        "duplicate_coco_annotation_id",
                        annotation_path,
                        detail=f"annotation id={annotation_id}",
                    )
                    mark_noncanonical("duplicate annotation id")
                annotation_ids.add(annotation_id)
            annotations_by_image[annotation_image_id].append(annotation)

        unknown_annotation_images = sorted(set(annotations_by_image) - set(image_by_id))
        if unknown_annotation_images:
            audit.issue(
                "annotation_for_missing_coco_image",
                annotation_path,
                detail=f"image_id(s)={unknown_annotation_images[:5]}",
            )

        for image_id, (path, width, height) in image_by_id.items():
            record = records[path]
            for annotation in annotations_by_image.get(image_id, []):
                try:
                    category_id = int(annotation["category_id"])
                    x, y, bw, bh = (float(value) for value in annotation["bbox"])
                except (KeyError, TypeError, ValueError) as exc:
                    audit.issue("malformed_coco_bbox", annotation_path, detail=f"image_id={image_id}: {exc}")
                    mark_noncanonical("malformed bbox")
                    continue
                values = np.asarray([x, y, bw, bh], dtype=np.float64)
                if not np.isfinite(values).all() or bw <= 0.0 or bh <= 0.0:
                    audit.issue("invalid_coco_bbox", annotation_path, detail=f"image_id={image_id}")
                    mark_noncanonical("invalid bbox")
                    continue
                if category_id not in category_id_to_class:
                    audit.issue("unmapped_coco_category", annotation_path, detail=f"image_id={image_id}, category_id={category_id}")
                    mark_noncanonical("unmapped category")
                    continue
                iscrowd = bool(annotation.get("iscrowd", 0))
                ignore = bool(annotation.get("ignore", 0))
                if ignore and not iscrowd:
                    audit.issue(
                        "nonstandard_coco_ignore",
                        annotation_path,
                        detail=f"image_id={image_id}; stock COCOeval only honors iscrowd",
                    )
                    mark_noncanonical("nonstandard ignore flag")
                    continue
                x1 = np.clip(x, 0.0, float(width))
                y1 = np.clip(y, 0.0, float(height))
                x2 = np.clip(x + bw, 0.0, float(width))
                y2 = np.clip(y + bh, 0.0, float(height))
                if x2 <= x1 or y2 <= y1:
                    audit.issue("clipped_empty_coco_bbox", annotation_path, detail=f"image_id={image_id}")
                    mark_noncanonical("empty clipped bbox")
                    continue
                if x1 != x or y1 != y or x2 != x + bw or y2 != y + bh:
                    audit.issue("clipped_coco_bbox", annotation_path, detail=f"image_id={image_id}", fatal=False)
                    mark_noncanonical("bbox clipped to image bounds")
                area = annotation.get("area")
                try:
                    area = float(area)
                    if not np.isfinite(area) or area < 0.0:
                        raise ValueError
                except (TypeError, ValueError):
                    audit.issue(
                        "invalid_coco_area",
                        annotation_path,
                        detail=f"image_id={image_id}; raw COCO evaluation needs a finite non-negative area",
                    )
                    mark_noncanonical("missing or invalid annotation area")
                    area = float((x2 - x1) * (y2 - y1))
                normalized = [
                    float((x1 + x2) * 0.5 / width),
                    float((y1 + y2) * 0.5 / height),
                    float((x2 - x1) / width),
                    float((y2 - y1) / height),
                ]
                class_id = category_id_to_class[category_id]
                record["eval_boxes"].append(normalized)
                record["eval_labels"].append(class_id)
                record["eval_iscrowd"].append(iscrowd)
                record["eval_ignore"].append(ignore)
                record["orig_area"].append(area)
                if not iscrowd and not ignore:
                    record["boxes"].append(normalized)
                    record["labels"].append(class_id)

        # The base class owns decoding, caching, and augmentation.  COCO
        # targets override YOLO text loading below, so no label cache is needed.
        kwargs = dict(kwargs)
        kwargs["cache_labels"] = False
        kwargs["allow_missing_labels"] = True
        kwargs["annotation_policy"] = annotation_policy
        super().__init__(ordered_paths, **kwargs)

        self.annotation_audit = audit
        self.is_coco = True
        self.coco_annotations_path = annotation_path
        self.coco_image_root = image_root
        self.category_id_to_class = category_id_to_class
        self.class_to_category_id = class_to_category_id
        self.class_names = {
            category_id_to_class[int(category["id"])]: str(category.get("name", category["id"]))
            for category in categories_payload
            if int(category["id"]) in category_id_to_class
        }
        self._coco_evaluator_cache = {}
        self._coco_evaluator_signature = self._annotation_signature()
        self.coco_metric_protocol = (
            "coco" if self._raw_coco_evaluation_safe else "coco_sanitized_noncanonical"
        )
        self.coco_metric_protocol_notes = tuple(self._raw_coco_evaluation_reasons)
        self._coco_targets_by_path = {}
        self._coco_targets_by_image_id = {}
        self._coco_image_sizes_by_id = {}
        for path in self.img_files:
            record = records[path]
            target = {
                "boxes": np.asarray(record["boxes"], dtype=np.float32).reshape(-1, 4),
                "labels": np.asarray(record["labels"], dtype=np.int64),
                "eval_boxes": np.asarray(record["eval_boxes"], dtype=np.float32).reshape(-1, 4),
                "eval_labels": np.asarray(record["eval_labels"], dtype=np.int64),
                "eval_iscrowd": np.asarray(record["eval_iscrowd"], dtype=bool),
                "eval_ignore": np.asarray(record["eval_ignore"], dtype=bool),
                "image_id": int(record["image_id"]),
                "orig_area": np.asarray(record["orig_area"], dtype=np.float64),
                "coco_category_ids": dict(self.class_to_category_id),
                "metric_protocol": self.coco_metric_protocol,
                "coco_annotations_path": self.coco_annotations_path,
            }
            self._coco_targets_by_path[path] = target
            self._coco_targets_by_image_id[int(record["image_id"])] = target
            _, width, height = image_by_id[int(record["image_id"])]
            self._coco_image_sizes_by_id[int(record["image_id"])] = (int(height), int(width))

    def _annotation_signature(self):
        stat = os.stat(self.coco_annotations_path)
        return (int(stat.st_size), int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))))

    def get_coco_evaluator(self, image_ids, *, max_dets=100, operating_conf=0.25, operating_iou=0.50):
        """Return an immutable raw-COCO evaluator cached by protocol and file version.

        Rebuilding ``COCO`` from a large JSON each epoch can dominate validation
        time.  This cache is safe because it keys every material protocol
        setting and invalidates as soon as the annotation file changes.
        """
        signature = self._annotation_signature()
        if signature != self._coco_evaluator_signature:
            self._coco_evaluator_cache.clear()
            self._coco_evaluator_signature = signature
        key = (
            tuple(int(image_id) for image_id in image_ids),
            int(max_dets),
            float(operating_conf),
            float(operating_iou),
        )
        evaluator = self._coco_evaluator_cache.get(key)
        if evaluator is None:
            from fotonet.metrics.map import CocoMapEvaluator

            if self._raw_coco_evaluation_safe:
                evaluator = CocoMapEvaluator.from_coco_json(
                    self.coco_annotations_path,
                    key[0],
                    self.class_to_category_id,
                    class_names=self.class_names,
                    max_dets=key[1],
                    operating_conf=key[2],
                    operating_iou=key[3],
                )
            else:
                try:
                    targets = [self._coco_targets_by_image_id[int(image_id)] for image_id in key[0]]
                    image_sizes = [self._coco_image_sizes_by_id[int(image_id)] for image_id in key[0]]
                except KeyError as exc:
                    raise ValueError(f"COCO image ID {exc.args[0]} is not available in this dataset") from exc
                evaluator = CocoMapEvaluator(
                    [target["eval_boxes"] for target in targets],
                    [target["eval_labels"] for target in targets],
                    image_ids=key[0],
                    image_sizes=image_sizes,
                    num_classes=len(self.class_to_category_id),
                    max_dets=key[1],
                    gt_iscrowd=[target["eval_iscrowd"] for target in targets],
                    gt_ignore=[target["eval_ignore"] for target in targets],
                    class_to_category_id=self.class_to_category_id,
                    class_names=self.class_names,
                    metric_protocol=self.coco_metric_protocol,
                    operating_conf=key[2],
                    operating_iou=key[3],
                )
                evaluator.sanitization_reasons = self.coco_metric_protocol_notes
            self._coco_evaluator_cache[key] = evaluator
        return evaluator

    def _get_label(self, idx):
        target = self._coco_targets_by_path[self.img_files[idx]]
        target = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in target.items()
        }
        return target
