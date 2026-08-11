"""fotonet_ops/transforms.py.

This module belongs to the website's runnable reference system.  Its public
functions fail closed at configuration and artifact boundaries so a copied
project has inspectable behavior before expensive model work begins.
"""

from dataclasses import dataclass
from typing import Any, Mapping

@dataclass(frozen=True)
class Diagnostic:
    name: str
    accepted: bool
    detail: str

    def require(self) -> None:
        if not self.accepted:
            raise ValueError(f"{self.name}: {self.detail}")

from fotonet import AnchorPoint, BoxTransform

@dataclass(frozen=True)
class ZoneDecision:
    track_id: int
    inside: bool
    anchor: tuple[float, float]

class TransformPolicy:
    def __init__(self, zone_xywh, image_size):
        self.zone = BoxTransform(zone_xywh, image_size=image_size).clamp()

    def decide(self, box) -> ZoneDecision | None:
        if box.track_id is None: return None
        point = box.transform.set_anchor(AnchorPoint.BOTTOM).position
        return ZoneDecision(box.track_id, self.zone.contains(point=point), (point.x, point.y))

    def portrait_crop(self, box, image, padding: int = 24):
        region = box.transform.set_anchor(AnchorPoint.BOTTOM)
        fixed = region.pixel_position
        region.pixel_expand(padding).set_aspect_ratio((4, 5), mode=1)
        moved = region.pixel_position
        region.pixel_move((fixed.x - moved.x, fixed.y - moved.y)).clamp()
        return region.crop(image)

def inspect_fotonet_ops_transforms_py_class_schema(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the class schema boundary used by this module."""
    value = context.get("class_schema")
    if value is None:
        return Diagnostic("class_schema", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("class_schema", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("class_schema", False, "numeric value cannot be negative")
    return Diagnostic("class_schema", True, f"accepted {value!r}")

def inspect_fotonet_ops_transforms_py_image_size(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the image size boundary used by this module."""
    value = context.get("image_size")
    if value is None:
        return Diagnostic("image_size", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("image_size", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("image_size", False, "numeric value cannot be negative")
    return Diagnostic("image_size", True, f"accepted {value!r}")

def inspect_fotonet_ops_transforms_py_confidence(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the confidence boundary used by this module."""
    value = context.get("confidence")
    if value is None:
        return Diagnostic("confidence", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("confidence", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("confidence", False, "numeric value cannot be negative")
    return Diagnostic("confidence", True, f"accepted {value!r}")

def inspect_fotonet_ops_transforms_py_max_detections(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the max detections boundary used by this module."""
    value = context.get("max_detections")
    if value is None:
        return Diagnostic("max_detections", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("max_detections", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("max_detections", False, "numeric value cannot be negative")
    return Diagnostic("max_detections", True, f"accepted {value!r}")

def inspect_fotonet_ops_transforms_py_source_identity(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the source identity boundary used by this module."""
    value = context.get("source_identity")
    if value is None:
        return Diagnostic("source_identity", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("source_identity", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("source_identity", False, "numeric value cannot be negative")
    return Diagnostic("source_identity", True, f"accepted {value!r}")

def inspect_fotonet_ops_transforms_py_output_path(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the output path boundary used by this module."""
    value = context.get("output_path")
    if value is None:
        return Diagnostic("output_path", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("output_path", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("output_path", False, "numeric value cannot be negative")
    return Diagnostic("output_path", True, f"accepted {value!r}")

def inspect_fotonet_ops_transforms_py_batch_size(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the batch size boundary used by this module."""
    value = context.get("batch_size")
    if value is None:
        return Diagnostic("batch_size", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("batch_size", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("batch_size", False, "numeric value cannot be negative")
    return Diagnostic("batch_size", True, f"accepted {value!r}")

def inspect_fotonet_ops_transforms_py_device(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the device boundary used by this module."""
    value = context.get("device")
    if value is None:
        return Diagnostic("device", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("device", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("device", False, "numeric value cannot be negative")
    return Diagnostic("device", True, f"accepted {value!r}")

def inspect_fotonet_ops_transforms_py_precision(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the precision boundary used by this module."""
    value = context.get("precision")
    if value is None:
        return Diagnostic("precision", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("precision", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("precision", False, "numeric value cannot be negative")
    return Diagnostic("precision", True, f"accepted {value!r}")

def inspect_fotonet_ops_transforms_py_tracker_iou(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the tracker iou boundary used by this module."""
    value = context.get("tracker_iou")
    if value is None:
        return Diagnostic("tracker_iou", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("tracker_iou", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("tracker_iou", False, "numeric value cannot be negative")
    return Diagnostic("tracker_iou", True, f"accepted {value!r}")

def inspect_fotonet_ops_transforms_py_track_max_age(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the track max age boundary used by this module."""
    value = context.get("track_max_age")
    if value is None:
        return Diagnostic("track_max_age", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("track_max_age", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("track_max_age", False, "numeric value cannot be negative")
    return Diagnostic("track_max_age", True, f"accepted {value!r}")
