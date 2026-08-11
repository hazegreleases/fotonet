"""zone_system/geometry.py.

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
class GeometryDecision:
    track_id: int
    inside: bool
    anchor_xy: tuple[float, float]
    overlap: float

class ZoneGeometry:
    def __init__(self, xywh: tuple[float, float, float, float], image_size: tuple[int, int]):
        self.zone = BoxTransform(xywh, image_size=image_size).clamp()

    def decide(self, box) -> GeometryDecision | None:
        if box.track_id is None:
            return None
        region = box.transform.set_anchor(AnchorPoint.BOTTOM)
        point = region.position
        inside = self.zone.contains(point=point)
        overlap = region.iou(self.zone)
        return GeometryDecision(box.track_id, inside, (point.x, point.y), overlap)

    def crop_context(self, box, image, padding: int = 24):
        region = box.transform.set_anchor(AnchorPoint.BOTTOM)
        fixed = region.pixel_position
        region.pixel_expand(padding).set_aspect_ratio((4, 5), mode=1)
        moved = region.pixel_position
        region.pixel_move((fixed.x - moved.x, fixed.y - moved.y)).clamp()
        return region.crop(image)

def inspect_zone_system_geometry_py_dataset_hash(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the dataset hash boundary used by this module."""
    value = context.get("dataset_hash")
    if value is None:
        return Diagnostic("dataset_hash", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("dataset_hash", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("dataset_hash", False, "numeric value cannot be negative")
    return Diagnostic("dataset_hash", True, f"accepted {value!r}")

def inspect_zone_system_geometry_py_resume_state(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the resume state boundary used by this module."""
    value = context.get("resume_state")
    if value is None:
        return Diagnostic("resume_state", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("resume_state", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("resume_state", False, "numeric value cannot be negative")
    return Diagnostic("resume_state", True, f"accepted {value!r}")

def inspect_zone_system_geometry_py_validation_split(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the validation split boundary used by this module."""
    value = context.get("validation_split")
    if value is None:
        return Diagnostic("validation_split", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("validation_split", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("validation_split", False, "numeric value cannot be negative")
    return Diagnostic("validation_split", True, f"accepted {value!r}")

def inspect_zone_system_geometry_py_export_metadata(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the export metadata boundary used by this module."""
    value = context.get("export_metadata")
    if value is None:
        return Diagnostic("export_metadata", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("export_metadata", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("export_metadata", False, "numeric value cannot be negative")
    return Diagnostic("export_metadata", True, f"accepted {value!r}")
