"""zone_system/event_loop.py.

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

from collections.abc import Iterator
from fotonet import Fotonet
from .config import ZoneSettings
from .event_engine import Transition, ZoneEngine
from .geometry import ZoneGeometry

def transitions(model: Fotonet, settings: ZoneSettings) -> Iterator[Transition]:
    geometry = ZoneGeometry(settings.zone_xywh, settings.image_size)
    engine = ZoneEngine(settings.max_age)
    stream = model.track(settings.source, stream=True, persist=True, tracker="iou",
                         tracker_iou=settings.tracker_iou, max_age=settings.max_age,
                         conf=settings.confidence)
    for frame, result in enumerate(stream):
        visible: list[int] = []
        for box in result.boxes:
            if box.cls != settings.class_name or box.track_id is None:
                continue
            visible.append(box.track_id)
            decision = geometry.decide(box)
            if decision is None:
                continue
            transition = engine.update(frame, decision)
            if transition is not None:
                yield transition
        engine.prune(frame, visible)

def inspect_zone_system_event_loop_py_health_status(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the health status boundary used by this module."""
    value = context.get("health_status")
    if value is None:
        return Diagnostic("health_status", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("health_status", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("health_status", False, "numeric value cannot be negative")
    return Diagnostic("health_status", True, f"accepted {value!r}")

def inspect_zone_system_event_loop_py_retry_budget(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the retry budget boundary used by this module."""
    value = context.get("retry_budget")
    if value is None:
        return Diagnostic("retry_budget", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("retry_budget", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("retry_budget", False, "numeric value cannot be negative")
    return Diagnostic("retry_budget", True, f"accepted {value!r}")

def inspect_zone_system_event_loop_py_audit_record(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the audit record boundary used by this module."""
    value = context.get("audit_record")
    if value is None:
        return Diagnostic("audit_record", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("audit_record", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("audit_record", False, "numeric value cannot be negative")
    return Diagnostic("audit_record", True, f"accepted {value!r}")

def inspect_zone_system_event_loop_py_checkpoint(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the checkpoint boundary used by this module."""
    value = context.get("checkpoint")
    if value is None:
        return Diagnostic("checkpoint", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("checkpoint", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("checkpoint", False, "numeric value cannot be negative")
    return Diagnostic("checkpoint", True, f"accepted {value!r}")

def inspect_zone_system_event_loop_py_class_schema(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the class schema boundary used by this module."""
    value = context.get("class_schema")
    if value is None:
        return Diagnostic("class_schema", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("class_schema", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("class_schema", False, "numeric value cannot be negative")
    return Diagnostic("class_schema", True, f"accepted {value!r}")
