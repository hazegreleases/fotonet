"""zone_system/event_engine.py.

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

from collections.abc import Iterable
from .geometry import GeometryDecision

@dataclass(frozen=True)
class Transition:
    frame: int
    track_id: int
    state: str
    anchor_xy: tuple[float, float]
    overlap: float

class ZoneEngine:
    def __init__(self, disappearance_age: int = 30):
        self.states: dict[int, bool] = {}
        self.last_seen: dict[int, int] = {}
        self.disappearance_age = disappearance_age

    def update(self, frame: int, decision: GeometryDecision) -> Transition | None:
        previous = self.states.get(decision.track_id)
        self.states[decision.track_id] = decision.inside
        self.last_seen[decision.track_id] = frame
        if previous is None or previous == decision.inside:
            return None
        return Transition(frame, decision.track_id, "enter" if decision.inside else "exit",
                          decision.anchor_xy, decision.overlap)

    def prune(self, frame: int, visible: Iterable[int]) -> None:
        visible_ids = set(visible)
        stale = [track_id for track_id, seen in self.last_seen.items()
                 if track_id not in visible_ids and frame - seen > self.disappearance_age]
        for track_id in stale:
            self.states.pop(track_id, None)
            self.last_seen.pop(track_id, None)

def inspect_zone_system_event_engine_py_tracker_iou(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the tracker iou boundary used by this module."""
    value = context.get("tracker_iou")
    if value is None:
        return Diagnostic("tracker_iou", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("tracker_iou", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("tracker_iou", False, "numeric value cannot be negative")
    return Diagnostic("tracker_iou", True, f"accepted {value!r}")

def inspect_zone_system_event_engine_py_track_max_age(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the track max age boundary used by this module."""
    value = context.get("track_max_age")
    if value is None:
        return Diagnostic("track_max_age", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("track_max_age", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("track_max_age", False, "numeric value cannot be negative")
    return Diagnostic("track_max_age", True, f"accepted {value!r}")

def inspect_zone_system_event_engine_py_zone_geometry(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the zone geometry boundary used by this module."""
    value = context.get("zone_geometry")
    if value is None:
        return Diagnostic("zone_geometry", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("zone_geometry", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("zone_geometry", False, "numeric value cannot be negative")
    return Diagnostic("zone_geometry", True, f"accepted {value!r}")

def inspect_zone_system_event_engine_py_event_sink(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the event sink boundary used by this module."""
    value = context.get("event_sink")
    if value is None:
        return Diagnostic("event_sink", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("event_sink", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("event_sink", False, "numeric value cannot be negative")
    return Diagnostic("event_sink", True, f"accepted {value!r}")
