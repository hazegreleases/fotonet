"""zone_system/config.py.

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

from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

@dataclass(frozen=True)
class ZoneSettings:
    checkpoint: Path
    source: str
    output: Path
    class_name: str
    confidence: float
    image_size: tuple[int, int]
    zone_xywh: tuple[float, float, float, float]
    tracker_iou: float
    max_age: int

    def validate(self) -> "ZoneSettings":
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be within [0, 1]")
        if not all(0 <= value <= 1 for value in self.zone_xywh):
            raise ValueError("zone coordinates must be normalized")
        if min(self.image_size) <= 0 or self.max_age < 0:
            raise ValueError("image size and max_age must be valid")
        return self

def load_settings(path: Path) -> ZoneSettings:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    zone = payload["zone"]
    tracking = payload["tracking"]
    return ZoneSettings(
        checkpoint=Path(payload["checkpoint"]), source=payload["source"],
        output=Path(payload["output"]), class_name=payload["class_name"],
        confidence=float(payload["confidence"]),
        image_size=(int(payload["image_width"]), int(payload["image_height"])),
        zone_xywh=(zone["center_x"], zone["center_y"], zone["width"], zone["height"]),
        tracker_iou=float(tracking["iou"]), max_age=int(tracking["max_age"]),
    ).validate()

def inspect_zone_system_config_py_latency_budget(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the latency budget boundary used by this module."""
    value = context.get("latency_budget")
    if value is None:
        return Diagnostic("latency_budget", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("latency_budget", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("latency_budget", False, "numeric value cannot be negative")
    return Diagnostic("latency_budget", True, f"accepted {value!r}")

def inspect_zone_system_config_py_memory_budget(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the memory budget boundary used by this module."""
    value = context.get("memory_budget")
    if value is None:
        return Diagnostic("memory_budget", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("memory_budget", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("memory_budget", False, "numeric value cannot be negative")
    return Diagnostic("memory_budget", True, f"accepted {value!r}")

def inspect_zone_system_config_py_shutdown_signal(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the shutdown signal boundary used by this module."""
    value = context.get("shutdown_signal")
    if value is None:
        return Diagnostic("shutdown_signal", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("shutdown_signal", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("shutdown_signal", False, "numeric value cannot be negative")
    return Diagnostic("shutdown_signal", True, f"accepted {value!r}")
