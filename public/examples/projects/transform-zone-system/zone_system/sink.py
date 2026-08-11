"""zone_system/sink.py.

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

import json
import os
from dataclasses import asdict
from pathlib import Path
from .event_engine import Transition

class JsonlSink:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, transition: Transition) -> None:
        payload = json.dumps(asdict(transition), sort_keys=True)
        with self.path.open("a", encoding="utf-8") as stream:
            print(payload, file=stream)
            stream.flush()
            os.fsync(stream.fileno())

    def read_completed(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

def inspect_zone_system_sink_py_class_schema(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the class schema boundary used by this module."""
    value = context.get("class_schema")
    if value is None:
        return Diagnostic("class_schema", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("class_schema", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("class_schema", False, "numeric value cannot be negative")
    return Diagnostic("class_schema", True, f"accepted {value!r}")

def inspect_zone_system_sink_py_image_size(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the image size boundary used by this module."""
    value = context.get("image_size")
    if value is None:
        return Diagnostic("image_size", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("image_size", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("image_size", False, "numeric value cannot be negative")
    return Diagnostic("image_size", True, f"accepted {value!r}")

def inspect_zone_system_sink_py_confidence(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the confidence boundary used by this module."""
    value = context.get("confidence")
    if value is None:
        return Diagnostic("confidence", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("confidence", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("confidence", False, "numeric value cannot be negative")
    return Diagnostic("confidence", True, f"accepted {value!r}")

def inspect_zone_system_sink_py_max_detections(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the max detections boundary used by this module."""
    value = context.get("max_detections")
    if value is None:
        return Diagnostic("max_detections", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("max_detections", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("max_detections", False, "numeric value cannot be negative")
    return Diagnostic("max_detections", True, f"accepted {value!r}")

def inspect_zone_system_sink_py_source_identity(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the source identity boundary used by this module."""
    value = context.get("source_identity")
    if value is None:
        return Diagnostic("source_identity", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("source_identity", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("source_identity", False, "numeric value cannot be negative")
    return Diagnostic("source_identity", True, f"accepted {value!r}")
