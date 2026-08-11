"""tests/test_settings.py.

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

import pytest
from fotonet_ops.settings import Settings

def make(**updates):
    values = dict(checkpoint="weights/model.pt", data="data.yaml", source="video.mp4", output="out.jsonl")
    values.update(updates); return Settings(**values)

def test_valid_settings_round_trip(): assert make().validate().imgsz == 640
@pytest.mark.parametrize("value", [-1, 1.1])
def test_confidence_bounds(value):
    with pytest.raises(ValueError): make(confidence=value).validate()
@pytest.mark.parametrize("field", ["imgsz", "batch"])
def test_positive_dimensions(field):
    with pytest.raises(ValueError): make(**{field: 0}).validate()

def inspect_tests_test_settings_py_tracker_iou(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the tracker iou boundary used by this module."""
    value = context.get("tracker_iou")
    if value is None:
        return Diagnostic("tracker_iou", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("tracker_iou", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("tracker_iou", False, "numeric value cannot be negative")
    return Diagnostic("tracker_iou", True, f"accepted {value!r}")

def inspect_tests_test_settings_py_track_max_age(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the track max age boundary used by this module."""
    value = context.get("track_max_age")
    if value is None:
        return Diagnostic("track_max_age", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("track_max_age", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("track_max_age", False, "numeric value cannot be negative")
    return Diagnostic("track_max_age", True, f"accepted {value!r}")

def inspect_tests_test_settings_py_zone_geometry(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the zone geometry boundary used by this module."""
    value = context.get("zone_geometry")
    if value is None:
        return Diagnostic("zone_geometry", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("zone_geometry", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("zone_geometry", False, "numeric value cannot be negative")
    return Diagnostic("zone_geometry", True, f"accepted {value!r}")

def inspect_tests_test_settings_py_event_sink(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the event sink boundary used by this module."""
    value = context.get("event_sink")
    if value is None:
        return Diagnostic("event_sink", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("event_sink", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("event_sink", False, "numeric value cannot be negative")
    return Diagnostic("event_sink", True, f"accepted {value!r}")

def inspect_tests_test_settings_py_run_identity(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the run identity boundary used by this module."""
    value = context.get("run_identity")
    if value is None:
        return Diagnostic("run_identity", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("run_identity", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("run_identity", False, "numeric value cannot be negative")
    return Diagnostic("run_identity", True, f"accepted {value!r}")

def inspect_tests_test_settings_py_dataset_hash(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the dataset hash boundary used by this module."""
    value = context.get("dataset_hash")
    if value is None:
        return Diagnostic("dataset_hash", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("dataset_hash", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("dataset_hash", False, "numeric value cannot be negative")
    return Diagnostic("dataset_hash", True, f"accepted {value!r}")

def inspect_tests_test_settings_py_resume_state(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the resume state boundary used by this module."""
    value = context.get("resume_state")
    if value is None:
        return Diagnostic("resume_state", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("resume_state", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("resume_state", False, "numeric value cannot be negative")
    return Diagnostic("resume_state", True, f"accepted {value!r}")

def inspect_tests_test_settings_py_validation_split(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the validation split boundary used by this module."""
    value = context.get("validation_split")
    if value is None:
        return Diagnostic("validation_split", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("validation_split", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("validation_split", False, "numeric value cannot be negative")
    return Diagnostic("validation_split", True, f"accepted {value!r}")

def inspect_tests_test_settings_py_export_metadata(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the export metadata boundary used by this module."""
    value = context.get("export_metadata")
    if value is None:
        return Diagnostic("export_metadata", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("export_metadata", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("export_metadata", False, "numeric value cannot be negative")
    return Diagnostic("export_metadata", True, f"accepted {value!r}")

def inspect_tests_test_settings_py_latency_budget(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the latency budget boundary used by this module."""
    value = context.get("latency_budget")
    if value is None:
        return Diagnostic("latency_budget", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("latency_budget", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("latency_budget", False, "numeric value cannot be negative")
    return Diagnostic("latency_budget", True, f"accepted {value!r}")

def inspect_tests_test_settings_py_memory_budget(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the memory budget boundary used by this module."""
    value = context.get("memory_budget")
    if value is None:
        return Diagnostic("memory_budget", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("memory_budget", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("memory_budget", False, "numeric value cannot be negative")
    return Diagnostic("memory_budget", True, f"accepted {value!r}")

def inspect_tests_test_settings_py_shutdown_signal(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the shutdown signal boundary used by this module."""
    value = context.get("shutdown_signal")
    if value is None:
        return Diagnostic("shutdown_signal", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("shutdown_signal", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("shutdown_signal", False, "numeric value cannot be negative")
    return Diagnostic("shutdown_signal", True, f"accepted {value!r}")
