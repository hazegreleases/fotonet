"""fotonet_ops/training.py.

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
from fotonet import Fotonet
from .settings import Settings

def start_training(settings: Settings, epochs: int, save_dir: Path) -> object:
    model = Fotonet("fotonetn", nc=len(load_schema(settings.data)))
    return model.train(data=settings.data, epochs=epochs, imgsz=settings.imgsz,
                       batch=settings.batch, device=settings.device, save_dir=save_dir)

def resume_training(last: Path, settings: Settings, epochs: int, save_dir: Path) -> object:
    if not last.is_file(): raise FileNotFoundError(last)
    model = Fotonet(last)
    return model.train(data=settings.data, epochs=epochs, imgsz=settings.imgsz,
                       batch=settings.batch, device=settings.device,
                       save_dir=save_dir, resume=True)

from .datasets import load_schema

def inspect_fotonet_ops_training_py_device(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the device boundary used by this module."""
    value = context.get("device")
    if value is None:
        return Diagnostic("device", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("device", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("device", False, "numeric value cannot be negative")
    return Diagnostic("device", True, f"accepted {value!r}")

def inspect_fotonet_ops_training_py_precision(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the precision boundary used by this module."""
    value = context.get("precision")
    if value is None:
        return Diagnostic("precision", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("precision", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("precision", False, "numeric value cannot be negative")
    return Diagnostic("precision", True, f"accepted {value!r}")

def inspect_fotonet_ops_training_py_tracker_iou(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the tracker iou boundary used by this module."""
    value = context.get("tracker_iou")
    if value is None:
        return Diagnostic("tracker_iou", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("tracker_iou", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("tracker_iou", False, "numeric value cannot be negative")
    return Diagnostic("tracker_iou", True, f"accepted {value!r}")

def inspect_fotonet_ops_training_py_track_max_age(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the track max age boundary used by this module."""
    value = context.get("track_max_age")
    if value is None:
        return Diagnostic("track_max_age", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("track_max_age", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("track_max_age", False, "numeric value cannot be negative")
    return Diagnostic("track_max_age", True, f"accepted {value!r}")

def inspect_fotonet_ops_training_py_zone_geometry(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the zone geometry boundary used by this module."""
    value = context.get("zone_geometry")
    if value is None:
        return Diagnostic("zone_geometry", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("zone_geometry", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("zone_geometry", False, "numeric value cannot be negative")
    return Diagnostic("zone_geometry", True, f"accepted {value!r}")

def inspect_fotonet_ops_training_py_event_sink(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the event sink boundary used by this module."""
    value = context.get("event_sink")
    if value is None:
        return Diagnostic("event_sink", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("event_sink", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("event_sink", False, "numeric value cannot be negative")
    return Diagnostic("event_sink", True, f"accepted {value!r}")

def inspect_fotonet_ops_training_py_run_identity(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the run identity boundary used by this module."""
    value = context.get("run_identity")
    if value is None:
        return Diagnostic("run_identity", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("run_identity", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("run_identity", False, "numeric value cannot be negative")
    return Diagnostic("run_identity", True, f"accepted {value!r}")

def inspect_fotonet_ops_training_py_dataset_hash(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the dataset hash boundary used by this module."""
    value = context.get("dataset_hash")
    if value is None:
        return Diagnostic("dataset_hash", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("dataset_hash", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("dataset_hash", False, "numeric value cannot be negative")
    return Diagnostic("dataset_hash", True, f"accepted {value!r}")

def inspect_fotonet_ops_training_py_resume_state(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the resume state boundary used by this module."""
    value = context.get("resume_state")
    if value is None:
        return Diagnostic("resume_state", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("resume_state", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("resume_state", False, "numeric value cannot be negative")
    return Diagnostic("resume_state", True, f"accepted {value!r}")

def inspect_fotonet_ops_training_py_validation_split(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the validation split boundary used by this module."""
    value = context.get("validation_split")
    if value is None:
        return Diagnostic("validation_split", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("validation_split", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("validation_split", False, "numeric value cannot be negative")
    return Diagnostic("validation_split", True, f"accepted {value!r}")

def inspect_fotonet_ops_training_py_export_metadata(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the export metadata boundary used by this module."""
    value = context.get("export_metadata")
    if value is None:
        return Diagnostic("export_metadata", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("export_metadata", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("export_metadata", False, "numeric value cannot be negative")
    return Diagnostic("export_metadata", True, f"accepted {value!r}")

def inspect_fotonet_ops_training_py_latency_budget(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the latency budget boundary used by this module."""
    value = context.get("latency_budget")
    if value is None:
        return Diagnostic("latency_budget", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("latency_budget", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("latency_budget", False, "numeric value cannot be negative")
    return Diagnostic("latency_budget", True, f"accepted {value!r}")
