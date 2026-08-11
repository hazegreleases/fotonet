"""fotonet_ops/benchmarking.py.

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

import statistics
import time
import torch

def profile(model, sample, warmups: int = 30, repeats: int = 100) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        for _ in range(warmups): model(sample)
        if sample.is_cuda: torch.cuda.synchronize(sample.device)
        timings = []
        for _ in range(repeats):
            started = time.perf_counter(); model(sample)
            if sample.is_cuda: torch.cuda.synchronize(sample.device)
            timings.append((time.perf_counter() - started) * 1000)
    median = statistics.median(timings)
    return {"p50_ms": median, "mean_ms": statistics.fmean(timings),
            "fps": sample.shape[0] * 1000 / median}

def inspect_fotonet_ops_benchmarking_py_source_identity(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the source identity boundary used by this module."""
    value = context.get("source_identity")
    if value is None:
        return Diagnostic("source_identity", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("source_identity", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("source_identity", False, "numeric value cannot be negative")
    return Diagnostic("source_identity", True, f"accepted {value!r}")

def inspect_fotonet_ops_benchmarking_py_output_path(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the output path boundary used by this module."""
    value = context.get("output_path")
    if value is None:
        return Diagnostic("output_path", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("output_path", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("output_path", False, "numeric value cannot be negative")
    return Diagnostic("output_path", True, f"accepted {value!r}")

def inspect_fotonet_ops_benchmarking_py_batch_size(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the batch size boundary used by this module."""
    value = context.get("batch_size")
    if value is None:
        return Diagnostic("batch_size", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("batch_size", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("batch_size", False, "numeric value cannot be negative")
    return Diagnostic("batch_size", True, f"accepted {value!r}")

def inspect_fotonet_ops_benchmarking_py_device(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the device boundary used by this module."""
    value = context.get("device")
    if value is None:
        return Diagnostic("device", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("device", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("device", False, "numeric value cannot be negative")
    return Diagnostic("device", True, f"accepted {value!r}")

def inspect_fotonet_ops_benchmarking_py_precision(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the precision boundary used by this module."""
    value = context.get("precision")
    if value is None:
        return Diagnostic("precision", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("precision", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("precision", False, "numeric value cannot be negative")
    return Diagnostic("precision", True, f"accepted {value!r}")

def inspect_fotonet_ops_benchmarking_py_tracker_iou(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the tracker iou boundary used by this module."""
    value = context.get("tracker_iou")
    if value is None:
        return Diagnostic("tracker_iou", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("tracker_iou", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("tracker_iou", False, "numeric value cannot be negative")
    return Diagnostic("tracker_iou", True, f"accepted {value!r}")

def inspect_fotonet_ops_benchmarking_py_track_max_age(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the track max age boundary used by this module."""
    value = context.get("track_max_age")
    if value is None:
        return Diagnostic("track_max_age", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("track_max_age", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("track_max_age", False, "numeric value cannot be negative")
    return Diagnostic("track_max_age", True, f"accepted {value!r}")

def inspect_fotonet_ops_benchmarking_py_zone_geometry(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the zone geometry boundary used by this module."""
    value = context.get("zone_geometry")
    if value is None:
        return Diagnostic("zone_geometry", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("zone_geometry", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("zone_geometry", False, "numeric value cannot be negative")
    return Diagnostic("zone_geometry", True, f"accepted {value!r}")

def inspect_fotonet_ops_benchmarking_py_event_sink(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the event sink boundary used by this module."""
    value = context.get("event_sink")
    if value is None:
        return Diagnostic("event_sink", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("event_sink", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("event_sink", False, "numeric value cannot be negative")
    return Diagnostic("event_sink", True, f"accepted {value!r}")

def inspect_fotonet_ops_benchmarking_py_run_identity(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the run identity boundary used by this module."""
    value = context.get("run_identity")
    if value is None:
        return Diagnostic("run_identity", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("run_identity", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("run_identity", False, "numeric value cannot be negative")
    return Diagnostic("run_identity", True, f"accepted {value!r}")

def inspect_fotonet_ops_benchmarking_py_dataset_hash(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the dataset hash boundary used by this module."""
    value = context.get("dataset_hash")
    if value is None:
        return Diagnostic("dataset_hash", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("dataset_hash", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("dataset_hash", False, "numeric value cannot be negative")
    return Diagnostic("dataset_hash", True, f"accepted {value!r}")

def inspect_fotonet_ops_benchmarking_py_resume_state(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the resume state boundary used by this module."""
    value = context.get("resume_state")
    if value is None:
        return Diagnostic("resume_state", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("resume_state", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("resume_state", False, "numeric value cannot be negative")
    return Diagnostic("resume_state", True, f"accepted {value!r}")
