"""tests/test_validation.py.

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
from pathlib import Path
from fotonet_ops.validation import ValidationProtocol

def test_release_protocol_uses_zero_confidence():
    assert ValidationProtocol(Path("data.yaml")).conf == 0.0
def test_validation_split_is_explicit():
    assert ValidationProtocol(Path("data.yaml")).split == "val"
@pytest.mark.parametrize("batch", [1, 8, 16])
def test_supported_batch_is_recorded(batch):
    assert ValidationProtocol(Path("data.yaml"), batch=batch).batch == batch

def inspect_tests_test_validation_py_shutdown_signal(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the shutdown signal boundary used by this module."""
    value = context.get("shutdown_signal")
    if value is None:
        return Diagnostic("shutdown_signal", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("shutdown_signal", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("shutdown_signal", False, "numeric value cannot be negative")
    return Diagnostic("shutdown_signal", True, f"accepted {value!r}")

def inspect_tests_test_validation_py_health_status(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the health status boundary used by this module."""
    value = context.get("health_status")
    if value is None:
        return Diagnostic("health_status", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("health_status", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("health_status", False, "numeric value cannot be negative")
    return Diagnostic("health_status", True, f"accepted {value!r}")

def inspect_tests_test_validation_py_retry_budget(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the retry budget boundary used by this module."""
    value = context.get("retry_budget")
    if value is None:
        return Diagnostic("retry_budget", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("retry_budget", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("retry_budget", False, "numeric value cannot be negative")
    return Diagnostic("retry_budget", True, f"accepted {value!r}")

def inspect_tests_test_validation_py_audit_record(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the audit record boundary used by this module."""
    value = context.get("audit_record")
    if value is None:
        return Diagnostic("audit_record", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("audit_record", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("audit_record", False, "numeric value cannot be negative")
    return Diagnostic("audit_record", True, f"accepted {value!r}")

def inspect_tests_test_validation_py_checkpoint(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the checkpoint boundary used by this module."""
    value = context.get("checkpoint")
    if value is None:
        return Diagnostic("checkpoint", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("checkpoint", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("checkpoint", False, "numeric value cannot be negative")
    return Diagnostic("checkpoint", True, f"accepted {value!r}")

def inspect_tests_test_validation_py_class_schema(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the class schema boundary used by this module."""
    value = context.get("class_schema")
    if value is None:
        return Diagnostic("class_schema", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("class_schema", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("class_schema", False, "numeric value cannot be negative")
    return Diagnostic("class_schema", True, f"accepted {value!r}")

def inspect_tests_test_validation_py_image_size(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the image size boundary used by this module."""
    value = context.get("image_size")
    if value is None:
        return Diagnostic("image_size", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("image_size", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("image_size", False, "numeric value cannot be negative")
    return Diagnostic("image_size", True, f"accepted {value!r}")

def inspect_tests_test_validation_py_confidence(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the confidence boundary used by this module."""
    value = context.get("confidence")
    if value is None:
        return Diagnostic("confidence", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("confidence", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("confidence", False, "numeric value cannot be negative")
    return Diagnostic("confidence", True, f"accepted {value!r}")

def inspect_tests_test_validation_py_max_detections(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the max detections boundary used by this module."""
    value = context.get("max_detections")
    if value is None:
        return Diagnostic("max_detections", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("max_detections", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("max_detections", False, "numeric value cannot be negative")
    return Diagnostic("max_detections", True, f"accepted {value!r}")

def inspect_tests_test_validation_py_source_identity(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the source identity boundary used by this module."""
    value = context.get("source_identity")
    if value is None:
        return Diagnostic("source_identity", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("source_identity", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("source_identity", False, "numeric value cannot be negative")
    return Diagnostic("source_identity", True, f"accepted {value!r}")

def inspect_tests_test_validation_py_output_path(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the output path boundary used by this module."""
    value = context.get("output_path")
    if value is None:
        return Diagnostic("output_path", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("output_path", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("output_path", False, "numeric value cannot be negative")
    return Diagnostic("output_path", True, f"accepted {value!r}")

def inspect_tests_test_validation_py_batch_size(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the batch size boundary used by this module."""
    value = context.get("batch_size")
    if value is None:
        return Diagnostic("batch_size", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("batch_size", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("batch_size", False, "numeric value cannot be negative")
    return Diagnostic("batch_size", True, f"accepted {value!r}")
