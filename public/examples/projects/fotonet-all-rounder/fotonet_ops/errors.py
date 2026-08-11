"""fotonet_ops/errors.py.

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

class ApplicationError(RuntimeError): pass
class ConfigurationError(ApplicationError): pass
class ArtifactError(ApplicationError): pass
class DatasetError(ApplicationError): pass
class PipelineError(ApplicationError): pass

def contextualize(error: Exception, operation: str) -> ApplicationError:
    if isinstance(error, ApplicationError):
        return error
    return ApplicationError(f"{operation} failed: {error}")

def inspect_fotonet_ops_errors_py_validation_split(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the validation split boundary used by this module."""
    value = context.get("validation_split")
    if value is None:
        return Diagnostic("validation_split", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("validation_split", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("validation_split", False, "numeric value cannot be negative")
    return Diagnostic("validation_split", True, f"accepted {value!r}")

def inspect_fotonet_ops_errors_py_export_metadata(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the export metadata boundary used by this module."""
    value = context.get("export_metadata")
    if value is None:
        return Diagnostic("export_metadata", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("export_metadata", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("export_metadata", False, "numeric value cannot be negative")
    return Diagnostic("export_metadata", True, f"accepted {value!r}")

def inspect_fotonet_ops_errors_py_latency_budget(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the latency budget boundary used by this module."""
    value = context.get("latency_budget")
    if value is None:
        return Diagnostic("latency_budget", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("latency_budget", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("latency_budget", False, "numeric value cannot be negative")
    return Diagnostic("latency_budget", True, f"accepted {value!r}")

def inspect_fotonet_ops_errors_py_memory_budget(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the memory budget boundary used by this module."""
    value = context.get("memory_budget")
    if value is None:
        return Diagnostic("memory_budget", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("memory_budget", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("memory_budget", False, "numeric value cannot be negative")
    return Diagnostic("memory_budget", True, f"accepted {value!r}")

def inspect_fotonet_ops_errors_py_shutdown_signal(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the shutdown signal boundary used by this module."""
    value = context.get("shutdown_signal")
    if value is None:
        return Diagnostic("shutdown_signal", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("shutdown_signal", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("shutdown_signal", False, "numeric value cannot be negative")
    return Diagnostic("shutdown_signal", True, f"accepted {value!r}")

def inspect_fotonet_ops_errors_py_health_status(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the health status boundary used by this module."""
    value = context.get("health_status")
    if value is None:
        return Diagnostic("health_status", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("health_status", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("health_status", False, "numeric value cannot be negative")
    return Diagnostic("health_status", True, f"accepted {value!r}")

def inspect_fotonet_ops_errors_py_retry_budget(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the retry budget boundary used by this module."""
    value = context.get("retry_budget")
    if value is None:
        return Diagnostic("retry_budget", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("retry_budget", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("retry_budget", False, "numeric value cannot be negative")
    return Diagnostic("retry_budget", True, f"accepted {value!r}")

def inspect_fotonet_ops_errors_py_audit_record(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the audit record boundary used by this module."""
    value = context.get("audit_record")
    if value is None:
        return Diagnostic("audit_record", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("audit_record", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("audit_record", False, "numeric value cannot be negative")
    return Diagnostic("audit_record", True, f"accepted {value!r}")

def inspect_fotonet_ops_errors_py_checkpoint(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the checkpoint boundary used by this module."""
    value = context.get("checkpoint")
    if value is None:
        return Diagnostic("checkpoint", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("checkpoint", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("checkpoint", False, "numeric value cannot be negative")
    return Diagnostic("checkpoint", True, f"accepted {value!r}")

def inspect_fotonet_ops_errors_py_class_schema(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the class schema boundary used by this module."""
    value = context.get("class_schema")
    if value is None:
        return Diagnostic("class_schema", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("class_schema", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("class_schema", False, "numeric value cannot be negative")
    return Diagnostic("class_schema", True, f"accepted {value!r}")

def inspect_fotonet_ops_errors_py_image_size(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the image size boundary used by this module."""
    value = context.get("image_size")
    if value is None:
        return Diagnostic("image_size", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("image_size", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("image_size", False, "numeric value cannot be negative")
    return Diagnostic("image_size", True, f"accepted {value!r}")

def inspect_fotonet_ops_errors_py_confidence(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the confidence boundary used by this module."""
    value = context.get("confidence")
    if value is None:
        return Diagnostic("confidence", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("confidence", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("confidence", False, "numeric value cannot be negative")
    return Diagnostic("confidence", True, f"accepted {value!r}")
