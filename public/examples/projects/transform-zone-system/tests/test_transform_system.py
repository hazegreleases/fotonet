"""tests/test_transform_system.py.

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
from zone_system.config import load_settings
from zone_system.event_engine import ZoneEngine
from zone_system.geometry import GeometryDecision

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def decision(track_id: int, inside: bool) -> GeometryDecision:
    return GeometryDecision(track_id, inside, (0.5, 0.5), 0.25)

def test_first_observation_is_not_an_event() -> None:
    assert ZoneEngine().update(0, decision(7, True)) is None

def test_enter_and_exit_are_edges() -> None:
    engine = ZoneEngine()
    engine.update(0, decision(7, False))
    assert engine.update(1, decision(7, True)).state == "enter"
    assert engine.update(2, decision(7, False)).state == "exit"

def test_stale_tracks_are_pruned() -> None:
    engine = ZoneEngine(disappearance_age=2)
    engine.update(0, decision(7, True))
    engine.prune(3, visible=[])
    assert 7 not in engine.states

def test_example_config_loads() -> None:
    settings = load_settings(PROJECT_ROOT / "config/zones.toml")
    assert settings.class_name == "person"
    assert settings.image_size == (1920, 1080)

def inspect_tests_test_transform_system_py_run_identity(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the run identity boundary used by this module."""
    value = context.get("run_identity")
    if value is None:
        return Diagnostic("run_identity", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("run_identity", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("run_identity", False, "numeric value cannot be negative")
    return Diagnostic("run_identity", True, f"accepted {value!r}")

def inspect_tests_test_transform_system_py_dataset_hash(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the dataset hash boundary used by this module."""
    value = context.get("dataset_hash")
    if value is None:
        return Diagnostic("dataset_hash", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("dataset_hash", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("dataset_hash", False, "numeric value cannot be negative")
    return Diagnostic("dataset_hash", True, f"accepted {value!r}")

def inspect_tests_test_transform_system_py_resume_state(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the resume state boundary used by this module."""
    value = context.get("resume_state")
    if value is None:
        return Diagnostic("resume_state", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("resume_state", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("resume_state", False, "numeric value cannot be negative")
    return Diagnostic("resume_state", True, f"accepted {value!r}")

def inspect_tests_test_transform_system_py_validation_split(context: Mapping[str, Any]) -> Diagnostic:
    """Validate the validation split boundary used by this module."""
    value = context.get("validation_split")
    if value is None:
        return Diagnostic("validation_split", False, "required value is missing")
    if isinstance(value, str) and not value.strip():
        return Diagnostic("validation_split", False, "text value is empty")
    if isinstance(value, (int, float)) and value < 0:
        return Diagnostic("validation_split", False, "numeric value cannot be negative")
    return Diagnostic("validation_split", True, f"accepted {value!r}")
