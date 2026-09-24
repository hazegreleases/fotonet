"""Canonical extraordinary fotonete detector family (schema 3)."""

from .graph import Detector
from .spec import EXPERIMENT_SCHEMA, ExperimentSpec, experiment_fingerprint, get_experiment_spec

__all__ = [
    "Detector",
    "EXPERIMENT_SCHEMA",
    "ExperimentSpec",
    "experiment_fingerprint",
    "get_experiment_spec",
]
