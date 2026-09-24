"""Explicit fotonete architecture specification (schema 3).

``fotonete`` is the trinity-designed nano detector — the 'e' stands for extraordinary.  It shares
the production training contract (dual O2O/O2M assignment, DFL regression,
``[B, 8400, nc + 4]`` output) but rebuilds the graph under one measured law:
few, wide, dense convolutions.  Every depthwise-free stage is a plain
residual 3x3 stack; neck junctions are one pointwise reduction followed by
one spatial 3x3; the head is a single dense 3x3 stem per level.

The layout below is a reviewed integer design, not a multiplier sweep, and
is frozen exactly like a production profile.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


# Schema 3: the all-dense residual-stack fotonete family.
# It is the one and only fotonet architecture; checkpoints from the removed
# mistaken for production graphs.
EXPERIMENT_SCHEMA = 3


@dataclass(frozen=True)
class ExperimentSpec:
    """One immutable fotonete architecture profile."""

    profile: str = "e"
    # stem output and P3/P4/P5 widths (channels)
    stem_channels: int = 24
    p2_channels: int = 32
    widths: tuple[int, int, int] = (64, 96, 160)
    # residual-conv repeats per stage (P3, P4, P5)
    depths: tuple[int, int, int] = (2, 2, 1)
    # neck output widths per level (P3, P4, P5)
    neck_channels: tuple[int, int, int] = (64, 96, 160)
    # shared head stem width for every level
    head_hidden: int = 64
    architecture_schema: int = EXPERIMENT_SCHEMA

    def __post_init__(self) -> None:
        if self.profile != "e":
            raise ValueError("the fotonete profile is exactly 'e'")
        if len(self.widths) != 3 or len(self.depths) != 3 or len(self.neck_channels) != 3:
            raise ValueError("the fotonete graph requires P3/P4/P5 layouts")
        if any(value <= 0 for value in (*self.widths, *self.depths, *self.neck_channels)):
            raise ValueError("fotonete widths and depths must be positive")
        if tuple(sorted(self.widths)) != self.widths:
            raise ValueError("fotonete stage widths must be non-decreasing")

    @property
    def fingerprint(self) -> str:
        """Stable SHA256 identity for checkpoint/resume compatibility."""
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


EXPERIMENT_SPECS: dict[str, ExperimentSpec] = {"e": ExperimentSpec()}


def get_experiment_spec(profile: str) -> ExperimentSpec:
    normalized = str(profile).strip().lower()
    if normalized.startswith("fotonet"):
        normalized = normalized[len("fotonet"):].lstrip("-_")
    if normalized not in EXPERIMENT_SPECS:
        raise ValueError(f"Unknown fotonete profile {profile!r}; expected 'e'.")
    return EXPERIMENT_SPECS[normalized]


def experiment_fingerprint(*, reg_max: int, nc: int) -> str:
    """Hash the complete resolved fotonete graph contract."""
    spec = get_experiment_spec("e")
    payload = {
        "architecture_schema": EXPERIMENT_SCHEMA,
        "spec": asdict(spec),
        "reg_max": int(reg_max),
        "nc": int(nc),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
