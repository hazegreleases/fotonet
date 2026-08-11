"""Explicit production architecture specifications.

The production graph deliberately does not derive its graph from width/depth
multipliers.  Each public model size owns a reviewed integer channel/depth
layout, while P2 and non-P2 variants resolve to the same foundation profile.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


ARCHITECTURE_SCHEMA = 1


@dataclass(frozen=True)
class ArchitectureSpec:
    """One immutable production architecture profile."""

    profile: str
    # stem, P2, P3, P4, P5
    backbone_channels: tuple[int, int, int, int, int]
    # P2, P3, P4, P5 SpatialCSP repeat counts
    backbone_depths: tuple[int, int, int, int]
    # stem, P2, P3, P4, P5 transitions. The stem is always dense.
    downsample_modes: tuple[str, str, str, str, str]
    # P2, P3, P4, P5 output widths after neck projection/fusion
    neck_channels: tuple[int, int, int, int]
    architecture_schema: int = ARCHITECTURE_SCHEMA

    def __post_init__(self) -> None:
        if not self.profile or self.profile != self.profile.lower():
            raise ValueError("foundation profile must be a non-empty lowercase name")
        if len(self.backbone_channels) != 5 or len(self.backbone_depths) != 4:
            raise ValueError("foundation backbone requires five widths and four depths")
        if len(self.downsample_modes) != 5:
            raise ValueError("foundation backbone requires five downsample modes")
        if self.downsample_modes[0] != "dense":
            raise ValueError("the production stem must use dense downsampling")
        if any(mode not in {"dense", "efficient"} for mode in self.downsample_modes):
            raise ValueError("downsample modes must be 'dense' or 'efficient'")
        if len(self.neck_channels) != 4:
            raise ValueError("foundation neck requires P2-P5 widths")
        if any(value <= 0 for value in (*self.backbone_channels, *self.neck_channels)):
            raise ValueError("foundation channel widths must be positive")
        if any(value <= 0 for value in self.backbone_depths):
            raise ValueError("foundation stage depths must be positive")
        if tuple(sorted(self.backbone_channels)) != self.backbone_channels:
            raise ValueError("foundation backbone widths must be non-decreasing")

    @property
    def fingerprint(self) -> str:
        """Stable SHA256 identity for checkpoint/resume compatibility."""
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    @property
    def backbone_out_channels(self) -> tuple[int, int, int, int]:
        return self.backbone_channels[1:]


ARCHITECTURE_SPECS: dict[str, ArchitectureSpec] = {
    "n": ArchitectureSpec(
        profile="n",
        backbone_channels=(16, 32, 64, 112, 192),
        backbone_depths=(1, 2, 2, 1),
        downsample_modes=("dense", "dense", "dense", "dense", "dense"),
        neck_channels=(24, 48, 96, 160),
    ),
    "s": ArchitectureSpec(
        profile="s",
        backbone_channels=(24, 40, 80, 144, 256),
        backbone_depths=(1, 2, 3, 1),
        downsample_modes=("dense", "dense", "dense", "dense", "efficient"),
        neck_channels=(32, 72, 128, 224),
    ),
    "m": ArchitectureSpec(
        profile="m",
        backbone_channels=(24, 48, 112, 208, 368),
        backbone_depths=(1, 3, 4, 2),
        downsample_modes=("dense", "dense", "dense", "efficient", "efficient"),
        neck_channels=(48, 96, 176, 320),
    ),
    "l": ArchitectureSpec(
        profile="l",
        backbone_channels=(32, 64, 144, 272, 480),
        backbone_depths=(2, 3, 5, 2),
        downsample_modes=("dense", "dense", "dense", "efficient", "efficient"),
        neck_channels=(64, 128, 240, 416),
    ),
    "x": ArchitectureSpec(
        profile="x",
        backbone_channels=(40, 80, 176, 336, 608),
        backbone_depths=(2, 4, 6, 3),
        downsample_modes=("dense", "dense", "dense", "efficient", "efficient"),
        neck_channels=(80, 160, 304, 512),
    ),
}

def normalize_profile(profile: str) -> str:
    """Normalize a scale spelling to its P2-independent foundation profile."""
    normalized = str(profile).strip().lower()
    if normalized.startswith("fotonet"):
        normalized = normalized[len("fotonet"):].lstrip("-_")
    if normalized.endswith("-p2"):
        normalized = normalized[:-3]
    if normalized not in ARCHITECTURE_SPECS:
        valid = ", ".join(ARCHITECTURE_SPECS)
        raise ValueError(f"Unknown architecture profile {profile!r}; expected one of: {valid}.")
    return normalized


def get_architecture_spec(profile: str) -> ArchitectureSpec:
    return ARCHITECTURE_SPECS[normalize_profile(profile)]


def architecture_fingerprint(*, profile: str, p2: bool, reg_max: int, quality_head: bool, nc: int) -> str:
    """Hash the complete resolved graph contract, including its output width."""
    spec = get_architecture_spec(profile)
    payload = {
        "architecture_schema": ARCHITECTURE_SCHEMA,
        "profile": spec.profile,
        "spec": asdict(spec),
        "p2": bool(p2),
        "reg_max": int(reg_max),
        "quality_head": bool(quality_head),
        "nc": int(nc),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
