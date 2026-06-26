"""Named FOTO-NET model scale registry.

Scale definitions live in ``fotonet/config/models/*.yaml`` so the public API,
CLI, profiler, and docs all resolve model sizes from one place.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml


MODEL_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "models"
PUBLIC_SCALES = ("n", "n-p2", "s", "s-p2", "m", "m-p2", "l", "l-p2", "x", "x-p2", "e")
LEGACY_ALIASES = {"n0": "n"}


def _scale_path(name: str) -> Path:
    return MODEL_CONFIG_DIR / f"fotonet{name}.yaml"


def available_model_scales() -> tuple[str, ...]:
    return tuple(name for name in PUBLIC_SCALES if _scale_path(name).exists())


def _normalize_scale_name(model_ref) -> str | None:
    if model_ref is None:
        return None
    raw = os.fspath(model_ref).strip().lower()
    stem = Path(raw).stem
    for prefix in ("fotonet-", "fotonet"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    stem = LEGACY_ALIASES.get(stem, stem)
    return stem if stem in available_model_scales() else None


def is_model_scale_ref(model_ref) -> bool:
    return _normalize_scale_name(model_ref) is not None


def load_scale_config(model_ref) -> dict:
    name = _normalize_scale_name(model_ref)
    if name is None:
        valid = ", ".join(f"fotonet{k}" for k in available_model_scales())
        raise ValueError(f"Unknown FOTO-NET model scale '{model_ref}'. Use one of: {valid}.")

    path = _scale_path(name)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return normalize_model_config(data, name=name, source=str(path))


def normalize_model_config(data: dict, name: str | None = None, source: str | None = None) -> dict:
    cfg = {
        "name": name,
        "source": source,
        "nc": int(data.get("nc", 80)),
        "width_multiple": float(data.get("width_multiple", 0.2)),
        "depth_multiple": float(data.get("depth_multiple", 0.33)),
        "p2_head": bool(data.get("p2_head", data.get("use_p2", True))),
        "reg_max": int(data.get("reg_max", 1)),
        "p3_extra_blocks": int(data.get("p3_extra_blocks", 0)),
        "p4_extra_blocks": int(data.get("p4_extra_blocks", 0)),
        "p5_extra_blocks": int(data.get("p5_extra_blocks", 0)),
        "p5_gate_blocks": int(data.get("p5_gate_blocks", 0)),
        "arch_version": int(data.get("arch_version", 1)),
        "neck_fusion": str(data.get("neck_fusion", data.get("fusion", "concat"))),
        "p2_context_blocks": int(data.get("p2_context_blocks", 0)),
        "p3_context_blocks": int(data.get("p3_context_blocks", 0)),
        "quality_head": bool(data.get("quality_head", False)),
    }
    return cfg


def scale_fallback_configs() -> dict[str, dict]:
    """Checkpoint compatibility probes for older public scale names."""
    configs = {}
    for name in available_model_scales():
        cfg = load_scale_config(name)
        configs[name] = cfg
    for alias, target in LEGACY_ALIASES.items():
        if target in configs:
            legacy = dict(configs[target])
            legacy["name"] = alias
            configs[alias] = legacy
    configs["n-legacy"] = normalize_model_config(
        {"width_multiple": 0.25, "depth_multiple": 0.33, "p2_head": True, "reg_max": 1},
        name="n-legacy",
    )
    configs["s-legacy"] = normalize_model_config(
        {"width_multiple": 0.50, "depth_multiple": 0.33, "p2_head": True, "reg_max": 1},
        name="s-legacy",
    )
    return configs
