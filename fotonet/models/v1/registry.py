"""Canonical production model registry."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from .spec import ARCHITECTURE_SCHEMA, architecture_fingerprint, get_architecture_spec


MODEL_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "models"
MODEL_IDS = tuple(
    f"fotonet{profile}{suffix}"
    for profile in ("n", "s", "m", "l", "x")
    for suffix in ("", "-p2")
)


def available_models() -> tuple[str, ...]:
    return MODEL_IDS


def normalize_model_id(model_ref) -> str | None:
    if model_ref is None:
        return None
    raw = os.fspath(model_ref).strip().lower()
    stem = Path(raw).name
    for suffix in (".yaml", ".yml"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if not stem.startswith("fotonet"):
        return None
    return stem if stem in MODEL_IDS else None


def is_model_ref(model_ref) -> bool:
    return normalize_model_id(model_ref) is not None


def load_model_config(model_ref) -> dict:
    model_id = normalize_model_id(model_ref)
    if model_id is None:
        raise ValueError(
            f"Unknown model {model_ref!r}. Use one of: {', '.join(MODEL_IDS)}."
        )
    path = MODEL_CONFIG_DIR / f"{model_id}.yaml"
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return normalize_model_config(data, model_id=model_id, source=str(path))


def normalize_model_config(data: dict, *, model_id: str | None = None, source=None) -> dict:
    if not isinstance(data, dict):
        raise TypeError("Model configuration must be a mapping.")
    forbidden = set(data) - {
        "nc", "profile", "p2", "reg_max", "quality_head", "architecture_schema"
    }
    if forbidden:
        raise ValueError(
            "Unsupported model configuration fields: " + ", ".join(sorted(forbidden))
        )
    if model_id is not None:
        normalized_id = normalize_model_id(model_id)
        if normalized_id is None:
            raise ValueError(f"Invalid canonical model id {model_id!r}.")
        profile_from_id = normalized_id[len("fotonet") :].removesuffix("-p2")
        p2_from_id = normalized_id.endswith("-p2")
    else:
        profile_from_id = None
        p2_from_id = False
        normalized_id = None
    profile = str(data.get("profile", profile_from_id or "")).lower()
    spec = get_architecture_spec(profile)
    p2 = bool(data.get("p2", p2_from_id))
    if normalized_id is None:
        normalized_id = f"fotonet{profile}{'-p2' if p2 else ''}"
        if normalized_id not in MODEL_IDS:
            raise ValueError(f"Unsupported canonical model id {normalized_id!r}.")
    if profile != profile_from_id or p2 != p2_from_id:
        raise ValueError(f"Configuration does not match canonical model id {normalized_id!r}.")
    schema = int(data.get("architecture_schema", ARCHITECTURE_SCHEMA))
    if schema != ARCHITECTURE_SCHEMA:
        raise ValueError(
            f"Unsupported architecture_schema={schema}; expected {ARCHITECTURE_SCHEMA}."
        )
    return {
        "model_id": normalized_id,
        "source": source,
        "architecture_schema": schema,
        "architecture_fingerprint": architecture_fingerprint(
            profile=spec.profile,
            p2=p2,
            reg_max=int(data.get("reg_max", 12)),
            quality_head=bool(data.get("quality_head", False)),
            nc=int(data.get("nc", 80)),
        ),
        "profile": spec.profile,
        "nc": int(data.get("nc", 80)),
        "p2": p2,
        "reg_max": int(data.get("reg_max", 12)),
        "quality_head": bool(data.get("quality_head", False)),
        "backbone_out_channels": list(spec.backbone_out_channels),
        "neck_out_channels": list(spec.neck_channels if p2 else spec.neck_channels[1:]),
        "feature_strides": [4, 8, 16, 32] if p2 else [8, 16, 32],
    }
