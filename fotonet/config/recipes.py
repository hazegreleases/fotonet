"""Resolved, public training recipe loader."""
from __future__ import annotations

from pathlib import Path

import yaml


RECIPE_DIR = Path(__file__).resolve().parent / "recipes"


def available_training_recipes():
    return tuple(path.stem for path in sorted(RECIPE_DIR.glob("*.yaml")))


def load_training_recipe(recipe):
    """Load a recipe by packaged name or YAML path and flatten it for ``train``."""
    candidate = Path(recipe)
    path = candidate if candidate.is_file() else RECIPE_DIR / f"{candidate.stem}.yaml"
    if not path.is_file():
        valid = ", ".join(available_training_recipes()) or "(none packaged)"
        raise FileNotFoundError(f"Training recipe '{recipe}' was not found. Available recipes: {valid}.")
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Training recipe '{path}' must contain a YAML mapping.")
    train = dict(raw.get("train") or {})
    optimizer = dict(raw.get("optimizer") or {})
    settings = {
        **train,
        "optimizer": optimizer.pop("name", train.get("optimizer", "sgd")),
        **optimizer,
        "pretrained": bool(raw.get("pretrained", False)),
        "resume": bool(raw.get("resume", False)),
        "augment_hyp": dict(raw.get("augment") or {}),
        "loss_hyp": dict(raw.get("loss") or {}),
        "matcher_hyp": dict(raw.get("matcher") or {}),
    }
    return {
        "name": path.stem,
        "path": str(path),
        "model": raw.get("model"),
        "settings": settings,
        "smoke_limits": dict(raw.get("smoke_limits") or {}),
    }
