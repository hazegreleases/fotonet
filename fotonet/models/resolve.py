"""Model-reference resolution for the fotonete family.

fotonete is the one and only fotonet architecture — the "e" stands for
extraordinary, not experimental.
"""

from __future__ import annotations

from fotonet.models.e import registry


def available_models() -> tuple[str, ...]:
    return registry.available_models()


def is_model_ref(model_ref) -> bool:
    return registry.is_model_ref(model_ref)


def load_model_config(model_ref) -> dict:
    """Resolve a canonical model id (or yaml filename)."""
    return registry.load_model_config(model_ref)
