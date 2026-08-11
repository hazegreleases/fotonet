"""Stable import facade for the versioned training protocol."""

from fotonet.training.protocols.v1.protocol import (
    Trainer,
    _AverageMovementLRDropDown,
    _MedianLRDropDown,
)

__all__ = [
    "Trainer",
    "_AverageMovementLRDropDown",
    "_MedianLRDropDown",
]
