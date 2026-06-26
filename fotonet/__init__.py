"""FOTO-NET: fast Apache-friendly object detection with an Ultralytics-like API."""

from fotonet.engine.model import FOTONET, __version__
from fotonet.engine.results import AnchorPoint, BoxTransform, FocusRegion, Vector2

__all__ = ["FOTONET", "Vector2", "AnchorPoint", "FocusRegion", "BoxTransform", "__version__"]
