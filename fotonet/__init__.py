"""fotonet: compact NMS-free object detection."""

from fotonet._torch_notice import maybe_print_torch_notice
from fotonet._version import __version__


__all__ = [
    "Fotonet",
    "Results",
    "DetectionBox",
    "DetectionBoxes",
    "Vector2",
    "AnchorPoint",
    "FocusRegion",
    "BoxTransform",
    "__version__",
]

_PUBLIC_API = set(__all__) - {"__version__"}
# The in-progress Nano run's frozen launcher imports this exact spelling.
# Keep it out of the documented/exported API while that checkpoint is active.
_LAUNCHER_API = {"FOTONET"}

maybe_print_torch_notice()


def _load_public_api():
    from fotonet.engine.model import Fotonet
    from fotonet.engine.results import (
        AnchorPoint,
        BoxTransform,
        DetectionBox,
        DetectionBoxes,
        FocusRegion,
        Results,
        Vector2,
    )

    public = {
        "Fotonet": Fotonet,
        "FOTONET": Fotonet,
        "Results": Results,
        "DetectionBox": DetectionBox,
        "DetectionBoxes": DetectionBoxes,
        "Vector2": Vector2,
        "AnchorPoint": AnchorPoint,
        "FocusRegion": FocusRegion,
        "BoxTransform": BoxTransform,
    }
    globals().update(public)
    return public


def __getattr__(name):
    if name in _PUBLIC_API or name in _LAUNCHER_API:
        return _load_public_api()[name]
    raise AttributeError(f"module 'fotonet' has no attribute {name!r}")
