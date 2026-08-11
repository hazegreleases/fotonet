"""Thin composition root for the immutable V1 training protocol."""

from .checkpoint import CheckpointProtocolMixin
from .data import DataPipelineMixin
from ...engine.callbacks import DiagnosticsMixin
from .initialization import InitializationMixin
from .loop import TrainingLoopMixin
from .optimization import OptimizationProtocolMixin
from .recipe import RecipeProtocolMixin
from .schedules import (
    SchedulesProtocolMixin,
    _AverageMovementLRDropDown,
    _MedianLRDropDown,
)
from .validation import ValidationProtocolMixin


class Trainer(
    TrainingLoopMixin,
    InitializationMixin,
    DataPipelineMixin,
    ValidationProtocolMixin,
    DiagnosticsMixin,
    CheckpointProtocolMixin,
    OptimizationProtocolMixin,
    SchedulesProtocolMixin,
    RecipeProtocolMixin,
):
    """Public trainer for the exact V1 numerical and resume protocol."""

    _AMP_FINITE_CHECK_INTERVAL = 512
    _LIVE_STATUS_INTERVAL_SEC = 30.0
    _RUNNING_METRIC_NAMES = (
        "loss", "cls", "box", "regpc", "dflpc", "iou", "cons",
        "wcls", "wreg", "wdfl", "wiou", "wcons", "quality", "wquality",
        "awcls", "awbox", "awdfl", "awcons", "awquality", "qmix", "hneg",
        "pos_o2o", "pos_o2m", "o2m_active", "exact_o2o",
    )
    _LOG_VALIDATION_METRICS = (
        "precision", "recall", "mAP50", "mAP50_95", "mAP75",
        "mAP_small", "mAP_medium", "mAP_large", "coco_AR100",
        "dense_scene_min_objects", "dense_scene_images",
        "dense_scene_precision_at_conf", "dense_scene_recall_at_conf",
    )


__all__ = [
    "Trainer",
    "_AverageMovementLRDropDown",
    "_MedianLRDropDown",
]
