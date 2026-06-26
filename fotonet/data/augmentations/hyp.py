"""Augmentation hyperparameter defaults and sanitization."""
import math


DEFAULT_AUGMENT_HYP = {
    "mosaic": 0.909,
    "mixup": 0.012,
    "copy_paste": 0.075,
    "copy_paste_mode": "flip",
    "close_mosaic": 10,
    "degrees": 1.11,
    "translate": 0.071,
    "scale": 0.562,
    "shear": 1.46,
    "flipud": 0.0,
    "fliplr": 0.606,
    "hsv_h": 0.014,
    "hsv_s": 0.645,
    "hsv_v": 0.566,
    "bgr": 0.106,
    "mixup_alpha": 32.0,
}


def finite_float(value, default=0.0):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def clamp(value, low, high, default=0.0):
    return min(max(finite_float(value, default), low), high)


def build_augment_hyp(overrides=None):
    hyp = DEFAULT_AUGMENT_HYP.copy()
    if overrides:
        hyp.update(overrides)
    for key in ("mosaic", "mixup", "copy_paste", "flipud", "fliplr", "bgr"):
        hyp[key] = clamp(hyp.get(key, DEFAULT_AUGMENT_HYP[key]), 0.0, 1.0, DEFAULT_AUGMENT_HYP[key])
    hyp["close_mosaic"] = max(int(finite_float(hyp.get("close_mosaic", 0), 0)), 0)
    hyp["degrees"] = clamp(hyp.get("degrees", 0.0), 0.0, 180.0, DEFAULT_AUGMENT_HYP["degrees"])
    hyp["translate"] = clamp(hyp.get("translate", 0.0), 0.0, 0.50, DEFAULT_AUGMENT_HYP["translate"])
    hyp["scale"] = clamp(hyp.get("scale", 0.0), 0.0, 0.95, DEFAULT_AUGMENT_HYP["scale"])
    hyp["shear"] = clamp(hyp.get("shear", 0.0), 0.0, 45.0, DEFAULT_AUGMENT_HYP["shear"])
    hyp["hsv_h"] = clamp(hyp.get("hsv_h", 0.0), 0.0, 0.50, DEFAULT_AUGMENT_HYP["hsv_h"])
    hyp["hsv_s"] = clamp(hyp.get("hsv_s", 0.0), 0.0, 1.0, DEFAULT_AUGMENT_HYP["hsv_s"])
    hyp["hsv_v"] = clamp(hyp.get("hsv_v", 0.0), 0.0, 1.0, DEFAULT_AUGMENT_HYP["hsv_v"])
    hyp["mixup_alpha"] = max(
        finite_float(hyp.get("mixup_alpha", DEFAULT_AUGMENT_HYP["mixup_alpha"]), DEFAULT_AUGMENT_HYP["mixup_alpha"]),
        1e-3,
    )
    return hyp
