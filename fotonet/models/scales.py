"""Strict launcher-facing access to the production registry."""

from fotonet.models.v1.registry import (
    MODEL_IDS,
    available_models,
    is_model_ref,
    load_model_config,
    normalize_model_config,
)


def available_model_scales():
    return tuple(model_id.removeprefix("fotonet") for model_id in MODEL_IDS)


def is_model_scale_ref(value):
    return is_model_ref(value)


def load_scale_config(value):
    config = load_model_config(value)
    # The current launcher and in-progress checkpoint use this explicit
    # temporary identity. It does not enable alternate graph construction.
    return {
        **config,
        "architecture": "standard",
        "width_multiple": {
            "n": 0.25, "s": 0.50, "m": 0.75, "l": 1.0, "x": 1.25,
        }[config["profile"]],
        "depth_multiple": 1.0,
        "arch_version": 4,
        "head_version": 4,
        "foundation_version": 2,
        "foundation_profile": config["profile"],
        "foundation_fingerprint": _candidate_fingerprint(config["profile"]),
        "p2_head": config["p2"],
        "neck_fusion": "concat",
        "neck_iema": False,
        "p2_context_blocks": 0,
        "p3_context_blocks": 0,
        "p3_extra_blocks": 0,
        "p4_extra_blocks": 0,
        "p5_extra_blocks": 0,
        "p5_gate_blocks": 0,
        "p5_psa_blocks": 1,
        "fold": {
            "enabled": False,
            "factor": 2,
            "method": "pixel_unshuffle",
            "width_multiple": 1.0,
            "detail_path": True,
        },
    }


def _candidate_fingerprint(profile):
    from fotonet.models.foundation_specs import get_foundation_spec

    return get_foundation_spec(profile).fingerprint
