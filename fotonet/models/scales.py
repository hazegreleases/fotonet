"""Strict launcher-facing access to the fotonete registry.

The training protocol's checkpoint metadata goes through here; the shape of
this mapping is part of the launcher contract and stays frozen.
"""

from fotonet.models.e import registry as e_registry


def available_model_scales():
    return ("e",)


def is_model_scale_ref(value):
    return e_registry.is_model_ref(value)


def load_scale_config(value):
    config = e_registry.load_model_config(value)
    return {
        **config,
        "architecture": "standard",
        "width_multiple": 1.0,
        "depth_multiple": 1.0,
        "arch_version": 6,
        "head_version": 6,
        "foundation_version": 3,
        "foundation_profile": config["profile"],
        "foundation_fingerprint": config["architecture_fingerprint"],
        "p2_head": False,
        "neck_fusion": "concat",
        "neck_iema": False,
        "p2_context_blocks": 0,
        "p3_context_blocks": 0,
        "p3_extra_blocks": 0,
        "p4_extra_blocks": 0,
        "p5_extra_blocks": 0,
        "p5_gate_blocks": 0,
        "p5_psa_blocks": 0,
        "fold": {
            "enabled": False,
            "factor": 2,
            "method": "pixel_unshuffle",
            "width_multiple": 1.0,
            "detail_path": True,
        },
    }
