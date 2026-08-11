"""The sole production detector graph."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Backbone
from .head import Head
from .neck import Neck
from .spec import ARCHITECTURE_SCHEMA, architecture_fingerprint, get_architecture_spec


def _init_weights(module):
    if isinstance(module, nn.Conv2d):
        if getattr(module, "_fotonet_identity_gate_output", False):
            nn.init.zeros_(module.weight)
        else:
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="leaky_relu", a=0.01
            )
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0, 0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _fuse_conv_bn(conv, bn):
    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )
    weight = conv.weight.reshape(conv.out_channels, -1)
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    fused.weight.data.copy_((weight * scale.reshape(-1, 1)).reshape_as(conv.weight))
    bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    fused.bias.data.copy_(bn.bias + (bias - bn.running_mean) * scale)
    return fused


def _fuse_named_pair(module, conv_name, bn_name):
    conv = getattr(module, conv_name, None)
    bn = getattr(module, bn_name, None)
    if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
        setattr(module, conv_name, _fuse_conv_bn(conv, bn))
        setattr(module, bn_name, nn.Identity())


def _fuse_sequential_pairs(module):
    if not isinstance(module, nn.Sequential):
        return
    names = list(module._modules)
    for left, right in zip(names, names[1:]):
        conv, bn = module._modules[left], module._modules[right]
        if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
            module._modules[left] = _fuse_conv_bn(conv, bn)
            module._modules[right] = nn.Identity()


class Detector(nn.Module):
    """Backbone, neck, and head assembled from one explicit profile."""

    def __init__(
        self,
        *,
        nc=80,
        profile="n",
        use_p2=False,
        reg_max=12,
        quality_head=False,
        imgsz=640,
    ):
        super().__init__()
        self.spec = get_architecture_spec(profile)
        self.architecture_schema = ARCHITECTURE_SCHEMA
        self.profile = self.spec.profile
        self.architecture_fingerprint = architecture_fingerprint(
            profile=self.spec.profile,
            p2=use_p2,
            reg_max=reg_max,
            quality_head=quality_head,
            nc=nc,
        )
        self.use_p2 = bool(use_p2)
        self.reg_max = int(reg_max)
        self.quality_head = bool(quality_head)
        # Temporary identity consumed only by the active run's strict launcher
        # and checkpoint migration. It does not select another graph.
        from fotonet.models.foundation_specs import get_foundation_spec

        active_identity = get_foundation_spec(self.profile)
        self.architecture = "standard"
        self.arch_version = 4
        self.head_version = 4
        self.foundation_version = active_identity.foundation_version
        self.foundation_profile = active_identity.profile
        self.foundation_fingerprint = active_identity.fingerprint
        self.backbone = Backbone(self.spec)
        self.neck = Neck(self.backbone.out_channels, self.spec, use_p2=self.use_p2)
        self.backbone_out_channels = tuple(self.backbone.out_channels)
        self.neck_out_channels = tuple(self.neck.out_channels)
        self.feature_strides = (4, 8, 16, 32) if self.use_p2 else (8, 16, 32)
        self.model_config = {
            "model_id": f"fotonet{self.profile}{'-p2' if self.use_p2 else ''}",
            "architecture_schema": self.architecture_schema,
            "architecture_fingerprint": self.architecture_fingerprint,
            "profile": self.profile,
            "nc": int(nc),
            "p2": self.use_p2,
            "reg_max": self.reg_max,
            "quality_head": self.quality_head,
            "backbone_out_channels": list(self.backbone_out_channels),
            "neck_out_channels": list(self.neck_out_channels),
            "feature_strides": list(self.feature_strides),
        }
        self.head = Head(
            nc=nc,
            in_channels=self.neck.out_channels,
            imgsz=imgsz,
            strides=self.feature_strides,
            reg_max=self.reg_max,
            quality_head=self.quality_head,
        )
        self.apply(_init_weights)
        self.head._init_cls_bias()
        self._fused = False

    def forward(self, x, return_all=False, use_o2m=None):
        original_shape = torch._shape_as_tensor(x)
        original_imgsz = original_shape[[3, 2]].to(device=x.device, dtype=x.dtype)
        if self.training:
            actual_imgsz = original_imgsz
            box_scale = None
        else:
            stride = max(self.head.strides)
            if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
                pad_h = torch.remainder(-original_shape[2], stride)
                pad_w = torch.remainder(-original_shape[3], stride)
                x = F.pad(x, (0, pad_w, 0, pad_h), value=114.0 / 255.0)
            else:
                height, width = x.shape[-2:]
                pad_h, pad_w = (-height) % stride, (-width) % stride
                if pad_h or pad_w:
                    x = F.pad(x, (0, pad_w, 0, pad_h), value=114.0 / 255.0)
            padded_shape = torch._shape_as_tensor(x)
            actual_imgsz = padded_shape[[3, 2]].to(device=x.device, dtype=x.dtype)
            ratio = actual_imgsz / original_imgsz
            box_scale = torch.stack((ratio[0], ratio[1], ratio[0], ratio[1]))
        outputs = self.head(
            self.neck(self.backbone(x)),
            imgsz=actual_imgsz,
            return_all=return_all,
            use_o2m=use_o2m,
        )
        if box_scale is None:
            return outputs
        if isinstance(outputs, dict):
            outputs = dict(outputs)
            for key in ("pred_boxes_o2o", "pred_boxes_o2m"):
                if key in outputs:
                    outputs[key] = outputs[key] * box_scale
            outputs["original_imgsz"] = original_imgsz
            return outputs
        return torch.cat(
            (outputs[..., : self.head.nc], outputs[..., self.head.nc :] * box_scale), dim=-1
        )

    def strip_o2m_for_inference(self):
        self.head.strip_o2m_for_inference()
        return self

    def fuse(self):
        if self.training:
            raise RuntimeError("Call model.eval() before fusing Conv-BN modules.")
        if self._fused:
            return self
        for module in self.modules():
            _fuse_named_pair(module, "conv", "bn")
            _fuse_named_pair(module, "cv2", "bn")
            _fuse_named_pair(module, "dw", "bn")
            _fuse_named_pair(module, "pw", "pw_bn")
            _fuse_sequential_pairs(module)
        remaining = sum(isinstance(module, nn.BatchNorm2d) for module in self.modules())
        if remaining:
            raise RuntimeError(f"Deploy fusion left {remaining} BatchNorm2d modules unfused.")
        self._fused = True
        return self
