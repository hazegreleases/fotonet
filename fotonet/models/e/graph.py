"""The canonical extraordinary fotonete detector graph (schema 3).

Design law (measured on RTX 4060, eager FP16): few, wide, dense ops.
The verified envelope at 640x640 is ~1.70M deploy parameters, ~5.43 GFLOPs
(2xMACs), B1 3.31 ms, B8 5.70 ms - 31-36% faster than the production nano
while carrying ~56% more compute.  Zero depthwise convolutions anywhere.

The training/inference contract deliberately mirrors the production graph
so the existing loss, matcher, validation, and export paths run unmodified.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spec import EXPERIMENT_SCHEMA, ExperimentSpec, experiment_fingerprint, get_experiment_spec


def _init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(
            module.weight, mode="fan_out", nonlinearity="leaky_relu", a=0.01
        )
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
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


class Conv(nn.Module):
    """Dense convolution, batch normalization, and SiLU primitive."""

    def __init__(self, c1, c2, k=1, s=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.bn(self.conv(x)))


class ResidualConv(nn.Module):
    """Plain dense 3x3 residual step: one wide op, one free skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = Conv(channels, channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class SPPF(nn.Module):
    """Sequential fast spatial-pyramid pooling."""

    def __init__(self, c1: int, c2: int, kernel_size: int = 5):
        super().__init__()
        hidden = c1 // 2
        self.reduce = Conv(c1, hidden, 1)
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
        self.project = Conv(hidden * 4, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        first = self.pool(x)
        second = self.pool(first)
        return self.project(torch.cat((x, first, second, self.pool(second)), dim=1))


class Backbone(nn.Module):
    """All-dense residual-stack backbone returning P3/P4/P5."""

    def __init__(self, spec: ExperimentSpec):
        super().__init__()
        self.spec = spec
        c3, c4, c5 = spec.widths
        d3, d4, d5 = spec.depths
        self.stem = Conv(3, spec.stem_channels, 3, 2)
        self.p2 = Conv(spec.stem_channels, spec.p2_channels, 3, 2)
        self.down3 = Conv(spec.p2_channels, c3, 3, 2)
        self.stage3 = nn.Sequential(*(ResidualConv(c3) for _ in range(d3)))
        self.down4 = Conv(c3, c4, 3, 2)
        self.stage4 = nn.Sequential(*(ResidualConv(c4) for _ in range(d4)))
        self.down5 = Conv(c4, c5, 3, 2)
        self.stage5 = nn.Sequential(*(ResidualConv(c5) for _ in range(d5)), SPPF(c5, c5))
        self.out_channels = (c3, c4, c5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        p3 = self.stage3(self.down3(self.p2(self.stem(x))))
        p4 = self.stage4(self.down4(p3))
        p5 = self.stage5(self.down5(p4))
        return p3, p4, p5


class FuseJunction(nn.Module):
    """Concat join refined by one pointwise reduction and one spatial 3x3."""

    def __init__(self, c_cat: int, c_out: int):
        super().__init__()
        self.reduce = Conv(c_cat, c_out, 1)
        self.spatial = Conv(c_out, c_out, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.reduce(x))


class Neck(nn.Module):
    """Bidirectional pyramid with dense fuse junctions at every join."""

    def __init__(self, in_channels: tuple[int, int, int], spec: ExperimentSpec):
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError("The fotonete neck requires P3-P5 backbone channels.")
        self.spec = spec
        c3, c4, c5 = (int(value) for value in in_channels)
        n3, n4, n5 = spec.neck_channels
        self.fuse4_td = FuseJunction(n5 + c4, n4)
        self.fuse3_td = FuseJunction(n4 + n3, n3)
        self.bu4 = Conv(n3, n4, 3, 2)
        self.fuse4_bu = FuseJunction(2 * n4, n4)
        self.bu5 = Conv(n4, n5, 3, 2)
        self.fuse5_bu = FuseJunction(2 * n5, n5)
        self.out_channels = (n3, n4, n5)

    @staticmethod
    def _resize_like(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
            return F.interpolate(x, size=reference.shape[-2:], mode="nearest")
        if x.shape[-2:] == reference.shape[-2:]:
            return x
        return F.interpolate(x, size=reference.shape[-2:], mode="nearest")

    def forward(self, feats: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        if len(feats) != 3:
            raise ValueError("The fotonete neck requires three P3-P5 tensors.")
        p3, p4, p5 = feats
        t4 = self.fuse4_td(torch.cat((self._resize_like(p5, p4), p4), dim=1))
        t3 = self.fuse3_td(torch.cat((self._resize_like(t4, p3), p3), dim=1))
        n4 = self.fuse4_bu(torch.cat((self.bu4(t3), t4), dim=1))
        n5 = self.fuse5_bu(torch.cat((self.bu5(n4), p5), dim=1))
        return t3, n4, n5


class Head(nn.Module):
    """Dual-assignment head with the production output contract.

    One dense 3x3 stem per level feeds independent O2O and training-only
    O2M pointwise adapters, mirroring the production head's dict outputs so
    the shared loss and matcher train this graph unmodified.
    """

    default_strides = [8, 16, 32]

    def __init__(self, nc=80, in_channels=(64, 96, 160), reg_max=12, imgsz=640, strides=None):
        nn.Module.__init__(self)
        self.nc = int(nc)
        self.reg_max = int(reg_max)
        self.imgsz = imgsz
        if strides is None:
            strides = self.default_strides
        self.strides = [int(s) for s in strides]
        if len(self.strides) != len(in_channels):
            raise ValueError("Number of head strides must match number of feature maps.")
        reg_out = 4 * self.reg_max if self.reg_max > 1 else 4
        hidden = get_experiment_spec("e").head_hidden
        self.shared_channels = [hidden] * len(in_channels)
        self.shared_stem = nn.ModuleList(
            (Conv(in_ch, hidden, 3) for in_ch in in_channels)
        )

        def adapters(out_channels):
            return nn.ModuleList(
                (nn.Conv2d(hidden, out_channels, 1) for _ in in_channels)
            )

        self.cls_o2o = adapters(self.nc)
        self.reg_o2o = adapters(reg_out)
        self.cls_o2m = adapters(self.nc)
        self.reg_o2m = adapters(reg_out)
        self.register_buffer("proj", torch.arange(self.reg_max, dtype=torch.float32))
        self._init_cls_bias()
        self._grid_cache = {}
        self.inference_only = False

    @property
    def has_o2m(self):
        return len(self.cls_o2m) > 0 and len(self.reg_o2m) > 0

    def strip_o2m_for_inference(self):
        """Drop training-only O2M heads from the deployment module."""
        self.cls_o2m = nn.ModuleList()
        self.reg_o2m = nn.ModuleList()
        self.inference_only = True
        return self

    def _init_cls_bias(self):
        """Set cls head biases so initial foreground confidence ~= 1%."""
        bias_init = -4.595
        for m in self.cls_o2o:
            nn.init.constant_(m.bias, bias_init)
        for m in self.cls_o2m:
            nn.init.constant_(m.bias, bias_init)

    def _dfl(self, x):
        """DFL decode: [B, 4*reg_max, N] -> [B, N, 4] expected LTRB values."""
        (b, _, n) = x.shape
        if self.reg_max <= 1:
            return F.softplus(x.view(b, 4, n)).permute(0, 2, 1)
        x = x.view(b, 4, self.reg_max, n)
        x = x.softmax(2)
        x = (x * self.proj.view(1, 1, -1, 1)).sum(2)
        return x.permute(0, 2, 1)

    def _get_grid(self, nx, ny, device, dtype):
        """Get anchor centers at cell midpoints [1, nx*ny, 2]. Cached."""
        if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
            (yv, xv) = torch.meshgrid(
                torch.arange(ny, device=device, dtype=torch.float32) + 0.5,
                torch.arange(nx, device=device, dtype=torch.float32) + 0.5,
                indexing="ij",
            )
            return torch.stack((xv, yv), 2).view(1, -1, 2).to(dtype=dtype)
        key = (ny, nx, str(device), str(dtype))
        if key not in self._grid_cache:
            (yv, xv) = torch.meshgrid(
                torch.arange(ny, device=device, dtype=torch.float32) + 0.5,
                torch.arange(nx, device=device, dtype=torch.float32) + 0.5,
                indexing="ij",
            )
            if len(self._grid_cache) >= 16:
                self._grid_cache.pop(next(iter(self._grid_cache)))
            self._grid_cache[key] = torch.stack((xv, yv), 2).view(1, -1, 2).to(dtype=dtype)
        return self._grid_cache[key]

    def _dist2bbox(self, ltrb, anchor, stride, imgsz):
        """Convert LTRB distances (grid-cell units) to normalized xywh boxes."""
        (lt, rb) = ltrb.chunk(2, -1)
        x1y1 = (anchor - lt) * stride
        x2y2 = (anchor + rb) * stride
        c_xy = (x1y1 + x2y2) * 0.5
        wh = (x2y2 - x1y1).clamp(min=0)
        if torch.is_tensor(imgsz):
            size = imgsz.to(device=ltrb.device, dtype=ltrb.dtype).reshape(-1)
        else:
            size = ltrb.new_tensor([float(imgsz)])
        if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
            size = size.expand(2)
        elif size.numel() == 1:
            size = size.expand(2)
        elif size.numel() != 2:
            raise ValueError("imgsz must be a scalar or [width, height].")
        scale = torch.stack((size[0], size[1], size[0], size[1]))
        return torch.cat([c_xy, wh], -1) / scale

    def forward(self, feats, imgsz=None, return_all=False, use_o2m=None):
        tracing_export = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()
        if tracing_export:
            return_all = False
            use_o2m = None
        else:
            if not isinstance(return_all, bool):
                raise TypeError("return_all must be a bool.")
            if use_o2m is not None and (not isinstance(use_o2m, bool)):
                raise TypeError("use_o2m must be a bool or None.")
        actual_imgsz = imgsz if imgsz is not None else float(self.imgsz)
        requested_o2m = self.training if use_o2m is None else use_o2m
        run_o2o = self.training or return_all or (not requested_o2m)
        need_meta = self.training or return_all
        need_o2m = requested_o2m or (return_all and use_o2m is None)
        if need_o2m and (not self.has_o2m):
            raise RuntimeError("This checkpoint is inference-only: O2M heads were stripped.")
        out_o2o = []
        out_o2m = []
        dist_o2o_parts = []
        dist_o2m_parts = []
        anchor_parts = []
        stride_parts = []
        feat_shapes = []
        for (i, feat) in enumerate(feats):
            (b, _, h, w) = feat.shape
            stride = self.strides[i]
            grid = self._get_grid(w, h, feat.device, feat.dtype)
            feat_shapes.append((h, w))
            shared = self.shared_stem[i](feat)
            reg_ch = 4 * self.reg_max if self.reg_max > 1 else 4
            if run_o2o:
                cls_o2o = self.cls_o2o[i](shared).view(b, self.nc, -1).permute(0, 2, 1)
                reg_raw_o2o = self.reg_o2o[i](shared).view(b, reg_ch, -1)
                ltrb_o2o = self._dfl(reg_raw_o2o)
                boxes_o2o = self._dist2bbox(ltrb_o2o, grid, stride, actual_imgsz)
                out_o2o.append(torch.cat([cls_o2o, boxes_o2o], dim=-1))
            if need_meta:
                if run_o2o:
                    dist_o2o_parts.append(reg_raw_o2o.permute(0, 2, 1))
                n_anchors = h * w
                anchor_parts.append(grid.squeeze(0))
                stride_parts.append(
                    torch.full((n_anchors, 1), stride, device=feat.device, dtype=feat.dtype)
                )
            if need_o2m:
                cls_o2m = self.cls_o2m[i](shared).view(b, self.nc, -1).permute(0, 2, 1)
                reg_raw_o2m = self.reg_o2m[i](shared).view(b, reg_ch, -1)
                ltrb_o2m = self._dfl(reg_raw_o2m)
                boxes_o2m = self._dist2bbox(ltrb_o2m, grid, stride, actual_imgsz)
                out_o2m.append(torch.cat([cls_o2m, boxes_o2m], dim=-1))
                dist_o2m_parts.append(reg_raw_o2m.permute(0, 2, 1))
        preds_o2o = torch.cat(out_o2o, dim=1) if out_o2o else None
        if need_meta:
            if preds_o2o is None:
                raise RuntimeError("return_all requires the O2O branch.")
            result = {
                "pred_logits_o2o": preds_o2o[..., : self.nc],
                "pred_boxes_o2o": preds_o2o[..., self.nc :],
                "pred_dist_o2o": torch.cat(dist_o2o_parts, dim=1),
                "anchor_points": torch.cat(anchor_parts, dim=0),
                "stride_tensor": torch.cat(stride_parts, dim=0),
                "feat_shapes": torch.tensor(
                    feat_shapes, device=preds_o2o.device, dtype=torch.int32
                ),
                "imgsz": actual_imgsz,
            }
            if need_o2m:
                preds_o2m = torch.cat(out_o2m, dim=1)
                result.update(
                    {
                        "pred_logits_o2m": preds_o2m[..., : self.nc],
                        "pred_boxes_o2m": preds_o2m[..., self.nc :],
                        "pred_dist_o2m": torch.cat(dist_o2m_parts, dim=1),
                    }
                )
            return result
        return preds_o2o


class Detector(nn.Module):
    """Canonical backbone, neck, and head assembled from one frozen spec."""

    def __init__(self, *, nc=80, reg_max=12, imgsz=640):
        super().__init__()
        self.spec = get_experiment_spec("e")
        self.architecture_schema = EXPERIMENT_SCHEMA
        self.profile = self.spec.profile
        self.reg_max = int(reg_max)
        self.quality_head = False
        self.use_p2 = False
        self.architecture_fingerprint = experiment_fingerprint(
            reg_max=self.reg_max, nc=nc
        )
        self.backbone = Backbone(self.spec)
        self.neck = Neck(self.backbone.out_channels, self.spec)
        self.backbone_out_channels = tuple(self.backbone.out_channels)
        self.neck_out_channels = tuple(self.neck.out_channels)
        self.feature_strides = (8, 16, 32)
        self.model_config = {
            "model_id": "fotonete",
            "architecture_schema": self.architecture_schema,
            "architecture_fingerprint": self.architecture_fingerprint,
            "profile": self.profile,
            "nc": int(nc),
            "p2": False,
            "reg_max": self.reg_max,
            "quality_head": False,
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
            conv = getattr(module, "conv", None)
            bn = getattr(module, "bn", None)
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                module.conv = _fuse_conv_bn(conv, bn)
                module.bn = nn.Identity()
        remaining = sum(isinstance(module, nn.BatchNorm2d) for module in self.modules())
        if remaining:
            raise RuntimeError(f"Deploy fusion left {remaining} BatchNorm2d modules unfused.")
        self._fused = True
        return self
