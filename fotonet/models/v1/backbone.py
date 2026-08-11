"""Production fotonet backbone."""
from __future__ import annotations

import torch
import torch.nn as nn

from .spec import ArchitectureSpec, get_architecture_spec


class Conv(nn.Module):
    """Convolution, batch normalization, and SiLU primitive."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        padding = k // 2 if p is None else p
        self.conv = nn.Conv2d(
            c1, c2, k, s, padding, groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            nn.SiLU(inplace=True)
            if act is True
            else (act if isinstance(act, nn.Module) else nn.Identity())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SPPF(nn.Module):
    """Sequential fast spatial-pyramid pooling."""

    def __init__(self, c1: int, c2: int, kernel_size: int = 5):
        super().__init__()
        hidden = c1 // 2
        self.reduce = Conv(c1, hidden, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
        self.project = Conv(hidden * 4, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        first = self.pool(x)
        second = self.pool(first)
        return self.project(torch.cat((x, first, second, self.pool(second)), dim=1))


class SpatialResidual(nn.Module):
    """Bottlenecked residual with a real 3x3 spatial refinement path."""

    def __init__(self, channels: int, expansion: float = 0.5):
        super().__init__()
        hidden = max(8, int(round(channels * float(expansion))))
        self.reduce = Conv(channels, hidden, 1, 1)
        self.spatial = Conv(hidden, channels, 3, 1, act=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.spatial(self.reduce(x)))


class SpatialCSP(nn.Module):
    """Two-path CSP stage whose transformed path always mixes spatially."""

    def __init__(self, c1: int, c2: int, depth: int = 1, expansion: float = 0.5):
        super().__init__()
        hidden = max(8, int(round(c2 * float(expansion))))
        self.split = Conv(c1, 2 * hidden, 1, 1)
        self.blocks = nn.Sequential(
            *(SpatialResidual(hidden, expansion=0.5) for _ in range(max(int(depth), 1)))
        )
        self.project = Conv(2 * hidden, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bypass, transformed = self.split(x).chunk(2, dim=1)
        return self.project(torch.cat((bypass, self.blocks(transformed)), dim=1))


class DenseDownsample(nn.Module):
    """Hardware-friendly dense stride-2 spatial projection."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.proj = Conv(c1, c2, 3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class EfficientDownsample(nn.Module):
    """Stride-first depthwise reduction followed by low-resolution projection."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.spatial = Conv(c1, c1, 3, 2, g=c1)
        self.project = Conv(c1, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.spatial(x))


def _downsample(mode: str, c1: int, c2: int) -> nn.Module:
    if mode == "dense":
        return DenseDownsample(c1, c2)
    if mode == "efficient":
        return EfficientDownsample(c1, c2)
    raise ValueError(f"Unsupported downsample mode {mode!r}.")


class Backbone(nn.Module):
    """Production backbone returning the same P2-P5 tuple for every scale."""

    def __init__(self, profile: str | ArchitectureSpec):
        super().__init__()
        self.spec = profile if isinstance(profile, ArchitectureSpec) else get_architecture_spec(profile)
        stem, c2, c3, c4, c5 = self.spec.backbone_channels
        d2, d3, d4, d5 = self.spec.backbone_depths
        modes = self.spec.downsample_modes

        self.stem = _downsample(modes[0], 3, stem)
        self.stage1 = nn.Sequential(
            _downsample(modes[1], stem, c2), SpatialCSP(c2, c2, d2)
        )
        self.stage2 = nn.Sequential(
            _downsample(modes[2], c2, c3), SpatialCSP(c3, c3, d3)
        )
        self.stage3 = nn.Sequential(
            _downsample(modes[3], c3, c4), SpatialCSP(c4, c4, d4)
        )
        self.stage4 = nn.Sequential(
            _downsample(modes[4], c4, c5), SpatialCSP(c5, c5, d5), SPPF(c5, c5)
        )
        self.out_channels = self.spec.backbone_out_channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.stem(x)
        p2 = self.stage1(x)
        p3 = self.stage2(p2)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p2, p3, p4, p5
