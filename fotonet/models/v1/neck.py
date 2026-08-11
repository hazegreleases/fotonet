"""Production fotonet feature-pyramid neck."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Conv, SpatialCSP
from .spec import ArchitectureSpec, get_architecture_spec


class TopDownFusion(nn.Module):
    """Identity-preserving concat followed by one spatial CSP refinement."""

    def __init__(self, c1: int, c2: int, out_channels: int):
        super().__init__()
        self.refine = SpatialCSP(c1 + c2, out_channels, depth=1)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return self.refine(torch.cat((first, second), dim=1))


class BottomUpFusion(nn.Module):
    """Shallower pointwise/depthwise refinement for the PAN return path."""

    def __init__(self, c1: int, c2: int, out_channels: int):
        super().__init__()
        self.project = Conv(c1 + c2, out_channels, 1, 1)
        self.spatial = Conv(out_channels, out_channels, 3, 1, g=out_channels)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.project(torch.cat((first, second), dim=1)))


class Neck(nn.Module):
    """Concat FPN/PAN with independent neck widths and an optional narrow P2 path."""

    def __init__(
        self,
        in_channels: tuple[int, int, int, int],
        profile: str | ArchitectureSpec,
        use_p2: bool = False,
    ):
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError("Neck requires P2-P5 backbone channels.")
        self.spec = profile if isinstance(profile, ArchitectureSpec) else get_architecture_spec(profile)
        self.use_p2 = bool(use_p2)
        c2, c3, c4, c5 = (int(value) for value in in_channels)
        n2, n3, n4, n5 = self.spec.neck_channels

        if self.use_p2:
            self.lateral2 = Conv(c2, n2, 1, 1)
        self.lateral3 = Conv(c3, n3, 1, 1)
        self.lateral4 = Conv(c4, n4, 1, 1)
        self.lateral5 = Conv(c5, n5, 1, 1)

        self.reduce5 = Conv(n5, n4, 1, 1)
        self.fuse4_td = TopDownFusion(n4, n4, n4)
        self.reduce4 = Conv(n4, n3, 1, 1)
        self.fuse3_td = TopDownFusion(n3, n3, n3)

        if self.use_p2:
            self.reduce3 = Conv(n3, n2, 1, 1)
            self.fuse2_td = TopDownFusion(n2, n2, n2)
            self.down2 = Conv(n2, n3, 3, 2)
            self.fuse3_bu = BottomUpFusion(n3, n3, n3)

        self.down3 = Conv(n3, n4, 3, 2)
        self.fuse4_bu = BottomUpFusion(n4, n4, n4)
        self.down4 = Conv(n4, n5, 3, 2)
        self.fuse5_bu = BottomUpFusion(n5, n5, n5)

        self.out_channels = self.spec.neck_channels if self.use_p2 else self.spec.neck_channels[1:]

    @staticmethod
    def _resize_like(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Align joins exactly for odd/rectangular eager and exported inputs."""
        if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
            return F.interpolate(x, size=reference.shape[-2:], mode="nearest")
        if x.shape[-2:] == reference.shape[-2:]:
            return x
        return F.interpolate(x, size=reference.shape[-2:], mode="nearest")

    def forward(self, feats: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        if len(feats) != 4:
            raise ValueError("Neck requires four P2-P5 feature tensors.")
        p2, p3, p4, p5 = feats
        p3_lat = self.lateral3(p3)
        p4_lat = self.lateral4(p4)
        p5_lat = self.lateral5(p5)

        p5_up = self._resize_like(self.reduce5(p5_lat), p4_lat)
        p4_td = self.fuse4_td(p5_up, p4_lat)
        p4_up = self._resize_like(self.reduce4(p4_td), p3_lat)
        p3_td = self.fuse3_td(p4_up, p3_lat)

        if self.use_p2:
            p2_lat = self.lateral2(p2)
            p3_up = self._resize_like(self.reduce3(p3_td), p2_lat)
            p2_out = self.fuse2_td(p3_up, p2_lat)
            p2_down = self._resize_like(self.down2(p2_out), p3_td)
            n3 = self.fuse3_bu(p2_down, p3_td)
        else:
            n3 = p3_td

        n3_down = self._resize_like(self.down3(n3), p4_td)
        n4 = self.fuse4_bu(n3_down, p4_td)
        n4_down = self._resize_like(self.down4(n4), p5_lat)
        n5 = self.fuse5_bu(n4_down, p5_lat)
        return (p2_out, n3, n4, n5) if self.use_p2 else (n3, n4, n5)
