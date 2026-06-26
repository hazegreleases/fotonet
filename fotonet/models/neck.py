"""
FOTONET SOTA Neck - Hybrid PAN-FPN with YOLOv10/v11 features.
Features: C2fCIB (YOLOv10) for efficiency, standard PAN-FPN (YOLOv8/v11).
"""
import torch
import torch.nn as nn
from fotonet.models.backbone import Conv, C2fCIB


class WeightedFusion(nn.Module):
    """Learnable normalized weighted sum for same-channel feature fusion."""
    def __init__(self, n_inputs=2):
        super().__init__()
        self.raw = nn.Parameter(torch.ones(int(n_inputs), dtype=torch.float32))

    def forward(self, *features):
        weights = torch.relu(self.raw)
        weights = weights / (weights.sum() + 1e-4)
        out = features[0] * weights[0]
        for idx in range(1, len(features)):
            out = out + features[idx] * weights[idx]
        return out


class ContextBlock(nn.Module):
    """Cheap local context for high-resolution P2/P3 features."""
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 5, 1, 2, groups=c, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(c)

    def forward(self, x):
        y = self.pw_bn(self.pw(self.act(self.bn(self.dw(x)))))
        return self.act(x + y)


def _context_stack(c, n):
    return nn.Sequential(*(ContextBlock(c) for _ in range(max(int(n), 0)))) if int(n) > 0 else nn.Identity()


class iEMA(nn.Module):
    """Improved Efficient Multi-scale Attention (WI-YOLO / YOLO26 inspired)."""
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Export-friendly coordinate-style attention. Mean keeps it cheap and ONNX-safe.
        attn = x.mean(dim=3, keepdim=True) * x.mean(dim=2, keepdim=True)
        return x * self.sigmoid(self.conv(attn))

class Neck(nn.Module):
    """
    FOTONET SOTA Neck: Hybrid PAN-FPN with YOLOv10/v11 features.
    Features: C2fCIB (YOLOv10) for efficiency, standard PAN-FPN (YOLOv8/v11).
    """
    def __init__(
        self,
        in_channels=(64, 128, 256),
        d=1.0,
        use_p2=False,
        fusion="concat",
        p2_context_blocks=0,
        p3_context_blocks=0,
    ):
        super().__init__()
        self.use_p2 = bool(use_p2 or len(in_channels) == 4)
        self.fusion = str(fusion)
        self.weighted = self.fusion == "weighted"
        n = max(round(3 * d), 1)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        if self.use_p2:
            c2, c3, c4, c5 = in_channels

            self.conv_p5_p4 = Conv(c5, c4, 1, 1)
            self.c2f_p4_td  = C2fCIB(c4 if self.weighted else c4 + c4, c4, n, shortcut=False)

            self.conv_p4_p3 = Conv(c4, c3, 1, 1)
            self.c2f_p3_td  = C2fCIB(c3 if self.weighted else c3 + c3, c3, n, shortcut=False)

            self.conv_p3_p2 = Conv(c3, c2, 1, 1)
            self.c2f_p2_td  = C2fCIB(c2 if self.weighted else c2 + c2, c2, n, shortcut=False)

            self.down_p2_p3 = Conv(c2, c3 if self.weighted else c2, 3, 2)
            self.c2f_n3_bu  = C2fCIB(c3 if self.weighted else c2 + c3, c3, n, shortcut=False)

            self.down_p3_p4 = Conv(c3, c4 if self.weighted else c3, 3, 2)
            self.c2f_n4_bu  = C2fCIB(c4 if self.weighted else c3 + c4, c4, n, shortcut=False)

            self.down_p4_p5 = Conv(c4, c5 if self.weighted else c4, 3, 2)
            self.c2f_n5_bu  = C2fCIB(c5 if self.weighted else c4 + c5, c5, n, shortcut=False)

            self.fuse_p4_td = WeightedFusion(2)
            self.fuse_p3_td = WeightedFusion(2)
            self.fuse_p2_td = WeightedFusion(2)
            self.fuse_n3_bu = WeightedFusion(2)
            self.fuse_n4_bu = WeightedFusion(2)
            self.fuse_n5_bu = WeightedFusion(2)
            self.context_p2 = _context_stack(c2, p2_context_blocks)
            self.context_p3 = _context_stack(c3, p3_context_blocks)

            self.iema2 = iEMA(c2)
            self.iema3 = iEMA(c3)
            self.iema4 = iEMA(c4)
            self.iema5 = iEMA(c5)

            self.out_channels = (c2, c3, c4, c5)
        else:
            c3, c4, c5 = in_channels

            self.conv_p5_p4 = Conv(c5, c4, 1, 1)
            self.c2f_p4_td  = C2fCIB(c4 if self.weighted else c4 + c4, c4, n, shortcut=False)

            self.conv_p4_p3 = Conv(c4, c3, 1, 1)
            self.c2f_p3_td  = C2fCIB(c3 if self.weighted else c3 + c3, c3, n, shortcut=False)

            self.down_p3_p4 = Conv(c3, c4 if self.weighted else c3, 3, 2)
            self.c2f_n4_bu  = C2fCIB(c4 if self.weighted else c3 + c4, c4, n, shortcut=False)

            self.down_p4_p5 = Conv(c4, c5 if self.weighted else c4, 3, 2)
            self.c2f_n5_bu  = C2fCIB(c5 if self.weighted else c4 + c5, c5, n, shortcut=False)

            self.fuse_p4_td = WeightedFusion(2)
            self.fuse_p3_td = WeightedFusion(2)
            self.fuse_n4_bu = WeightedFusion(2)
            self.fuse_n5_bu = WeightedFusion(2)
            self.context_p3 = _context_stack(c3, p3_context_blocks)
            
            self.iema3 = iEMA(c3)
            self.iema4 = iEMA(c4)
            self.iema5 = iEMA(c5)

            self.out_channels = (c3, c4, c5)

    def _merge(self, fusion_module, *features):
        if self.weighted:
            return fusion_module(*features)
        return torch.cat(features, 1)

    def forward(self, feats):
        if self.use_p2:
            p2, p3, p4, p5 = feats

            p5_up  = self.up(self.conv_p5_p4(p5))
            p4_fuse = self.c2f_p4_td(self._merge(self.fuse_p4_td, p5_up, p4))

            p4_up  = self.up(self.conv_p4_p3(p4_fuse))
            p3_fuse = self.c2f_p3_td(self._merge(self.fuse_p3_td, p4_up, p3))

            p3_up  = self.up(self.conv_p3_p2(p3_fuse))
            p2_out  = self.c2f_p2_td(self._merge(self.fuse_p2_td, p3_up, p2))
            p2_out  = self.context_p2(p2_out)

            p2_down = self.down_p2_p3(p2_out)
            n3      = self.c2f_n3_bu(self._merge(self.fuse_n3_bu, p2_down, p3_fuse))
            n3      = self.context_p3(n3)

            n3_down = self.down_p3_p4(n3)
            n4      = self.c2f_n4_bu(self._merge(self.fuse_n4_bu, n3_down, p4_fuse))

            n4_down = self.down_p4_p5(n4)
            n5      = self.c2f_n5_bu(self._merge(self.fuse_n5_bu, n4_down, p5))

            return self.iema2(p2_out), self.iema3(n3), self.iema4(n4), self.iema5(n5)

        p3, p4, p5 = feats

        p5_up  = self.up(self.conv_p5_p4(p5))
        p4_fuse = self.c2f_p4_td(self._merge(self.fuse_p4_td, p5_up, p4))

        p4_up  = self.up(self.conv_p4_p3(p4_fuse))
        n3     = self.c2f_p3_td(self._merge(self.fuse_p3_td, p4_up, p3))
        n3     = self.context_p3(n3)

        n3_down = self.down_p3_p4(n3)
        n4      = self.c2f_n4_bu(self._merge(self.fuse_n4_bu, n3_down, p4_fuse))

        n4_down = self.down_p4_p5(n4)
        n5      = self.c2f_n5_bu(self._merge(self.fuse_n5_bu, n4_down, p5))

        return self.iema3(n3), self.iema4(n4), self.iema5(n5)
