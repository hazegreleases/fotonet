"""
FOTONET Dual Head — NMS-free with One-to-One (O2O) + One-to-Many (O2M) branches.

Architecture:
  - DFL (Distribution Focal Loss) regression with reg_max=16
  - LTRB-based box decoding via softmax distribution
  - For each scale: two parallel conv heads (cls + reg), one O2O and one O2M.
  - During training: returns dict with both branches + raw distributions for DFL loss.
  - During inference: ONLY the O2O branch is used (true NMS-free).

Accepts explicit channel counts from the neck output.
For nano (w=0.25): in_channels = (64, 128, 256)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_head_convs(in_ch, mid_ch, out_ch, depthwise=False):
    """Two-layer conv head: BN+SiLU conv, then 1x1 output. Optional depthwise for efficiency."""
    if depthwise:
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 1),
        )
    return nn.Sequential(
        nn.Conv2d(in_ch, mid_ch, 3, 1, 1, bias=False),
        nn.BatchNorm2d(mid_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(mid_ch, mid_ch, 3, 1, 1, bias=False),
        nn.BatchNorm2d(mid_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(mid_ch, out_ch, 1),
    )


def _make_quality_convs(in_ch):
    """Cheap quality tower for localization-aware ranking."""
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(in_ch, 1, 1),
    )


def _fuse_quality_logits(cls_logits, quality_logits):
    cls_prob = cls_logits.sigmoid()
    q_prob = quality_logits.sigmoid()
    fused = (cls_prob * q_prob).clamp(1e-5, 1.0 - 1e-5)
    return torch.log(fused / (1.0 - fused))


class FOTONETHead(nn.Module):
    """
    Dual-assignment detection head with DFL regression.
    SOTA: Distribution Focal Loss (reg_max=16) for precise box regression.
    O2O branch: one prediction per GT (NMS-free at inference).
    O2M branch: top-k predictions per GT (rich supervision during training).
    """
    default_strides = [8, 16, 32]

    def __init__(self, nc=80, in_channels=(64, 128, 256), reg_max=16, imgsz=640, strides=None, quality_head=False):
        super().__init__()
        self.nc      = nc
        self.reg_max = reg_max
        self.imgsz   = imgsz
        self.quality_head = bool(quality_head)
        if strides is None:
            strides = [4, 8, 16, 32] if len(in_channels) == 4 else self.default_strides
        self.strides = [int(s) for s in strides]
        if len(self.strides) != len(in_channels):
            raise ValueError("Number of head strides must match number of feature maps.")

        reg_out = 4 * reg_max if reg_max > 1 else 4
        # DFL models output 4 * reg_max bins; DFL-free models output direct LTRB distances.
        self.cls_o2o = nn.ModuleList(
            _make_head_convs(c, c, nc, depthwise=True) for c in in_channels
        )
        self.reg_o2o = nn.ModuleList(
            _make_head_convs(c, c, reg_out) for c in in_channels
        )
        self.cls_o2m = nn.ModuleList(
            _make_head_convs(c, c, nc, depthwise=True) for c in in_channels
        )
        self.reg_o2m = nn.ModuleList(
            _make_head_convs(c, c, reg_out) for c in in_channels
        )
        if self.quality_head:
            self.quality_o2o = nn.ModuleList(_make_quality_convs(c) for c in in_channels)
            self.quality_o2m = nn.ModuleList(_make_quality_convs(c) for c in in_channels)
        else:
            self.quality_o2o = nn.ModuleList()
            self.quality_o2m = nn.ModuleList()

        # DFL projection buffer: [0, 1, 2, ..., reg_max-1]
        self.register_buffer('proj', torch.arange(reg_max, dtype=torch.float32))

        # Init cls bias (called again after general weight init in FOTONETModel)
        self._init_cls_bias()

        self._grid_cache = {}
        self.inference_only = False

    @property
    def has_o2m(self):
        return len(self.cls_o2m) > 0 and len(self.reg_o2m) > 0

    @property
    def has_quality_o2o(self):
        return self.quality_head and len(self.quality_o2o) > 0

    @property
    def has_quality_o2m(self):
        return self.quality_head and len(self.quality_o2m) > 0

    def strip_o2m_for_inference(self):
        """Drop training-only O2M heads from the module so deployment carries O2O params only."""
        self.cls_o2m = nn.ModuleList()
        self.reg_o2m = nn.ModuleList()
        self.quality_o2m = nn.ModuleList()
        self.inference_only = True
        return self

    def _init_cls_bias(self):
        """Set cls head biases so initial foreground confidence ≈ 1% (not 50%)."""
        bias_init = -4.595  # sigmoid(-4.595) ≈ 0.01
        for m in self.cls_o2o:
            nn.init.constant_(m[-1].bias, bias_init)
        for m in self.cls_o2m:
            nn.init.constant_(m[-1].bias, bias_init)

    def _dfl(self, x):
        """DFL decode: [B, 4*reg_max, N] → [B, N, 4] expected LTRB values."""
        b, _, n = x.shape
        if self.reg_max <= 1:
            return F.softplus(x.view(b, 4, n)).permute(0, 2, 1)
        x = x.view(b, 4, self.reg_max, n)
        x = x.softmax(2)
        x = (x * self.proj.view(1, 1, -1, 1)).sum(2)  # [B, 4, N]
        return x.permute(0, 2, 1)  # [B, N, 4]

    def _get_grid(self, nx, ny, device):
        """Get anchor centers at cell midpoints [1, nx*ny, 2]. Cached."""
        key = (ny, nx, str(device))
        if key not in self._grid_cache:
            yv, xv = torch.meshgrid(
                torch.arange(ny, device=device, dtype=torch.float32) + 0.5,
                torch.arange(nx, device=device, dtype=torch.float32) + 0.5,
                indexing='ij'
            )
            self._grid_cache[key] = torch.stack((xv, yv), 2).view(1, -1, 2)
        return self._grid_cache[key]

    def _dist2bbox(self, ltrb, anchor, stride, imgsz):
        """Convert LTRB distances (grid-cell units) to normalized xywh boxes."""
        lt, rb = ltrb.chunk(2, -1)  # [B, N, 2] each
        x1y1 = (anchor - lt) * stride
        x2y2 = (anchor + rb) * stride
        c_xy = (x1y1 + x2y2) * 0.5
        wh   = (x2y2 - x1y1).clamp(min=0)
        return torch.cat([c_xy, wh], -1) / imgsz  # normalized xywh

    def forward(self, feats, imgsz=None, return_all=False, use_o2m=None):
        """
        Args:
            feats: list of feature tensors from neck [(B,C,H,W), ...]
            imgsz: actual image size (float/tensor). If None, falls back to self.imgsz.
            return_all: when True, returns both O2O and O2M branches even in eval mode.
            use_o2m: optional training-time switch for the O2M branch.
        """
        actual_imgsz = imgsz if imgsz is not None else float(self.imgsz)

        out_o2o = []
        need_meta = self.training or return_all
        need_o2m = (self.training if use_o2m is None else bool(use_o2m)) or return_all
        if need_o2m and not self.has_o2m:
            raise RuntimeError("This checkpoint is inference-only: O2M heads were stripped.")

        out_o2m = []
        dist_o2o_parts = []
        dist_o2m_parts = []
        quality_o2o_parts = []
        quality_o2m_parts = []
        anchor_parts = []
        stride_parts = []
        feat_shapes = []

        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            stride = self.strides[i]
            grid = self._get_grid(w, h, feat.device)
            feat_shapes.append((h, w))

            # --- O2O branch ---
            cls_o2o = self.cls_o2o[i](feat).view(b, self.nc, -1).permute(0, 2, 1)
            reg_ch = 4 * self.reg_max if self.reg_max > 1 else 4
            reg_raw_o2o = self.reg_o2o[i](feat).view(b, reg_ch, -1)
            ltrb_o2o = self._dfl(reg_raw_o2o)
            boxes_o2o = self._dist2bbox(ltrb_o2o, grid, stride, actual_imgsz)
            out_o2o.append(torch.cat([cls_o2o, boxes_o2o], dim=-1))
            if self.has_quality_o2o:
                q_o2o = self.quality_o2o[i](feat).view(b, 1, -1).permute(0, 2, 1)
                quality_o2o_parts.append(q_o2o)

            if need_meta:
                dist_o2o_parts.append(reg_raw_o2o.permute(0, 2, 1))
                n_anchors = h * w
                anchor_parts.append(grid.squeeze(0))
                stride_parts.append(torch.full((n_anchors, 1), stride,
                                               device=feat.device, dtype=torch.float32))

            if need_o2m:
                # --- O2M branch ---
                cls_o2m = self.cls_o2m[i](feat).view(b, self.nc, -1).permute(0, 2, 1)
                reg_raw_o2m = self.reg_o2m[i](feat).view(b, reg_ch, -1)
                ltrb_o2m = self._dfl(reg_raw_o2m)
                boxes_o2m = self._dist2bbox(ltrb_o2m, grid, stride, actual_imgsz)
                out_o2m.append(torch.cat([cls_o2m, boxes_o2m], dim=-1))
                if self.has_quality_o2m:
                    q_o2m = self.quality_o2m[i](feat).view(b, 1, -1).permute(0, 2, 1)
                    quality_o2m_parts.append(q_o2m)

                # Store raw distributions for DFL loss [B, N, 4*reg_max]
                dist_o2m_parts.append(reg_raw_o2m.permute(0, 2, 1))

        preds_o2o = torch.cat(out_o2o, dim=1)
        pred_quality_o2o = torch.cat(quality_o2o_parts, dim=1) if quality_o2o_parts else None

        if need_meta:
            shapes_tensor = torch.tensor(feat_shapes, device=preds_o2o.device, dtype=torch.int32)
            result = {
                "pred_logits_o2o": preds_o2o[..., :self.nc],
                "pred_boxes_o2o":  preds_o2o[..., self.nc:],
                "pred_dist_o2o":   torch.cat(dist_o2o_parts, dim=1),
                "anchor_points":   torch.cat(anchor_parts, dim=0),   # [total_N, 2]
                "stride_tensor":   torch.cat(stride_parts, dim=0),   # [total_N, 1]
                "feat_shapes":     shapes_tensor,
                "imgsz":           actual_imgsz,
            }
            if pred_quality_o2o is not None:
                result["pred_quality_o2o"] = pred_quality_o2o
            if need_o2m:
                preds_o2m = torch.cat(out_o2m, dim=1)
                result.update({
                    "pred_logits_o2m": preds_o2m[..., :self.nc],
                    "pred_boxes_o2m":  preds_o2m[..., self.nc:],
                    "pred_dist_o2m":   torch.cat(dist_o2m_parts, dim=1),
                })
                if quality_o2m_parts:
                    result["pred_quality_o2m"] = torch.cat(quality_o2m_parts, dim=1)
            return result

        if pred_quality_o2o is not None:
            fused_logits = _fuse_quality_logits(preds_o2o[..., :self.nc], pred_quality_o2o)
            preds_o2o = torch.cat([fused_logits, preds_o2o[..., self.nc:]], dim=-1)

        return preds_o2o
