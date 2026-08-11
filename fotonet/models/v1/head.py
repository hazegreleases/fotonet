"""Production NMS-free dual-assignment detection head."""
import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_quality_convs(in_ch):
    """Cheap quality tower for localization-aware ranking."""
    return nn.Sequential(nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False), nn.BatchNorm2d(in_ch), nn.SiLU(inplace=True), nn.Conv2d(in_ch, 1, 1))

def _shared_channels(strides):
    """Cap shared head width while retaining more capacity on coarser levels."""
    widths = {4: 40, 8: 56, 16: 72, 32: 96}
    return [widths.get(int(stride), 96) for stride in strides]

def _make_shared_stem(in_ch, hidden_ch):
    """One spatial mixer and one pointwise bottleneck shared by all tasks."""
    return nn.Sequential(nn.Conv2d(in_ch, in_ch, 5, 1, 2, groups=in_ch, bias=False), nn.BatchNorm2d(in_ch), nn.SiLU(inplace=True), nn.Conv2d(in_ch, hidden_ch, 1, bias=False), nn.BatchNorm2d(hidden_ch), nn.SiLU(inplace=True))

def _make_adapter(in_ch, hidden_ch, out_ch):
    """Tiny task/assignment-specific pointwise adapter and output projection."""
    return nn.Sequential(nn.Conv2d(in_ch, hidden_ch, 1, bias=False), nn.BatchNorm2d(hidden_ch), nn.SiLU(inplace=True), nn.Conv2d(hidden_ch, out_ch, 1))

def _fuse_quality_logits(cls_logits, quality_logits):
    cls_prob = cls_logits.sigmoid()
    q_prob = quality_logits.sigmoid()
    fused = (cls_prob * q_prob).clamp(1e-05, 1.0 - 1e-05)
    return torch.log(fused / (1.0 - fused))

class Head(nn.Module):
    """Production dual-assignment head. A shared per-level spatial stem feeds small independent O2O and training-only O2M adapters."""
    default_strides = [8, 16, 32]

    def __init__(self, nc=80, in_channels=(64, 128, 256), reg_max=16, imgsz=640, strides=None, quality_head=False):
        nn.Module.__init__(self)
        self.nc = int(nc)
        self.reg_max = int(reg_max)
        self.imgsz = imgsz
        self.quality_head = bool(quality_head)
        if strides is None:
            strides = [4, 8, 16, 32] if len(in_channels) == 4 else self.default_strides
        self.strides = [int(s) for s in strides]
        if len(self.strides) != len(in_channels):
            raise ValueError('Number of head strides must match number of feature maps.')
        reg_out = 4 * self.reg_max if self.reg_max > 1 else 4
        self.shared_channels = _shared_channels(self.strides)
        self.adapter_channels = [max(24, hidden // 2) for hidden in self.shared_channels]
        self.shared_stem = nn.ModuleList((_make_shared_stem(in_ch, hidden) for (in_ch, hidden) in zip(in_channels, self.shared_channels)))

        def adapters(out_channels):
            return nn.ModuleList((_make_adapter(hidden, adapter, out_channels) for (hidden, adapter) in zip(self.shared_channels, self.adapter_channels)))
        self.cls_o2o = adapters(self.nc)
        self.reg_o2o = adapters(reg_out)
        self.cls_o2m = adapters(self.nc)
        self.reg_o2m = adapters(reg_out)
        if self.quality_head:
            self.quality_o2o = nn.ModuleList((_make_quality_convs(hidden) for hidden in self.shared_channels))
            self.quality_o2m = nn.ModuleList((_make_quality_convs(hidden) for hidden in self.shared_channels))
        else:
            self.quality_o2o = nn.ModuleList()
            self.quality_o2m = nn.ModuleList()
        self.register_buffer('proj', torch.arange(self.reg_max, dtype=torch.float32))
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
        bias_init = -4.595
        for m in self.cls_o2o:
            nn.init.constant_(m[-1].bias, bias_init)
        for m in self.cls_o2m:
            nn.init.constant_(m[-1].bias, bias_init)

    def _dfl(self, x):
        """DFL decode: [B, 4*reg_max, N] → [B, N, 4] expected LTRB values."""
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
            (yv, xv) = torch.meshgrid(torch.arange(ny, device=device, dtype=torch.float32) + 0.5, torch.arange(nx, device=device, dtype=torch.float32) + 0.5, indexing='ij')
            return torch.stack((xv, yv), 2).view(1, -1, 2).to(dtype=dtype)
        key = (ny, nx, str(device), str(dtype))
        if key not in self._grid_cache:
            (yv, xv) = torch.meshgrid(torch.arange(ny, device=device, dtype=torch.float32) + 0.5, torch.arange(nx, device=device, dtype=torch.float32) + 0.5, indexing='ij')
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
            raise ValueError('imgsz must be a scalar or [width, height].')
        scale = torch.stack((size[0], size[1], size[0], size[1]))
        return torch.cat([c_xy, wh], -1) / scale

    def _level_features(self, feat, index):
        shared = self.shared_stem[index](feat)
        return (shared, shared, shared)

    def forward(self, feats, imgsz=None, return_all=False, use_o2m=None):
        tracing_export = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing()
        if tracing_export:
            return_all = False
            use_o2m = None
        else:
            if not isinstance(return_all, bool):
                raise TypeError('return_all must be a bool.')
            if use_o2m is not None and (not isinstance(use_o2m, bool)):
                raise TypeError('use_o2m must be a bool or None.')
        actual_imgsz = imgsz if imgsz is not None else float(self.imgsz)
        requested_o2m = self.training if use_o2m is None else use_o2m
        run_o2o = self.training or return_all or (not requested_o2m)
        need_meta = self.training or return_all
        need_o2m = requested_o2m or (return_all and use_o2m is None)
        if need_o2m and (not self.has_o2m):
            raise RuntimeError('This checkpoint is inference-only: O2M heads were stripped.')
        out_o2o = []
        out_o2m = []
        dist_o2o_parts = []
        dist_o2m_parts = []
        quality_o2o_parts = []
        quality_o2m_parts = []
        anchor_parts = []
        stride_parts = []
        feat_shapes = []
        for (i, feat) in enumerate(feats):
            (b, _, h, w) = feat.shape
            stride = self.strides[i]
            grid = self._get_grid(w, h, feat.device, feat.dtype)
            feat_shapes.append((h, w))
            (cls_feat, reg_feat, quality_feat) = self._level_features(feat, i)
            reg_ch = 4 * self.reg_max if self.reg_max > 1 else 4
            if run_o2o:
                cls_o2o = self.cls_o2o[i](cls_feat).view(b, self.nc, -1).permute(0, 2, 1)
                reg_raw_o2o = self.reg_o2o[i](reg_feat).view(b, reg_ch, -1)
                ltrb_o2o = self._dfl(reg_raw_o2o)
                boxes_o2o = self._dist2bbox(ltrb_o2o, grid, stride, actual_imgsz)
                out_o2o.append(torch.cat([cls_o2o, boxes_o2o], dim=-1))
                if self.has_quality_o2o:
                    quality_o2o_parts.append(self.quality_o2o[i](quality_feat).view(b, 1, -1).permute(0, 2, 1))
            if need_meta:
                if run_o2o:
                    dist_o2o_parts.append(reg_raw_o2o.permute(0, 2, 1))
                n_anchors = h * w
                anchor_parts.append(grid.squeeze(0))
                stride_parts.append(torch.full((n_anchors, 1), stride, device=feat.device, dtype=feat.dtype))
            if need_o2m:
                cls_o2m = self.cls_o2m[i](cls_feat).view(b, self.nc, -1).permute(0, 2, 1)
                reg_raw_o2m = self.reg_o2m[i](reg_feat).view(b, reg_ch, -1)
                ltrb_o2m = self._dfl(reg_raw_o2m)
                boxes_o2m = self._dist2bbox(ltrb_o2m, grid, stride, actual_imgsz)
                out_o2m.append(torch.cat([cls_o2m, boxes_o2m], dim=-1))
                if self.has_quality_o2m:
                    quality_o2m_parts.append(self.quality_o2m[i](quality_feat).view(b, 1, -1).permute(0, 2, 1))
                dist_o2m_parts.append(reg_raw_o2m.permute(0, 2, 1))
        preds_o2o = torch.cat(out_o2o, dim=1) if out_o2o else None
        pred_quality_o2o = torch.cat(quality_o2o_parts, dim=1) if quality_o2o_parts else None
        if need_meta:
            if preds_o2o is None:
                raise RuntimeError('return_all requires the O2O branch.')
            result = {'pred_logits_o2o': preds_o2o[..., :self.nc], 'pred_boxes_o2o': preds_o2o[..., self.nc:], 'pred_dist_o2o': torch.cat(dist_o2o_parts, dim=1), 'anchor_points': torch.cat(anchor_parts, dim=0), 'stride_tensor': torch.cat(stride_parts, dim=0), 'feat_shapes': torch.tensor(feat_shapes, device=preds_o2o.device, dtype=torch.int32), 'imgsz': actual_imgsz}
            if pred_quality_o2o is not None:
                result['pred_quality_o2o'] = pred_quality_o2o
            if need_o2m:
                preds_o2m = torch.cat(out_o2m, dim=1)
                result.update({'pred_logits_o2m': preds_o2m[..., :self.nc], 'pred_boxes_o2m': preds_o2m[..., self.nc:], 'pred_dist_o2m': torch.cat(dist_o2m_parts, dim=1)})
                if quality_o2m_parts:
                    result['pred_quality_o2m'] = torch.cat(quality_o2m_parts, dim=1)
            return result
        if not run_o2o:
            preds_o2m = torch.cat(out_o2m, dim=1)
            if quality_o2m_parts:
                quality_o2m = torch.cat(quality_o2m_parts, dim=1)
                fused_logits = _fuse_quality_logits(preds_o2m[..., :self.nc], quality_o2m)
                preds_o2m = torch.cat([fused_logits, preds_o2m[..., self.nc:]], dim=-1)
            return preds_o2m
        if pred_quality_o2o is not None:
            fused_logits = _fuse_quality_logits(preds_o2o[..., :self.nc], pred_quality_o2o)
            preds_o2o = torch.cat([fused_logits, preds_o2o[..., self.nc:]], dim=-1)
        return preds_o2o
__all__ = ['Head']
