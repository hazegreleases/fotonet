"""
FOTONET Loss Functions.
DualLoss: O2O + O2M assignment with Focal + DFL + CIoU.
Uses Distribution Focal Loss (DFL) for regression instead of L1.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fotonet.utils.boxes import xywh_to_xyxy


def bbox_ciou(box1_xyxy, box2_xyxy, eps=1e-7):
    """CIoU between paired boxes. [N,4] vs [N,4]."""
    b1x1, b1y1, b1x2, b1y2 = box1_xyxy.unbind(1)
    b2x1, b2y1, b2x2, b2y2 = box2_xyxy.unbind(1)

    inter = (torch.min(b1x2, b2x2) - torch.max(b1x1, b2x1)).clamp(0) * \
            (torch.min(b1y2, b2y2) - torch.max(b1y1, b2y1)).clamp(0)
    w1, h1 = b1x2 - b1x1, b1y2 - b1y1
    w2, h2 = b2x2 - b2x1, b2y2 - b2y1
    union  = w1*h1 + w2*h2 - inter + eps
    iou    = inter / union
    if iou.numel() == 0: return iou

    cw = torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1)
    ch = torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)
    c2 = cw**2 + ch**2 + eps
    rho2 = ((b1x1+b1x2 - b2x1-b2x2)**2 + (b1y1+b1y2 - b2y1-b2y2)**2) / 4
    v = (4 / math.pi**2) * (torch.atan(w2/(h2+eps)) - torch.atan(w1/(h1+eps)))**2
    with torch.no_grad():
        alpha = v / (v - iou + 1 + eps)
    return iou - (rho2/c2 + v*alpha)


def _dfl_loss(pred_dist, target, reg_max):
    """
    Distribution Focal Loss for one coordinate.
    pred_dist: [N, reg_max] raw logits
    target: [N] continuous regression target in [0, reg_max-1]
    Returns: [N] per-sample loss
    """
    target = target.clamp(0, reg_max - 1 - 0.01)
    tl = target.long()
    tr = tl + 1
    wl = tr.float() - target
    wr = 1.0 - wl
    return (F.cross_entropy(pred_dist, tl, reduction='none') * wl +
            F.cross_entropy(pred_dist, tr, reduction='none') * wr)


def _bbox2dist(anchor_points, gt_boxes_xywh, stride, imgsz, reg_max=None):
    """
    Convert GT xywh (normalized) to LTRB distances in grid-cell units.
    anchor_points: [M, 2] in grid-cell units (0.5, 1.5, ...)
    gt_boxes_xywh: [M, 4] normalized (cx, cy, w, h)
    stride: [M, 1] or [M] pixel stride per anchor
    imgsz: float image size
    Returns: [M, 4] LTRB in grid-cell units, clamped to [0, reg_max-1]
    """
    stride = stride.view(-1)
    cx, cy, w, h = gt_boxes_xywh.unbind(-1)
    # Convert normalized coords to grid-cell units
    x1 = (cx - w * 0.5) * imgsz / stride
    y1 = (cy - h * 0.5) * imgsz / stride
    x2 = (cx + w * 0.5) * imgsz / stride
    y2 = (cy + h * 0.5) * imgsz / stride

    l = anchor_points[:, 0] - x1
    t = anchor_points[:, 1] - y1
    r = x2 - anchor_points[:, 0]
    b = y2 - anchor_points[:, 1]
    dist = torch.stack([l, t, r, b], -1).clamp_min(0)
    if reg_max is not None and reg_max > 1:
        dist = dist.clamp(0, reg_max - 1 - 0.01)
    return dist


def calc_branch_loss(pred_logits, pred_boxes, pred_dist, targets, indices, nc,
                     anchor_points, stride_tensor, reg_max, imgsz,
                     class_weights=None, label_smoothing=0.05,
                     focal_gamma_pos=2.0, focal_gamma_neg=3.0,
                     quality_targets=True, quality_power=1.0, quality_floor=0.05,
                     quality_mix=1.0, qfl_beta=2.0, hard_negative_topk=256,
                     hard_negative_weight=0.15, hard_negative_min_score=0.15,
                     dfl_mean_over_coords=True):
    """
    Compute classification (focal) + DFL + CIoU loss for one branch.
    Returns SUMS of (cls_loss, dfl_loss, ciou_loss).
    """
    bs, n_queries, _ = pred_logits.shape
    device = pred_logits.device

    # Stability: clamp logits to prevent extreme sigmoids
    pred_logits = pred_logits.clamp(-30, 30)

    tgt_onehot = torch.zeros_like(pred_logits)
    all_p_idx = []
    all_t_idx = []
    all_b_idx = []
    for i, (p_idx, t_idx) in enumerate(indices):
        if len(p_idx) > 0:
            all_p_idx.append(p_idx)
            all_t_idx.append(t_idx)
            all_b_idx.append(torch.full_like(p_idx, i))

    if not all_p_idx:
        # All-background batch
        tgt_smoothed = torch.zeros_like(pred_logits)
        ce = F.binary_cross_entropy_with_logits(pred_logits, tgt_smoothed, reduction='none')
        p     = torch.sigmoid(pred_logits)
        focal = ce * (p ** focal_gamma_neg)
        focal_f32 = focal.to(torch.float32)
        loss_cls = focal_f32.sum()
        if hard_negative_topk > 0 and hard_negative_weight > 0:
            neg_scores = p.detach().reshape(-1)
            hard_mask = neg_scores > hard_negative_min_score
            hard_losses = focal_f32.reshape(-1)[hard_mask]
            if hard_losses.numel() > 0:
                k = min(int(hard_negative_topk), hard_losses.numel())
                loss_cls = loss_cls + hard_negative_weight * hard_losses.topk(k).values.sum()
        return loss_cls, torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    all_p_idx = torch.cat(all_p_idx)
    all_b_idx = torch.cat(all_b_idx)

    all_tgt_labels = torch.cat([targets[i]["labels"][indices[i][1]] for i in range(bs) if len(indices[i][1]) > 0])
    all_tgt_boxes  = torch.cat([targets[i]["boxes"][indices[i][1]] for i in range(bs) if len(indices[i][1]) > 0])

    tgt_onehot[all_b_idx, all_p_idx, all_tgt_labels] = 1.0

    # --- CIoU Loss (on decoded boxes) ---
    all_pred_boxes = pred_boxes[all_b_idx, all_p_idx]
    p_xyxy = xywh_to_xyxy(all_pred_boxes)
    t_xyxy = xywh_to_xyxy(all_tgt_boxes)
    matched_ciou = bbox_ciou(p_xyxy, t_xyxy)
    loss_ciou = (1 - matched_ciou).sum()

    # --- DFL Loss (on raw distributions) ---
    matched_dist = pred_dist[all_b_idx, all_p_idx]  # [M, 4*reg_max]
    matched_anchors = anchor_points[all_p_idx]       # [M, 2]
    matched_strides = stride_tensor[all_p_idx]       # [M, 1]

    if reg_max <= 1:
        gt_ltrb = _bbox2dist(matched_anchors, all_tgt_boxes, matched_strides, imgsz, reg_max=None)
        pred_ltrb = F.softplus(matched_dist.view(-1, 4))
        dfl_sum = F.smooth_l1_loss(pred_ltrb, gt_ltrb, reduction="sum")
        if dfl_mean_over_coords:
            dfl_sum = dfl_sum / 4.0
    else:
        gt_ltrb = _bbox2dist(matched_anchors, all_tgt_boxes, matched_strides, imgsz, reg_max)
        matched_dist = matched_dist.view(-1, 4, reg_max)  # [M, 4, reg_max]

        dfl_sum = torch.tensor(0.0, device=device)
        for j in range(4):
            dfl_sum = dfl_sum + _dfl_loss(matched_dist[:, j], gt_ltrb[:, j], reg_max).sum()
        if dfl_mean_over_coords:
            dfl_sum = dfl_sum / 4.0

    # --- Quality-aware Focal Classification Loss ---
    p = torch.sigmoid(pred_logits)
    pos_mask = tgt_onehot > 0.5

    if quality_targets:
        tgt_scores = torch.zeros_like(pred_logits)
        quality = matched_ciou.detach().clamp(min=quality_floor, max=1.0).pow(quality_power)
        hard_target = torch.full_like(quality, 1.0 - label_smoothing)
        quality_target = quality * (1.0 - label_smoothing)
        pos_target = hard_target.lerp(quality_target, quality_mix)
        tgt_scores[all_b_idx, all_p_idx, all_tgt_labels] = pos_target.to(dtype=tgt_scores.dtype)
        ce = F.binary_cross_entropy_with_logits(pred_logits, tgt_scores, reduction='none')
        focal_factor = torch.where(
            pos_mask,
            (tgt_scores - p).abs().pow(qfl_beta),
            p.pow(focal_gamma_neg),
        )
    else:
        tgt_scores = torch.where(pos_mask, 1.0 - label_smoothing, 0.0)
        ce = F.binary_cross_entropy_with_logits(pred_logits, tgt_scores, reduction='none')
        focal_factor = torch.where(
            pos_mask,
            (1.0 - p) ** focal_gamma_pos,
            p ** focal_gamma_neg,
        )

    focal = ce * focal_factor
    focal_f32 = focal.to(torch.float32)

    if class_weights is not None:
        w = 1.0 + (class_weights - 1.0) * tgt_onehot
        loss_cls = (w * focal_f32).sum()
    else:
        loss_cls = focal_f32.sum()

    if hard_negative_topk > 0 and hard_negative_weight > 0:
        neg_mask = ~pos_mask
        hard_candidate = neg_mask & (p.detach() > hard_negative_min_score)
        hard_losses = focal_f32[hard_candidate]
        if hard_losses.numel() > 0:
            k = min(int(hard_negative_topk), hard_losses.numel())
            loss_cls = loss_cls + hard_negative_weight * hard_losses.topk(k).values.sum()

    return loss_cls, dfl_sum, loss_ciou


def calc_quality_loss(pred_quality, pred_boxes, targets, indices):
    """BCE quality target: positives get detached CIoU quality, negatives get 0."""
    if pred_quality is None:
        return None
    bs, n_queries, _ = pred_quality.shape
    device = pred_quality.device
    targets_quality = torch.zeros_like(pred_quality)

    all_p_idx = []
    all_b_idx = []
    all_tgt_boxes = []
    for batch_idx, (p_idx, t_idx) in enumerate(indices):
        if len(p_idx) == 0:
            continue
        all_p_idx.append(p_idx)
        all_b_idx.append(torch.full_like(p_idx, batch_idx))
        all_tgt_boxes.append(targets[batch_idx]["boxes"][t_idx])

    if all_p_idx:
        all_p_idx = torch.cat(all_p_idx)
        all_b_idx = torch.cat(all_b_idx)
        all_tgt_boxes = torch.cat(all_tgt_boxes).to(device=device, dtype=pred_boxes.dtype)
        matched_boxes = pred_boxes[all_b_idx, all_p_idx]
        quality = bbox_ciou(
            xywh_to_xyxy(matched_boxes.float()),
            xywh_to_xyxy(all_tgt_boxes.float()),
        ).detach().clamp(0.0, 1.0)
        targets_quality[all_b_idx, all_p_idx, 0] = quality.to(dtype=targets_quality.dtype)

    loss = F.binary_cross_entropy_with_logits(
        pred_quality.float(),
        targets_quality.float(),
        reduction="sum",
    )
    return loss / max(float(bs * n_queries), 1.0)


def calc_consistency_loss(outputs, targets, indices, cls_weight=0.25, box_weight=0.75):
    """
    O2O <- O2M consistency on O2O positives.
    The O2M branch sees richer supervision, so its detached predictions act as a
    teacher for the NMS-free O2O branch once training has warmed up.
    """
    device = outputs["pred_logits_o2o"].device
    bs = outputs["pred_logits_o2o"].shape[0]
    all_p_idx, all_b_idx = [], []
    for i, (p_idx, t_idx) in enumerate(indices):
        if len(p_idx) > 0:
            all_p_idx.append(p_idx)
            all_b_idx.append(torch.full_like(p_idx, i))

    if not all_p_idx:
        return torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)

    all_p_idx = torch.cat(all_p_idx)
    all_b_idx = torch.cat(all_b_idx)
    all_tgt_labels = torch.cat([
        targets[i]["labels"][indices[i][1]]
        for i in range(bs)
        if len(indices[i][1]) > 0
    ])

    o2o_logits = outputs["pred_logits_o2o"][all_b_idx, all_p_idx]
    o2m_logits = outputs["pred_logits_o2m"][all_b_idx, all_p_idx].detach()
    row = torch.arange(all_tgt_labels.numel(), device=device)
    teacher_scores = torch.sigmoid(o2m_logits[row, all_tgt_labels]).clamp(0.01, 0.99)
    cls_loss = F.binary_cross_entropy_with_logits(
        o2o_logits[row, all_tgt_labels],
        teacher_scores,
        reduction="sum",
    )

    o2o_boxes = outputs["pred_boxes_o2o"][all_b_idx, all_p_idx]
    o2m_boxes = outputs["pred_boxes_o2m"][all_b_idx, all_p_idx].detach()
    box_loss = (1.0 - bbox_ciou(xywh_to_xyxy(o2o_boxes), xywh_to_xyxy(o2m_boxes))).sum()

    return cls_weight * cls_loss + box_weight * box_loss, torch.as_tensor(
        max(float(all_p_idx.numel()), 1.0), device=device
    )


class DualLoss(nn.Module):
    """
    Dual Assignment Loss: O2O + O2M branches.
    Uses Focal (cls) + DFL (reg distribution) + CIoU (box geometry).
    Loss weights are configurable so recipe-specific tuning stays outside the model graph.
    """
    def __init__(self, nc, matcher_o2o, matcher_o2m, reg_max=16,
                 w_cls=1.0, w_box=7.5, w_dfl=1.5, class_weights=None,
                 o2o_label_smoothing=0.01, o2m_label_smoothing=0.05,
                 focal_gamma_pos=2.0, focal_gamma_neg=2.0,
                 quality_targets=True, quality_power=1.0, quality_floor=0.05,
                 quality_start_epoch=25, quality_warmup_epochs=50,
                 qfl_beta=2.0, hard_negative_topk=256,
                 hard_negative_weight=0.10, hard_negative_min_score=0.15,
                 hard_negative_start_epoch=8, hard_negative_warmup_epochs=32,
                 o2o_weight=1.0, o2m_weight=1.0,
                 consistency_weight=0.35, consistency_start_epoch=10,
                 consistency_warmup_epochs=40, consistency_cls_weight=0.25,
                 consistency_box_weight=0.75,
                 prog_loss=True, prog_start_epoch=1, prog_warmup_epochs=60,
                 cls_start_factor=1.50, box_start_factor=1.30,
                 dfl_start_factor=0.85, dfl_mean_over_coords=True,
                 o2m_full_epochs=None, o2m_sparse_epochs=None,
                 o2m_every_n_steps=1, o2m_disable_epoch=None,
                 exact_o2o_warmup_epochs=None, exact_o2o_period=1,
                 exact_o2o_period_end_epoch=None,
                 quality=1.0, quality_loss_start_epoch=1,
                 quality_loss_warmup_epochs=25):
        super().__init__()
        self.nc          = nc
        self.reg_max     = reg_max
        self.matcher_o2o = matcher_o2o
        self.matcher_o2m = matcher_o2m
        self.w_cls  = w_cls
        self.w_box  = w_box
        self.w_dfl  = w_dfl
        self.o2o_label_smoothing = o2o_label_smoothing
        self.o2m_label_smoothing = o2m_label_smoothing
        self.focal_gamma_pos = focal_gamma_pos
        self.focal_gamma_neg = focal_gamma_neg
        self.quality_targets = quality_targets
        self.quality_power = quality_power
        self.quality_floor = quality_floor
        self.quality_start_epoch = quality_start_epoch
        self.quality_warmup_epochs = quality_warmup_epochs
        self.qfl_beta = qfl_beta
        self.hard_negative_topk = hard_negative_topk
        self.hard_negative_weight = hard_negative_weight
        self.hard_negative_min_score = hard_negative_min_score
        self.hard_negative_start_epoch = hard_negative_start_epoch
        self.hard_negative_warmup_epochs = hard_negative_warmup_epochs
        self.o2o_weight = float(o2o_weight)
        self.o2m_weight = float(o2m_weight)
        self.consistency_weight = float(consistency_weight)
        self.consistency_start_epoch = consistency_start_epoch
        self.consistency_warmup_epochs = consistency_warmup_epochs
        self.consistency_cls_weight = float(consistency_cls_weight)
        self.consistency_box_weight = float(consistency_box_weight)
        self.prog_loss = bool(prog_loss)
        self.prog_start_epoch = prog_start_epoch
        self.prog_warmup_epochs = prog_warmup_epochs
        self.cls_start_factor = float(cls_start_factor)
        self.box_start_factor = float(box_start_factor)
        self.dfl_start_factor = float(dfl_start_factor)
        self.dfl_mean_over_coords = bool(dfl_mean_over_coords)
        self.w_quality = float(quality)
        self.quality_loss_start_epoch = int(quality_loss_start_epoch)
        self.quality_loss_warmup_epochs = int(quality_loss_warmup_epochs)
        self.o2m_full_epochs = None if o2m_full_epochs is None else int(o2m_full_epochs)
        self.o2m_sparse_epochs = None if o2m_sparse_epochs is None else int(o2m_sparse_epochs)
        self.o2m_every_n_steps = max(int(o2m_every_n_steps or 1), 1)
        self.o2m_disable_epoch = None if o2m_disable_epoch is None else int(o2m_disable_epoch)
        self.exact_o2o_warmup_epochs = None if exact_o2o_warmup_epochs is None else int(exact_o2o_warmup_epochs)
        self.exact_o2o_period = max(int(exact_o2o_period or 1), 1)
        self.exact_o2o_period_end_epoch = (
            None if exact_o2o_period_end_epoch is None else int(exact_o2o_period_end_epoch)
        )
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.view(1, 1, -1))
        else:
            self.register_buffer("class_weights", None)

    @staticmethod
    def _epoch_ramp(current_epoch, start_epoch, warmup_epochs):
        """1-indexed linear ramp. Returns 0 before start, 1 after warmup."""
        epoch = float(current_epoch) + 1.0
        start = max(float(start_epoch), 1.0)
        warmup = max(float(warmup_epochs), 0.0)
        if epoch < start:
            return 0.0
        if warmup <= 0:
            return 1.0
        return max(0.0, min(1.0, (epoch - start + 1.0) / warmup))

    def use_o2m(self, current_epoch=0, global_step=0):
        """Return whether this step should compute the training-only O2M branch."""
        epoch = int(current_epoch) + 1
        if self.o2m_disable_epoch is not None and epoch > self.o2m_disable_epoch:
            return False
        if self.o2m_full_epochs is None or epoch <= self.o2m_full_epochs:
            return self.o2m_weight > 0
        if self.o2m_sparse_epochs is not None and epoch > self.o2m_sparse_epochs:
            return False
        return self.o2m_weight > 0 and (int(global_step) % self.o2m_every_n_steps == 0)

    def _use_exact_o2o(self, current_epoch=0, global_step=0):
        epoch = int(current_epoch) + 1
        if self.exact_o2o_warmup_epochs is None:
            return bool(getattr(self.matcher_o2o, "exact", True))
        if epoch <= self.exact_o2o_warmup_epochs:
            return True
        if self.exact_o2o_period_end_epoch is not None and epoch > self.exact_o2o_period_end_epoch:
            return False
        return int(global_step) % self.exact_o2o_period == 0

    def forward(self, outputs, targets, current_epoch=0, max_epochs=1, global_step=0):
        feat_shapes    = outputs.get("feat_shapes", None)
        anchor_points  = outputs["anchor_points"]   # [total_N, 2]
        stride_tensor  = outputs["stride_tensor"]   # [total_N, 1]
        imgsz          = outputs["imgsz"]

        o2o_in = {
            "pred_logits": outputs["pred_logits_o2o"],
            "pred_boxes":  outputs["pred_boxes_o2o"],
        }
        old_exact = getattr(self.matcher_o2o, "exact", None)
        if old_exact is not None:
            self.matcher_o2o.exact = self._use_exact_o2o(current_epoch, global_step)
        try:
            idx_o2o = self.matcher_o2o(o2o_in, targets, feat_shapes=feat_shapes)
        finally:
            if old_exact is not None:
                self.matcher_o2o.exact = old_exact

        has_o2m_outputs = all(k in outputs for k in ("pred_logits_o2m", "pred_boxes_o2m", "pred_dist_o2m"))
        use_o2m = has_o2m_outputs and self.use_o2m(current_epoch, global_step)
        if use_o2m:
            o2m_in = {
                "pred_logits": outputs["pred_logits_o2m"],
                "pred_boxes":  outputs["pred_boxes_o2m"],
            }
            idx_o2m = self.matcher_o2m(o2m_in, targets, feat_shapes=feat_shapes)
        else:
            o2m_in = None
            idx_o2m = [
                (
                    torch.empty(0, dtype=torch.long, device=outputs["pred_logits_o2o"].device),
                    torch.empty(0, dtype=torch.long, device=outputs["pred_logits_o2o"].device),
                )
                for _ in targets
            ]

        n_o2o = max(float(sum(len(i[0]) for i in idx_o2o)), 1.0)
        n_o2m_raw = float(sum(len(i[0]) for i in idx_o2m))
        n_o2m = max(n_o2m_raw, 1.0)
        quality_mix = self._epoch_ramp(
            current_epoch, self.quality_start_epoch, self.quality_warmup_epochs
        ) if self.quality_targets else 0.0
        hard_negative_weight = self.hard_negative_weight * self._epoch_ramp(
            current_epoch, self.hard_negative_start_epoch, self.hard_negative_warmup_epochs
        )
        prog_mix = self._epoch_ramp(
            current_epoch, self.prog_start_epoch, self.prog_warmup_epochs
        ) if self.prog_loss else 1.0
        cls_factor = self.cls_start_factor + (1.0 - self.cls_start_factor) * prog_mix
        box_factor = self.box_start_factor + (1.0 - self.box_start_factor) * prog_mix
        dfl_factor = self.dfl_start_factor + (1.0 - self.dfl_start_factor) * prog_mix
        active_w_cls = self.w_cls * cls_factor
        active_w_box = self.w_box * box_factor
        active_w_dfl = self.w_dfl * dfl_factor
        active_w_cons = self.consistency_weight * self._epoch_ramp(
            current_epoch, self.consistency_start_epoch, self.consistency_warmup_epochs
        )
        active_w_quality = self.w_quality * self._epoch_ramp(
            current_epoch,
            self.quality_loss_start_epoch,
            self.quality_loss_warmup_epochs,
        )

        # O2O Branch (sharper label smoothing)
        cls_o2o, dfl_o2o, iou_o2o = calc_branch_loss(
            o2o_in["pred_logits"], o2o_in["pred_boxes"],
            outputs["pred_dist_o2o"], targets, idx_o2o, self.nc,
            anchor_points, stride_tensor, self.reg_max, imgsz,
            self.class_weights,
            label_smoothing=self.o2o_label_smoothing,
            focal_gamma_pos=self.focal_gamma_pos,
            focal_gamma_neg=self.focal_gamma_neg,
            quality_targets=self.quality_targets,
            quality_power=self.quality_power,
            quality_floor=self.quality_floor,
            quality_mix=quality_mix,
            qfl_beta=self.qfl_beta,
            hard_negative_topk=self.hard_negative_topk,
            hard_negative_weight=hard_negative_weight,
            hard_negative_min_score=self.hard_negative_min_score,
            dfl_mean_over_coords=self.dfl_mean_over_coords,
        )

        if use_o2m:
            # O2M Branch (standard smoothing)
            cls_o2m, dfl_o2m, iou_o2m = calc_branch_loss(
                o2m_in["pred_logits"], o2m_in["pred_boxes"],
                outputs["pred_dist_o2m"], targets, idx_o2m, self.nc,
                anchor_points, stride_tensor, self.reg_max, imgsz,
                self.class_weights,
                label_smoothing=self.o2m_label_smoothing,
                focal_gamma_pos=self.focal_gamma_pos,
                focal_gamma_neg=self.focal_gamma_neg,
                quality_targets=self.quality_targets,
                quality_power=self.quality_power,
                quality_floor=self.quality_floor,
                quality_mix=quality_mix,
                qfl_beta=self.qfl_beta,
                hard_negative_topk=self.hard_negative_topk,
                hard_negative_weight=hard_negative_weight,
                hard_negative_min_score=self.hard_negative_min_score,
                dfl_mean_over_coords=self.dfl_mean_over_coords,
            )
            active_o2m_weight = self.o2m_weight
        else:
            zero = cls_o2o.new_tensor(0.0)
            cls_o2m = dfl_o2m = iou_o2m = zero
            active_o2m_weight = 0.0

        q_o2o = calc_quality_loss(
            outputs.get("pred_quality_o2o"),
            o2o_in["pred_boxes"],
            targets,
            idx_o2o,
        )
        if q_o2o is None:
            q_o2o = cls_o2o.new_tensor(0.0)

        if use_o2m:
            q_o2m = calc_quality_loss(
                outputs.get("pred_quality_o2m"),
                o2m_in["pred_boxes"],
                targets,
                idx_o2m,
            )
            if q_o2m is None:
                q_o2m = cls_o2o.new_tensor(0.0)
        else:
            q_o2m = cls_o2o.new_tensor(0.0)

        # Normalize branches independently, then blend O2O/O2M explicitly.
        loss_cls  = self.o2o_weight * (cls_o2o / n_o2o) + active_o2m_weight * (cls_o2m / n_o2m)
        loss_ciou = self.o2o_weight * (iou_o2o / n_o2o) + active_o2m_weight * (iou_o2m / n_o2m)
        loss_dfl  = self.o2o_weight * (dfl_o2o / n_o2o) + active_o2m_weight * (dfl_o2m / n_o2m)
        loss_quality = self.o2o_weight * q_o2o + active_o2m_weight * q_o2m
        if use_o2m and active_w_cons > 0:
            cons_raw, n_cons = calc_consistency_loss(
                outputs,
                targets,
                idx_o2o,
                cls_weight=self.consistency_cls_weight,
                box_weight=self.consistency_box_weight,
            )
            loss_cons = cons_raw / n_cons
        else:
            loss_cons = loss_cls.new_tensor(0.0)

        total = (
            active_w_cls * loss_cls
            + active_w_box * loss_ciou
            + active_w_dfl * loss_dfl
            + active_w_cons * loss_cons
            + active_w_quality * loss_quality
        )

        return {
            "loss":              total,
            "loss_cls":          loss_cls.detach(),
            "loss_box":          loss_dfl.detach(),    # DFL replaces old L1 "box" loss
            "loss_dfl_per_coord": (loss_dfl if self.dfl_mean_over_coords else loss_dfl / 4.0).detach(),
            "loss_ciou":         loss_ciou.detach(),
            "loss_cls_weighted":  (active_w_cls * loss_cls).detach(),
            "loss_dfl_weighted":  (active_w_dfl * loss_dfl).detach(),
            "loss_ciou_weighted": (active_w_box * loss_ciou).detach(),
            "loss_consistency":    loss_cons.detach(),
            "loss_consistency_weighted": (active_w_cons * loss_cons).detach(),
            "loss_quality":        loss_quality.detach(),
            "loss_quality_weighted": (active_w_quality * loss_quality).detach(),
            "active_w_cls":       torch.as_tensor(active_w_cls, device=loss_cls.device),
            "active_w_box":       torch.as_tensor(active_w_box, device=loss_cls.device),
            "active_w_dfl":       torch.as_tensor(active_w_dfl, device=loss_cls.device),
            "active_w_consistency": torch.as_tensor(active_w_cons, device=loss_cls.device),
            "active_w_quality":   torch.as_tensor(active_w_quality, device=loss_cls.device),
            "quality_mix":        torch.as_tensor(quality_mix, device=loss_cls.device),
            "hard_negative_weight": torch.as_tensor(hard_negative_weight, device=loss_cls.device),
            "num_pos_o2o":        torch.as_tensor(n_o2o, device=loss_cls.device),
            "num_pos_o2m":        torch.as_tensor(n_o2m_raw, device=loss_cls.device),
            "o2o_weight":         torch.as_tensor(self.o2o_weight, device=loss_cls.device),
            "o2m_weight":         torch.as_tensor(active_o2m_weight, device=loss_cls.device),
            "o2m_active":         torch.as_tensor(float(use_o2m), device=loss_cls.device),
            "exact_o2o_active":   torch.as_tensor(float(self._use_exact_o2o(current_epoch, global_step)), device=loss_cls.device),
        }


def get_loss(model, nc, class_weights=None, loss_hyp=None, matcher_hyp=None):
    from fotonet.utils.matcher import FastO2OAssigner, TaskAlignedAssigner
    # Get reg_max from head
    reg_max = getattr(model.head, 'reg_max', 16)
    loss_hyp = loss_hyp or {}
    matcher_hyp = matcher_hyp or {}
    cw_tensor = None
    if class_weights is not None:
        device = next(model.parameters()).device
        cw_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    o2o_assigner = FastO2OAssigner(
        cost_class=matcher_hyp.get("cost_class", 2.0),
        cost_bbox=matcher_hyp.get("cost_bbox", 2.0),
        cost_giou=matcher_hyp.get("cost_giou", 1.0),
        exact=matcher_hyp.get("exact_o2o", True),
    )
    o2m_assigner = TaskAlignedAssigner(
        topk=matcher_hyp.get("topk", 8),
        alpha=matcher_hyp.get("alpha", 1.0),
        beta=matcher_hyp.get("beta", 2.0),
        min_small_assign=matcher_hyp.get("min_small_assign", 4),
        small_obj_px=matcher_hyp.get("small_obj_px", 8.0),
        strides=matcher_hyp.get("strides", getattr(model.head, "strides", [8, 16, 32])),
    )
    return DualLoss(
        nc,
        o2o_assigner,
        o2m_assigner,
        reg_max=reg_max,
        class_weights=cw_tensor,
        w_cls=loss_hyp.get("cls", 1.0),
        w_box=loss_hyp.get("box", 7.5),
        w_dfl=loss_hyp.get("dfl", 1.5),
        o2o_label_smoothing=loss_hyp.get("o2o_label_smoothing", 0.01),
        o2m_label_smoothing=loss_hyp.get("o2m_label_smoothing", 0.05),
        focal_gamma_pos=loss_hyp.get("focal_gamma_pos", 2.0),
        focal_gamma_neg=loss_hyp.get("focal_gamma_neg", 2.0),
        quality_targets=loss_hyp.get("quality_targets", True),
        quality_power=loss_hyp.get("quality_power", 1.0),
        quality_floor=loss_hyp.get("quality_floor", 0.05),
        quality_start_epoch=loss_hyp.get("quality_start_epoch", 25),
        quality_warmup_epochs=loss_hyp.get("quality_warmup_epochs", 50),
        qfl_beta=loss_hyp.get("qfl_beta", 2.0),
        hard_negative_topk=loss_hyp.get("hard_negative_topk", 256),
        hard_negative_weight=loss_hyp.get("hard_negative_weight", 0.10),
        hard_negative_min_score=loss_hyp.get("hard_negative_min_score", 0.15),
        hard_negative_start_epoch=loss_hyp.get("hard_negative_start_epoch", 8),
        hard_negative_warmup_epochs=loss_hyp.get("hard_negative_warmup_epochs", 32),
        o2o_weight=loss_hyp.get("o2o_weight", 1.0),
        o2m_weight=loss_hyp.get("o2m_weight", 1.0),
        consistency_weight=loss_hyp.get("consistency_weight", 0.35),
        consistency_start_epoch=loss_hyp.get("consistency_start_epoch", 10),
        consistency_warmup_epochs=loss_hyp.get("consistency_warmup_epochs", 40),
        consistency_cls_weight=loss_hyp.get("consistency_cls_weight", 0.25),
        consistency_box_weight=loss_hyp.get("consistency_box_weight", 0.75),
        prog_loss=loss_hyp.get("prog_loss", True),
        prog_start_epoch=loss_hyp.get("prog_start_epoch", 1),
        prog_warmup_epochs=loss_hyp.get("prog_warmup_epochs", 60),
        cls_start_factor=loss_hyp.get("cls_start_factor", 1.50),
        box_start_factor=loss_hyp.get("box_start_factor", 1.30),
        dfl_start_factor=loss_hyp.get("dfl_start_factor", 0.85),
        dfl_mean_over_coords=loss_hyp.get("dfl_mean_over_coords", True),
        o2m_full_epochs=loss_hyp.get("o2m_full_epochs", None),
        o2m_sparse_epochs=loss_hyp.get("o2m_sparse_epochs", None),
        o2m_every_n_steps=loss_hyp.get("o2m_every_n_steps", 1),
        o2m_disable_epoch=loss_hyp.get("o2m_disable_epoch", None),
        exact_o2o_warmup_epochs=matcher_hyp.get("exact_o2o_warmup_epochs", None),
        exact_o2o_period=matcher_hyp.get("exact_o2o_period", 1),
        exact_o2o_period_end_epoch=matcher_hyp.get("exact_o2o_period_end_epoch", None),
        quality=loss_hyp.get("quality", 1.0),
        quality_loss_start_epoch=loss_hyp.get("quality_loss_start_epoch", loss_hyp.get("quality_head_start_epoch", 1)),
        quality_loss_warmup_epochs=loss_hyp.get("quality_loss_warmup_epochs", loss_hyp.get("quality_head_warmup_epochs", 25)),
    )
