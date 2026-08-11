"""
FOTONET Loss Functions.
DualLoss: O2O + O2M assignment with Focal + DFL + CIoU.
Uses Distribution Focal Loss (DFL) for regression instead of L1.
"""
import contextlib
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from fotonet.utils.boxes import xywh_to_xyxy


@dataclass
class BranchLossResult:
    cls_loss: torch.Tensor
    reg_loss: torch.Tensor
    ciou_loss: torch.Tensor
    reg_kind: str
    dfl_loss: torch.Tensor
    direct_ltrb_loss: torch.Tensor

    def __iter__(self):
        yield self.cls_loss
        yield self.reg_loss
        yield self.ciou_loss


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
    imgsz: scalar image size or ``[width, height]`` input size
    Returns: [M, 4] LTRB in grid-cell units, clamped to [0, reg_max-1]
    """
    stride = stride.to(device=anchor_points.device, dtype=anchor_points.dtype).view(-1)
    if torch.is_tensor(imgsz):
        image_size = imgsz.to(device=anchor_points.device, dtype=anchor_points.dtype).reshape(-1)
    else:
        image_size = torch.as_tensor(imgsz, device=anchor_points.device, dtype=anchor_points.dtype).reshape(-1)
    if image_size.numel() == 1:
        image_w = image_h = image_size[0]
    elif image_size.numel() == 2:
        # The model forwards spatial shape as [width, height].  Applying a
        # two-element tensor directly to [num_matches] coordinates used to
        # broadcast each GT into two fake targets (or crash outright).
        image_w, image_h = image_size[0], image_size[1]
    else:
        raise ValueError("imgsz must be a scalar or a [width, height] pair")
    cx, cy, w, h = gt_boxes_xywh.unbind(-1)
    # Convert normalized coords to grid-cell units
    x1 = (cx - w * 0.5) * image_w / stride
    y1 = (cy - h * 0.5) * image_h / stride
    x2 = (cx + w * 0.5) * image_w / stride
    y2 = (cy + h * 0.5) * image_h / stride

    l = anchor_points[:, 0] - x1
    t = anchor_points[:, 1] - y1
    r = x2 - anchor_points[:, 0]
    b = y2 - anchor_points[:, 1]
    dist = torch.stack([l, t, r, b], -1).clamp_min(0)
    if reg_max is not None and reg_max > 1:
        dist = dist.clamp(0, reg_max - 1 - 0.01)
    return dist


def _hard_negative_topk_sum(losses, candidate_mask, topk):
    """Sum up to ``topk`` candidate losses without materializing a dynamic gather.

    Focal losses are non-negative, so a negative finite sentinel cannot be a
    genuine candidate.  Replacing that sentinel after fixed-size ``topk``
    retains the exact gather-and-topk value while keeping CUDA work tensorized.
    """
    flat_losses = losses.reshape(-1)
    k = min(int(topk), int(flat_losses.numel()))
    if k <= 0:
        return flat_losses.new_zeros(())
    sentinel = flat_losses.new_tensor(-1.0)
    selected = flat_losses.masked_fill(~candidate_mask.reshape(-1), sentinel).topk(k).values
    return selected.masked_fill(selected.eq(sentinel), 0.0).sum()


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

    # ``binary_cross_entropy_with_logits`` is numerically stable for extreme
    # finite logits. Do not clamp here: a clamp would give an extreme false
    # positive a large loss but a zero corrective gradient.

    tgt_onehot = torch.zeros_like(pred_logits)
    all_p_idx = []
    all_b_idx = []
    all_tgt_labels = []
    all_tgt_boxes = []
    for i, (p_idx, t_idx) in enumerate(indices):
        if len(p_idx) > 0:
            p_idx = p_idx.to(device=device, dtype=torch.long, non_blocking=True)
            target_labels = targets[i]["labels"]
            target_boxes = targets[i]["boxes"]
            local_t_idx = t_idx.to(device=target_labels.device, dtype=torch.long, non_blocking=True)
            if target_labels.ndim != 1 or target_boxes.ndim != 2 or target_boxes.shape[-1] != 4:
                raise ValueError("Each loss target must provide labels [N] and boxes [N, 4]")
            if target_labels.shape[0] != target_boxes.shape[0]:
                raise ValueError("Target labels and boxes must have matching lengths")
            all_p_idx.append(p_idx)
            all_b_idx.append(torch.full((p_idx.numel(),), i, dtype=torch.long, device=device))
            all_tgt_labels.append(target_labels[local_t_idx].to(device=device, dtype=torch.long, non_blocking=True))
            all_tgt_boxes.append(target_boxes[local_t_idx].to(device=device, dtype=pred_boxes.dtype, non_blocking=True))

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
            loss_cls = loss_cls + hard_negative_weight * _hard_negative_topk_sum(
                focal_f32, hard_mask, hard_negative_topk
            )
        zero = torch.tensor(0.0, device=device)
        return BranchLossResult(
            cls_loss=loss_cls,
            reg_loss=zero,
            ciou_loss=zero,
            reg_kind="direct_ltrb" if reg_max <= 1 else "dfl",
            dfl_loss=zero,
            direct_ltrb_loss=zero,
        )

    all_p_idx = torch.cat(all_p_idx)
    all_b_idx = torch.cat(all_b_idx)
    all_tgt_labels = torch.cat(all_tgt_labels)
    all_tgt_boxes = torch.cat(all_tgt_boxes)

    tgt_onehot[all_b_idx, all_p_idx, all_tgt_labels] = 1.0

    # --- CIoU Loss (on decoded boxes) ---
    all_pred_boxes = pred_boxes[all_b_idx, all_p_idx]
    p_xyxy = xywh_to_xyxy(all_pred_boxes)
    t_xyxy = xywh_to_xyxy(all_tgt_boxes)
    matched_ciou = bbox_ciou(p_xyxy, t_xyxy)
    loss_ciou = (1 - matched_ciou).sum()

    # --- DFL Loss (on raw distributions) ---
    matched_dist = pred_dist[all_b_idx, all_p_idx]  # [M, 4*reg_max]
    matched_anchors = anchor_points.to(device=device, dtype=pred_boxes.dtype)[all_p_idx]  # [M, 2]
    matched_strides = stride_tensor.to(device=device, dtype=pred_boxes.dtype)[all_p_idx]  # [M, 1]

    if reg_max <= 1:
        gt_ltrb = _bbox2dist(matched_anchors, all_tgt_boxes, matched_strides, imgsz, reg_max=None)
        pred_ltrb = F.softplus(matched_dist.view(-1, 4))
        reg_sum = F.smooth_l1_loss(pred_ltrb, gt_ltrb, reduction="sum")
        if dfl_mean_over_coords:
            reg_sum = reg_sum / 4.0
        dfl_sum = reg_sum.new_tensor(0.0)
        direct_ltrb_sum = reg_sum
        reg_kind = "direct_ltrb"
    else:
        gt_ltrb = _bbox2dist(matched_anchors, all_tgt_boxes, matched_strides, imgsz, reg_max)
        matched_dist = matched_dist.view(-1, 4, reg_max)  # [M, 4, reg_max]

        reg_sum = torch.tensor(0.0, device=device)
        for j in range(4):
            reg_sum = reg_sum + _dfl_loss(matched_dist[:, j], gt_ltrb[:, j], reg_max).sum()
        if dfl_mean_over_coords:
            reg_sum = reg_sum / 4.0
        dfl_sum = reg_sum
        direct_ltrb_sum = reg_sum.new_tensor(0.0)
        reg_kind = "dfl"

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
        w = 1.0 + (class_weights.to(device=device, dtype=tgt_onehot.dtype) - 1.0) * tgt_onehot
        loss_cls = (w * focal_f32).sum()
    else:
        loss_cls = focal_f32.sum()

    if hard_negative_topk > 0 and hard_negative_weight > 0:
        neg_mask = ~pos_mask
        hard_candidate = neg_mask & (p.detach() > hard_negative_min_score)
        loss_cls = loss_cls + hard_negative_weight * _hard_negative_topk_sum(
            focal_f32, hard_candidate, hard_negative_topk
        )

    return BranchLossResult(
        cls_loss=loss_cls,
        reg_loss=reg_sum,
        ciou_loss=loss_ciou,
        reg_kind=reg_kind,
        dfl_loss=dfl_sum,
        direct_ltrb_loss=direct_ltrb_sum,
    )


def calc_quality_loss(pred_quality, pred_boxes, targets, indices, beta=2.0):
    """Positive-normalized quality focal loss.

    Quality is a localization-ranking signal for already-classified anchors;
    it is not a second dense objectness classifier.  Summing ordinary BCE over
    every dense background query made the background gradient dominate a
    perfect positive by orders of magnitude.  We train all assigned positives
    (including zero-IoU ones) and normalize by their target quality mass.
    """
    if pred_quality is None:
        return None
    bs, n_queries, _ = pred_quality.shape
    device = pred_quality.device
    targets_quality = torch.zeros_like(pred_quality)
    positive_mask = torch.zeros_like(pred_quality, dtype=torch.bool)

    all_p_idx = []
    all_b_idx = []
    all_tgt_boxes = []
    for batch_idx, (p_idx, t_idx) in enumerate(indices):
        if len(p_idx) == 0:
            continue
        p_idx = p_idx.to(device=device, dtype=torch.long, non_blocking=True)
        target_boxes = targets[batch_idx]["boxes"]
        local_t_idx = t_idx.to(device=target_boxes.device, dtype=torch.long, non_blocking=True)
        if target_boxes.ndim != 2 or target_boxes.shape[-1] != 4:
            raise ValueError("Quality-loss targets must provide boxes with shape [N, 4]")
        all_p_idx.append(p_idx)
        all_b_idx.append(torch.full((p_idx.numel(),), batch_idx, dtype=torch.long, device=device))
        all_tgt_boxes.append(target_boxes[local_t_idx])

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
        positive_mask[all_b_idx, all_p_idx, 0] = True

    logits = pred_quality.float()
    targets_f = targets_quality.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets_f, reduction="none")
    focal = (targets_f - logits.sigmoid()).abs().pow(float(beta))
    # Dense masking has the same positive-only loss and normalization as
    # boolean indexing, without materializing a data-dependent CUDA gather.
    positive_weight = positive_mask.to(dtype=targets_f.dtype)
    normalizer = (targets_f * positive_weight).sum().clamp_min(1.0)
    return ((bce * focal) * positive_weight).sum() / normalizer


def calc_consistency_loss(
    outputs,
    targets,
    indices_o2o,
    indices_o2m=None,
    cls_weight=0.25,
    box_weight=0.75,
):
    """
    O2O <- O2M consistency on common GT identities.
    The O2M branch sees richer supervision, so its detached predictions act as a
    teacher for the NMS-free O2O branch once training has warmed up.
    """
    device = outputs["pred_logits_o2o"].device
    bs = outputs["pred_logits_o2o"].shape[0]
    # Anchor indices are branch-local: an O2O positive at anchor 17 is not
    # necessarily the O2M positive for the same object.  Without O2M indices
    # there is no safe correspondence, so omit this auxiliary loss.
    if indices_o2m is None:
        return torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)

    student_logits_parts, teacher_logits_parts = [], []
    student_boxes_parts, teacher_boxes_parts = [], []
    label_parts = []
    for i, ((o2o_p, o2o_t), (o2m_p, o2m_t)) in enumerate(zip(indices_o2o, indices_o2m)):
        o2o_p = o2o_p.to(device=device, dtype=torch.long, non_blocking=True)
        o2o_t = o2o_t.to(device=device, dtype=torch.long, non_blocking=True)
        o2m_p = o2m_p.to(device=device, dtype=torch.long, non_blocking=True)
        o2m_t = o2m_t.to(device=device, dtype=torch.long, non_blocking=True)
        if o2o_p.numel() == 0 or o2m_p.numel() == 0:
            continue

        labels = targets[i]["labels"].to(device=device, dtype=torch.long)
        target_boxes = targets[i]["boxes"].to(
            device=device,
            dtype=outputs["pred_boxes_o2m"].dtype,
        )
        o2o_labels = labels[o2o_t]
        o2o_targets = target_boxes[o2o_t]
        teacher_logits = outputs["pred_logits_o2m"][i, o2m_p].detach()
        teacher_boxes = outputs["pred_boxes_o2m"][i, o2m_p].detach()

        # Pick the strongest O2M prediction for each O2O GT.  Matching by GT
        # index, rather than identical anchor index, preserves branch-local
        # assignment semantics.  CIoU helps prefer a localized teacher where
        # multiple O2M positives belong to that GT.
        common = o2o_t[:, None].eq(o2m_t[None, :])
        n_o2o, n_o2m = common.shape
        teacher_pair_boxes = teacher_boxes.unsqueeze(0).expand(n_o2o, -1, -1).reshape(-1, 4)
        target_pair_boxes = o2o_targets.unsqueeze(1).expand(-1, n_o2m, -1).reshape(-1, 4)
        pair_iou = bbox_ciou(
            xywh_to_xyxy(teacher_pair_boxes),
            xywh_to_xyxy(target_pair_boxes),
        ).reshape(n_o2o, n_o2m).clamp(0.0, 1.0)
        class_scores = teacher_logits[:, o2o_labels].transpose(0, 1).sigmoid()
        teacher_rank = class_scores * (0.5 + 0.5 * pair_iou)
        valid = common.any(dim=1)
        chosen = teacher_rank.masked_fill(~common, float("-inf")).argmax(dim=1)
        chosen = chosen[valid]

        student_logits_parts.append(outputs["pred_logits_o2o"][i, o2o_p[valid]])
        teacher_logits_parts.append(teacher_logits[chosen])
        student_boxes_parts.append(outputs["pred_boxes_o2o"][i, o2o_p[valid]])
        teacher_boxes_parts.append(teacher_boxes[chosen])
        label_parts.append(o2o_labels[valid])

    if not student_logits_parts:
        return torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)

    o2o_logits = torch.cat(student_logits_parts)
    o2m_logits = torch.cat(teacher_logits_parts)
    o2o_boxes = torch.cat(student_boxes_parts)
    o2m_boxes = torch.cat(teacher_boxes_parts)
    labels = torch.cat(label_parts)
    row = torch.arange(labels.numel(), device=device)
    teacher_scores = torch.sigmoid(o2m_logits[row, labels]).clamp(0.01, 0.99)
    cls_loss = F.binary_cross_entropy_with_logits(
        o2o_logits[row, labels], teacher_scores, reduction="sum"
    )
    box_loss = (1.0 - bbox_ciou(xywh_to_xyxy(o2o_boxes), xywh_to_xyxy(o2m_boxes))).sum()
    count = max(int(labels.numel()), 1)
    return cls_weight * cls_loss + box_weight * box_loss, torch.as_tensor(float(count), device=device)


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
                 dual_branch_curriculum=False, o2o_start_weight=None,
                 o2m_start_weight=None, o2o_end_weight=None, o2m_end_weight=None,
                 dual_branch_start_epoch=1, dual_branch_warmup_epochs=100,
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
        self.dual_branch_curriculum = bool(dual_branch_curriculum)
        self.o2o_start_weight = float(self.o2o_weight if o2o_start_weight is None else o2o_start_weight)
        self.o2m_start_weight = float(self.o2m_weight if o2m_start_weight is None else o2m_start_weight)
        self.o2o_end_weight = float(self.o2o_weight if o2o_end_weight is None else o2o_end_weight)
        self.o2m_end_weight = float(self.o2m_weight if o2m_end_weight is None else o2m_end_weight)
        self.dual_branch_start_epoch = int(dual_branch_start_epoch)
        self.dual_branch_warmup_epochs = int(dual_branch_warmup_epochs)
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

    @staticmethod
    def _epoch_span_mix(current_epoch, start_epoch, warmup_epochs):
        """1-indexed inclusive ramp: first active epoch is 0.0, final warmup epoch is 1.0."""
        epoch = float(current_epoch) + 1.0
        start = max(float(start_epoch), 1.0)
        warmup = max(float(warmup_epochs), 1.0)
        if epoch < start:
            return 0.0
        if warmup <= 1.0:
            return 1.0
        return max(0.0, min(1.0, (epoch - start) / (warmup - 1.0)))

    def branch_weights(self, current_epoch=0):
        if not self.dual_branch_curriculum:
            return self.o2o_weight, self.o2m_weight
        mix = self._epoch_span_mix(
            current_epoch,
            self.dual_branch_start_epoch,
            self.dual_branch_warmup_epochs,
        )
        o2o = self.o2o_start_weight + (self.o2o_end_weight - self.o2o_start_weight) * mix
        o2m = self.o2m_start_weight + (self.o2m_end_weight - self.o2m_start_weight) * mix
        return float(o2o), float(o2m)

    def use_o2m(self, current_epoch=0, global_step=0, active_o2m_weight=None):
        """Return whether this step should compute the training-only O2M branch."""
        epoch = int(current_epoch) + 1
        o2m_weight = self.o2m_weight if active_o2m_weight is None else float(active_o2m_weight)
        if self.o2m_disable_epoch is not None and epoch > self.o2m_disable_epoch:
            return False
        if self.o2m_full_epochs is None or epoch <= self.o2m_full_epochs:
            return o2m_weight > 0
        if self.o2m_sparse_epochs is not None and epoch > self.o2m_sparse_epochs:
            return False
        return o2m_weight > 0 and (int(global_step) % self.o2m_every_n_steps == 0)

    def _use_exact_o2o(self, current_epoch=0, global_step=0):
        # Exact matching is an explicit opt-in. A schedule must not silently
        # re-enable costly CPU/SciPy assignment after the caller chose greedy.
        if not bool(getattr(self.matcher_o2o, "exact", False)):
            return False
        epoch = int(current_epoch) + 1
        if self.exact_o2o_warmup_epochs is None:
            return True
        if epoch <= self.exact_o2o_warmup_epochs:
            return True
        if self.exact_o2o_period_end_epoch is not None and epoch > self.exact_o2o_period_end_epoch:
            return False
        return int(global_step) % self.exact_o2o_period == 0

    def forward(
        self,
        outputs,
        targets,
        current_epoch=0,
        max_epochs=1,
        global_step=0,
        profiler=None,
    ):
        def profile_stage(name):
            if profiler is None:
                return contextlib.nullcontext()
            return profiler.stage(name)

        feat_shapes    = outputs.get("feat_shapes", None)
        anchor_points  = outputs["anchor_points"]   # [total_N, 2]
        stride_tensor  = outputs["stride_tensor"]   # [total_N, 1]
        imgsz          = outputs["imgsz"]
        active_o2o_weight, scheduled_o2m_weight = self.branch_weights(current_epoch)

        o2o_in = {
            "pred_logits": outputs["pred_logits_o2o"],
            "pred_boxes":  outputs["pred_boxes_o2o"],
        }
        # The head already knows exact per-query grids and the real input
        # shape. Forward that metadata into both task-aligned matchers instead
        # of reconstructing normalized anchors from feature-map shapes.
        for key in ("anchor_points", "stride_tensor", "imgsz"):
            if key in outputs:
                o2o_in[key] = outputs[key]
        shared_geometry = None
        prepare_geometry = getattr(self.matcher_o2o, "prepare_geometry", None)
        if callable(prepare_geometry):
            with profile_stage("matcher_geometry"):
                shared_geometry = prepare_geometry(
                    o2o_in,
                    targets,
                    feat_shapes,
                )
        old_exact = getattr(self.matcher_o2o, "exact", None)
        if old_exact is not None:
            self.matcher_o2o.exact = self._use_exact_o2o(current_epoch, global_step)
        try:
            with profile_stage("matcher_o2o"):
                idx_o2o = self.matcher_o2o(
                    o2o_in,
                    targets,
                    feat_shapes=feat_shapes,
                    **(
                        {"geometry": shared_geometry}
                        if shared_geometry is not None
                        else {}
                    ),
                )
        finally:
            if old_exact is not None:
                self.matcher_o2o.exact = old_exact

        has_o2m_outputs = all(k in outputs for k in ("pred_logits_o2m", "pred_boxes_o2m", "pred_dist_o2m"))
        use_o2m = has_o2m_outputs and self.use_o2m(current_epoch, global_step, scheduled_o2m_weight)
        if use_o2m:
            o2m_in = {
                "pred_logits": outputs["pred_logits_o2m"],
                "pred_boxes":  outputs["pred_boxes_o2m"],
            }
            for key in ("anchor_points", "stride_tensor", "imgsz"):
                if key in outputs:
                    o2m_in[key] = outputs[key]
            with profile_stage("matcher_o2m"):
                idx_o2m = self.matcher_o2m(
                    o2m_in,
                    targets,
                    feat_shapes=feat_shapes,
                    **(
                        {"geometry": shared_geometry}
                        if shared_geometry is not None
                        else {}
                    ),
                )
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
        with profile_stage("loss_o2o"):
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
            with profile_stage("loss_o2m"):
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
            active_o2m_weight = scheduled_o2m_weight
        else:
            zero = cls_o2o.new_tensor(0.0)
            cls_o2m = dfl_o2m = iou_o2m = zero
            active_o2m_weight = 0.0

        q_o2o = calc_quality_loss(
            outputs.get("pred_quality_o2o"),
            o2o_in["pred_boxes"],
            targets,
            idx_o2o,
            beta=self.qfl_beta,
        )
        if q_o2o is None:
            q_o2o = cls_o2o.new_tensor(0.0)

        if use_o2m:
            q_o2m = calc_quality_loss(
                outputs.get("pred_quality_o2m"),
                o2m_in["pred_boxes"],
                targets,
                idx_o2m,
                beta=self.qfl_beta,
            )
            if q_o2m is None:
                q_o2m = cls_o2o.new_tensor(0.0)
        else:
            q_o2m = cls_o2o.new_tensor(0.0)

        # Normalize branches independently, then blend O2O/O2M explicitly.
        loss_cls  = active_o2o_weight * (cls_o2o / n_o2o) + active_o2m_weight * (cls_o2m / n_o2m)
        loss_ciou = active_o2o_weight * (iou_o2o / n_o2o) + active_o2m_weight * (iou_o2m / n_o2m)
        loss_reg  = active_o2o_weight * (dfl_o2o / n_o2o) + active_o2m_weight * (dfl_o2m / n_o2m)
        loss_quality = active_o2o_weight * q_o2o + active_o2m_weight * q_o2m
        if use_o2m and active_w_cons > 0:
            with profile_stage("consistency_loss"):
                cons_raw, n_cons = calc_consistency_loss(
                    outputs,
                    targets,
                    idx_o2o,
                    idx_o2m,
                    cls_weight=self.consistency_cls_weight,
                    box_weight=self.consistency_box_weight,
                )
            loss_cons = cons_raw / n_cons
        else:
            loss_cons = loss_cls.new_tensor(0.0)

        total = (
            active_w_cls * loss_cls
            + active_w_box * loss_ciou
            + active_w_dfl * loss_reg
            + active_w_cons * loss_cons
            + active_w_quality * loss_quality
        )
        reg_kind = "direct_ltrb" if self.reg_max <= 1 else "dfl"
        zero_reg = loss_reg.new_tensor(0.0)
        loss_dfl_component = loss_reg if reg_kind == "dfl" else zero_reg
        loss_direct_component = loss_reg if reg_kind == "direct_ltrb" else zero_reg

        return {
            "loss":              total,
            "loss_cls":          loss_cls.detach(),
            "loss_box":          loss_reg.detach(),
            "loss_reg":          loss_reg.detach(),
            "loss_dfl":          loss_dfl_component.detach(),
            "loss_direct_ltrb":   loss_direct_component.detach(),
            "loss_reg_per_coord": (loss_reg if self.dfl_mean_over_coords else loss_reg / 4.0).detach(),
            "loss_dfl_per_coord": (loss_dfl_component if self.dfl_mean_over_coords else loss_dfl_component / 4.0).detach(),
            "loss_ciou":         loss_ciou.detach(),
            "loss_cls_weighted":  (active_w_cls * loss_cls).detach(),
            "loss_reg_weighted":  (active_w_dfl * loss_reg).detach(),
            "loss_dfl_weighted":  (active_w_dfl * loss_dfl_component).detach(),
            "loss_direct_ltrb_weighted": (active_w_dfl * loss_direct_component).detach(),
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
            "o2o_weight":         torch.as_tensor(active_o2o_weight, device=loss_cls.device),
            "o2m_weight":         torch.as_tensor(active_o2m_weight, device=loss_cls.device),
            "reg_kind":           reg_kind,
            "o2m_active":         torch.as_tensor(float(use_o2m), device=loss_cls.device),
            "exact_o2o_active":   torch.as_tensor(float(self._use_exact_o2o(current_epoch, global_step)), device=loss_cls.device),
        }


def get_loss(model, nc, class_weights=None, loss_hyp=None, matcher_hyp=None):
    from fotonet.utils.matcher import TaskAlignedAssigner, TaskAlignedO2OAssigner
    # Get reg_max from head
    reg_max = getattr(model.head, 'reg_max', 16)
    loss_hyp = loss_hyp or {}
    matcher_hyp = matcher_hyp or {}
    cw_tensor = None
    if class_weights is not None:
        device = next(model.parameters()).device
        cw_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    strides = matcher_hyp.get("strides", getattr(model.head, "strides", [8, 16, 32]))
    if matcher_hyp.get("o2o_assigner", "task_aligned") != "task_aligned":
        raise ValueError("Only the task_aligned O2O assigner is supported")
    o2o_assigner = TaskAlignedO2OAssigner(
        alpha=matcher_hyp.get("alpha", 1.0),
        beta=matcher_hyp.get("beta", 2.0),
        strides=strides,
        center_radius_cells=matcher_hyp.get("center_radius_cells", 2.5),
        center_prior_cells=matcher_hyp.get("center_prior_cells", 5.0),
        proposal_rounds=matcher_hyp.get("proposal_rounds", 4),
        exact=matcher_hyp.get("exact_o2o", False),
    )
    o2m_assigner = TaskAlignedAssigner(
        topk=matcher_hyp.get("topk", 8),
        alpha=matcher_hyp.get("alpha", 1.0),
        beta=matcher_hyp.get("beta", 2.0),
        min_small_assign=matcher_hyp.get("min_small_assign", 4),
        small_obj_px=matcher_hyp.get("small_obj_px", 8.0),
        strides=strides,
        center_radius_cells=matcher_hyp.get("center_radius_cells", 2.5),
        center_prior_cells=matcher_hyp.get("center_prior_cells", 5.0),
        proposal_rounds=matcher_hyp.get("proposal_rounds", 4),
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
        dual_branch_curriculum=loss_hyp.get("dual_branch_curriculum", False),
        o2o_start_weight=loss_hyp.get("o2o_start_weight", None),
        o2m_start_weight=loss_hyp.get("o2m_start_weight", None),
        o2o_end_weight=loss_hyp.get("o2o_end_weight", None),
        o2m_end_weight=loss_hyp.get("o2m_end_weight", None),
        dual_branch_start_epoch=loss_hyp.get("dual_branch_start_epoch", 1),
        dual_branch_warmup_epochs=loss_hyp.get("dual_branch_warmup_epochs", 100),
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
