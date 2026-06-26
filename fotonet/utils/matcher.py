"""
FOTONET Matchers — fully vectorized, GPU-resident.

FastO2OAssigner (replaces HungarianMatcher):
  - Builds cost matrix on GPU
  - Resolves conflicts via one sorted transfer to CPU (no per-GPU-element .item() calls)
  - M is typically 1-20, so the tiny Python greedy loop terminates in ~M steps

TaskAlignedAssigner (O2M):
  - Fully vectorized - zero Python loops over M
  - Uses real feat_shapes from head output — no back-inference formula
"""
import math
import torch
import torch.nn as nn
from fotonet.utils.boxes import box_giou, box_iou, xywh_to_xyxy

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


class FastO2OAssigner(nn.Module):
    """
    Fast One-to-One assigner. Fully vectorized except for the tiny greedy
    deduplication step which runs on CPU lists (no GPU syncs inside the loop).
    """
    def __init__(self, cost_class=2.0, cost_bbox=2.0, cost_giou=1.0, exact=True):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox  = cost_bbox
        self.cost_giou  = cost_giou
        self.exact      = bool(exact)

    @torch.no_grad()
    def forward(self, outputs, targets, feat_shapes=None):
        device = outputs["pred_logits"].device
        bs     = outputs["pred_logits"].shape[0]

        all_costs = []
        batch_m = []
        batch_n = []

        for i in range(bs):
            pred_logits = outputs["pred_logits"][i].float()
            pred_boxes  = outputs["pred_boxes"][i].float()
            tgt_boxes   = targets[i]["boxes"].float()
            tgt_labels  = targets[i]["labels"]
            M = tgt_boxes.shape[0]
            N = pred_logits.shape[0]
            batch_m.append(M)
            batch_n.append(N)

            if M == 0:
                all_costs.append(None)
                continue

            probs    = pred_logits.sigmoid()
            cost_cls = -probs[:, tgt_labels]
            cost_l1  = torch.cdist(pred_boxes, tgt_boxes, p=1)
            cost_iou = -box_giou(xywh_to_xyxy(pred_boxes), xywh_to_xyxy(tgt_boxes))

            C = (self.cost_class * cost_cls + self.cost_bbox * cost_l1 + self.cost_giou * cost_iou)
            C = torch.nan_to_num(C, nan=1e6, posinf=1e6, neginf=-1e6)
            all_costs.append(C)

        flat_costs_cpu = []
        for i in range(bs):
            if all_costs[i] is not None and not (self.exact and linear_sum_assignment is not None):
                flat_costs_cpu.append(all_costs[i].view(-1).argsort().cpu().numpy())
            else:
                flat_costs_cpu.append(None)

        indices = []
        for i in range(bs):
            M, N = batch_m[i], batch_n[i]
            if M == 0:
                indices.append((torch.empty(0, dtype=torch.long, device=device),
                                torch.empty(0, dtype=torch.long, device=device)))
                continue

            if self.exact and linear_sum_assignment is not None:
                rows, cols = linear_sum_assignment(all_costs[i].detach().cpu().numpy())
                p_assign = [0] * M
                for row, col in zip(rows, cols):
                    p_assign[int(col)] = int(row)
                indices.append((torch.tensor(p_assign, dtype=torch.long, device=device),
                                torch.arange(M, dtype=torch.long, device=device)))
                continue

            flat_order = flat_costs_cpu[i]
            used_anc = [False] * N
            used_gt  = [False] * M
            p_assign = [0] * M
            done = 0
            for idx in flat_order:
                a = int(idx) // M
                g = int(idx) % M
                if not used_anc[a] and not used_gt[g]:
                    p_assign[g] = a
                    used_anc[a] = True
                    used_gt[g]  = True
                    done += 1
                    if done == M: break

            indices.append((torch.tensor(p_assign, dtype=torch.long, device=device),
                            torch.arange(M, dtype=torch.long, device=device)))

        return indices


class TaskAlignedAssigner(nn.Module):
    """
    Task-Aligned One-to-Many assigner. Fully vectorized — zero Python loops
    over the number of GT objects (M). Accepts real feat_shapes from the head.
    """
    def __init__(self, topk=10, alpha=1.0, beta=2.0, min_small_assign=4,
                 small_obj_px=8.0, strides=(8, 16, 32)):
        super().__init__()
        self.topk   = topk
        self.alpha  = alpha
        self.beta   = beta
        self.min_small_assign = int(min_small_assign)
        self.small_obj_px = float(small_obj_px)
        self._anchor_cache = {}
        self.strides = [int(s) for s in strides]

    def _get_anchors(self, device, feat_shapes):
        """
        Build normalized anchor centers [total_anchors, 2] in [0, 1] coords.
        Each anchor is the center of its grid cell, normalized by image dimensions.
        feat_shapes: list or tensor of (H, W) per stride level.
        """
        # Ensure hashable key for cache
        if torch.is_tensor(feat_shapes):
            fs_key = tuple(map(tuple, feat_shapes.tolist()))
        else:
            fs_key = tuple(feat_shapes)

        key = (str(device), fs_key, tuple(self.strides))
        if key not in self._anchor_cache:
            parts = []
            for i in range(len(feat_shapes)):
                ny, nx = int(feat_shapes[i][0]), int(feat_shapes[i][1])
                stride = self.strides[i]
                # Image size = stride * grid_size
                img_h = float(stride * ny)
                img_w = float(stride * nx)
                yv, xv = torch.meshgrid(
                    torch.arange(ny, device=device, dtype=torch.float32),
                    torch.arange(nx, device=device, dtype=torch.float32),
                    indexing='ij'
                )
                # Anchor centers at cell midpoints, normalized to [0,1] image coords
                g = torch.stack((
                    (xv + 0.5) * stride / img_w,
                    (yv + 0.5) * stride / img_h,
                ), 2).view(-1, 2)
                parts.append(g)
            self._anchor_cache[key] = torch.cat(parts, 0)
        return self._anchor_cache[key]

    def _infer_feat_shapes(self, N):
        """
        Fallback: infer feat_shapes from total anchor count N.
        Works for square images: N = (S/8)^2 + (S/16)^2 + (S/32)^2
              = S^2*(1/64 + 1/256 + 1/1024) = S^2 * 21/1024
        """
        denom = sum(1.0 / float(s * s) for s in self.strides)
        S = int(round(math.sqrt(N / max(denom, 1e-12))))
        return [(S // s, S // s) for s in self.strides]

    @torch.no_grad()
    def forward(self, outputs, targets, feat_shapes=None):
        device = outputs["pred_logits"].device
        bs, N  = outputs["pred_logits"].shape[:2]

        # Use real feat_shapes from the head when available; fall back for robustness
        if feat_shapes is None:
            feat_shapes = self._infer_feat_shapes(N)

        anchors = self._get_anchors(device, feat_shapes)
        indices = []

        for i in range(bs):
            pred_logits = outputs["pred_logits"][i].float().sigmoid()
            pred_boxes  = outputs["pred_boxes"][i].float()
            tgt_boxes   = targets[i]["boxes"].float()
            tgt_labels  = targets[i]["labels"]
            M = tgt_boxes.shape[0]

            if M == 0:
                indices.append((
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                ))
                continue

            tgt_xyxy = xywh_to_xyxy(tgt_boxes)
            anc_x = anchors[:, 0]
            anc_y = anchors[:, 1]
            in_x  = (anc_x[:, None] > tgt_xyxy[None, :, 0]) & \
                    (anc_x[:, None] < tgt_xyxy[None, :, 2])
            in_y  = (anc_y[:, None] > tgt_xyxy[None, :, 1]) & \
                    (anc_y[:, None] < tgt_xyxy[None, :, 3])
            in_gt = in_x & in_y

            # Robustness: For extremely small objects that don't cover any anchor centers,
            # enforce a minimum assignment radius (e.g., ~2.5 grid cells at largest feature map)
            center_dists = torch.cdist(anchors, tgt_boxes[:, :2])
            in_center    = center_dists < 0.025
            in_gt       |= in_center

            # STAL: Small-Target-Aware Label Assignment
            tgt_wh = tgt_boxes[:, 2:4]
            tgt_areas = tgt_wh[:, 0] * tgt_wh[:, 1]
            size_penalty = torch.clamp(1.0 / (torch.sqrt(tgt_areas) * 10.0 + 0.1), min=1.0, max=3.0)

            cls_score = pred_logits[:, tgt_labels]
            iou       = box_iou(xywh_to_xyxy(pred_boxes), tgt_xyxy).clamp_min(0.0)
            align     = cls_score.pow(self.alpha) * iou.pow(self.beta) * in_gt * size_penalty[None, :]

            # Early training can produce zero IoU for every candidate of a GT. In that
            # state task-aligned assignment becomes almost random and O2M supervision
            # thins out badly. Fall back to a center prior until decoded boxes improve.
            empty_gt = align.max(dim=0).values <= 0
            if empty_gt.any():
                center_prior = (1.0 - center_dists / 0.05).clamp_min(0.0)
                align[:, empty_gt] = (
                    center_prior[:, empty_gt]
                    * in_gt[:, empty_gt]
                    * size_penalty[None, empty_gt]
                )

            topk         = min(self.topk, N)
            _, topk_idx  = align.topk(topk, dim=0)
            mask         = torch.zeros(N, M, dtype=torch.bool, device=device)
            mask.scatter_(0, topk_idx, True)
            mask        &= in_gt

            _, best_gt  = align.max(dim=1)

            gt_range         = torch.arange(M, device=device)
            anchor_best_mask = (best_gt[:, None] == gt_range[None, :])
            pos_mask         = mask & anchor_best_mask

            pair_anc, pair_gt = pos_mask.nonzero(as_tuple=True)

            # Fallback: nearest anchor for any unmatched GT
            gt_assigned = torch.zeros(M, dtype=torch.bool, device=device)
            if pair_gt.numel() > 0:
                gt_assigned.scatter_(0, pair_gt, True)
            missing = (~gt_assigned).nonzero(as_tuple=True)[0]
            if missing.numel() > 0:
                miss_centers = tgt_boxes[missing, :2]
                dists        = torch.cdist(anchors, miss_centers)
                nearest      = dists.argmin(dim=0)
                pair_anc = torch.cat([pair_anc, nearest])
                pair_gt  = torch.cat([pair_gt,  missing])

            # STAL: tiny objects should receive several anchors, not just one.
            if self.min_small_assign > 1:
                img_side = float(max(int(feat_shapes[j][0]) * self.strides[j] for j in range(len(feat_shapes))))
                small_norm = self.small_obj_px / max(img_side, 1.0)
                small_gt = (torch.sqrt(tgt_areas.clamp_min(0.0)) < small_norm).nonzero(as_tuple=True)[0]
                if small_gt.numel() > 0:
                    add_anc = []
                    add_gt = []
                    for gt_idx in small_gt.detach().cpu().tolist():
                        current = pair_anc[pair_gt == gt_idx]
                        need = self.min_small_assign - int(current.numel())
                        if need <= 0:
                            continue
                        existing = set(current.detach().cpu().tolist())
                        order = center_dists[:, gt_idx].argsort().detach().cpu().tolist()
                        chosen = []
                        for anc_idx in order:
                            if anc_idx not in existing:
                                chosen.append(anc_idx)
                                existing.add(anc_idx)
                                if len(chosen) == need:
                                    break
                        if chosen:
                            add_anc.append(torch.tensor(chosen, dtype=torch.long, device=device))
                            add_gt.append(torch.full((len(chosen),), gt_idx, dtype=torch.long, device=device))
                    if add_anc:
                        pair_anc = torch.cat([pair_anc, *add_anc])
                        pair_gt = torch.cat([pair_gt, *add_gt])

            indices.append((pair_anc, pair_gt))

        return indices


HungarianMatcher = FastO2OAssigner
