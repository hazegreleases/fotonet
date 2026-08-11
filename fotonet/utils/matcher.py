"""
FOTONET matchers — GPU score construction with compact CPU conflict
resolution only when a one-to-one collision requires it.

TaskAlignedAssigner (O2M):
  - Fully vectorized - zero Python loops over M
  - Uses real feat_shapes from head output — no back-inference formula

TaskAlignedO2OAssigner (NMS-free branch):
  - Uses an on-device no-conflict fast path
  - Resolves collisions from O(M * min(N, M)) candidate data, not repeated
    full N x M GPU scans
"""
import math
import numpy as np
import torch
import torch.nn as nn
from fotonet.utils.boxes import box_iou, xywh_to_xyxy

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


def _empty_indices(device):
    return (
        torch.empty(0, dtype=torch.long, device=device),
        torch.empty(0, dtype=torch.long, device=device),
    )


def validate_target_mapping(target):
    """Validate normalized detection targets before they reach a CUDA matcher.

    Dataset/prefetch callers run this while targets are still on CPU, so the
    error path cannot serialize a training stream.  Direct matcher callers are
    validated too, including CUDA inputs where a malformed public input is
    worth the one-time synchronization needed to report it safely.
    """
    try:
        boxes = target["boxes"]
        labels = target["labels"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Each target must contain tensor 'boxes' and 'labels' entries") from exc
    if not torch.is_tensor(boxes) or not torch.is_tensor(labels):
        raise ValueError("Target 'boxes' and 'labels' entries must be tensors")
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"Target boxes must have shape [N, 4], got {tuple(boxes.shape)}")
    if labels.ndim != 1 or labels.shape[0] != boxes.shape[0]:
        raise ValueError(
            "Target labels must have shape [N] and match target boxes; "
            f"got labels={tuple(labels.shape)}, boxes={tuple(boxes.shape)}"
        )
    if labels.is_floating_point():
        labels_valid = torch.isfinite(labels).all() & labels.eq(labels.round()).all()
    else:
        labels_valid = torch.ones((), dtype=torch.bool, device=labels.device)
    geometry = torch.isfinite(boxes).all()
    if boxes.numel():
        centers = boxes[:, :2]
        wh = boxes[:, 2:]
        corners = torch.cat((centers - wh * 0.5, centers + wh * 0.5), dim=1)
        geometry = geometry & centers.ge(0).all() & centers.le(1).all()
        geometry = geometry & wh.gt(0).all() & wh.le(1).all()
        geometry = geometry & corners.ge(0).all() & corners.le(1).all()
    labels_valid = labels_valid & labels.ge(0).all()
    if not bool(geometry & labels_valid):
        raise ValueError(
            "Targets must use finite normalized xywh boxes fully inside [0, 1] with positive width/height, "
            "and non-negative integer labels."
        )


def _target_tensors(target, device, box_dtype):
    """Move one target mapping to the prediction device with shape checks."""
    if not bool(target.get("_fotonet_target_validated", False)):
        validate_target_mapping(target)
    try:
        boxes = target["boxes"].to(device=device, dtype=box_dtype, non_blocking=True)
        labels = target["labels"].to(device=device, dtype=torch.long, non_blocking=True)
    except (KeyError, AttributeError) as exc:
        raise ValueError("Each target must contain tensor 'boxes' and 'labels' entries") from exc
    return boxes, labels


def _compact_greedy_candidates_cpu(
    candidate_values,
    candidate_preds,
    n_preds,
    n_targets,
    *,
    largest,
):
    """CPU portion of compact greedy matching.

    Inputs are already compact NumPy arrays. Keeping transfer outside this
    helper lets a whole training batch cross the CUDA/CPU boundary together
    instead of forcing one synchronization per image.
    """
    k = min(n_preds, n_targets)
    expected_shape = (k, n_targets)
    candidate_values = np.asarray(candidate_values)
    candidate_preds = np.asarray(candidate_preds)
    if candidate_values.shape != expected_shape or candidate_preds.shape != expected_shape:
        raise ValueError("compact O2O candidates must have shape [min(N, M), M]")

    values_cpu = candidate_values.reshape(-1)
    preds_cpu = candidate_preds.reshape(-1)
    targets_cpu = np.broadcast_to(
        np.arange(n_targets, dtype=np.int64),
        expected_shape,
    ).reshape(-1)
    primary_key = -values_cpu if largest else values_cpu
    order = np.lexsort((targets_cpu, preds_cpu, primary_key))

    used_preds = np.zeros(n_preds, dtype=np.bool_)
    used_targets = np.zeros(n_targets, dtype=np.bool_)
    matched_preds = np.empty(k, dtype=np.int64)
    matched_targets = np.empty(k, dtype=np.int64)
    matched = 0
    for flat_index in order:
        pred_index = int(preds_cpu[flat_index])
        target_index = int(targets_cpu[flat_index])
        if used_preds[pred_index] or used_targets[target_index]:
            continue
        used_preds[pred_index] = True
        used_targets[target_index] = True
        matched_preds[matched] = pred_index
        matched_targets[matched] = target_index
        matched += 1
        if matched == k:
            break

    if matched != k:
        raise RuntimeError(
            "Internal O2O candidate matching invariant failed: expected "
            f"{k} unique pairs, produced {matched}."
        )
    return matched_preds, matched_targets


def _compact_greedy_candidates(candidate_values, candidate_preds, n_preds, n_targets, *, largest):
    """Resolve complete one-to-one greedy matches from top-K candidates.

    Each target contributes ``K = min(n_preds, n_targets)`` distinct anchors.
    When ``n_preds >= n_targets``, that gives every target ``n_targets``
    candidates; when ``n_preds < n_targets``, it gives every target all
    predictions. Any maximal matching over this graph therefore has exactly K
    pairs: before K predictions are claimed, every unmatched target still has
    an unused candidate. This removes repeated full ``N x M`` GPU scans while
    retaining complete cardinality.

    With strictly ordered values, global greedy never needs an anchor below
    rank K for a target, because fewer than K anchors can be unavailable when
    that target is selected. Thus this matches the full-matrix greedy result
    for non-tied values. Ties are deterministic over the candidate set
    (value, anchor, target); CUDA ``topk`` may select a different subset at an
    all-equal K-th boundary, which remains an explicit non-exact tie policy.
    """
    # One compact transfer replaces K full GPU mask/reduction launches. The
    # final two keys reproduce the old anchor-major flattened tie preference.
    values_cpu = candidate_values.detach().contiguous().cpu().numpy()
    preds_cpu = candidate_preds.detach().contiguous().cpu().numpy()
    matched_preds, matched_targets = _compact_greedy_candidates_cpu(
        values_cpu,
        preds_cpu,
        n_preds,
        n_targets,
        largest=largest,
    )

    device = candidate_values.device
    return (
        torch.from_numpy(matched_preds).to(device=device, dtype=torch.long),
        torch.from_numpy(matched_targets).to(device=device, dtype=torch.long),
    )


class TaskAlignedAssigner(nn.Module):
    """
    Task-Aligned One-to-Many assigner. Fully vectorized — zero Python loops
    over the number of GT objects (M). Accepts real feat_shapes from the head.
    """
    def __init__(self, topk=10, alpha=1.0, beta=2.0, min_small_assign=4,
                 small_obj_px=8.0, strides=(8, 16, 32),
                 center_radius_cells=2.5, center_prior_cells=5.0,
                 proposal_rounds=4):
        super().__init__()
        self.topk   = topk
        self.alpha  = alpha
        self.beta   = beta
        self.min_small_assign = int(min_small_assign)
        self.small_obj_px = float(small_obj_px)
        self.center_radius_cells = float(center_radius_cells)
        self.center_prior_cells = float(center_prior_cells)
        self.proposal_rounds = max(int(proposal_rounds), 1)
        if self.center_radius_cells <= 0 or self.center_prior_cells <= 0:
            raise ValueError("center_radius_cells and center_prior_cells must be positive")
        self._anchor_cache = {}
        self.strides = [int(s) for s in strides]

    def _get_anchors(self, device, feat_shapes):
        """
        Build normalized anchor centers [total_anchors, 2] in [0, 1] coords.
        Each anchor is the center of its grid cell, normalized by image dimensions.
        feat_shapes: list or tensor of (H, W) per stride level.
        """
        # Ensure hashable key for cache. Production loss calls pass head anchor
        # metadata and bypass this fallback, avoiding a GPU ``tolist()`` sync.
        if torch.is_tensor(feat_shapes):
            fs_key = tuple(tuple(int(x) for x in row) for row in feat_shapes.detach().cpu().tolist())
        else:
            fs_key = tuple(tuple(int(x) for x in row) for row in feat_shapes)

        if len(fs_key) != len(self.strides):
            raise ValueError(
                f"feat_shapes has {len(fs_key)} levels but matcher has {len(self.strides)} strides"
            )
        if any(ny <= 0 or nx <= 0 for ny, nx in fs_key):
            raise ValueError(f"feat_shapes must contain positive [height, width] pairs, got {fs_key}")

        key = (str(device), fs_key, tuple(self.strides))
        if key not in self._anchor_cache:
            parts = []
            for i, (ny, nx) in enumerate(fs_key):
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

    def _anchors_for_outputs(self, outputs, device, n_queries, feat_shapes):
        """Use the head's exact decoded-grid metadata when it is available.

        Reconstructing normalized anchors from feature shapes was only exact for
        conveniently divisible square inputs and forced ``feat_shapes.tolist``
        to synchronize CUDA.  The head already supplies the true per-level
        grid and spatial size, so use it for both correctness and throughput.
        """
        anchor_points = outputs.get("anchor_points")
        stride_tensor = outputs.get("stride_tensor")
        imgsz = outputs.get("imgsz")
        has_any_metadata = any(value is not None for value in (anchor_points, stride_tensor, imgsz))
        if has_any_metadata:
            if anchor_points is None or stride_tensor is None or imgsz is None:
                raise ValueError(
                    "Task-aligned assignment metadata must include anchor_points, stride_tensor, and imgsz together"
                )
            anchor_points = anchor_points.to(device=device, dtype=torch.float32, non_blocking=True)
            stride_tensor = stride_tensor.to(device=device, dtype=torch.float32, non_blocking=True)
            if anchor_points.ndim != 2 or anchor_points.shape != (n_queries, 2):
                raise ValueError(
                    "anchor_points must have shape [num_queries, 2]; "
                    f"got {tuple(anchor_points.shape)} for {n_queries} queries"
                )
            if stride_tensor.numel() != n_queries:
                raise ValueError(
                    "stride_tensor must contain one stride per query; "
                    f"got {tuple(stride_tensor.shape)} for {n_queries} queries"
                )
            stride_tensor = stride_tensor.reshape(-1, 1)
            if torch.is_tensor(imgsz):
                image_size = imgsz.to(device=device, dtype=torch.float32, non_blocking=True).reshape(-1)
            else:
                image_size = torch.as_tensor(imgsz, device=device, dtype=torch.float32).reshape(-1)
            if image_size.numel() == 1:
                image_size = image_size.repeat(2)
            elif image_size.numel() != 2:
                raise ValueError("imgsz must be a scalar or [width, height] pair")
            return (
                anchor_points * stride_tensor / image_size.view(1, 2),
                stride_tensor.reshape(-1),
                image_size,
            )

        if feat_shapes is None:
            feat_shapes = self._infer_feat_shapes(n_queries)
        anchors = self._get_anchors(device, feat_shapes)
        if anchors.shape[0] != n_queries:
            raise ValueError(
                "Feature shapes do not match prediction query count: "
                f"got {anchors.shape[0]} anchors for {n_queries} predictions"
            )
        image_h = max(int(feat_shapes[j][0]) * self.strides[j] for j in range(len(feat_shapes)))
        image_w = max(int(feat_shapes[j][1]) * self.strides[j] for j in range(len(feat_shapes)))
        anchor_strides = torch.cat([
            torch.full(
                (int(feat_shapes[j][0]) * int(feat_shapes[j][1]),),
                float(self.strides[j]),
                dtype=torch.float32,
                device=device,
            )
            for j in range(len(feat_shapes))
        ])
        image_size = torch.tensor([image_w, image_h], dtype=torch.float32, device=device)
        return anchors, anchor_strides, image_size

    def _infer_feat_shapes(self, N):
        """
        Fallback: infer feat_shapes from total anchor count N.
        Works for square images: N = (S/8)^2 + (S/16)^2 + (S/32)^2
              = S^2*(1/64 + 1/256 + 1/1024) = S^2 * 21/1024
        """
        denom = sum(1.0 / float(s * s) for s in self.strides)
        S = int(round(math.sqrt(N / max(denom, 1e-12))))
        return [(S // s, S // s) for s in self.strides]

    def _alignment_scores(
        self,
        pred_probs,
        pred_boxes,
        tgt_boxes,
        tgt_labels,
        anchors,
        feat_shapes,
        *,
        anchor_strides,
        image_size,
    ):
        """Return the task-aligned score matrix and priors shared by O2M/O2O."""
        in_gt, center_dists, tgt_areas, tgt_xyxy = self._target_geometry(
            tgt_boxes,
            anchors,
            anchor_strides,
            image_size,
        )

        cls_score = pred_probs[:, tgt_labels]
        iou = box_iou(xywh_to_xyxy(pred_boxes), tgt_xyxy).clamp_min(0.0)
        # A target-wide size multiplier cannot affect the top-k ordering within
        # that target. Small-target coverage is handled explicitly after
        # ownership resolution instead of pretending this scalar is STAL.
        align = cls_score.pow(self.alpha) * iou.pow(self.beta) * in_gt

        # Early training can produce zero IoU for every candidate of a GT. In that
        # state task-aligned assignment becomes almost random and supervision thins
        # out badly. Fall back to a center prior until decoded boxes improve.
        empty_gt = align.max(dim=0).values <= 0
        anchor_radius = anchor_strides.to(
            device=anchors.device, dtype=anchors.dtype
        ).reshape(-1, 1)
        center_prior = (1.0 - center_dists / (anchor_radius * self.center_prior_cells).clamp_min(1e-6)).clamp_min(0.0)
        align = torch.where(
            empty_gt.unsqueeze(0),
            center_prior * in_gt.to(dtype=center_prior.dtype),
            align,
        )

        return align, in_gt, center_dists, tgt_areas

    def _target_geometry(self, tgt_boxes, anchors, anchor_strides, image_size):
        tgt_xyxy = xywh_to_xyxy(tgt_boxes)
        anc_x = anchors[:, 0]
        anc_y = anchors[:, 1]
        in_x = (anc_x[:, None] > tgt_xyxy[None, :, 0]) & (
            anc_x[:, None] < tgt_xyxy[None, :, 2]
        )
        in_y = (anc_y[:, None] > tgt_xyxy[None, :, 1]) & (
            anc_y[:, None] < tgt_xyxy[None, :, 3]
        )
        in_gt = in_x & in_y
        image_scale = image_size.to(
            device=anchors.device, dtype=anchors.dtype
        ).view(1, 1, 2)
        center_delta = (
            anchors[:, None, :] - tgt_boxes[None, :, :2]
        ) * image_scale
        center_dists = center_delta.square().sum(dim=-1).sqrt()
        anchor_radius = anchor_strides.to(
            device=anchors.device, dtype=anchors.dtype
        ).reshape(-1, 1)
        in_gt |= center_dists <= anchor_radius * self.center_radius_cells
        tgt_wh = tgt_boxes[:, 2:4]
        tgt_areas = tgt_wh[:, 0] * tgt_wh[:, 1]
        return in_gt, center_dists, tgt_areas, tgt_xyxy

    def prepare_geometry(self, outputs, targets, feat_shapes=None):
        """Prepare branch-independent target geometry once per microbatch."""
        device = outputs["pred_logits"].device
        batch, queries = outputs["pred_logits"].shape[:2]
        if len(targets) != batch:
            raise ValueError(
                f"Batch has {batch} predictions but {len(targets)} target mappings"
            )
        anchors, anchor_strides, image_size = self._anchors_for_outputs(
            outputs, device, queries, feat_shapes
        )
        prepared = []
        for target in targets:
            boxes, labels = _target_tensors(
                target,
                device,
                outputs["pred_boxes"].dtype,
            )
            if boxes.numel():
                geometry = self._target_geometry(
                    boxes.float(),
                    anchors,
                    anchor_strides,
                    image_size,
                )
            else:
                geometry = None
            prepared.append((boxes, labels, geometry))
        return {
            "anchors": anchors,
            "anchor_strides": anchor_strides,
            "image_size": image_size,
            "items": prepared,
        }

    @staticmethod
    def _owner_counts(owner, num_targets):
        """Count dense owner assignments without compacting CUDA indices."""
        valid = owner.ge(0)
        return torch.zeros(num_targets, dtype=torch.long, device=owner.device).scatter_add_(
            0,
            owner.clamp_min(0),
            valid.to(dtype=torch.long),
        )

    @staticmethod
    def _inverse_stable_rank(values):
        """Return a deterministic 0..N-1 rank (lower input values first)."""
        order = torch.argsort(values, stable=True)
        ranks = torch.empty_like(order)
        return ranks.scatter_(0, order, torch.arange(values.numel(), device=values.device))

    @staticmethod
    def _owner_to_indices(owner):
        """Compact dense ownership once at the matcher API boundary."""
        valid = owner.ge(0)
        anchors = torch.arange(owner.numel(), device=owner.device, dtype=torch.long)
        return anchors[valid], owner[valid]

    def _claim_slots(self, owner, score, eligible, request, priority, *, allow_global_residual=False):
        """Claim requested GT slots with bounded, tensor-only conflict resolution.

        ``owner`` stays dense until the final API conversion.  Four proposal
        rounds select target-local candidates without Python loops over GTs.
        O2M passes ``allow_global_residual=False`` so an unavailable local
        candidate is never fabricated as a far-away regression positive. O2O
        can opt into a nearest-anchor residual, matching its one-to-one
        fallback contract while still avoiding arbitrary zero-score ties.
        """
        n_anchors, n_targets = score.shape
        quota = int(request.shape[1])
        if n_anchors == 0 or n_targets == 0 or quota == 0:
            return owner

        slot_gt = torch.arange(n_targets, device=owner.device, dtype=torch.long).repeat_interleave(quota)
        slot_q = torch.arange(quota, device=owner.device, dtype=torch.long).repeat(n_targets)
        request_flat = request.reshape(-1)
        priority_flat = priority.reshape(-1)
        total_slots = int(slot_gt.numel())
        if priority_flat.numel() != total_slots:
            raise ValueError("slot priority must have the same shape as request")

        safe_score = torch.nan_to_num(score, nan=-1e6, neginf=-1e6, posinf=1e6)
        local_score = safe_score.masked_fill(~eligible, float("-inf"))
        free_initial = owner.eq(-1)
        ranked_score = local_score.masked_fill(~free_initial[:, None], float("-inf"))
        proposal_count = min(n_anchors, max(1, self.proposal_rounds * quota))
        candidates = ranked_score.topk(proposal_count, dim=0, largest=True, sorted=True).indices.transpose(0, 1)
        accepted = torch.zeros(total_slots, dtype=torch.bool, device=owner.device)
        invalid_priority = total_slots

        for round_index in range(self.proposal_rounds):
            candidate_rank = round_index * quota + slot_q
            has_candidate = candidate_rank.lt(proposal_count)
            proposal = candidates[slot_gt, candidate_rank.clamp_max(proposal_count - 1)]
            proposal_score = local_score[proposal, slot_gt]
            active = (
                request_flat
                & ~accepted
                & has_candidate
                & owner[proposal].eq(-1)
                & torch.isfinite(proposal_score)
            )
            key = torch.where(active, priority_flat, torch.full_like(priority_flat, invalid_priority))
            winning_key = torch.full(
                (n_anchors,),
                invalid_priority,
                dtype=priority_flat.dtype,
                device=owner.device,
            )
            winning_key.scatter_reduce_(0, proposal, key, reduce="amin", include_self=True)
            wins = active & key.eq(winning_key[proposal])

            owner_update = torch.zeros_like(owner)
            owner_update.scatter_reduce_(
                0,
                proposal,
                torch.where(wins, slot_gt + 1, torch.zeros_like(slot_gt)),
                reduce="amax",
                include_self=True,
            )
            owner = torch.where(owner_update.gt(0), owner_update - 1, owner)
            accepted |= wins

        if not allow_global_residual:
            return owner

        # O2O's documented fallback is the nearest *free* anchor if a GT had
        # no winning TAL candidate. The cumsum map assigns each remaining
        # request a unique free anchor in deterministic priority order.
        residual = request_flat & ~accepted
        requested_by_priority = torch.zeros(total_slots, dtype=torch.bool, device=owner.device)
        requested_by_priority.scatter_(0, priority_flat, residual)
        rank_by_priority = requested_by_priority.to(torch.long).cumsum(0) - 1
        slot_rank = rank_by_priority[priority_flat]
        free = owner.eq(-1)
        can_fill = residual & slot_rank.lt(free.to(torch.long).sum())
        fallback_score = safe_score.max(dim=1).values
        fallback_order = torch.where(
            free,
            fallback_score,
            fallback_score.amin() - 1.0,
        ).argsort(descending=True, stable=True)
        fallback_anchor = fallback_order[slot_rank.clamp(min=0, max=n_anchors - 1)]
        owner_update = torch.zeros_like(owner)
        owner_update.scatter_reduce_(
            0,
            fallback_anchor,
            torch.where(can_fill, slot_gt + 1, torch.zeros_like(slot_gt)),
            reduce="amax",
            include_self=True,
        )
        return torch.where(owner_update.gt(0), owner_update - 1, owner)

    def _small_target_mask(self, tgt_boxes, image_size):
        """Use geometric-mean side length in pixels for rectangle-safe small-GT coverage."""
        pixel_wh = tgt_boxes[:, 2:4] * image_size.to(device=tgt_boxes.device, dtype=tgt_boxes.dtype).view(1, 2)
        return pixel_wh.prod(dim=1).clamp_min(0.0).sqrt() < self.small_obj_px

    @staticmethod
    def _alignment_groups(items, num_queries, max_elements=4_000_000):
        """Bucket non-empty images for bounded-memory batched alignment."""
        ordered = sorted(
            (
                (index, int(item[0].shape[0]))
                for index, item in enumerate(items)
                if int(item[0].shape[0]) > 0
            ),
            key=lambda pair: (pair[1], pair[0]),
        )
        groups = []
        current = []
        maximum_targets = 0
        for index, target_count in ordered:
            candidate_max = max(maximum_targets, target_count)
            candidate_size = (
                (len(current) + 1) * int(num_queries) * candidate_max
            )
            if current and candidate_size > max_elements:
                groups.append(tuple(current))
                current = []
                maximum_targets = 0
            current.append(index)
            maximum_targets = max(maximum_targets, target_count)
        if current:
            groups.append(tuple(current))
        return tuple(groups)

    def _batched_alignment(self, outputs, geometry):
        """Compute class/IoU alignment in bounded image batches.

        Assignment ownership remains image-local, but batching the expensive
        sigmoid, gather, pairwise IoU, and center-prior kernels removes most of
        the per-image CUDA launch overhead. Padding is masked and groups are
        capped to avoid a pathological crowded image inflating the full
        microbatch allocation.
        """
        items = geometry["items"]
        # Preserve the matcher subclass extension point. Custom assigners may
        # override _alignment_scores to define eligibility/priors; the
        # production base implementation alone uses the fused batch path.
        if type(self)._alignment_scores is not TaskAlignedAssigner._alignment_scores:
            per_image = []
            for index, (boxes, labels, _shared) in enumerate(items):
                if not len(boxes):
                    per_image.append(None)
                    continue
                alignment, in_gt, distances, areas = self._alignment_scores(
                    outputs["pred_logits"][index].float().sigmoid(),
                    outputs["pred_boxes"][index].float(),
                    boxes.float(),
                    labels,
                    geometry["anchors"],
                    None,
                    anchor_strides=geometry["anchor_strides"],
                    image_size=geometry["image_size"],
                )
                original = items[index][2]
                target_xyxy = (
                    original[3]
                    if original is not None
                    else xywh_to_xyxy(boxes.float())
                )
                items[index] = (
                    boxes,
                    labels,
                    (in_gt, distances, areas, target_xyxy),
                )
                per_image.append(alignment)
            return per_image
        batch, queries = outputs["pred_logits"].shape[:2]
        result = [None] * batch
        for group in self._alignment_groups(items, queries):
            maximum_targets = max(int(items[index][0].shape[0]) for index in group)
            group_tensor = torch.tensor(
                group,
                dtype=torch.long,
                device=outputs["pred_logits"].device,
            )
            probabilities = (
                outputs["pred_logits"].index_select(0, group_tensor).float().sigmoid()
            )
            prediction_boxes = outputs["pred_boxes"].index_select(
                0, group_tensor
            ).float()
            target_boxes = prediction_boxes.new_zeros(
                (len(group), maximum_targets, 4)
            )
            target_labels = torch.zeros(
                (len(group), maximum_targets),
                dtype=torch.long,
                device=prediction_boxes.device,
            )
            valid = torch.zeros(
                (len(group), maximum_targets),
                dtype=torch.bool,
                device=prediction_boxes.device,
            )
            in_gt = torch.zeros(
                (len(group), queries, maximum_targets),
                dtype=torch.bool,
                device=prediction_boxes.device,
            )
            center_dists = prediction_boxes.new_zeros(
                (len(group), queries, maximum_targets)
            )
            target_xyxy = prediction_boxes.new_zeros(
                (len(group), maximum_targets, 4)
            )
            for local, index in enumerate(group):
                boxes, labels, shared = items[index]
                count = int(boxes.shape[0])
                target_boxes[local, :count] = boxes.to(
                    dtype=prediction_boxes.dtype
                )
                target_labels[local, :count] = labels
                valid[local, :count] = True
                if shared is not None:
                    in_gt[local, :, :count] = shared[0]
                    center_dists[local, :, :count] = shared[1]
                    target_xyxy[local, :count] = shared[3]

            class_scores = probabilities.gather(
                2,
                target_labels[:, None, :].expand(-1, queries, -1),
            )
            prediction_xyxy = xywh_to_xyxy(prediction_boxes)
            inter_x1 = torch.maximum(
                prediction_xyxy[:, :, None, 0],
                target_xyxy[:, None, :, 0],
            )
            inter_y1 = torch.maximum(
                prediction_xyxy[:, :, None, 1],
                target_xyxy[:, None, :, 1],
            )
            inter_x2 = torch.minimum(
                prediction_xyxy[:, :, None, 2],
                target_xyxy[:, None, :, 2],
            )
            inter_y2 = torch.minimum(
                prediction_xyxy[:, :, None, 3],
                target_xyxy[:, None, :, 3],
            )
            intersection = (inter_x2 - inter_x1).clamp(0) * (
                inter_y2 - inter_y1
            ).clamp(0)
            prediction_area = (
                prediction_xyxy[:, :, 2] - prediction_xyxy[:, :, 0]
            ) * (
                prediction_xyxy[:, :, 3] - prediction_xyxy[:, :, 1]
            )
            target_area = (
                target_xyxy[:, :, 2] - target_xyxy[:, :, 0]
            ) * (
                target_xyxy[:, :, 3] - target_xyxy[:, :, 1]
            )
            iou = intersection / (
                prediction_area[:, :, None]
                + target_area[:, None, :]
                - intersection
                + 1e-7
            )
            iou = iou.clamp_min(0.0)
            alignment = (
                class_scores.pow(self.alpha)
                * iou.pow(self.beta)
                * in_gt
            )
            empty = alignment.max(dim=1).values.le(0) & valid
            anchor_radius = geometry["anchor_strides"].reshape(1, -1, 1)
            center_prior = (
                1.0
                - center_dists
                / (
                    anchor_radius * self.center_prior_cells
                ).clamp_min(1e-6)
            ).clamp_min(0.0)
            alignment = torch.where(
                empty[:, None, :],
                center_prior * in_gt.to(dtype=center_prior.dtype),
                alignment,
            )
            alignment = alignment * valid[:, None, :]
            for local, index in enumerate(group):
                count = int(items[index][0].shape[0])
                result[index] = alignment[local, :, :count]
        return result

    @torch.no_grad()
    def forward(self, outputs, targets, feat_shapes=None, geometry=None):
        device = outputs["pred_logits"].device
        bs, N  = outputs["pred_logits"].shape[:2]

        if len(targets) != bs:
            raise ValueError(f"Batch has {bs} predictions but {len(targets)} target mappings")
        if N == 0:
            return [_empty_indices(device) for _ in range(bs)]
        if geometry is None:
            geometry = self.prepare_geometry(outputs, targets, feat_shapes)
        anchors = geometry["anchors"]
        anchor_strides = geometry["anchor_strides"]
        image_size = geometry["image_size"]
        batched_alignment = self._batched_alignment(outputs, geometry)
        indices = []

        for i in range(bs):
            pred_logits = outputs["pred_logits"][i].float().sigmoid()
            pred_boxes  = outputs["pred_boxes"][i].float()
            tgt_boxes, tgt_labels, shared_geometry = geometry["items"][i]
            tgt_boxes = tgt_boxes.to(dtype=pred_boxes.dtype)
            M = tgt_boxes.shape[0]

            if M == 0:
                indices.append(_empty_indices(device))
                continue

            if shared_geometry is None:
                indices.append(_empty_indices(device))
                continue
            in_gt, center_dists, _, _tgt_xyxy = shared_geometry
            align = batched_alignment[i]

            topk = min(self.topk, N)
            if topk <= 0:
                indices.append(_empty_indices(device))
                continue

            # Break only numerical ties with a cyclic anchor preference. In
            # crowded/identical-GT cases, plain `topk` gives every target the
            # same low-index anchors; this deterministic sub-epsilon term
            # spreads equal-score targets across distinct local candidates
            # without changing meaningful TAL ordering.
            anchor_order = torch.arange(N, device=device, dtype=torch.long).view(-1, 1)
            target_order = torch.arange(M, device=device, dtype=torch.long).view(1, -1)
            tie_rank = torch.remainder(anchor_order - target_order, N).to(dtype=align.dtype)
            selection_align = align - tie_rank * (1e-6 / max(N, 1))

            _, topk_idx = selection_align.topk(topk, dim=0)
            mask         = torch.zeros(N, M, dtype=torch.bool, device=device)
            mask.scatter_(0, topk_idx, True)
            mask        &= in_gt

            small_gt = self._small_target_mask(tgt_boxes, image_size)
            candidate_counts = in_gt.sum(dim=0).to(torch.long)
            # Small targets receive the first recovery claims under capacity
            # pressure; within each class, constrained GTs go first.
            recovery_key = candidate_counts + (~small_gt).to(torch.long) * (N + 1)
            recovery_priority = self._inverse_stable_rank(recovery_key)
            recovery_score = (
                2.0 * in_gt.to(dtype=align.dtype)
                + selection_align
                - center_dists / image_size.max().clamp_min(1.0) * 1e-6
            )
            # Reserve contested anchors for targets with fewer local options.
            # The priority offset is deliberately larger than the bounded TAL
            # score range; without it, a broad target can take the sole anchor
            # of a constrained overlapping target and make a feasible matching
            # impossible before recovery sees the alternate anchor.
            scarcity = candidate_counts.max() - candidate_counts
            reservation_bonus = (
                scarcity.to(dtype=selection_align.dtype)
                + small_gt.to(dtype=selection_align.dtype) * 0.5
            ).view(1, -1) * 4.0
            candidate_scores = (selection_align + reservation_bonus).masked_fill(~mask, float("-inf"))
            best_scores, best_gt = candidate_scores.max(dim=1)
            owner = torch.where(
                torch.isfinite(best_scores),
                best_gt,
                torch.full_like(best_gt, -1),
            )
            owner = self._claim_slots(
                owner,
                recovery_score,
                in_gt,
                self._owner_counts(owner, M).eq(0).unsqueeze(1),
                recovery_priority.unsqueeze(1),
            )

            if self.min_small_assign > 1:
                quota = min(self.min_small_assign, N)
                slot = torch.arange(quota, device=device, dtype=torch.long)
                needed = (quota - self._owner_counts(owner, M)).clamp_min(0)
                small_request = small_gt[:, None] & slot.unsqueeze(0).lt(needed[:, None])
                physical_area = (tgt_boxes[:, 2:4] * image_size.to(dtype=tgt_boxes.dtype).view(1, 2)).prod(dim=1)
                area_rank = self._inverse_stable_rank(physical_area)
                small_priority = area_rank[:, None] * quota + slot.unsqueeze(0)
                owner = self._claim_slots(
                    owner,
                    recovery_score,
                    in_gt,
                    small_request,
                    small_priority,
                )

            indices.append(self._owner_to_indices(owner))

        return indices


class TaskAlignedO2OAssigner(TaskAlignedAssigner):
    """
    Task-aligned one-to-one assigner for the NMS-free branch.
    Uses the same class confidence, IoU, inside-GT, center prior, and
    feature-shape anchor signal as TaskAlignedAssigner, then resolves to one
    prediction per GT with unique prediction indices whenever possible.

    ``exact=True`` keeps the opt-in full SciPy/Hungarian solution. The normal
    path has a GPU no-conflict fast path. Only collided cases transfer each
    target's ``min(num_predictions, num_targets)`` best candidates to a compact
    CPU greedy resolver, avoiding a ``min(N, M)``-iteration GPU scan of the
    full ``N x M`` score matrix.
    """
    def __init__(self, topk=1, alpha=1.0, beta=2.0, min_small_assign=1,
                 small_obj_px=8.0, strides=(8, 16, 32), exact=False,
                 center_radius_cells=2.5, center_prior_cells=5.0,
                 proposal_rounds=4):
        super().__init__(
            topk=topk,
            alpha=alpha,
            beta=beta,
            min_small_assign=min_small_assign,
            small_obj_px=small_obj_px,
            strides=strides,
            center_radius_cells=center_radius_cells,
            center_prior_cells=center_prior_cells,
            proposal_rounds=proposal_rounds,
        )
        self.exact = bool(exact)

    def _scores_with_nearest_fallback(self, align, center_dists):
        # Any real TAL positive outranks a fallback. A target that loses its
        # only positive therefore takes its nearest remaining anchor instead
        # of an arbitrary zero-score/outside-GT index.
        return torch.where(align > 0, align, -center_dists)

    @staticmethod
    def _compact_greedy_candidates(candidate_scores, candidate_preds, n_preds, n_targets):
        """Score-maximizing specialization of the shared compact resolver."""
        return _compact_greedy_candidates(
            candidate_scores,
            candidate_preds,
            n_preds,
            n_targets,
            largest=True,
        )

    def _resolve_one_to_one(self, scores, center_dists):
        device = scores.device
        N, M = scores.shape
        if M == 0 or N == 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )

        scores = self._scores_with_nearest_fallback(scores, center_dists)
        score_limit = torch.finfo(scores.dtype).max
        scores = torch.nan_to_num(scores, nan=-score_limit, neginf=-score_limit, posinf=score_limit)
        if self.exact and linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment((-scores).detach().cpu().numpy())
            return (
                torch.as_tensor(rows, dtype=torch.long, device=device),
                torch.as_tensor(cols, dtype=torch.long, device=device),
            )

        # The common non-collision case stays entirely on the GPU. One scalar
        # check is far cheaper than the old per-target full-matrix reductions.
        if N >= M:
            _, best_preds = scores.max(dim=0)
            claims = torch.zeros(N, dtype=torch.int32, device=device)
            claims.scatter_add_(0, best_preds, torch.ones_like(best_preds, dtype=torch.int32))
            if bool(claims.max().le(1)):
                return best_preds, torch.arange(M, dtype=torch.long, device=device)

        # Collision path: retain only enough per-target candidates to prove a
        # complete matching, then resolve globally without an M-step GPU loop.
        k = min(N, M)
        candidate_scores, candidate_preds = scores.topk(k, dim=0, largest=True, sorted=True)
        return self._compact_greedy_candidates(candidate_scores, candidate_preds, N, M)

    def _resolve_non_exact_batch(self, score_matrices, best_predictions, conflicts, device):
        """Resolve a microbatch with one collision check and compact transfer.

        The old path converted one CUDA scalar to ``bool`` for every non-empty
        image.  Retaining each image's exact ``max`` result while batching the
        flags and collided candidate buffers preserves assignment and tie
        semantics but removes nearly all image-level stream drains.
        """
        active_conflicts = [
            bool(value)
            for value in torch.stack(conflicts).detach().cpu().tolist()
        ] if conflicts else []
        metadata = []
        value_parts = []
        pred_parts = []
        active_index = 0
        for scores in score_matrices:
            if scores is None:
                metadata.append(None)
                continue
            n_preds, n_targets = scores.shape
            collided = active_conflicts[active_index]
            active_index += 1
            if not collided:
                metadata.append(("fast", n_preds, n_targets, 0))
                continue
            k = min(n_preds, n_targets)
            values, predictions = scores.topk(
                k, dim=0, largest=True, sorted=True
            )
            num_candidates = values.numel()
            metadata.append(("collision", n_preds, n_targets, num_candidates))
            value_parts.append(values.reshape(-1))
            pred_parts.append(predictions.reshape(-1))

        cpu_results = []
        if value_parts:
            values_cpu = torch.cat(value_parts).detach().contiguous().cpu().numpy()
            preds_cpu = torch.cat(pred_parts).detach().contiguous().cpu().numpy()
            candidate_offset = 0
            for item in metadata:
                if item is None or item[0] != "collision":
                    cpu_results.append(None)
                    continue
                _, n_preds, n_targets, num_candidates = item
                k = min(n_preds, n_targets)
                next_offset = candidate_offset + num_candidates
                cpu_results.append(
                    _compact_greedy_candidates_cpu(
                        values_cpu[candidate_offset:next_offset].reshape(k, n_targets),
                        preds_cpu[candidate_offset:next_offset].reshape(k, n_targets),
                        n_preds,
                        n_targets,
                        largest=True,
                    )
                )
                candidate_offset = next_offset
        else:
            cpu_results = [None] * len(metadata)

        packed = None
        collision_results = [item for item in cpu_results if item is not None]
        if collision_results:
            packed_cpu = np.concatenate(
                [
                    np.stack((predictions, targets), axis=1)
                    for predictions, targets in collision_results
                ],
                axis=0,
            )
            packed = torch.from_numpy(packed_cpu).to(
                device=device, dtype=torch.long
            )

        indices = []
        fast_index = 0
        packed_offset = 0
        for item, cpu_result in zip(metadata, cpu_results):
            if item is None:
                indices.append(_empty_indices(device))
                continue
            kind, n_preds, n_targets, _ = item
            if kind == "fast":
                predictions = best_predictions[fast_index]
                indices.append(
                    (
                        predictions,
                        torch.arange(n_targets, dtype=torch.long, device=device),
                    )
                )
            else:
                count = min(n_preds, n_targets)
                pairs = packed[packed_offset:packed_offset + count]
                indices.append((pairs[:, 0], pairs[:, 1]))
                packed_offset += count
            fast_index += 1
        return indices

    @torch.no_grad()
    def forward(self, outputs, targets, feat_shapes=None, geometry=None):
        device = outputs["pred_logits"].device
        bs, N = outputs["pred_logits"].shape[:2]

        if len(targets) != bs:
            raise ValueError(f"Batch has {bs} predictions but {len(targets)} target mappings")
        if N == 0:
            return [_empty_indices(device) for _ in range(bs)]
        if geometry is None:
            geometry = self.prepare_geometry(outputs, targets, feat_shapes)
        anchors = geometry["anchors"]
        anchor_strides = geometry["anchor_strides"]
        image_size = geometry["image_size"]
        batched_alignment = self._batched_alignment(outputs, geometry)
        indices = []
        score_matrices = []
        best_predictions = []
        conflicts = []

        for i in range(bs):
            pred_probs = outputs["pred_logits"][i].float().sigmoid()
            pred_boxes = outputs["pred_boxes"][i].float()
            tgt_boxes, tgt_labels, shared_geometry = geometry["items"][i]
            tgt_boxes = tgt_boxes.to(dtype=pred_boxes.dtype)
            M = tgt_boxes.shape[0]

            if M == 0:
                if self.exact:
                    indices.append(_empty_indices(device))
                else:
                    score_matrices.append(None)
                continue

            if shared_geometry is None:
                if self.exact:
                    indices.append(_empty_indices(device))
                else:
                    score_matrices.append(None)
                continue
            _in_gt, center_dists, _, _tgt_xyxy = shared_geometry
            align = batched_alignment[i]
            if self.exact:
                indices.append(self._resolve_one_to_one(align, center_dists))
                continue
            scores = self._scores_with_nearest_fallback(align, center_dists)
            score_limit = torch.finfo(scores.dtype).max
            scores = torch.nan_to_num(
                scores,
                nan=-score_limit,
                neginf=-score_limit,
                posinf=score_limit,
            )
            score_matrices.append(scores)
            if N >= M:
                _, best = scores.max(dim=0)
                claims = torch.zeros(
                    N, dtype=torch.int32, device=device
                )
                claims.scatter_add_(
                    0,
                    best,
                    torch.ones_like(best, dtype=torch.int32),
                )
                conflict = claims.max().gt(1)
            else:
                best = torch.empty(0, dtype=torch.long, device=device)
                conflict = scores.new_tensor(True, dtype=torch.bool)
            best_predictions.append(best)
            conflicts.append(conflict)

        if self.exact:
            return indices
        return self._resolve_non_exact_batch(
            score_matrices,
            best_predictions,
            conflicts,
            device,
        )
