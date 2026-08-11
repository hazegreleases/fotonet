"""Small dependency-free tracker used by :meth:`FOTONET.track`.

[SIMPLIFIED] This is an IoU/class association tracker, not a motion-model
tracker. It provides stable IDs for ordered image/video frames without
pretending to be ByteTrack, BoT-SORT, or a replacement for a dedicated MOT
package.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from fotonet.engine.results import Results
from fotonet.utils.boxes import box_iou, xywh_to_xyxy


@dataclass
class _Track:
    box: torch.Tensor
    cls: int
    age: int = 0


class IoUTracker:
    """Greedy score-ordered, class-aware IoU tracker."""

    def __init__(self, iou_threshold=0.3, max_age=30):
        self.iou_threshold = float(iou_threshold)
        self.max_age = max(int(max_age), 0)
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ValueError("tracker IoU threshold must be in (0, 1].")
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def reset(self):
        self._tracks.clear()
        self._next_id = 1

    def update(self, result: Results) -> Results:
        boxes = result.boxes_tensor.detach().cpu()
        classes = result.classes.detach().cpu().long()
        scores = result.scores.detach().cpu()
        ids = torch.full((len(boxes),), -1, dtype=torch.long)

        for track in self._tracks.values():
            track.age += 1

        active_ids = [track_id for track_id, track in self._tracks.items() if track.age <= self.max_age]
        if len(boxes) and active_ids:
            track_boxes = torch.stack([self._tracks[track_id].box for track_id in active_ids])
            track_classes = torch.tensor([self._tracks[track_id].cls for track_id in active_ids])
            ious = box_iou(xywh_to_xyxy(boxes), xywh_to_xyxy(track_boxes))
            valid = (classes[:, None] == track_classes[None, :]) & (ious >= self.iou_threshold)
            # Highest confidence detections get first choice, then highest IoU.
            matched_dets, matched_tracks = set(), set()
            for det_idx in scores.argsort(descending=True).tolist():
                candidates = torch.where(valid[det_idx])[0]
                if not len(candidates):
                    continue
                ranked = candidates[ious[det_idx, candidates].argsort(descending=True)]
                for track_col in ranked.tolist():
                    track_id = active_ids[track_col]
                    if track_id not in matched_tracks:
                        ids[det_idx] = track_id
                        self._tracks[track_id] = _Track(boxes[det_idx].clone(), int(classes[det_idx]), 0)
                        matched_dets.add(det_idx)
                        matched_tracks.add(track_id)
                        break

        for det_idx in range(len(boxes)):
            if int(ids[det_idx]) >= 0:
                continue
            track_id = self._next_id
            self._next_id += 1
            ids[det_idx] = track_id
            self._tracks[track_id] = _Track(boxes[det_idx].clone(), int(classes[det_idx]), 0)

        self._tracks = {track_id: track for track_id, track in self._tracks.items() if track.age <= self.max_age}
        return Results(
            result.orig_img,
            result.boxes_tensor,
            result.scores,
            result.classes,
            result.names,
            ids.to(result.boxes_tensor.device),
            result.orig_shape,
        )
