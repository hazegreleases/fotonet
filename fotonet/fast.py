"""FastPredictor — accelerated BGR inference for fotonete.

The fast path keeps the exact detection semantics of
``Fotonet.predict_bgr`` (two-stage NMS-free selection, identical letterbox
math, identical clamping) while removing the per-frame CPU costs:

* zero-allocation letterbox into a reusable uint8 canvas,
* host->device transfer as uint8 through a pinned staging buffer (a
  pageable transfer is silently synchronous; pinned is genuinely async and
  moves 4x fewer bytes than float32),
* fused GPU-side BGR->RGB, float conversion, and /255 normalization,
* un-letterboxing fused on the GPU, with a single ``[N, 6]`` copy back.

``Fotonet.predict_bgr`` routes BGR NumPy frames on CUDA through the same
staging (see ``Fotonet._predict_bgr_fast``), so the official Results API
gets the same speedup with its contract unchanged.
"""

import numpy as np
import torch

__all__ = ["FastPredictor"]


def _letterbox_params(shape, target_h, target_w):
    """Gain/padding exactly matching Fotonet._preprocess_rgb_array."""
    orig_h, orig_w = shape[:2]
    gain = min(target_h / max(orig_h, 1), target_w / max(orig_w, 1))
    new_w = max(int(round(orig_w * gain)), 1)
    new_h = max(int(round(orig_h * gain)), 1)
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    return orig_h, orig_w, gain, new_h, new_w, pad_h, pad_w


class FastPredictor:
    """Standalone fast wrapper returning ``[N, 6]`` arrays.

    Rows are ``x1, y1, x2, y2, score, class_id`` in original-frame pixel
    coordinates, sorted by descending score.  Accepts a loaded
    ``Fotonet``, a raw ``nn.Module`` deploy graph (``[B, 8400, nc + 4]``
    output, normalized cxcywh + logits), or a checkpoint path.
    """

    def __init__(self, model_or_checkpoint, device="cuda", imgsz=640,
                 half=None, conf=0.25, max_det=300):
        from fotonet import Fotonet

        self.device = torch.device(device)
        if isinstance(model_or_checkpoint, Fotonet):
            fotonet = model_or_checkpoint
            self.model = fotonet.model
            self.names = fotonet.names
            self.nc = int(fotonet.nc)
        elif isinstance(model_or_checkpoint, torch.nn.Module):
            self.model = model_or_checkpoint
            self.names = {}
            self.nc = int(getattr(self.model, "nc", 80))
        else:
            fotonet = Fotonet(model_or_checkpoint, device=device)
            self.model = fotonet.model
            self.names = fotonet.names
            self.nc = int(fotonet.nc)

        self.model.to(self.device).eval()
        if half is not None:
            self.model.half() if half and self.device.type == "cuda" else self.model.float()
        param = next(self.model.parameters(), None)
        self.dtype = param.dtype if param is not None else torch.float32
        self.conf = float(conf)
        self.max_det = int(max_det)
        th, tw = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        self.target_h, self.target_w = int(th), int(tw)

        # Reusable staging: one heap canvas + one pinned host buffer, sized
        # once.  The pinned buffer is only overwritten after the previous
        # frame's ``.cpu()`` has synchronized the stream.
        self._canvas = np.full((self.target_h, self.target_w, 3), 114,
                               dtype=np.uint8)
        self._pinned = (
            torch.empty_like(torch.from_numpy(self._canvas)).pin_memory()
            if self.device.type == "cuda" else None
        )

    @torch.inference_mode()
    def predict(self, frame_bgr, conf=None):
        import cv2

        conf_thresh = self.conf if conf is None else float(conf)
        orig_h, orig_w, gain, new_h, new_w, pad_h, pad_w = _letterbox_params(
            frame_bgr.shape, self.target_h, self.target_w)

        self._canvas.fill(114)
        # resize writes straight into the canvas view — no intermediate copy
        cv2.resize(frame_bgr, (new_w, new_h),
                   dst=self._canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w],
                   interpolation=cv2.INTER_LINEAR)

        host = torch.from_numpy(self._canvas)
        if self._pinned is not None:
            self._pinned.copy_(host)
            host = self._pinned
        gpu = host.to(self.device, non_blocking=True)

        # BGR uint8 HWC -> RGB NCHW on the device; normalize in fp32 first so
        # pixel values are bit-identical to the portable CPU path before the
        # model-dtype cast (dividing inside fp16 shifts the last bit).
        tensor = gpu.permute(2, 0, 1).unsqueeze(0)[:, [2, 1, 0]]
        tensor = tensor.to(torch.float32).div_(255.0).to(self.dtype)

        out = self.model(tensor)
        pred_logits = out[:, :, :self.nc].squeeze(0)
        pred_boxes = out[0, :, self.nc:]

        # Official two-stage NMS-free selection (allows two classes on one
        # anchor), fused on the GPU.
        probs = pred_logits.sigmoid()
        num_anchors, num_classes = probs.shape
        topk = min(self.max_det, num_anchors)
        _, anchor_idx = probs.amax(-1).topk(topk)
        scores, pair_idx = probs[anchor_idx].flatten().topk(topk)
        rows = anchor_idx[pair_idx.div(num_classes, rounding_mode="floor")]
        classes = pair_idx.remainder(num_classes)
        mask = scores > conf_thresh
        rows, scores, classes = rows[mask], scores[mask], classes[mask]
        boxes = pred_boxes[rows]

        if boxes.numel() == 0:
            return np.zeros((0, 6), dtype=np.float32)

        # Fused un-letterbox to original-frame pixels
        cx = boxes[:, 0] * float(self.target_w)
        cy = boxes[:, 1] * float(self.target_h)
        bw = boxes[:, 2] * float(self.target_w)
        bh = boxes[:, 3] * float(self.target_h)
        x1 = ((cx - bw * 0.5 - float(pad_w)) / float(gain)).clamp(0, float(orig_w))
        y1 = ((cy - bh * 0.5 - float(pad_h)) / float(gain)).clamp(0, float(orig_h))
        x2 = ((cx + bw * 0.5 - float(pad_w)) / float(gain)).clamp(0, float(orig_w))
        y2 = ((cy + bh * 0.5 - float(pad_h)) / float(gain)).clamp(0, float(orig_h))

        dets = torch.stack([x1, y1, x2, y2, scores, classes.float()], dim=-1)
        # .cpu() synchronizes the stream before the pinned buffer is reused
        return dets.cpu().numpy()
