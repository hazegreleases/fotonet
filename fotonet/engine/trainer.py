"""
FOTONET Trainer — fast, production-grade training loop.

Speed features:
  - Thread-based prefetcher: loads next batch while GPU trains current one (safe on Windows)
  - torch.compile() for GPU kernel fusion (PyTorch 2.0+)
  - Gradient accumulation (nominal batch size controlled by nbs)
  - Mixed-precision (AMP) with GradScaler
  - Cosine LR decay with linear warmup
  - Auto-save best checkpoint

Fixes applied:
  - torch.compile(mode="default") — no CUDA graph tracing warmup lag
  - Removed spurious optimizer.step() before training
  - Removed torch.cuda.empty_cache() — lets caching allocator work
  - Simplified collate_fn — no per-image interpolation
  - Fixed gradient accumulation remainder detection
  - Warmup starts at 1% LR (not 0.1%)
  - Ramping EMA decay for faster early convergence
  - DFL-compatible loss dict keys
"""
import gc
import math
import os
import json
import time
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from fotonet.engine.runtime import EMA, EpochCutSampler as _EpochCutSampler, ThreadPrefetcher as _ThreadPrefetcher
from fotonet.utils.loss import get_loss
from tqdm import tqdm


class Trainer:
    def __init__(self, model, data_cfg, epochs=100, imgsz=640,
                 lr0=0.001, lrf=0.01, batch=16, val_batch=None, nbs=128,
                 warmup_epochs=3, save_dir=".", val_split=0.2, val_period=1,
                 workers=None, pin_memory=True, cache_to_ram=True, ram_cache_images=1024, amp=True, val_amp=None, cos_lr=True,
                 cuda_graphs=False, profile=False, resume_ckpt=None, epoch_cut=1, compile_model=False,
                 save_period=-1, slim_best=True, best_metric="mAP50_95", save_last=True,
                 momentum=0.937, weight_decay=0.0005,
                 augment_hyp=None, loss_hyp=None, matcher_hyp=None,
                 imgsz_schedule=None, val_subset_size=0, full_val_after=1.0,
                 cache_labels=True, disk_cache_images=False, disk_cache_dir=None,
                 distill_teacher=None, distill_weight=0.0,
                 distill_start_epoch=1, distill_warmup_epochs=1,
                 distill_end_epoch=None, distill_conf=0.25, distill_topk=64):
        self.model         = model
        self.epochs        = epochs
        self.imgsz         = int(imgsz)
        self.imgsz_schedule = self._normalize_imgsz_schedule(imgsz_schedule, int(imgsz))
        self.current_imgsz = int(imgsz)
        self.lr0           = lr0
        self.lrf           = lrf
        self.batch_size    = batch
        self.val_batch_size = batch if val_batch is None else int(val_batch)
        self.nbs           = nbs
        self.warmup_epochs = warmup_epochs
        self.save_dir      = save_dir
        self.nc            = model.head.nc
        self.data_cfg      = data_cfg
        self.val_period    = max(1, val_period)
        self.epoch_cut     = max(1, int(epoch_cut))

        # Hardware parameters
        self.workers       = workers if workers is not None else max(os.cpu_count() - 2, 1)
        self.pin_memory    = pin_memory and (next(model.parameters()).device.type == "cuda")
        self.cache_to_ram  = cache_to_ram
        self.ram_cache_images = ram_cache_images
        self.use_amp       = amp
        self.val_amp       = amp if val_amp is None else bool(val_amp)
        self.cos_lr        = cos_lr
        self.cuda_graphs   = cuda_graphs
        self.profile       = profile
        self.compile_model = compile_model
        self.save_period   = int(save_period)
        self.slim_best     = slim_best
        self.best_metric   = best_metric
        self.save_last     = bool(save_last)
        self.momentum      = momentum
        self.weight_decay  = weight_decay
        self.augment_hyp   = augment_hyp or {}
        self.loss_hyp      = loss_hyp or {}
        self.matcher_hyp   = matcher_hyp or {}
        self.val_subset_size = max(int(val_subset_size or 0), 0)
        self.full_val_after = float(full_val_after if full_val_after is not None else 1.0)
        self.cache_labels = bool(cache_labels)
        self.disk_cache_images = bool(disk_cache_images)
        self.disk_cache_dir = disk_cache_dir
        self.distill_teacher_path = distill_teacher
        self.distill_weight = float(distill_weight or 0.0)
        self.distill_start_epoch = int(distill_start_epoch or 1)
        self.distill_warmup_epochs = max(int(distill_warmup_epochs or 1), 1)
        self.distill_end_epoch = None if distill_end_epoch is None else int(distill_end_epoch)
        self.distill_conf = float(distill_conf)
        self.distill_topk = int(distill_topk)
        
        # Compile model — use "default" mode to avoid CUDA graph capture warmup lag
        if self.compile_model and torch.cuda.is_available():
            try:
                print("[INFO] Compiling model with torch.compile(mode='default')...")
                self.model = torch.compile(self.model, mode="default")
                print("[INFO] Model compiled successfully!")
            except Exception as e:
                print(f"[WARN] torch.compile() failed, using uncompiled model: {e}")

        self.val_split     = val_split if val_split is not None else 0.2

        self.device = next(self.model.parameters()).device
        
        # Get original model (for optimizer, EMA, etc.)
        original_model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        
        # SGD with decoupled weight decay
        g_bnw, g_w, g_b = self._split_optimizer_params(original_model)

        self.optimizer = optim.SGD(g_bnw, lr=lr0, momentum=momentum, nesterov=True)
        self.optimizer.add_param_group({'params': g_w, 'weight_decay': weight_decay})
        self.optimizer.add_param_group({'params': g_b})
        self.accum_steps = max(round(nbs / batch), 1)

        class_weights = self.data_cfg.get("class_weights", None) if isinstance(self.data_cfg, dict) else None
        self.criterion = get_loss(
            original_model,
            self.nc,
            class_weights,
            loss_hyp=self.loss_hyp,
            matcher_hyp=self.matcher_hyp,
        )

        # Estimate total optimizer steps for EMA ramp
        # Rough estimate: ~6500 steps/epoch for COCO, * epochs
        est_steps_per_epoch = max(100000 // max(batch, 1), 100)
        total_ema_steps = est_steps_per_epoch * epochs

        self.ema = EMA(original_model, decay_start=0.99, decay_end=0.9999,
                       total_steps=total_ema_steps)

        self.scaler    = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.best_loss = float("inf")
        self.best_map  = 0.0
        self.start_epoch = 0
        self.global_step = 0
        self.resume_ckpt = resume_ckpt
        
        # Persistent dataset and loader references for reuse
        self.full_train_set = None
        self.chunk_indices = []
        self.train_loader = None
        self.train_prefetcher = None
        self.val_dataset = None
        self.full_val_dataset = None
        self.active_val_dataset = None
        self._val_subset_indices = None
        self._coco_evaluator_cache = None
        self.distill_teacher = None

        # Restore full training state from a resume checkpoint
        if resume_ckpt is not None:
            self.start_epoch = resume_ckpt.get("epoch", 0)
            ckpt_metric = resume_ckpt.get("best_metric", "mAP50")
            if ckpt_metric == self.best_metric:
                self.best_map = resume_ckpt.get("best_score", resume_ckpt.get("best_map", resume_ckpt.get("mAP", 0.0)))
            elif self.best_metric == "mAP50":
                self.best_map = resume_ckpt.get("best_map", resume_ckpt.get("mAP", 0.0))
            else:
                self.best_map = 0.0
            if "optimizer_state" in resume_ckpt:
                try:
                    self.optimizer.load_state_dict(resume_ckpt["optimizer_state"])
                    self._move_optimizer_state_to_device()
                    print(f"[INFO] Restored optimizer state.")
                except Exception as e:
                    print(f"[WARN] Failed to restore optimizer state (architecture change?): {e}")
            
            if "scaler_state" in resume_ckpt:
                self.scaler.load_state_dict(resume_ckpt["scaler_state"])
                print(f"[INFO] Restored AMP scaler state.")
            
            if "ema_state" in resume_ckpt:
                try:
                    self.ema.ema.load_state_dict(resume_ckpt["ema_state"])
                    self.ema.step_count = int(resume_ckpt.get("ema_updates", self.ema.step_count))
                    print(f"[INFO] Restored EMA state.")
                except Exception as e:
                    print(f"[WARN] Failed to restore EMA state: {e}")
            
            print(f"[INFO] Resuming from epoch {self.start_epoch} | best_{self.best_metric}={self.best_map:.4f}")
            self.global_step = int(resume_ckpt.get("global_step", self.global_step))

        # Setup val logging path
        now = datetime.now()
        log_dir = os.path.join(save_dir, "logs", now.strftime("train-%Y/%m/%d/%H/%M"))
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "metrics.json")
        self.live_status_file = os.path.join(log_dir, "live_status.json")
        self.preview_dir = log_dir
        self.preview_interval_sec = 60.0
        self._last_preview_at = 0.0
        self._preview_index = 0
        self._last_iter_rate_step = None
        self._last_iter_rate_at = None
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("[]")  # Init as empty list
        self._write_live_status({
            "running": False,
            "epoch": int(self.start_epoch),
            "epochs": int(self.epochs),
            "timestamp": datetime.now().isoformat(),
        })

        self._setup_distillation_teacher()

    @staticmethod
    def _split_optimizer_params(model):
        """Return no-decay, decay, and bias parameter groups without dropping direct Parameters."""
        g_bnw, g_w, g_b = [], [], []
        seen = set()
        norm_types = (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.SyncBatchNorm)

        for module in model.modules():
            for name, param in module.named_parameters(recurse=False):
                if not param.requires_grad or id(param) in seen:
                    continue
                seen.add(id(param))
                if name == "bias":
                    g_b.append(param)
                elif isinstance(module, norm_types) or param.ndim <= 1:
                    g_bnw.append(param)
                else:
                    g_w.append(param)

        return g_bnw, g_w, g_b

    @staticmethod
    def _assert_finite_loss_dict(loss_dict, epoch, step, global_step):
        bad = []
        for name, value in loss_dict.items():
            if not str(name).startswith("loss"):
                continue
            if torch.is_tensor(value):
                finite = torch.isfinite(value.detach()).all()
                if bool(finite):
                    continue
                if value.numel() == 1:
                    shown = float(value.detach().float().cpu())
                else:
                    shown = f"shape={tuple(value.shape)}"
                bad.append(f"{name}={shown}")
            elif isinstance(value, (float, int)) and not math.isfinite(float(value)):
                bad.append(f"{name}={float(value)}")
        if bad:
            raise FloatingPointError(
                "Non-finite loss at "
                f"epoch {int(epoch) + 1} step {int(step) + 1} global_step {int(global_step)}: "
                + ", ".join(bad)
            )

    @staticmethod
    def _normalize_imgsz_schedule(schedule, default_imgsz):
        if not schedule:
            return [(1.0, int(default_imgsz))]
        pairs = []
        for item in schedule:
            if isinstance(item, dict):
                frac = item.get("fraction", item.get("until", item.get("pct")))
                size = item.get("imgsz", item.get("size"))
            else:
                frac, size = item
            pairs.append((float(frac), int(size)))
        total = sum(frac for frac, _ in pairs)
        if total <= 1.0001:
            out = []
            acc = 0.0
            for frac, size in pairs:
                acc += frac
                out.append((min(acc, 1.0), size))
            out[-1] = (1.0, out[-1][1])
            return out
        out = [(min(max(frac, 0.0), 1.0), size) for frac, size in pairs]
        out = sorted(out, key=lambda x: x[0])
        if out[-1][0] < 1.0:
            out.append((1.0, out[-1][1]))
        return out

    def _imgsz_for_epoch(self, epoch):
        progress = (int(epoch) + 1) / max(int(self.epochs), 1)
        for limit, size in self.imgsz_schedule:
            if progress <= limit + 1e-9:
                return int(size)
        return int(self.imgsz_schedule[-1][1])

    def _imgsz_phase_for_epoch(self, epoch):
        progress = (int(epoch) + 1) / max(int(self.epochs), 1)
        for index, (limit, _size) in enumerate(self.imgsz_schedule, start=1):
            if progress <= limit + 1e-9:
                return index, len(self.imgsz_schedule)
        return len(self.imgsz_schedule), len(self.imgsz_schedule)

    def _write_live_status(self, payload):
        path = getattr(self, "live_status_file", None)
        if not path:
            return
        if hasattr(self, "best_metric") and hasattr(self, "best_map"):
            payload = dict(payload)
            payload.setdefault("best_metric", self.best_metric)
            payload.setdefault("best_score", float(self.best_map))
            payload.setdefault(f"best_{self.best_metric}", float(self.best_map))
        try:
            tmp_path = f"{path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp_path, path)
        except OSError:
            pass

    def _live_status_payload(self, epoch, step, total_steps, running, n_steps, extra=None, running_flag=True, elapsed_sec=None):
        denom = max(int(n_steps), 1)
        cut_total = max(int(getattr(self, "epoch_cut", 1) or 1), 1)
        cut_index = ((int(epoch) % cut_total) + 1) if cut_total > 1 else None
        has_steps = int(n_steps) > 0
        box_loss = round(float(running["box"]) / denom, 4) if has_steps else None
        dfl_loss = round(float(running["dflpc"]) / denom, 4) if has_steps else None
        payload = {
            "running": bool(running_flag),
            "epoch": int(epoch) + 1,
            "epochs": int(self.epochs),
            "step": int(step),
            "steps": int(total_steps),
            "epoch_progress": float(min(max(step / max(total_steps, 1), 0.0), 1.0)),
            "imgsz": int(self.current_imgsz),
            "cut": int(cut_index) if cut_index is not None else None,
            "cut_total": int(cut_total),
            "loss": round(float(running["loss"]) / denom, 4) if has_steps else None,
            "cls_loss": round(float(running["cls"]) / denom, 4) if has_steps else None,
            "box_loss": box_loss,
            "dfl_loss": dfl_loss,
            "iou_loss": round(float(running["iou"]) / denom, 4) if has_steps else None,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "global_step": int(self.global_step),
            "timestamp": datetime.now().isoformat(),
        }
        if has_steps and elapsed_sec and elapsed_sec > 0:
            payload["iter_rate"] = self._recent_iter_rate(n_steps, elapsed_sec)
        if extra:
            payload.update(extra)
        return payload

    def _recent_iter_rate(self, n_steps, elapsed_sec):
        fallback = float(n_steps) / max(float(elapsed_sec), 1e-9)
        now = time.time()
        current_step = int(getattr(self, "global_step", 0))
        last_step = getattr(self, "_last_iter_rate_step", None)
        last_at = getattr(self, "_last_iter_rate_at", None)
        rate = fallback
        if last_step is not None and last_at is not None:
            delta_steps = current_step - int(last_step)
            delta_time = now - float(last_at)
            if delta_steps > 0 and delta_time > 1e-9:
                rate = float(delta_steps) / delta_time
        self._last_iter_rate_step = current_step
        self._last_iter_rate_at = now
        return float(rate)

    def _maybe_write_run_previews(self, imgs, targets, outputs, now=None):
        now = time.time() if now is None else float(now)
        if now - getattr(self, "_last_preview_at", 0.0) < self.preview_interval_sec:
            return
        self._last_preview_at = now
        try:
            n_preview = min(int(imgs.shape[0]), 6)
            for slot in range(n_preview):
                self._write_prediction_preview(imgs, targets, outputs, slot=slot, sample_index=slot)
                self._write_feature_preview(outputs, slot=slot, sample_index=slot)
            self._preview_index += n_preview
        except Exception:
            pass

    @staticmethod
    def _xywhn_to_xyxy_px(boxes, width, height):
        if boxes.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        boxes = np.nan_to_num(boxes.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
        x1 = (boxes[:, 0] - boxes[:, 2] * 0.5) * width
        y1 = (boxes[:, 1] - boxes[:, 3] * 0.5) * height
        x2 = (boxes[:, 0] + boxes[:, 2] * 0.5) * width
        y2 = (boxes[:, 1] + boxes[:, 3] * 0.5) * height
        out = np.stack(
            (
                np.minimum(x1, x2),
                np.minimum(y1, y2),
                np.maximum(x1, x2),
                np.maximum(y1, y2),
            ),
            axis=1,
        ).astype(np.float32, copy=False)
        out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, width - 1)
        out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, height - 1)
        return out

    @staticmethod
    def _draw_label(draw, xy, text, fill, font):
        x, y = xy
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        y0 = max(0, y - th - 6)
        draw.rectangle((x, y0, x + tw + 8, y0 + th + 6), fill=fill)
        draw.text((x + 4, y0 + 3), text, fill=(255, 255, 255), font=font)

    def _write_prediction_preview(self, imgs, targets, outputs, slot=0, sample_index=0):
        from PIL import Image, ImageDraw, ImageFont

        image = imgs[int(sample_index)].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        image = (image * 255.0).round().astype(np.uint8)
        base_canvas = Image.fromarray(image, mode="RGB")
        canvas = base_canvas.copy()
        gt_canvas = base_canvas.copy()
        pred_canvas = base_canvas.copy()
        draw = ImageDraw.Draw(canvas)
        gt_draw = ImageDraw.Draw(gt_canvas)
        pred_draw = ImageDraw.Draw(pred_canvas)
        font = ImageFont.load_default()
        height, width = image.shape[:2]

        target = targets[int(sample_index)] if targets and int(sample_index) < len(targets) else {}
        gt_boxes = target.get("boxes")
        gt_labels = target.get("labels")
        if gt_boxes is not None:
            gt_boxes_np = gt_boxes.detach().float().cpu().numpy()
            gt_labels_np = gt_labels.detach().cpu().numpy() if gt_labels is not None else np.zeros((len(gt_boxes_np),), dtype=np.int64)
            for box, label in zip(self._xywhn_to_xyxy_px(gt_boxes_np, width, height)[:64], gt_labels_np[:64]):
                x1, y1, x2, y2 = [int(v) for v in box]
                if x2 <= x1 or y2 <= y1:
                    continue
                for layer_draw in (draw, gt_draw):
                    layer_draw.rectangle((x1, y1, x2, y2), outline=(76, 175, 80), width=3)
                    self._draw_label(layer_draw, (x1, y1), f"gt {int(label)}", (76, 175, 80), font)

        logits = outputs["pred_logits_o2o"][int(sample_index)].detach().float()
        boxes = outputs["pred_boxes_o2o"][int(sample_index)].detach().float()
        scores, classes = logits.sigmoid().max(-1)
        topk = min(24, int(scores.numel()))
        if topk > 0:
            top_scores, top_idx = torch.topk(scores, topk)
            keep = top_scores > 0.05
            if not keep.any():
                keep[: min(6, topk)] = True
            pred_boxes = boxes[top_idx[keep]].cpu().numpy()
            pred_scores = top_scores[keep].cpu().numpy()
            pred_classes = classes[top_idx[keep]].cpu().numpy()
            for box, score, cls in zip(self._xywhn_to_xyxy_px(pred_boxes, width, height), pred_scores, pred_classes):
                x1, y1, x2, y2 = [int(v) for v in box]
                if x2 <= x1 or y2 <= y1:
                    continue
                for layer_draw in (draw, pred_draw):
                    layer_draw.rectangle((x1, y1, x2, y2), outline=(3, 169, 244), width=3)
                    self._draw_label(layer_draw, (x1, y1), f"p {int(cls)} {float(score):.2f}", (3, 169, 244), font)

        layers = {
            "overview": canvas,
            "base": base_canvas,
            "gt": gt_canvas,
            "pred": pred_canvas,
        }
        for name, layer in layers.items():
            path = os.path.join(self.preview_dir, f"training_prediction_{name}_{int(slot):02d}.png")
            tmp_path = f"{path}.tmp.png"
            layer.save(tmp_path)
            os.replace(tmp_path, path)

    @staticmethod
    def _heatmap_rgb(gray):
        x = gray.astype(np.float32) / 255.0
        red = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
        green = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
        blue = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
        return (np.stack([red, green, blue], axis=-1) * 255.0).round().astype(np.uint8)

    def _write_feature_preview(self, outputs, slot=0, sample_index=0):
        from PIL import Image

        size_value = outputs.get("imgsz", self.current_imgsz)
        size = int(size_value.detach().cpu().item()) if torch.is_tensor(size_value) else int(size_value)
        preview_features = outputs.get("preview_features")
        if preview_features is not None:
            heat = preview_features[int(sample_index)].detach().float().abs().mean(0, keepdim=True).unsqueeze(0)
            heat = F.interpolate(heat, size=(size, size), mode="bilinear", align_corners=False)[0, 0]
        else:
            logits = outputs["pred_logits_o2o"][int(sample_index)].detach().float()
            scores = logits.sigmoid().amax(-1)
            shapes = outputs.get("feat_shapes")
            if shapes is None:
                return
            shapes = [(int(h), int(w)) for h, w in shapes.detach().cpu().tolist()]
            heat = None
            cursor = 0
            for h, w in shapes:
                count = h * w
                if cursor + count > scores.numel():
                    break
                level = scores[cursor:cursor + count].reshape(1, 1, h, w)
                level = F.interpolate(level, size=(size, size), mode="bilinear", align_corners=False)[0, 0]
                heat = level if heat is None else torch.maximum(heat, level)
                cursor += count
        if heat is None:
            return
        heat = heat - heat.min()
        heat = heat / heat.max().clamp_min(1e-6)
        gray = (heat.clamp(0, 1) * 255.0).byte().cpu().numpy()
        image = Image.fromarray(self._heatmap_rgb(gray), mode="RGB")
        path = os.path.join(self.preview_dir, f"training_feature_map_{int(slot):02d}.png")
        tmp_path = f"{path}.tmp.png"
        image.save(tmp_path)
        os.replace(tmp_path, path)

    def _set_dataset_imgsz(self, dataset, imgsz):
        if dataset is None:
            return
        if hasattr(dataset, "set_imgsz"):
            dataset.set_imgsz(imgsz)
        elif hasattr(dataset, "dataset"):
            self._set_dataset_imgsz(dataset.dataset, imgsz)

    def _make_train_loader(self, train_sampler, train_shuffle, num_workers, train_pf_factor):
        return DataLoader(
            self.full_train_set, batch_size=self.batch_size, shuffle=train_shuffle,
            sampler=train_sampler,
            collate_fn=self._collate_fn, num_workers=num_workers,
            pin_memory=self.pin_memory, prefetch_factor=train_pf_factor,
            persistent_workers=True if num_workers > 0 else False,
        )

    def _rebuild_train_prefetcher(self, train_sampler, train_shuffle, num_workers, train_pf_factor):
        if self.train_prefetcher is not None:
            self.train_prefetcher.shutdown()
        self.train_loader = self._make_train_loader(train_sampler, train_shuffle, num_workers, train_pf_factor)
        self.train_prefetcher = _ThreadPrefetcher(
            self.train_loader,
            device=self.device,
            queue_size=4 if self.epoch_cut > 1 else 2,
        )
        return self.train_prefetcher

    def _train_augment_hyp(self):
        dataset = self.full_train_set
        seen = set()
        while dataset is not None and id(dataset) not in seen:
            seen.add(id(dataset))
            hyp = getattr(dataset, "augment_hyp", None)
            if isinstance(hyp, dict):
                return hyp
            dataset = getattr(dataset, "dataset", None)
        return {}

    def _close_mosaic_boundary(self):
        try:
            close_mosaic = int(self._train_augment_hyp().get("close_mosaic", 0) or 0)
        except (TypeError, ValueError):
            close_mosaic = 0
        if close_mosaic <= 0:
            return None
        return max(int(self.epochs) - close_mosaic, 0)

    def _init_val_subset(self, val_dataset):
        n_val = len(val_dataset)
        if self.val_subset_size <= 0 or self.val_subset_size >= n_val:
            self._val_subset_indices = None
            return
        generator = torch.Generator()
        generator.manual_seed(2026)
        self._val_subset_indices = torch.randperm(n_val, generator=generator)[:self.val_subset_size].tolist()

    def _val_dataset_for_epoch(self, epoch):
        is_last = int(epoch) == self.epochs - 1
        progress = (int(epoch) + 1) / max(int(self.epochs), 1)
        force_full_last = is_last and self.full_val_after <= 1.0
        use_full = force_full_last or self._val_subset_indices is None or progress >= self.full_val_after
        if use_full:
            return self.full_val_dataset, "full"
        return Subset(self.full_val_dataset, self._val_subset_indices), f"subset{len(self._val_subset_indices)}"

    def _setup_distillation_teacher(self):
        if not self.distill_teacher_path or self.distill_weight <= 0:
            return
        if not os.path.exists(str(self.distill_teacher_path)):
            print(f"[distill] teacher missing; disabled: {self.distill_teacher_path}")
            return
        try:
            from fotonet import FOTONET
            teacher_api = FOTONET(str(self.distill_teacher_path))
            self.distill_teacher = teacher_api.model.to(self.device).eval()
            for p in self.distill_teacher.parameters():
                p.requires_grad_(False)
            print(f"[distill] teacher={self.distill_teacher_path} weight={self.distill_weight:g}")
        except Exception as exc:
            self.distill_teacher = None
            print(f"[distill] failed to load teacher; disabled: {exc}")

    def _distill_factor(self, epoch):
        if self.distill_teacher is None or self.distill_weight <= 0:
            return 0.0
        epoch1 = int(epoch) + 1
        if epoch1 < self.distill_start_epoch:
            return 0.0
        if self.distill_end_epoch is not None and epoch1 > self.distill_end_epoch:
            return 0.0
        ramp = min(1.0, (epoch1 - self.distill_start_epoch + 1) / max(self.distill_warmup_epochs, 1))
        return self.distill_weight * max(ramp, 0.0)

    def _distillation_loss(self, outputs, imgs, epoch):
        factor = self._distill_factor(epoch)
        if factor <= 0:
            return outputs["pred_logits_o2o"].new_tensor(0.0)

        from fotonet.utils.boxes import box_giou, xywh_to_xyxy
        from fotonet.utils.loss import bbox_ciou

        with torch.no_grad():
            teacher = self.distill_teacher(imgs)
            if isinstance(teacher, dict):
                t_logits = teacher["pred_logits_o2o"]
                t_boxes = teacher["pred_boxes_o2o"]
            else:
                t_logits = teacher[..., :self.nc]
                t_boxes = teacher[..., self.nc:]
            t_scores, t_labels = t_logits.sigmoid().max(-1)

        s_logits = outputs["pred_logits_o2o"]
        s_boxes = outputs["pred_boxes_o2o"]
        losses = []

        for b in range(s_logits.shape[0]):
            scores = t_scores[b]
            keep = scores > self.distill_conf
            if not keep.any():
                continue
            boxes = t_boxes[b][keep].detach().float()
            labels = t_labels[b][keep].detach()
            q_scores = scores[keep].detach().float()
            if self.distill_topk > 0 and q_scores.numel() > self.distill_topk:
                q_scores, idx = torch.topk(q_scores, self.distill_topk)
                boxes = boxes[idx]
                labels = labels[idx]

            pred_logits = s_logits[b].float()
            pred_boxes = s_boxes[b].float()
            probs = pred_logits.sigmoid()
            cost_cls = -probs[:, labels]
            cost_l1 = torch.cdist(pred_boxes, boxes, p=1)
            cost_iou = -box_giou(xywh_to_xyxy(pred_boxes), xywh_to_xyxy(boxes))
            cost = torch.nan_to_num(cost_cls + 2.0 * cost_l1 + cost_iou, nan=1e6, posinf=1e6, neginf=-1e6)
            p_idx = cost.argmin(dim=0)

            cls_loss = F.binary_cross_entropy_with_logits(
                pred_logits[p_idx, labels],
                q_scores.clamp(0.01, 0.99),
                reduction="mean",
            )
            box_loss = (1.0 - bbox_ciou(
                xywh_to_xyxy(pred_boxes[p_idx]),
                xywh_to_xyxy(boxes),
            )).mean()
            losses.append(cls_loss + 2.0 * box_loss)

        if not losses:
            return s_logits.new_tensor(0.0)
        return s_logits.new_tensor(float(factor)) * torch.stack(losses).mean()

    def _print_train_header(self, n_train, n_val):
        schedule_txt = " -> ".join(
            f"{int(limit * 100)}%:{size}" for limit, size in self.imgsz_schedule
        )
        val_txt = "full"
        if self.val_subset_size > 0:
            if self.full_val_after > 1.0:
                val_txt = f"subset={self.val_subset_size}"
            else:
                val_txt = f"subset={self.val_subset_size} until {int(self.full_val_after * 100)}%, then full"
        print(
            "\n"
            "FOTO-NET TRAIN\n"
            "-------------\n"
            f"images: train={n_train} val={n_val} | imgsz_schedule={schedule_txt}\n"
            f"validation: {val_txt}\n"
            f"batch: train={self.batch_size} val={self.val_batch_size} | nbs={self.nbs} accum={self.accum_steps}\n"
            f"optim: lr0={self.lr0:g} lrf={self.lrf:g} momentum={self.momentum:g} weight_decay={self.weight_decay:g}\n"
            f"amp: train={self.use_amp} val={self.val_amp} | cache_to_ram={self.cache_to_ram}\n"
        )

    @staticmethod
    def _state_dict_for_save(model, half=False, strip_o2m=False):
        state = model.state_dict()
        if not half and not strip_o2m:
            return state
        slim_state = {}
        for k, v in state.items():
            if strip_o2m and (
                k.startswith("head.cls_o2m.")
                or k.startswith("head.reg_o2m.")
                or k.startswith("head.quality_o2m.")
            ):
                continue
            if not torch.is_tensor(v):
                slim_state[k] = v
                continue
            v = v.detach().cpu()
            slim_state[k] = v.half() if v.dtype.is_floating_point else v
        return slim_state

    def _raw_model(self):
        return self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model

    def _checkpoint_model_meta(self):
        raw = self._raw_model()
        return {
            "w": getattr(raw, "w", getattr(raw, "width_multiple", None)),
            "d": getattr(raw, "d", getattr(raw, "depth_multiple", None)),
            "use_p2": getattr(raw, "use_p2", False),
            "reg_max": getattr(raw, "reg_max", getattr(getattr(raw, "head", None), "reg_max", 16)),
            "p3_extra_blocks": getattr(raw, "p3_extra_blocks", 0),
            "p4_extra_blocks": getattr(raw, "p4_extra_blocks", 0),
            "p5_extra_blocks": getattr(raw, "p5_extra_blocks", 0),
            "p5_gate_blocks": getattr(raw, "p5_gate_blocks", 0),
            "arch_version": getattr(raw, "arch_version", 1),
            "neck_fusion": getattr(raw, "neck_fusion", "concat"),
            "p2_context_blocks": getattr(raw, "p2_context_blocks", 0),
            "p3_context_blocks": getattr(raw, "p3_context_blocks", 0),
            "quality_head": getattr(raw, "quality_head", False),
        }

    def _restore_scheduler_lr(self, scheduler):
        """LambdaLR state restores counters; explicitly sync optimizer LRs too."""
        for group, lr in zip(self.optimizer.param_groups, scheduler.get_last_lr()):
            group["lr"] = lr

    def _move_optimizer_state_to_device(self):
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device, non_blocking=True)

    def _save_last_checkpoint(self, scheduler, epoch):
        model_meta = self._checkpoint_model_meta()
        torch.save({
            "model":           self._raw_model().state_dict(),
            "ema_state":       self.ema.ema.state_dict(),
            "ema_updates":     self.ema.step_count,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state":    self.scaler.state_dict(),
            "nc":              self.nc,
            "epoch":           epoch + 1,
            "global_step":     self.global_step,
            "best_map":        self.best_map,
            "best_metric":     self.best_metric,
            "best_score":      self.best_map,
            "mAP":             self.best_map,
            "w":               model_meta["w"],
            "d":               model_meta["d"],
            "use_p2":           model_meta["use_p2"],
            "reg_max":          model_meta["reg_max"],
            "p3_extra_blocks":   model_meta["p3_extra_blocks"],
            "p4_extra_blocks":   model_meta["p4_extra_blocks"],
            "p5_extra_blocks":   model_meta["p5_extra_blocks"],
            "p5_gate_blocks":    model_meta["p5_gate_blocks"],
            "arch_version":      model_meta["arch_version"],
            "neck_fusion":       model_meta["neck_fusion"],
            "p2_context_blocks": model_meta["p2_context_blocks"],
            "p3_context_blocks": model_meta["p3_context_blocks"],
            "quality_head":      model_meta["quality_head"],
        }, os.path.join(self.save_dir, "fotonet_last.pt"))

    def _save_best_checkpoint(self, epoch):
        model_meta = self._checkpoint_model_meta()
        if self.slim_best:
            ckpt = {
                "model":    self._state_dict_for_save(self.ema.ema, half=True, strip_o2m=True),
                "nc":       self.nc,
                "epoch":    epoch + 1,
                "global_step": self.global_step,
                "best_map": self.best_map,
                "best_metric": self.best_metric,
                "best_score": self.best_map,
                "mAP":      self.best_map,
                "w":        model_meta["w"],
                "d":        model_meta["d"],
                "use_p2":    model_meta["use_p2"],
                "reg_max":   model_meta["reg_max"],
                "p3_extra_blocks": model_meta["p3_extra_blocks"],
                "p4_extra_blocks": model_meta["p4_extra_blocks"],
                "p5_extra_blocks": model_meta["p5_extra_blocks"],
                "p5_gate_blocks": model_meta["p5_gate_blocks"],
                "arch_version": model_meta["arch_version"],
                "neck_fusion": model_meta["neck_fusion"],
                "p2_context_blocks": model_meta["p2_context_blocks"],
                "p3_context_blocks": model_meta["p3_context_blocks"],
                "quality_head": model_meta["quality_head"],
                "inference_only": True,
                "stripped_o2m": True,
                "has_o2m": False,
                "nms_free": True,
            }
        else:
            ckpt = {
                "model":     self.ema.ema.state_dict(),
                "ema_state": self.ema.ema.state_dict(),
                "nc":        self.nc,
                "epoch":     epoch + 1,
                "global_step": self.global_step,
                "best_map":  self.best_map,
                "best_metric": self.best_metric,
                "best_score": self.best_map,
                "mAP":       self.best_map,
                "w":         model_meta["w"],
                "d":         model_meta["d"],
                "use_p2":     model_meta["use_p2"],
                "reg_max":    model_meta["reg_max"],
                "p3_extra_blocks": model_meta["p3_extra_blocks"],
                "p4_extra_blocks": model_meta["p4_extra_blocks"],
                "p5_extra_blocks": model_meta["p5_extra_blocks"],
                "p5_gate_blocks": model_meta["p5_gate_blocks"],
                "arch_version": model_meta["arch_version"],
                "neck_fusion": model_meta["neck_fusion"],
                "p2_context_blocks": model_meta["p2_context_blocks"],
                "p3_context_blocks": model_meta["p3_context_blocks"],
                "quality_head": model_meta["quality_head"],
                "nms_free": True,
            }
        torch.save(ckpt, os.path.join(self.save_dir, "fotonet_best.pt"))

    def _save_periodic_checkpoint(self, scheduler, epoch):
        if self.save_period <= 0 or (epoch + 1) % self.save_period != 0:
            return
        model_meta = self._checkpoint_model_meta()
        torch.save({
            "model":           self._raw_model().state_dict(),
            "ema_state":       self.ema.ema.state_dict(),
            "ema_updates":     self.ema.step_count,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state":    self.scaler.state_dict(),
            "nc":              self.nc,
            "epoch":           epoch + 1,
            "global_step":     self.global_step,
            "best_map":        self.best_map,
            "best_metric":     self.best_metric,
            "best_score":      self.best_map,
            "mAP":             self.best_map,
            "w":               model_meta["w"],
            "d":               model_meta["d"],
            "use_p2":           model_meta["use_p2"],
            "reg_max":          model_meta["reg_max"],
            "p3_extra_blocks":   model_meta["p3_extra_blocks"],
            "p4_extra_blocks":   model_meta["p4_extra_blocks"],
            "p5_extra_blocks":   model_meta["p5_extra_blocks"],
            "p5_gate_blocks":    model_meta["p5_gate_blocks"],
            "arch_version":      model_meta["arch_version"],
            "neck_fusion":       model_meta["neck_fusion"],
            "p2_context_blocks": model_meta["p2_context_blocks"],
            "p3_context_blocks": model_meta["p3_context_blocks"],
            "quality_head":      model_meta["quality_head"],
        }, os.path.join(self.save_dir, f"fotonet_epoch_{epoch+1}.pt"))

    def _make_val_loader(self, val_dataset, batch_size, num_workers, pf_factor, persistent_workers=True):
        return DataLoader(
            val_dataset, batch_size=int(batch_size), shuffle=False,
            collate_fn=self._collate_fn, num_workers=num_workers,
            pin_memory=self.pin_memory, prefetch_factor=pf_factor,
            persistent_workers=persistent_workers if num_workers > 0 else False
        )

    @staticmethod
    def _is_recoverable_cuda_error(exc):
        msg = str(exc).lower()
        return any(token in msg for token in (
            "out of memory",
            "cudnn_status_execution_failed",
            "cudart",
            "cuda error",
        ))

    def _cleanup_cuda_after_error(self):
        gc.collect()
        if self.device.type == "cuda":
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()

    def _validate_with_retries(self, loader, epoch, num_workers, pf_factor):
        try:
            return self.validate(loader, epoch)
        except RuntimeError as exc:
            if not self._is_recoverable_cuda_error(exc) or self.active_val_dataset is None:
                raise

            print(f"[WARN] Validation hit a recoverable CUDA/cuDNN error: {exc}")
            print("[WARN] Retrying validation with safer settings instead of killing the run.")
            self._cleanup_cuda_after_error()

        original_val_amp = self.val_amp
        fallback_batches = []
        b = max(1, int(self.val_batch_size))
        while b > 1:
            b = max(1, b // 2)
            if b not in fallback_batches:
                fallback_batches.append(b)
        if 1 not in fallback_batches:
            fallback_batches.append(1)

        for batch_size in fallback_batches:
            try:
                self.val_amp = True if self.device.type == "cuda" else original_val_amp
                safe_workers = max(0, min(int(num_workers), 2))
                safe_pf = 1 if safe_workers > 0 else None
                safe_loader = self._make_val_loader(
                    self.active_val_dataset,
                    batch_size=batch_size,
                    num_workers=safe_workers,
                    pf_factor=safe_pf,
                    persistent_workers=False,
                )
                print(
                    f"[INFO] Validation retry: batch={batch_size}, "
                    f"workers={safe_workers}, val_amp={self.val_amp}"
                )
                stats = self.validate(safe_loader, epoch)
                stats["validation_retried"] = True
                stats["validation_retry_batch"] = int(batch_size)
                self.val_amp = original_val_amp
                return stats
            except RuntimeError as exc:
                self._cleanup_cuda_after_error()
                if not self._is_recoverable_cuda_error(exc):
                    self.val_amp = original_val_amp
                    raise
                print(f"[WARN] Validation retry with batch={batch_size} failed: {exc}")

        self.val_amp = original_val_amp
        raise RuntimeError("Validation failed even after safe retry batches.")

    def _lr_lambda(self, epoch):
        """Warmup (1% → 100%) then cosine annealing."""
        if epoch < self.warmup_epochs:
            # Start at 1% LR and scale linearly to 100%
            return 0.01 + 0.99 * (epoch + 1) / self.warmup_epochs
        
        progress = (epoch - self.warmup_epochs) / max(self.epochs - self.warmup_epochs, 1)
        if self.cos_lr:
            cos_val  = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.lrf + (1.0 - self.lrf) * cos_val
        else:
            return 1.0 - (1.0 - self.lrf) * progress

    def _set_backbone_frozen(self, freeze, context=None):
        """Keep backbone trainability aligned with the current training phase."""
        orig = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        if not hasattr(orig, "backbone"):
            return

        for p in orig.backbone.parameters():
            p.requires_grad_(not freeze)

        if context:
            state = "frozen" if freeze else "unfrozen"
            print(f"[INFO] Backbone {state} ({context}).")

    def train(self, dataset, frozen_epochs=0, unfreeze_backbone_at=None):
        """
        Main training loop. 
        dataset: Optional FOTONETDataset used only for dynamic val_split training.
        """
        from fotonet.data.dataset import FOTONETDataset
        if unfreeze_backbone_at is not None:
            frozen_epochs = 0 if int(unfreeze_backbone_at) < 0 else int(unfreeze_backbone_at)
        frozen_epochs = max(int(frozen_epochs or 0), 0)
        initial_imgsz = self._imgsz_for_epoch(self.start_epoch)
        self.current_imgsz = initial_imgsz
        cut_total = max(int(getattr(self, "epoch_cut", 1) or 1), 1)
        cut_index = ((int(self.start_epoch) % cut_total) + 1) if cut_total > 1 else None
        self._write_live_status({
            "running": True,
            "stage": "loading data",
            "epoch": int(self.start_epoch) + 1,
            "epochs": int(self.epochs),
            "epoch_progress": 0.0,
            "imgsz": int(self.current_imgsz),
            "cut": int(cut_index) if cut_index is not None else None,
            "cut_total": int(cut_total),
            "timestamp": datetime.now().isoformat(),
        })

        # ---------------------------------------------------------------
        # 1. Dataset Setup (Split vs Standard Folders)
        # ---------------------------------------------------------------
        if self.val_split == 0 and isinstance(self.data_cfg, dict):
            print("[data] split=folders")
            train_path = os.path.join(self.data_cfg["path"], self.data_cfg["train"])
            val_path   = os.path.join(self.data_cfg["path"], self.data_cfg["val"])
            
            self.full_train_set = FOTONETDataset(
                img_dir = train_path,
                imgsz   = initial_imgsz,
                augment = True,
                cache_labels = self.cache_labels,
                cache_to_ram = self.cache_to_ram,
                ram_cache_images = self.ram_cache_images,
                disk_cache_images = self.disk_cache_images,
                disk_cache_dir = self.disk_cache_dir,
                augment_hyp = self.augment_hyp,
                num_classes = self.nc,
            )
            val_dataset = FOTONETDataset(
                img_dir = val_path,
                imgsz   = initial_imgsz,
                augment = False,
                cache_labels = self.cache_labels,
                disk_cache_images = self.disk_cache_images,
                disk_cache_dir = self.disk_cache_dir,
                cache_to_ram = False,
                num_classes = self.nc,
            )
        else:
            print(f"[data] split=dynamic val={self.val_split*100:.1f}%")
            self._set_dataset_imgsz(dataset, initial_imgsz)
            n_all    = len(dataset)
            n_val    = int(n_all * self.val_split)
            n_train  = n_all - n_val
            indices  = torch.randperm(n_all).tolist()
            train_idx = indices[:n_train]
            val_idx   = indices[n_train:]

            self.full_train_set = Subset(dataset, train_idx)
            val_files = [dataset.img_files[i] for i in val_idx]
            val_dataset = FOTONETDataset(
                img_dir  = val_files,
                imgsz    = initial_imgsz,
                augment  = False,
                cache_labels = self.cache_labels,
                disk_cache_images = self.disk_cache_images,
                disk_cache_dir = self.disk_cache_dir,
                cache_to_ram = False,
                num_classes = self.nc,
            )
        self.full_val_dataset = val_dataset
        self.val_dataset = val_dataset
        self.active_val_dataset = val_dataset
        self._init_val_subset(val_dataset)

        if hasattr(self.full_train_set, "set_total_epochs"):
            self.full_train_set.set_total_epochs(self.epochs)
        elif hasattr(self.full_train_set, "dataset") and hasattr(self.full_train_set.dataset, "set_total_epochs"):
            self.full_train_set.dataset.set_total_epochs(self.epochs)

        # ---------------------------------------------------------------
        # 2. Dataloader Parameters
        # ---------------------------------------------------------------
        num_workers = self.workers
        pf_factor   = 2 if num_workers > 0 else None
        train_pf_factor = 4 if num_workers > 0 and self.epoch_cut > 1 else pf_factor
        
        self._print_train_header(n_train=len(self.full_train_set), n_val=len(val_dataset))

        n_train = len(self.full_train_set)
        train_sampler = None
        train_shuffle = True
        if self.epoch_cut > 1:
            train_sampler = _EpochCutSampler(self.full_train_set, epoch_cut=self.epoch_cut, seed=42)
            train_sampler.set_epoch(self.start_epoch)
            train_shuffle = False
            chunk_size = len(train_sampler)
            print(
                f"[train] epoch_cut={self.epoch_cut} chunk_images={chunk_size}; "
                "validation skips are folded into epoch summaries."
            )

        self.train_sampler = train_sampler
        self._rebuild_train_prefetcher(train_sampler, train_shuffle, num_workers, train_pf_factor)
        steps_per_epoch = max(math.ceil(len(self.train_loader) / max(self.accum_steps, 1)), 1)
        self.ema.total_steps = max(steps_per_epoch * max(self.epochs - self.start_epoch, 1), self.ema.step_count + 1)

        if frozen_epochs > 0:
            freeze_backbone = self.start_epoch < frozen_epochs
            phase = f"epoch {self.start_epoch} resume state" if self.start_epoch > 0 else "training start"
            self._set_backbone_frozen(freeze_backbone, context=phase)

        for param_group in self.optimizer.param_groups:
            param_group.setdefault("initial_lr", self.lr0)

        scheduler_last_epoch = -1
        if self.resume_ckpt is not None and self.start_epoch > 0 and "scheduler_state" not in self.resume_ckpt:
            scheduler_last_epoch = self.start_epoch - 1

        scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._lr_lambda,
            last_epoch=scheduler_last_epoch,
        )

        if self.resume_ckpt is not None:
            if "scheduler_state" in self.resume_ckpt:
                scheduler.load_state_dict(self.resume_ckpt["scheduler_state"])
                self._restore_scheduler_lr(scheduler)
                print("[INFO] Restored LR scheduler state.")
            elif self.start_epoch > 0:
                print(f"[INFO] Rebuilt LR scheduler state for resume at epoch {self.start_epoch}.")

        # Profiler setup
        prof = None
        if self.profile:
            from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=tensorboard_trace_handler(os.path.join(self.save_dir, "profile")),
                record_shapes=True,
                with_stack=True
            )
            prof.start()

        close_mosaic_boundary = self._close_mosaic_boundary()
        close_mosaic_workers_refreshed = False

        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()
            self.model.train()
            desired_imgsz = self._imgsz_for_epoch(epoch)
            if desired_imgsz != self.current_imgsz:
                self.current_imgsz = desired_imgsz
                self._set_dataset_imgsz(self.full_train_set, desired_imgsz)
                self._set_dataset_imgsz(self.full_val_dataset, desired_imgsz)
                self._rebuild_train_prefetcher(train_sampler, train_shuffle, num_workers, train_pf_factor)
                print(f"[schedule] epoch={epoch+1} imgsz={desired_imgsz}")
            chunk_note = ""

            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
                chunk_note = f" chunk={(epoch % self.epoch_cut) + 1}/{self.epoch_cut}"

            if hasattr(self.full_train_set, 'set_epoch'):
                self.full_train_set.set_epoch(epoch)
            elif hasattr(self.full_train_set, 'dataset') and hasattr(self.full_train_set.dataset, 'set_epoch'):
                self.full_train_set.dataset.set_epoch(epoch)

            if (
                close_mosaic_boundary is not None
                and not close_mosaic_workers_refreshed
                and num_workers > 0
                and epoch >= close_mosaic_boundary
            ):
                self._rebuild_train_prefetcher(train_sampler, train_shuffle, num_workers, train_pf_factor)
                close_mosaic_workers_refreshed = True
                print(f"[schedule] epoch={epoch+1} close_mosaic workers refreshed")

            current_prefetcher = self.train_prefetcher

            if frozen_epochs > 0 and epoch == frozen_epochs and self.start_epoch < frozen_epochs:
                print(f"\n[STABILITY] Epoch {epoch}: Unfreezing backbone...")
                self._set_backbone_frozen(False)

            running = {
                "loss": 0.0, "cls": 0.0, "box": 0.0, "iou": 0.0,
                "dflpc": 0.0, "cons": 0.0, "wcls": 0.0, "wdfl": 0.0, "wiou": 0.0,
                "wcons": 0.0, "awcls": 0.0, "awbox": 0.0, "awdfl": 0.0, "awcons": 0.0,
                "quality": 0.0, "wquality": 0.0, "awquality": 0.0,
                "qmix": 0.0, "hneg": 0.0, "pos_o2o": 0.0, "pos_o2m": 0.0,
                "o2m_active": 0.0, "exact_o2o": 0.0, "distill": 0.0,
            }
            n_steps = 0
            last_accum_step = -1  # Track whether last batch was an accum boundary
            self.optimizer.zero_grad()
            total_batches = max(len(current_prefetcher), 1)
            last_live_status_at = 0.0
            self._last_iter_rate_step = self.global_step
            self._last_iter_rate_at = epoch_start
            self._write_live_status(
                self._live_status_payload(epoch, 0, total_batches, running, n_steps, elapsed_sec=0.0)
            )

            pbar = tqdm(
                current_prefetcher,
                desc=f"Epoch {epoch+1:>3}/{self.epochs}",
            )
            for i, (imgs, targets) in enumerate(pbar):

                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=torch.float16):
                    use_o2m = self.criterion.use_o2m(epoch, self.global_step) if hasattr(self.criterion, "use_o2m") else True
                    preview_due = time.time() - getattr(self, "_last_preview_at", 0.0) >= self.preview_interval_sec
                    outputs = self.model(imgs, use_o2m=use_o2m, return_preview=preview_due)
                    loss_dict = self.criterion(
                        outputs,
                        targets,
                        current_epoch=epoch,
                        max_epochs=self.epochs,
                        global_step=self.global_step,
                    )
                    distill_loss = self._distillation_loss(outputs, imgs, epoch)
                    loss_dict["loss"] = loss_dict["loss"] + distill_loss
                    loss_dict["loss_distill"] = distill_loss.detach()
                    self._assert_finite_loss_dict(loss_dict, epoch, i, self.global_step)
                    loss      = loss_dict["loss"] / self.accum_steps

                self.scaler.scale(loss).backward()

                if (i + 1) % self.accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.ema.update(self.model)
                    last_accum_step = i

                if prof: prof.step()

                real_loss = loss_dict["loss"].item()
                running["loss"] += real_loss
                running["cls"]  += loss_dict["loss_cls"].item()
                running["box"]  += loss_dict["loss_box"].item()
                running["dflpc"] += loss_dict["loss_dfl_per_coord"].item()
                running["iou"]  += loss_dict["loss_ciou"].item()
                running["cons"] += loss_dict["loss_consistency"].item()
                running["wcls"] += loss_dict["loss_cls_weighted"].item()
                running["wdfl"] += loss_dict["loss_dfl_weighted"].item()
                running["wiou"] += loss_dict["loss_ciou_weighted"].item()
                running["wcons"] += loss_dict["loss_consistency_weighted"].item()
                running["quality"] += float(loss_dict.get("loss_quality", 0.0))
                running["wquality"] += float(loss_dict.get("loss_quality_weighted", 0.0))
                running["awcls"] += float(loss_dict["active_w_cls"])
                running["awbox"] += float(loss_dict["active_w_box"])
                running["awdfl"] += float(loss_dict["active_w_dfl"])
                running["awcons"] += float(loss_dict["active_w_consistency"])
                running["awquality"] += float(loss_dict.get("active_w_quality", 0.0))
                running["qmix"] += float(loss_dict["quality_mix"])
                running["hneg"] += float(loss_dict["hard_negative_weight"])
                running["pos_o2o"] += float(loss_dict["num_pos_o2o"])
                running["pos_o2m"] += float(loss_dict["num_pos_o2m"])
                running["o2m_active"] += float(loss_dict.get("o2m_active", 0.0))
                running["exact_o2o"] += float(loss_dict.get("exact_o2o_active", 0.0))
                running["distill"] += float(loss_dict.get("loss_distill", 0.0))
                n_steps += 1
                self.global_step += 1

                denom = max(n_steps, 1)
                pbar.set_postfix(
                    loss=f"{running['loss'] / denom:.2f}",
                    cls=f"{running['cls'] / denom:.2f}",
                    box=f"{running['box'] / denom:.2f}",
                    dfl=f"{running['dflpc'] / denom:.2f}",
                    iou=f"{running['iou'] / denom:.2f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                )
                now_live_status = time.time()
                if now_live_status - last_live_status_at >= 1.0 or i + 1 >= total_batches:
                    self._write_live_status(
                        self._live_status_payload(
                            epoch,
                            i + 1,
                            total_batches,
                            running,
                            n_steps,
                            elapsed_sec=now_live_status - epoch_start,
                        )
                    )
                    last_live_status_at = now_live_status
                self._maybe_write_run_previews(imgs, targets, outputs, now_live_status)

            # Flush remaining accumulated gradients only if last batch wasn't an accum boundary
            if last_accum_step != i and n_steps > 0:
                try:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                except RuntimeError:
                    pass
                self.optimizer.zero_grad()
                self.ema.update(self.model)

            scheduler.step()

            # Log training metrics per epoch
            log_entry = {
                "epoch":        epoch + 1,
                "imgsz":        int(self.current_imgsz),
                "loss":         round(running['loss']/max(n_steps,1), 4),
                "cls_loss":     round(running['cls']/max(n_steps,1), 4),
                "box_loss":     round(running['box']/max(n_steps,1), 4),
                "dfl_loss":     round(running['dflpc']/max(n_steps,1), 4),
                "iou_loss":     round(running['iou']/max(n_steps,1), 4),
                "consistency_loss": round(running['cons']/max(n_steps,1), 4),
                "quality_loss": round(running['quality']/max(n_steps,1), 4),
                "weighted_cls_loss": round(running['wcls']/max(n_steps,1), 4),
                "weighted_dfl_loss": round(running['wdfl']/max(n_steps,1), 4),
                "weighted_iou_loss": round(running['wiou']/max(n_steps,1), 4),
                "weighted_consistency_loss": round(running['wcons']/max(n_steps,1), 4),
                "weighted_quality_loss": round(running['wquality']/max(n_steps,1), 4),
                "active_w_cls": round(running['awcls']/max(n_steps,1), 4),
                "active_w_box": round(running['awbox']/max(n_steps,1), 4),
                "active_w_dfl": round(running['awdfl']/max(n_steps,1), 4),
                "active_w_consistency": round(running['awcons']/max(n_steps,1), 4),
                "active_w_quality": round(running['awquality']/max(n_steps,1), 4),
                "quality_mix": round(running['qmix']/max(n_steps,1), 4),
                "hard_negative_weight": round(running['hneg']/max(n_steps,1), 4),
                "avg_pos_o2o": round(running['pos_o2o']/max(n_steps,1), 2),
                "avg_pos_o2m": round(running['pos_o2m']/max(n_steps,1), 2),
                "o2m_active_ratio": round(running['o2m_active']/max(n_steps,1), 4),
                "exact_o2o_ratio": round(running['exact_o2o']/max(n_steps,1), 4),
                "distill_loss": round(running['distill']/max(n_steps,1), 4),
                "lr":           self.optimizer.param_groups[0]['lr'],
                "timestamp":    datetime.now().isoformat()
            }

            is_last = (epoch == self.epochs - 1)
            will_validate = (epoch + 1) % self.val_period == 0 or is_last

            # Save before validation only when validation actually runs. If
            # validation hits a driver/cuDNN hiccup, resume starts after the
            # trained epoch instead of wasting the completed work.
            if self.save_last and will_validate:
                self._save_last_checkpoint(scheduler, epoch)

            # Per-Epoch Validation Loop (Controlled by val_period)
            if will_validate:
                self.active_val_dataset, val_mode = self._val_dataset_for_epoch(epoch)
                self.val_dataset = self.active_val_dataset
                val_loader = self._make_val_loader(
                    self.active_val_dataset,
                    self.val_batch_size,
                    num_workers,
                    pf_factor,
                    persistent_workers=False,
                )
                val_stats = self._validate_with_retries(val_loader, epoch, num_workers, pf_factor)
                val_stats["val_mode"] = val_mode
                log_entry.update(val_stats)

                # Auto-save best model by the configured validation metric.
                metric_score = float(val_stats.get(self.best_metric, val_stats["mAP50"]))
                if metric_score > self.best_map:
                    self.best_map = metric_score
                    self._save_best_checkpoint(epoch)

            epoch_seconds = time.time() - epoch_start
            log_entry["epoch_time_sec"] = round(epoch_seconds, 2)
            if will_validate:
                val_text = (
                    f"val({log_entry.get('val_mode', 'full')}) "
                    f"P={log_entry['precision']:.4f} R={log_entry['recall']:.4f} "
                    f"mAP50={log_entry['mAP50']:.4f} mAP50-95={log_entry['mAP50_95']:.4f} "
                    f"{log_entry.get('val_images', 0)}img {log_entry.get('val_time_sec', 0):.1f}s"
                )
            else:
                next_val = min(((epoch + 1) // self.val_period + 1) * self.val_period, self.epochs)
                val_text = f"val=skip next={next_val}"
            print(
                f"[epoch {epoch+1}/{self.epochs}{chunk_note}] "
                f"imgsz={self.current_imgsz} loss={log_entry['loss']:.4f} cls={log_entry['cls_loss']:.4f} "
                f"box={log_entry['box_loss']:.4f} dfl={log_entry['dfl_loss']:.4f} lr={log_entry['lr']:.2e} "
                f"o2m={log_entry['o2m_active_ratio']:.2f} exact={log_entry['exact_o2o_ratio']:.2f} "
                f"{val_text} time={epoch_seconds:.1f}s"
            )

            with open(self.log_file, "r+", encoding="utf-8") as f:
                data = json.load(f)
                data.append(log_entry)
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
            self._write_live_status(
                self._live_status_payload(
                    epoch,
                    total_batches,
                    total_batches,
                    running,
                    n_steps,
                    extra=log_entry,
                    elapsed_sec=max(time.time() - epoch_start, 1e-9),
                )
            )

            # Save resumable last checkpoint every epoch; periodic milestones are opt-in.
            # This second save preserves updated best-score metadata after validation.
            if self.save_last:
                self._save_last_checkpoint(scheduler, epoch)
            self._save_periodic_checkpoint(scheduler, epoch)

        self._write_live_status({
            "running": False,
            "epoch": int(self.epochs),
            "epochs": int(self.epochs),
            "timestamp": datetime.now().isoformat(),
        })

    @torch.inference_mode()
    def validate(self, loader, epoch):
        """Validation using EMA model with AMP for speed."""
        from fotonet.metrics.map import CocoMapEvaluator
        self.ema.ema.eval()
        
        preds_b, preds_s, preds_c = [], [], []
        gts_b, gts_c = [], []
        image_ids = []
        
        val_start = time.time()
        n_images  = 0

        for imgs, targets in tqdm(
            loader,
            desc="Validating",
            leave=False,
        ):
            imgs = imgs.to(self.device, non_blocking=True)
            n_images += imgs.shape[0]

            use_val_amp = self.val_amp and self.device.type == "cuda"
            with torch.amp.autocast("cuda", enabled=use_val_amp, dtype=torch.float16):
                outputs = self.ema.ema(imgs)

            for b_idx in range(outputs.shape[0]):
                out = outputs[b_idx]
                scores, classes = out[:, :self.nc].sigmoid().max(1)
                boxes = out[:, self.nc:]

                # O2O branch: one prediction per GT by design — no NMS needed.
                # Low pre-filter for AP ranking; deploy inference can still use 0.25.
                mask = scores > 0.001
                b, s, c = boxes[mask], scores[mask], classes[mask]

                if len(b) > 100:
                    s, idx = torch.topk(s, 100)
                    b = b[idx]
                    c = c[idx]

                preds_b.append(b.cpu().numpy() if len(b) > 0 else np.empty((0, 4)))
                preds_s.append(s.cpu().numpy() if len(s) > 0 else np.empty(0))
                preds_c.append(c.cpu().numpy() if len(c) > 0 else np.empty(0, dtype=np.int32))

                gts_b.append(targets[b_idx]["boxes"].cpu().numpy())
                gts_c.append(targets[b_idx]["labels"].cpu().numpy())
                image_ids.append(int(targets[b_idx].get("image_id", torch.tensor([len(image_ids)])).reshape(-1)[0].item()))

        # Restore training model to train mode
        self.model.train()

        cache_key = (
            int(self.nc),
            tuple(int(x) for x in image_ids),
            tuple(int(len(x)) for x in gts_b),
            tuple(tuple(np.asarray(x, dtype=np.int64).tolist()) for x in gts_c),
        )
        cached_key, evaluator = self._coco_evaluator_cache or (None, None)
        if cached_key != cache_key or evaluator is None:
            evaluator = CocoMapEvaluator(gts_b, gts_c, image_ids=image_ids, num_classes=self.nc, max_dets=100)
            self._coco_evaluator_cache = (cache_key, evaluator)
        metrics = evaluator.evaluate(preds_b, preds_s, preds_c)

        val_time = time.time() - val_start
        fps = n_images / val_time if val_time > 0 else 0

        log_entry = {
            "mAP50":        round(float(metrics["mAP50"]), 4),
            "mAP50_95":     round(float(metrics["mAP50_95"]), 4),
            "precision":    round(float(metrics["precision"]), 4),
            "recall":       round(float(metrics["recall"]), 4),
            "coco_AR100":    round(float(metrics["coco_AR100"]), 4),
            "metric_backend": metrics["metric_backend"],
            "per_class":     metrics["per_class"],
            "val_images":   int(n_images),
            "val_time_sec": round(val_time, 2),
            "avg_fps":      round(fps, 1)
        }
        return log_entry

    @staticmethod
    def _collate_fn(batch):
        """Simple collate: stack images directly (all are same size from dataset)."""
        imgs, targets = zip(*batch)
        return torch.stack(imgs, 0), list(targets)
