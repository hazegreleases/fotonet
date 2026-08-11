"""Construction of exact V1 trainer state from resolved arguments."""

import math
import os
import uuid
from datetime import datetime

import torch

from fotonet.engine.optim import build_optimizer, split_optimizer_params
from fotonet.engine.runtime import EMA
from fotonet.utils.config import default_class_names, normalize_class_schema
from fotonet.utils.loss import get_loss


class InitializationMixin:
    @staticmethod
    def _resolve_training_run_id(resume_ckpt, requested_run_id=None):
        if requested_run_id is not None:
            if not isinstance(requested_run_id, str) or not requested_run_id.strip():
                raise ValueError("_training_run_id must be a non-empty string when provided")
            if isinstance(resume_ckpt, dict):
                checkpoint_run_id = resume_ckpt.get("training_run_id")
                if checkpoint_run_id != requested_run_id:
                    raise ValueError(
                        "_training_run_id does not match the resume checkpoint training_run_id"
                    )
            return requested_run_id

        if isinstance(resume_ckpt, dict):
            run_id = resume_ckpt.get("training_run_id")
            if not run_id:
                raise ValueError("Resume checkpoint is missing training_run_id")
            return str(run_id)
        return uuid.uuid4().hex

    def __init__(
        self, model, data_cfg, epochs=100, imgsz=640, lr0=0.01, lrf=0.01,
        batch=16, val_batch=None, nbs=128, warmup_epochs=3, save_dir=".",
        val_split=None, val_period=1, workers=None, pin_memory=True,
        cache_to_ram=True, ram_cache_images=1024, amp=True,
        amp_init_scale=65536.0, val_amp=None, cos_lr=True,
        lr_scheduler="Cosine", lr_drop_factor=0.92, lr_drop_patience=5,
        lr_drop_threshold=0.001, lr_drop_min_lr=1e-5, resume_ckpt=None,
        compile_model=False, save_period=-1, slim_best=True,
        best_metric="mAP50_95", save_last=True, optimizer="sgd",
        momentum=0.937, weight_decay=0.0005, augment_hyp=None,
        augmentation_passes=None, loss_hyp=None, matcher_hyp=None,
        imgsz_schedule=None, val_subset_size=0, full_val_after=1.0,
        cache_labels=True, disk_cache_images=False, disk_cache_dir=None,
        coco_max_dets=100, val_conf=0.0, operating_conf=0.25,
        operating_iou=0.50, annotation_policy="fix",
        allow_missing_labels=False, source_recursive=True,
        _training_run_id=None,
    ):
        self.model         = model
        self.epochs        = int(epochs)
        if self.epochs < -1:
            raise ValueError("epochs must be a non-negative integer or -1 for an explicit unbounded run")
        self.infinite_epochs = self.epochs < 0
        self.imgsz         = int(imgsz)
        self.max_stride = self._model_max_stride(model)
        self._validate_imgsz_alignment(self.imgsz, self.max_stride, "imgsz")
        self.imgsz_schedule = self._normalize_imgsz_schedule(imgsz_schedule, int(imgsz))
        for schedule_index, (_until, schedule_imgsz) in enumerate(self.imgsz_schedule, start=1):
            self._validate_imgsz_alignment(
                schedule_imgsz,
                self.max_stride,
                f"imgsz_schedule entry {schedule_index}",
            )
        self.current_imgsz = int(imgsz)
        self.lr0           = float(lr0)
        self.lrf           = float(lrf)
        self.batch_size    = int(batch)
        self.val_batch_size = self.batch_size if val_batch is None else int(val_batch)
        self.nbs           = int(nbs)
        self.warmup_epochs = float(warmup_epochs)
        if not math.isfinite(self.lr0) or self.lr0 <= 0:
            raise ValueError("lr0 must be a finite positive learning rate")
        if not math.isfinite(self.lrf) or not 0.0 <= self.lrf <= 1.0:
            raise ValueError("lrf must be a finite final-LR fraction in [0, 1]")
        if self.batch_size <= 0 or self.val_batch_size <= 0 or self.nbs <= 0:
            raise ValueError("batch, val_batch, and nbs must be positive integers")
        if not math.isfinite(self.warmup_epochs) or self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be a finite non-negative number")
        self.save_dir      = save_dir
        self.nc, _         = normalize_class_schema(model.head.nc, context="model")
        self.data_cfg      = dict(data_cfg) if isinstance(data_cfg, dict) else data_cfg
        self._configured_class_names = None
        self.class_names = default_class_names(self.nc)
        if isinstance(self.data_cfg, dict):
            configured_nc, configured_names = normalize_class_schema(
                self.data_cfg.get("nc"),
                self.data_cfg.get("names"),
                context="data config",
            )
            if configured_nc is not None and configured_nc != self.nc:
                raise ValueError(
                    f"Dataset nc={configured_nc} does not match model nc={self.nc}. "
                    "Construct a matching model before creating Trainer."
                )
            if configured_nc is not None:
                self.data_cfg["nc"] = configured_nc
            if configured_names is not None:
                self.data_cfg["names"] = configured_names
                self._configured_class_names = configured_names
                self.class_names = dict(configured_names)
        self.val_period    = int(val_period)
        if self.val_period <= 0:
            raise ValueError("val_period must be a positive integer")

        # Hardware parameters
        cpu_count = os.cpu_count() or 1
        self.workers       = int(workers) if workers is not None else max(cpu_count - 2, 1)
        if self.workers < 0:
            raise ValueError("workers must be non-negative")
        self.pin_memory    = pin_memory and (next(model.parameters()).device.type == "cuda")
        self.cache_to_ram  = cache_to_ram
        self.ram_cache_images = ram_cache_images
        self.use_amp       = amp
        self.amp_init_scale = float(amp_init_scale)
        self.val_amp       = amp if val_amp is None else bool(val_amp)
        self.cos_lr        = cos_lr
        self.lr_scheduler  = self._normalize_lr_scheduler(lr_scheduler)
        self.lr_drop_factor = float(lr_drop_factor)
        self.lr_drop_patience = max(int(lr_drop_patience or 1), 1)
        self.lr_drop_threshold = float(lr_drop_threshold)
        self.lr_drop_min_lr = float(lr_drop_min_lr)
        self.compile_model = compile_model
        self.save_period   = int(save_period)
        self.slim_best     = slim_best
        self.best_metric   = self._normalize_best_metric(best_metric)
        self.save_last     = bool(save_last)
        self.optimizer_name = str(optimizer or "sgd")
        self.momentum      = float(momentum)
        self.weight_decay  = float(weight_decay)
        if not math.isfinite(self.momentum) or self.momentum < 0:
            raise ValueError("momentum must be a finite non-negative value")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be a finite non-negative value")
        self.augment_hyp   = augment_hyp or {}
        self.augmentation_passes = None
        self.augmentation_pass_names = None
        if augmentation_passes is not None:
            if not isinstance(augmentation_passes, (list, tuple)) or not augmentation_passes:
                raise TypeError("augmentation_passes must be a non-empty list of augmentation mappings")
            if len(augmentation_passes) != int(self.epochs):
                raise ValueError(
                    f"augmentation_passes has {len(augmentation_passes)} entries but training has "
                    f"{self.epochs} passes; provide exactly one augmentation mapping per pass."
                )
            from fotonet.data.augmentations import build_augment_hyp
            profiles, names = [], []
            for index, profile in enumerate(augmentation_passes):
                if not isinstance(profile, dict):
                    raise TypeError(f"augmentation_passes[{index}] must be a mapping")
                profile = dict(profile)
                names.append(str(profile.pop("name", f"augmentation-{index + 1}")))
                profile["close_mosaic"] = 0
                profiles.append(build_augment_hyp(profile))
            self.augmentation_passes = tuple(profiles)
            self.augmentation_pass_names = tuple(names)
        self._active_augmentation_pass = None
        self.loss_hyp      = loss_hyp or {}
        self.matcher_hyp   = matcher_hyp or {}
        self.val_subset_size = max(int(val_subset_size or 0), 0)
        self.full_val_after = float(full_val_after if full_val_after is not None else 1.0)
        if self.val_subset_size > 0:
            raise ValueError(
                "val_subset_size is disabled because subset metrics must never select a best checkpoint or drive "
                "the validation scheduler. Use full validation for comparable AP."
            )
        self.cache_labels = bool(cache_labels)
        self.disk_cache_images = bool(disk_cache_images)
        self.disk_cache_dir = disk_cache_dir
        self.coco_max_dets = int(coco_max_dets)
        self.val_conf = float(val_conf)
        self.operating_conf = float(operating_conf)
        self.operating_iou = float(operating_iou)
        self.annotation_policy = str(annotation_policy or "fix")
        self.allow_missing_labels = bool(allow_missing_labels)
        self.source_recursive = bool(source_recursive)
        if self.coco_max_dets < 1:
            raise ValueError("coco_max_dets must be positive")
        if not 0.0 <= self.val_conf <= 1.0:
            raise ValueError("val_conf must be in [0, 1]")
        if not 0.0 <= self.operating_conf <= 1.0:
            raise ValueError("operating_conf must be in [0, 1]")
        if self.val_conf > self.operating_conf:
            raise ValueError(
                "val_conf cannot exceed operating_conf: validation would discard predictions "
                "needed to report the requested operating point."
            )
        if not 0.0 < self.operating_iou <= 1.0:
            raise ValueError("operating_iou must be in (0, 1]")
        
        # Compile model — use "default" mode to avoid CUDA graph capture warmup lag
        if self.compile_model and torch.cuda.is_available():
            try:
                print("[INFO] Compiling model with torch.compile(mode='default')...")
                self.model = torch.compile(self.model, mode="default")
                print("[INFO] Model compiled successfully!")
            except Exception as e:
                print(f"[WARN] torch.compile() failed, using uncompiled model: {e}")

        if val_split is None:
            self.val_split = 0.0
            self.validation_protocol = "configured_val"
        else:
            self.val_split = float(val_split)
            if not 0.0 <= self.val_split < 1.0:
                raise ValueError("val_split must be None/0 for configured validation or satisfy 0 < val_split < 1")
            self.validation_protocol = "configured_val" if self.val_split == 0.0 else "random_train_split"

        self.device = next(self.model.parameters()).device
        requested_amp = bool(self.use_amp)
        self.use_amp = requested_amp and self.device.type == "cuda"
        self.val_amp = bool(self.val_amp) and self.device.type == "cuda"
        if requested_amp and self.device.type != "cuda":
            print(f"[INFO] AMP disabled on {self.device.type}; CUDA AMP is unavailable.")
        
        # Get original model (for optimizer, EMA, etc.)
        original_model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        
        self.optimizer = build_optimizer(
            original_model,
            name=self.optimizer_name,
            lr=self.lr0,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self._install_optimizer_step_tracker()
        self.accum_steps = max(round(self.nbs / self.batch_size), 1)

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
        est_steps_per_epoch = max(100000 // max(self.batch_size, 1), 100)
        total_ema_steps = est_steps_per_epoch * (self.epochs if not self.infinite_epochs else 500)

        self.ema = EMA(
            original_model,
            decay_start=0.0,
            decay_end=0.9999,
            total_steps=total_ema_steps,
            warmup_steps=2000,
        )

        self.scaler = torch.amp.GradScaler(
            "cuda" if self.device.type == "cuda" else "cpu",
            enabled=self.use_amp,
            init_scale=self.amp_init_scale,
        )

        self.best_loss = float("inf")
        # A fresh run has no selected validation result yet. -inf guarantees
        # the first real validation, including a valid zero-AP run, replaces a
        # stale checkpoint instead of silently inheriting it.
        self.best_map  = float("-inf")
        self.start_epoch = 0
        self.global_step = 0
        self.images_seen = 0
        self.labels_seen = 0
        self.pixel_equivalent_images = 0.0
        self.training_wall_time_sec = 0.0
        self.ap_threshold_times = {}
        self.resume_ckpt = resume_ckpt
        if isinstance(resume_ckpt, dict):
            self._apply_resume_class_schema(resume_ckpt)
        self._resume_ema_total_steps = None
        self.optimizer_step_count = 0
        self._lr_total_steps = 1
        self._warmup_steps = 0
        self._active_scheduler = None
        self.schedule_progress = 0.0

        self.training_run_id = self._resolve_training_run_id(
            resume_ckpt,
            _training_run_id,
        )
        self.last_checkpoint_path = os.path.join(self.save_dir, "fotonet_last.pt")
        self.best_checkpoint_path = os.path.join(self.save_dir, "fotonet_best.pt")
        
        # Persistent dataset and loader references for reuse
        self.full_train_set = None
        self.chunk_indices = []
        self.train_loader = None
        self.train_prefetcher = None
        self.val_dataset = None
        self.full_val_dataset = None
        self.active_val_dataset = None
        self._val_subset_indices = None
        self._validation_loaders = {}

        # Restore full training state from a resume checkpoint
        if resume_ckpt is not None:
            if not isinstance(resume_ckpt, dict):
                raise ValueError("resume_ckpt must be a full checkpoint mapping")
            self.start_epoch = resume_ckpt.get("epoch", 0)
            ckpt_metric_raw = resume_ckpt.get("best_metric", "mAP50")
            try:
                ckpt_metric = self._normalize_best_metric(ckpt_metric_raw)
            except ValueError:
                ckpt_metric = None
            if ckpt_metric == self.best_metric:
                self.best_map = float(resume_ckpt.get("best_score", resume_ckpt.get("best_map", resume_ckpt.get("mAP", float("-inf")))))
            elif self.best_metric == "mAP50":
                self.best_map = float(resume_ckpt.get("best_map", resume_ckpt.get("mAP", float("-inf"))))
            else:
                self.best_map = float("-inf")
            required_state = ("optimizer_state", "scaler_state", "ema_state")
            missing_state = [key for key in required_state if key not in resume_ckpt]
            if missing_state:
                raise ValueError(
                    "Cannot safely resume: checkpoint is missing " + ", ".join(missing_state) + ". "
                    "Start a fresh pretrained run instead."
                )
            try:
                self.optimizer.load_state_dict(resume_ckpt["optimizer_state"])
                self._move_optimizer_state_to_device()
                self.scaler.load_state_dict(resume_ckpt["scaler_state"])
                self.ema.ema.load_state_dict(resume_ckpt["ema_state"])
            except Exception as exc:
                raise ValueError(
                    "Cannot safely resume: optimizer, scaler, or EMA state is incompatible with the model. "
                    "Start a fresh pretrained run instead."
                ) from exc
            self.ema.step_count = int(resume_ckpt.get("ema_updates", self.ema.step_count))
            self.ema.warmup_steps = max(int(resume_ckpt.get("ema_warmup_steps", self.ema.warmup_steps)), 1)
            self.ema.decay_start = float(resume_ckpt.get("ema_decay_start", self.ema.decay_start))
            self.ema.decay_end = float(resume_ckpt.get("ema_decay_end", self.ema.decay_end))
            print("[INFO] Restored optimizer, AMP scaler, and EMA state.")
            if resume_ckpt.get("ema_total_steps") is not None:
                self._resume_ema_total_steps = int(resume_ckpt["ema_total_steps"])
            self.optimizer_step_count = int(resume_ckpt.get("optimizer_step_count", 0))
            self._restore_rng_state(resume_ckpt.get("rng_state"))

            if not self._resume_best_checkpoint_matches():
                print("[WARN] Resume best checkpoint is missing, stale, or incompatible; best score will restart from this run.")
                self.best_map = float("-inf")
            print(f"[INFO] Resuming from epoch {self.start_epoch} | best_{self.best_metric}={self.best_map:.4f}")
            self.global_step = int(resume_ckpt.get("global_step", self.global_step))
            self.images_seen = int(resume_ckpt.get("images_seen", self.images_seen))
            self.labels_seen = int(
                resume_ckpt.get("labels_seen", self.labels_seen)
            )
            self.pixel_equivalent_images = float(
                resume_ckpt.get(
                    "pixel_equivalent_images",
                    self.pixel_equivalent_images,
                )
            )
            self.training_wall_time_sec = float(
                resume_ckpt.get(
                    "training_wall_time_sec",
                    self.training_wall_time_sec,
                )
            )
            self.ap_threshold_times = {}
            for key, value in dict(
                resume_ckpt.get("ap_threshold_times", {})
            ).items():
                self.ap_threshold_times[str(key)] = (
                    dict(value)
                    if isinstance(value, dict)
                    else {"wall_time_sec": float(value)}
                )
            self.schedule_progress = float(
                resume_ckpt.get(
                    "schedule_progress",
                    self.start_epoch / max(int(self.epochs), 1),
                )
            )

        # Setup val logging path
        now = datetime.now()
        log_dir = os.path.join(save_dir, "logs", now.strftime("train-%Y/%m/%d/%H/%M"))
        os.makedirs(log_dir, exist_ok=True)
        # Append-only JSONL avoids reading/re-writing every prior epoch.
        self.log_file = os.path.join(log_dir, "metrics.jsonl")
        self.live_status_file = os.path.join(log_dir, "live_status.json")
        self._last_iter_rate_step = None
        self._last_iter_rate_at = None
        open(self.log_file, "w", encoding="utf-8").close()
        self._write_live_status({
            "running": False,
            "epoch": int(self.start_epoch),
            "epochs": int(self.epochs),
            "timestamp": datetime.now().isoformat(),
        })


    @staticmethod
    def _split_optimizer_params(model):
        """Return no-decay, decay, and bias parameter groups without dropping direct Parameters."""
        return split_optimizer_params(model)


__all__ = ["InitializationMixin"]
