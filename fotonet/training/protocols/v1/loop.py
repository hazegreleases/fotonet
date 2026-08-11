"""Exact V1 epoch/batch coordinator and update ordering."""

import json
import math
import os
import time
from datetime import datetime

import torch
from torch.utils.data import Subset
from tqdm import tqdm


class TrainingLoopMixin:
    def train(self, dataset, frozen_epochs=0, unfreeze_backbone_at=None):
        """
        Main training loop. 
        dataset: Optional DetectionDataset used only for dynamic val_split training.
        """
        from fotonet.data.dataset import build_detection_dataset
        if unfreeze_backbone_at is not None:
            frozen_epochs = 0 if int(unfreeze_backbone_at) < 0 else int(unfreeze_backbone_at)
        frozen_epochs = max(int(frozen_epochs or 0), 0)
        initial_imgsz = self._imgsz_for_epoch(self.start_epoch)
        self.current_imgsz = initial_imgsz
        self._write_live_status({
            "running": True,
            "stage": "loading data",
            "epoch": int(self.start_epoch) + 1,
            "epochs": int(self.epochs),
            "epoch_progress": 0.0,
            "imgsz": int(self.current_imgsz),
            "timestamp": datetime.now().isoformat(),
        })

        # ---------------------------------------------------------------
        # 1. Dataset Setup (Split vs Standard Folders)
        # ---------------------------------------------------------------
        if self.val_split == 0 and isinstance(self.data_cfg, dict):
            if "train" not in self.data_cfg or "val" not in self.data_cfg:
                raise ValueError(
                    "Configured validation requires both 'train' and 'val' in the data config. "
                    "FOTONET never falls back to evaluating the training set."
                )
            if dataset is None:
                self.full_train_set = build_detection_dataset(
                    self.data_cfg["train"],
                    imgsz=initial_imgsz,
                    augment=True,
                    cache_labels=self.cache_labels,
                    cache_to_ram=self.cache_to_ram,
                    ram_cache_images=self.ram_cache_images,
                    disk_cache_images=self.disk_cache_images,
                    disk_cache_dir=self.disk_cache_dir,
                    augment_hyp=self.augment_hyp,
                    num_classes=self.nc,
                    coco_images=self.data_cfg.get("train_images", self.data_cfg.get("coco_images")),
                )
            else:
                self.full_train_set = dataset
            val_dataset = build_detection_dataset(
                self.data_cfg["val"],
                imgsz=initial_imgsz,
                augment=False,
                cache_labels=self.cache_labels,
                disk_cache_images=self.disk_cache_images,
                disk_cache_dir=self.disk_cache_dir,
                cache_to_ram=False,
                num_classes=self.nc,
                annotation_policy=self.annotation_policy,
                allow_missing_labels=self.allow_missing_labels,
                source_recursive=self.source_recursive,
                coco_images=self.data_cfg.get("val_images", self.data_cfg.get("coco_images")),
            )
            if dataset is None:
                self._assert_disjoint_train_val_images(self.full_train_set, val_dataset)
        else:
            if dataset is None:
                raise ValueError("An explicit dynamic val_split requires a train dataset instance")
            if getattr(dataset, "is_coco", False):
                raise ValueError(
                    "Dynamic val_split is not supported for direct COCO JSON because it would no longer be a canonical "
                    "COCO validation protocol. Configure an explicit 'val' source instead."
                )
            self._set_dataset_imgsz(dataset, initial_imgsz)
            n_all    = len(dataset)
            n_val    = int(n_all * self.val_split)
            n_train  = n_all - n_val
            if n_all < 2 or n_val < 1 or n_train < 1:
                raise ValueError(f"Dynamic val_split={self.val_split} needs at least one train and one val image; got {n_all}")
            split_generator = torch.Generator()
            split_generator.manual_seed(42)
            indices  = torch.randperm(n_all, generator=split_generator).tolist()
            train_idx = indices[:n_train]
            val_idx   = indices[n_train:]

            if hasattr(dataset, "set_sampling_indices"):
                dataset.set_sampling_indices(train_idx)
            self.full_train_set = Subset(dataset, train_idx)
            val_files = [dataset.img_files[i] for i in val_idx]
            val_dataset = build_detection_dataset(
                val_files,
                imgsz=initial_imgsz,
                augment=False,
                cache_labels=self.cache_labels,
                disk_cache_images=self.disk_cache_images,
                disk_cache_dir=self.disk_cache_dir,
                cache_to_ram=False,
                num_classes=self.nc,
                annotation_policy=self.annotation_policy,
                allow_missing_labels=self.allow_missing_labels,
                source_recursive=self.source_recursive,
            )

        self.full_val_dataset = val_dataset
        self.val_dataset = val_dataset
        self.active_val_dataset = val_dataset
        self._init_val_subset(val_dataset)

        if self.augmentation_passes is not None:
            initial_pass = min(int(self.start_epoch), len(self.augmentation_passes) - 1)
            self._set_train_augmentation_pass(initial_pass)
            print(
                f"[pass {initial_pass + 1}/{len(self.augmentation_passes)}] "
                f"augmentation={self.augmentation_pass_names[initial_pass]}"
            )

        if hasattr(self.full_train_set, "set_total_epochs"):
            self.full_train_set.set_total_epochs(self.epochs)
        elif hasattr(self.full_train_set, "dataset") and hasattr(self.full_train_set.dataset, "set_total_epochs"):
            self.full_train_set.dataset.set_total_epochs(self.epochs)

        # ---------------------------------------------------------------
        # 2. Dataloader Parameters
        # ---------------------------------------------------------------
        num_workers = self.workers
        pf_factor   = 2 if num_workers > 0 else None
        train_pf_factor = pf_factor
        
        self._print_train_header(n_train=len(self.full_train_set), n_val=len(val_dataset))

        n_train = len(self.full_train_set)
        train_sampler = None
        train_shuffle = True

        self.train_sampler = train_sampler
        self._rebuild_train_prefetcher(
            train_sampler, train_shuffle, num_workers, train_pf_factor
        )
        steps_per_epoch = max(
            math.ceil(len(self.train_loader) / max(self.accum_steps, 1)),
            1,
        )
        ema_epochs = 500 if self.infinite_epochs else max(int(self.epochs), 1)
        scheduled_ema_steps = steps_per_epoch * ema_epochs
        restored_ema_steps = int(self._resume_ema_total_steps or 0)
        self.ema.total_steps = max(scheduled_ema_steps, restored_ema_steps, self.ema.step_count + 1)
        self._lr_total_steps = max(scheduled_ema_steps, 1)
        # A short finite run must still reach the base LR before it ends;
        # otherwise a one-epoch smoke/finetune would stay at warmup LR.
        requested_warmup_steps = max(int(round(float(self.warmup_epochs) * steps_per_epoch)), 0)
        self._warmup_steps = min(requested_warmup_steps, self._lr_total_steps)

        if frozen_epochs > 0:
            freeze_backbone = self.start_epoch < frozen_epochs
            phase = f"epoch {self.start_epoch} resume state" if self.start_epoch > 0 else "training start"
            self._set_backbone_frozen(freeze_backbone, context=phase)

        for param_group in self.optimizer.param_groups:
            param_group.setdefault("initial_lr", self.lr0)

        scheduler_last_epoch = -1
        if self.resume_ckpt is not None and self.start_epoch > 0:
            ckpt_scheduler = self._normalize_lr_scheduler(self.resume_ckpt.get("lr_scheduler", "Cosine"))
            expected_unit = self._scheduler_step_unit()
            can_restore_scheduler = (
                "scheduler_state" in self.resume_ckpt
                and ckpt_scheduler == self.lr_scheduler
                and self.resume_ckpt.get("scheduler_step_unit") == expected_unit
            )
            if not can_restore_scheduler and self.lr_scheduler == "Cosine":
                scheduler_last_epoch = max(self.optimizer_step_count - 1, -1)

        scheduler = self._build_lr_scheduler(scheduler_last_epoch)
        self._active_scheduler = scheduler
        self._restore_lr_scheduler_state(scheduler)

        close_mosaic_boundary = self._close_mosaic_boundary()
        close_mosaic_workers_refreshed = False

        epoch = self.start_epoch
        last_val_stats = None
        while self.infinite_epochs or epoch < self.epochs:
            epoch_start = time.time()
            epoch_images = 0
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            self.model.train()
            if self.augmentation_passes is not None and self._active_augmentation_pass != epoch:
                self._set_train_augmentation_pass(epoch)
                self._rebuild_train_prefetcher(train_sampler, train_shuffle, num_workers, train_pf_factor)
                print(
                    f"[pass {epoch + 1}/{len(self.augmentation_passes)}] "
                    f"augmentation={self.augmentation_pass_names[epoch]}"
                )
            if frozen_epochs > 0 and epoch < frozen_epochs:
                self._set_backbone_frozen(True)
            desired_imgsz = self._imgsz_for_epoch(epoch)
            if desired_imgsz != self.current_imgsz:
                self.current_imgsz = desired_imgsz
                self._set_dataset_imgsz(self.full_train_set, desired_imgsz)
                self._set_dataset_imgsz(self.full_val_dataset, desired_imgsz)
                self._rebuild_train_prefetcher(train_sampler, train_shuffle, num_workers, train_pf_factor)
                print(f"[schedule] epoch={epoch+1} imgsz={desired_imgsz}")
            chunk_note = ""

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
                "regpc": 0.0, "dflpc": 0.0, "cons": 0.0, "wcls": 0.0, "wreg": 0.0, "wdfl": 0.0, "wiou": 0.0,
                "wcons": 0.0, "awcls": 0.0, "awbox": 0.0, "awdfl": 0.0, "awcons": 0.0,
                "quality": 0.0, "wquality": 0.0, "awquality": 0.0,
                "qmix": 0.0, "hneg": 0.0, "pos_o2o": 0.0, "pos_o2m": 0.0,
                "o2m_active": 0.0, "exact_o2o": 0.0,
            }
            running_accumulator = None
            n_steps = 0
            epoch_had_optimizer_step = False
            last_accum_step = -1  # Track whether last batch was an accum boundary
            last_batch_index = -1
            amp_finite_window = None
            amp_finite_entries = []
            self.optimizer.zero_grad()
            total_batches = max(len(current_prefetcher), 1)
            final_accum_size = total_batches % self.accum_steps or self.accum_steps
            last_live_status_at = epoch_start
            self._last_iter_rate_step = self.global_step
            self._last_iter_rate_at = epoch_start
            self._write_live_status(
                self._live_status_payload(epoch, 0, total_batches, running, n_steps, elapsed_sec=0.0)
            )

            progress_unit = "Pass" if self.augmentation_passes is not None else "Epoch"
            pbar = tqdm(
                current_prefetcher,
                desc=f"{progress_unit} {epoch+1:>3}/{self._epoch_label()}",
                leave=False,
            )
            for i, (imgs, targets) in enumerate(pbar):
                last_batch_index = i
                input_tensor = next(iter(imgs.values())) if isinstance(imgs, dict) else imgs
                batch_images = int(input_tensor.shape[0])
                image_height, image_width = map(int, input_tensor.shape[-2:])
                epoch_images += batch_images
                self.images_seen += batch_images
                batch_labels = sum(
                    int(target.get("labels", ()).numel())
                    if torch.is_tensor(target.get("labels"))
                    else len(target.get("labels", ()))
                    for target in targets
                )
                self.labels_seen += batch_labels
                reference_resolution = max(int(self.current_imgsz), 1)
                self.pixel_equivalent_images += (
                    batch_images
                    * image_height
                    * image_width
                    / float(reference_resolution * reference_resolution)
                )

                with torch.amp.autocast(self.device.type, enabled=self.use_amp, dtype=torch.float16):
                    use_o2m = self._should_compute_o2m(self.criterion, epoch, self.global_step)
                    outputs = self.model(imgs, use_o2m=use_o2m)
                    criterion_kwargs = {
                        "current_epoch": epoch,
                        "max_epochs": self.epochs,
                        "global_step": self.global_step,
                    }
                    loss_dict = self.criterion(
                        outputs,
                        targets,
                        **criterion_kwargs,
                    )
                    if self.device.type == "cuda" and self.use_amp:
                        finite_flag = self._loss_dict_finite_flag(loss_dict, self.device)
                        amp_finite_window = finite_flag if amp_finite_window is None else (amp_finite_window & finite_flag)
                        amp_finite_entries.append((
                            epoch,
                            i,
                            self.global_step,
                            {
                                name: value.detach() if torch.is_tensor(value) else value
                                for name, value in loss_dict.items()
                            },
                        ))
                    if self._should_eager_finite_loss_check(self.global_step):
                        self._assert_finite_loss_dict(loss_dict, epoch, i, self.global_step)
                    in_final_partial_accum = (
                        final_accum_size < self.accum_steps
                        and i >= total_batches - final_accum_size
                    )
                    loss_divisor = (
                        final_accum_size
                        if in_final_partial_accum
                        else self.accum_steps
                    )
                    loss = loss_dict["loss"] / loss_divisor
                    accumulation_boundary = (i + 1) % self.accum_steps == 0

                self.scaler.scale(loss).backward()

                if accumulation_boundary:
                    stepped = self._optimizer_step(scheduler)
                    if self.device.type == "cuda" and self.use_amp and not stepped:
                        self._assert_finite_loss_window(amp_finite_window, amp_finite_entries)
                    amp_finite_window = None
                    amp_finite_entries = []
                    epoch_had_optimizer_step = stepped or epoch_had_optimizer_step
                    last_accum_step = i


                metric_values = self._loss_metric_tensor(loss_dict)
                if running_accumulator is None:
                    running_accumulator = torch.zeros_like(metric_values)
                running_accumulator.add_(metric_values)
                n_steps += 1
                self.global_step += 1

                now_live_status = time.time()
                if (
                    now_live_status - last_live_status_at >= self._LIVE_STATUS_INTERVAL_SEC
                    or i + 1 >= total_batches
                ):
                    self._flush_running_metrics(
                        running,
                        running_accumulator,
                    )
                    last_live_status_at = now_live_status
                    denom = max(n_steps, 1)
                    # Snapshot avoids mixing values if a future callback mutates
                    # the running aggregate while tqdm formats its postfix.
                    running_snapshot = dict(running)
                    regression_name = self._regression_loss_name()
                    pbar.set_postfix(
                        loss=f"{running_snapshot['loss'] / denom:.2f}",
                        cls=f"{running_snapshot['cls'] / denom:.2f}",
                        **{regression_name: f"{running_snapshot['regpc'] / denom:.2f}"},
                        iou=f"{running_snapshot['iou'] / denom:.2f}",
                        lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    )
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

            self._flush_running_metrics(
                running,
                running_accumulator,
            )

            # Flush remaining accumulated gradients only if last batch wasn't an accum boundary
            if last_accum_step != last_batch_index and n_steps > 0:
                stepped = self._optimizer_step(scheduler)
                if self.device.type == "cuda" and self.use_amp and not stepped:
                    self._assert_finite_loss_window(amp_finite_window, amp_finite_entries)
                amp_finite_window = None
                amp_finite_entries = []
                epoch_had_optimizer_step = stepped or epoch_had_optimizer_step

            # Log training metrics per epoch
            regression_name = self._regression_loss_name()
            log_entry = {
                "epoch": epoch + 1,
                "imgsz": int(self.current_imgsz),
                "loss": round(running['loss']/max(n_steps,1), 4),
                "cls_loss": round(running['cls']/max(n_steps,1), 4),
                f"{regression_name}_loss": round(running['regpc']/max(n_steps,1), 4),
                "iou_loss": round(running['iou']/max(n_steps,1), 4),
                "lr": self.optimizer.param_groups[0]['lr'],
            }
            is_last = (not self.infinite_epochs) and (epoch == self.epochs - 1)
            will_validate = (epoch + 1) % self.val_period == 0 or is_last

            # Persist the completed optimization epoch before entering
            # validation. Validation is intentionally strict and may fail on
            # malformed/non-finite model output or external resource errors;
            # none of those failures should discard an otherwise completed
            # training epoch. The normal post-validation save below overwrites
            # this checkpoint with updated best-score/scheduler metadata.
            if will_validate and self.save_last:
                self._save_last_checkpoint(scheduler, epoch)

            # Per-Epoch Validation Loop (Controlled by val_period)
            if will_validate:
                self.active_val_dataset, val_mode = self._val_dataset_for_epoch(epoch)
                self.val_dataset = self.active_val_dataset
                val_workers = max(0, min(int(num_workers), 2))
                val_pf_factor = 1 if val_workers > 0 else None
                val_loader = self._get_validation_loader(
                    self.active_val_dataset,
                    self.val_batch_size,
                    val_workers,
                    val_pf_factor,
                )
                val_stats = self._validate_with_retries(
                    val_loader,
                    epoch,
                    val_workers,
                    val_pf_factor,
                )
                val_stats["val_mode"] = val_mode
                last_val_stats = dict(val_stats)
                log_entry.update(self._compact_validation_metrics(val_stats, self.coco_max_dets))
                per_class = val_stats.get("per_class")
                if isinstance(per_class, dict):
                    log_entry["per_class"] = per_class
                    defined = [
                        value
                        for value in per_class.values()
                        if int(value.get("num_gt", 0)) > 0
                    ]
                    defined.sort(
                        key=lambda value: (
                            int(value.get("num_gt", 0)),
                            int(value.get("category_id", 0)),
                        )
                    )
                    rare_count = max(math.ceil(len(defined) * 0.25), 1)
                    rare_recall = [
                        float(value[f"ar{self.coco_max_dets}"])
                        for value in defined[:rare_count]
                        if value.get(f"ar{self.coco_max_dets}") is not None
                    ]
                    log_entry["rare_class_count"] = rare_count
                    log_entry["rare_class_recall"] = (
                        sum(rare_recall) / len(rare_recall)
                        if rare_recall
                        else None
                    )

                # Only a full, protocol-comparable evaluation may select a
                # checkpoint or drive a validation-based scheduler.
                if self.best_metric not in val_stats:
                    raise KeyError(
                        f"Validation did not produce configured best_metric={self.best_metric!r}; "
                        f"available metrics: {sorted(val_stats)}"
                    )
                metric_score = float(val_stats[self.best_metric])
                if val_mode == "full" and self._should_save_best(
                    metric_score, self.best_map, self.best_checkpoint_path
                ):
                    self.best_map = metric_score
                    self._save_best_checkpoint(epoch)
                if val_mode == "full" and self.lr_scheduler == "LRDropDown":
                    old_lr = float(self.optimizer.param_groups[0]["lr"])
                    scheduler.step(metric_score)
                    new_lr = float(self.optimizer.param_groups[0]["lr"])
                    log_entry["lr"] = new_lr
                    if new_lr < old_lr:
                        print(
                            f"[lr] LRDropDown {old_lr:.2e} -> {new_lr:.2e} "
                            f"avg_move={scheduler.last_average_movement:.4f} "
                            f"avg={scheduler.last_average:.4f} prev_avg={scheduler.last_previous_average:.4f}"
                        )

            epoch_seconds = time.time() - epoch_start
            self.training_wall_time_sec += epoch_seconds
            log_entry["timestamp"] = datetime.now().isoformat()
            log_entry["epoch_time_sec"] = round(epoch_seconds, 2)
            log_entry["images_seen"] = int(self.images_seen)
            log_entry["labels_seen"] = int(self.labels_seen)
            log_entry["optimizer_steps"] = int(self.optimizer_step_count)
            log_entry["pixel_equivalent_images"] = round(
                self.pixel_equivalent_images,
                3,
            )
            log_entry["full_dataset_equivalent_epochs"] = round(
                self.images_seen / max(len(self.full_train_set), 1),
                6,
            )
            log_entry["pixel_equivalent_epochs"] = round(
                self.pixel_equivalent_images
                / max(len(self.full_train_set), 1),
                6,
            )
            log_entry["training_wall_time_sec"] = round(
                self.training_wall_time_sec,
                3,
            )
            log_entry["images_per_sec"] = round(epoch_images / max(epoch_seconds, 1e-9), 2)
            if will_validate:
                measured_ap = float(log_entry["mAP50_95"])
                elapsed_hours = self.training_wall_time_sec / 3600.0
                log_entry["ap_per_hour"] = (
                    measured_ap / elapsed_hours
                    if elapsed_hours > 0
                    else None
                )
                for threshold_index in range(1, 11):
                    threshold = threshold_index * 0.05
                    key = f"{threshold:.2f}"
                    if measured_ap >= threshold and key not in self.ap_threshold_times:
                        self.ap_threshold_times[key] = {
                            "wall_time_sec": float(
                                self.training_wall_time_sec
                            ),
                            "optimizer_steps": int(
                                self.optimizer_step_count
                            ),
                            "images_seen": int(self.images_seen),
                            "epoch": int(epoch + 1),
                        }
                log_entry["ap_threshold_times"] = dict(
                    self.ap_threshold_times
                )
            if self.device.type == "cuda":
                log_entry["peak_training_vram_mb"] = round(torch.cuda.max_memory_allocated(self.device) / 2**20, 2)
            if will_validate:
                val_text = (
                    f"val({log_entry.get('val_mode', 'full')}) "
                    f"P@{self.operating_conf:g}={log_entry['precision']:.4f} "
                    f"R@{self.operating_conf:g}={log_entry['recall']:.4f} "
                    f"mAP@.50={log_entry['mAP50']:.4f} mAP@.50:.95={log_entry['mAP50_95']:.4f} "
                    f"AR{self.coco_max_dets}={log_entry[f'coco_AR{self.coco_max_dets}']:.4f}"
                )
            else:
                next_val = ((epoch + 1) // self.val_period + 1) * self.val_period
                if not self.infinite_epochs:
                    next_val = min(next_val, self.epochs)
                val_text = f"val=skip next={next_val}"
            log_unit = "pass" if self.augmentation_passes is not None else "epoch"
            print(
                f"[{log_unit} {epoch+1}/{self._epoch_label()}{chunk_note}] "
                f"imgsz={self.current_imgsz} loss={log_entry['loss']:.4f} cls={log_entry['cls_loss']:.4f} "
                f"{regression_name}={log_entry[f'{regression_name}_loss']:.4f} "
                f"iou={log_entry['iou_loss']:.4f} lr={log_entry['lr']:.2e} "
                f"{val_text} time={epoch_seconds:.1f}s"
            )

            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, default=str) + "\n")
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
            epoch += 1

        self._write_live_status({
            "running": False,
            "epoch": int(self.epochs),
            "epochs": int(self.epochs),
            "timestamp": datetime.now().isoformat(),
        })
        result = {
            "status": "completed",
            "training_run_id": self.training_run_id,
            "save_dir": os.fspath(self.save_dir),
            "epochs_requested": int(self.epochs),
            "epochs_completed": int(epoch),
            "global_step": int(self.global_step),
            "optimizer_steps": int(self.optimizer_step_count),
            "images_seen": int(self.images_seen),
            "labels_seen": int(self.labels_seen),
            "pixel_equivalent_images": float(
                self.pixel_equivalent_images
            ),
            "full_dataset_equivalent_epochs": (
                self.images_seen / max(len(self.full_train_set), 1)
            ),
            "pixel_equivalent_epochs": (
                self.pixel_equivalent_images
                / max(len(self.full_train_set), 1)
            ),
            "training_wall_time_sec": float(
                self.training_wall_time_sec
            ),
            "ap_threshold_times": dict(self.ap_threshold_times),
            "sampling_strategy": "uniform",
            "verified_batch_size_lower_bound": int(self.batch_size),
            "best_metric": self.best_metric,
            "best_score": float(self.best_map) if math.isfinite(self.best_map) else None,
            "best_checkpoint": self.best_checkpoint_path if os.path.isfile(self.best_checkpoint_path) else None,
            "last_checkpoint": self.last_checkpoint_path if os.path.isfile(self.last_checkpoint_path) else None,
            "last_validation": last_val_stats,
        }
        self.last_train_result = result
        return result


__all__ = ["TrainingLoopMixin"]
