"""Exact V1 scheduler implementations and progression semantics."""

import math
import numpy as np
import torch
import torch.optim as optim


class _AverageMovementLRDropDown:
    """Validation rolling-average movement LR dropper with resumable state."""

    def __init__(
        self,
        optimizer,
        mode="max",
        factor=0.92,
        patience=5,
        threshold=0.001,
        min_lr=1e-5,
        initial_best=None,
    ):
        self.optimizer = optimizer
        self.mode = str(mode or "max").lower()
        if self.mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.factor = float(factor)
        if not 0.0 < self.factor < 1.0:
            raise ValueError("factor must be between 0 and 1")
        self.patience = max(int(patience or 1), 1)
        self.threshold = float(threshold)
        self.min_lr = float(min_lr)
        self.history = [] if initial_best is None else [float(initial_best)]
        self.last_epoch = -1
        self.last_metric = None
        self.last_average = None
        self.last_previous_average = None
        self.last_average_movement = None
        self.num_drops = 0

    def get_last_lr(self):
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def step(self, metric=None):
        self.last_epoch += 1
        if metric is None:
            return self.get_last_lr()

        metric = float(metric)
        if not math.isfinite(metric):
            return self.get_last_lr()

        self.last_metric = metric
        self.history.append(metric)

        if len(self.history) < self.patience + 1:
            return self.get_last_lr()

        recent_values = self.history[-self.patience:]
        previous_values = self.history[-self.patience - 1:-1]

        recent_avg = float(np.mean(np.asarray(recent_values, dtype=np.float64)))
        previous_avg = float(np.mean(np.asarray(previous_values, dtype=np.float64)))
        raw_movement = recent_avg - previous_avg
        average_movement = raw_movement if self.mode == "max" else -raw_movement

        self.last_average = recent_avg
        self.last_previous_average = previous_avg
        self.last_average_movement = average_movement

        if average_movement < self.threshold:
            old_lrs = self.get_last_lr()
            for group in self.optimizer.param_groups:
                group["lr"] = max(float(group["lr"]) * self.factor, self.min_lr)
            new_lrs = self.get_last_lr()
            if any(new < old for new, old in zip(new_lrs, old_lrs)):
                self.num_drops += 1
        return self.get_last_lr()

    def state_dict(self):
        return {
            "mode": self.mode,
            "factor": self.factor,
            "patience": self.patience,
            "threshold": self.threshold,
            "min_lr": self.min_lr,
            "history": list(self.history),
            "last_epoch": self.last_epoch,
            "last_metric": self.last_metric,
            "last_average": self.last_average,
            "last_previous_average": self.last_previous_average,
            "last_average_movement": self.last_average_movement,
            "num_drops": self.num_drops,
        }

    def load_state_dict(self, state):
        self.mode = str(state.get("mode", self.mode)).lower()
        self.factor = float(state.get("factor", self.factor))
        self.patience = max(int(state.get("patience", self.patience)), 1)
        self.threshold = float(state.get("threshold", self.threshold))
        self.min_lr = float(state.get("min_lr", self.min_lr))
        self.history = [float(x) for x in state.get("history", self.history)]
        if not self.history and state.get("best_median") is not None:
            self.history = [float(state["best_median"])]
        self.last_epoch = int(state.get("last_epoch", self.last_epoch))
        last_metric = state.get("last_metric", self.last_metric)
        self.last_metric = None if last_metric is None else float(last_metric)
        last_average = state.get("last_average", state.get("last_median", self.last_average))
        self.last_average = None if last_average is None else float(last_average)
        last_previous_average = state.get("last_previous_average", self.last_previous_average)
        self.last_previous_average = None if last_previous_average is None else float(last_previous_average)
        last_average_movement = state.get("last_average_movement", self.last_average_movement)
        self.last_average_movement = None if last_average_movement is None else float(last_average_movement)
        self.num_drops = int(state.get("num_drops", self.num_drops))

_MedianLRDropDown = _AverageMovementLRDropDown


class SchedulesProtocolMixin:
    def _restore_scheduler_lr(self, scheduler):
        """Scheduler state restores counters; explicitly sync optimizer LRs too."""
        for group, lr in zip(self.optimizer.param_groups, scheduler.get_last_lr()):
            group["lr"] = lr

    def _scheduler_step_unit(self):
        if self.lr_scheduler == "LRDropDown":
            return "validation"
        return "optimizer"

    def _build_lr_scheduler(self, scheduler_last_epoch=-1):
        if self.lr_scheduler == "LRDropDown":
            initial_best = self.best_map if self.best_map > 0 else None
            return _AverageMovementLRDropDown(
                self.optimizer,
                mode="max",
                factor=self.lr_drop_factor,
                patience=self.lr_drop_patience,
                threshold=self.lr_drop_threshold,
                min_lr=self.lr_drop_min_lr,
                initial_best=initial_best,
            )
        return optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._lr_lambda,
            last_epoch=scheduler_last_epoch,
        )

    def _restore_lr_scheduler_state(self, scheduler):
        if self.resume_ckpt is None:
            return
        ckpt_scheduler = self._normalize_lr_scheduler(self.resume_ckpt.get("lr_scheduler", "Cosine"))
        expected_unit = self._scheduler_step_unit()
        checkpoint_unit = self.resume_ckpt.get("scheduler_step_unit")
        if (
            "scheduler_state" in self.resume_ckpt
            and ckpt_scheduler == self.lr_scheduler
            and checkpoint_unit == expected_unit
        ):
            scheduler.load_state_dict(self.resume_ckpt["scheduler_state"])
            self._restore_scheduler_lr(scheduler)
            print(f"[INFO] Restored {self.lr_scheduler} scheduler state.")
            return
        raise ValueError(
            "Resume checkpoint scheduler identity/state does not match the current protocol"
        )

    def _lr_lambda(self, optimizer_step):
        """Warmup (1% → 100%) then cosine annealing."""
        step = max(int(optimizer_step), 0)
        warmup_steps = max(int(getattr(self, "_warmup_steps", 0)), 0)
        total_steps = max(int(getattr(self, "_lr_total_steps", 1)), 1)
        if warmup_steps > 0 and step < warmup_steps:
            # Start exactly at 1% and arrive at 100% on the final warmup step.
            # ``LambdaLR`` evaluates step 0 during construction, so ``+1``
            # here would accidentally start one warmup increment too high.
            if warmup_steps == 1:
                return 1.0
            return 0.01 + 0.99 * step / float(warmup_steps - 1)

        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        if self.cos_lr:
            cos_val  = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.lrf + (1.0 - self.lrf) * cos_val
        return max(self.lrf, 1.0 - (1.0 - self.lrf) * progress)

    def _set_backbone_frozen(self, freeze, context=None):
        """Keep backbone trainability aligned with the current training phase."""
        orig = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        component = getattr(orig, "backbone", None)
        if component is None:
            return

        for p in component.parameters():
            p.requires_grad_(not freeze)
        for module in component.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval() if freeze else module.train()

        if context:
            state = "frozen" if freeze else "unfrozen"
            print(f"[INFO] Backbone {state} ({context}).")


__all__ = ["SchedulesProtocolMixin", "_AverageMovementLRDropDown", "_MedianLRDropDown"]
