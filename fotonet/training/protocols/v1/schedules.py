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


WSD_DECAY_SHAPES = ("linear", "cosine", "sqrt")


class WSDScheduler:
    """Warmup-Stable-Decay LR schedule stepped per optimizer update.

    Warmup rises linearly from ``min_lr`` to ``lr0``; the stable phase holds
    ``lr0``; decay then falls from the anchored LR back to ``min_lr`` over
    ``decay_iters`` optimizer steps using one of three shapes.  Decay normally
    starts at the scheduled step, and an external trigger (a ``DECAY`` file in
    the run directory or an explicit epoch) may only pull that start earlier -
    the earliest trigger wins.  State round-trips through ``state_dict`` so
    resume is exact.
    """

    def __init__(
        self,
        optimizer,
        lr0: float,
        min_lr: float,
        warmup_iters: int,
        decay_iters: int,
        decay_start_iter: int,
        decay_shape: str = "linear",
    ):
        if decay_shape not in WSD_DECAY_SHAPES:
            raise ValueError(f"decay_shape must be one of {WSD_DECAY_SHAPES}, got {decay_shape!r}")
        self.optimizer = optimizer
        self.lr0 = float(lr0)
        self.min_lr = float(min_lr)
        self.warmup_iters = max(int(warmup_iters), 0)
        self.decay_iters = max(int(decay_iters), 1)
        self.decay_shape = decay_shape
        self.scheduled_decay_start = max(int(decay_start_iter), 0)
        self.earliest_decay_start: int | None = None
        self.anchor_lr: float | None = None
        self.it = 0

    @property
    def decay_start_iter(self) -> int:
        """Effective decay start: the earliest of scheduled and triggered."""
        starts = [self.scheduled_decay_start]
        if self.earliest_decay_start is not None:
            starts.append(self.earliest_decay_start)
        return min(starts)

    def phase(self, it: int | None = None) -> str:
        it = self.it if it is None else int(it)
        if it < self.warmup_iters:
            return "warmup"
        if it < self.decay_start_iter:
            return "stable"
        return "decay"

    def lr_at(self, it: int) -> float:
        it = max(int(it), 0)
        start = self.decay_start_iter
        if it >= start:
            t = (it - start) / float(self.decay_iters)
            t = min(max(t, 0.0), 1.0)
            anchor = self.anchor_lr if self.anchor_lr is not None else self.lr_at_pre_decay(start)
            if self.decay_shape == "linear":
                f = 1.0 - t
            elif self.decay_shape == "cosine":
                f = 0.5 * (1.0 + math.cos(math.pi * t))
            else:  # sqrt: 1 - sqrt(t), the shape that holds LR high longest
                f = 1.0 - math.sqrt(t)
            return self.min_lr + (anchor - self.min_lr) * f
        if it < self.warmup_iters:
            span = max(self.warmup_iters, 1)
            return self.min_lr + (self.lr0 - self.min_lr) * (it / float(span))
        return self.lr0

    def lr_at_pre_decay(self, it: int) -> float:
        """Schedule value ignoring decay (used to anchor the decay curve)."""
        if it < self.warmup_iters:
            span = max(self.warmup_iters, 1)
            return self.min_lr + (self.lr0 - self.min_lr) * (it / float(span))
        return self.lr0

    def step(self, it: int | None = None) -> float:
        if it is None:
            self.it += 1
            it = self.it
        else:
            it = max(int(it), 0)
            self.it = it
        lr = self.lr_at(it)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def trigger_early_decay(self, at_iter: int) -> bool:
        """Pull the decay start earlier if the request precedes the plan."""
        at_iter = max(int(at_iter), 0)
        if at_iter >= self.decay_start_iter:
            return False
        self.earliest_decay_start = at_iter
        self.anchor_lr = self.lr_at_pre_decay(at_iter)
        return True

    def decay_finished(self, it: int | None = None) -> bool:
        it = self.it if it is None else int(it)
        return it >= self.decay_start_iter + self.decay_iters

    def get_last_lr(self):
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {
            "lr0": self.lr0,
            "min_lr": self.min_lr,
            "warmup_iters": self.warmup_iters,
            "decay_iters": self.decay_iters,
            "decay_shape": self.decay_shape,
            "scheduled_decay_start": self.scheduled_decay_start,
            "earliest_decay_start": self.earliest_decay_start,
            "anchor_lr": self.anchor_lr,
            "it": self.it,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.lr0 = float(sd["lr0"])
        self.min_lr = float(sd["min_lr"])
        self.warmup_iters = int(sd["warmup_iters"])
        self.decay_iters = int(sd["decay_iters"])
        self.decay_shape = str(sd["decay_shape"])
        self.scheduled_decay_start = int(sd["scheduled_decay_start"])
        earliest = sd.get("earliest_decay_start")
        self.earliest_decay_start = None if earliest is None else int(earliest)
        anchor = sd.get("anchor_lr")
        self.anchor_lr = None if anchor is None else float(anchor)
        self.it = int(sd.get("it", 0))
        self.step(self.it)


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
        if self.lr_scheduler == "WSD":
            decay_iters = max(int(round(self._wsd_decay_iters)), 1)
            scheduled_start = max(self._lr_total_steps - decay_iters, 0)
            if self.wsd_decay_start_epoch is not None:
                epoch_start = max(int(self.wsd_decay_start_epoch), 0)
                scheduled_start = min(
                    scheduled_start,
                    epoch_start * max(int(getattr(self, "_wsd_steps_per_epoch", 1)), 1),
                )
            return WSDScheduler(
                self.optimizer,
                lr0=self.lr0,
                min_lr=self.lrf * self.lr0,
                warmup_iters=self._warmup_steps,
                decay_iters=decay_iters,
                decay_start_iter=scheduled_start,
                decay_shape=self.wsd_decay_shape,
            )
        return optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._lr_lambda,
            last_epoch=scheduler_last_epoch,
        )

    def _wsd_phase_label(self) -> str:
        """Current WSD phase for telemetry; empty for non-WSD schedules."""
        if self.lr_scheduler != "WSD":
            return ""
        scheduler = getattr(self, "_active_scheduler", None)
        if isinstance(scheduler, WSDScheduler):
            return scheduler.phase()
        return "stable"

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
            return
        raise ValueError(
            "Resume checkpoint scheduler identity/state does not match the current protocol"
        )

    def _lr_lambda(self, optimizer_step):
        """Warmup (1% -> 100%) then cosine annealing."""
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


__all__ = [
    "SchedulesProtocolMixin",
    "WSDScheduler",
    "WSD_DECAY_SHAPES",
    "_AverageMovementLRDropDown",
    "_MedianLRDropDown",
]
