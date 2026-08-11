"""Exact V1 finite-loss, AMP, optimizer-step, EMA, and telemetry rules."""

import math
import torch


class OptimizationProtocolMixin:
    @staticmethod
    def _assert_finite_loss_dict(loss_dict, epoch, step, global_step):
        bad = []
        tensor_groups = {}
        for name, value in loss_dict.items():
            if not str(name).startswith("loss"):
                continue
            if torch.is_tensor(value):
                tensor_groups.setdefault(value.device, []).append((name, value))
            elif isinstance(value, (float, int)) and not math.isfinite(float(value)):
                bad.append(f"{name}={float(value)}")

        # A scalar host read is necessary to fail before corrupt gradients are
        # applied, but do it once per device rather than synchronizing every
        # individual loss component on every batch.
        for entries in tensor_groups.values():
            all_finite = torch.stack([
                torch.isfinite(value.detach()).all() for _, value in entries
            ]).all()
            if bool(all_finite):
                continue
            for name, value in entries:
                if bool(torch.isfinite(value.detach()).all()):
                    continue
                if value.numel() == 1:
                    shown = float(value.detach().float().cpu())
                else:
                    shown = f"shape={tuple(value.shape)}"
                bad.append(f"{name}={shown}")
        if bad:
            raise FloatingPointError(
                "Non-finite loss at "
                f"epoch {int(epoch) + 1} step {int(step) + 1} global_step {int(global_step)}: "
                + ", ".join(bad)
            )

    @staticmethod
    def _loss_dict_finite_flag(loss_dict, device):
        """Check the differentiable total; expand components only on failure."""
        value = loss_dict.get("loss")
        if torch.is_tensor(value):
            return torch.isfinite(value.detach()).all()
        if isinstance(value, (float, int)):
            return torch.as_tensor(math.isfinite(float(value)), device=device)
        if value is None:
            return torch.ones((), dtype=torch.bool, device=device)
        raise TypeError("loss_dict['loss'] must be a tensor or finite scalar")

    def _should_eager_finite_loss_check(self, global_step):
        """Keep strict checks everywhere except CUDA AMP's scaler-guarded hot path."""
        if self.device.type != "cuda" or not self.use_amp:
            return True
        return int(global_step) % self._AMP_FINITE_CHECK_INTERVAL == 0

    @classmethod
    def _assert_finite_loss_window(cls, finite_flag, entries):
        """Diagnose a scaler-rejected accumulation window only when it is bad."""
        if bool(finite_flag):
            return
        for epoch, step, global_step, loss_dict in entries:
            cls._assert_finite_loss_dict(loss_dict, epoch, step, global_step)
        raise FloatingPointError("AMP scaler rejected a non-finite loss accumulation window.")

    @staticmethod
    def _loss_metric_tensor(loss_dict):
        """Pack hot-path telemetry into one detached device tensor.

        Individual ``.item()`` calls synchronize CUDA. Keeping the aggregate
        on-device lets model work and input prefetch proceed until the bounded
        progress/status reporting cadence needs one host transfer.
        """
        reference = loss_dict["loss"]
        if torch.is_tensor(reference):
            device = reference.device
            # Loss values may be autocast FP16; telemetry accumulation should
            # not overflow before the next one-second host flush.
            dtype = torch.float32
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        def scalar(value):
            if torch.is_tensor(value):
                return value.detach().to(device=device, dtype=dtype).reshape(())
            return torch.as_tensor(value, device=device, dtype=dtype).reshape(())

        return torch.stack((
            scalar(loss_dict["loss"]),
            scalar(loss_dict["loss_cls"]),
            scalar(loss_dict["loss_box"]),
            scalar(loss_dict.get("loss_reg_per_coord", loss_dict["loss_box"])),
            scalar(loss_dict["loss_dfl_per_coord"]),
            scalar(loss_dict["loss_ciou"]),
            scalar(loss_dict["loss_consistency"]),
            scalar(loss_dict["loss_cls_weighted"]),
            scalar(loss_dict.get("loss_reg_weighted", loss_dict["loss_dfl_weighted"])),
            scalar(loss_dict["loss_dfl_weighted"]),
            scalar(loss_dict["loss_ciou_weighted"]),
            scalar(loss_dict["loss_consistency_weighted"]),
            scalar(loss_dict.get("loss_quality", 0.0)),
            scalar(loss_dict.get("loss_quality_weighted", 0.0)),
            scalar(loss_dict["active_w_cls"]),
            scalar(loss_dict["active_w_box"]),
            scalar(loss_dict["active_w_dfl"]),
            scalar(loss_dict["active_w_consistency"]),
            scalar(loss_dict.get("active_w_quality", 0.0)),
            scalar(loss_dict["quality_mix"]),
            scalar(loss_dict["hard_negative_weight"]),
            scalar(loss_dict["num_pos_o2o"]),
            scalar(loss_dict["num_pos_o2m"]),
            scalar(loss_dict.get("o2m_active", 0.0)),
            scalar(loss_dict.get("exact_o2o_active", 0.0)),
        ))

    @classmethod
    def _flush_running_metrics(cls, running, accumulator, metric_names=None):
        """Transfer one packed metric vector to host and reset it in-place."""
        if accumulator is None:
            return
        values = accumulator.detach().cpu().tolist()
        names = cls._RUNNING_METRIC_NAMES if metric_names is None else metric_names
        for key, value in zip(names, values):
            running[key] += float(value)
        accumulator.zero_()

    def _install_optimizer_step_tracker(self):
        """Observe real optimizer calls without GradScaler.get_scale() syncs."""
        self._optimizer_invocation_count = 0

        def record_step(_optimizer, _args, _kwargs):
            self._optimizer_invocation_count += 1

        register = getattr(self.optimizer, "register_step_post_hook", None)
        self._optimizer_step_tracker_handle = (
            register(record_step) if callable(register) else None
        )

    @staticmethod
    def _should_compute_o2m(criterion, current_epoch, global_step):
        """Use the exact curriculum weight that the loss will use later."""
        use_o2m = getattr(criterion, "use_o2m", None)
        if not callable(use_o2m):
            return True
        branch_weights = getattr(criterion, "branch_weights", None)
        if not callable(branch_weights):
            return bool(use_o2m(current_epoch, global_step))
        _o2o_weight, active_o2m_weight = branch_weights(current_epoch)
        return bool(
            use_o2m(
                current_epoch,
                global_step,
                active_o2m_weight=active_o2m_weight,
            )
        )

    def _optimizer_step(self, scheduler=None):
        """Run one AMP-aware optimizer step and report whether it really happened."""
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        invocation_before = getattr(self, "_optimizer_invocation_count", None)
        # Compatibility fallback is used only by external/fake optimizers that
        # do not support step hooks. Production torch optimizers use the
        # zero-synchronization invocation tracker.
        old_scale = (
            self.scaler.get_scale()
            if invocation_before is None and hasattr(self.scaler, "get_scale")
            else None
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        if invocation_before is not None:
            stepped = self._optimizer_invocation_count > invocation_before
        elif old_scale is None:
            stepped = True
        else:
            new_scale = self.scaler.get_scale()
            stepped = float(new_scale) >= float(old_scale)
        if stepped:
            self.ema.update(self.model)
            self.optimizer_step_count = int(getattr(self, "optimizer_step_count", 0)) + 1
            if scheduler is not None and getattr(self, "lr_scheduler", "Cosine") == "Cosine":
                scheduler.step()
        return stepped


__all__ = ["OptimizationProtocolMixin"]
