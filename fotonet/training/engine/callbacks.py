
"""Structured training status and console diagnostics."""

import json
import os
import time
from datetime import datetime


class DiagnosticsMixin:
    def _write_live_status(self, payload):
        path = getattr(self, "live_status_file", None)
        if not path:
            return
        try:
            tmp_path = f"{path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp_path, path)
        except OSError:
            pass

    def _live_status_payload(self, epoch, step, total_steps, running, n_steps, extra=None, running_flag=True, elapsed_sec=None):
        denom = max(int(n_steps), 1)
        has_steps = int(n_steps) > 0
        regression_name = self._regression_loss_name()
        regression_loss = round(float(running["regpc"]) / denom, 4) if has_steps else None
        payload = {
            "running": bool(running_flag),
            "epoch": int(epoch) + 1,
            "epochs": int(self.epochs),
            "step": int(step),
            "steps": int(total_steps),
            "epoch_progress": float(min(max(step / max(total_steps, 1), 0.0), 1.0)),
            "imgsz": int(self.current_imgsz),
            "loss": round(float(running["loss"]) / denom, 4) if has_steps else None,
            "cls_loss": round(float(running["cls"]) / denom, 4) if has_steps else None,
            f"{regression_name}_loss": regression_loss,
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

    def _regression_loss_name(self):
        """Return the truthful public name for the active regression loss."""
        reg_max = int(getattr(getattr(self, "criterion", None), "reg_max", 1))
        return "dfl" if reg_max > 1 else "reg"

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
        unit = "passes" if self.augmentation_passes is not None else "epochs"
        print(
            f"\n[train] train={n_train} val={n_val} {unit}={self._epoch_label()} "
            f"batch={self.batch_size} accum={self.accum_steps} imgsz={schedule_txt} "
            f"optimizer={self.optimizer_name} lr={self.lr0:g} scheduler={self.lr_scheduler} "
            f"amp={self.use_amp} validation={val_txt}"
        )



__all__ = ["DiagnosticsMixin"]
