
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
        from fotonet.utils.console import print_train_card

        schedule_txt = " -> ".join(
            f"{int(limit * 100)}%:{size}" for limit, size in self.imgsz_schedule
        )
        val_txt = "full"
        if self.val_subset_size > 0:
            if self.full_val_after > 1.0:
                val_txt = f"subset={self.val_subset_size}"
            else:
                val_txt = f"subset={self.val_subset_size} until {int(self.full_val_after * 100)}%, then full"

        model_name = getattr(self.model, "model_config", {}).get("model_id", getattr(self, "model_id", "fotonet"))
        param_count = sum(p.numel() for p in self.model.parameters()) if hasattr(self.model, "parameters") else None
        device_str = str(getattr(self, "device", "cpu"))
        if getattr(self, "device", None) and getattr(self.device, "type", None) == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    name = torch.cuda.get_device_name(self.device)
                    device_str = f"{self.device} ({name})"
            except Exception:
                pass

        print_train_card(
            model_name=model_name,
            param_count=param_count,
            device=device_str,
            n_train=n_train,
            n_val=n_val,
            epochs=self._epoch_label(),
            imgsz=schedule_txt,
            batch_size=self.batch_size,
            accum_steps=self.accum_steps,
            optimizer_name=self.optimizer_name,
            lr0=self.lr0,
            scheduler_name=self.lr_scheduler,
            amp=self.use_amp,
            val_txt=val_txt,
            resume_path=getattr(self, "resume_info", None),
            recipe_name=getattr(self, "recipe_name", None),
        )


__all__ = ["DiagnosticsMixin"]
