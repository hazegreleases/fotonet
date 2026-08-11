"""Exact V1 validation selection, retry, and reporting policy."""

import gc
import math
import time
import torch
from torch.utils.data import DataLoader
from fotonet.engine.runtime import configure_data_worker


class ValidationProtocolMixin:
    @classmethod
    def _compact_validation_metrics(cls, metrics, max_dets=100):
        """Keep only epoch-level validation signals useful for run tracking."""
        compact = {
            key: metrics[key]
            for key in cls._LOG_VALIDATION_METRICS
            if key in metrics and metrics[key] is not None
        }
        dynamic_ar = f"coco_AR{int(max_dets)}"
        if dynamic_ar not in compact and metrics.get(dynamic_ar) is not None:
            compact[dynamic_ar] = metrics[dynamic_ar]
        for key in ("operating_conf", "operating_iou", "val_mode", "val_images"):
            if metrics.get(key) is not None:
                compact[key] = metrics[key]
        return compact

    def _make_val_loader(self, val_dataset, batch_size, num_workers, pf_factor, persistent_workers=True):
        return DataLoader(
            val_dataset, batch_size=int(batch_size), shuffle=False,
            collate_fn=self._collate_fn, num_workers=num_workers,
            pin_memory=self.pin_memory, prefetch_factor=pf_factor,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            worker_init_fn=configure_data_worker if num_workers > 0 else None,
        )

    def _get_validation_loader(
        self,
        val_dataset,
        batch_size,
        num_workers,
        pf_factor,
    ):
        """Reuse validation workers for an unchanged dataset/protocol."""
        key = (
            id(val_dataset),
            int(batch_size),
            int(num_workers),
            None if pf_factor is None else int(pf_factor),
        )
        loaders = getattr(self, "_validation_loaders", None)
        if loaders is None:
            loaders = {}
            self._validation_loaders = loaders
        loader = loaders.get(key)
        if loader is None:
            loader = self._make_val_loader(
                val_dataset,
                batch_size,
                num_workers,
                pf_factor,
                persistent_workers=num_workers > 0,
            )
            loaders[key] = loader
        return loader

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

    def _retry_validation_after_host_oom(self, error_message, epoch):
        original_val_conf = float(self.val_conf)
        fallback_floors = []
        for floor in (0.05, 0.10, 0.20, float(self.operating_conf)):
            floor = min(float(floor), float(self.operating_conf))
            if floor > original_val_conf and floor not in fallback_floors:
                fallback_floors.append(floor)

        print(f"[WARN] COCO validation exhausted host RAM: {error_message}")
        print("[WARN] Retrying with a bounded score floor and zero validation workers.")
        self._cleanup_cuda_after_error()
        last_error = None
        for floor in fallback_floors:
            self.val_conf = floor
            safe_loader = self._make_val_loader(
                self.active_val_dataset,
                batch_size=min(max(int(self.val_batch_size), 1), 4),
                num_workers=0,
                pf_factor=None,
                persistent_workers=False,
            )
            try:
                print(f"[INFO] Host-memory validation retry: val_conf={floor:g}, workers=0")
                stats = self.validate(safe_loader, epoch)
            except MemoryError as exc:
                last_error = str(exc)
                exc.__traceback__ = None
                self._cleanup_cuda_after_error()
                print(f"[WARN] Validation retry with val_conf={floor:g} exhausted host RAM.")
                continue

            if math.isfinite(self.best_map):
                print(
                    "[WARN] Validation score-floor protocol changed; "
                    "resetting the incomparable best score."
                )
                self.best_map = float("-inf")
            stats["validation_retried"] = True
            stats["validation_retry_reason"] = "host_memory"
            stats["validation_retry_val_conf"] = float(floor)
            return stats

        self.val_conf = original_val_conf
        detail = last_error or error_message
        raise MemoryError(
            "COCO validation exhausted host RAM even after bounded-score retries: "
            f"{detail}"
        )

    def _validate_with_retries(self, loader, epoch, num_workers, pf_factor):
        try:
            return self.validate(loader, epoch)
        except FloatingPointError as exc:
            # A finite FP32 model can overflow inside an FP16-only inference
            # operator (for example, attention matmul) even though AMP
            # training remains healthy under GradScaler.  Validation has no
            # scaler, so retry the complete metric pass in FP32.  Do not hide
            # genuinely corrupt weights: a non-finite FP32 retry is raised.
            if not self.val_amp:
                raise

            error_message = str(exc)
            # Drop frames retaining the partially accumulated metric payload
            # before beginning a complete FP32 pass.
            exc.__traceback__ = None
            print(f"[WARN] AMP validation produced a non-finite output: {error_message}")
            print("[WARN] Retrying the complete validation pass in FP32.")
            original_val_amp = self.val_amp
            self.val_amp = False
            self._cleanup_cuda_after_error()
            try:
                stats = self._validate_with_retries(
                    loader,
                    epoch,
                    num_workers,
                    pf_factor,
                )
            finally:
                self.val_amp = original_val_amp
            secondary_reason = stats.get("validation_retry_reason")
            if secondary_reason and secondary_reason != "nonfinite_amp":
                stats["validation_retry_secondary_reason"] = secondary_reason
            stats["validation_retried"] = True
            stats["validation_retry_reason"] = "nonfinite_amp"
            stats["validation_retry_val_amp"] = False
            return stats
        except MemoryError as exc:
            error_message = str(exc)
            # NumPy's ArrayMemoryError traceback retains the large COCOeval
            # arrays that triggered it. Break that chain before retrying.
            exc.__traceback__ = None
            return self._retry_validation_after_host_oom(error_message, epoch)
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
                # Keep retry metrics numerically comparable with the configured
                # validation protocol. Lowering batch/workers is a memory
                # recovery; silently switching precision can alter thresholded
                # AP and should not choose a best checkpoint.
                self.val_amp = original_val_amp
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

    def validate(self, loader, epoch):
        """Validate EMA through the same protocol used by ``Fotonet.val``."""
        del epoch
        from fotonet.engine.validation import evaluate_detection_model

        val_start = time.time()
        metrics = evaluate_detection_model(
            self.ema.ema,
            loader,
            self.device,
            self.nc,
            conf=self.val_conf,
            coco_max_dets=self.coco_max_dets,
            operating_conf=self.operating_conf,
            operating_iou=self.operating_iou,
            amp=self.val_amp,
            class_names=self.class_names,
        )
        val_time = max(time.time() - val_start, 1e-9)
        metrics["val_time_sec"] = float(val_time)
        metrics["avg_fps"] = float(metrics["val_images"] / val_time)
        return metrics


__all__ = ["ValidationProtocolMixin"]
