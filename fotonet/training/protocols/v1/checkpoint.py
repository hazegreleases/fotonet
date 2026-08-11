"""Exact V1 checkpoint metadata, resume normalization, and save transport."""

import math
import os
import random
import numpy as np
import torch
from fotonet.utils.config import normalize_class_schema


class CheckpointProtocolMixin:
    @staticmethod
    def _capture_rng_state():
        """Capture reproducible RNG state using only weights-only-safe values."""
        np_state = np.random.get_state()
        # ``np.random.get_state`` contains an ndarray, which forces PyTorch to
        # invoke NumPy pickle reconstruction on load. Persist its raw uint32
        # keys as a tensor instead so a checkpoint made here works with
        # ``torch.load(..., weights_only=True)``.
        np_keys = np.asarray(np_state[1], dtype=np.uint32).copy()
        state = {
            "python": random.getstate(),
            "numpy": {
                "format": "numpy-random-state-v1",
                "bit_generator": str(np_state[0]),
                "keys": torch.from_numpy(np_keys),
                "pos": int(np_state[2]),
                "has_gauss": int(np_state[3]),
                "cached_gaussian": float(np_state[4]),
            },
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state):
        if not state:
            print("[WARN] Resume checkpoint has no RNG state; continuation is not bitwise reproducible.")
            return
        try:
            if state.get("python") is not None:
                random.setstate(state["python"])
            numpy_state = state.get("numpy")
            if numpy_state is not None:
                if isinstance(numpy_state, dict) and numpy_state.get("format") == "numpy-random-state-v1":
                    keys = numpy_state.get("keys")
                    if torch.is_tensor(keys):
                        keys = keys.detach().cpu().numpy()
                    keys = np.asarray(keys, dtype=np.uint32).reshape(-1).copy()
                    np.random.set_state((
                        str(numpy_state["bit_generator"]),
                        keys,
                        int(numpy_state["pos"]),
                        int(numpy_state["has_gauss"]),
                        float(numpy_state["cached_gaussian"]),
                    ))
                else:
                    raise ValueError("Unsupported NumPy RNG state format in checkpoint")
            if state.get("torch") is not None:
                torch.set_rng_state(state["torch"])
            if state.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["cuda"])
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("Cannot safely restore RNG state from resume checkpoint.") from exc

    @staticmethod
    def _should_save_best(metric_score, best_score, best_path):
        """Choose the first artifact even for a valid zero-valued metric."""
        score = float(metric_score)
        if not math.isfinite(score):
            return False
        if not best_path or not os.path.isfile(os.fspath(best_path)):
            return True
        # A zero metric must still be able to replace an empty placeholder.
        if float(best_score) <= 0.0 and score <= 0.0:
            return True
        return score > float(best_score)

    def _apply_resume_class_schema(self, checkpoint):
        """Validate and adopt class names from a full resume checkpoint.

        A matching ``nc`` alone is insufficient: ``[cat, dog]`` and
        ``[dog, cat]`` have the same tensor shapes but invert every supervised
        class. Checkpoints must therefore carry normalized names.
        """
        if not isinstance(checkpoint, dict):
            raise ValueError("resume_ckpt must be a full checkpoint mapping")
        checkpoint_nc, checkpoint_names = normalize_class_schema(
            checkpoint.get("nc", self.nc),
            checkpoint.get("names"),
            context="resume checkpoint",
        )
        if checkpoint_nc != self.nc:
            raise ValueError(
                f"Resume checkpoint nc={checkpoint_nc} does not match model/data nc={self.nc}. "
                "Use a checkpoint trained for the same class set."
            )
        if self._configured_class_names is not None:
            if checkpoint_names is None:
                raise ValueError("Resume checkpoint is missing class names")
            if checkpoint_names != self._configured_class_names:
                raise ValueError(
                    "Resume checkpoint class names/order do not match the data config. "
                    "Refusing to continue with semantically permuted class targets."
                )
            self.class_names = dict(self._configured_class_names)
        elif checkpoint_names is not None:
            self.class_names = dict(checkpoint_names)
        return checkpoint_names

    def _checkpoint_matches_current_model(self, checkpoint):
        """Reject best artifacts from a different architecture/run shape."""
        if not isinstance(checkpoint, dict):
            return False
        raw = self._raw_model() if hasattr(self, "_raw_model") else getattr(self, "model", None)
        if raw is None:
            return False
        if str(checkpoint.get("architecture", "standard")) != "standard":
            return False
        current_values = {
            "nc": getattr(getattr(raw, "head", None), "nc", None),
            "w": next((getattr(raw, name) for name in ("w", "width_multiple") if hasattr(raw, name)), None),
            "d": next((getattr(raw, name) for name in ("d", "depth_multiple") if hasattr(raw, name)), None),
            "use_p2": getattr(raw, "use_p2", None),
            "reg_max": getattr(raw, "reg_max", getattr(getattr(raw, "head", None), "reg_max", None)),
            "p3_extra_blocks": getattr(raw, "p3_extra_blocks", None),
            "p4_extra_blocks": getattr(raw, "p4_extra_blocks", None),
            "p5_extra_blocks": getattr(raw, "p5_extra_blocks", None),
            "p5_gate_blocks": getattr(raw, "p5_gate_blocks", None),
            "arch_version": getattr(raw, "arch_version", None),
            "head_version": getattr(raw, "head_version", getattr(raw, "arch_version", None)),
            "p5_psa_blocks": getattr(raw, "p5_psa_blocks", None),
            "neck_fusion": getattr(raw, "neck_fusion", None),
            "neck_iema": getattr(raw, "neck_iema", None),
            "p2_context_blocks": getattr(raw, "p2_context_blocks", None),
            "p3_context_blocks": getattr(raw, "p3_context_blocks", None),
            "quality_head": getattr(raw, "quality_head", None),
        }
        bool_keys = {"use_p2", "neck_iema", "quality_head"}
        float_keys = {"w", "d"}
        for checkpoint_key, current in current_values.items():
            saved = checkpoint.get(checkpoint_key)
            if saved is None or current is None:
                continue
            if checkpoint_key in bool_keys or isinstance(saved, bool) or isinstance(current, bool):
                if bool(saved) != bool(current):
                    return False
            elif checkpoint_key in float_keys:
                if not math.isclose(float(saved), float(current), rel_tol=1e-6, abs_tol=1e-8):
                    return False
            elif isinstance(current, str) or isinstance(saved, str):
                if str(saved) != str(current):
                    return False
            elif int(saved) != int(current):
                return False
        try:
            current_nc = getattr(self, "nc", current_values.get("nc"))
            checkpoint_nc, checkpoint_names = normalize_class_schema(
                checkpoint.get("nc", current_nc),
                checkpoint.get("names"),
                context="checkpoint",
            )
        except (TypeError, ValueError):
            return False
        if current_nc is not None and checkpoint_nc != int(current_nc):
            return False
        current_names = getattr(self, "class_names", None)
        if checkpoint_names is not None and current_names is not None and checkpoint_names != current_names:
            return False
        return True

    def _resume_best_checkpoint_matches(self):
        """Return whether the best artifact belongs to this resumed run.

        Best files are accepted only when they carry this run's identity.
        """
        resume_ckpt = getattr(self, "resume_ckpt", None)
        path = getattr(self, "best_checkpoint_path", None)
        if resume_ckpt is None or not path or not os.path.isfile(path):
            return False
        try:
            # Best-artifact inspection must never execute checkpoint pickle.
            # A safe-load failure simply means the artifact cannot be trusted
            # for resume identity and a later full validation will replace it.
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            return False
        if not self._checkpoint_matches_current_model(checkpoint):
            return False
        expected_run_id = str(getattr(self, "training_run_id", "") or resume_ckpt.get("training_run_id", ""))
        if not expected_run_id:
            return False
        observed_run_id = checkpoint.get("training_run_id") if isinstance(checkpoint, dict) else None
        if str(observed_run_id) != expected_run_id:
            return False
        expected_metric = getattr(self, "best_metric", None)
        checkpoint_metric = checkpoint.get("best_metric") if isinstance(checkpoint, dict) else None
        if checkpoint_metric is None:
            return True
        try:
            return self._normalize_best_metric(checkpoint_metric) == expected_metric
        except ValueError:
            return False

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
        from fotonet.models.scales import load_scale_config

        model_id = raw.model_config["model_id"]
        launcher_config = load_scale_config(model_id)
        launcher_config.update({
            "nc": int(self.nc),
            "model_id": model_id,
            "architecture_schema": int(raw.architecture_schema),
            "architecture_fingerprint": str(raw.architecture_fingerprint),
            "profile": str(raw.profile),
            "p2": bool(raw.use_p2),
        })
        return {
            "architecture": launcher_config["architecture"],
            "model_config": launcher_config,
            "model_id": model_id,
            "architecture_schema": int(raw.architecture_schema),
            "architecture_fingerprint": str(raw.architecture_fingerprint),
            "foundation_version": launcher_config["foundation_version"],
            "foundation_profile": launcher_config["foundation_profile"],
            "foundation_fingerprint": launcher_config["foundation_fingerprint"],
            "w": launcher_config["width_multiple"],
            "d": launcher_config["depth_multiple"],
            "use_p2": launcher_config["p2_head"],
            "reg_max": launcher_config["reg_max"],
            "p3_extra_blocks": launcher_config["p3_extra_blocks"],
            "p4_extra_blocks": launcher_config["p4_extra_blocks"],
            "p5_extra_blocks": launcher_config["p5_extra_blocks"],
            "p5_gate_blocks": launcher_config["p5_gate_blocks"],
            "arch_version": launcher_config["arch_version"],
            "head_version": launcher_config["head_version"],
            "p5_psa_blocks": launcher_config["p5_psa_blocks"],
            "neck_fusion": launcher_config["neck_fusion"],
            "neck_iema": launcher_config["neck_iema"],
            "p2_context_blocks": launcher_config["p2_context_blocks"],
            "p3_context_blocks": launcher_config["p3_context_blocks"],
            "quality_head": launcher_config["quality_head"],
        }

    def _sampling_checkpoint_fields(self):
        """Frozen disabled-policy metadata required by the active launcher."""
        from fotonet.engine.sampling import (
            normalize_sampling_hyp,
            sampling_policy_fingerprint,
        )

        sampling = normalize_sampling_hyp({
            "strategy": "uniform",
            "seed": 0,
            "dataset_mismatch": "error",
        })
        return {
            "training_system_version": 1,
            "sampling_strategy": "uniform",
            "sampling_config": sampling,
            "sampling_policy_fingerprint": sampling_policy_fingerprint(sampling),
            "sampling_state": None,
            "afss_orbit_config": {
                "enabled": False,
                "num_shards": 6,
                "shards_per_pass": 2,
                "anchor_fraction": 0.07,
                "refresh_interval_orbits": 5,
                "consolidation_interval_orbits": 10,
                "full_consolidation": True,
                "adaptive": True,
                "control": "intelligent",
                "seed": 0,
            },
            "afss_orbit_state": None,
            "art_config": {
                "enabled": False,
                "max_resolution": int(self.imgsz),
                "resolutions": (640, 576, 512, 448, 320),
                "probabilities": (0.45, 0.25, 0.15, 0.1, 0.05),
                "deterministic_quota": True,
                "quota_length": 20,
                "max_repeats": 2,
                "adaptive": True,
                "update_interval_orbits": 5,
                "adaptation_rate": 0.05,
                "metric_ema": 0.8,
                "small_object_force_max": True,
                "minimum_object_pixels": 6.0,
                "dynamic_batch": False,
                "batch_sizes": "auto",
                "prewarm_shapes": True,
                "eval_resolution": int(self.imgsz),
                "seed": 0,
            },
            "art_state": None,
        }

    @staticmethod
    def _schema_fields(model_meta):
        return {
            "checkpoint_format": 1,
            "training_protocol": 1,
            "model_id": model_meta["model_id"],
            "architecture_schema": model_meta["architecture_schema"],
            "architecture_fingerprint": model_meta["architecture_fingerprint"],
        }

    def _move_optimizer_state_to_device(self):
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device, non_blocking=True)

    def _save_last_checkpoint(self, scheduler, epoch):
        model_meta = self._checkpoint_model_meta()
        torch.save({
            **self._schema_fields(model_meta),
            "model":           self._raw_model().state_dict(),
            "ema_state":       self.ema.ema.state_dict(),
            "ema_updates":     self.ema.step_count,
            "ema_total_steps": self.ema.total_steps,
            "ema_warmup_steps": self.ema.warmup_steps,
            "ema_decay_start": self.ema.decay_start,
            "ema_decay_end": self.ema.decay_end,
            "optimizer_state": self.optimizer.state_dict(),
            "optimizer_name":  self.optimizer_name,
            "lr_scheduler":    self.lr_scheduler,
            "scheduler_step_unit": self._scheduler_step_unit(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state":    self.scaler.state_dict(),
            "nc":              self.nc,
            "names":           dict(self.class_names),
            "epoch":           epoch + 1,
            "global_step":     self.global_step,
            "optimizer_step_count": self.optimizer_step_count,
            "images_seen":       getattr(self, "images_seen", 0),
            "schedule_progress": float(getattr(self, "schedule_progress", 0.0)),
            "rng_state":       self._capture_rng_state(),
            "training_run_id": self.training_run_id,
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
            "head_version":      model_meta.get("head_version", model_meta["arch_version"]),
            "p5_psa_blocks":     model_meta["p5_psa_blocks"],
            "neck_fusion":       model_meta["neck_fusion"],
            "neck_iema":         model_meta["neck_iema"],
            "p2_context_blocks": model_meta["p2_context_blocks"],
            "p3_context_blocks": model_meta["p3_context_blocks"],
            "quality_head":      model_meta["quality_head"],
            "architecture":      model_meta.get("architecture", "standard"),
            "model_config":      model_meta.get("model_config"),
            "foundation_version": model_meta.get("foundation_version"),
            "foundation_profile": model_meta.get("foundation_profile"),
            "foundation_fingerprint": model_meta.get("foundation_fingerprint"),
            **self._sampling_checkpoint_fields(),
        }, self.last_checkpoint_path)

    def _save_best_checkpoint(self, epoch):
        model_meta = self._checkpoint_model_meta()
        if self.slim_best:
            ckpt = {
                **self._schema_fields(model_meta),
                "model":    self._state_dict_for_save(self.ema.ema, half=True, strip_o2m=True),
                "nc":       self.nc,
                "names":    dict(self.class_names),
                "epoch":    epoch + 1,
                "global_step": self.global_step,
                "images_seen": getattr(self, "images_seen", 0),
                "training_run_id": self.training_run_id,
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
                "head_version": model_meta.get("head_version", model_meta["arch_version"]),
                "p5_psa_blocks": model_meta["p5_psa_blocks"],
                "neck_fusion": model_meta["neck_fusion"],
                "neck_iema": model_meta["neck_iema"],
                "p2_context_blocks": model_meta["p2_context_blocks"],
                "p3_context_blocks": model_meta["p3_context_blocks"],
                "quality_head": model_meta["quality_head"],
                "architecture": model_meta.get("architecture", "standard"),
                "model_config": model_meta.get("model_config"),
                "foundation_version": model_meta.get("foundation_version"),
                "foundation_profile": model_meta.get("foundation_profile"),
                "foundation_fingerprint": model_meta.get("foundation_fingerprint"),
                "inference_only": True,
                "stripped_o2m": True,
                "has_o2m": False,
                "nms_free": True,
            }
        else:
            ckpt = {
                **self._schema_fields(model_meta),
                "model":     self.ema.ema.state_dict(),
                "ema_state": self.ema.ema.state_dict(),
                "nc":        self.nc,
                "names":     dict(self.class_names),
                "epoch":     epoch + 1,
                "global_step": self.global_step,
                "images_seen": getattr(self, "images_seen", 0),
                "training_run_id": self.training_run_id,
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
                "head_version": model_meta.get("head_version", model_meta["arch_version"]),
                "p5_psa_blocks": model_meta["p5_psa_blocks"],
                "neck_fusion": model_meta["neck_fusion"],
                "neck_iema": model_meta["neck_iema"],
                "p2_context_blocks": model_meta["p2_context_blocks"],
                "p3_context_blocks": model_meta["p3_context_blocks"],
                "quality_head": model_meta["quality_head"],
                "architecture": model_meta.get("architecture", "standard"),
                "model_config": model_meta.get("model_config"),
                "foundation_version": model_meta.get("foundation_version"),
                "foundation_profile": model_meta.get("foundation_profile"),
                "foundation_fingerprint": model_meta.get("foundation_fingerprint"),
                "nms_free": True,
            }
        torch.save(ckpt, self.best_checkpoint_path)

    def _save_periodic_checkpoint(self, scheduler, epoch):
        if self.save_period <= 0 or (epoch + 1) % self.save_period != 0:
            return
        model_meta = self._checkpoint_model_meta()
        torch.save({
            **self._schema_fields(model_meta),
            "model":           self._raw_model().state_dict(),
            "ema_state":       self.ema.ema.state_dict(),
            "ema_updates":     self.ema.step_count,
            "ema_total_steps": self.ema.total_steps,
            "ema_warmup_steps": self.ema.warmup_steps,
            "ema_decay_start": self.ema.decay_start,
            "ema_decay_end": self.ema.decay_end,
            "optimizer_state": self.optimizer.state_dict(),
            "optimizer_name":  self.optimizer_name,
            "lr_scheduler":    self.lr_scheduler,
            "scheduler_step_unit": self._scheduler_step_unit(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state":    self.scaler.state_dict(),
            "nc":              self.nc,
            "names":           dict(self.class_names),
            "epoch":           epoch + 1,
            "global_step":     self.global_step,
            "optimizer_step_count": self.optimizer_step_count,
            "images_seen":       getattr(self, "images_seen", 0),
            "schedule_progress": float(getattr(self, "schedule_progress", 0.0)),
            "rng_state":       self._capture_rng_state(),
            "training_run_id": self.training_run_id,
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
            "head_version":      model_meta.get("head_version", model_meta["arch_version"]),
            "p5_psa_blocks":     model_meta["p5_psa_blocks"],
            "neck_fusion":       model_meta["neck_fusion"],
            "neck_iema":         model_meta["neck_iema"],
            "p2_context_blocks": model_meta["p2_context_blocks"],
            "p3_context_blocks": model_meta["p3_context_blocks"],
            "quality_head":      model_meta["quality_head"],
            "architecture":      model_meta.get("architecture", "standard"),
            "model_config":      model_meta.get("model_config"),
            "foundation_version": model_meta.get("foundation_version"),
            "foundation_profile": model_meta.get("foundation_profile"),
            "foundation_fingerprint": model_meta.get("foundation_fingerprint"),
            **self._sampling_checkpoint_fields(),
        }, os.path.join(self.save_dir, f"fotonet_epoch_{epoch+1}.pt"))


__all__ = ["CheckpointProtocolMixin"]
