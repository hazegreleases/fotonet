"""Public fotonet model API."""
import io
import json
import os
import copy
import math
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None

from fotonet._version import __version__
from fotonet.models.v1 import Detector
from fotonet.models.v1.registry import available_models, is_model_ref, load_model_config
from fotonet.engine.results import Results
from fotonet.utils.general import check_device
from fotonet.utils.config import (
    class_names_are_default,
    default_class_names,
    load_data_cfg,
    load_model_cfg,
    normalize_class_schema,
)

_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}


class Fotonet:
    """
    NMS-free object detection with one self-identifying production graph.

    Use ``Fotonet('fotonetn')`` for an untrained architecture or
    ``Fotonet('path/to.pt')`` for a supported checkpoint.
    """
    MODELS = available_models()

    def __init__(self, model_path=None, nc=None, task="detect", device=None):
        # Preserve the zero-argument device hook used by downstream callers;
        # only pass an argument when the caller explicitly chose a device.
        self.device = check_device() if device is None else check_device(device)
        self.task = str(task).lower()
        if self.task != "detect":
            raise ValueError("fotonet supports task='detect' only.")
        self.nc, _ = normalize_class_schema(80 if nc is None else nc, context="model")
        self.model  = None
        self._model_cfg = None
        self.model_path = model_path
        self.loaded_weight_path = None
        self.feature_init_path = None
        self.runtime_format = None
        self._runtime_imgsz = None
        self._runtime_dynamic = None
        self._runtime_schema_pending = False
        # Deployment transforms are applied to a clone.  This keeps the graph
        # with BatchNorm/O2M heads intact if the caller later invokes train().
        self._training_model_before_deploy = None
        self._apply_model_cfg(load_model_config("fotonetn"), update_nc=False)
        self.names = default_class_names(self.nc)

        if model_path is None:
            self.model = self._new_model()
            return

        model_path = os.fspath(model_path)
        path_lower = model_path.lower()

        if os.path.isfile(model_path) and path_lower.endswith((".yaml", ".yml")):
            cfg = load_model_cfg(model_path)
            self._model_cfg = cfg
            self._apply_model_cfg(cfg, update_nc=nc is None)
            self.names = default_class_names(self.nc)
            self.model = self._new_model()

        elif os.path.isfile(model_path) and path_lower.endswith((".onnx", ".torchscript", ".ts")):
            self._load_runtime_artifact(model_path)

        elif os.path.isfile(model_path) and path_lower.endswith((".engine", ".mlmodel", ".mlpackage")):
            raise RuntimeError(
                f"'{Path(model_path).suffix}' runtime loading is not available in this Python build. "
                "Use the native TensorRT/CoreML runtime integration for that platform, or load the ONNX/TorchScript artifact."
            )

        elif os.path.isfile(model_path):
            self.model = self._new_model()
            self.load(model_path)
            self.loaded_weight_path = model_path

        elif is_model_ref(path_lower):
            self._apply_model_cfg(load_model_config(path_lower), update_nc=nc is None)
            self.names = default_class_names(self.nc)
            self.model = self._new_model()
            return

        else:
            valid = ", ".join(available_models())
            raise ValueError(f"Unknown model '{model_path}'. Use one of: {valid}, a YAML file, or a checkpoint.")

    def _load_runtime_artifact(self, model_path):
        """Load an exported inference artifact with its adjacent metadata contract."""
        metadata_path = Path(model_path).with_suffix(Path(model_path).suffix + ".metadata.json")
        suffix = Path(model_path).suffix.lower()
        if suffix == ".onnx":
            from fotonet.engine.runtimes import ONNXRuntime, validate_artifact_metadata

            if not metadata_path.is_file():
                raise RuntimeError(
                    f"ONNX artifact is missing required metadata sidecar '{metadata_path}'. Re-export it."
                )
            try:
                with open(metadata_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Could not read ONNX metadata '{metadata_path}'.") from exc
            metadata = validate_artifact_metadata(metadata, "onnx")
            runtime = ONNXRuntime(
                model_path,
                metadata.get("nc", self.nc),
                self.device,
                strides=metadata.get("strides"),
            )
            # ONNX preserves its output width, so class count and input shape
            # can still be recovered when a sidecar was not copied alongside
            # the artifact.
            if runtime.output_nc is not None:
                if "nc" in metadata and int(metadata["nc"]) != runtime.output_nc:
                    raise RuntimeError(
                        f"ONNX metadata nc={int(metadata['nc'])} does not match the graph output nc="
                        f"{runtime.output_nc}. Restore the matching sidecar or re-export the artifact."
                    )
                metadata = {**metadata, "nc": runtime.output_nc}
            graph_imgsz = getattr(runtime, "input_imgsz", None)
            if graph_imgsz is not None:
                declared_imgsz = tuple(metadata["imgsz"]) if "imgsz" in metadata else None
                if declared_imgsz is not None and declared_imgsz != tuple(graph_imgsz):
                    raise RuntimeError(
                        f"ONNX metadata imgsz={declared_imgsz} does not match graph input HxW={tuple(graph_imgsz)}. "
                        "Restore the matching sidecar or re-export the artifact."
                    )
                metadata = {**metadata, "imgsz": list(graph_imgsz)}
            graph_batch = getattr(runtime, "input_batch", None)
            if graph_batch is not None:
                if "batch" in metadata and int(metadata["batch"]) != int(graph_batch):
                    raise RuntimeError(
                        f"ONNX metadata batch={int(metadata['batch'])} does not match graph input batch="
                        f"{int(graph_batch)}. Restore the matching sidecar or re-export the artifact."
                    )
                metadata = {**metadata, "batch": int(graph_batch)}
            graph_dynamic = getattr(runtime, "dynamic_input", None)
            if graph_dynamic is not None:
                if "dynamic" in metadata and metadata["dynamic"] != bool(graph_dynamic):
                    raise RuntimeError(
                        f"ONNX metadata dynamic={metadata['dynamic']} does not match the graph input contract "
                        f"(dynamic={bool(graph_dynamic)}). Restore the matching sidecar or re-export the artifact."
                    )
                metadata = {**metadata, "dynamic": bool(graph_dynamic)}
            # A dynamic ONNX graph often leaves its output width symbolic.
            # When the signed sidecar supplies both class count and stride
            # geometry, that schema is complete enough to preallocate the
            # CUDA I/O-binding output. Leaving ``schema_pending`` true here
            # silently forces every GPU request through CPU copies.
            if "nc" in metadata and "strides" in metadata:
                runtime.set_output_schema(metadata["nc"])
            self.runtime_format = "onnx"
        else:
            from fotonet.engine.runtimes import TorchScriptRuntime, validate_artifact_metadata

            runtime = TorchScriptRuntime(model_path, self.nc, self.device)
            if runtime.metadata:
                # The archive is authoritative: a stale sidecar must not make
                # runtime preprocessing disagree with the serialized graph.
                metadata = runtime.metadata
            elif metadata_path.is_file():
                try:
                    with open(metadata_path, "r", encoding="utf-8") as handle:
                        metadata = json.load(handle)
                except (OSError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        f"Could not read TorchScript runtime metadata '{metadata_path}'. "
                        "Re-export the artifact to embed its contract."
                    ) from exc
                metadata = validate_artifact_metadata(metadata, "torchscript")
            else:
                raise RuntimeError(
                    "TorchScript artifact is missing both embedded FOTO-NET metadata and its "
                    "'.metadata.json' sidecar. A traced module does not reliably retain its "
                    "class count or static input HxW. Re-export it with "
                    "Fotonet(...).export(format='torchscript', path='model.torchscript')."
                )
            self.runtime_format = "torchscript"
        try:
            self.nc, artifact_names = normalize_class_schema(
                metadata.get("nc", getattr(runtime.head, "nc", self.nc)),
                metadata.get("names"),
                context="runtime artifact metadata",
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Runtime artifact has an invalid class schema: {exc}") from exc
        runtime.head.nc = self.nc
        self.model = runtime
        artifact_imgsz = metadata.get("imgsz")
        if isinstance(artifact_imgsz, (list, tuple)) and len(artifact_imgsz) == 2:
            self._runtime_imgsz = tuple(int(value) for value in artifact_imgsz)
        self._runtime_dynamic = bool(metadata["dynamic"]) if "dynamic" in metadata else None
        self.names = dict(artifact_names) if artifact_names is not None else default_class_names(self.nc)
        if self._runtime_imgsz is None:
            self._runtime_imgsz = getattr(self.model, "input_imgsz", None)
        if self._runtime_dynamic is None:
            self._runtime_dynamic = getattr(self.model, "dynamic_input", None)
        # A symbolic ONNX output is still fully specified when its sidecar
        # supplies both class count and stride geometry. Without either, do
        # not let validation assume the constructor's default COCO schema.
        self._runtime_schema_pending = bool(
            getattr(self.model, "schema_pending", False)
            and ("nc" not in metadata or "strides" not in metadata)
        )
        self.loaded_weight_path = model_path

    def _sync_runtime_output_schema(self, output):
        """Adopt the graph's class count once a symbolic runtime output resolves."""
        if self.runtime_format is None or not torch.is_tensor(output):
            return
        if output.ndim != 3:
            raise RuntimeError(
                f"Runtime artifact returned invalid output shape {tuple(output.shape)}; "
                "expected raw [B,N,nc+4]."
            )
        output_nc = int(output.shape[-1]) - 4
        if output_nc <= 0:
            raise RuntimeError(
                f"Runtime artifact returned invalid output shape {tuple(output.shape)}; "
                "expected raw [B,N,nc+4]."
            )
        if output_nc == self.nc:
            self._runtime_schema_pending = False
            return
        self.nc = output_nc
        self.model.head.nc = output_nc
        expected_keys = set(range(output_nc))
        if not isinstance(self.names, dict) or set(self.names) != expected_keys:
            self.names = default_class_names(output_nc)
        self._runtime_schema_pending = False

    def _apply_model_cfg(self, cfg, update_nc=True):
        self._model_cfg = dict(cfg)
        if update_nc:
            self.nc, _ = normalize_class_schema(cfg.get("nc", self.nc), context="model config")
        self.model_id = str(cfg["model_id"])
        self.architecture_schema = int(cfg["architecture_schema"])
        self.architecture_fingerprint = str(cfg["architecture_fingerprint"])
        self.profile = str(cfg["profile"])
        self.use_p2 = bool(cfg["p2"])
        self.reg_max = int(cfg["reg_max"])
        self.quality_head = bool(cfg.get("quality_head", getattr(self, "quality_head", False)))

    def _new_model(self):
        model = Detector(
            nc=self.nc,
            profile=self.profile,
            use_p2=self.use_p2,
            reg_max=self.reg_max,
            quality_head=self.quality_head,
        ).to(self.device)
        self.architecture_fingerprint = model.architecture_fingerprint
        model.model_config = {
            "model_id": self.model_id,
            "architecture_schema": self.architecture_schema,
            "architecture_fingerprint": self.architecture_fingerprint,
            "profile": self.profile,
            "nc": int(self.nc),
            "p2": bool(self.use_p2),
            "reg_max": int(self.reg_max),
            "quality_head": bool(self.quality_head),
            "backbone_out_channels": list(model.backbone_out_channels),
            "neck_out_channels": list(model.neck_out_channels),
            "feature_strides": list(model.feature_strides),
        }
        return model

    CHECKPOINT_FORMAT = 1
    TRAINING_PROTOCOL = 1
    _ACTIVE_CANDIDATE_FINGERPRINT = (
        "4d8755c10962efbcfc1a78ae9719b9101e7d8b28c7f9ccae82567747dacfd6de"
    )

    @classmethod
    def _checkpoint_config(cls, checkpoint):
        """Validate the sole production identity or the active-run migration identity."""
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a self-identifying tensor-only mapping.")
        if checkpoint.get("checkpoint_format") == cls.CHECKPOINT_FORMAT:
            required = (
                "architecture_schema", "architecture_fingerprint", "model_id",
                "model_config", "nc",
            )
            missing = [key for key in required if checkpoint.get(key) is None]
            if missing:
                raise ValueError("Checkpoint is missing required identity: " + ", ".join(missing))
            config = load_model_config(checkpoint["model_id"])
            config["nc"] = int(checkpoint["nc"])
            expected = Detector(
                nc=config["nc"], profile=config["profile"], use_p2=config["p2"],
                reg_max=config["reg_max"], quality_head=config["quality_head"],
            ).architecture_fingerprint
            if int(checkpoint["architecture_schema"]) != config["architecture_schema"]:
                raise ValueError("Unsupported checkpoint architecture_schema.")
            if str(checkpoint["architecture_fingerprint"]) != expected:
                raise ValueError("Checkpoint architecture_fingerprint does not match its model identity.")
            embedded = checkpoint["model_config"]
            if not isinstance(embedded, dict):
                raise ValueError("Checkpoint model_config must be a mapping.")
            for key in ("model_id", "architecture_schema", "profile", "p2", "reg_max", "quality_head"):
                if embedded.get(key) != config.get(key):
                    raise ValueError(f"Checkpoint has conflicting model_config.{key} metadata.")
            return config

        model_config = checkpoint.get("model_config") or {}
        candidate = (
            checkpoint.get("arch_version") == 4
            and checkpoint.get("head_version") == 4
            and checkpoint.get("foundation_version") == 2
            and checkpoint.get("foundation_profile") == "n"
            and checkpoint.get("foundation_fingerprint") == cls._ACTIVE_CANDIDATE_FINGERPRINT
            and model_config.get("foundation_profile") == "n"
            and model_config.get("p2_head") is False
            and int(model_config.get("reg_max", -1)) == 12
            and model_config.get("quality_head") is False
        )
        if not candidate:
            raise ValueError(
                "Unsupported checkpoint identity. Only checkpoint_format=1 artifacts and the "
                "explicitly identified active fotonetn training checkpoint are accepted."
            )
        config = load_model_config("fotonetn")
        config["nc"] = int(checkpoint.get("nc", model_config.get("nc", 80)))
        return config

    def _restore_training_model(self):
        """Restore the untouched training graph after a deploy-only transform."""
        if self._training_model_before_deploy is not None:
            self.model = self._training_model_before_deploy.to(self.device)
            self._training_model_before_deploy = None

    def _assert_trainable_graph(self):
        """Reject deploy-fused native graphs that cannot be faithfully unfused."""
        raw = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        if bool(getattr(raw, "_fused", False)):
            raise RuntimeError(
                "This model contains a fused deployment graph and cannot be trained safely. "
                "Reload an unfused training checkpoint, or use the original model before fusion."
            )
        return self.model

    def _ensure_deployment_copy(self):
        """Switch inference to an isolated copy, retaining the trainable graph."""
        if self.runtime_format is not None or self._training_model_before_deploy is not None:
            return self.model
        try:
            deploy = copy.deepcopy(self.model)
        except Exception as exc:
            raise RuntimeError(
                "Could not create an isolated deployment copy. Reload the model and export without "
                "prepare_for_inference() rather than fusing the live training graph."
            ) from exc
        self._training_model_before_deploy = self.model
        self.model = deploy
        return self.model

    @staticmethod
    def _safe_torch_load(path):
        """Load supported checkpoints without enabling pickle execution."""
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(
                f"Could not safely load {path!r}; supported checkpoints must be tensor-only."
            ) from exc

    @staticmethod
    def _checkpoint_state_dict(checkpoint):
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
            raise ValueError("Checkpoint is missing its model tensor mapping.")
        return checkpoint["model"]

    @classmethod
    def _checkpoint_inference_state_dict(cls, checkpoint):
        state = checkpoint.get("ema_state") if isinstance(checkpoint, dict) else None
        return state if isinstance(state, dict) else cls._checkpoint_state_dict(checkpoint)

    @staticmethod
    def _checkpoint_is_inference_only(checkpoint):
        return bool(
            checkpoint.get("inference_only", False)
            or checkpoint.get("stripped_o2m", False)
            or checkpoint.get("has_o2m") is False
        )

    @staticmethod
    def _checkpoint_is_full_training_resume(checkpoint):
        required = (
            "model", "ema_state", "optimizer_state", "optimizer_name",
            "scheduler_state", "lr_scheduler", "scheduler_step_unit",
            "scaler_state", "epoch", "global_step", "optimizer_step_count",
            "rng_state", "training_run_id", "nc", "names", "model_config",
        )
        return isinstance(checkpoint, dict) and all(key in checkpoint for key in required)

    @staticmethod
    def _resume_checkpoint_error(weights_path):
        return (
            f"Cannot resume from {weights_path!r}: use a full fotonet training checkpoint "
            "containing model, optimizer, scheduler, scaler, EMA, and epoch state."
        )

    def _apply_checkpoint_config(self, config):
        self._apply_model_cfg(config, update_nc=True)
        self.names = default_class_names(self.nc)
        self.model = self._new_model()

    def _load_model_state(self, state, *, inference_only=False):
        if not isinstance(state, dict):
            raise ValueError("Checkpoint model state must be a tensor mapping.")
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        allowed_missing = (
            "head.cls_o2m.", "head.reg_o2m.", "head.quality_o2m.",
            "head.cls_adapter_o2m.", "head.reg_adapter_o2m.",
        )
        illegal_missing = [key for key in missing if not key.startswith(allowed_missing)]
        if unexpected or illegal_missing or (missing and not inference_only):
            raise ValueError(
                "Checkpoint tensors do not exactly match the declared graph: "
                f"missing={illegal_missing or missing}, unexpected={unexpected}."
            )
        return missing

    def load_weights(self, weights_path):
        """Strict-load a supported self-identifying checkpoint."""
        if self.runtime_format is not None:
            raise RuntimeError("Exported runtime artifacts cannot accept PyTorch checkpoints.")
        path = os.fspath(weights_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = self._safe_torch_load(path)
        config = self._checkpoint_config(checkpoint)
        self._apply_checkpoint_config(config)
        state = self._checkpoint_inference_state_dict(checkpoint)
        inference_only = self._checkpoint_is_inference_only(checkpoint)
        self._load_model_state(state, inference_only=inference_only)
        self.loaded_weight_path = path
        return True

    def _ensure_training_o2m_heads(self, source_state=None, mirror=False):
        raw = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        if bool(getattr(raw.head, "has_o2m", True)):
            return
        state = source_state or raw.state_dict()
        self.model = self._new_model()
        self._load_model_state(state, inference_only=True)
        if mirror:
            self._mirror_o2o_to_o2m()

    def freeze_backbone(self, freeze=True):
        """Freeze/unfreeze backbone weights (useful for first N epochs)."""
        raw = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        for p in raw.backbone.parameters():
            p.requires_grad_(not freeze)
        state = "frozen" if freeze else "unfrozen"
        print(f"[INFO] Backbone {state}.")

    def train_from_recipe(self, data, recipe="fotonetn_scratch", **overrides):
        """Train from a packaged/resolved recipe, with explicit overrides.

        This method does not execute a hidden recipe: it returns the same value
        as :meth:`train` and prints the source recipe before any training begins.
        """
        from fotonet.config.recipes import load_training_recipe

        resolved = load_training_recipe(recipe)
        recipe_model = resolved.get("model")
        if recipe_model and not is_model_ref(recipe_model):
            raise ValueError(f"Recipe '{resolved['name']}' names unsupported model '{recipe_model}'.")
        settings = dict(resolved["settings"])
        settings.update(overrides)
        print(f"[INFO] Using training recipe '{resolved['name']}' from '{resolved['path']}'.")
        return self.train(data=data, **settings)

    def train(
        self, data, epochs=100, imgsz=640, batch=16, lr0=0.01, lrf=0.01,
        nbs=128, warmup_epochs=3, frozen_epochs=0, save_dir=".", val_split=None,
        val_period=1, val_batch=None, workers=None, pin_memory=True,
        cache_to_ram=True, ram_cache_images=1024, amp=True,
        amp_init_scale=65536.0, val_amp=None, cos_lr=True,
        lr_scheduler="Cosine", lr_drop_factor=0.92, lr_drop_patience=5,
        lr_drop_threshold=0.001, lr_drop_min_lr=1e-5, weights=None,
        resume=False, pretrained=False, compile_model=False, save_period=-1,
        slim_best=True, best_metric="mAP50_95", save_last=True,
        optimizer="sgd", momentum=0.937, weight_decay=0.0005,
        augment_hyp=None, augmentation_passes=None, loss_hyp=None,
        matcher_hyp=None, imgsz_schedule=None, val_subset_size=0,
        full_val_after=1.0, cache_labels=True, disk_cache_images=False,
        disk_cache_dir=None, unfreeze_backbone_at=None, coco_max_dets=100,
        val_conf=0.0, operating_conf=0.25, operating_iou=0.50,
        annotation_policy="fix", allow_missing_labels=False,
        source_recursive=True, train_dataset=None, _training_run_id=None,
        sampling_hyp=None, afss_orbit_hyp=None, art_hyp=None,
        distill=None, distill_teacher=None, distill_weight=0.0,
        distill_labels_dir=None, allow_unscored_pseudo_labels=False,
        profile=False, profiling_hyp=None, epoch_cut=1, cuda_graphs=False,
    ):
        """Train or resume the uniform production protocol."""
        self._restore_training_model()
        self._assert_trainable_graph()
        epochs = int(epochs)
        if epochs < 1:
            raise ValueError("epochs must be a positive integer")
        max_stride = max(int(stride) for stride in self.model.head.strides)
        for label, size in [("imgsz", imgsz)]:
            if int(size) <= 0 or int(size) % max_stride:
                raise ValueError(f"{label} must be positive and divisible by {max_stride}")
        if imgsz_schedule:
            for index, item in enumerate(imgsz_schedule):
                size = item.get("imgsz", item.get("size")) if isinstance(item, dict) else item[1]
                if int(size) <= 0 or int(size) % max_stride:
                    raise ValueError(f"imgsz_schedule[{index}] must be divisible by {max_stride}")
        if isinstance(resume, (str, os.PathLike)):
            weights, resume = os.fspath(resume), True
        weights = os.fspath(weights) if weights is not None else None
        if resume and pretrained:
            raise ValueError("resume and pretrained are mutually exclusive")
        locked_off = {
            "sampling_hyp": sampling_hyp not in (None, {}, {"strategy": "uniform", "seed": 0, "dataset_mismatch": "error"}),
            "afss_orbit_hyp": bool((afss_orbit_hyp or {}).get("enabled", False)),
            "art_hyp": bool((art_hyp or {}).get("enabled", False)),
            "distill": distill not in (None, False),
            "distill_teacher": distill_teacher not in (None, False, ""),
            "distill_weight": float(distill_weight or 0.0) != 0.0,
            "distill_labels_dir": distill_labels_dir not in (None, False, ""),
            "allow_unscored_pseudo_labels": bool(allow_unscored_pseudo_labels),
            "profile": bool(profile) or bool((profiling_hyp or {}).get("enabled", False)),
            "epoch_cut": int(epoch_cut) != 1,
            "cuda_graphs": bool(cuda_graphs),
        }
        enabled = [name for name, value in locked_off.items() if value]
        if enabled:
            raise ValueError("Removed training options cannot be enabled: " + ", ".join(enabled))
        if unfreeze_backbone_at is not None:
            frozen_epochs = max(int(unfreeze_backbone_at), 0)

        data_config = load_data_cfg(data)
        data_nc, data_names = data_config.get("nc"), data_config.get("names")
        resume_checkpoint = None
        if resume:
            checkpoint_path = weights or self.loaded_weight_path
            if not checkpoint_path:
                raise ValueError("resume requires weights='.../fotonet_last.pt'")
            resume_checkpoint = self._safe_torch_load(checkpoint_path)
            config = self._checkpoint_config(resume_checkpoint)
            if not self._checkpoint_is_full_training_resume(resume_checkpoint):
                raise ValueError(self._resume_checkpoint_error(checkpoint_path))
            candidate = resume_checkpoint.get("checkpoint_format") is None
            if resume_checkpoint.get("training_protocol") != self.TRAINING_PROTOCOL and not candidate:
                raise ValueError("Unsupported or missing training_protocol")
            if candidate:
                states = (resume_checkpoint.get("sampling_state"), resume_checkpoint.get("afss_orbit_state"), resume_checkpoint.get("art_state"))
                if any(value is not None for value in states):
                    raise ValueError("Active-run migration requires controller-free state")
                if any((resume_checkpoint.get(key) or {}).get("enabled", False) for key in ("afss_orbit_hyp", "art_hyp")):
                    raise ValueError("Active-run migration rejects enabled adaptive controllers")
            checkpoint_nc, checkpoint_names = normalize_class_schema(
                resume_checkpoint.get("nc"), resume_checkpoint.get("names"),
                context="resume checkpoint",
            )
            if data_nc is not None and int(data_nc) != checkpoint_nc:
                raise ValueError("Resume checkpoint class count does not match the dataset")
            if data_names is not None and checkpoint_names is not None and data_names != checkpoint_names:
                raise ValueError("Resume checkpoint class order does not match the dataset")
            config["nc"] = checkpoint_nc
            self._apply_checkpoint_config(config)
            self.names = dict(checkpoint_names or data_names or default_class_names(checkpoint_nc))
            self.model.load_state_dict(self._checkpoint_state_dict(resume_checkpoint), strict=True)
            optimizer = resume_checkpoint.get("optimizer_name", optimizer)
            self.loaded_weight_path = checkpoint_path
        elif pretrained:
            checkpoint_path = weights or self.loaded_weight_path
            if not checkpoint_path:
                raise ValueError("pretrained requires a supported checkpoint path")
            checkpoint = self._safe_torch_load(checkpoint_path)
            config = self._checkpoint_config(checkpoint)
            checkpoint_nc, checkpoint_names = normalize_class_schema(
                checkpoint.get("nc"), checkpoint.get("names"), context="pretrained checkpoint"
            )
            if data_nc is not None and int(data_nc) != checkpoint_nc:
                raise ValueError("Pretrained checkpoint class count does not match the dataset")
            config["nc"] = checkpoint_nc
            self._apply_checkpoint_config(config)
            self.names = dict(checkpoint_names or data_names or default_class_names(checkpoint_nc))
            state = self._checkpoint_inference_state_dict(checkpoint)
            self._load_model_state(state, inference_only=self._checkpoint_is_inference_only(checkpoint))
            self._ensure_training_o2m_heads(state, mirror=True)
            self.loaded_weight_path = checkpoint_path
        else:
            if data_nc is not None and int(data_nc) != self.nc:
                self.nc = int(data_nc)
                self.model = self._new_model()
            if data_names is not None:
                self.names = dict(data_names)

        dataset = train_dataset
        if dataset is not None and val_split not in (None, 0, 0.0):
            raise ValueError("train_dataset cannot be combined with val_split")
        if dataset is None and val_split not in (None, 0, 0.0):
            from fotonet.data.dataset import build_detection_dataset
            if "train" not in data_config:
                raise ValueError("val_split requires a train source")
            dataset = build_detection_dataset(
                data_config["train"], imgsz=imgsz, augment=True,
                cache_labels=cache_labels, disk_cache_images=disk_cache_images,
                disk_cache_dir=disk_cache_dir, cache_to_ram=cache_to_ram,
                ram_cache_images=ram_cache_images, augment_hyp=augment_hyp,
                num_classes=self.nc, annotation_policy=annotation_policy,
                allow_missing_labels=allow_missing_labels, source_recursive=source_recursive,
                coco_images=data_config.get("train_images", data_config.get("coco_images")),
            )
        from fotonet.engine.trainer import Trainer
        trainer = Trainer(
            self.model, data_config, epochs=epochs, imgsz=imgsz, lr0=lr0, lrf=lrf,
            batch=batch, val_batch=val_batch, nbs=nbs, warmup_epochs=warmup_epochs,
            save_dir=save_dir, val_split=val_split, val_period=val_period,
            workers=workers, pin_memory=pin_memory, cache_to_ram=cache_to_ram,
            ram_cache_images=ram_cache_images, amp=amp, amp_init_scale=amp_init_scale,
            val_amp=val_amp, cos_lr=cos_lr, lr_scheduler=lr_scheduler,
            lr_drop_factor=lr_drop_factor, lr_drop_patience=lr_drop_patience,
            lr_drop_threshold=lr_drop_threshold, lr_drop_min_lr=lr_drop_min_lr,
            resume_ckpt=resume_checkpoint, compile_model=compile_model,
            save_period=save_period, slim_best=slim_best, best_metric=best_metric,
            save_last=save_last, optimizer=optimizer, momentum=momentum,
            weight_decay=weight_decay, augment_hyp=augment_hyp,
            augmentation_passes=augmentation_passes, loss_hyp=loss_hyp,
            matcher_hyp=matcher_hyp, imgsz_schedule=imgsz_schedule,
            val_subset_size=val_subset_size, full_val_after=full_val_after,
            cache_labels=cache_labels, disk_cache_images=disk_cache_images,
            disk_cache_dir=disk_cache_dir, coco_max_dets=coco_max_dets,
            val_conf=val_conf, operating_conf=operating_conf,
            operating_iou=operating_iou, annotation_policy=annotation_policy,
            allow_missing_labels=allow_missing_labels, source_recursive=source_recursive,
            _training_run_id=_training_run_id,
        )
        return trainer.train(dataset, frozen_epochs=frozen_epochs)
    def prepare_for_inference(self, device=None, half=False, strip_o2m=False, fuse=True, matmul_precision=None):
        """Prepare an isolated deploy graph without mutating the train graph.

        Fusion, O2M stripping, and FP16 conversion happen on a deep copy.  If
        the user subsequently calls :meth:`train`, the original graph is
        restored before any optimizer or loss state is constructed.
        """
        device = device or self.device
        if self.runtime_format is None and (fuse or strip_o2m or half):
            self._ensure_deployment_copy()
        self.model.to(device).eval()
        if strip_o2m and self.runtime_format is None:
            self.strip_o2m_for_inference()
        if fuse and self.runtime_format is None:
            raw = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
            raw.fuse()
        if half and torch.device(device).type == "cuda":
            self.model.half()
        if matmul_precision is not None:
            if matmul_precision not in {"highest", "high", "medium"}:
                raise ValueError("matmul_precision must be 'highest', 'high', 'medium', or None.")
            torch.set_float32_matmul_precision(matmul_precision)
        return self

    @staticmethod
    def _normalize_imgsz(imgsz):
        if isinstance(imgsz, int):
            h = w = int(imgsz)
        elif isinstance(imgsz, (tuple, list)) and len(imgsz) == 2:
            h, w = (int(imgsz[0]), int(imgsz[1]))
        else:
            raise TypeError("imgsz must be a positive integer or an (height, width) pair.")
        if h <= 0 or w <= 0:
            raise ValueError("imgsz dimensions must be positive.")
        return h, w

    def _resolve_predict_imgsz(self, imgsz):
        """Resolve a default inference size and enforce static artifact shape."""
        target = self._normalize_imgsz((640, 640) if imgsz is None else imgsz)
        if imgsz is None and self._runtime_imgsz is not None:
            target = self._runtime_imgsz
        if (
            self.runtime_format is not None
            and self._runtime_dynamic is False
            and self._runtime_imgsz is not None
            and tuple(target) != tuple(self._runtime_imgsz)
        ):
            raise ValueError(
                f"This static {self.runtime_format} artifact accepts imgsz={tuple(self._runtime_imgsz)}, "
                f"not {tuple(target)}. Re-export with dynamic=True for variable shapes."
            )
        return target

    @staticmethod
    def _is_url(value):
        return isinstance(value, str) and urlparse(value).scheme in {"http", "https"}

    @classmethod
    def _is_video_source(cls, source):
        if isinstance(source, int):
            return True
        if not isinstance(source, (str, Path)):
            return False
        value = os.fspath(source)
        parsed = urlparse(value)
        suffix = Path(parsed.path if parsed.scheme else value).suffix.lower()
        return suffix in _VIDEO_SUFFIXES

    @classmethod
    def _expand_sources(cls, source):
        if isinstance(source, (str, Path)):
            value = os.fspath(source)
            if os.path.isdir(value):
                files = [
                    str(path) for path in Path(value).rglob("*")
                    if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
                ]
                if not files:
                    raise FileNotFoundError(f"No supported images were found under '{value}'.")
                return sorted(files)
            return [value]
        if isinstance(source, (list, tuple)):
            expanded = []
            for item in source:
                if cls._is_video_source(item):
                    raise ValueError("Video/webcam inputs must be passed as one source, not mixed with image batches.")
                expanded.extend(cls._expand_sources(item))
            if not expanded:
                raise ValueError("source cannot be an empty list.")
            return expanded
        return [source]

    @staticmethod
    def _read_image_source(source):
        if isinstance(source, Path):
            source = os.fspath(source)
        if isinstance(source, str):
            if Fotonet._is_url(source):
                request = Request(source, headers={"User-Agent": "fotonet/0.2"})
                with urlopen(request, timeout=20) as response:
                    return Image.open(io.BytesIO(response.read())).convert("RGB")
            if not os.path.isfile(source):
                raise FileNotFoundError(f"Image source not found: {source}")
            return Image.open(source).convert("RGB")
        if isinstance(source, (Image.Image, np.ndarray)):
            return source
        raise TypeError("source must be an image path/URL, directory, PIL image, NumPy array, tensor, video path, or webcam index.")

    def _predict_video(self, source, imgsz, conf, max_det, device, retain_images=True):
        if cv2 is None:
            raise RuntimeError("Video/webcam prediction requires opencv-python.")
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to open video/webcam source: {source}")
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield self._predict_batch_impl(
                    [frame], imgsz, conf, device=device, bgr=True, max_det=max_det,
                    retain_images=retain_images,
                )[0]
        finally:
            capture.release()

    def predict(
        self,
        source,
        imgsz=None,
        conf=0.25,
        max_det=300,
        device=None,
        batch=16,
        stream=False,
        retain_images=None,
        **kwargs,
    ):
        """Run detection and return a list of Results, or an iterator when ``stream=True``.

        ``retain_images`` defaults to ``True`` for decoded images and ``False``
        for tensor input.  The tensor default avoids a full GPU-to-CPU image
        copy just to make plotting available; pass ``True`` when pixels are
        needed for ``plot()``, ``save()``, or crops.
        """
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported predict option(s): {names}")
        target_h, target_w = self._resolve_predict_imgsz(imgsz)
        conf = float(conf)
        if not 0.0 <= conf <= 1.0:
            raise ValueError("conf must be in [0, 1].")
        max_det = max(int(max_det), 1)
        device = torch.device(device or self.device)
        self.model.to(device).eval()
        batch = max(int(batch or 1), 1)
        retain_images_arg = retain_images
        retain_images = True if retain_images is None else bool(retain_images)

        if self._is_video_source(source):
            frames = self._predict_video(
                source, (target_h, target_w), conf, max_det, device, retain_images=retain_images
            )
            return frames if stream else list(frames)
        if stream:
            raise ValueError("stream=True is supported for video/webcam sources only.")

        if isinstance(source, torch.Tensor):
            tensor = source.unsqueeze(0) if source.ndim == 3 else source
            if tensor.ndim != 4 or tensor.shape[1] != 3:
                raise ValueError("Tensor source must have shape [3,H,W] or [B,3,H,W].")
            # Tensor sources intentionally default to no host copy.  Explicit
            # retain_images=True restores visualization-compatible behavior.
            tensor_retain_images = False if retain_images_arg is None else bool(retain_images_arg)
            return self._predict_tensor(
                tensor.to(device),
                self._orig_images_from_tensor(tensor, retain_images=tensor_retain_images),
                conf,
                orig_shapes=[tuple(int(v) for v in tensor.shape[-2:])] * int(tensor.shape[0]),
                max_det=max_det,
                input_range_known=False,
            )

        sources = self._expand_sources(source)
        results = []
        for start in range(0, len(sources), batch):
            results.extend(
                self._predict_batch_impl(
                    sources[start:start + batch], (target_h, target_w), conf, device=device,
                    max_det=max_det, retain_images=retain_images,
                )
            )
        return results

    def _predict_batch(self, sources, imgsz, conf=0.25, device=None, max_det=300,
                       retain_images=True):
        return self._predict_batch_impl(
            sources, imgsz, conf, device=device, bgr=False, max_det=max_det,
            retain_images=retain_images,
        )

    def predict_bgr(self, source, imgsz=None, conf=0.25, max_det=300, device=None,
                    batch=16, retain_images=True):
        """Run RGB-model inference from BGR OpenCV frames and return a list of Results."""
        target_h, target_w = self._resolve_predict_imgsz(imgsz)
        device = torch.device(device or self.device)
        self.model.to(device).eval()
        batch = max(int(batch or 1), 1)
        sources = list(source) if isinstance(source, (list, tuple)) else [source]
        results = []
        for start in range(0, len(sources), batch):
            results.extend(
                self._predict_batch_impl(
                    sources[start:start + batch], (target_h, target_w), conf, device=device,
                    bgr=True, max_det=max_det, retain_images=bool(retain_images),
                )
            )
        return results

    def _predict_batch_impl(self, sources, imgsz, conf=0.25, device=None, bgr=False,
                            max_det=300, retain_images=True):
        if not sources:
            return []
        device = device or self.device
        tensors = []
        orig_imgs = []
        orig_shapes = []
        metas = []
        for source in sources:
            img = source if bgr else self._read_image_source(source)
            tensor, meta = self._preprocess_numpy_bgr(img, imgsz, device=None) if bgr else self._preprocess_image(img, imgsz, device=None)
            tensors.append(tensor)
            image_array = np.asarray(img)
            orig_shapes.append(tuple(int(v) for v in image_array.shape[:2]))
            orig_imgs.append(np.asarray(img)[..., ::-1].copy() if bgr and retain_images else (img if retain_images else None))
            metas.append(meta)
        batch_tensor = torch.cat(tensors, 0).to(device, non_blocking=True)
        return self._predict_tensor(
            batch_tensor, orig_imgs, conf, letterbox_meta=metas, orig_shapes=orig_shapes,
            max_det=max_det, input_range_known=True,
        )

    @staticmethod
    def _orig_images_from_tensor(tensor, retain_images=True):
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError("Tensor source must have shape [B,3,H,W].")
        if not retain_images:
            return [None] * int(tensor.shape[0])
        value = tensor.detach().float().cpu()
        if value.ndim == 3:
            value = value.unsqueeze(0)
        if value.ndim != 4:
            raise ValueError("Tensor source must have shape [B,3,H,W].")
        value_min, value_max = Fotonet._tensor_value_range(value)
        if not math.isfinite(value_min) or not math.isfinite(value_max) or value_min < 0.0 or value_max > 255.0:
            raise ValueError("Tensor image values must be finite and in [0,1] or [0,255].")
        if value_max <= 1.0:
            value = value * 255.0
        images = [
            image.permute(1, 2, 0).clamp(0, 255).byte().numpy().copy()
            for image in value
        ]
        return images

    @staticmethod
    def _tensor_value_range(tensor):
        """Return a tensor's min/max through one host transfer when it lives on CUDA."""
        value_min, value_max = torch.aminmax(tensor.detach())
        values = torch.stack((value_min, value_max)).cpu().tolist()
        return float(values[0]), float(values[1])

    @staticmethod
    def _normalize_tensor_input(tensor, *, input_range_known=False):
        """Validate and normalize public tensor input without repeated CUDA scalar reads."""
        if input_range_known:
            return tensor

        value_min, value_max = Fotonet._tensor_value_range(tensor)
        if not math.isfinite(value_min) or not math.isfinite(value_max) or value_min < 0.0 or value_max > 255.0:
            raise ValueError("Tensor image values must be finite and in [0,1] or [0,255].")
        if not tensor.is_floating_point():
            # Integer image tensors conventionally encode pixel intensities in
            # [0, 255], including the degenerate all-zero/all-one cases.
            return tensor.float().div_(255.0)
        if value_max > 1.0:
            return tensor.float().div_(255.0)
        return tensor

    @staticmethod
    def _as_rgb_uint8(img, *, bgr=False):
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
            raise ValueError("Image arrays must have shape [H,W], [H,W,3], or [H,W,4].")
        arr = arr[..., :3]
        if np.issubdtype(arr.dtype, np.floating):
            if not np.isfinite(arr).all() or arr.min() < 0.0 or arr.max() > 255.0:
                raise ValueError("Floating image arrays must contain finite values in [0,1] or [0,255].")
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.rint(arr).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if bgr:
            arr = arr[..., ::-1]
        return np.ascontiguousarray(arr)

    @classmethod
    def _preprocess_rgb_array(cls, arr, imgsz, device=None):
        """One RGB letterbox adapter shared by PIL, NumPy, and BGR entry points."""
        target_h, target_w = cls._normalize_imgsz(imgsz)
        arr = cls._as_rgb_uint8(arr)
        orig_h, orig_w = arr.shape[:2]
        gain = min(target_h / max(orig_h, 1), target_w / max(orig_w, 1))
        new_w = max(int(round(orig_w * gain)), 1)
        new_h = max(int(round(orig_h * gain)), 1)
        if cv2 is not None:
            resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = np.asarray(Image.fromarray(arr).resize((new_w, new_h), Image.BILINEAR))
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        tensor = torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1))).float().div_(255.0).unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device, non_blocking=True)
        return tensor, {
            "input_w": float(target_w),
            "input_h": float(target_h),
            "gain": float(gain),
            "pad_w": float(pad_w),
            "pad_h": float(pad_h),
            "orig_w": float(orig_w),
            "orig_h": float(orig_h),
        }

    @staticmethod
    def _preprocess_image(img, imgsz, device):
        """Letterbox image to the training geometry and return scale metadata."""
        if not isinstance(img, Image.Image):
            return Fotonet._preprocess_rgb_array(img, imgsz, device)
        return Fotonet._preprocess_rgb_array(np.asarray(img.convert("RGB")), imgsz, device)

    @staticmethod
    def _preprocess_numpy_rgb(img, imgsz, device=None):
        """Fast NumPy/OpenCV letterbox path for already-decoded RGB frames."""
        return Fotonet._preprocess_rgb_array(img, imgsz, device)

    @staticmethod
    def _preprocess_numpy_bgr(img, imgsz, device=None):
        """Fast letterbox path for OpenCV BGR frames, converting to RGB only after resize."""
        return Fotonet._preprocess_rgb_array(Fotonet._as_rgb_uint8(img, bgr=True), imgsz, device)

    @staticmethod
    def _scale_boxes_from_letterbox(boxes, meta):
        """Map normalized xywh boxes from letterboxed square input back to the original image."""
        if meta is None:
            return boxes
        input_w = float(meta.get("input_w", meta.get("imgsz")))
        input_h = float(meta.get("input_h", meta.get("imgsz")))
        gain = max(float(meta["gain"]), 1e-9)
        pad_w = float(meta["pad_w"])
        pad_h = float(meta["pad_h"])
        orig_w = max(float(meta["orig_w"]), 1.0)
        orig_h = max(float(meta["orig_h"]), 1.0)

        x, y, w, h = boxes.unbind(-1)
        x1 = (x - w * 0.5) * input_w
        y1 = (y - h * 0.5) * input_h
        x2 = (x + w * 0.5) * input_w
        y2 = (y + h * 0.5) * input_h

        x1 = ((x1 - pad_w) / gain).clamp(0, orig_w)
        y1 = ((y1 - pad_h) / gain).clamp(0, orig_h)
        x2 = ((x2 - pad_w) / gain).clamp(0, orig_w)
        y2 = ((y2 - pad_h) / gain).clamp(0, orig_h)

        cx = ((x1 + x2) * 0.5) / orig_w
        cy = ((y1 + y2) * 0.5) / orig_h
        bw = (x2 - x1).clamp_min(0) / orig_w
        bh = (y2 - y1).clamp_min(0) / orig_h
        return torch.stack((cx, cy, bw, bh), -1).clamp(0, 1)

    @staticmethod
    def _clip_normalized_xywh(boxes):
        """Clip normalized boxes through xyxy, preserving valid edge geometry."""
        x, y, w, h = boxes.unbind(-1)
        x1 = (x - w * 0.5).clamp(0, 1)
        y1 = (y - h * 0.5).clamp(0, 1)
        x2 = (x + w * 0.5).clamp(0, 1)
        y2 = (y + h * 0.5).clamp(0, 1)
        return torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1), dim=-1)

    @staticmethod
    def _letterbox_tensor_batch(tensor, target_h, target_w):
        """Letterbox a normalized NCHW batch without a CPU image round-trip."""
        if tensor.ndim != 4 or tensor.shape[1] != 3:
            raise ValueError("Tensor source must have shape [B,3,H,W].")
        orig_h, orig_w = (int(tensor.shape[-2]), int(tensor.shape[-1]))
        gain = min(target_h / max(orig_h, 1), target_w / max(orig_w, 1))
        new_h = max(int(round(orig_h * gain)), 1)
        new_w = max(int(round(orig_w * gain)), 1)
        resized = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
        canvas = tensor.new_full((tensor.shape[0], 3, target_h, target_w), 114.0 / 255.0)
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        canvas[..., pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        meta = {
            "input_w": float(target_w),
            "input_h": float(target_h),
            "gain": float(gain),
            "pad_w": float(pad_w),
            "pad_h": float(pad_h),
            "orig_w": float(orig_w),
            "orig_h": float(orig_h),
        }
        return canvas, [dict(meta) for _ in range(int(tensor.shape[0]))]

    def _predict_tensor(
        self,
        tensor,
        orig_img,
        conf=0.25,
        letterbox_meta=None,
        orig_shapes=None,
        force_list=True,
        max_det=300,
        input_range_known=False,
    ):
        device = tensor.device
        try:
            model_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            model_dtype = getattr(self.model, "input_torch_dtype", torch.float32)
        tensor = self._normalize_tensor_input(tensor, input_range_known=input_range_known)
        if tensor.dtype != model_dtype:
            tensor = tensor.to(dtype=model_dtype)

        # A fixed-shape artifact cannot consume arbitrary direct tensors.  For
        # the public tensor API we letterbox on-device and carry the inverse
        # transform into Results; decoded image sources already have this shape
        # because ``_resolve_predict_imgsz`` selected the artifact contract.
        if (
            self.runtime_format is not None
            and self._runtime_dynamic is False
            and self._runtime_imgsz is not None
            and tuple(int(value) for value in tensor.shape[-2:]) != tuple(self._runtime_imgsz)
        ):
            if letterbox_meta is not None:
                raise ValueError(
                    f"This static {self.runtime_format} artifact expects HxW={tuple(self._runtime_imgsz)}; "
                    "decoded inputs must be preprocessed at that size."
                )
            tensor, letterbox_meta = self._letterbox_tensor_batch(tensor, *self._runtime_imgsz)

        with torch.inference_mode():
            out = self.model(tensor)
        self._sync_runtime_output_schema(out)
        pred_logits = out[:, :, :self.model.head.nc]
        pred_boxes  = out[:, :, self.model.head.nc:]

        if pred_logits.ndim == 2:
            pred_logits = pred_logits.unsqueeze(0)
            pred_boxes = pred_boxes.unsqueeze(0)

        batched_orig = orig_img if isinstance(orig_img, list) else [orig_img]
        batched_meta = letterbox_meta if isinstance(letterbox_meta, list) else [letterbox_meta] * pred_logits.shape[0]
        batched_shapes = orig_shapes if isinstance(orig_shapes, list) else [orig_shapes] * pred_logits.shape[0]
        results = []
        for index in range(pred_logits.shape[0]):
            boxes, scores, classes = self._postprocess_single(
                pred_logits[index],
                pred_boxes[index],
                conf=conf,
                letterbox_meta=batched_meta[index],
                max_det=max_det,
            )
            results.append(
                Results(
                    batched_orig[index], boxes, scores, classes, names=self.names,
                    orig_shape=batched_shapes[index],
                )
            )

        return results

    def _postprocess_single(self, pred_logits, pred_boxes, conf=0.25,
                            letterbox_meta=None, max_det=300):
        scores, classes = pred_logits.sigmoid().max(-1)

        topk = min(int(max_det), int(scores.numel()))
        scores, indices = torch.topk(scores, topk)
        boxes, classes = pred_boxes[indices], classes[indices]
        mask = scores > conf
        boxes, scores, classes = boxes[mask], scores[mask], classes[mask]

        boxes = self._scale_boxes_from_letterbox(boxes, letterbox_meta)
        # Tensor input has no letterbox inverse transform. Clamp it too, so
        # right/bottom stride padding never leaks out-of-image coordinates.
        return self._clip_normalized_xywh(boxes), scores, classes

    def __call__(self, source, **kwargs):
        return self.predict(source, **kwargs)

    def track(self, source, *, persist=False, tracker="iou", tracker_iou=0.3, max_age=30, **kwargs):
        """Track ordered frames with explicit, dependency-free IoU association.

        [SIMPLIFIED] ``tracker='iou'`` associates same-class detections by IoU;
        it intentionally does not claim motion estimation or re-identification.
        """
        if str(tracker).lower() != "iou":
            raise ValueError("Only the built-in tracker='iou' is available.")
        from fotonet.engine.tracker import IoUTracker

        current = getattr(self, "_tracker", None)
        if not persist or current is None or current.iou_threshold != float(tracker_iou) or current.max_age != int(max_age):
            current = IoUTracker(iou_threshold=tracker_iou, max_age=max_age)
            if persist:
                self._tracker = current

        stream = bool(kwargs.get("stream", False))
        predictions = self.predict(source, **kwargs)
        if stream:
            return (current.update(result) for result in predictions)
        return [current.update(result) for result in predictions]

    def val(
        self,
        data,
        imgsz=None,
        batch=8,
        conf=0.0,
        max_det=100,
        operating_conf=0.25,
        operating_iou=0.50,
        workers=0,
        annotation_policy="fix",
        allow_missing_labels=False,
        source_recursive=True,
        progress=False,
        progress_interval=100,
    ):
        """Evaluate an explicit validation source using the trainer's COCO protocol.

        ``conf=0`` retains every finite detection so COCO maxDets performs the
        only score-ranked truncation.  A nonzero ``conf`` is explicitly labeled
        as a score-floor protocol. ``operating_conf`` and ``operating_iou``
        define the separately reported P/R point. A ``val`` entry is mandatory:
        validation never falls back to ``train``. A static ONNX/TorchScript
        artifact uses its exported input HxW when ``imgsz`` is omitted.
        """
        from torch.utils.data import DataLoader

        from fotonet.data.dataset import build_detection_dataset
        from fotonet.engine.validation import evaluate_detection_model

        cfg = load_data_cfg(data)
        if "val" not in cfg:
            raise ValueError(
                "Fotonet.val() requires an explicit 'val' source in the data config; "
                "it will not evaluate the training set."
            )
        if getattr(self, "_runtime_schema_pending", False):
            raise RuntimeError(
                "This metadata-less dynamic ONNX artifact has a symbolic output width, so its class count "
                "is not known yet. Run predict() once to resolve the runtime schema, or restore the export "
                "metadata/re-export before validating."
            )
        if not hasattr(self.model, "head") or not hasattr(self.model.head, "nc"):
            raise RuntimeError("Validation requires a detector runtime with a head.nc output contract.")
        model_nc = int(self.model.head.nc)
        configured_nc = cfg.get("nc")
        if configured_nc is not None and int(configured_nc) != model_nc:
            raise ValueError(
                f"Dataset nc={int(configured_nc)} does not match model nc={model_nc}. "
                "Load a compatible checkpoint/model instead of mutating the model during validation."
            )
        configured_names = cfg.get("names")
        model_names = None
        if configured_names is not None:
            _, configured_names = normalize_class_schema(
                model_nc,
                configured_names,
                context="data config",
            )
            raw_model_names = getattr(self, "names", None)
            if raw_model_names is not None:
                _, model_names = normalize_class_schema(
                    model_nc,
                    raw_model_names,
                    context="model",
                )
            if (
                model_names is not None
                and not class_names_are_default(model_names, model_nc)
                and model_names != configured_names
            ):
                raise ValueError(
                    "Dataset class names/order do not match the loaded model checkpoint. "
                    "Use a data config with the same class mapping."
                )
        batch = int(batch)
        if batch < 1:
            raise ValueError("batch must be positive")
        workers = int(workers)
        if workers < 0:
            raise ValueError("workers must be non-negative")

        runtime_format = getattr(self, "runtime_format", None)
        runtime_dynamic = getattr(self, "_runtime_dynamic", None)
        runtime_imgsz = getattr(self, "_runtime_imgsz", None)
        if runtime_format is None:
            eval_imgsz = self._normalize_imgsz(640 if imgsz is None else imgsz)
        elif runtime_dynamic is False:
            if runtime_imgsz is None:
                raise RuntimeError(
                    f"Cannot validate this static {runtime_format} artifact without its input-shape metadata. "
                    "Re-export it with FOTO-NET so the adjacent metadata file is produced."
                )
            eval_imgsz = tuple(int(value) for value in runtime_imgsz)
            if imgsz is not None and self._normalize_imgsz(imgsz) != eval_imgsz:
                raise ValueError(
                    f"This static {runtime_format} artifact accepts imgsz={eval_imgsz}, "
                    f"not {self._normalize_imgsz(imgsz)}. Re-export with dynamic=True for another validation shape."
                )
        else:
            if runtime_dynamic is None and runtime_imgsz is None:
                raise RuntimeError(
                    f"Cannot determine the input contract for '{getattr(self, 'loaded_weight_path', None)}'. "
                    "Keep the export metadata sidecar or load a native PyTorch checkpoint."
                )
            eval_imgsz = self._normalize_imgsz(640 if imgsz is None else imgsz)

        if progress:
            print("[validate] building validation dataset...", flush=True)
        dataset = build_detection_dataset(
            cfg["val"],
            imgsz=eval_imgsz,
            augment=False,
            cache_labels=True,
            cache_to_ram=False,
            num_classes=model_nc,
            annotation_policy=annotation_policy,
            allow_missing_labels=allow_missing_labels,
            source_recursive=source_recursive,
            coco_images=cfg.get("val_images", cfg.get("coco_images")),
        )
        def collate_detection_batch(items):
            images, targets = zip(*items)
            if isinstance(images[0], dict):
                common_keys = set(images[0]).intersection(*(set(image) for image in images[1:]))
                return {
                    key: torch.stack([image[key] for image in images], 0)
                    for key in common_keys
                }, list(targets)
            return torch.stack(images, 0), list(targets)

        loader = DataLoader(
            dataset,
            batch_size=batch,
            shuffle=False,
            num_workers=workers,
            pin_memory=torch.device(self.device).type == "cuda",
            collate_fn=collate_detection_batch,
            persistent_workers=workers > 0,
        )
        if progress:
            print(
                f"[validate] dataset ready: images={len(dataset)} batches={len(loader)}",
                flush=True,
            )
        metrics = evaluate_detection_model(
            self.model,
            loader,
            self.device,
            model_nc,
            conf=conf,
            coco_max_dets=max_det,
            operating_conf=operating_conf,
            operating_iou=operating_iou,
            amp=False,
            class_names=(
                configured_names
                if configured_names is not None
                else getattr(self, "names", default_class_names(model_nc))
            ),
            progress=progress,
            progress_interval=progress_interval,
        )
        # Maintain the historic spelling without making it a separate metric.
        metrics["mAP50-95"] = metrics["mAP50_95"]
        return metrics

    def _mirror_o2o_to_o2m(self):
        """Rebuild training-only O2M heads when loading slim inference checkpoints."""
        state = self.model.state_dict()
        for key in list(state.keys()):
            if "head.cls_o2m." in key:
                src = key.replace("head.cls_o2m.", "head.cls_o2o.")
            elif "head.reg_o2m." in key:
                src = key.replace("head.reg_o2m.", "head.reg_o2o.")
            elif "head.quality_o2m." in key:
                src = key.replace("head.quality_o2m.", "head.quality_o2o.")
            else:
                continue
            if src in state and state[src].shape == state[key].shape:
                state[key].copy_(state[src])
        self.model.load_state_dict(state)

    def strip_o2m_for_inference(self):
        """Remove O2M heads from an isolated inference graph.

        The original graph is retained for a later ``train()`` call, matching
        the non-mutating deploy behavior of ``prepare_for_inference()``.
        """
        if self.runtime_format is None:
            self._ensure_deployment_copy()
        raw = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        if hasattr(raw, "strip_o2m_for_inference"):
            raw.strip_o2m_for_inference()
            raw.eval()
        return self

    def load(self, model_path):
        """Load one explicit production checkpoint; identity guessing is prohibited."""
        if self.runtime_format is not None:
            raise RuntimeError("Cannot load a native checkpoint into an exported runtime")
        path = os.fspath(model_path)
        checkpoint = self._safe_torch_load(path)
        config = self._checkpoint_config(checkpoint)
        checkpoint_nc, checkpoint_names = normalize_class_schema(
            checkpoint.get("nc"), checkpoint.get("names"), context="checkpoint"
        )
        config["nc"] = checkpoint_nc
        self._apply_checkpoint_config(config)
        self.names = dict(checkpoint_names or default_class_names(checkpoint_nc))
        state = self._checkpoint_inference_state_dict(checkpoint)
        self._load_model_state(state, inference_only=self._checkpoint_is_inference_only(checkpoint))
        self.loaded_weight_path = path
        return self

    @staticmethod
    def _state_dict_for_save(model, half=False, strip_o2m=False):
        state = model.state_dict()
        out = {}
        for k, v in state.items():
            if strip_o2m and (
                k.startswith("head.cls_o2m.")
                or k.startswith("head.reg_o2m.")
                or k.startswith("head.quality_o2m.")
            ):
                continue
            if torch.is_tensor(v):
                v = v.detach().cpu()
                out[k] = v.half() if half and v.dtype.is_floating_point else v
            else:
                out[k] = v
        return out

    def save(self, model_path, inference_only=False, half=False):
        model_path = os.fspath(model_path)
        parent = os.path.dirname(model_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # ``prepare_for_inference`` intentionally keeps an unfused graph for
        # future training. A normal save must serialize that graph, not the
        # disposable fused deployment copy, or a reloaded checkpoint would
        # silently lose BatchNorm training behavior.
        source_model = self.model
        if not inference_only and self._training_model_before_deploy is not None:
            source_model = self._training_model_before_deploy
        raw = source_model._orig_mod if hasattr(source_model, "_orig_mod") else source_model
        if not inference_only and bool(getattr(raw, "_fused", False)):
            raise RuntimeError(
                "Refusing to save a fused graph as a trainable checkpoint. "
                "Use inference_only=True or reload/restore an unfused training model first."
            )
        strip_o2m = bool(inference_only)
        checkpoint = {
            "model": self._state_dict_for_save(raw, half=half, strip_o2m=strip_o2m),
            "checkpoint_format": self.CHECKPOINT_FORMAT,
            "architecture_schema": self.architecture_schema,
            "architecture_fingerprint": self.architecture_fingerprint,
            "training_protocol": None if inference_only else self.TRAINING_PROTOCOL,
            "model_id": self.model_id,
            "nc": self.nc,
            "names": self.names,
            "inference_only": bool(inference_only),
            "stripped_o2m": strip_o2m,
            "has_o2m": not strip_o2m,
            "fused": bool(getattr(raw, "_fused", False)),
            "model_config": getattr(raw, "model_config", self._model_cfg),
            "raw_output": "[B,N,nc+4] class_logits+normalized_xywh",
        }
        torch.save(checkpoint, model_path)
        print(f"[INFO] Saved to '{model_path}'.")

    def export(self, path=None, format="onnx", imgsz=640, batch=1, dynamic=False,
               half=False, simplify=True, int8=False, opset=17, device=None, verify=True, **kwargs):
        """Export an inference graph and return ``{'artifact', 'metadata'}``.

        Returning metadata makes preprocessing and coordinate contracts visible
        to deployment callers instead of burying them beside the file.
        """
        from fotonet.export import export_model
        return export_model(
            self,
            path=path,
            format=format,
            imgsz=imgsz,
            batch=batch,
            dynamic=dynamic,
            half=half,
            simplify=simplify,
            int8=int8,
            opset=opset,
            device=device,
            verify=verify,
            **kwargs,
        )
