"""
FOTO-NET main model API with an Ultralytics-style interface.
"""
import os
import torch
import numpy as np
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None

# Enable TensorFloat32 for better performance on RTX 30/40 series GPUs
torch.set_float32_matmul_precision('high')

from fotonet.models.fotonet import FOTONETModel
from fotonet.models.scales import available_model_scales, is_model_scale_ref, load_scale_config, scale_fallback_configs
from fotonet.data.dataset import FOTONETDataset
from fotonet.engine.trainer import Trainer
from fotonet.engine.results import Results
from fotonet.utils.general import check_device
from fotonet.utils.config import load_data_cfg, load_model_cfg

__version__ = "0.1.0a0"


class FOTONET:
    """
    FOTO-NET: NMS-free object detection with custom backbone/neck/head.
    API: FOTONET('fotonetn') | FOTONET('path/to.pt') | FOTONET(nc=80)
    """
    SCALES = available_model_scales()

    def __init__(self, model_path=None, nc=None, task="detect"):
        self.device = check_device()
        self.task   = task
        self.nc     = nc or 80
        self.model  = None
        self._model_cfg = None
        self.model_path = model_path
        self.loaded_weight_path = None
        self._apply_model_cfg(load_scale_config("n"), update_nc=False)
        self.names = {i: f"class_{i}" for i in range(self.nc)}

        if model_path is None:
            self.model = self._new_model()
            return

        model_path = os.fspath(model_path)
        path_lower = model_path.lower()

        if os.path.isfile(model_path) and path_lower.endswith((".yaml", ".yml")):
            cfg = load_model_cfg(model_path)
            self._model_cfg = cfg
            self._apply_model_cfg(cfg, update_nc=True)
            self.model = self._new_model()

        elif os.path.isfile(model_path):
            self.model = self._new_model()
            self.load(model_path)
            self.loaded_weight_path = model_path

        elif is_model_scale_ref(path_lower):
            self._apply_model_cfg(load_scale_config(path_lower), update_nc=True)
            self.model = self._new_model()
            return

        else:
            valid = ", ".join(f"fotonet{k}" for k in available_model_scales())
            raise ValueError(f"Unknown model '{model_path}'. Use one of: {valid}, a YAML file, or a checkpoint.")

    def _apply_model_cfg(self, cfg, update_nc=True):
        if update_nc:
            self.nc = int(cfg.get("nc", self.nc))
        self.w = float(cfg.get("width_multiple", cfg.get("w", getattr(self, "w", 0.2))))
        self.d = float(cfg.get("depth_multiple", cfg.get("d", getattr(self, "d", 0.33))))
        self.use_p2 = bool(cfg.get("p2_head", cfg.get("use_p2", getattr(self, "use_p2", True))))
        self.reg_max = int(cfg.get("reg_max", getattr(self, "reg_max", 1)))
        self.p3_extra_blocks = max(int(cfg.get("p3_extra_blocks", getattr(self, "p3_extra_blocks", 0))), 0)
        self.p4_extra_blocks = max(int(cfg.get("p4_extra_blocks", getattr(self, "p4_extra_blocks", 0))), 0)
        self.p5_extra_blocks = max(int(cfg.get("p5_extra_blocks", getattr(self, "p5_extra_blocks", 0))), 0)
        self.p5_gate_blocks = max(int(cfg.get("p5_gate_blocks", getattr(self, "p5_gate_blocks", 0))), 0)
        self.arch_version = int(cfg.get("arch_version", getattr(self, "arch_version", 1)))
        self.neck_fusion = str(cfg.get("neck_fusion", getattr(self, "neck_fusion", "concat")))
        self.p2_context_blocks = max(int(cfg.get("p2_context_blocks", getattr(self, "p2_context_blocks", 0))), 0)
        self.p3_context_blocks = max(int(cfg.get("p3_context_blocks", getattr(self, "p3_context_blocks", 0))), 0)
        self.quality_head = bool(cfg.get("quality_head", getattr(self, "quality_head", False)))

    def _new_model(self):
        return FOTONETModel(
            nc=self.nc,
            w=self.w,
            d=self.d,
            use_p2=self.use_p2,
            reg_max=self.reg_max,
            p3_extra_blocks=self.p3_extra_blocks,
            p4_extra_blocks=self.p4_extra_blocks,
            p5_extra_blocks=self.p5_extra_blocks,
            p5_gate_blocks=self.p5_gate_blocks,
            arch_version=self.arch_version,
            neck_fusion=self.neck_fusion,
            p2_context_blocks=self.p2_context_blocks,
            p3_context_blocks=self.p3_context_blocks,
            quality_head=self.quality_head,
        ).to(self.device)

    def load_weights(self, weights_path, auto_transfer=True):
        """
        Loads weights from a .pt or .pth file.
        """
        if not os.path.exists(weights_path):
            print(f"[ERROR] Weights file not found: {weights_path}")
            return False

        print(f"[INFO] Loading weights from '{weights_path}' ...")
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict):
            if "model" in ckpt:
                sd = ckpt["model"]
            elif "model_state_dict" in ckpt:
                sd = ckpt["model_state_dict"]
            else:
                sd = ckpt
        else:
            sd = ckpt

        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()

        # 1. Official YOLOX-Nano mapping
        is_official_yolox = any(k.startswith("backbone.backbone.") for k in sd.keys())
        if is_official_yolox and auto_transfer:
            print("[INFO] Official YOLOX-Nano weights detected. Mapping layers...")
            # Key mappings: YOLOX official -> FOTONET
            mapping = {
                "backbone.backbone.": "backbone.",
                "backbone.lateral_conv0": "neck.conv_p5_p4",
                "backbone.C3_p4":        "neck.c2f_p4_td",
                "backbone.reduce_conv1": "neck.conv_p4_p3",
                "backbone.C3_p3":        "neck.c2f_p3_td",
                "backbone.bu_conv2":     "neck.down_p3_p4",
                "backbone.C3_n3":        "neck.c2f_n4_bu",
                "backbone.bu_conv1":     "neck.down_p4_p5",
                "backbone.C3_n4":        "neck.c2f_n5_bu",
            }
            new_sd = {}
            for k, v in sd.items():
                mapped = False
                for old_pfx, new_pfx in mapping.items():
                    if k.startswith(old_pfx):
                        new_k = k.replace(old_pfx, new_pfx, 1)
                        new_sd[new_k] = v
                        mapped = True
                        break
                if not mapped:
                    new_sd[k] = v
            sd = new_sd

        # Legacy check
        is_yolo8 = any(k.startswith("model.0.") for k in sd.keys())
        if is_yolo8 and auto_transfer:
             print("[INFO] YOLOv8 weights detected. Automatic mapping has been deprecated.")
             pass

        cur_sd = self.model.state_dict()
        matched = 0
        skipped = 0

        for k, v in sd.items():
            if k in cur_sd:
                if cur_sd[k].shape == v.shape:
                    cur_sd[k] = v.to(self.device).float()
                    matched += 1
                else:
                    # Silently skip head mismatches unless debug? 
                    # For now just log it.
                    skipped += 1
            else:
                pass

        self.model.load_state_dict(cur_sd)
        print(f"[INFO] Loaded {matched} layers successfully from backbone/neck.")
        self.loaded_weight_path = weights_path
        return True

    def _load_backbone_only(self, weights_path):
        if not weights_path or not os.path.exists(weights_path):
            raise FileNotFoundError(f"Backbone weights not found: {weights_path}")

        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()

        self.model = self._new_model()
        cur_sd = self.model.state_dict()
        matched, skipped = 0, 0
        for k, v in sd.items():
            if not k.startswith("backbone."):
                continue
            if k in cur_sd and cur_sd[k].shape == v.shape:
                cur_sd[k] = v.to(self.device).float()
                matched += 1
            else:
                skipped += 1
        self.model.load_state_dict(cur_sd)
        print(f"[INFO] pretrained=False with checkpoint init: loaded backbone only matched={matched}, skipped={skipped}.")

    def _load_matched_state_dict(self, sd):
        """Load only tensors whose keys and shapes fit the current model."""
        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()
        cur_sd = self.model.state_dict()
        matched, skipped = 0, 0
        for k, v in sd.items():
            if k in cur_sd and hasattr(v, "shape") and cur_sd[k].shape == v.shape:
                target = cur_sd[k]
                cur_sd[k] = v.to(device=self.device, dtype=target.dtype)
                matched += 1
            else:
                skipped += 1
        self.model.load_state_dict(cur_sd)
        return matched, skipped

    @staticmethod
    def _checkpoint_state_dict(ckpt):
        if isinstance(ckpt, dict):
            sd = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        else:
            sd = ckpt
        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()
        return sd

    @staticmethod
    def _checkpoint_is_inference_only(ckpt, sd=None):
        if not isinstance(ckpt, dict):
            return False
        explicit = bool(ckpt.get("inference_only", False) or ckpt.get("stripped_o2m", False) or ckpt.get("has_o2m") is False)
        if explicit:
            return True
        if sd is None:
            sd = FOTONET._checkpoint_state_dict(ckpt)
        return isinstance(sd, dict) and not any(
            str(k).startswith(("head.cls_o2m.", "head.reg_o2m.", "head.quality_o2m.")) for k in sd.keys()
        )

    @staticmethod
    def _checkpoint_is_full_training_resume(ckpt):
        if not isinstance(ckpt, dict):
            return False
        required = ("model", "optimizer_state", "scheduler_state", "scaler_state", "epoch")
        return all(k in ckpt for k in required)

    @staticmethod
    def _resume_checkpoint_error(weights_path):
        return (
            f"Cannot resume training from '{weights_path}'. This checkpoint is not a full training resume file. "
            "If you want to start a fresh run from these weights, use pretrained=True and resume=False. "
            "If you want to continue an interrupted run, use the resumable checkpoint 'fotonet_last.pt'."
        )

    def _ensure_training_o2m_heads(self, source_sd=None, mirror=False):
        """Ensure training-only O2M heads exist and are initialized from O2O heads."""
        raw = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        head = getattr(raw, "head", None)
        has_o2m = bool(getattr(head, "has_o2m", True))
        rebuilt = False
        if not has_o2m:
            current_sd = source_sd or raw.state_dict()
            self.model = self._new_model()
            matched, skipped = self._load_matched_state_dict(current_sd)
            print(f"[INFO] Rebuilt O2M heads from slim checkpoint: matched={matched}, skipped={skipped}.")
            rebuilt = True
        if rebuilt or mirror:
            self._mirror_o2o_to_o2m()

    def freeze_backbone(self, freeze=True):
        """Freeze/unfreeze backbone weights (useful for first N epochs)."""
        for p in self.model.backbone.parameters():
            p.requires_grad_(not freeze)
        state = "frozen" if freeze else "unfrozen"
        print(f"[INFO] Backbone {state}.")

    def train(self, data, epochs=100, imgsz=640, batch=16, lr0=0.001, lrf=0.01,
              nbs=128, warmup_epochs=3, frozen_epochs=0, save_dir=".", val_split=0.2, val_period=1, val_batch=None,
              workers=None, pin_memory=True, cache_to_ram=True, ram_cache_images=1024, amp=True, val_amp=None, cos_lr=True,
              cuda_graphs=False, profile=False,
              weights=None, resume=False, pretrained=True,
              epoch_cut=1, compile_model=False,
              save_period=-1, slim_best=True, best_metric="mAP50_95", save_last=True,
              momentum=0.937, weight_decay=0.0005,
              augment_hyp=None, loss_hyp=None, matcher_hyp=None,
              imgsz_schedule=None, val_subset_size=0, full_val_after=1.0,
              cache_labels=True, disk_cache_images=False, disk_cache_dir=None,
              unfreeze_backbone_at=None,
              distill=None,
              distill_teacher=None, distill_weight=0.0,
              distill_start_epoch=1, distill_warmup_epochs=1,
              distill_end_epoch=None, distill_conf=0.25, distill_topk=64):
        """
        Train the FOTONET model.

        Preferred API:
            weights (str | None): Optional checkpoint path.
            resume (bool | str): Full checkpoint resume from a training checkpoint such as fotonet_last.pt.
            pretrained (bool): Fresh optimizer/scheduler from current or weights=path model weights.

        Args:
            weights: Optional `.pt` checkpoint path.

            resume: If True, restores a full checkpoint saved by this
                trainer (must contain optimizer_state, scaler_state, ema_state, epoch).
                Fully resumes all training state so the run picks up exactly where it
                left off, including LR schedule position and best_mAP.

            pretrained: If True, loads model weights only and starts fresh optimizer
                state. If False with a checkpoint-backed model, keeps compatible
                backbone weights only.

            frozen_epochs: Number of initial epochs to keep the backbone frozen.
                ``unfreeze_backbone_at`` remains accepted as a deprecated alias.

            distill: Optional distillation config as a teacher checkpoint path
                or dict, e.g. {"teacher": "teacher.pt", "weight": 0.25}.
                ``distill_teacher`` and ``distill_weight`` remain accepted as
                compatibility aliases.
        """
        resume_ckpt = None
        resume_sd = None

        if isinstance(resume, str):
            weights = resume
            resume = True

        if bool(resume) and bool(pretrained):
            raise ValueError(
                "resume=True and pretrained=True are mutually exclusive. "
                "Use resume=True with 'fotonet_last.pt' to continue an interrupted run, "
                "or pretrained=True with resume=False to start a fresh run from model weights."
            )

        if unfreeze_backbone_at is not None:
            frozen_epochs = 0 if int(unfreeze_backbone_at) < 0 else int(unfreeze_backbone_at)

        if distill not in (None, False):
            if isinstance(distill, (str, os.PathLike)):
                distill_teacher = os.fspath(distill)
                if float(distill_weight or 0.0) <= 0.0:
                    distill_weight = 1.0
            elif isinstance(distill, dict):
                distill_teacher = distill.get("teacher", distill.get("weights", distill.get("model", distill_teacher)))
                if "weight" in distill:
                    distill_weight = float(distill["weight"])
                elif distill_teacher and float(distill_weight or 0.0) <= 0.0:
                    distill_weight = 1.0
                distill_start_epoch = distill.get("start_epoch", distill_start_epoch)
                distill_warmup_epochs = distill.get("warmup_epochs", distill_warmup_epochs)
                distill_end_epoch = distill.get("end_epoch", distill_end_epoch)
                distill_conf = distill.get("conf", distill_conf)
                distill_topk = distill.get("topk", distill_topk)
            else:
                raise ValueError("distill must be None, a teacher checkpoint path, or a dict with teacher/weight settings.")

        if resume:
            weights_path = weights or self.loaded_weight_path
            if not weights_path:
                raise ValueError(
                    "resume=True requires a full training checkpoint. "
                    "Initialize with FOTONET('fotonet_last.pt') or pass weights='fotonet_last.pt'."
                )
            print(f"[INFO] resume=True: loading full checkpoint from '{weights_path}'")
            resume_ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
            resume_sd = self._checkpoint_state_dict(resume_ckpt)
            if self._checkpoint_is_inference_only(resume_ckpt, resume_sd) or not self._checkpoint_is_full_training_resume(resume_ckpt):
                raise ValueError(self._resume_checkpoint_error(weights_path))

        cfg = load_data_cfg(data)
        if "nc" in cfg and cfg["nc"] != self.nc:
            self.nc = cfg["nc"]
            self.model = self._new_model()

        if "names" in cfg:
            self.names = cfg["names"] if isinstance(cfg["names"], dict) else {i: n for i, n in enumerate(cfg["names"])}

        self._ensure_training_o2m_heads()

        if resume:
            weights_path = weights or self.loaded_weight_path
            ckpt = resume_ckpt
            sd = resume_sd
            inferred_p2, inferred_reg_max, inferred_quality = self._infer_arch_from_state_dict(
                sd, self.use_p2, self.reg_max, self.quality_head
            ) if isinstance(sd, dict) else (self.use_p2, self.reg_max, self.quality_head)
            ckpt_nc = ckpt.get("nc", self.nc) if isinstance(ckpt, dict) else self.nc
            ckpt_w = ckpt.get("w", self.w) if isinstance(ckpt, dict) else self.w
            ckpt_d = ckpt.get("d", self.d) if isinstance(ckpt, dict) else self.d
            ckpt_p2 = bool(ckpt.get("use_p2", ckpt.get("p2_head", inferred_p2))) if isinstance(ckpt, dict) else inferred_p2
            ckpt_reg_max = int(ckpt.get("reg_max", inferred_reg_max)) if isinstance(ckpt, dict) else inferred_reg_max
            ckpt_quality = bool(ckpt.get("quality_head", inferred_quality)) if isinstance(ckpt, dict) else inferred_quality
            ckpt_p3_extra = int(ckpt.get("p3_extra_blocks", 0)) if isinstance(ckpt, dict) else self.p3_extra_blocks
            ckpt_p4_extra = int(ckpt.get("p4_extra_blocks", 0)) if isinstance(ckpt, dict) else self.p4_extra_blocks
            ckpt_p5_extra = int(ckpt.get("p5_extra_blocks", 0)) if isinstance(ckpt, dict) else self.p5_extra_blocks
            ckpt_p5_gate = int(ckpt.get("p5_gate_blocks", 0)) if isinstance(ckpt, dict) else self.p5_gate_blocks
            if (ckpt_nc != self.nc or ckpt_w != self.w or ckpt_d != self.d or
                    ckpt_p2 != self.use_p2 or ckpt_reg_max != self.reg_max or
                    ckpt_p3_extra != self.p3_extra_blocks or ckpt_p4_extra != self.p4_extra_blocks or
                    ckpt_p5_extra != self.p5_extra_blocks or ckpt_p5_gate != self.p5_gate_blocks or
                    ckpt_quality != self.quality_head):
                self.nc, self.w, self.d = ckpt_nc, ckpt_w, ckpt_d
                self.use_p2, self.reg_max = ckpt_p2, ckpt_reg_max
                self.p3_extra_blocks = ckpt_p3_extra
                self.p4_extra_blocks = ckpt_p4_extra
                self.p5_extra_blocks = ckpt_p5_extra
                self.p5_gate_blocks = ckpt_p5_gate
                self.quality_head = ckpt_quality
                self.model = self._new_model()
            self.model.load_state_dict(sd)
            # Pass the full checkpoint dict to Trainer for optimizer/scaler/ema restore
            resume_ckpt = ckpt

        elif pretrained:
            weights_path = weights if isinstance(weights, str) else self.loaded_weight_path
            if weights_path:
                print(f"[INFO] pretrained=True: loading model weights only from '{weights_path}'")
                ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
                sd = ckpt.get("model", ckpt)
                if hasattr(sd, "state_dict"):
                    sd = sd.state_dict()
                stripped_o2m = bool(ckpt.get("stripped_o2m", False)) if isinstance(ckpt, dict) else False
                has_o2m_keys = any(str(k).startswith(("head.cls_o2m.", "head.reg_o2m.")) for k in sd.keys())
                matched, skipped = self._load_matched_state_dict(sd)
                if stripped_o2m or not has_o2m_keys:
                    self._ensure_training_o2m_heads(mirror=True)
                self.loaded_weight_path = weights_path
                print(f"[INFO] pretrained: matched={matched}, skipped={skipped}. Starting from epoch 0.")
            else:
                print("[INFO] pretrained=True: using current model weights, fresh optimizer/scheduler.")

        elif not pretrained and (weights or self.loaded_weight_path):
            self._load_backbone_only(weights or self.loaded_weight_path)

        # --- Build dataset and start Trainer ---
        dataset = None
        if val_split != 0:
            dataset = FOTONETDataset(
                img_dir=cfg["train"],
                imgsz=imgsz,
                augment=True,
                cache_labels=cache_labels,
                disk_cache_images=disk_cache_images,
                disk_cache_dir=disk_cache_dir,
                cache_to_ram=cache_to_ram,
                ram_cache_images=ram_cache_images,
                augment_hyp=augment_hyp,
                num_classes=self.nc,
            )
        trainer = Trainer(
            self.model, cfg, epochs=epochs, imgsz=imgsz,
            lr0=lr0, lrf=lrf, batch=batch, val_batch=val_batch, nbs=nbs,
            warmup_epochs=warmup_epochs, save_dir=save_dir,
            val_split=val_split, val_period=val_period,
            workers=workers, pin_memory=pin_memory,
            cache_to_ram=cache_to_ram, ram_cache_images=ram_cache_images, amp=amp, val_amp=val_amp, cos_lr=cos_lr,
            cuda_graphs=cuda_graphs, profile=profile,
            resume_ckpt=resume_ckpt, epoch_cut=epoch_cut,
            compile_model=compile_model, save_period=save_period, slim_best=slim_best,
            best_metric=best_metric, save_last=save_last,
            momentum=momentum, weight_decay=weight_decay,
            augment_hyp=augment_hyp, loss_hyp=loss_hyp, matcher_hyp=matcher_hyp,
            imgsz_schedule=imgsz_schedule,
            val_subset_size=val_subset_size,
            full_val_after=full_val_after,
            cache_labels=cache_labels,
            disk_cache_images=disk_cache_images,
            disk_cache_dir=disk_cache_dir,
            distill_teacher=distill_teacher,
            distill_weight=distill_weight,
            distill_start_epoch=distill_start_epoch,
            distill_warmup_epochs=distill_warmup_epochs,
            distill_end_epoch=distill_end_epoch,
            distill_conf=distill_conf,
            distill_topk=distill_topk,
        )
        trainer.train(dataset, frozen_epochs=frozen_epochs)

    def prepare_for_inference(self, device=None, half=False, strip_o2m=False):
        """Put the loaded model in deploy mode without modifying the checkpoint file."""
        device = device or self.device
        self.model.to(device).eval()
        if strip_o2m:
            self.strip_o2m_for_inference()
        if half and torch.device(device).type == "cuda":
            self.model.half()
        return self

    def predict(self, source, imgsz=640, conf=0.25, use_nms=False, device=None, batch=16, **kwargs):
        device = device or self.device
        self.model.to(device).eval()
        batch = max(int(batch or 1), 1)

        if isinstance(source, list):
            results = []
            for start in range(0, len(source), batch):
                results.extend(self._predict_batch(source[start:start + batch], imgsz, conf, use_nms, device))
            return results
        if isinstance(source, str) and os.path.isdir(source):
            files = [os.path.join(source, f) for f in os.listdir(source)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            return self.predict(sorted(files), imgsz, conf, use_nms, device, batch=batch)

        if isinstance(source, str):
            img = Image.open(source).convert("RGB")
        elif isinstance(source, np.ndarray):
            img = source
        elif isinstance(source, torch.Tensor):
            # If it's already a tensor, just assume it's preprocessed or handle batching
            if source.ndim == 3:
                source = source.unsqueeze(0)
            tensor = source.to(device)
            return self._predict_tensor(
                tensor,
                self._orig_images_from_tensor(source),
                conf,
                use_nms,
                force_list=tensor.shape[0] > 1,
            )
        else:
            img = source

        tensor, letterbox_meta = self._preprocess_image(img, imgsz, device)
        return self._predict_tensor(tensor, img, conf, use_nms, letterbox_meta=letterbox_meta)

    def _predict_batch(self, sources, imgsz, conf=0.25, use_nms=False, device=None):
        return self._predict_batch_impl(sources, imgsz, conf, use_nms, device, bgr=False)

    def predict_bgr(self, source, imgsz=640, conf=0.25, use_nms=False, device=None, batch=16):
        """Fast OpenCV-frame inference. Input is BGR uint8, output boxes stay normalized xywh."""
        device = device or self.device
        self.model.to(device).eval()
        batch = max(int(batch or 1), 1)
        if isinstance(source, list):
            results = []
            for start in range(0, len(source), batch):
                results.extend(self._predict_batch_impl(source[start:start + batch], imgsz, conf, use_nms, device, bgr=True))
            return results
        tensor, letterbox_meta = self._preprocess_numpy_bgr(source, imgsz, device)
        return self._predict_tensor(tensor, source, conf, use_nms, letterbox_meta=letterbox_meta)

    def _predict_batch_impl(self, sources, imgsz, conf=0.25, use_nms=False, device=None, bgr=False):
        if not sources:
            return []
        device = device or self.device
        tensors = []
        orig_imgs = []
        metas = []
        for source in sources:
            if isinstance(source, str):
                img = Image.open(source).convert("RGB")
            elif isinstance(source, np.ndarray):
                img = source
            else:
                img = source
            tensor, meta = self._preprocess_numpy_bgr(img, imgsz, device=None) if bgr else self._preprocess_image(img, imgsz, device=None)
            tensors.append(tensor)
            orig_imgs.append(img)
            metas.append(meta)
        batch_tensor = torch.cat(tensors, 0).to(device, non_blocking=True)
        return self._predict_tensor(batch_tensor, orig_imgs, conf, use_nms, letterbox_meta=metas, force_list=True)

    @staticmethod
    def _orig_images_from_tensor(tensor):
        if tensor.ndim == 4:
            _, _, h, w = tensor.shape
            if int(tensor.shape[0]) == 1:
                return np.zeros((int(h), int(w), 3), dtype=np.uint8)
            return [np.zeros((int(h), int(w), 3), dtype=np.uint8) for _ in range(int(tensor.shape[0]))]
        if tensor.ndim == 3:
            _, h, w = tensor.shape
            return np.zeros((int(h), int(w), 3), dtype=np.uint8)
        return tensor

    @staticmethod
    def _preprocess_image(img, imgsz, device):
        """Letterbox image to the training geometry and return scale metadata."""
        if isinstance(img, np.ndarray):
            return FOTONET._preprocess_numpy_rgb(img, imgsz, device)
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        img = img.convert("RGB")
        orig_w, orig_h = img.size
        imgsz = int(imgsz)
        gain = min(imgsz / max(orig_h, 1), imgsz / max(orig_w, 1))
        new_w = max(int(round(orig_w * gain)), 1)
        new_h = max(int(round(orig_h * gain)), 1)
        resized = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
        pad_w = (imgsz - new_w) // 2
        pad_h = (imgsz - new_h) // 2
        canvas.paste(resized, (pad_w, pad_h))

        arr = np.array(canvas, copy=True)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float().div_(255.0).unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device, non_blocking=True)
        return tensor, {
            "imgsz": imgsz,
            "gain": float(gain),
            "pad_w": float(pad_w),
            "pad_h": float(pad_h),
            "orig_w": float(orig_w),
            "orig_h": float(orig_h),
        }

    @staticmethod
    def _preprocess_numpy_rgb(img, imgsz, device=None):
        """Fast NumPy/OpenCV letterbox path for already-decoded RGB frames."""
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        arr = np.ascontiguousarray(arr)
        orig_h, orig_w = arr.shape[:2]
        imgsz = int(imgsz)
        gain = min(imgsz / max(orig_h, 1), imgsz / max(orig_w, 1))
        new_w = max(int(round(orig_w * gain)), 1)
        new_h = max(int(round(orig_h * gain)), 1)
        if cv2 is not None:
            resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = np.asarray(Image.fromarray(arr).resize((new_w, new_h), Image.BILINEAR))
        canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        pad_w = (imgsz - new_w) // 2
        pad_h = (imgsz - new_h) // 2
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        tensor = torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1))).float().div_(255.0).unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device, non_blocking=True)
        return tensor, {
            "imgsz": imgsz,
            "gain": float(gain),
            "pad_w": float(pad_w),
            "pad_h": float(pad_h),
            "orig_w": float(orig_w),
            "orig_h": float(orig_h),
        }

    @staticmethod
    def _preprocess_numpy_bgr(img, imgsz, device=None):
        """Fast letterbox path for OpenCV BGR frames, converting to RGB only after resize."""
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        arr = np.ascontiguousarray(arr)
        orig_h, orig_w = arr.shape[:2]
        imgsz = int(imgsz)
        gain = min(imgsz / max(orig_h, 1), imgsz / max(orig_w, 1))
        new_w = max(int(round(orig_w * gain)), 1)
        new_h = max(int(round(orig_h * gain)), 1)
        if cv2 is not None:
            resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = np.asarray(Image.fromarray(arr[..., ::-1]).resize((new_w, new_h), Image.BILINEAR))[..., ::-1]
        canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        pad_w = (imgsz - new_w) // 2
        pad_h = (imgsz - new_h) // 2
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        rgb_chw = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1))
        tensor = torch.from_numpy(rgb_chw).float().div_(255.0).unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device, non_blocking=True)
        return tensor, {
            "imgsz": imgsz,
            "gain": float(gain),
            "pad_w": float(pad_w),
            "pad_h": float(pad_h),
            "orig_w": float(orig_w),
            "orig_h": float(orig_h),
        }

    @staticmethod
    def _scale_boxes_from_letterbox(boxes, meta):
        """Map normalized xywh boxes from letterboxed square input back to the original image."""
        if meta is None or boxes.numel() == 0:
            return boxes
        inp = float(meta["imgsz"])
        gain = max(float(meta["gain"]), 1e-9)
        pad_w = float(meta["pad_w"])
        pad_h = float(meta["pad_h"])
        orig_w = max(float(meta["orig_w"]), 1.0)
        orig_h = max(float(meta["orig_h"]), 1.0)

        x, y, w, h = boxes.unbind(-1)
        x1 = (x - w * 0.5) * inp
        y1 = (y - h * 0.5) * inp
        x2 = (x + w * 0.5) * inp
        y2 = (y + h * 0.5) * inp

        x1 = ((x1 - pad_w) / gain).clamp(0, orig_w)
        y1 = ((y1 - pad_h) / gain).clamp(0, orig_h)
        x2 = ((x2 - pad_w) / gain).clamp(0, orig_w)
        y2 = ((y2 - pad_h) / gain).clamp(0, orig_h)

        cx = ((x1 + x2) * 0.5) / orig_w
        cy = ((y1 + y2) * 0.5) / orig_h
        bw = (x2 - x1).clamp_min(0) / orig_w
        bh = (y2 - y1).clamp_min(0) / orig_h
        return torch.stack((cx, cy, bw, bh), -1).clamp(0, 1)

    def _predict_tensor(self, tensor, orig_img, conf=0.25, use_nms=False, letterbox_meta=None, force_list=False):
        device = tensor.device
        self.model.to(device).eval()
        model_dtype = next(self.model.parameters()).dtype
        if not tensor.is_floating_point():
            tensor = tensor.float().div_(255.0)
        if tensor.is_floating_point() and tensor.dtype != model_dtype:
            tensor = tensor.to(dtype=model_dtype)

        with torch.inference_mode():
            has_o2m = bool(getattr(self.model.head, "has_o2m", False))
            if use_nms and has_o2m:
                out_dict = self.model(tensor, return_all=True)
                pred_logits = out_dict["pred_logits_o2m"]
                pred_boxes  = out_dict["pred_boxes_o2m"]
            else:
                out = self.model(tensor)
                pred_logits = out[:, :, :self.nc]
                pred_boxes  = out[:, :, self.nc:]

        if pred_logits.ndim == 2:
            pred_logits = pred_logits.unsqueeze(0)
            pred_boxes = pred_boxes.unsqueeze(0)

        batched_orig = orig_img if isinstance(orig_img, list) else [orig_img]
        batched_meta = letterbox_meta if isinstance(letterbox_meta, list) else [letterbox_meta] * pred_logits.shape[0]
        results = []
        for index in range(pred_logits.shape[0]):
            boxes, scores, classes = self._postprocess_single(
                pred_logits[index],
                pred_boxes[index],
                conf=conf,
                use_nms=use_nms,
                letterbox_meta=batched_meta[index],
            )
            results.append(Results(batched_orig[index], boxes.cpu(), scores.cpu(), classes.cpu(), names=self.names))

        return results if force_list or len(results) != 1 else results[0]

    def _postprocess_single(self, pred_logits, pred_boxes, conf=0.25, use_nms=False, letterbox_meta=None):
        scores, classes = pred_logits.sigmoid().max(-1)

        if use_nms:
            from fotonet.utils.nms import batched_nms
            boxes, scores, classes = batched_nms(pred_boxes, scores, classes, iou_threshold=0.45, score_threshold=conf)
        else:
            mask = scores > conf
            boxes, scores, classes = pred_boxes[mask], scores[mask], classes[mask]
            if len(boxes) > 300:
                _, idx = torch.topk(scores, 300)
                boxes, scores, classes = boxes[idx], scores[idx], classes[idx]

        boxes = self._scale_boxes_from_letterbox(boxes, letterbox_meta)
        return boxes, scores, classes

    def __call__(self, source, **kwargs):
        return self.predict(source, **kwargs)

    def track(self, source, **kwargs):
        return self.predict(source, **kwargs)

    def val(self, data, imgsz=640, batch=8, iou_threshold=0.5):
        from fotonet.metrics.map import compute_coco_metrics
        from torch.utils.data import DataLoader

        cfg      = load_data_cfg(data)
        self.nc  = cfg.get("nc", self.nc)
        val_path = cfg.get("val", cfg.get("train"))
        if isinstance(val_path, list):
            val_path = val_path[0]

        dataset = FOTONETDataset(img_dir=val_path, imgsz=imgsz, augment=False, num_classes=self.nc)
        loader  = DataLoader(dataset, batch_size=batch, shuffle=False,
                             collate_fn=lambda b: (
                                 torch.stack([x[0] for x in b]),
                                 [x[1] for x in b]
                             ))
        self.model.eval()
        nc = self.model.head.nc
        preds_b, preds_s, preds_c = [], [], []
        gts_b,   gts_c            = [], []
        image_ids                 = []

        with torch.no_grad():
            for imgs, targets in loader:
                imgs = imgs.to(self.device)
                out  = self.model(imgs)
                for i, t in enumerate(targets):
                    logits = out[i, :, :nc]
                    boxes  = out[i, :, nc:]
                    scores, classes = logits.sigmoid().max(-1)
                    mask    = scores > 0.001
                    b, s, c = boxes[mask].cpu(), scores[mask].cpu(), classes[mask].cpu()
                    if len(b) > 100:
                        _, idx = torch.topk(s, 100)
                        b, s, c = b[idx], s[idx], c[idx]
                    preds_b.append(b.numpy()  if len(b) > 0 else np.empty((0,4)))
                    preds_s.append(s.numpy()  if len(s) > 0 else np.empty(0))
                    preds_c.append(c.numpy()  if len(c) > 0 else np.empty(0, dtype=np.int64))
                    gts_b.append(t["boxes"].numpy())
                    gts_c.append(t["labels"].numpy())
                    image_ids.append(int(t.get("image_id", torch.tensor([len(image_ids)])).reshape(-1)[0].item()))

        metrics = compute_coco_metrics(
            preds_b, preds_s, preds_c, gts_b, gts_c,
            image_ids=image_ids, num_classes=nc
        )
        return {
            "mAP50": metrics["mAP50"],
            "mAP50_95": metrics["mAP50_95"],
            "mAP50-95": metrics["mAP50_95"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "coco_AR100": metrics["coco_AR100"],
            "metric_backend": metrics["metric_backend"],
            "per_class": metrics["per_class"],
        }

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
        """Permanently remove training-only O2M modules from this loaded model."""
        raw = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        if hasattr(raw, "strip_o2m_for_inference"):
            raw.strip_o2m_for_inference()
        return self

    @staticmethod
    def _infer_arch_from_state_dict(sd, default_use_p2=True, default_reg_max=1, default_quality_head=False):
        """Infer architecture flags from older checkpoints that lack metadata."""
        keys = list(sd.keys())
        use_p2 = any(k.startswith("head.cls_o2o.3.") for k in keys)
        if not use_p2 and not any(k.startswith("head.cls_o2o.2.") for k in keys):
            use_p2 = bool(default_use_p2)
        quality_head = any(k.startswith("head.quality_o2o.") for k in keys)
        if not quality_head:
            quality_head = bool(default_quality_head)

        reg_max = int(default_reg_max)
        for k, v in sd.items():
            if k.startswith("head.reg_o2o.0.6.weight") and torch.is_tensor(v) and v.ndim >= 1:
                reg_max = max(int(v.shape[0]) // 4, 1)
                break
        return bool(use_p2), int(reg_max), bool(quality_head)

    def load(self, model_path):
        ckpt     = torch.load(model_path, map_location="cpu", weights_only=False)
        new_nc   = self.nc
        w, d     = self.w, self.d
        use_p2   = self.use_p2
        reg_max  = self.reg_max
        p3_extra_blocks = self.p3_extra_blocks
        p4_extra_blocks = self.p4_extra_blocks
        p5_extra_blocks = self.p5_extra_blocks
        p5_gate_blocks = self.p5_gate_blocks
        arch_version = self.arch_version
        neck_fusion = self.neck_fusion
        p2_context_blocks = self.p2_context_blocks
        p3_context_blocks = self.p3_context_blocks
        quality_head = self.quality_head
        sd       = ckpt
        stripped_o2m = False
        if isinstance(ckpt, dict) and "model" in ckpt:
            sd     = ckpt["model"]
            new_nc = ckpt.get("nc", self.nc)
            w      = ckpt.get("w",  ckpt.get("cfg", {}).get("width_multiple",  w))
            d      = ckpt.get("d",  ckpt.get("cfg", {}).get("depth_multiple",  d))
            inferred_p2, inferred_reg_max, inferred_quality = self._infer_arch_from_state_dict(sd, use_p2, reg_max, quality_head)
            use_p2 = bool(ckpt.get("use_p2", ckpt.get("p2_head", inferred_p2)))
            reg_max = int(ckpt.get("reg_max", inferred_reg_max))
            p3_extra_blocks = int(ckpt.get("p3_extra_blocks", ckpt.get("cfg", {}).get("p3_extra_blocks", 0)))
            p4_extra_blocks = int(ckpt.get("p4_extra_blocks", ckpt.get("cfg", {}).get("p4_extra_blocks", 0)))
            p5_extra_blocks = int(ckpt.get("p5_extra_blocks", ckpt.get("cfg", {}).get("p5_extra_blocks", 0)))
            p5_gate_blocks = int(ckpt.get("p5_gate_blocks", ckpt.get("cfg", {}).get("p5_gate_blocks", 0)))
            arch_version = int(ckpt.get("arch_version", ckpt.get("cfg", {}).get("arch_version", arch_version)))
            neck_fusion = str(ckpt.get("neck_fusion", ckpt.get("cfg", {}).get("neck_fusion", neck_fusion)))
            p2_context_blocks = int(ckpt.get("p2_context_blocks", ckpt.get("cfg", {}).get("p2_context_blocks", p2_context_blocks)))
            p3_context_blocks = int(ckpt.get("p3_context_blocks", ckpt.get("cfg", {}).get("p3_context_blocks", p3_context_blocks)))
            quality_head = bool(ckpt.get("quality_head", ckpt.get("cfg", {}).get("quality_head", inferred_quality)))
            stripped_o2m = bool(ckpt.get("stripped_o2m", False))
        elif isinstance(sd, dict):
            use_p2, reg_max, quality_head = self._infer_arch_from_state_dict(sd, use_p2, reg_max, quality_head)

        if (new_nc != self.nc or w != self.w or d != self.d or use_p2 != self.use_p2 or
                reg_max != self.reg_max or p3_extra_blocks != self.p3_extra_blocks or
                p4_extra_blocks != self.p4_extra_blocks or p5_extra_blocks != self.p5_extra_blocks or
                p5_gate_blocks != self.p5_gate_blocks or arch_version != self.arch_version or
                neck_fusion != self.neck_fusion or p2_context_blocks != self.p2_context_blocks or
                p3_context_blocks != self.p3_context_blocks or quality_head != self.quality_head):
            print(
                f"[INFO] Re-init model: nc={new_nc}, w={w}, d={d}, "
                f"use_p2={use_p2}, reg_max={reg_max}, "
                f"p3_extra_blocks={p3_extra_blocks}, p4_extra_blocks={p4_extra_blocks}, "
                f"p5_extra_blocks={p5_extra_blocks}, p5_gate_blocks={p5_gate_blocks}, "
                f"arch_version={arch_version}, neck_fusion={neck_fusion}, quality_head={quality_head}"
            )
            self.nc, self.w, self.d = new_nc, w, d
            self.use_p2, self.reg_max = bool(use_p2), int(reg_max)
            self.p3_extra_blocks = max(int(p3_extra_blocks), 0)
            self.p4_extra_blocks = max(int(p4_extra_blocks), 0)
            self.p5_extra_blocks = max(int(p5_extra_blocks), 0)
            self.p5_gate_blocks = max(int(p5_gate_blocks), 0)
            self.arch_version = int(arch_version)
            self.neck_fusion = str(neck_fusion)
            self.p2_context_blocks = max(int(p2_context_blocks), 0)
            self.p3_context_blocks = max(int(p3_context_blocks), 0)
            self.quality_head = bool(quality_head)
            self.model = self._new_model()

        try:
            self.model.load_state_dict(sd, strict=not stripped_o2m)
            if stripped_o2m:
                self.strip_o2m_for_inference()
                print("[INFO] Loaded slim inference checkpoint; O2M heads are not present in memory.")
        except RuntimeError as e:
            for name, scale in scale_fallback_configs().items():
                try:
                    tmp = FOTONETModel(
                        nc=self.nc,
                        w=scale["width_multiple"],
                        d=scale["depth_multiple"],
                        use_p2=bool(scale.get("p2_head", False)),
                        reg_max=int(scale.get("reg_max", self.reg_max)),
                        p3_extra_blocks=int(scale.get("p3_extra_blocks", 0)),
                        p4_extra_blocks=int(scale.get("p4_extra_blocks", 0)),
                        p5_extra_blocks=int(scale.get("p5_extra_blocks", 0)),
                        p5_gate_blocks=int(scale.get("p5_gate_blocks", 0)),
                        arch_version=int(scale.get("arch_version", 1)),
                        neck_fusion=str(scale.get("neck_fusion", "concat")),
                        p2_context_blocks=int(scale.get("p2_context_blocks", 0)),
                        p3_context_blocks=int(scale.get("p3_context_blocks", 0)),
                        quality_head=bool(scale.get("quality_head", False)),
                    ).to(self.device)
                    tmp.load_state_dict(sd, strict=not stripped_o2m)
                    self.model = tmp
                    self._apply_model_cfg(scale, update_nc=False)
                    if stripped_o2m:
                        self.strip_o2m_for_inference()
                    print(f"[INFO] Loaded with scale '{name}'.")
                    break
                except Exception:
                    continue
            else:
                print(f"[WARN] load_state_dict failed: {e}")
                raise e

        self.model.to(self.device)
        self.loaded_weight_path = model_path

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
        raw = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        strip_o2m = bool(inference_only)
        torch.save({
            "model": self._state_dict_for_save(raw, half=half, strip_o2m=strip_o2m),
            "nc":    self.nc,
            "w":     self.w,
            "d":     self.d,
            "use_p2": self.use_p2,
            "reg_max": self.reg_max,
            "p3_extra_blocks": self.p3_extra_blocks,
            "p4_extra_blocks": self.p4_extra_blocks,
            "p5_extra_blocks": self.p5_extra_blocks,
            "p5_gate_blocks": self.p5_gate_blocks,
            "arch_version": self.arch_version,
            "neck_fusion": self.neck_fusion,
            "p2_context_blocks": self.p2_context_blocks,
            "p3_context_blocks": self.p3_context_blocks,
            "quality_head": self.quality_head,
            "inference_only": bool(inference_only),
            "stripped_o2m": strip_o2m,
            "has_o2m": not strip_o2m,
            "nms_free": True,
        }, model_path)
        print(f"[INFO] Saved to '{model_path}'.")

    def export(self, path=None, format="onnx", imgsz=640, batch=1, dynamic=False,
               half=False, simplify=True, int8=False, opset=17, device=None, **kwargs):
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
            **kwargs,
        )["artifact"]
