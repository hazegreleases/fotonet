"""Exact V1 recipe normalization and resolution-phase semantics."""

class RecipeProtocolMixin:
    @staticmethod
    def _normalize_imgsz_schedule(schedule, default_imgsz):
        if not schedule:
            return [(1.0, int(default_imgsz))]
        pairs = []
        for item in schedule:
            if isinstance(item, dict):
                frac = item.get("fraction", item.get("until", item.get("pct")))
                size = item.get("imgsz", item.get("size"))
            else:
                frac, size = item
            pairs.append((float(frac), int(size)))
        total = sum(frac for frac, _ in pairs)
        if total <= 1.0001:
            out = []
            acc = 0.0
            for frac, size in pairs:
                acc += frac
                out.append((min(acc, 1.0), size))
            out[-1] = (1.0, out[-1][1])
            return out
        out = [(min(max(frac, 0.0), 1.0), size) for frac, size in pairs]
        out = sorted(out, key=lambda x: x[0])
        if out[-1][0] < 1.0:
            out.append((1.0, out[-1][1]))
        return out

    @staticmethod
    def _model_max_stride(model):
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        head = getattr(raw, "head", None)
        strides = getattr(head, "strides", None)
        if not strides:
            raise ValueError("Trainer model must expose a non-empty head.strides sequence")
        try:
            max_stride = max(int(stride) for stride in strides)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Model head.strides is invalid: {strides!r}") from exc
        if max_stride <= 0:
            raise ValueError(f"Model head.strides must be positive, got {strides!r}")
        return max_stride

    @staticmethod
    def _validate_imgsz_alignment(imgsz, max_stride, setting):
        size = int(imgsz)
        if size <= 0:
            raise ValueError(f"{setting} must be a positive integer, got {imgsz!r}")
        if size % int(max_stride) != 0:
            raise ValueError(
                f"{setting}={size} is not divisible by model max stride {max_stride}. "
                "Training does not pad inputs like inference, so use a stride-aligned size "
                "to keep the train/eval anchor graph identical."
            )

    @staticmethod
    def _normalize_lr_scheduler(name):
        raw = str(name or "Cosine").strip().lower().replace("_", "").replace("-", "")
        if raw in {"cosine", "cos"}:
            return "Cosine"
        if raw in {"lrdropdown", "dropdown", "plateau", "reducelronplateau"}:
            return "LRDropDown"
        raise ValueError("lr_scheduler must be 'Cosine' or 'LRDropDown'")

    @staticmethod
    def _normalize_best_metric(name):
        raw = str(name or "mAP50_95").strip().lower().replace("-", "").replace("_", "")
        if raw in {"map5095", "cocomap", "map"}:
            return "mAP50_95"
        if raw in {"map50", "ap50"}:
            return "mAP50"
        raise ValueError("best_metric must be 'mAP50_95' or 'mAP50'")

    def _epoch_count_for_progress(self):
        if not getattr(self, "infinite_epochs", False):
            return max(int(self.epochs), 1)
        return max(int(getattr(self, "start_epoch", 0)) + 500, 500)

    def _epoch_label(self):
        return "inf" if getattr(self, "infinite_epochs", False) else str(int(self.epochs))

    def _imgsz_for_epoch(self, epoch):
        progress = (int(epoch) + 1) / self._epoch_count_for_progress()
        for limit, size in self.imgsz_schedule:
            if progress <= limit + 1e-9:
                return int(size)
        return int(self.imgsz_schedule[-1][1])

    def _imgsz_phase_for_epoch(self, epoch):
        progress = (int(epoch) + 1) / self._epoch_count_for_progress()
        for index, (limit, _size) in enumerate(self.imgsz_schedule, start=1):
            if progress <= limit + 1e-9:
                return index, len(self.imgsz_schedule)
        return len(self.imgsz_schedule), len(self.imgsz_schedule)


__all__ = ["RecipeProtocolMixin"]
