"""Exact V1 data-loader lifecycle and dataset mutation protocol."""

import os
import torch
from torch.utils.data import DataLoader, Subset
from fotonet.engine.runtime import (
    ThreadPrefetcher as _ThreadPrefetcher,
    configure_data_worker,
)


class DataPipelineMixin:
    def _set_dataset_imgsz(self, dataset, imgsz):
        if dataset is None:
            return
        if hasattr(dataset, "set_imgsz"):
            dataset.set_imgsz(imgsz)
        elif hasattr(dataset, "dataset"):
            self._set_dataset_imgsz(dataset.dataset, imgsz)

    def _make_train_loader(self, train_sampler, train_shuffle, num_workers, train_pf_factor):
        common = {
            "dataset": self.full_train_set,
            "collate_fn": self._collate_fn,
            "num_workers": num_workers,
            "pin_memory": self.pin_memory,
            "prefetch_factor": train_pf_factor,
            "persistent_workers": True if num_workers > 0 else False,
            "worker_init_fn": configure_data_worker if num_workers > 0 else None,
        }
        return DataLoader(
            batch_size=self.batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            **common,
        )

    def _rebuild_train_prefetcher(self, train_sampler, train_shuffle, num_workers, train_pf_factor):
        if self.train_prefetcher is not None:
            self.train_prefetcher.shutdown()
        self.train_loader = self._make_train_loader(train_sampler, train_shuffle, num_workers, train_pf_factor)
        self.train_prefetcher = _ThreadPrefetcher(
            self.train_loader,
            device=self.device,
            queue_size=2,
        )
        return self.train_prefetcher

    def _train_augment_hyp(self):
        dataset = self.full_train_set
        seen = set()
        while dataset is not None and id(dataset) not in seen:
            seen.add(id(dataset))
            hyp = getattr(dataset, "augment_hyp", None)
            if isinstance(hyp, dict):
                return hyp
            dataset = getattr(dataset, "dataset", None)
        return {}

    def _set_train_augmentation_pass(self, pass_index):
        if self.augmentation_passes is None:
            return False
        index = int(pass_index)
        if not 0 <= index < len(self.augmentation_passes):
            raise IndexError(f"augmentation pass index {index} is outside the configured pass schedule")
        dataset = self.full_train_set
        seen = set()
        while dataset is not None and id(dataset) not in seen:
            seen.add(id(dataset))
            if hasattr(dataset, "augment_hyp"):
                dataset.augment = True
                dataset.augment_hyp = dict(self.augmentation_passes[index])
                self._active_augmentation_pass = index
                return True
            dataset = getattr(dataset, "dataset", None)
        raise TypeError("augmentation_passes requires a mutable fotonet training dataset")

    def _close_mosaic_boundary(self):
        if self.augmentation_passes is not None:
            return None
        if getattr(self, "infinite_epochs", False):
            return None
        try:
            close_mosaic = int(self._train_augment_hyp().get("close_mosaic", 0) or 0)
        except (TypeError, ValueError):
            close_mosaic = 0
        if close_mosaic <= 0:
            return None
        if int(self.epochs) <= close_mosaic:
            return None
        return max(int(self.epochs) - close_mosaic, 0)

    def _init_val_subset(self, val_dataset):
        n_val = len(val_dataset)
        if self.val_subset_size <= 0 or self.val_subset_size >= n_val:
            self._val_subset_indices = None
            return
        generator = torch.Generator()
        generator.manual_seed(2026)
        self._val_subset_indices = torch.randperm(n_val, generator=generator)[:self.val_subset_size].tolist()

    def _val_dataset_for_epoch(self, epoch):
        is_last = (not getattr(self, "infinite_epochs", False)) and int(epoch) == self.epochs - 1
        progress = 0.0 if getattr(self, "infinite_epochs", False) else (int(epoch) + 1) / max(int(self.epochs), 1)
        force_full_last = is_last and self.full_val_after <= 1.0
        use_full = force_full_last or self._val_subset_indices is None or progress >= self.full_val_after
        if use_full:
            return self.full_val_dataset, "full"
        return Subset(self.full_val_dataset, self._val_subset_indices), f"subset{len(self._val_subset_indices)}"

    @staticmethod
    def _assert_disjoint_train_val_images(train_dataset, val_dataset):
        """Reject validation leakage before any optimization can begin."""
        train_files = getattr(train_dataset, "img_files", None)
        val_files = getattr(val_dataset, "img_files", None)
        if train_files is None or val_files is None:
            raise RuntimeError(
                "Cannot verify configured train/val independence because a dataset does not expose img_files. "
                "Use FOTO-NET's built-in dataset sources or an explicit disjoint dynamic val_split."
            )

        def canonical_paths(paths):
            return {
                os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(path))))
                for path in paths
            }

        overlap = canonical_paths(train_files) & canonical_paths(val_files)
        if overlap:
            examples = ", ".join(sorted(overlap)[:3])
            suffix = "" if len(overlap) == 1 else "s"
            raise ValueError(
                f"Configured train and val sources overlap on {len(overlap)} image{suffix} ({examples}). "
                "Use disjoint images so validation AP is not leaked or optimistic."
            )

    @staticmethod
    def _collate_fn(batch):
        """Simple collate: stack images directly (all are same size from dataset)."""
        imgs, targets = zip(*batch)
        if isinstance(imgs[0], dict):
            common_keys = set(imgs[0]).intersection(*(set(item) for item in imgs[1:]))
            return {key: torch.stack([item[key] for item in imgs], 0) for key in common_keys}, list(targets)
        return torch.stack(imgs, 0), list(targets)


__all__ = ["DataPipelineMixin"]
