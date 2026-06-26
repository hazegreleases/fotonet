"""Training runtime helpers."""
import copy
import math
import queue
import threading

import torch
from torch.utils.data import Sampler


class ThreadPrefetcher:
    """
    Thread-based data prefetcher. Wraps a DataLoader and pre-fetches next batch
    in a daemon thread while GPU trains current one. Safe on Windows.
    """
    def __init__(self, loader, device, queue_size=2):
        self.loader = loader
        self.device = device
        self.queue_size = queue_size
        self._queue = None
        self._thread = None
        self._stop_event = threading.Event()

    def __iter__(self):
        self._queue = queue.Queue(maxsize=self.queue_size)
        self._stop_event.clear()
        stop_token = object()

        def _worker():
            try:
                for batch in self.loader:
                    if self._stop_event.is_set():
                        break
                    imgs, targets = batch
                    imgs_gpu = imgs.to(self.device, non_blocking=True)
                    tgts_gpu = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]
                    self._queue.put((imgs_gpu, tgts_gpu))
            finally:
                self._queue.put(stop_token)

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

        while True:
            item = self._queue.get()
            if item is stop_token:
                break
            yield item

    def shutdown(self):
        self._stop_event.set()
        if self._queue is not None:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def __len__(self):
        return len(self.loader)


class EpochCutSampler(Sampler):
    """
    Samples one deterministic chunk per epoch without recreating DataLoader.
    Useful for quick bounded smoke runs over large datasets.
    """
    def __init__(self, data_source, epoch_cut=1, seed=42):
        self.data_source = data_source
        self.epoch_cut = max(1, int(epoch_cut))
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return int(math.ceil(len(self.data_source) / self.epoch_cut))

    def __iter__(self):
        n = len(self.data_source)
        if n <= 0:
            return iter(())
        chunk_size = len(self)
        cycle_start_epoch = (self.epoch // self.epoch_cut) * self.epoch_cut
        cycle_idx = self.epoch % self.epoch_cut
        generator = torch.Generator()
        generator.manual_seed(self.seed + cycle_start_epoch)
        perm = torch.randperm(n, generator=generator).tolist()
        start = cycle_idx * chunk_size
        end = min(start + chunk_size, n)
        if start >= n:
            return iter(())
        return iter(perm[start:end])


class EMA:
    """Exponential Moving Average of model weights with ramping decay."""
    def __init__(self, model, decay_start=0.99, decay_end=0.9999, total_steps=None):
        self.original_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        self.ema = copy.deepcopy(self.original_model).eval()
        self.decay_start = decay_start
        self.decay_end = decay_end
        self.total_steps = total_steps or 100000
        self.step_count = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            t = min(self.step_count / max(self.total_steps, 1), 1.0)
            decay = self.decay_start + (self.decay_end - self.decay_start) * t
            self.step_count += 1

            current_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            model_state = current_model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.copy_(decay * v + (1.0 - decay) * model_state[k].detach())
