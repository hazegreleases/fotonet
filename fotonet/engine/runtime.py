
"""Training runtime helpers."""
import copy
import math
import queue
import threading

import torch
from fotonet.utils.matcher import validate_target_mapping


def configure_data_worker(_worker_id=None):
    """Keep spawned loader workers from oversubscribing the host CPU.

    OpenCV defaults to all logical CPUs in every Windows worker.  Four workers
    on the current 4-core host therefore launched up to 32 OpenCV threads.
    OpenCV's image operations are row independent, so limiting each process to
    one thread preserves pixels while allowing DataLoader process parallelism
    to provide the outer level of concurrency.
    """

    try:
        import cv2

        cv2.setNumThreads(1)
    except (ImportError, AttributeError):
        pass
    # PyTorch normally applies this inside DataLoader workers itself.  Keep the
    # invariant explicit for direct/custom worker launchers.
    torch.set_num_threads(1)


class ThreadPrefetcher:
    """Prefetch host batches and asynchronously stage CUDA batches safely.

    Loader exceptions are delivered to the training thread instead of being
    converted into a normal end-of-epoch sentinel.  CUDA work is submitted by
    the consumer thread on a dedicated stream; invoking CUDA from the Python
    loader thread made overlap unreliable and could silently consume extra VRAM
    on the default stream.
    """

    def __init__(
        self,
        loader,
        device,
        queue_size=2,
        move_targets=True,
    ):
        self.loader = loader
        self.device = torch.device(device)
        self.queue_size = max(int(queue_size), 1)
        self.move_targets = bool(move_targets)
        self._queue = None
        self._thread = None
        self._stop_event = threading.Event()
        self._worker_error = None
        self._cuda_stream = None

    @staticmethod
    def _move_target(target, device):
        # Validate on the CPU side before asynchronous H2D staging.  The
        # matcher sees the marker and avoids a per-image CUDA scalar read in
        # its hot path; direct matcher users remain validated there.
        if "boxes" in target and "labels" in target:
            validate_target_mapping(target)
        moved = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in target.items()
        }
        if "boxes" in target and "labels" in target:
            moved["_fotonet_target_validated"] = True
        return moved

    def _stage_batch(self, batch):
        imgs, targets = batch
        def move_inputs(value):
            if torch.is_tensor(value):
                return value.to(self.device, non_blocking=True)
            if isinstance(value, dict):
                return {key: move_inputs(item) for key, item in value.items()}
            return value
        if self.device.type != "cuda":
            return (
                move_inputs(imgs),
                (
                    [self._move_target(target, self.device) for target in targets]
                    if self.move_targets
                    else targets
                ),
                None,
            )

        with torch.cuda.stream(self._cuda_stream):
            staged_imgs = move_inputs(imgs)
            staged_targets = (
                [self._move_target(target, self.device) for target in targets]
                if self.move_targets
                else targets
            )
            ready = torch.cuda.Event()
            ready.record(self._cuda_stream)
        return staged_imgs, staged_targets, ready

    def _wait_until_ready(self, staged):
        if len(staged) == 3:
            imgs, targets, ready = staged
            started = None
        else:
            imgs, targets, ready, started = staged
        if ready is not None:
            torch.cuda.current_stream(self.device).wait_event(ready)
        return imgs, targets

    def __iter__(self):
        # A caller may abandon a previous iterator early.
        self.shutdown()
        self._queue = queue.Queue(maxsize=self.queue_size)
        self._stop_event.clear()
        self._worker_error = None
        stop_token = object()
        error_token = object()
        not_ready = object()

        def _put(item):
            while not self._stop_event.is_set():
                try:
                    self._queue.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def _worker():
            try:
                for batch in self.loader:
                    if self._stop_event.is_set():
                        break
                    if not _put(batch):
                        return
            except BaseException as exc:  # Preserve DataLoader worker failures too.
                self._worker_error = exc
                _put((error_token, exc))
            finally:
                _put(stop_token)

        self._thread = threading.Thread(target=_worker, daemon=True, name="fotonet-prefetch")
        self._thread.start()
        if self.device.type == "cuda":
            self._cuda_stream = torch.cuda.Stream(device=self.device)

        def _next_raw(*, wait=True):
            while True:
                try:
                    item = self._queue.get(timeout=0.1) if wait else self._queue.get_nowait()
                except queue.Empty:
                    if self._worker_error is not None and (self._thread is None or not self._thread.is_alive()):
                        raise RuntimeError("Data prefetch worker failed") from self._worker_error
                    if not wait:
                        return not_ready
                    continue
                if item is stop_token:
                    if self._worker_error is not None:
                        raise RuntimeError("Data prefetch worker failed") from self._worker_error
                    return None
                if isinstance(item, tuple) and len(item) == 2 and item[0] is error_token:
                    raise RuntimeError("Data prefetch worker failed") from item[1]
                return item

        try:
            raw_current = _next_raw()
            if raw_current is None:
                return
            staged_current = self._stage_batch(raw_current)

            while staged_current is not None:
                # Start next-batch H2D early when CPU loading already won the
                # race, but never wait for a slow *next* decode before
                # yielding a ready current batch. The worker keeps filling the
                # bounded CPU queue while the caller computes. CUDA staging
                # remains exclusively on this consumer thread and its
                # dedicated stream.
                deferred_error = None
                try:
                    raw_next = _next_raw(wait=False)
                except RuntimeError as exc:
                    raw_next = None
                    deferred_error = exc
                if raw_next is not_ready:
                    staged_next = not_ready
                elif raw_next is None:
                    staged_next = None
                else:
                    staged_next = self._stage_batch(raw_next)

                yield self._wait_until_ready(staged_current)
                if deferred_error is not None:
                    raise deferred_error
                if staged_next is None:
                    break
                if staged_next is not_ready:
                    raw_next = _next_raw(wait=True)
                    if raw_next is None:
                        break
                    staged_current = self._stage_batch(raw_next)
                else:
                    staged_current = staged_next
        finally:
            self.shutdown()

    def shutdown(self):
        self._stop_event.set()
        if self._queue is not None:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._cuda_stream = None

    def __len__(self):
        return len(self.loader)


class EMA:
    """Exponential Moving Average with a zero-to-target decay warmup.

    Starting at a high decay leaves validation evaluating an almost-random
    initialization for the first part of a run.  The exponential warmup copies
    early optimizer updates nearly verbatim and asymptotically approaches the
    requested long-horizon decay.
    """

    def __init__(
        self,
        model,
        decay_start=0.0,
        decay_end=0.9999,
        total_steps=None,
        warmup_steps=None,
    ):
        self.original_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        self.ema = copy.deepcopy(self.original_model).eval()
        self.decay_start = float(decay_start)
        self.decay_end = float(decay_end)
        self.total_steps = total_steps or 100000
        self.warmup_steps = max(int(warmup_steps or 2000), 1)
        self.step_count = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def current_decay(self, step_count=None, progress=None):
        """Return the decay used for the next/selected EMA update."""
        if progress is None:
            step = self.step_count if step_count is None else int(step_count)
        else:
            progress = min(max(float(progress), 0.0), 1.0)
            step = progress * max(int(self.total_steps), 1)
        ramp = 1.0 - math.exp(-max(step, 0) / float(self.warmup_steps))
        return self.decay_start + (self.decay_end - self.decay_start) * ramp

    def update(self, model, progress=None):
        with torch.no_grad():
            decay = self.current_decay(self.step_count + 1, progress=progress)
            self.step_count += 1

            current_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            model_state = current_model.state_dict()
            floating_groups = {}
            for key, ema_value in self.ema.state_dict().items():
                source = model_state[key].detach()
                if ema_value.dtype.is_floating_point:
                    group_key = (ema_value.device, ema_value.dtype)
                    ema_values, source_values = floating_groups.setdefault(
                        group_key, ([], [])
                    )
                    ema_values.append(ema_value)
                    source_values.append(source)
                else:
                    ema_value.copy_(source)
            # One foreach launch per device/dtype replaces a Python-launched
            # kernel for every floating parameter and buffer. State names,
            # values, decay schedule, and checkpoint representation are
            # unchanged.
            update_weight = 1.0 - decay
            for ema_values, source_values in floating_groups.values():
                torch._foreach_lerp_(ema_values, source_values, update_weight)
