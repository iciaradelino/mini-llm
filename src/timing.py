"""Per-step timing and exposed-communication measurement.

StepTimer records wall time per phase (data, fwd_bwd, optim) and per step, with
torch.cuda.synchronize() at every boundary on GPU so asynchronous kernels are
attributed to the right phase. The first `warmup_steps` steps are discarded and
the summary reports medians.

    timer = StepTimer(info.device, warmup_steps=20)
    for step in range(steps):
        timer.start_step()
        with timer.phase("data"):
            x, y = get_batch(...)
        with timer.phase("fwd_bwd"):
            loss = model(x, y); loss.backward()
        with timer.phase("optim"):
            opt.step(); opt.zero_grad(set_to_none=True)
        timer.end_step()
    timer.medians()  # {"data": ms, "fwd_bwd": ms, "optim": ms, "step": ms}

Gradient communication is measured separately by measure_exposed_comm(); see
results/README.md ("How communication time is measured") for the method.
"""

import contextlib
import statistics
import time
from collections import defaultdict
from typing import Callable

import torch
import torch.distributed as dist


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class StepTimer:
    def __init__(self, device: torch.device, warmup_steps: int = 20):
        self.device = device
        self.warmup_steps = warmup_steps
        self.step = 0
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._current: dict[str, float] = defaultdict(float)
        self._step_t0 = 0.0

    def start_step(self) -> None:
        _sync(self.device)
        self._current.clear()
        self._step_t0 = time.perf_counter()

    @contextlib.contextmanager
    def phase(self, name: str):
        _sync(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync(self.device)
            self._current[name] += (time.perf_counter() - t0) * 1000.0

    def end_step(self) -> None:
        _sync(self.device)
        step_ms = (time.perf_counter() - self._step_t0) * 1000.0
        if self.step >= self.warmup_steps:
            for name, ms in self._current.items():
                self._samples[name].append(ms)
            self._samples["step"].append(step_ms)
        self.step += 1

    @property
    def measured_steps(self) -> int:
        return len(self._samples["step"])

    def medians(self) -> dict[str, float]:
        """Median ms per phase over the post-warm-up steps (empty if none were measured)."""
        return {name: statistics.median(v) for name, v in self._samples.items() if v}


def measure_exposed_comm(
    model: torch.nn.Module,
    fwd_bwd: Callable[[], None],
    device: torch.device,
    n_steps: int = 20,
    warmup: int = 3,
) -> dict[str, float]:
    """Estimate the gradient communication DDP does not hide behind backward.

    Runs `fwd_bwd` (forward + loss.backward() on a fresh batch) alternately with
    normal DDP gradient sync and under model.no_sync(), and returns

        fwd_bwd_sync_ms    median fwd+bwd with all-reduce (overlapped)
        fwd_bwd_nosync_ms  median fwd+bwd without any all-reduce
        comm_ms            max(0, sync - nosync): exposed communication time

    No optimizer step is taken and gradients are zeroed after every step, so the
    model weights are unchanged: call it after training (or after the step-500
    evaluation) so it cannot influence the loss numbers.

    Returns zeros for a non-DDP model.
    """
    if not hasattr(model, "no_sync") or not dist.is_initialized():
        return {"fwd_bwd_sync_ms": 0.0, "fwd_bwd_nosync_ms": 0.0, "comm_ms": 0.0}

    samples: dict[bool, list[float]] = {True: [], False: []}
    for i in range(warmup + n_steps):
        # Alternate modes so drift (clock boost, thermal, other jobs) hits both equally.
        for sync_grads in (True, False):
            dist.barrier()  # align ranks so no_sync times don't include waiting skew
            _sync(device)
            t0 = time.perf_counter()
            ctx = contextlib.nullcontext() if sync_grads else model.no_sync()
            with ctx:
                fwd_bwd()
            _sync(device)
            ms = (time.perf_counter() - t0) * 1000.0
            model.zero_grad(set_to_none=True)
            if i >= warmup:
                samples[sync_grads].append(ms)

    sync_ms = statistics.median(samples[True])
    nosync_ms = statistics.median(samples[False])
    return {
        "fwd_bwd_sync_ms": sync_ms,
        "fwd_bwd_nosync_ms": nosync_ms,
        "comm_ms": max(0.0, sync_ms - nosync_ms),
    }
