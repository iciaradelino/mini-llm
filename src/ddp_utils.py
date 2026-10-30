"""DDP setup helpers.

Works both under torchrun (multi-process DDP) and plain `python` (single process).
torchrun sets RANK / LOCAL_RANK / WORLD_SIZE; if RANK is absent we run without a
process group and every helper degrades to a no-op.

Typical use in train.py:

    info = ddp_utils.setup()                       # picks nccl on GPU, gloo on CPU
    batch = ddp_utils.per_rank_batch(64, info.world_size)
    model = ddp_utils.wrap(model.to(info.device), info)
    ...
    if info.is_master: runlog.append_row(...)
    ddp_utils.cleanup(info)
"""

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class DistInfo:
    enabled: bool          # True if a process group was initialized
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str | None    # "nccl", "gloo" or None

    @property
    def is_master(self) -> bool:
        return self.rank == 0


def setup(device: str | None = None, backend: str | None = None) -> DistInfo:
    """Initialize the process group (if launched by torchrun) and pick the device.

    device:  "cuda" / "cpu"; default is cuda when available.
    backend: "nccl" / "gloo"; default is nccl for cuda, gloo for cpu.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if "RANK" not in os.environ:
        dev = torch.device("cuda:0" if device == "cuda" else "cpu")
        if device == "cuda":
            torch.cuda.set_device(dev)
        return DistInfo(False, 0, 0, 1, dev, None)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if backend is None:
        backend = "nccl" if device == "cuda" else "gloo"

    if device == "cuda":
        # One GPU per process; LOCAL_RANK indexes GPUs on this node.
        dev = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(dev)
    else:
        dev = torch.device("cpu")

    dist.init_process_group(backend=backend)
    return DistInfo(True, rank, local_rank, world_size, dev, backend)


def cleanup(info: DistInfo) -> None:
    if info.enabled:
        dist.destroy_process_group()


def barrier(info: DistInfo) -> None:
    if info.enabled:
        dist.barrier()


def per_rank_batch(global_batch: int, world_size: int) -> int:
    """Global batch stays fixed; each rank gets global_batch / world_size sequences."""
    if global_batch % world_size != 0:
        raise ValueError(f"global_batch={global_batch} not divisible by world_size={world_size}")
    return global_batch // world_size


def wrap(model: torch.nn.Module, info: DistInfo) -> torch.nn.Module:
    """Wrap in DDP when running distributed; otherwise return the model unchanged."""
    if not info.enabled:
        return model
    if info.device.type == "cuda":
        return DDP(model, device_ids=[info.local_rank], output_device=info.local_rank)
    return DDP(model)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """The underlying module (for saving, parameter counts, generation)."""
    return model.module if isinstance(model, DDP) else model


def seed_everything(seed: int) -> None:
    """Same seed on every rank: identical model init (DDP also broadcasts rank 0's weights)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def data_rng(seed: int, info: DistInfo) -> np.random.Generator:
    """Per-rank RNG for sampling batch offsets, so ranks don't train on identical batches."""
    return np.random.default_rng(seed + 1000 * info.rank)


def all_reduce_mean(value: float, info: DistInfo) -> float:
    """Average a scalar (e.g. validation loss) across ranks."""
    if not info.enabled:
        return value
    t = torch.tensor([value], dtype=torch.float64, device=info.device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() / info.world_size
