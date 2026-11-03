"""Append one row per run to results/runs.csv.

COLUMNS must match the schema in results/README.md exactly. append_row() refuses
rows with missing or unknown keys and refuses to write to a CSV whose header
differs, so schema drift fails loudly instead of producing a misaligned file.
Pass None (or "") for columns that legitimately have no value.

Only rank 0 should call append_row(); the caller checks info.is_master.
"""

import csv
import os
import sys
from datetime import datetime, timezone

import torch

COLUMNS = [
    "run_id", "timestamp", "job", "experiment", "slurm_job_id", "nodes", "device",
    "gpu_model", "world_size", "num_threads", "n_layer", "n_embd", "n_head", "n_params",
    "seq_len", "global_batch", "precision", "seed", "steps", "warmup_steps",
    "nccl_p2p_disable", "t_data_ms", "t_fwd_bwd_ms", "t_optim_ms", "t_comm_ms",
    "t_step_ms", "tokens_per_s", "peak_mem_mb", "val_loss_500", "final_val_loss",
    "valid", "status", "notes",
]

DEFAULT_PATH = os.path.join("results", "runs.csv")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_run_id(index: int = 0) -> str:
    """`<slurm_job_id>_<index>`; outside Slurm, `local-<YYYYmmddHHMMSS>_<index>`."""
    job = os.environ.get("SLURM_JOB_ID") or "local-" + datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{job}_{index}"


def slurm_fields() -> dict:
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "nodes": os.environ.get("SLURM_JOB_NODELIST", ""),
        "nccl_p2p_disable": int(os.environ.get("NCCL_P2P_DISABLE", "0") or 0),
    }


def device_fields(device: torch.device) -> dict:
    if device.type == "cuda":
        return {"device": "cuda", "gpu_model": torch.cuda.get_device_name(device),
                "num_threads": torch.get_num_threads()}
    return {"device": "cpu", "gpu_model": "", "num_threads": torch.get_num_threads()}


def peak_mem_mb(device: torch.device) -> float | None:
    """Peak GPU memory allocated by PyTorch, or peak process RSS on CPU."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    try:
        import resource  # Unix only (the cluster); absent on Windows
    except ImportError:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 1024  # bytes on macOS, KiB on Linux


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return v


def append_row(row: dict, path: str = DEFAULT_PATH) -> None:
    missing = [c for c in COLUMNS if c not in row]
    extra = [k for k in row if k not in COLUMNS]
    if missing or extra:
        raise ValueError(f"runs.csv row mismatch: missing={missing} extra={extra}")

    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    if not write_header:
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != COLUMNS:
            raise ValueError(f"{path} header does not match runlog.COLUMNS; "
                             "update both together (see docs/interfaces.md)")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow({k: _fmt(row[k]) for k in COLUMNS})
