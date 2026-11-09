#!/usr/bin/env python3
"""
src/train.py  (steps 6a + 6b)

GPT model and training loop adapted from nanoGPT (Karpathy, MIT licence).
Runs single-process with `python` or data-parallel (DDP) under `torchrun`
(nccl on GPU, gloo on CPU; see src/ddp_utils.py).

Baseline (README): 6 layers, width 384, 6 heads, sequence length 256,
global batch 64 sequences, vocab 8192 (from data/tokenized/meta.json).
Under DDP the global batch stays fixed: each rank processes
global_batch / world_size sequences per step.

Data: flat uint16 streams read with np.memmap (data/tokenized/{train,val,tiny}.bin).
If the files are missing, or --dummy_data is given, random tokens are used so the
loop can be tested before the real data exists.

Reproducibility: one --seed drives model init and training batches (rank r samples
with seed + 1000*r, so ranks see different windows); validation batches come from
a separate fixed-seed generator and are sharded over ranks, so every run evaluates
exactly the same windows whatever the world size.

Validation loss is evaluated every --eval_interval steps and always at step 500
(`val_loss_500`, used by the 2% correctness rule) and at the final step.

Timing (src/timing.py): every step is split into data / fwd_bwd / optim with
torch.cuda.synchronize() at phase boundaries; the first --timing_warmup steps are
dropped. Exposed gradient communication is measured after training with the
no_sync() comparison (results/README.md). Rank 0 appends one row to --runs_csv.

Usage
    python src/train.py --dataset tiny --max_steps 50 --eval_interval 10
    python src/train.py --dummy_data --max_steps 20 --device cpu
    python src/train.py --precision bf16 --max_steps 1000 --job A --experiment baseline_fp32
    torchrun --standalone --nproc_per_node=2 src/train.py --device cpu --dataset tiny
"""

import argparse
import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ddp_utils
import runlog
from timing import StepTimer, measure_exposed_comm

BASE_DIR = Path(__file__).resolve().parent.parent
LOSS_CHECK_STEP = 500  # step whose validation loss feeds the correctness rule


# --------------------------------------------------------------------------- model
@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 8192
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd, bias=cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        self.apply(self._init_weights)
        for name, p in self.named_parameters():  # GPT-2 scaled init for residual projections
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())  # tied weights are counted once
        return n - self.transformer.wpe.weight.numel() if non_embedding else n

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.cfg.block_size, f"sequence {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

    def configure_optimizer(self, weight_decay, lr, betas, device_type):
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ]
        fused = device_type == "cuda"
        return torch.optim.AdamW(groups, lr=lr, betas=betas, **({"fused": True} if fused else {}))


# --------------------------------------------------------------------------- data
class BatchLoader:
    """Random windows from a memmapped uint16 stream (or random tokens if no file)."""

    def __init__(self, path, vocab_size, seq_len, seed, device):
        self.seq_len, self.vocab_size, self.device = seq_len, vocab_size, device
        self.gen = torch.Generator().manual_seed(seed)
        self.path = path
        self.dummy = path is None
        if not self.dummy:
            self.data = np.memmap(path, dtype=np.uint16, mode="r")
            assert len(self.data) > seq_len + 1, f"{path} has too few tokens"
            assert int(self.data.max()) < vocab_size, f"{path} has ids >= vocab_size"
            self.n_tokens = len(self.data)
        else:
            self.n_tokens = 10_000_000

    def get_batch(self, batch_size):
        T = self.seq_len
        if self.dummy:
            x = torch.randint(0, self.vocab_size, (batch_size, T), generator=self.gen)
            y = torch.randint(0, self.vocab_size, (batch_size, T), generator=self.gen)
        else:
            ix = torch.randint(self.n_tokens - T - 1, (batch_size,), generator=self.gen).tolist()
            x = torch.from_numpy(np.stack([self.data[i:i + T].astype(np.int64) for i in ix]))
            y = torch.from_numpy(np.stack([self.data[i + 1:i + 1 + T].astype(np.int64) for i in ix]))
        if torch.device(self.device).type == "cuda":
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)


def resolve_split(data_dir, name):
    p = Path(data_dir) / f"{name}.bin"
    return str(p) if p.exists() else None


# --------------------------------------------------------------------------- loop
def get_lr(step, a):
    if step < a.warmup_steps:
        return a.lr * (step + 1) / a.warmup_steps
    if step >= a.max_steps:
        return a.min_lr
    ratio = (step - a.warmup_steps) / max(1, a.max_steps - a.warmup_steps)
    return a.min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (a.lr - a.min_lr)


@torch.no_grad()
def estimate_val_loss(model, a, val_seed, ctx, info):
    """Mean loss over a FIXED set of validation windows (same for every run).

    Every rank draws the same eval_iters batches; rank r evaluates batches
    r, r + world_size, ... and the per-batch losses are summed across ranks, so the
    result does not depend on the world size. `model` is the unwrapped module.
    """
    loader = BatchLoader(a.val_path, a.vocab_size, a.seq_len, val_seed, a.device)
    model.eval()
    losses = torch.zeros(a.eval_iters, dtype=torch.float64, device=info.device)
    for k in range(a.eval_iters):
        x, y = loader.get_batch(a.eval_batch)  # always drawn, to keep the generator in step
        if k % info.world_size != info.rank:
            continue
        with ctx:
            _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    if info.enabled:
        torch.distributed.all_reduce(losses, op=torch.distributed.ReduceOp.SUM)
    return losses.mean().item()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    p.add_argument("--data_dir", default=str(BASE_DIR / "data" / "tokenized"))
    p.add_argument("--dataset", default="train", choices=["train", "tiny"], help="training stream")
    p.add_argument("--dummy_data", action="store_true", help="random tokens instead of .bin files")
    p.add_argument("--vocab_size", type=int, default=None, help="default: from meta.json, else 8192")
    # model
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    # optimisation
    p.add_argument("--global_batch", type=int, default=64, help="sequences per optimizer step")
    p.add_argument("--micro_batch", type=int, default=None,
                   help="sequences per forward pass per rank (gradient accumulation); "
                        "default = global_batch / world_size")
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min_lr", type=float, default=6e-5)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # run
    p.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num_threads", type=int, default=None, help="torch CPU threads")
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--eval_iters", type=int, default=20, help="validation batches per evaluation")
    p.add_argument("--eval_batch", type=int, default=None,
                   help="sequences per validation batch (independent of world size); default = global_batch")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--summary_json", default=None, help="write the final summary dict here")
    # instrumentation / results (step 6b)
    p.add_argument("--timing_warmup", type=int, default=20, help="steps excluded from the timing medians")
    p.add_argument("--comm_measure_steps", type=int, default=20,
                   help="extra fwd+bwd pairs for the no_sync() comm measurement (DDP only; 0 = skip)")
    p.add_argument("--runs_csv", default=str(BASE_DIR / "results" / "runs.csv"),
                   help="rank 0 appends one row here; 'none' disables")
    p.add_argument("--job", default="dev", help="runs.csv `job`: A, B, C, D, E, opt, final, smoke, dev")
    p.add_argument("--experiment", default=None, help="runs.csv `experiment` label (default: derived)")
    p.add_argument("--notes", default="", help="free text for runs.csv `notes`")
    return p.parse_args(argv)


def build_row(a, info, n_params, med, comm, val_loss_500, last_val, status, notes):
    """One runs.csv row (schema: results/README.md). `valid` is filled in later by
    src/check_correctness.py against the fp32 single-GPU baseline."""
    step_ms = med.get("step")
    return {
        "run_id": runlog.next_run_id(a.runs_csv),
        "timestamp": runlog.now_iso(),
        "job": a.job,
        "experiment": a.experiment or f"{info.device.type}_ws{info.world_size}_{a.precision}",
        **runlog.slurm_fields(),
        **runlog.device_fields(info.device),
        "world_size": info.world_size,
        "n_layer": a.n_layer, "n_embd": a.n_embd, "n_head": a.n_head, "n_params": n_params,
        "seq_len": a.seq_len, "global_batch": a.global_batch,
        "precision": a.precision, "seed": a.seed,
        "steps": a.max_steps, "warmup_steps": a.timing_warmup,
        "t_data_ms": med.get("data"), "t_fwd_bwd_ms": med.get("fwd_bwd"),
        "t_optim_ms": med.get("optim"), "t_comm_ms": comm["comm_ms"],
        "t_step_ms": step_ms,
        "tokens_per_s": a.global_batch * a.seq_len / (step_ms / 1000.0) if step_ms else None,
        "peak_mem_mb": runlog.peak_mem_mb(info.device),
        "val_loss_500": val_loss_500, "final_val_loss": last_val,
        "valid": None, "status": status, "notes": notes,
    }


def main(argv=None):
    a = parse_args(argv)
    if a.device == "auto":
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = a.device.split(":")[0]
    info = ddp_utils.setup(device=device_type)
    if not info.enabled:  # single process: honour an explicit cuda:N
        info.device = torch.device(a.device)
        if device_type == "cuda":
            torch.cuda.set_device(info.device)
    a.device = str(info.device)
    log = print if info.is_master else (lambda *args, **kwargs: None)
    if a.num_threads:
        torch.set_num_threads(a.num_threads)
    ddp_utils.seed_everything(a.seed)  # same init on every rank

    # global batch fixed; each rank takes global_batch / world_size, optionally in micro-batches
    per_rank = ddp_utils.per_rank_batch(a.global_batch, info.world_size)
    micro = a.micro_batch or per_rank
    assert per_rank % micro == 0, "global_batch / world_size must be a multiple of micro_batch"
    accum = per_rank // micro
    a.eval_batch = a.eval_batch or a.global_batch

    # data / vocab
    meta_path = Path(a.data_dir) / "meta.json"
    if a.vocab_size is None:
        a.vocab_size = json.loads(meta_path.read_text())["vocab_size"] if meta_path.exists() else 8192
    if a.dummy_data:
        a.train_path = a.val_path = None
        log("[INFO] --dummy_data: training on random tokens (loss will stay near ln(vocab))")
    else:
        a.train_path = resolve_split(a.data_dir, a.dataset)
        a.val_path = resolve_split(a.data_dir, "val")
        missing = [f"{n}.bin" for n, p in ((a.dataset, a.train_path), ("val", a.val_path)) if p is None]
        if missing:
            raise SystemExit(f"[ERROR] {', '.join(missing)} not found in {a.data_dir}. "
                            "Run slurm/tokenize.sbatch, check --data_dir, "
                            "or pass --dummy_data to train on random tokens on purpose.")
    data_kind = "random" if a.train_path is None else Path(a.train_path).name
    # per-rank seed so ranks sample different windows (rank 0 = single-process stream)
    train_loader = BatchLoader(a.train_path, a.vocab_size, a.seq_len, a.seed + 1000 * info.rank, a.device)

    # model
    cfg = GPTConfig(block_size=a.seq_len, vocab_size=a.vocab_size, n_layer=a.n_layer,
                    n_head=a.n_head, n_embd=a.n_embd, dropout=a.dropout)
    raw_model = GPT(cfg).to(info.device)
    n_params = raw_model.num_params()
    model = ddp_utils.wrap(raw_model, info)
    log(f"[INFO] params {n_params / 1e6:.2f}M ({raw_model.num_params(True) / 1e6:.2f}M excl. positions) | "
        f"device {a.device} | world {info.world_size} ({info.backend or 'no DDP'}) | {a.precision} | "
        f"data {data_kind} | batch {a.global_batch} (per rank {per_rank} = micro {micro} x accum {accum}) | "
        f"seq {a.seq_len} | seed {a.seed}")

    optimizer = raw_model.configure_optimizer(a.weight_decay, a.lr, (a.beta1, a.beta2), device_type)
    ctx = (torch.autocast(device_type=device_type, dtype=torch.bfloat16)
           if a.precision == "bf16" else nullcontext())
    val_seed = 10_000  # fixed: identical validation windows in every run

    def get_batches():
        return [train_loader.get_batch(micro) for _ in range(accum)]

    def fwd_bwd(batches):
        """Forward + backward over this rank's share of one global batch; returns the mean loss."""
        loss_sum = torch.zeros((), device=info.device)
        for i, (x, y) in enumerate(batches):
            # DDP all-reduces on every backward(); only sync on the last micro-batch
            last = i == len(batches) - 1
            with nullcontext() if last or not info.enabled else model.no_sync():
                with ctx:
                    _, loss = model(x, y)
                (loss / accum).backward()
            loss_sum += loss.detach()
        return loss_sum / accum

    timer = StepTimer(info.device, warmup_steps=a.timing_warmup)
    comm = {"fwd_bwd_sync_ms": 0.0, "fwd_bwd_nosync_ms": 0.0, "comm_ms": 0.0}
    val_loss_500, last_val, train_loss, first_loss = None, None, float("nan"), None
    history, t0 = [], time.time()
    model.train()
    try:
        for step in range(1, a.max_steps + 1):
            lr = get_lr(step - 1, a)
            for g in optimizer.param_groups:
                g["lr"] = lr
            timer.start_step()
            with timer.phase("data"):
                batches = get_batches()
            with timer.phase("fwd_bwd"):
                loss = fwd_bwd(batches)
            with timer.phase("optim"):
                if a.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), a.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            timer.end_step()

            # averaged over ranks so every rank takes the same branches below
            train_loss = ddp_utils.all_reduce_mean(loss.item(), info)
            first_loss = first_loss if first_loss is not None else train_loss
            if not math.isfinite(train_loss):
                raise RuntimeError(f"non-finite training loss at step {step}")

            if step % a.log_interval == 0 or step == 1:
                dt = (time.time() - t0) / step
                log(f"step {step:5d} | loss {train_loss:.4f} | lr {lr:.2e} | {dt * 1000:.0f} ms/step")
            if step % a.eval_interval == 0 or step == LOSS_CHECK_STEP or step == a.max_steps:
                last_val = estimate_val_loss(raw_model, a, val_seed, ctx, info)
                history.append({"step": step, "train_loss": train_loss, "val_loss": last_val})
                if step == LOSS_CHECK_STEP:
                    val_loss_500 = last_val
                log(f"step {step:5d} | VAL loss {last_val:.4f}" + ("  <- step 500" if step == LOSS_CHECK_STEP else ""))
        wall_s = round(time.time() - t0, 2)

        # exposed gradient communication: after training, so the losses above are unaffected
        if info.enabled and a.comm_measure_steps > 0:
            fixed = get_batches()  # same batch every time: the comparison is pure compute vs comm
            comm = measure_exposed_comm(model, lambda: fwd_bwd(fixed), info.device,
                                        n_steps=a.comm_measure_steps)
    except Exception as e:
        if info.is_master and a.runs_csv.lower() != "none":
            note = "; ".join(filter(None, [a.notes, f"{type(e).__name__}: {e}"[:300]]))
            runlog.append_row(build_row(a, info, n_params, timer.medians(), comm,
                                        val_loss_500, last_val, "failed", note), a.runs_csv)
        ddp_utils.cleanup(info)
        raise

    med = timer.medians()
    summary = {
        "n_params": n_params, "steps": a.max_steps, "seed": a.seed, "precision": a.precision,
        "global_batch": a.global_batch, "seq_len": a.seq_len, "device": a.device, "data": data_kind,
        "world_size": info.world_size, "backend": info.backend,
        "first_train_loss": first_loss, "final_train_loss": train_loss,
        "val_loss_500": val_loss_500, "final_val_loss": last_val,
        "timing_ms": med, "timed_steps": timer.measured_steps, "comm": comm,
        "wall_s": wall_s, "history": history,
    }
    log("SUMMARY " + json.dumps({k: v for k, v in summary.items() if k != "history"}))
    if info.is_master:
        if a.summary_json:
            Path(a.summary_json).write_text(json.dumps(summary, indent=2))
        if a.runs_csv.lower() != "none":
            auto = f"data {data_kind}; train loss {first_loss:.3f}->{train_loss:.3f}"
            if info.enabled:
                auto += (f"; fwd_bwd sync {comm['fwd_bwd_sync_ms']:.2f}ms "
                         f"nosync {comm['fwd_bwd_nosync_ms']:.2f}ms")
            runlog.append_row(build_row(a, info, n_params, med, comm, val_loss_500, last_val, "ok",
                                        "; ".join(filter(None, [a.notes, auto]))), a.runs_csv)
            log(f"[INFO] appended run to {a.runs_csv}")
    ddp_utils.cleanup(info)
    return summary


if __name__ == "__main__":
    main()
