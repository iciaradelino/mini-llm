"""Stand-in harness for the DDP + instrumentation code until train.py exists.

Trains a tiny next-token model on random tokens using ddp_utils, timing and
runlog exactly the way train.py will, then writes one CSV row from rank 0.

    # single process
    python src/ddp_smoke.py
    # 2 processes on CPU (gloo)
    torchrun --standalone --nproc_per_node=2 src/ddp_smoke.py

Native Windows can't run the multi-process case: the PyTorch 2.8 Windows wheel
has no usable gloo transport ("unsupported gloo device"). Use WSL or the
cluster's cpu partition instead.

Writes to results/smoke_runs.csv by default (git-ignored) so smoke runs never
end up in results/runs.csv.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import ddp_utils
import runlog
from timing import StepTimer, measure_exposed_comm


class TinyLM(nn.Module):
    def __init__(self, vocab: int, n_embd: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, n_embd)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))
        self.head = nn.Linear(n_embd, vocab)

    def forward(self, x, y):
        h = self.emb(x)
        h = h + self.mlp(h)
        logits = self.head(h)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--comm_measure_steps", type=int, default=10)
    p.add_argument("--global_batch", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--vocab", type=int, default=512)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    p.add_argument("--out", default=os.path.join("results", "smoke_runs.csv"))
    args = p.parse_args()

    info = ddp_utils.setup(device=args.device)
    ddp_utils.seed_everything(args.seed)
    batch = ddp_utils.per_rank_batch(args.global_batch, info.world_size)
    rng = ddp_utils.data_rng(args.seed, info)

    model = TinyLM(args.vocab, args.n_embd).to(info.device)
    n_params = sum(p.numel() for p in model.parameters())
    model = ddp_utils.wrap(model, info)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def get_batch():
        x = torch.from_numpy(rng.integers(0, args.vocab, (batch, args.seq_len))).long()
        y = torch.roll(x, -1, dims=1)  # learnable target so the loss visibly drops
        return x.to(info.device), y.to(info.device)

    timer = StepTimer(info.device, warmup_steps=args.warmup_steps)
    first_loss = None
    for step in range(args.steps):
        timer.start_step()
        with timer.phase("data"):
            x, y = get_batch()
        with timer.phase("fwd_bwd"):
            loss = model(x, y)
            loss.backward()
        with timer.phase("optim"):
            opt.step()
            opt.zero_grad(set_to_none=True)
        timer.end_step()
        if first_loss is None:
            first_loss = loss.item()

    final_loss = ddp_utils.all_reduce_mean(loss.item(), info)

    def fwd_bwd():
        x, y = get_batch()
        model(x, y).backward()

    comm = measure_exposed_comm(model, fwd_bwd, info.device, n_steps=args.comm_measure_steps)

    med = timer.medians()
    if info.is_master:
        tokens_per_s = args.global_batch * args.seq_len / (med["step"] / 1000.0)
        row = {
            "run_id": runlog.make_run_id(0),
            "timestamp": runlog.now_iso(),
            "job": "smoke",
            "experiment": f"ddp_smoke_ws{info.world_size}",
            **runlog.slurm_fields(),
            **runlog.device_fields(info.device),
            "world_size": info.world_size,
            "n_layer": 1, "n_embd": args.n_embd, "n_head": 0, "n_params": n_params,
            "seq_len": args.seq_len, "global_batch": args.global_batch,
            "precision": "fp32", "seed": args.seed,
            "steps": args.steps, "warmup_steps": args.warmup_steps,
            "t_data_ms": med["data"], "t_fwd_bwd_ms": med["fwd_bwd"],
            "t_optim_ms": med["optim"], "t_comm_ms": comm["comm_ms"],
            "t_step_ms": med["step"], "tokens_per_s": tokens_per_s,
            "peak_mem_mb": runlog.peak_mem_mb(info.device),
            "val_loss_500": None, "final_val_loss": final_loss,
            "valid": "n/a", "status": "ok",
            "notes": f"train loss {first_loss:.3f}->{final_loss:.3f}; "
                     f"fwd_bwd sync {comm['fwd_bwd_sync_ms']:.2f}ms nosync {comm['fwd_bwd_nosync_ms']:.2f}ms",
        }
        runlog.append_row(row, args.out)
        print(f"[rank0] world_size={info.world_size} backend={info.backend} "
              f"loss {first_loss:.3f}->{final_loss:.3f} "
              f"step {med['step']:.2f}ms comm {comm['comm_ms']:.2f}ms -> {args.out}")

    ddp_utils.cleanup(info)


if __name__ == "__main__":
    main()
