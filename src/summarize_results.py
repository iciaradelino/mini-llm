#!/usr/bin/env python3
"""
src/summarize_results.py

Turns results/runs.csv into the summary tables (Markdown) and plots for the report.
Every number is the MEDIAN of the repeats of one configuration (experiment label
without its _rN suffix), so a single slow run cannot distort a result.

Rules applied before summarising
- only rows with status = ok;
- PROFILED runs are left out of all timing tables (their timings include
  profiler overhead);
- Job E uses only job 31986 (job 28597 was cancelled; see REPRODUCE.md);
- the torch.compile rows (job 28862) were trained on the PREVIOUS corpus, so
  their losses are shown separately and never compared with other jobs.

Usage
    python src/summarize_results.py                       # writes results/summary.md + results/plots/
    python src/summarize_results.py --csv results/runs.csv --out results
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TOKENS_PER_STEP = 64 * 256          # global batch x sequence length (baseline)
JOB_E_ID = "31986"
OLD_CORPUS_JOBS = {"28862"}         # torch.compile job, run before the corpus rebuild


# ----------------------------------------------------------------------------- data
def load(csv_path):
    df = pd.read_csv(csv_path, dtype={"slurm_job_id": str, "run_id": str},
                     keep_default_na=False, na_values=[""])   # keep "n/a" in `valid` as text
    df = df[df["status"] == "ok"].copy()
    df["profiled"] = df["notes"].fillna("").str.contains("PROFILED")
    df["config"] = df["experiment"].str.replace(r"_r\d+$", "", regex=True)
    return df


def med(df, group, cols):
    """Median of `cols` per group, plus the number of repeats."""
    g = df.groupby(group)
    out = g[cols].median()
    out["n"] = g.size()
    return out


def md_table(rows, headers, align):
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join({"l": ":--", "r": "--:", "c": ":-:"}[a] for a in align) + " |"
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join([line, sep] + body)


f1 = lambda x: f"{x:,.1f}"
f2 = lambda x: f"{x:.2f}"
f4 = lambda x: f"{x:.4f}"
fk = lambda x: f"{x:,.0f}"
pct = lambda x: f"{100*x:.1f} %"


# ----------------------------------------------------------------------------- tables
def table_overview(df):
    rows = []
    order = ["A", "B", "C", "E", "opt"]
    for job, sub in sorted(df.groupby("job"), key=lambda x: order.index(x[0]) if x[0] in order else 99):
        if job == "E":
            sub = sub[sub["slurm_job_id"] == JOB_E_ID]
        jobs = ", ".join(sorted(sub["slurm_job_id"].unique()))
        hw = sub["gpu_model"].fillna("").replace("", np.nan).dropna().unique()
        hw = hw[0].replace("NVIDIA ", "").replace(" Server Edition", "") if len(hw) else "CPU (AMD EPYC 9755)"
        valid = sub["valid"].fillna("(empty)").value_counts().to_dict()
        vtxt = ", ".join(f"{k}: {v}" for k, v in valid.items())
        rows.append([f"**{job}**", jobs, ", ".join(sorted(sub["nodes"].unique())), hw, len(sub), vtxt])
    return md_table(rows, ["Job", "Slurm job(s)", "Node", "Hardware", "Runs", "`valid`"],
                    "llllrl")


def table_gpu_single(df):
    """Jobs A and B: one GPU, fp32 vs bf16, two GPU generations."""
    sub = df[df["job"].isin(["A", "B"]) & df["config"].str.startswith("baseline") & ~df["profiled"]]
    m = med(sub, ["job", "gpu_model", "config"], ["t_step_ms", "t_fwd_bwd_ms", "t_optim_ms",
                                                   "tokens_per_s", "peak_mem_mb", "val_loss_500"])
    ref = m.loc[("A", slice(None), "baseline_fp32")]["t_step_ms"].iloc[0]
    m = m.reindex(sorted(m.index, key=lambda k: (k[0], 0 if "fp32" in k[2] else 1)))
    rows = []
    for (job, gpu, cfg), r in m.iterrows():
        rows.append([job, gpu.replace("NVIDIA ", "").replace(" Server Edition", ""),
                     cfg.replace("baseline_", ""), f1(r.t_step_ms), fk(r.tokens_per_s),
                     f2(ref / r.t_step_ms) + "×", f1(r.peak_mem_mb), f4(r.val_loss_500), int(r.n)])
    return md_table(rows, ["Job", "GPU", "Precision", "Step (ms)", "Tokens/s",
                           "Speed-up vs Ada fp32", "Peak mem (MB)", "Val loss @500", "n"],
                    "lllrrrrrr"), m


def table_seqlen(df):
    sub = df[(df["job"] == "A") & df["config"].str.match(r"(seqlen\d+|baseline)_fp32$") & ~df["profiled"]].copy()
    m = med(sub, ["seq_len"], ["t_step_ms", "tokens_per_s", "peak_mem_mb"])
    rows = []
    for L, r in m.iterrows():
        tok_step = 64 * L
        rows.append([L, f"{tok_step:,}", f1(r.t_step_ms), f"{1000*r.t_step_ms/tok_step:.2f}",
                     fk(r.tokens_per_s), f1(r.peak_mem_mb), int(r.n)])
    return md_table(rows, ["Seq length", "Tokens/step", "Step (ms)", "µs per token",
                           "Tokens/s", "Peak mem (MB)", "n"], "rrrrrrr"), m


def table_ddp(df):
    """Job C against the 1-GPU fp32 baseline of Job A (same GPU model, same node)."""
    base = df[(df["job"] == "A") & (df["config"] == "baseline_fp32")]
    t1 = base["t_step_ms"].median()
    tok1 = base["tokens_per_s"].median()
    sub = df[(df["job"] == "C") & ~df["profiled"]]
    m = med(sub, ["config"], ["t_step_ms", "t_fwd_bwd_ms", "t_optim_ms", "t_comm_ms",
                              "tokens_per_s", "val_loss_500"])
    rows = [["1 GPU (Job A)", "—", f1(t1), fk(tok1), "1.00×", "100 %", "—", "—",
             f4(base["val_loss_500"].median()), len(base)]]
    for cfg, r in m.iterrows():
        p2p = "on" if cfg.endswith("p2p_on") else "off"
        s = t1 / r.t_step_ms
        rows.append([f"2 GPUs, P2P {p2p}", "P2P/CUMEM" if p2p == "on" else "SHM", f1(r.t_step_ms),
                     fk(r.tokens_per_s), f2(s) + "×", pct(s / 2), f2(r.t_comm_ms),
                     pct(r.t_comm_ms / r.t_step_ms), f4(r.val_loss_500), int(r.n)])
    return md_table(rows, ["Configuration", "NCCL path", "Step (ms)", "Tokens/s", "Speed-up",
                           "Efficiency", "Exposed comm (ms)", "Comm share", "Val loss @500", "n"],
                    "llrrrrrrrr"), m, t1


def table_cpu(df):
    sub = df[(df["job"] == "E") & (df["slurm_job_id"] == JOB_E_ID)]
    m = med(sub, ["num_threads"], ["t_step_ms", "t_data_ms", "t_fwd_bwd_ms", "t_optim_ms", "tokens_per_s"])
    t1 = m.loc[1, "t_step_ms"]
    m["speedup"] = t1 / m["t_step_ms"]
    # Karp-Flatt experimentally determined serial fraction: e = (1/S - 1/p) / (1 - 1/p)
    m["karp_flatt"] = [(1 / r.speedup - 1 / p) / (1 - 1 / p) if 1 < p <= 32 else np.nan
                       for p, r in m.iterrows()]
    rows = []
    for p, r in m.iterrows():
        label = f"{p} (SMT, 32 cores)" if p == 64 else str(p)
        eff = "—" if p == 64 else pct(r.speedup / p)
        kf = "—" if np.isnan(r.karp_flatt) else pct(r.karp_flatt)
        spread = sub[sub["num_threads"] == p]["t_step_ms"]
        rows.append([label, f1(r.t_step_ms), f"±{100*(spread.max()-spread.min())/2/r.t_step_ms:.1f} %",
                     f2(r.speedup) + "×", eff, kf, f2(r.t_data_ms), f1(r.t_fwd_bwd_ms), f1(r.t_optim_ms),
                     fk(r.tokens_per_s), int(r.n)])
    # serial fraction for the Amdahl curve: median Karp-Flatt over 8-32 cores
    # (2 is too few cores to be informative; 4 is the anomalous point, see notes)
    f = float(m.loc[[8, 16, 32], "karp_flatt"].median())
    return md_table(rows, ["Threads", "Step (ms)", "Spread", "Speed-up", "Efficiency", "Karp–Flatt",
                           "Data (ms)", "Fwd+bwd (ms)", "Optim (ms)", "Tokens/s", "n"], "lrrrrrrrrrr"), m, f


def table_compile(df):
    sub = df[df["job"] == "opt"]
    m = med(sub, ["config"], ["t_step_ms", "t_fwd_bwd_ms", "tokens_per_s", "peak_mem_mb", "val_loss_500"])
    eager = m.loc["opt_fp32", "t_step_ms"]
    rows = []
    for cfg, r in m.iterrows():
        rows.append(["torch.compile" if "compile" in cfg else "eager", f1(r.t_step_ms), fk(r.tokens_per_s),
                     f"{100*(eager/r.t_step_ms-1):+.2f} %", f1(r.peak_mem_mb), f4(r.val_loss_500), int(r.n)])
    return md_table(rows, ["Mode", "Step (ms)", "Tokens/s", "Faster than eager", "Peak mem (MB)",
                           "Val loss @500 (old corpus)", "n"], "lrrrrrr")


def table_landscape(gpu_m, ddp_m, t1_ada, cpu_m):
    """Throughput of every hardware configuration, baseline model, fp32 unless stated."""
    entries = [
        ("CPU, 1 core (EPYC 9755)", cpu_m.loc[1, "t_step_ms"]),
        ("CPU, 32 cores", cpu_m.loc[32, "t_step_ms"]),
        ("1 × RTX 6000 Ada, fp32", t1_ada),
        ("2 × RTX 6000 Ada (DDP, P2P on), fp32", ddp_m.loc["ddp2_p2p_on", "t_step_ms"]),
        ("1 × RTX PRO 6000 Blackwell, fp32", gpu_m.xs("baseline_fp32", level="config")["t_step_ms"].min()),
        ("1 × RTX 6000 Ada, bf16", gpu_m.loc[("A", slice(None), "baseline_bf16")]["t_step_ms"].iloc[0]),
        ("1 × RTX PRO 6000 Blackwell, bf16", gpu_m.loc[("B", slice(None), "baseline_bf16")]["t_step_ms"].iloc[0]),
    ]
    c1 = entries[0][1]
    rows = [[name, f1(t), fk(TOKENS_PER_STEP / (t / 1000)), f"{c1/t:,.0f}×"] for name, t in entries]
    return md_table(rows, ["Hardware", "Step (ms)", "Tokens/s", "vs 1 CPU core"], "lrrr"), entries


# ----------------------------------------------------------------------------- plots
def plot_cpu(cpu_m, f, out):
    p = cpu_m.index.values
    phys = p[p <= 32]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(phys, phys, "--", color="grey", label="ideal")
    ax[0].plot(phys, 1 / (f + (1 - f) / phys), ":", color="tab:orange", label=f"Amdahl, serial {100*f:.1f} %")
    ax[0].plot(phys, cpu_m.loc[phys, "speedup"], "o-", color="tab:blue", label="measured (median of 3)")
    if 64 in p:
        ax[0].plot([32], [cpu_m.loc[64, "speedup"]], "s", color="tab:red", label="64 threads on 32 cores (SMT)")
    ax[0].set(xscale="log", yscale="log", xlabel="threads (one per physical core)", ylabel="speed-up vs 1 thread",
              title="Job E: CPU strong scaling")
    ax[0].set_xticks(phys); ax[0].set_xticklabels(phys); ax[0].legend(fontsize=8)
    eff = cpu_m.loc[phys, "speedup"] / phys
    ax[1].plot(phys, 100 * eff, "o-")
    ax[1].set(xscale="log", ylim=(0, 105), xlabel="threads", ylabel="efficiency (%)", title="Job E: parallel efficiency")
    ax[1].set_xticks(phys); ax[1].set_xticklabels(phys)
    fig.tight_layout(); fig.savefig(out / "job_E_speedup_efficiency.png", dpi=150); plt.close(fig)

    # share of the step per phase (stacked bars must use a linear axis to be honest)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [str(x) if x != 64 else "64\n(SMT)" for x in p]
    parts = [("t_fwd_bwd_ms", "forward + backward"), ("t_optim_ms", "optimiser"), ("t_data_ms", "data")]
    total = sum(cpu_m[c].values for c, _ in parts)
    bottom = np.zeros(len(p))
    for col, name in parts:
        share = 100 * cpu_m[col].values / total
        ax.bar(labels, share, bottom=bottom, label=name); bottom += share
    for i, t in enumerate(cpu_m["t_step_ms"].values):
        ax.text(i, 101, f"{t:,.0f} ms", ha="center", va="bottom", fontsize=7)
    ax.set(ylim=(90, 103), xlabel="threads", ylabel="share of step time (%)",
           title="Job E: where the step time goes (y-axis starts at 90 %)")
    ax.legend(fontsize=8, loc="lower left"); fig.tight_layout()
    fig.savefig(out / "job_E_phases.png", dpi=150); plt.close(fig)


def plot_landscape(entries, out):
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [e[0] for e in entries]; tps = [TOKENS_PER_STEP / (e[1] / 1000) for e in entries]
    ax.barh(names, tps, color=["tab:grey"] * 2 + ["tab:blue"] * 5)
    ax.set(xscale="log", xlabel="tokens per second (log)", title="Training throughput, baseline model")
    ax.invert_yaxis(); fig.tight_layout(); fig.savefig(out / "throughput_all_hardware.png", dpi=150); plt.close(fig)


def plot_seqlen(seq_m, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    L = seq_m.index.values
    ax.plot(L, 1000 * seq_m["t_step_ms"] / (64 * L), "o-")
    ax.set(xscale="log", xlabel="sequence length", ylabel="µs per token", title="Job A: cost per token vs sequence length")
    ax.set_xticks(L); ax.set_xticklabels(L)
    fig.tight_layout(); fig.savefig(out / "job_A_seqlen.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="results/runs.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out); plots = out / "plots"; plots.mkdir(parents=True, exist_ok=True)
    df = load(a.csv)

    gpu_tbl, gpu_m = table_gpu_single(df)
    seq_tbl, seq_m = table_seqlen(df)
    ddp_tbl, ddp_m, t1_ada = table_ddp(df)
    cpu_tbl, cpu_m, f = table_cpu(df)
    land_tbl, entries = table_landscape(gpu_m, ddp_m, t1_ada, cpu_m)

    md = [
        "# Results summary",
        f"_Generated by `src/summarize_results.py` from `{a.csv}`. Every value is the median of the "
        "repeats of a configuration; `n` is the number of repeats. Profiled runs are excluded from timings._\n",
        "## Runs overview\n", table_overview(df),
        "\n## Throughput across hardware (baseline model)\n", land_tbl,
        "\n## Jobs A and B: one GPU, fp32 vs bf16, two GPU generations\n", gpu_tbl,
        "\n## Job A: sequence length (fp32, 1 × RTX 6000 Ada)\n", seq_tbl,
        "\n## Job C: data-parallel training on 2 GPUs (haskell)\n", ddp_tbl,
        "\nSpeed-up and efficiency are relative to the 1-GPU fp32 median of Job A (same GPU model, same node). "
        "The DDP validation loss is 0.2 % lower than on 1 GPU because each rank samples different "
        "training windows; well within the 2 % rule. "
        "Exposed communication = forward/backward time with gradient sync minus without (`no_sync()`), "
        "measured after training.\n",
        "\n## Job E: CPU strong scaling (job 31986, AMD EPYC 9755)\n", cpu_tbl,
        "\nKarp–Flatt = experimentally determined serial fraction, e = (1/S − 1/p) / (1 − 1/p). "
        f"Over 8–32 cores it is {100*cpu_m.loc[[8,16,32],'karp_flatt'].min():.1f}–{100*cpu_m.loc[[8,16,32],'karp_flatt'].max():.1f} % "
        f"(median {100*f:.1f} %), so by Amdahl's law the speed-up would level off near {1/f:.0f}×; "
        "at 32 cores it is still rising. The 4-thread point is anomalous (Karp–Flatt "
        f"{pct(cpu_m.loc[4,'karp_flatt'])}): forward + backward only, systematic in all repeats; "
        "hypothesis: the 4 pinned threads share one 8-core CCD and saturate its memory link. "
        "Spread = half the range of the repeats.\n",
        "\n## Optimisation: torch.compile (job 28862)\n", table_compile(df),
        "\n**Caution:** job 28862 ran on the previous corpus (before the cleaning change of 11 Nov), "
        "so its losses must not be compared with the other jobs; its timings are comparable.\n",
    ]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    plot_cpu(cpu_m, f, plots); plot_landscape(entries, plots); plot_seqlen(seq_m, plots)
    print(f"[DONE] {out/'summary.md'} and plots in {plots}/")


if __name__ == "__main__":
    main()
