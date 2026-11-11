# Cluster facts

> Fill in from what the team has already gathered (`sinfo`, `scontrol show node`,
> `sacctmgr show assoc user=$USER`, `module avail`). Paste raw command output in
> the "Raw output" section so the numbers are traceable.

## Partitions

| Partition | Nodes | Time limit | Notes |
|-----------|-------|------------|-------|
| cpu       | fortran, haskell, julia | 7 days | Default partition (`*` in `sinfo`). Used for data prep, CPU smoke tests, OpenMP thread sweep |
| gpu       | fortran, haskell, julia | 3 days | GPU jobs: `--partition=gpu --gres=gpu:N` |
| interactive | fortran, haskell, julia | 8 hours | |

All three partitions list the same three nodes (see "Raw output", `sinfo`).

## GPU nodes

| Node    | GPU model | GPUs/node | GPU memory | CPU cores | RAM  | Interconnect |
|---------|-----------|-----------|------------|-----------|------|--------------|
| haskell | NVIDIA RTX 6000 Ada (sm_89) | TODO | TODO | TODO | TODO | TODO |
| julia   | NVIDIA RTX PRO 6000 Blackwell (sm_120) | TODO | TODO | TODO | TODO | TODO |
| fortran | TODO (in the `gpu` partition; model not recorded yet) | TODO | TODO | TODO | TODO | TODO |

GPU models for haskell and julia come from the help text of the `pytorch/2.8.0`
module (see "Raw output"). Confirm them with the `nvidia-smi` output at the top
of the first GPU job log, and fill in the rest from `scontrol show node`.

Intra-node GPU link (NVLink / PCIe, from `nvidia-smi topo -m`): TODO

## Accounts and limits

| Item                  | Value |
|-----------------------|-------|
| Account(s)            | TODO (gpu-prior+ — pending instructor confirmation) |
| Max jobs / user       | TODO  |
| Max GPUs / user       | TODO  |
| Max wall time         | TODO  |
| QOS                   | TODO  |

## Modules

| Module          | Notes |
|-----------------|-------|
| pytorch/2.8.0   | PyTorch 2.8.0 with CUDA 12.8, cuDNN 9 and NCCL, packaged in an Apptainer container. Plain `python3` does not see torch: run Python with `torch-python script.py` and other commands with `torch-exec` (e.g. `torch-exec torchrun ...`). `/home`, `/data` and `/scratch` are mounted inside. Loads and runs inside batch jobs on the `cpu` partition (jobs 26102, 28297). The module help says "run on a GPU node only" and that multi-GPU on haskell needs `NCCL_P2P_DISABLE=1` |

## Raw output

Transcribed from terminal output of the login node `rust` (1 Nov 2026).

```
# sinfo
PARTITION   AVAIL  TIMELIMIT  NODES  STATE NODELIST
cpu*           up 7-00:00:00      2    mix fortran,haskell
cpu*           up 7-00:00:00      1   idle julia
gpu            up 3-00:00:00      2    mix fortran,haskell
gpu            up 3-00:00:00      1   idle julia
interactive    up    8:00:00      2    mix fortran,haskell
interactive    up    8:00:00      1   idle julia
```

```
# module show pytorch/2.8.0   (help text, abridged)
PyTorch 2.8.0 with CUDA 12.8, cuDNN 9 and NCCL, in an Apptainer container.
Supports both GPU generations in the cluster: RTX 6000 Ada (haskell, sm_89) and
RTX PRO 6000 Blackwell (julia, sm_120).
Run on a GPU node only (--partition=gpu --gres=gpu:N).
Usage: torch-python train.py | torch-exec python -c "..." | torch-shell
Multi-GPU (DDP) on haskell - NCCL_P2P_DISABLE=1 is REQUIRED there and is also
faster than the peer-to-peer path (pending IOMMU fix).
/home, /data and /scratch are bind-mounted inside the container.
```

```
# scontrol show node haskell
TODO
```

```
# scontrol show node julia
TODO
```

```
# scontrol show node fortran
TODO
```
