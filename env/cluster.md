# Cluster facts

> Fill in from what the team has already gathered (`sinfo`, `scontrol show node`,
> `sacctmgr show assoc user=$USER`, `module avail`). Paste raw command output in
> the "Raw output" section so the numbers are traceable.
>
> Items marked **TODO** are still unconfirmed; the command to get each one is given.

## Partitions

| Partition   | Nodes                   | Time limit | Notes                                                                                        |
|-------------|-------------------------|------------|----------------------------------------------------------------------------------------------|
| cpu         | fortran, haskell, julia | 7 days     | Default partition (`*` in `sinfo`). Used for data prep, CPU smoke tests, OpenMP thread sweep |
| gpu         | fortran, haskell, julia | 3 days     | GPU jobs: `--partition=gpu --gres=gpu:<type>:N`                                              |
| interactive | fortran, haskell, julia | 8 hours    | JupyterHub / interactive sessions. Never train on the login node                             |

All three partitions list the same three nodes (see "Raw output", `sinfo`).

## GPU nodes

| Node    | GPU model                                       | GPUs/node | GPU memory  | CPU cores | RAM (Slurm)                                                       | Slurm GRES type | Interconnect                                                       |
|---------|-------------------------------------------------|-----------|-------------|-----------|-------------------------------------------------------------------|-----------------|--------------------------------------------------------------------|
| haskell | NVIDIA RTX 6000 Ada (Ada Lovelace, 2022, sm_89) | 2         | 48 GB each  | 512       | ~397 GiB (the maintenance notice says 471 GB: both figures noted) | `rtx6000ada`    | PCIe, no NVLink (~26 GB/s direct PCIe per the notice)              |
| julia   | NVIDIA RTX PRO 6000 Blackwell* (2025, sm_120)   | 1         | 96 GB*      | 512       | ~503 GiB                                                          | `rtx6000bb`     | single GPU                                                         |
| fortran | NVIDIA RTX PRO 6000 Blackwell* (2025)           | 2         | 96 GB each* | 512       | ~738 GiB (Slurm `RealMemory=756000` MiB)                          | `rtx6000bb`     | **TODO** (PCIe or NVLink; topology command failed, see Raw output) |

\* Slurm labels the type `rtx6000bb`. GPU models for haskell and julia are also in the
help text of the `pytorch/2.8.0` module (see "Raw output"); fortran's model comes from
the team's cluster data and is not in that help text. Confirm every model with the `nvidia-smi`
output at the top of the GPU job logs (`slurm/logs/job_A_*.out`, Job B for julia, Job D for fortran).

Intra-node GPU link (NVLink / PCIe, from `nvidia-smi topo -m`): **haskell: `NODE` (PCIe path through PCIe host bridges within the same NUMA node; no NVLink).** Job C prints the matrix at the start of its log; paste the matrix into "Raw output" below.

Inter-node network (fortran <-> julia, needed for Job D): **TODO**. Job D prints
`ip -brief addr` and `ibstat` per node and a NCCL probe (the `NET/...` line says which one NCCL used).

Missing node data: `scontrol show node fortran | grep -E "CPUTot|RealMemory|Gres"`.

## Accounts and limits

| Item            | Value                                                                                                                                                               |
|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Account(s)      | none needed: GPU jobs are submitted **without** `--account` (team plan, Nov 2026). The earlier "gpu-prior+" note is superseded unless the instructor says otherwise |
| QOS             | none needed                                                                                                                                                         |
| Max jobs / user | **TODO** (association output does not report a value; QOS limits not yet checked)                                                                                   |
| Max GPUs / user | **TODO** (association output does not report a value; QOS limits not yet checked)                                                                                   |
| Max wall time   | cpu 7 days, gpu 3 days, interactive 8 hours (partition limits)                                                                                                      |

GPU job header used by the project:

```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000ada:N      # haskell
#SBATCH --gres=gpu:rtx6000bb:N       # julia, fortran
```

Every GPU log starts with `nvidia-smi`.

## Modules

| Module        | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| pytorch/2.8.0 | PyTorch 2.8.0 with CUDA 12.8, cuDNN 9 and NCCL, packaged in an Apptainer container. Plain `python3` does not see torch: run Python with `torch-python script.py` and other commands with `torch-exec` (e.g. `torch-exec torchrun ...`). `/home`, `/data` and `/scratch` are mounted inside. Loads and runs inside batch jobs on the `cpu` partition (jobs 26102, 26995, 28297). The module help says "run on a GPU node only" and that multi-GPU on haskell needs `NCCL_P2P_DISABLE=1` |

Versions seen inside batch jobs (`env/modules.md`): Python 3.11.13, torch 2.8.0+cu128.
NCCL version: **2.27.3+cuda12.9** (from Job C, `slurm/logs/job_C_32368.out`).

## What this means for the experiments

- Memory is not a constraint: the base model needs ~2 GB (the ~91M model ~5-6 GB) on 48-96 GB GPUs.
- Job B compares an older generation (Ada, haskell) with a newer one (Blackwell, julia).
- **Job C: the module help says `NCCL_P2P_DISABLE=1` is REQUIRED for multi-GPU on haskell and
  is also faster than the peer-to-peer path (pending an IOMMU fix).** So P2P off is the working
  configuration and the P2P-on runs may hang or fail. `slurm/job_C_haskell_2gpu.sbatch` runs
  P2P off first and gives P2P on a timeout; a hang or failure is reported as a measured result
  with its log. The Ada cards have no NVLink, so the comparison is direct PCIe vs host memory.
- Job D (fortran + julia) uses the same GPU model on both nodes. The P2P note is about haskell only.
- Job E: 512 cores per node, so 64 threads is not a limit.

## Raw output

Transcribed from terminal output of the login node `rust` (1 Nov 2026), plus subsequent
GPU-job and login-node checks on 18 Nov 2026.

```text
# sinfo
PARTITION   AVAIL  TIMELIMIT  NODES  STATE NODELIST
cpu*           up 7-00:00:00      2    mix fortran,haskell
cpu*           up 7-00:00:00      1   idle julia
gpu            up 3-00:00:00      2    mix fortran,haskell
gpu            up 3-00:00:00      1   idle julia
interactive    up    8:00:00      2    mix fortran,haskell
interactive    up    8:00:00      1   idle julia
```

```text
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

```text
# scontrol show node haskell
   CPUAlloc=6 CPUEfctv=512 CPUTot=512 CPULoad=0.39
   Gres=gpu:rtx6000ada:2
   RealMemory=406528 AllocMem=20480 FreeMem=14377 Sockets=2 Boards=1
```

```text
# scontrol show node julia
   CPUAlloc=4 CPUEfctv=512 CPUTot=512 CPULoad=0.38
   Gres=gpu:rtx6000bb:1
   RealMemory=515002 AllocMem=8192 FreeMem=137000 Sockets=2 Boards=1
```

```text
# scontrol show node fortran | grep -E "CPUTot|RealMemory|Gres"
   CPUAlloc=284 CPUEfctv=512 CPUTot=512 CPULoad=407.79
   Gres=gpu:rtx6000bb:2
   RealMemory=756000 AllocMem=180224 FreeMem=41388 Sockets=2 Boards=1
```

```text
# nvidia-smi topo -m (from Job C log: slurm/logs/job_C_32368.out)
        GPU0    GPU1    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NODE    0-3,256-259     0               N/A
GPU1    NODE     X      0-3,256-259     0               N/A

Legend:
  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE  = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

```text
# NCCL version (from Job C log: slurm/logs/job_C_32368.out)
haskell:611170:611170 [0] NCCL INFO NCCL version 2.27.3+cuda12.9
haskell:611171:611171 [1] NCCL INFO NCCL version 2.27.3+cuda12.9
NCCL version 2.27.3+cuda12.9
```

```text
# sacctmgr show assoc user=hpca06 format=Account,Partition,QOS,MaxJobs,MaxWall%20,GrpTRES%30
   Account  Partition                  QOS MaxJobs              MaxWall                        GrpTRES
---------- ---------- -------------------- ------- -------------------- ------------------------------
  hpcclass            student,student-2gpu
```

```text
# nvidia-smi topo -m (run directly on login node fortran)
Failed to run topology matrix
```

```text
# srun topology attempt (from login node fortran)
srun --partition=gpu --nodelist=fortran --gres=gpu:rtx6000bb:2 --ntasks=1 --cpus-per-task=2 --time=00:05:00 nvidia-smi topo -m
srun: error: Unable to create step for job 32350: Invalid generic resource (gres) specification
```

```text
# GPU model confirmation: Job A, slurm/logs/job_A_32017.out
|   0  NVIDIA RTX 6000 Ada Gene...    On  |   00000000:61:00.0 Off |                  Off |
torch 2.8.0+cu128 | cuda True | NVIDIA RTX 6000 Ada Generation
```

```text
# GPU model confirmation: Job B, slurm/logs/job_B_32366.out
|   0  NVIDIA RTX PRO 6000 Blac...    Off |   00000001:C5:00.0 Off |                    0 |
NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, 580.126.09
torch 2.8.0+cu128 | cuda True | NVIDIA RTX PRO 6000 Blackwell Server Edition
```
