# Cluster facts

> Fill in from what the team has already gathered (`sinfo`, `scontrol show node`,
> `sacctmgr show assoc user=$USER`, `module avail`). Paste raw command output in
> the "Raw output" section so the numbers are traceable.

## Partitions

| Partition | Nodes | Time limit | Notes |
|-----------|-------|------------|-------|
| cpu       | TODO  | TODO       | Used for data prep, CPU smoke tests, OpenMP thread sweep |
| TODO (GPU)| TODO  | TODO       |       |

## GPU nodes

| Node    | GPU model | GPUs/node | GPU memory | CPU cores | RAM  | Interconnect |
|---------|-----------|-----------|------------|-----------|------|--------------|
| haskell | TODO      | TODO      | TODO       | TODO      | TODO | TODO         |
| julia   | TODO      | TODO      | TODO       | TODO      | TODO | TODO         |

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
| pytorch/2.8.0   | Must verify it loads inside a job (step 7) |

## Raw output

```
# sinfo
```

```
# scontrol show node haskell
```

```
# scontrol show node julia
```
