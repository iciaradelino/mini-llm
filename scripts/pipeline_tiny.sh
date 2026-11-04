#!/usr/bin/env bash
# scripts/pipeline_tiny.sh
#
# End-to-end smoke pipeline: prepare -> tokenizer -> encode -> train (~50 steps).
# Run from anywhere; it cd's to the repo root. Expects the modules to be loaded
# already (slurm/cpu_smoke.sbatch does `module load pytorch/2.8.0`).
#
# Outputs go to a SEPARATE directory (data/tokenized_smoke) so the smoke test never
# overwrites the real data/tokenized/ and the smoke row goes to results/smoke_runs.csv,
# not results/runs.csv.
#
# Overridable environment variables (defaults in brackets):
#   PY            python launcher                      [torch-python]
#   STEPS         training steps                       [50]
#   OUT_DIR       tokenizer / .bin output directory    [data/tokenized_smoke]
#   RUNS_CSV      where train.py appends its row       [results/smoke_runs.csv]
#   THREADS       CPU threads for torch                [$SLURM_CPUS_PER_TASK or 4]
#   EXPECTED_RAW  raw Gutenberg files that must exist  [30]
#   RUN_DDP       1 = also run a 2-process gloo DDP    [0]
#                 check after the single-process run

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"

PY="${PY:-torch-python}"
STEPS="${STEPS:-50}"
OUT_DIR="${OUT_DIR:-data/tokenized_smoke}"
RUNS_CSV="${RUNS_CSV:-results/smoke_runs.csv}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
EXPECTED_RAW="${EXPECTED_RAW:-30}"
RUN_DDP="${RUN_DDP:-0}"
TAG="${SLURM_JOB_ID:-local}"

export OMP_NUM_THREADS="$THREADS"

stage() { echo; echo "=== [STAGE $1] $(date '+%F %T') ==="; }
fail()  { echo "[PIPELINE FAILED] $*" >&2; exit 1; }
trap 'echo "[PIPELINE FAILED] line $LINENO: $BASH_COMMAND" >&2' ERR

echo "repo root: $ROOT | python: $PY | steps: $STEPS | out: $OUT_DIR | threads: $THREADS"

# ---------------------------------------------------------------- 0. checks
stage "0/5 environment"
command -v "$PY" >/dev/null || fail "'$PY' not found - did you 'module load pytorch/2.8.0'?"
"$PY" -c "import torch, numpy, tokenizers; print('torch', torch.__version__, '| numpy', numpy.__version__, '| tokenizers', tokenizers.__version__)" \
  || fail "missing python packages; run: pip install --user -r env/requirements.txt"

# Compute nodes may have no internet and prepare_corpus.py has no download timeout,
# so fail fast if the offline raw copies are not all there.
n_raw=$(ls data/raw/*.txt 2>/dev/null | wc -l || true)
[ "$n_raw" -ge "$EXPECTED_RAW" ] \
  || fail "only $n_raw/$EXPECTED_RAW raw files in data/raw/ - copy the offline Gutenberg downloads there first"
echo "raw files found: $n_raw"

# ---------------------------------------------------------------- 1. prepare
stage "1/5 prepare corpus (src/prepare_corpus.py)"
"$PY" src/prepare_corpus.py
n_train=$(ls data/clean/train/*.txt | wc -l)
n_val=$(ls data/clean/val/*.txt | wc -l)
echo "clean files: train=$n_train val=$n_val"
[ "$n_train" -gt 0 ] && [ "$n_val" -gt 0 ] || fail "empty clean/train or clean/val"

# ---------------------------------------------------------------- 2. tokenizer
stage "2/5 train BPE tokenizer (src/train_tokenizer.py)"
mkdir -p "$OUT_DIR"
"$PY" src/train_tokenizer.py --out "$OUT_DIR/tokenizer.json"

# ---------------------------------------------------------------- 3. encode
stage "3/5 encode (src/encode.py)"
"$PY" src/encode.py --out_dir "$OUT_DIR"
for f in train.bin val.bin tiny.bin meta.json tokenizer.json; do
  [ -s "$OUT_DIR/$f" ] || fail "$OUT_DIR/$f missing or empty"
done
"$PY" - "$OUT_DIR" <<'EOF'
import sys, json, numpy as np
d = sys.argv[1]
meta = json.load(open(f"{d}/meta.json"))
for n in ("train", "val", "tiny"):
    a = np.memmap(f"{d}/{n}.bin", dtype=np.uint16, mode="r")
    assert a.size == meta["splits"][n]["tokens"], f"{n}: size mismatch"
    assert int(a.max()) < meta["vocab_size"], f"{n}: id >= vocab"
    print(f"memmap OK: {n}.bin {a.size:,} tokens")
EOF

# ---------------------------------------------------------------- 4. train
stage "4/5 train $STEPS steps on tiny.bin, CPU (src/train.py)"
mkdir -p results slurm/logs
rows_before=$( [ -f "$RUNS_CSV" ] && wc -l < "$RUNS_CSV" || echo 0 )

TRAIN_ARGS=(--data_dir "$OUT_DIR" --dataset tiny --device cpu --seed 1337
            --max_steps "$STEPS" --global_batch 16 --warmup_steps 5
            --eval_interval 25 --eval_iters 5 --log_interval 10
            --num_threads "$THREADS"
            --job smoke --runs_csv "$RUNS_CSV")

"$PY" src/train.py "${TRAIN_ARGS[@]}" \
      --experiment "smoke_single_${TAG}" \
      --summary_json "slurm/logs/smoke_summary_${TAG}.json"

rows_after=$(wc -l < "$RUNS_CSV")
[ "$rows_after" -gt "$rows_before" ] || fail "train.py did not append a row to $RUNS_CSV"
tail -n 1 "$RUNS_CSV" | grep -q ',ok,' || fail "last row in $RUNS_CSV does not have status ok"
echo "runs csv row appended ($rows_before -> $rows_after lines), status ok"

# ---------------------------------------------------------------- 5. optional DDP
if [ "$RUN_DDP" = "1" ]; then
  stage "5/5 optional 2-process gloo DDP check"
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=2 src/train.py \
        "${TRAIN_ARGS[@]}" --max_steps 30 --comm_measure_steps 5 \
        --num_threads "$(( THREADS / 2 > 0 ? THREADS / 2 : 1 ))" \
        --experiment "smoke_ddp_cpu_${TAG}"
  tail -n 1 "$RUNS_CSV" | grep -q ',ok,' || fail "DDP row not ok"
else
  stage "5/5 DDP check skipped (set RUN_DDP=1 to enable)"
fi

echo
echo "[PIPELINE OK] $(date '+%F %T')"
