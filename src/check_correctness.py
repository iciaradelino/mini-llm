"""Apply the 2% correctness rule to results/runs.csv and fill in the `valid` column.

Rule (README): a run is valid if its validation loss after 500 steps is within 2%
of the single-GPU fp32 baseline, same seed, same global batch. The operational
definition is frozen in docs/interfaces.md section 5:

- Baseline rows: precision fp32, world_size 1, status ok, non-empty val_loss_500.
- A run is compared with the baseline rows that share every column in MATCH_KEYS;
  the reference is the median val_loss_500 of those rows.
- |val_loss_500 - reference| / reference <= tolerance  ->  yes, otherwise no.
  Baseline rows are tested against their own median as well.
- n/a: smoke/dev row, status not ok, no val_loss_500, or no matching baseline.

The script recomputes every row on each run, so it can be rerun after every job
(new baselines may change earlier verdicts). It only ever writes the `valid`
column; every other cell is written back unchanged.

Usage (from the repo root; standard library only, no torch needed):
    python3 src/check_correctness.py                    # update results/runs.csv
    python3 src/check_correctness.py --dry-run          # print the table, write nothing
    python3 src/check_correctness.py path/to/runs.csv --strict   # exit 1 if any run is invalid
"""

import argparse
import csv
import math
import os
import statistics
import sys
import tempfile

DEFAULT_CSV = os.path.join("results", "runs.csv")
TOLERANCE = 0.02

# Columns that must be identical between a run and its baseline (interfaces.md section 5).
# `steps` is --max_steps: the cosine learning-rate schedule ends at max_steps, so the
# learning rate at step 500 (and therefore val_loss_500) depends on it. A 1000-step run
# is not comparable with a 500-step baseline and gets n/a instead of a wrong `no`.
MATCH_KEYS = ["n_layer", "n_embd", "n_head", "seq_len", "global_batch", "seed",
              "steps", "device", "gpu_model"]
NUMERIC_KEYS = {"n_layer", "n_embd", "n_head", "seq_len", "global_batch", "seed", "steps"}
REQUIRED_COLUMNS = set(MATCH_KEYS) | {"run_id", "job", "experiment", "precision",
                                      "world_size", "val_loss_500", "valid", "status"}
NON_EXPERIMENT_JOBS = {"smoke", "dev"}  # these belong in results/smoke_runs.csv


def norm(key, value):
    """Normalise a cell for matching: '256', '256.0' and ' 256 ' are the same value."""
    value = (value or "").strip()
    if key in NUMERIC_KEYS and value:
        try:
            return str(int(float(value)))
        except ValueError:
            pass
    return value


def config_key(row):
    return tuple(norm(k, row.get(k)) for k in MATCH_KEYS)


def parse_loss(value):
    """Return the loss as a float, or None if the cell is empty."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return float("nan")


def is_baseline(row):
    loss = parse_loss(row.get("val_loss_500"))
    return (row.get("precision", "").strip() == "fp32"
            and norm("world_size", row.get("world_size")) in ("1", "1.0")
            and row.get("status", "").strip() == "ok"
            and row.get("job", "").strip() not in NON_EXPERIMENT_JOBS
            and loss is not None and math.isfinite(loss))


def no_baseline_reason(row, baselines):
    """Explain a missing baseline; name the columns that differ from the closest one."""
    if not baselines:
        return "no matching fp32 single-device baseline"
    key = config_key(row)
    closest = min(baselines, key=lambda b: sum(x != y for x, y in zip(key, b)))
    diff = [f"{k}={a or '-'} (baseline {b or '-'})"
            for k, a, b in zip(MATCH_KEYS, key, closest) if a != b]
    return "no matching fp32 single-device baseline; closest differs in " + ", ".join(diff)


def evaluate(rows, tolerance):
    """Return one result dict per row: valid, reference, rel_diff, n_baselines, reason."""
    baselines = {}
    for row in rows:
        if is_baseline(row):
            baselines.setdefault(config_key(row), []).append(parse_loss(row["val_loss_500"]))

    results = []
    for row in rows:
        res = {"valid": "n/a", "reference": None, "rel_diff": None, "n_baselines": 0, "reason": ""}
        job = row.get("job", "").strip()
        status = row.get("status", "").strip()
        loss = parse_loss(row.get("val_loss_500"))
        refs = baselines.get(config_key(row), [])

        if job in NON_EXPERIMENT_JOBS:
            res["reason"] = f"job '{job}' is not an experiment (belongs in smoke_runs.csv)"
        elif status != "ok":
            res["reason"] = f"status {status or 'empty'}"
        elif loss is None:
            res["reason"] = "no val_loss_500 (run shorter than 500 steps)"
        elif not refs:
            res["reason"] = no_baseline_reason(row, baselines)
        else:
            reference = statistics.median(refs)
            res["reference"], res["n_baselines"] = reference, len(refs)
            if not math.isfinite(loss):
                res["valid"], res["reason"] = "no", "non-finite val_loss_500"
            else:
                rel = abs(loss - reference) / reference
                res["rel_diff"] = rel
                res["valid"] = "yes" if rel <= tolerance else "no"
                if len(refs) < 3:
                    res["reason"] = f"only {len(refs)} baseline repeat(s)"
        results.append(res)
    return results


def print_table(rows, results, tolerance):
    print(f"{'run_id':<16} {'job':<5} {'experiment':<24} {'val500':>8} {'ref':>8} "
          f"{'diff':>7} {'n':>2}  {'valid':<5} note")
    for row, res in zip(rows, results):
        loss = parse_loss(row.get("val_loss_500"))
        ref, rel = res["reference"], res["rel_diff"]
        print(f"{row.get('run_id', '')[:16]:<16} {row.get('job', '')[:5]:<5} "
              f"{row.get('experiment', '')[:24]:<24} "
              f"{'' if loss is None else f'{loss:.4f}':>8} "
              f"{'' if ref is None else f'{ref:.4f}':>8} "
              f"{'' if rel is None else f'{100 * rel:.2f}%':>7} "
              f"{res['n_baselines'] or '':>2}  {res['valid']:<5} {res['reason']}")
    counts = {v: sum(r["valid"] == v for r in results) for v in ("yes", "no", "n/a")}
    print(f"\n{len(rows)} rows | tolerance {100 * tolerance:g}% | "
          f"yes {counts['yes']} | no {counts['no']} | n/a {counts['n/a']}")
    return counts


def write_csv(path, header, rows, newline):
    """Rewrite the CSV atomically with the csv dialect of src/runlog.py, keeping the
    file's existing line endings so git only sees the `valid` cells change."""
    folder = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".runs_", suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header, lineterminator=newline)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?", default=DEFAULT_CSV, help=f"runs CSV (default {DEFAULT_CSV})")
    p.add_argument("--tolerance", type=float, default=TOLERANCE, help="relative tolerance (default 0.02)")
    p.add_argument("--dry-run", action="store_true", help="print the verdicts, do not write the CSV")
    p.add_argument("--strict", action="store_true", help="exit with code 1 if any run is invalid")
    a = p.parse_args(argv)

    with open(a.csv, "rb") as f:
        newline = "\r\n" if b"\r\n" in f.read() else "\n"
    with open(a.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)

    missing = REQUIRED_COLUMNS - set(header)
    if missing:
        sys.exit(f"[ERROR] {a.csv} is missing columns {sorted(missing)} (schema: results/README.md)")
    if not rows:
        print(f"{a.csv}: no runs yet, nothing to check")
        return 0

    results = evaluate(rows, a.tolerance)
    counts = print_table(rows, results, a.tolerance)

    changed = sum(row.get("valid", "") != res["valid"] for row, res in zip(rows, results))
    if a.dry_run:
        print(f"dry run: {changed} row(s) would change, {a.csv} not written")
    else:
        for row, res in zip(rows, results):
            row["valid"] = res["valid"]
        write_csv(a.csv, header, rows, newline)
        print(f"wrote `valid` for {len(rows)} row(s) in {a.csv} ({changed} changed)")

    return 1 if a.strict and counts["no"] else 0


if __name__ == "__main__":
    sys.exit(main())