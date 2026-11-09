"""Test src/check_correctness.py on tests/sample_runs.csv.

The sample CSV holds SYNTHETIC rows (made-up losses and GPU names) covering every
case of the rule in docs/interfaces.md section 5. Each row's `notes` says the
verdict it must get ("expect valid=..."). The test runs the script on a temporary
copy, so tests/sample_runs.csv itself is never modified.

Run from the repo root (standard library only):
    python3 tests/test_check_correctness.py
"""

import csv
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import check_correctness  # noqa: E402

SAMPLE = os.path.join(ROOT, "tests", "sample_runs.csv")


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "runs.csv")
        shutil.copy(SAMPLE, path)
        check_correctness.main([path])
        before, after = read(SAMPLE), read(path)
        with open(SAMPLE, "rb") as f1, open(path, "rb") as f2:
            same_endings = (b"\r\n" in f1.read()) == (b"\r\n" in f2.read())

    failures = []
    for b, a in zip(before, after):
        expected = b["notes"].split("expect valid=")[1].strip()
        if a["valid"] != expected:
            failures.append(f"{a['run_id']} ({a['experiment']}): got {a['valid']!r}, expected {expected!r}")
        changed = [c for c in b if c != "valid" and b[c] != a[c]]
        if changed:
            failures.append(f"{a['run_id']}: columns other than `valid` changed: {changed}")
    if len(before) != len(after):
        failures.append(f"row count changed: {len(before)} -> {len(after)}")
    if not same_endings:
        failures.append("line endings changed")

    print()
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        return 1
    print(f"PASSED: {len(after)} rows got the expected verdict; no other cell changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())