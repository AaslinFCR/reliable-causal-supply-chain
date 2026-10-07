"""Re-run the complete evaluation and compare numerical artifact hashes."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "results.csv",
    "market_results.csv",
    "market_results_by_region.csv",
    "logistics_results.csv",
    "causal_diagnostics.json",
    "shift_detection.json",
    "threshold_sweep_validation.csv",
]


def hashes():
    return {
        name: hashlib.sha256((ROOT / "experiments" / name).read_bytes()).hexdigest()
        for name in FILES
    }


def main():
    before = hashes()
    subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "scrc.eval.replay"], cwd=ROOT, check=True
    )
    after = hashes()
    mismatches = [name for name in FILES if before[name] != after[name]]
    result = {
        "identical_numerical_artifacts": not mismatches,
        "mismatches": mismatches,
        "sha256": after,
        "scope": "Full rerun in the same installed environment and frozen real-data inputs; not an independent clean-environment replication.",
    }
    (ROOT / "reports/reproducibility.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if mismatches:
        raise SystemExit(
            "Numerical artifacts differed. Inspect the reproducibility report."
        )


if __name__ == "__main__":
    main()
