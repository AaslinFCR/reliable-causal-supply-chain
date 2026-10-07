"""Execute the complete real-data research workflow with recorded stage status."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime

from scrc.common import ROOT, config, digest, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", help="Research settings path relative to the project root."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require verified local inputs; make no network requests.",
    )
    args = parser.parse_args()
    if args.config:
        os.environ["SCRC_CONFIG"] = args.config
    cfg = config()
    stages = []

    def stage(name, operation):
        print(f"\n{name}", flush=True)
        start = time.perf_counter()
        try:
            operation()
            stages.append(
                {
                    "stage": name,
                    "status": "passed",
                    "seconds": round(time.perf_counter() - start, 2),
                }
            )
        except Exception as error:
            stages.append({"stage": name, "status": "failed", "error": str(error)})
            raise
        finally:
            save_json(
                ROOT / "reports/pipeline_status.json",
                {
                    "python": sys.version,
                    "stages": stages,
                    "updated_utc": datetime.now(UTC).isoformat(),
                },
            )

    def acquire():
        official = cfg["market_source"] == "official"
        if not args.offline:
            if official:
                from scrc.data.official import download
            else:
                from scrc.data.download_archive import main as download
            download()
        sources = json.loads(
            (
                ROOT
                / (
                    "data/raw/official_manifest.json"
                    if official
                    else "data/raw/archive_manifest.json"
                )
            ).read_text()
        )
        from scrc.data.official import official_jobs

        if len(sources) != (len(official_jobs(cfg)) if official else 7):
            raise ValueError("Source manifest is incomplete.")
        for source in sources:
            if digest(ROOT / source["file"]) != source["sha256"]:
                raise ValueError(f"Source checksum mismatch: {source['file']}")
        path = ROOT / "data/raw/delhivery.csv"
        if not path.exists():
            if args.offline:
                raise FileNotFoundError("Delhivery input is missing.")
            import requests

            from scrc.data.download import DELHIVERY

            response = requests.get(DELHIVERY, timeout=180)
            response.raise_for_status()
            path.write_bytes(response.content)
        expected_delhivery = (
            "1062cb1d666394bf9ced980db9dbc6c1601a58b1449c388a42f6d0142b69f3d9"
        )
        if digest(path) != expected_delhivery:
            raise ValueError(
                "Delhivery differs from the frozen research input; inspect source changes before proceeding."
            )
        save_json(
            ROOT / "data/raw/delhivery_manifest.json",
            {
                "file": "data/raw/delhivery.csv",
                "sha256": expected_delhivery,
                "bytes": path.stat().st_size,
                "url": "https://raw.githubusercontent.com/shekshavalipattan/Delhivery-Logistics-Data-Pipeline-and-Feature-Engineering/master/delhivery.csv",
                "licence": "Redistribution terms unverified.",
            },
        )
        save_json(
            ROOT / "reports/input_verification.json",
            {
                "market_files_verified": len(sources),
                "delhivery_sha256": digest(path),
                "network_disabled": args.offline,
            },
        )
        save_json(
            ROOT / "reports/source_manifest.json",
            {
                "market_monthly_sources": sources,
                "delhivery": json.loads(
                    (ROOT / "data/raw/delhivery_manifest.json").read_text()
                ),
                "note": "Provenance metadata only; raw source responses are retained locally and excluded from the code ZIP.",
            },
        )

    def run_module(name, *arguments):
        subprocess.run(
            [sys.executable, str(ROOT / "run.py"), name, *arguments],
            cwd=ROOT,
            check=True,
        )

    stage("1. Acquire and verify original real datasets", acquire)
    stage(
        "2. Clean data and prepare past-only panels",
        lambda: run_module("scrc.data.build"),
    )
    stage(
        "3. Run forecasts, conformal, drift, causal estimates, gates and reports",
        lambda: run_module("scrc.eval.replay"),
    )
    stage("4. Run regression and leakage checks", lambda: run_module("pytest", "-q"))
    stage(
        "5. Check lint and formatting",
        lambda: (
            run_module("ruff", "check", "scrc", "tests", "scripts", "run.py"),
            run_module(
                "black",
                "--check",
                "--workers",
                "1",
                "scrc",
                "tests",
                "scripts",
                "run.py",
            ),
        ),
    )
    checklist = """# Implementation and evaluation completion

| Plan phase | Delivered evidence |
|---|---|
| Scaffold | Python package, YAML settings, isolated runtime |
| Real data | Government AGMARKNET monthly records, Delhivery CSV, checksums and data cards |
| Preparation | Units, chronological splits, missing dates and descriptive shortage proxy |
| Forecast and conformal | LightGBM, split and adaptive intervals, held-out results |
| Shift monitoring | Past-only scores, real event windows, alert/delay report |
| Observational causal estimates | T-learner, overlap, date bootstrap, placebo/subset/specification checks, DR scoring |
| Selection and gate | Conservative route policy; reject unsupported causal automation |
| Baselines and replay | Historical/rolling means, point forecasts, conformal and policy references |
| Ablations and figures | No-uncertainty/no-gate/no-shift/no-width comparisons; validation threshold sweep |
| Quality | Tests, lint, formatting, pipeline log, dependency lock and repeatability script |
| Paper support | Generated experimental setup/results fragment and measured tables |

Data-limited items: actual fill rates, operational stock-out reduction, inventory-policy costs, warehouse transfers, intervention costs, a joined market-shipment causal network, and live ERP execution cannot be evaluated from these public datasets. They are not represented as achieved results. Negative or inconclusive research findings are retained.
"""
    (ROOT / "reports/COMPLETION_CHECKLIST.md").write_text(checklist, encoding="utf-8")
    stage(
        "6. Package source, results and figures",
        lambda: subprocess.run(
            [sys.executable, str(ROOT / "scripts/build_delivery.py")],
            cwd=ROOT,
            check=True,
        ),
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_delivery.py")], cwd=ROOT, check=True
    )
    print("\nComplete workflow finished. Read reports/EVALUATION.md.", flush=True)


if __name__ == "__main__":
    main()
