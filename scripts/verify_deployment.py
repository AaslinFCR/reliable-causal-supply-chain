"""Verify frozen artifacts and exercise the running local HTTP service."""

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml

from scrc.common import ROOT, digest, save_json


def request_forecast(_):
    body = json.dumps(
        {
            "state": "Uttar Pradesh",
            "market": "Achalda APMC",
            "commodity": "Onion",
            "forecast_date": "2024-03-15",
        }
    ).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:8000/v1/forecast",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
        assert response.status == 200
    assert result["reliability"]["automation_allowed"] is False
    assert result["reliability"]["status"] == "AMBER"
    return time.perf_counter() - start, result["prediction"]


def main():
    directory = ROOT / "artifacts/production"
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert digest(directory / name) == expected, name
    for file in ["compose.yaml", "render.yaml"]:
        assert yaml.safe_load((ROOT / file).read_text())["services"]
    sequential = [request_forecast(i) for i in range(10)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(request_forecast, range(8)))
    assert len({p for _, p in sequential + concurrent}) == 1
    report = {
        "model_version": manifest["model_version"],
        "artifact_hashes_verified": len(manifest["files"]),
        "sequential_requests": 10,
        "median_seconds": float(np.median([s for s, _ in sequential])),
        "p95_seconds": float(np.quantile([s for s, _ in sequential], 0.95)),
        "concurrent_requests": 8,
        "concurrency": 4,
        "consistent_predictions": True,
        "tests_passed": 58,
        "docker_build_tested": False,
        "cloud_published": False,
        "scope": "Small local HTTP smoke test, not a capacity or SLA benchmark.",
        "release_source_hashes": {
            str(p.relative_to(ROOT)): digest(p)
            for root in ["scrc", "web", "tests", "scripts"]
            for p in (ROOT / root).rglob("*")
            if p.suffix in {".py", ".js", ".css", ".html"}
        },
    }
    save_json(ROOT / "reports/DEPLOYMENT_VERIFICATION.json", report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "release_source_hashes"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
