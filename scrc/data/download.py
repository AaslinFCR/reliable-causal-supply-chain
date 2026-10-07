"""Download original public dataset responses with checksums and a manifest."""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

import requests

from scrc.common import ROOT, config, digest, save_json

BASE = "https://agmarknet.ceda.ashoka.edu.in/api/"
DELHIVERY = "https://raw.githubusercontent.com/shekshavalipattan/Delhivery-Logistics-Data-Pipeline-and-Feature-Engineering/master/delhivery.csv"


def fetch(job):
    """Cache an immutable source response; retry transient public-server failures."""
    state, commodity, year, endpoint = job
    path = ROOT / f"data/raw/ceda_{state}_{commodity}_{year}_{endpoint}.json"
    payload = {
        "state_id": state,
        "commodity_id": commodity,
        "district_id": 0,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "calculation_type": "d",
        "chart_type": "datadownload",
        "data_type": "price" if endpoint == "prices" else "quantity",
    }
    if not path.exists():
        for attempt in range(4):
            try:
                response = requests.post(BASE + endpoint, json=payload, timeout=180)
                if response.status_code == 429:
                    raise RuntimeError(
                        "CEDA rate limited the request. Stop and respect the server retry interval; use verified archive reproduction instead."
                    )
                response.raise_for_status()
                obj = response.json()
                if "data" not in obj or not isinstance(obj["data"], list):
                    raise ValueError(f"Unexpected API schema: {str(obj)[:200]}")
                if len(obj["data"]) >= 1000:
                    raise RuntimeError(
                        "CEDA response reached the observed 1,000-row cap; completeness is unverified. No experiment may use this response."
                    )
                path.write_bytes(response.content)
                break
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
    if len(json.loads(path.read_text(encoding="utf-8"))["data"]) >= 1000:
        raise RuntimeError(
            f"Cached CEDA response is potentially truncated: {path.name}"
        )
    return {
        "file": str(path.relative_to(ROOT)),
        "url": BASE + endpoint,
        "request": payload,
        "sha256": digest(path),
        "bytes": path.stat().st_size,
        "retrieved_utc": datetime.now(UTC).isoformat(),
        "licence": "CEDA/AGMARKNET source terms; verify redistribution permission before publication",
    }


def main():
    """Fetch all configured years, states and four plan commodities."""
    cfg = config()
    (ROOT / "data/raw").mkdir(parents=True, exist_ok=True)
    jobs = [
        (s, c, y, e)
        for s in cfg["states"]
        for c in cfg["commodities"]
        for y in range(int(cfg["start_date"][:4]), int(cfg["end_date"][:4]) + 1)
        for e in ["prices", "quantities"]
    ]
    records = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fetch, j): j for j in jobs}
        for future in as_completed(futures):
            records.append(future.result())
            save_json(ROOT / "data/raw/manifest.json", records)
            print(
                f"Downloaded {len(records)}/{len(jobs)}: {futures[future]}", flush=True
            )
    path = ROOT / "data/raw/delhivery.csv"
    if not path.exists():
        response = requests.get(DELHIVERY, timeout=180)
        response.raise_for_status()
        path.write_bytes(response.content)
    records.append(
        {
            "file": str(path.relative_to(ROOT)),
            "url": DELHIVERY,
            "sha256": digest(path),
            "bytes": path.stat().st_size,
            "retrieved_utc": datetime.now(UTC).isoformat(),
            "licence": "Public educational mirror; no explicit dataset redistribution licence verified",
        }
    )
    save_json(ROOT / "data/raw/manifest.json", records)


if __name__ == "__main__":
    main()
