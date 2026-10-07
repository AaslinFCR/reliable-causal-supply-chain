"""Fetch documented AGMARKNET historical records, never generated substitutes."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import requests

from scrc.common import ROOT, digest, save_json

BASE = "https://raw.githubusercontent.com/iancovert/Agmarknet/master/"


def fetch(job):
    crop, year = job
    url = BASE + f"{crop}/{year}.csv"
    path = ROOT / f"data/raw/agmarknet_{crop}_{year}.csv"
    if not path.exists():
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    print(f"Downloaded {crop} {year}: {path.stat().st_size:,} bytes", flush=True)
    return {
        "file": str(path.relative_to(ROOT)),
        "url": url,
        "sha256": digest(path),
        "bytes": path.stat().st_size,
        "retrieved_utc": datetime.now(UTC).isoformat(),
        "original_source": "Government AGMARKNET, archived by Ian Covert in June 2017",
        "source_documentation": "https://github.com/iancovert/Agmarknet",
        "licence": "No explicit archive redistribution licence verified; cite original source and archive. Do not redistribute raw data without verification.",
    }


def main():
    (ROOT / "data/raw").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as p:
        records = list(
            p.map(
                fetch, [(c, y) for c in ["Wheat", "Rice"] for y in [2014, 2015, 2016]]
            )
        )
    path = ROOT / "data/raw/delhivery.csv"
    if not path.exists():
        from scrc.data.download import DELHIVERY

        r = requests.get(DELHIVERY, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    records.append(
        {
            "file": str(path.relative_to(ROOT)),
            "sha256": digest(path),
            "bytes": path.stat().st_size,
            "url": "https://raw.githubusercontent.com/shekshavalipattan/Delhivery-Logistics-Data-Pipeline-and-Feature-Engineering/master/delhivery.csv",
            "retrieved_utc": datetime.now(UTC).isoformat(),
            "licence": "No explicit redistribution licence verified.",
        }
    )
    save_json(ROOT / "data/raw/archive_manifest.json", records)


if __name__ == "__main__":
    main()
