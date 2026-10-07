"""Original government AGMARKNET acquisition and normalized market observations."""

import gzip
import hashlib
import json
import time
from datetime import UTC, datetime

import pandas as pd
import requests

from scrc.common import ROOT, config, digest, save_json
from scrc.data.panel import market_features

URL = "https://api.agmarknet.gov.in/v1/prices-and-arrivals/date-wise/specific-commodity"
STATES = {34: "Uttar Pradesh", 16: "Karnataka", 36: "West Bengal", 20: "Maharashtra"}
CROPS = {1: "Wheat", 3: "Rice", 23: "Onion", 24: "Potato"}
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Origin": "https://agmarknet.gov.in",
    "Referer": "https://agmarknet.gov.in/",
    "Accept": "application/json",
}


class SourceRateLimit(RuntimeError):
    """A publisher throttle requiring a cooldown before any new request."""

    def __init__(self, response):
        super().__init__("Government API rate limit")
        value = response.headers.get("Retry-After", "")
        self.seconds = max(900, int(value)) if value.isdigit() else 900
        self.headers = {
            k: v
            for k, v in response.headers.items()
            if k.lower()
            in {
                "date",
                "retry-after",
                "ratelimit-limit",
                "ratelimit-remaining",
                "ratelimit-reset",
                "x-ratelimit-limit",
                "x-ratelimit-remaining",
                "x-ratelimit-reset",
            }
        }


def fetch(job):
    state, crop, year, month = job
    path = ROOT / f"data/raw/official/{state}_{crop}_{year}_{month:02d}.json.gz"
    params = {"stateId": state, "commodityId": crop, "year": year, "month": month}
    if not path.exists():
        time.sleep(2)
        for attempt in range(4):
            try:
                response = requests.get(
                    URL, params=params, headers=HEADERS, timeout=120
                )
                if response.status_code == 429:
                    raise SourceRateLimit(response)
                response.raise_for_status()
                obj = response.json()
                if obj.get("success") is not True or not isinstance(
                    obj.get("markets"), list
                ):
                    raise ValueError("Unexpected official AGMARKNET response schema.")
                path.write_bytes(gzip.compress(response.content, mtime=0))
                break
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(3 * (attempt + 1))
    content = gzip.decompress(path.read_bytes())
    obj = json.loads(content)
    title = obj.get("title", "")
    if CROPS[crop] not in title or STATES[state] not in title:
        raise ValueError(
            "Official response title does not match the requested state and commodity."
        )
    dates = [x["arrivalDate"] for m in obj["markets"] for x in m["dates"]]
    if any((int(d[6:10]), int(d[3:5])) != (year, month) for d in dates):
        raise ValueError(
            "Government response contains dates outside the requested month."
        )
    return {
        "file": str(path.relative_to(ROOT)),
        "url": URL,
        "parameters": params,
        "sha256": digest(path),
        "response_sha256": hashlib.sha256(content).hexdigest(),
        "market_count": len(obj["markets"]),
        "market_days": len(dates),
        "bytes": path.stat().st_size,
        "retrieved_utc": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
        "verified_utc": datetime.now(UTC).isoformat(),
        "source": "Government of India AGMARKNET",
        "licence": "Government source attribution retained; redistribution terms require verification.",
    }


def official_jobs(cfg):
    return [
        (s, c, y, m)
        for s in cfg["states"]
        for c in cfg["commodities"]
        for y in range(int(cfg["start_date"][:4]), int(cfg["end_date"][:4]) + 1)
        for m in range(1, 13)
    ]


def download():
    """Cache all 2021–2023 monthly responses with a resumable audit manifest."""
    (ROOT / "data/raw/official").mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "data/raw/official_manifest.json"
    if manifest.exists():
        for record in json.loads(manifest.read_text()):
            if digest(ROOT / record["file"]) != record["sha256"]:
                raise ValueError(f"Cached source checksum mismatch: {record['file']}")
    jobs = official_jobs(config())
    records = []
    for job in jobs:
        for retry in range(3):
            try:
                records.append(fetch(job))
                break
            except SourceRateLimit as error:
                save_json(
                    ROOT / "reports/source_rate_limit.json",
                    {
                        "observed_utc": datetime.now(UTC).isoformat(),
                        "job": job,
                        "cooldown_seconds": error.seconds,
                        "headers": error.headers,
                        "retry": retry + 1,
                    },
                )
                if retry == 2:
                    raise
                print(
                    f"Government source rate limited; pausing {error.seconds} seconds before retrying the same request.",
                    flush=True,
                )
                time.sleep(error.seconds)
        if len(records) % 12 == 0 or len(records) == len(jobs):
            save_json(ROOT / "data/raw/official_manifest.json", records)
            print(
                f"Official AGMARKNET: {len(records)}/{len(jobs)} monthly files",
                flush=True,
            )
    save_json(
        ROOT / "data/raw/official_manifest.json",
        sorted(records, key=lambda x: x["file"]),
    )


def rows(path):
    """Yield market-day summaries; quantity totals are not repeated per variety."""
    s, c, _, _ = path.name.split("_")
    state = STATES[int(s)]
    crop = CROPS[int(c)]
    obj = json.loads(gzip.decompress(path.read_bytes()))
    for market in obj["markets"]:
        for day in market["dates"]:
            varieties = day["data"]
            prices = []
            amounts = []
            for v in varieties:
                price = pd.to_numeric(v.get("modalPrice"), errors="coerce")
                quantity = pd.to_numeric(v.get("arrivals"), errors="coerce")
                if pd.notna(price) and price > 0:
                    prices.append(float(price))
                    amounts.append(
                        max(0, float(quantity)) if pd.notna(quantity) else 0.0
                    )
            total = pd.to_numeric(day.get("total_arrivals"), errors="coerce")
            if not prices or pd.isna(total) or total < 0:
                continue
            modal = (
                sum(p * q for p, q in zip(prices, amounts)) / sum(amounts)
                if sum(amounts) > 0
                else float(pd.Series(prices).median())
            )
            yield {
                "state": state,
                "market": market["marketName"],
                "commodity": crop,
                "date": pd.Timestamp(
                    year=int(day["arrivalDate"][6:10]),
                    month=int(day["arrivalDate"][3:5]),
                    day=int(day["arrivalDate"][:2]),
                ),
                "arrivals": float(total),
                "modal_price": modal,
            }


def market_panel(cfg):
    """Freeze a training-only market selection then retain actual observations."""
    expected = [
        ROOT / f"data/raw/official/{s}_{c}_{y}_{m:02d}.json.gz"
        for s, c, y, m in official_jobs(cfg)
    ]
    paths = sorted(p for p in expected if p.exists())
    if len(paths) != len(expected):
        raise FileNotFoundError(
            f"Expected {len(expected)} complete monthly files; found {len(paths)}."
        )
    selection = (
        ROOT / "configs" / cfg.get("selected_market_file", "official_markets.json")
    )
    if selection.exists():
        selected = json.loads(selection.read_text())
    else:
        counts = {}
        for path in paths:
            if int(path.name.split("_")[2]) > 2022:
                continue
            for row in rows(path):
                if row["date"] >= pd.Timestamp("2022-07-01"):
                    continue
                key = (row["state"], row["market"], row["commodity"])
                counts[key] = counts.get(key, 0) + 1
        table = (
            pd.Series(counts)
            .unstack(fill_value=0)
            .reindex(columns=list(CROPS.values()), fill_value=0)
        )
        table["score"] = 10 * table.min(axis=1) + table.sum(axis=1)
        selected = []
        for state in [STATES[s] for s in cfg["states"]]:
            rank = table.loc[state].sort_values("score", ascending=False, kind="stable")
            selected.extend(
                [[state, m] for m in rank.head(cfg["markets_per_state"]).index]
            )
        save_json(selection, selected)
    chosen = set(map(tuple, selected))
    selected_rows = []
    total_rows = 0
    for path in paths:
        for row in rows(path):
            total_rows += 1
            if (row["state"], row["market"]) in chosen:
                selected_rows.append(row)
    observed = pd.DataFrame(selected_rows).drop_duplicates()
    keys = ["state", "market", "commodity", "date"]
    if observed.duplicated(keys).any():
        raise ValueError("Duplicate official market-day keys.")
    observed = observed.sort_values(keys).reset_index(drop=True)
    observed.to_parquet(
        ROOT / "data/processed/market_observations.parquet", index=False
    )
    panel = market_features(observed, cfg)
    published_days = sum(
        r["market_days"]
        for r in json.loads((ROOT / "data/raw/official_manifest.json").read_text())
    )
    save_json(
        ROOT / "reports/market_data_card.json",
        {
            "source": "Government of India AGMARKNET public API",
            "documentation": "https://agmarknet.gov.in/",
            "raw_archive_rows": total_rows,
            "published_market_days": published_days,
            "unusable_market_days": published_days - total_rows,
            "selected_markets": len(selected),
            "states": sorted(observed.state.unique()),
            "monthly_source_files": len(paths),
            "selected_real_observations": len(observed),
            "calendar_panel_rows": len(panel),
            "unobserved_calendar_rows": int((~panel.observed).sum()),
            "date_min": str(observed.date.min().date()),
            "date_max": str(observed.date.max().date()),
            "commodities": sorted(observed.commodity.unique()),
            "units": {"arrivals": "metric tonnes", "prices": "INR per quintal"},
            "imputation": "No source targets or prices imputed; missing calendar dates remain NaN.",
            "aggregation": "Use published total_arrivals once per market-day; arrivals-weighted variety modal price.",
            "selection": "Markets ranked on January 2021–June 2022 commodity completeness; frozen before test evaluation.",
            "coverage_by_state_commodity": {
                str(k): int(v)
                for k, v in observed.groupby(["state", "commodity"]).size().items()
            },
            "missingness": observed.isna().sum().to_dict(),
            "deviations": [
                "Market arrivals are supply inflows, not observed demand or warehouse inventory.",
                "Market observations and Delhivery trips cannot be linked to shared shipments.",
            ],
        },
    )
    return panel


if __name__ == "__main__":
    download()
