"""Inspect the public official AGMARKNET schema with normal site headers."""

import json
from pathlib import Path

import requests

base = "https://api.agmarknet.gov.in/v1/"
session = requests.Session()
session.headers.update(
    {
        "User-Agent": "Mozilla/5.0",
        "Origin": "https://agmarknet.gov.in",
        "Referer": "https://agmarknet.gov.in/",
        "Accept": "application/json",
    }
)
r = session.get(base + "daily-price-arrival/filters", timeout=60)
r.raise_for_status()
d = r.json()
Path("data/raw/official_filters.json").write_text(json.dumps(d), encoding="utf-8")
print("states", d["data"]["state_data"], flush=True)
print(
    "commodities",
    [
        x
        for x in d["data"]["cmdt_data"]
        if x["cmdt_name"] in ["Rice", "Wheat", "Onion", "Potato"]
    ],
    flush=True,
)
r = session.get(
    base + "prices-and-arrivals/date-wise/specific-commodity",
    params={"year": 2023, "month": 12, "stateId": 9, "commodityId": 1},
    timeout=90,
)
print(r.status_code, r.text[:6500], flush=True)
Path("reports/official_probe.json").write_text(r.text, encoding="utf-8")
