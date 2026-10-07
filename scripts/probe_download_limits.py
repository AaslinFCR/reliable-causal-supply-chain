from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

base = "https://agmarknet.ceda.ashoka.edu.in/api/"


def probe(s):
    r = requests.get(base + f"districts?state_id={s}", timeout=40)
    r.raise_for_status()
    print(s, r.text, flush=True)
    Path(f"data/raw/districts_{s}.json").write_text(r.text, encoding="utf-8")


with ThreadPoolExecutor(max_workers=4) as p:
    list(p.map(probe, [9, 29, 19, 27]))
payload = {
    "state_id": 9,
    "commodity_id": 1,
    "district_id": 140,
    "start_date": "2021-01-01",
    "end_date": "2021-03-31",
    "calculation_type": "d",
    "chart_type": "datadownload",
    "data_type": "price",
}
for e in ["prices", "quantities"]:
    r = requests.post(base + e, json=payload, timeout=60)
    print(e, r.status_code, r.text[:1000], flush=True)
    if r.ok:
        d = r.json()["data"]
        print(
            "rows",
            len(d),
            "dates",
            min(x["t"] for x in d),
            max(x["t"] for x in d),
            flush=True,
        )
