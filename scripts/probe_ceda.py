from pathlib import Path

import requests

base = "https://agmarknet.ceda.ashoka.edu.in/api/"
for endpoint in ["states", "commodities"]:
    r = requests.get(base + endpoint, timeout=40)
    r.raise_for_status()
    Path("data/raw/" + endpoint + ".json").write_text(r.text, encoding="utf-8")
    print(endpoint, r.text[:4500], flush=True)
payload = {
    "state_id": 9,
    "commodity_id": 1,
    "district_id": 0,
    "start_date": "2022-01-01",
    "end_date": "2022-01-31",
    "calculation_type": "d",
    "chart_type": "datadownload",
    "data_type": "price",
}
for endpoint in ["prices", "quantities"]:
    r = requests.post(base + endpoint, json=payload, timeout=90)
    print(endpoint, r.status_code, r.text[:6500], flush=True)
    Path("reports/probe_" + endpoint + ".json").write_text(r.text, encoding="utf-8")
