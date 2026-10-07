"""Inspect public CEDA website assets to document its public data interface."""

import re
from pathlib import Path

import requests

base = "https://agmarknet.ceda.ashoka.edu.in"
out = Path(__file__).resolve().parents[1] / "reports" / "source_inspection.txt"
lines = []
html = requests.get(base, timeout=40).text
for src in re.findall(r'<script src="([^"]+)"', html):
    content = requests.get(base + src, timeout=40).text
    for pattern in [r".{0,350}/api/.{0,700}", r".{0,250}calculation_type.{0,500}"]:
        lines.extend(re.findall(pattern, content))
out.write_text("\n".join(lines), encoding="utf-8")
print(out.read_text(encoding="utf-8")[:24000])
