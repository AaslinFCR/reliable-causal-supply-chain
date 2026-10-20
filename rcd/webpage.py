"""Build a self-contained results web page (report.html) from results/ and figures/."""
import base64, html, json, os, re, time
import pandas as pd

CSS = """
:root{--bg:#f7f7f4;--card:#fff;--ink:#16161a;--muted:#5d5c58;--line:#e4e3de;--accent:#2a78d6;--good:#1a8f5f;--bad:#c4472a}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--card:#1d1d21;--ink:#ececea;--muted:#a3a29d;--line:#33333a;--accent:#6aa6f0;--good:#4cc28f;--bad:#f0805f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:32px 16px 8px;max-width:1100px;margin:auto}h1{margin:0 0 4px;font-size:26px}
.sub{color:var(--muted);font-size:14px}nav{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);z-index:2}
nav div{max-width:1100px;margin:auto;padding:8px 16px;display:flex;gap:16px;overflow-x:auto;white-space:nowrap}
nav a{color:var(--muted);text-decoration:none;font-size:14px}nav a:hover{color:var(--accent)}
main{max-width:1100px;margin:auto;padding:16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:8px 0 24px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi .v{font-size:24px;font-weight:650}.kpi .l{color:var(--muted);font-size:13px}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin:0 0 20px}
h2{margin:0 0 6px;font-size:19px}section p{color:var(--muted);margin:0 0 14px}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600}tr.hl td{background:color-mix(in srgb,var(--accent) 12%,transparent);font-weight:600}
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px;margin-top:16px}
figure{margin:0}figure img{width:100%;border-radius:8px;background:#fff;border:1px solid var(--line)}
figcaption{color:var(--muted);font-size:13px;margin-top:4px}footer{color:var(--muted);font-size:13px;text-align:center;padding:12px 16px 32px}
"""


def _img(figdir, name, caption):
    p = os.path.join(figdir, name + ".png")
    if not os.path.exists(p):
        return ""
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    return (f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{b64}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


def _table(df, highlight=None, single_seed=False):
    df = df.dropna(axis=1, how="all").copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:,.2f}")
        elif single_seed:  # "x ± 0.00" carries no information with one seed
            df[c] = df[c].astype(str).str.replace(r"\s*±.*$", "", regex=True)
    cols = [html.escape(str(c)) for c in df.columns]
    rows = []
    for _, r in df.iterrows():
        cls = ' class="hl"' if highlight and re.search(highlight, str(r.iloc[0])) else ""
        rows.append(f"<tr{cls}>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>")
    return ('<div class="tw"><table><thead><tr>' + "".join(f"<th>{c}</th>" for c in cols)
            + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def _csv(outdir, name):
    return pd.read_csv(os.path.join(outdir, name), encoding="utf-8-sig")


def make(outdir="results", figdir="figures", path="report.html"):
    summ = json.load(open(os.path.join(outdir, "summary.json")))
    one = summ.get("n_seeds", 1) == 1
    fc = _csv(outdir, "table_forecast.csv")
    cf = _csv(outdir, "table_conformal.csv")
    ca = _csv(outdir, "table_causal.csv").rename(columns={"Unnamed: 0": "method"})
    cl = _csv(outdir, "table_closed_loop.csv")

    def num(s):
        return float(str(s).split("±")[0])

    sel = summ.get("selected", "-")
    sel = ", ".join(dict.fromkeys(sel)) if isinstance(sel, list) else str(sel)  # one entry per seed
    prop = cl[cl["policy"].str.startswith("Proposed")].iloc[0]
    best_fc = max(fc["Accuracy"].map(num))
    cusum = summ.get("detection", {}).get("CUSUM", {})
    kpis = [
        (f"{num(prop['saving_vs_static_%']):+.1f}%", "cost saving vs static safety stock (proposed policy)"),
        (f"{num(prop['fill_rate_%']):.2f}%", "fill rate (proposed policy)"),
        (f"{best_fc:.1f}%", "best 7-day forecast accuracy"),
        (f"{cusum.get('detect_rate', float('nan')):.0f}%", "disruptions detected by CUSUM"),
        (sel, "causal learner selected by OPE"),
    ]

    det = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)}
                        for k, v in summ.get("detection", {}).items()}).T.reset_index()
    det = det.rename(columns={"index": "monitor"})
    keep = ["method"] + [c for c in cf.columns if c.endswith(("coverage", "rel_width")) or c == "min_series_cov"]
    cf = cf[[c for c in keep if c in cf.columns]]

    sections = [
        ("closed", "Closed-loop results (2025 test year)",
         "Total cost, service level and decision regret of each replenishment policy. Highlighted: the proposed reliability-gated causal policy.",
         _table(cl, "^Proposed", one), ["fig_cost"]),
        ("forecast", "Demand forecasting",
         "7-day demand forecast error by model (lower MAE/WAPE is better).",
         _table(fc, None, one), ["fig_demand"]),
        ("conformal", "Uncertainty intervals (conformal, 90% target)",
         "Coverage and relative width of prediction intervals, overall and during shift periods.",
         _table(cf), ["fig_coverage"]),
        ("shift", "Shift detection",
         "How quickly and reliably each monitor flags real disruptions.",
         _table(det) if len(det) else "", ["fig_cusum"]),
        ("causal", "Causal decision quality",
         "Accuracy of choosing the right intervention, and regret vs the oracle, on held-out planner logs.",
         _table(ca, "^(" + "|".join(map(re.escape, sel.split(", "))) + ")$"), ["fig_risk_coverage", "fig_zones"]),
    ]
    caps = {"fig_cost": "Cost by policy", "fig_demand": "Demand and forecast", "fig_coverage": "Interval coverage over time",
            "fig_cusum": "CUSUM shift monitor", "fig_risk_coverage": "Risk vs coverage of the reliability gate",
            "fig_zones": "GREEN / AMBER / RED decision zones"}

    body = "".join(
        f'<section id="{sid}"><h2>{title}</h2><p>{desc}</p>{tbl}'
        f'<div class="figs">{"".join(_img(figdir, f, caps[f]) for f in figs)}</div></section>'
        for sid, title, desc, tbl, figs in sections)
    nav = "".join(f'<a href="#{sid}">{title.split(" (")[0]}</a>' for sid, title, *_ in sections)
    tiles = "".join(f'<div class="kpi"><div class="v">{html.escape(str(v))}</div><div class="l">{l}</div></div>' for v, l in kpis)
    seeds = ", ".join(str(s) for s in (summ.get("seeds") if isinstance(summ.get("seeds"), list) else [summ.get("seeds")]))

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Supply Chain Results</title><style>{CSS}</style></head>
<body><header><h1>Reliable Causal Decisions for Supply Chains Under Shift</h1>
<div class="sub">{summ.get('n_seeds')} simulated world(s) &middot; seeds {seeds} &middot; generated {time.strftime('%Y-%m-%d %H:%M')}</div></header>
<nav><div>{nav}</div></nav><main><div class="kpis">{tiles}</div>{body}</main>
<footer>All data are simulated; costs are illustrative planning units (INR-like), not measured company costs.</footer></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return os.path.abspath(path)
