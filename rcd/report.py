"""Aggregate multi-seed results into tables (CSV/JSON) and paper figures."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from . import config as C, simulator as SIM

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9890", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.5, "legend.frameon": False, "savefig.dpi": 300})


def ms(vals, f="{:.2f}"):
    v = np.asarray(vals, dtype=float)
    return (f + " ± " + f).format(np.nanmean(v), np.nanstd(v, ddof=1) if len(v) > 1 else 0.0)


def tci(diffs):
    d = np.asarray(diffs, float); n = len(d)
    if n < 2:
        return d.mean(), np.nan, np.nan
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return d.mean(), d.mean() - h, d.mean() + h


def make(out, outdir="results", figdir="figures"):
    os.makedirs(outdir, exist_ok=True); os.makedirs(figdir, exist_ok=True)
    summ = {"n_seeds": len(out), "seeds": [r["seed"] for r in out]}
    # ---- forecasting
    models = ["naive", "ma28", "RF", "GBDT"]
    rows = []
    for m in models:
        rows.append({"model": m, **{k: ms([r["prepare"]["point"][m][k] for r in out]) for k in
                                    ["MAE", "RMSE", "WAPE", "Accuracy", "Bias", "R2"]}})
    pd.DataFrame(rows).to_csv(f"{outdir}/table_forecast.csv", index=False, encoding="utf-8-sig")
    summ["forecast"] = {m: {k: float(np.mean([r["prepare"]["point"][m][k] for r in out]))
                            for k in ["MAE", "RMSE", "WAPE", "Accuracy", "Bias", "R2"]} for m in models}
    summ["forecast_sd"] = {m: {k: float(np.std([r["prepare"]["point"][m][k] for r in out], ddof=1))
                               for k in ["WAPE", "Accuracy"]} for m in models}
    # ---- conformal
    meths = ["split", "mondrian", "aci", "waci", "sacp"]
    rows = []; summ["conformal"] = {}
    for m in meths:
        e = {}
        for lab in ["all", "shift", "normal"]:
            for k in ["coverage", "rel_width", "int_score"]:
                e[f"{lab}_{k}"] = float(np.mean([r["prepare"]["conformal"][m][lab][k] for r in out]))
                e[f"{lab}_{k}_sd"] = float(np.std([r["prepare"]["conformal"][m][lab][k] for r in out], ddof=1))
        e["min_series_cov"] = float(np.mean([r["prepare"]["conformal"][m]["min_series_cov"] for r in out]))
        summ["conformal"][m] = e; rows.append({"method": m, **e})
    pd.DataFrame(rows).to_csv(f"{outdir}/table_conformal.csv", index=False, encoding="utf-8-sig")
    # ---- detection
    summ["detection"] = {}
    for d in ["CUSUM", "KS-14"]:
        summ["detection"][d] = {k: float(np.mean([r["prepare"]["detection"][d][k] for r in out]))
                                for k in ["detect_rate", "median_delay", "mean_delay", "false_alarm", "n_events"]}
        kinds = {}
        for r in out:
            for k, v in r["prepare"]["detection"][d]["by_kind"].items():
                kinds.setdefault(k, []).append(v)
        summ["detection"][d]["by_kind"] = {k: float(np.mean(v)) for k, v in kinds.items()}
    summ["detection"]["Transit-overdue"] = {k: float(np.mean([r["prepare"]["detection"]["Transit-overdue"][k]
                                                              for r in out])) for k in ["recall_days", "false_alarm"]}
    # ---- causal (offline)
    lnames = ["Always none", "Logged planner", "Naive", "S-learner", "T-learner", "DR-learner", "Oracle"]
    summ["causal"] = {}
    for l in lnames:
        summ["causal"][l] = {k: float(np.nanmean([r["decision_quality"][l][k] for r in out]))
                             for k in ["pehe", "decision_acc", "regret", "harmful", "value"]}
        summ["causal"][l]["regret_sd"] = float(np.std([r["decision_quality"][l]["regret"] for r in out], ddof=1))
        summ["causal"][l]["acc_sd"] = float(np.std([r["decision_quality"][l]["decision_acc"] for r in out], ddof=1))
    summ["selected"] = [r["selected"] for r in out]
    summ["ope"] = {k: float(np.mean([r["ope"][k] for r in out])) for k in out[0]["ope"]}
    summ["cf_coverage"] = {a: float(np.mean([r["cf_coverage"][a] for r in out])) for a in C.ACTIONS}
    summ["n_logs"] = out[0]["n_logs"]; summ["logged_shares"] = np.mean([r["logged_shares"] for r in out], 0).tolist()
    pd.DataFrame(summ["causal"]).T.to_csv(f"{outdir}/table_causal.csv", encoding="utf-8-sig")
    # ---- closed loop
    pols = list(out[0]["closed"])
    keys = ["total_cost", "holding", "shortage", "action", "fill_rate", "stockout_days", "regret", "harmful",
            "decision_acc", "regret_shift", "regret_normal"]
    summ["closed"] = {}
    base = "Static safety stock (MA)"; prop = "Proposed: gated (amber approved)"
    for p in pols:
        e = {k: float(np.mean([r["closed"][p][k] for r in out])) for k in keys}
        e.update({k + "_sd": float(np.std([r["closed"][p][k] for r in out], ddof=1)) for k in keys})
        sav = [100 * (1 - r["closed"][p]["total_cost"] / r["closed"][base]["total_cost"]) for r in out]
        e["saving_vs_static"] = float(np.mean(sav)); e["saving_vs_static_sd"] = float(np.std(sav, ddof=1))
        diff = [r["closed"][prop]["total_cost"] - r["closed"][p]["total_cost"] for r in out]
        m_, lo, hi = tci(diff)
        e["prop_minus_this"] = [m_ / 1e5, lo / 1e5, hi / 1e5]
        e["prop_wins"] = int(sum(d < 0 for d in diff))
        e["act_share"] = {a: float(np.mean([r["closed"][p]["act_share"][a] for r in out])) for a in C.ACTIONS}
        if "zones" in out[0]["closed"][p]:
            e["zones"] = {z: {k: float(np.nanmean([r["closed"][p]["zones"][z][k] for r in out]))
                              for k in out[0]["closed"][p]["zones"][z]} for z in ["green", "amber", "red"]}
            e["auto_action_precision"] = float(np.nanmean([r["closed"][p]["auto_action_precision"] for r in out]))
            e["auto_actions"] = float(np.mean([r["closed"][p]["auto_actions"] for r in out]))
            e["escalated"] = float(np.mean([r["closed"][p]["escalated"] for r in out]))
        summ["closed"][p] = e
    rows = [{"policy": p, "total_cost_lakh": ms([r["closed"][p]["total_cost"] / 1e5 for r in out]),
             "saving_vs_static_%": ms([100 * (1 - r["closed"][p]["total_cost"] / r["closed"][base]["total_cost"]) for r in out], "{:.1f}"),
             "fill_rate_%": ms([r["closed"][p]["fill_rate"] for r in out]),
             "stockout_days_%": ms([r["closed"][p]["stockout_days"] for r in out]),
             "regret_INR": ms([r["closed"][p]["regret"] for r in out], "{:.0f}"),
             "harmful_%": ms([r["closed"][p]["harmful"] for r in out], "{:.1f}")} for p in pols]
    pd.DataFrame(rows).to_csv(f"{outdir}/table_closed_loop.csv", index=False, encoding="utf-8-sig")
    # ---- gate ablation
    summ["gate_ablation"] = {k: {kk: float(np.nanmean([r["gate_ablation"][k][kk] for r in out]))
                                 for kk in out[0]["gate_ablation"][k]} for k in out[0]["gate_ablation"]}
    summ["gate_thresholds"] = {k: float(np.mean([r["gate_thresholds"][k] for r in out])) for k in out[0]["gate_thresholds"]}
    json.dump(summ, open(f"{outdir}/summary.json", "w"), indent=1, default=float)
    figures(out, summ, figdir)
    return summ


def figures(out, summ, figdir):
    r0 = out[0]
    # Fig: demand example with events (seed 0 world)
    w = SIM.generate(r0["seed"])
    s = C.REGIONS.index("Mumbai") * len(C.PRODUCTS) + C.PRODUCTS.index("biscuits")
    fig, ax = plt.subplots(figsize=(3.5, 1.8))
    ax.plot(w["dates"], w["demand"][s], color=BLUE, lw=0.6)
    tr = w["shift_dem"][s]
    ax.fill_between(w["dates"], 0, w["demand"][s].max() * 1.05, where=tr, color=ORANGE, alpha=0.18, lw=0,
                    label="ground-truth demand shift")
    for y in [2023, 2024, 2025]:
        ax.axvline(pd.Timestamp(f"{y}-01-01"), color=GRAY, lw=0.6, ls=":")
    ax.set_ylabel("daily demand (units)"); ax.set_title("Mumbai - biscuits", fontsize=8, color=INK)
    ax.legend(loc="upper left", fontsize=6.5)
    fig.tight_layout(); fig.savefig(f"{figdir}/fig_demand.pdf"); fig.savefig(f"{figdir}/fig_demand.png"); plt.close(fig)
    # Fig: rolling coverage
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    dts = r0["prepare"]["dates_eval"]
    for m, col, lab in [("split", GRAY, "Split CP"), ("aci", ORANGE, "ACI"), ("sacp", BLUE, "SC-ACI (ours)")]:
        ax.plot(dts, 100 * np.asarray(r0["prepare"]["cov_roll"][m]), color=col, lw=1.2, label=lab)
    ax.axhline(90, color=INK, lw=0.6, ls="--")
    ax.set_ylabel("28-day rolling coverage (%)"); ax.set_ylim(60, 100)
    ax.legend(ncol=3, fontsize=6.5, loc="lower left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout(); fig.savefig(f"{figdir}/fig_coverage.pdf"); fig.savefig(f"{figdir}/fig_coverage.png"); plt.close(fig)
    # Fig: CUSUM example
    cs, truth = r0["prepare"]["cusum_example"]
    t25 = np.where(w["dates"].year == 2025)[0]
    fig, ax = plt.subplots(figsize=(3.5, 1.8))
    ax.plot(w["dates"][t25], cs[s, t25], color=BLUE, lw=0.9, label="Mumbai biscuits, 2025")
    ax.fill_between(w["dates"][t25], 0, np.nanmax(cs[s, t25]) * 1.05, where=truth[s, t25], color=ORANGE,
                    alpha=0.18, lw=0, label="ground-truth disruption")
    ax.axhline(1, color=INK, lw=0.6, ls="--")
    ax.set_ylabel("CUSUM / threshold"); ax.legend(fontsize=6.5, loc="upper left", ncol=2)
    ax.set_ylim(0, np.nanmax(cs[s, t25]) * 1.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout(); fig.savefig(f"{figdir}/fig_cusum.pdf"); fig.savefig(f"{figdir}/fig_cusum.png"); plt.close(fig)
    # Fig: closed-loop cost
    pols = list(summ["closed"])
    short = {"Static safety stock (MA)": "Static SS", "ML point forecast": "ML point", "Rule-based planner": "Rule planner",
             "Conformal rule (no causal)": "Conformal rule", "Causal, no gate": "Causal, no gate",
             "Proposed: gated (amber approved)": "Gated (ours)", "Proposed: gated (strict)": "Gated strict (ours)",
             "Oracle (myopic)": "Oracle"}
    vals = [np.array([r["closed"][p]["total_cost"] / 1e5 for r in out]) for p in pols]
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    cols = [BLUE if "ours" in short[p] else (AQUA if p.startswith("Oracle") else GRAY) for p in pols]
    y = np.arange(len(pols))[::-1]
    ax.barh(y, [v.mean() for v in vals], xerr=[v.std(ddof=1) if len(v) > 1 else 0 for v in vals], color=cols,
            height=0.62, error_kw=dict(lw=0.7, ecolor=INK, capsize=1.5))
    ax.set_yticks(y); ax.set_yticklabels([short[p] for p in pols]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("total 2025 cost (lakh units), mean ± SD")
    for yy, v in zip(y, vals):
        ax.text(v.mean() + (v.std(ddof=1) if len(v) > 1 else 0) + 0.6, yy, f"{v.mean():.1f}", va="center", fontsize=6.5, color=MUTED)
    ax.set_xlim(0, max(v.mean() for v in vals) * 1.18)
    fig.tight_layout(); fig.savefig(f"{figdir}/fig_cost.pdf"); fig.savefig(f"{figdir}/fig_cost.png"); plt.close(fig)
    # Fig: risk-coverage
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    cov = 100 * np.asarray(out[0]["risk_coverage"]["coverage"])
    R = np.array([r["risk_coverage"]["regret"] for r in out])
    ax.plot(cov, R.mean(0), color=BLUE, lw=1.4, marker="o", ms=3, label="ranked by gate signals")
    ax.fill_between(cov, R.min(0), R.max(0), color=BLUE, alpha=0.15, lw=0, label="min-max over seeds")
    ax.axhline(np.mean([r["risk_coverage"]["all_regret"] for r in out]), color=GRAY, ls="--", lw=0.8,
               label="all decisions (no selection)")
    ax.set_xlabel("decisions auto-executed (%)"); ax.set_ylabel("mean regret (units)")
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{figdir}/fig_risk_coverage.pdf"); fig.savefig(f"{figdir}/fig_risk_coverage.png"); plt.close(fig)
    # Fig: zone regret
    z = summ["closed"]["Proposed: gated (amber approved)"]["zones"]
    fig, ax = plt.subplots(figsize=(3.5, 1.8))
    xs = np.arange(3); wdt = 0.36
    ax.bar(xs - wdt / 2 - 0.01, [z[k]["rec_regret"] for k in ["green", "amber", "red"]], wdt, color=BLUE, label="causal recommendation")
    ax.bar(xs + wdt / 2 + 0.01, [z[k]["fb_regret"] for k in ["green", "amber", "red"]], wdt, color=ORANGE, label="conformal fallback rule")
    ax.set_xticks(xs); ax.set_xticklabels([f"{k.capitalize()} ({z[k]['share']:.0f}%)" for k in ["green", "amber", "red"]])
    ax.set_ylabel("mean regret (units)"); ax.grid(axis="x", visible=False); ax.legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(f"{figdir}/fig_zones.pdf"); fig.savefig(f"{figdir}/fig_zones.png"); plt.close(fig)
