"""Generate the final official-source results report and research figures."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from scrc.common import ROOT, config


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def markdown_table(frame, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in frame.iterrows():
        vals = []
        for col in cols:
            value = row.get(col)
            vals.append(
                "—"
                if pd.isna(value)
                else f"{value:.4f}" if isinstance(value, float) else str(value)
            )
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    cfg = config()
    out = ROOT / "reports/figures"
    out.mkdir(parents=True, exist_ok=True)
    market = pd.read_parquet(ROOT / "experiments/market_replay.parquet")
    trips = pd.read_parquet(ROOT / "experiments/logistics_replay.parquet")
    metrics = pd.read_csv(ROOT / "experiments/market_results.csv")
    policies = pd.read_csv(ROOT / "experiments/logistics_results.csv")
    regions = pd.read_csv(ROOT / "experiments/market_results_by_region.csv")
    card = read_json("reports/market_data_card.json")
    lcard = read_json("reports/logistics_data_card.json")
    diag = read_json("experiments/causal_diagnostics.json")
    detection = pd.DataFrame(read_json("experiments/shift_detection.json"))
    events = [
        e
        for e in yaml.safe_load((ROOT / "configs/events.yaml").read_text())
        if ((market.date >= e["start"]) & (market.date <= e["end"])).any()
    ]
    overall = metrics.loc[metrics.window == "overall"]
    cols = ["method", "n", "mae", "rmse", "coverage", "mean_width", "acceptance"]
    adaptive = overall.loc[overall.method == "adaptive_conformal"].iloc[0]
    fixed = overall.loc[overall.method == "split_conformal"].iloc[0]
    rolling = overall.loc[overall.method == "rolling_mean"].iloc[0]
    selected = overall.loc[overall.method == "predictive_gate"].iloc[0]
    coverage_finding = (
        "Adaptive calibration met the nominal 90% target empirically."
        if adaptive.coverage >= 0.9
        else f"Adaptive calibration fell below the nominal 90% target by {100*(0.9-adaptive.coverage):.2f} percentage points."
    )
    gate_finding = (
        "The gate accepted every held-out prediction, so its ablations are identical here; this experiment shows no selective-prediction benefit from the gate."
        if selected.acceptance == 1
        else "The gate metrics concern its accepted subset and must be compared with its acceptance rate."
    )
    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    for name, kind in [
        ("Split conformal", "split"),
        ("Adaptive conformal", "adaptive"),
    ]:
        daily = (
            market.assign(
                covered=(market.arrivals >= market["lower_" + kind])
                & (market.arrivals <= market["upper_" + kind])
            )
            .groupby("date")
            .covered.mean()
        )
        ax.plot(daily.index, daily.rolling(14, min_periods=1).mean(), label=name)
    ax.axhline(0.9, color="black", linestyle="--", label="90% target")
    for event in events:
        ax.axvline(pd.Timestamp(event["start"]), color="grey", alpha=0.5, linewidth=0.8)
    ax.set(ylabel="14-day rolling mean daily coverage", ylim=(0, 1.02))
    ax.legend(loc="lower left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "coverage_under_shift.png", dpi=300)
    plt.close(fig)
    tuned = read_json("experiments/tuned_predictive_gate.json")
    fig, ax = plt.subplots(figsize=(7.1, 2.8))
    daily = market.groupby("date").shift_score.mean()
    ax.plot(daily.index, daily, label="Mean drift score")
    ax.axhline(
        tuned["tau_shift"],
        color="red",
        linestyle="--",
        label="Validation-selected threshold",
    )
    for event in events:
        ax.axvline(pd.Timestamp(event["start"]), color="grey", alpha=0.5)
    ax.set(ylabel="Shift score", ylim=(0, 1))
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "shift_score.png", dpi=300)
    plt.close(fig)
    sweep = pd.read_csv(ROOT / "experiments/threshold_sweep_validation.csv")
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.scatter(sweep.acceptance, sweep.coverage, s=22, color="#24768e")
    ax.axhline(0.9, color="black", linestyle="--")
    ax.set(
        xlabel="Validation prediction acceptance",
        ylabel="Accepted interval coverage",
        ylim=(0, 1.02),
    )
    fig.tight_layout()
    fig.savefig(out / "predictive_safety_acceptance.png", dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
    for arm, name in [(0, "Carting"), (1, "FTL")]:
        axes[0].hist(
            trips.loc[trips.treatment == arm, "propensity"],
            bins=20,
            range=(0, 1),
            alpha=0.6,
            label=name,
        )
    axes[0].axvspan(
        cfg["propensity_min"], cfg["propensity_max"], alpha=0.08, color="green"
    )
    axes[0].set(xlabel="Estimated P(FTL | covariates)", ylabel="Held-out trips")
    axes[0].legend()
    counts = trips.gate.value_counts().reindex(["GREEN", "AMBER", "RED"], fill_value=0)
    axes[1].bar(counts.index, counts.values, color=["green", "orange", "firebrick"])
    axes[1].set(ylabel="Trips", title="Evidence-gated actions")
    fig.tight_layout()
    fig.savefig(out / "causal_overlap_and_gate.png", dpi=300)
    plt.close(fig)
    overall[cols].to_csv(ROOT / "reports/main_forecast_table.csv", index=False)
    latex = [
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        "Method & N & MAE & RMSE & Coverage & Width & Acceptance " + r"\\",
        r"\hline",
    ]
    for _, row in overall.iterrows():
        values = [str(row["method"]).replace("_", r"\_"), str(int(row["n"]))] + [
            "--" if pd.isna(row[c]) else f"{row[c]:.4f}" for c in cols[2:]
        ]
        latex.append(" & ".join(values) + " " + r"\\")
    latex.extend([r"\hline", r"\end{tabular}"])
    (ROOT / "reports/main_forecast_table.tex").write_text(
        "\n".join(latex), encoding="utf-8"
    )
    descriptions = []
    for event in events:
        subset = metrics.loc[metrics.window == event["name"]]
        descriptions.append(
            f"### {event['name']}\n\nSource: [{event['name']}]({event['source']}). {event['note']}\n\n"
            + markdown_table(subset, cols)
        )
    proxy = (
        market.groupby("event")
        .agg(
            observed_rows=("arrivals", "size"),
            valid_proxy_rows=("shortage_proxy", "count"),
            shortage_pressure_rate=("shortage_proxy", "mean"),
        )
        .reset_index()
    )
    proxy.to_csv(ROOT / "experiments/shortage_proxy_descriptive.csv", index=False)
    gain = "Empirical overlap is insufficient for policy evaluation."
    cr = policies.loc[policies.method == "conservative_lower_bound_policy"]
    if len(cr):
        row = cr.iloc[0]
        gain = f"Conservative policy estimated gain: {row.estimated_gain_vs_carting:.4f} delay-factor units; date-cluster interval [{row.gain_ci_low:.4f}, {row.gain_ci_high:.4f}]. "
        gain += (
            "The interval includes zero."
            if row.gain_ci_low <= 0 <= row.gain_ci_high
            else "The interval excludes zero conditional on the fitted observational model."
        )
    report = f"""# Reliable Causal Decisions for Supply Chains Under Distribution Shift

Implementation and evaluation based on the supplied M.Tech plan. The download, preparation, forecasting, calibration, monitoring, conservative intervention estimation, gate, baselines, ablations, tables and figures are implemented. Operational stock-out reduction is not identifiable from these public datasets.

Measured result: split-conformal coverage {fixed.coverage:.2%}; adaptive coverage {adaptive.coverage:.2%}, with {100*(1-adaptive.mean_width/fixed.mean_width):.2f}% narrower mean intervals. {coverage_finding} {gate_finding} No causal automation benefit is established.

## Data actually used

**AGMARKNET:** {card['selected_real_observations']:,} real market-day-commodity observations; {card['selected_markets']} markets in {', '.join(card['states'])}; rice, wheat, onion and potato; {card['date_min']}–{card['date_max']}. Source: [Government AGMARKNET](https://agmarknet.gov.in/). All {card['monthly_source_files']} monthly responses required by this cohort and their checksums are retained locally. Source columns define arrivals in metric tonnes and prices in INR/quintal. Published daily totals are counted once rather than repeated for every variety. Calendar gaps add {card['unobserved_calendar_rows']:,} missing rows; these are never treated as observations or scored as zero.

**Delhivery:** {lcard['raw_rows']:,} scan records aggregated to {lcard['usable_trips']:,} distinct usable trips, September–October 2018. Source: [documented educational mirror](https://github.com/shekshavalipattan/Delhivery-Logistics-Data-Pipeline-and-Feature-Engineering). Cumulative scans are aggregated by OD-leg maxima, with legs ordered chronologically before trip-level endpoints are selected.

The original four-state rice/wheat archive experiment is preserved in `runs/archive_2014_2016/`; it is not mixed into this official-source experiment. The attempted four-state 2021–2023 download stopped at 184/576 monthly files because the government service continued returning HTTP 429 after extended cooldowns. The complete Uttar Pradesh cohort has 144/144 files. Incomplete Karnataka files and unavailable West Bengal/Maharashtra files are excluded. India Data Portal's public page was readable but its data service required a signed-in session. This is a documented regional limitation relative to the four-region plan, not a claim of four-region validation. Partial CEDA downloads are also excluded.

## Experimental protocol

Market selection uses January 2021–June 2022 completeness only. Training ends June 2022, calibration covers July–December 2022, validation covers January–June 2023, and the held-out test covers July–December 2023. Calendar lags and rolling means use strictly earlier dates. Test outcomes update online conformal calibration only after their predictions are logged; they never tune model or gate settings. Actual eligible counts appear in `experiments/market_splits.json`.

The forecaster is fixed LightGBM on log arrivals. Residuals are normalized by a prior rolling mean and calibrated by state/commodity. Adaptive calibration is a clipped, batched ACI-inspired implementation; original ACI theoretical guarantees do not automatically transfer. Drift combines feature KS effect sizes and past miscoverage. The predictive gate is selected using validation data only. Selective prediction acceptance is not successful-action precision.

Logistics evaluation has separate chronological partitions. Training trips unresolved at the next boundary are purged. T-learner outcome regressions and a propensity model support a conservative lower-bound route selector. Model uncertainty uses {cfg['causal_bootstrap_models']} training-date bootstrap refits; held-out policy loss intervals use {cfg['bootstrap_repeats']} date-cluster resamples. Only {diag['independent_test_dates']} independent held-out dates are available. Carting is an observed reference route, not a verified no-action option.

## Overall measured forecast results

{markdown_table(overall,cols)}

Adaptive coverage is **{adaptive.coverage:.2%}** on {int(adaptive['n']):,} held-out observations. Its mean width is {adaptive.mean_width:.4f} tonnes versus {fixed.mean_width:.4f} for split conformal. The rolling-mean baseline has MAE {rolling.mae:.4f} and RMSE {rolling.rmse:.4f}; LightGBM has MAE {adaptive.mae:.4f} and RMSE {adaptive.rmse:.4f}. Comparisons may favour different methods on different metrics; no universal superiority is asserted. State/commodity coverage ranges from {regions.coverage.min():.2%} to {regions.coverage.max():.2%}.

The validation-selected predictive gate accepts {selected.acceptance:.2%} of test predictions. Its no-shift and no-width ablations are included in the table; changes in accepted-subset metrics must be interpreted alongside acceptance rates.

## Real event windows

{chr(10).join(descriptions)}

{markdown_table(detection,['event','detected','delay_days','outside_window_alert_rate'])}

The frozen validation-selected threshold produced alerts in {int(detection.detected.sum())}/{len(detection)} annotated event windows. Missing detection delays mean that no alert occurred in that window; they are not zero-day detections.

Policy dates annotate possible changes; they are not exhaustive drift ground truth or proof of a causal effect. `experiments/shift_detection.json` reports event alerts and delays. Outside-window alerts cannot automatically be labelled false alarms. Generic Rice includes varieties potentially unaffected by a specific export restriction.

## Observational logistics evaluation

{markdown_table(policies,['method','n','mean_dr_delay_loss','estimated_gain_vs_carting','gain_ci_low','gain_ci_high','ftl_recommendation_fraction','status'])}

{gain} Overlap includes {diag['test_overlap_rows']:,}/{diag['test_rows']:,} held-out trips ({diag['test_overlap_fraction']:.2%}). These estimates condition on unverified observational assumptions. Aggregate OSRM distance may be treatment-dependent, and shipment load/fleet constraints are unobserved. Placebo, subset, omitted-distance and additive-bias diagnostics are in `experiments/causal_diagnostics.json`; they do not prove identification.

The causal gate withholds automatic actions because verified pre-dispatch confounders and causal identification are absent. The all-red result is a legitimate evidence rejection, not proof of improved supply-chain outcomes. Zero action cost is a declared delay-factor scenario; monetary savings are not inferred.

## Shortage-pressure proxy

{markdown_table(proxy,['event','observed_rows','valid_proxy_rows','shortage_pressure_rate'])}

This descriptive arrivals/price proxy is not a measured stock-out or fill rate. No counterfactual improvement in this proxy can be attributed to Delhivery route choices.

## Limits on the original plan

Public AGMARKNET markets and Delhivery shipments have no shared shipment identifiers and cover different years. They cannot be validly joined into one causal warehouse simulation. Static safety-stock costs, actual fill rates, transfers, supplier switching, inventory savings and green-action success are unobservable. They are marked unavailable rather than fabricated. There is no live ERP/WMS execution.

The implementation uses transparent scikit-learn/LightGBM estimators and explicit doubly robust scores rather than EconML/DoWhy. This software choice does not change the need for identification assumptions. The source supports Python 3.11+; the final isolated environment version is recorded in `experiments/run_manifest.json`.

## Reproduction

`python run.py scrc.pipeline` performs source acquisition (using immutable cache), preparation, evaluation, tests and delivery packaging. `run.ps1` launches it using the project environment. `scripts/verify_reproducibility.py` repeats the numerical evaluation; its actual result is in `reports/reproducibility.json`. Raw/processed inputs stay local; the code ZIP includes source, aggregate metrics and figures. See `README.md` for environment setup.

## Method references

[Gibbs and Candès, ACI, NeurIPS 2021](https://arxiv.org/abs/2106.00170); [Dudík, Langford and Li, doubly robust policy evaluation, ICML 2011](https://arxiv.org/abs/1103.4601); [Künzel et al., metalearners, PNAS 2019](https://doi.org/10.1073/pnas.1804597116).
"""
    (ROOT / "reports/EVALUATION.md").write_text(report, encoding="utf-8")
    fragment = rf"""\section{{Experimental Setup}}
We evaluate {card['selected_real_observations']:,} observed AGMARKNET market-day records for rice, wheat, onion and potato from 2021--2023. Training, calibration, validation and testing are separated chronologically. A separate observational logistics experiment uses {lcard['usable_trips']:,} Delhivery trips; it is not joined to the market records.
\section{{Results and Limitations}}
Adaptive interval coverage was {100*adaptive.coverage:.2f}\% on {int(adaptive['n']):,} held-out observations, with mean width {adaptive.mean_width:.4f} tonnes. Split-conformal coverage was {100*fixed.coverage:.2f}\%, with mean width {fixed.mean_width:.4f} tonnes. Predictive gate acceptance was {100*selected.acceptance:.2f}\%. These are empirical observations and do not establish conditional coverage guarantees under arbitrary shift.
The causal gate withheld automatic interventions because the public logistics data do not establish causal identification. Actual inventory outcomes and monetary intervention costs are absent. The evaluation therefore does not claim measured stock-out reduction or operational cost savings.
"""
    (ROOT / "reports/EXPERIMENTAL_SETUP_AND_RESULTS.tex").write_text(
        fragment, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
