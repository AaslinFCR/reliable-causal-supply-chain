"""Create publication-size figures and a report from actual experiment outputs."""

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scrc.common import ROOT


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main():
    """Regenerate plots and Markdown/LaTeX results without hand-entered results."""
    from scrc.common import config

    if config().get("market_source") == "official":
        from scrc.eval.final_report import main as final_report

        return final_report()
    out = ROOT / "reports/figures"
    out.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(ROOT / "experiments/market_results.csv")
    logistics = pd.read_csv(ROOT / "experiments/logistics_results.csv")
    replay = pd.read_parquet(ROOT / "experiments/market_replay.parquet")
    causal = pd.read_parquet(ROOT / "experiments/logistics_replay.parquet")
    card = read_json("reports/market_data_card.json")
    regional_results = pd.read_csv(ROOT / "experiments/market_results_by_region.csv")
    lcard = read_json("reports/logistics_data_card.json")
    diag = read_json("experiments/causal_diagnostics.json")
    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    for label, lo, hi in [
        ("Split", "lower_split", "upper_split"),
        ("Adaptive", "lower_adaptive", "upper_adaptive"),
    ]:
        daily = (
            replay.assign(
                covered=(replay.arrivals >= replay[lo])
                & (replay.arrivals <= replay[hi])
            )
            .groupby("date")
            .covered.mean()
        )
        ax.plot(daily.index, daily.rolling(14, min_periods=1).mean(), label=label)
    ax.axhline(0.9, color="black", linestyle="--", linewidth=0.8, label="90% target")
    ax.axvspan(
        pd.Timestamp("2016-11-09"),
        pd.Timestamp("2016-12-08"),
        alpha=0.12,
        color="orange",
        label="Demonetisation window",
    )
    ax.set(ylabel="14-day rolling daily coverage", ylim=(0, 1.02))
    ax.legend(loc="lower left", ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "coverage_under_shift.png", dpi=300)
    plt.close(fig)
    daily = replay.groupby("date").shift_score.mean()
    tuned = read_json("experiments/tuned_predictive_gate.json")
    fig, ax = plt.subplots(figsize=(7.1, 2.8))
    ax.plot(daily.index, daily, label="Mean feature / past-coverage drift")
    ax.axhline(
        tuned["tau_shift"],
        color="red",
        linestyle="--",
        label="Validation-selected predictive threshold",
    )
    ax.axvspan(
        pd.Timestamp("2016-11-09"),
        pd.Timestamp("2016-12-08"),
        alpha=0.12,
        color="orange",
    )
    ax.set(ylabel="Shift score", ylim=(0, 1))
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "shift_score.png", dpi=300)
    plt.close(fig)
    sweep = pd.read_csv(ROOT / "experiments/threshold_sweep_validation.csv")
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.scatter(
        sweep.acceptance, sweep.coverage, c=sweep.tau_shift, cmap="viridis", s=22
    )
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
            causal.loc[causal.treatment == arm, "propensity"],
            bins=np.linspace(0, 1, 21),
            alpha=0.6,
            label=name,
        )
    axes[0].axvspan(0.1, 0.9, alpha=0.08, color="green")
    axes[0].set(xlabel="Estimated P(FTL | covariates)", ylabel="Held-out trips")
    axes[0].legend()
    counts = causal.gate.value_counts().reindex(["GREEN", "AMBER", "RED"], fill_value=0)
    axes[1].bar(counts.index, counts.values, color=["#24835b", "#db9e2f", "#bb4242"])
    axes[1].set(ylabel="Trips", title="Causal gate: evidence required")
    fig.tight_layout()
    fig.savefig(out / "causal_overlap_and_gate.png", dpi=300)
    plt.close(fig)
    overall = results.loc[results.window == "overall"].copy()
    selected_cols = [
        "method",
        "n",
        "mae",
        "rmse",
        "coverage",
        "mean_width",
        "acceptance",
    ]
    overall[selected_cols].to_csv(ROOT / "reports/main_forecast_table.csv", index=False)
    latex_rows = [
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        "Method & N & MAE & RMSE & Coverage & Width & Acceptance " + r"\\",
        r"\hline",
    ]
    for _, row in overall.iterrows():
        values = [str(row["method"]).replace("_", r"\_"), str(int(row["n"]))]
        values += [
            "--" if pd.isna(row[c]) else f"{row[c]:.4f}" for c in selected_cols[2:]
        ]
        latex_rows.append(" & ".join(values) + " " + r"\\")
    latex_rows.extend([r"\hline", r"\end{tabular}"])
    (ROOT / "reports/main_forecast_table.tex").write_text(
        "\n".join(latex_rows), encoding="utf-8"
    )

    def table(frame, columns):
        lines = [
            "| " + " | ".join(columns) + " |",
            "|" + "|".join(["---"] * len(columns)) + "|",
        ]
        for _, row in frame.iterrows():
            vals = []
            for c in columns:
                value = row.get(c)
                vals.append(
                    "—"
                    if pd.isna(value)
                    else (
                        f"{value:.4f}"
                        if isinstance(value, (float, np.floating))
                        else str(value)
                    )
                )
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    adaptive_row = overall.loc[overall.method == "adaptive_conformal"].iloc[0]
    split_row = overall.loc[overall.method == "split_conformal"].iloc[0]
    rolling_row = overall.loc[overall.method == "rolling_mean"].iloc[0]
    gate_row = overall.loc[overall.method == "predictive_gate"].iloc[0]
    width_reduction = 1 - adaptive_row.mean_width / split_row.mean_width
    conservative = logistics.loc[logistics.method == "conservative_lower_bound_policy"]
    gain_finding = (
        "Policy superiority is not estimable because empirical overlap is insufficient."
    )
    if len(conservative):
        cr = conservative.iloc[0]
        gain_finding = f"The conservative policy's estimated gain over Carting is {cr.estimated_gain_vs_carting:.4f} delay-factor units, with a date-cluster interval [{cr.gain_ci_low:.4f}, {cr.gain_ci_high:.4f}]. "
        gain_finding += (
            "This interval includes zero, so the evaluation does not establish an intervention benefit."
            if cr.gain_ci_low <= 0 <= cr.gain_ci_high
            else "This is an observational model estimate and remains subject to the identification limitations below."
        )
    text = f"""# Reliable Causal Decisions for Supply Chains Under Distribution Shift

## Evaluation status

This is a completed real-data prototype evaluation with a narrower, explicitly documented scope than the supplied project plan. It is not evidence of operational stock-out reduction or deployable causal automation.

AGMARKNET: **{card['selected_real_observations']:,} observed market-day-commodity records**, {card['selected_markets']} markets, rice and wheat, {card['date_min']}–{card['date_max']}. Calendar padding adds {card['unobserved_calendar_rows']:,} missing rows; these are not observations and are not scored. Raw target values were never generated or imputed.

Delhivery: **{lcard['raw_rows']:,} scan records**, aggregated to **{lcard['usable_trips']:,} usable trips**, {lcard['start']}–{lcard['end']}. OD-leg cumulative maxima are summed before trip aggregation.

## Findings

Adaptive interval coverage is {adaptive_row.coverage:.2%} on {int(adaptive_row['n']):,} held-out market observations, with {width_reduction:.2%} narrower intervals than split conformal. This is observed coverage, not a universal guarantee under shift. The forecasting model has MAE {adaptive_row.mae:.4f} versus {rolling_row.mae:.4f} for the rolling mean, but RMSE {adaptive_row.rmse:.4f} versus {rolling_row.rmse:.4f}; it does not dominate that baseline.

The validation-selected predictive gate accepts {gate_row.acceptance:.1%} of test predictions. At this operating point, the gate and its ablations do not demonstrate an additional reliability benefit. The gate sweep is a prediction-acceptance study, not a cost-savings curve.

Coverage varies across state/commodity groups from {regional_results.coverage.min():.2%} to {regional_results.coverage.max():.2%}. Overall coverage therefore does not establish conditional reliability for every region or product. At the selected drift threshold, no alert is raised in the demonetisation window; this experiment does not demonstrate successful event detection.

{gain_finding}

## Forecast results on held-out real observations

{table(overall,selected_cols)}

MAE and width use tonnes of arrivals. Forecasting arrivals is supply-flow forecasting, not measured customer demand. Historical mean and rolling mean are forecast references; they are not experimentally evaluated inventory policies. Predictive gate acceptance is selective interval reporting, not successful automatic intervention precision. Accepted subsets differ; compare acceptance and coverage together.

## Event-window results

{table(results.loc[(results.window=='demonetisation_2016')],selected_cols)}

The legal tender withdrawal took effect on 9 November 2016. The fixed 30-day event window is an evaluation annotation; timing alone does not establish a causal market effect. Outside-window alerts are reported as alerts, not verified false positives. See `experiments/shift_detection.json`.

## Observational logistics policy estimates

{table(logistics,['method','n','mean_dr_delay_loss','loss_ci_low','loss_ci_high','estimated_gain_vs_carting','gain_ci_low','gain_ci_high','ftl_recommendation_fraction','status'])}

Test overlap includes {diag['test_overlap_rows']:,}/{diag['test_rows']:,} trips ({diag['test_overlap_fraction']:.1%}); empirical support check: {diag['empirical_support_pass']}. Only {diag['independent_test_dates']} held-out calendar days underpin the cluster intervals. Lower estimated delay loss is better; intervals are exploratory and do not account for unmeasured confounding. They are computed for the overlap subset, not the full logistics population. Carting is a reference route type, not a documented no-action treatment. Action cost is set to zero in delay-factor units as a declared scenario; no monetary savings are inferred.

The causal gate withholds every intervention because verified pre-dispatch adjustment variables and identification are absent. Retrospectively aggregated OSRM distance may be affected by route choice; shipment load, available fleet and dispatch constraints are not observed. Narrow model intervals and placebo checks cannot fix these identification limits. No realised green-zone action precision or gate policy value can be measured from the logs.

## Method and leakage controls

- Whole dates are separated into training, calibration, validation and test. Features use strictly prior calendar observations. Categorical vocabularies are frozen on training data. Observations without sufficient past history are excluded from forecasting; see `experiments/market_splits.json` for exact eligible counts. Historical-mean forecasts for new market/commodity combinations fall back to a training-only state/commodity mean.
- One fixed LightGBM model predicts log arrivals. Absolute residuals are normalized by a prior 28-day mean, calibrated by state and commodity, then returned to original units.
- Split conformal uses the finite-sample corrected order statistic. Adaptive calibration updates after each same-day batch of outcomes; this clipped, batched ACI-inspired implementation is an empirical adaptation. Original ACI theoretical results do not automatically transfer to this implementation or establish conditional guarantees for dependent panels.
- Predictive thresholds maximize validation acceptance subject to empirical 90% accepted coverage and at least 30 accepted records. Test labels never select thresholds.
- Logistics models exclude training trips unresolved at the next split boundary. Potential outcome models and propensities fit on historical trips; date-cluster bootstrap refits estimate model uncertainty. Held-out doubly robust estimates use a frozen propensity overlap rule.
- No synthetic source records are used. Placebo analysis permutes existing treatment labels as a falsification check only.

## Deviations and unavailable claims

The CEDA public endpoint returned capped 1,000-row downloads and rate limiting; the government API returned HTTP 403. Complete rice/wheat data came from the documented Ian Covert AGMARKNET archive. Onion/potato and 2023 event experiments are not completed in this run because the archive does not contain them. The true observation count is reported rather than padded to 50,000.

AGMARKNET and Delhivery have no verified common shipment keys and cover different years. They are evaluated as separate evidence streams; no artificial geographic/time merge was made. Actual inventory, demand fulfilment, stock-outs, intervention costs, warehouse transfers and supplier switching are unavailable. The arrivals/price shortage proxy is descriptive, not a validated stock-out label. Static safety-stock costs, proxy fill-rate improvements under interventions, end-to-end causal policy superiority, and real-world automatic action precision are not estimable from these sources.

The prototype uses a transparent scikit-learn/LightGBM T-learner and manual doubly robust scoring rather than EconML/DoWhy. The available interpreter is Python 3.13; source supports Python 3.11+. No results are fabricated to satisfy planned performance tests.

## Reproduction and artifacts

See `README.md`. Results are in `experiments/results.csv`; per-date predictions in `experiments/market_replay.parquet`; policy diagnostics in `experiments/causal_diagnostics.json`; source checksums in `data/raw/archive_manifest.json`; configuration hashes in `experiments/run_manifest.json`. Figures and tables regenerate from these outputs.

## Sources

- AGMARKNET original government records and archive collection documentation: https://github.com/iancovert/Agmarknet
- Delhivery public educational dataset mirror: https://github.com/shekshavalipattan/Delhivery-Logistics-Data-Pipeline-and-Feature-Engineering
- RBI event notification: https://www.rbi.org.in/commonman/english/Scripts/Notification.aspx?Id=1938
- CEDA data portal: https://ceda.ashoka.edu.in/data-portal/

Method references: Gibbs and Candès, *Adaptive Conformal Inference Under Distribution Shift*, NeurIPS 2021 ([paper](https://arxiv.org/abs/2106.00170)); Dudík, Langford and Li, *Doubly Robust Policy Evaluation and Learning*, ICML 2011 ([paper](https://arxiv.org/abs/1103.4601)); Künzel et al., *Metalearners for estimating heterogeneous treatment effects using machine learning*, PNAS 2019 ([paper](https://doi.org/10.1073/pnas.1804597116)). These references motivate the methods; they do not validate the causal assumptions in these datasets.

Dataset redistribution licences were not verified for the archives. Raw data is kept locally and excluded from the code delivery ZIP.
"""
    (ROOT / "reports/EVALUATION.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
