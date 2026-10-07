"""Run separate market forecasting and exploratory logistics policy evaluations."""

import os
import platform
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd
import yaml

from scrc.baselines.rules import policy_baselines
from scrc.causal.effects import (
    ObservationalEffects,
    clustered_interval,
    doubly_robust_scores,
)
from scrc.causal.refutation import diagnostics
from scrc.causal.selector import select
from scrc.common import ROOT, config, digest, save_json
from scrc.data.panel import chronological_splits
from scrc.eval.metrics import forecast_metrics
from scrc.gate.gate import gate
from scrc.models.conformal import AdaptiveConformal, conformal_quantile
from scrc.models.forecaster import forecaster
from scrc.models.shift_detector import shift_score

FEATURES = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean",
    "rolling_std",
    "price_lag",
    "weekday",
    "month",
    "year_day",
    "state_code",
    "market_code",
    "commodity_code",
]


def encode_market(frame, train):
    """Freeze categorical vocabularies on the training period."""
    frame = frame.copy()
    for col in ["state", "market", "commodity"]:
        mapping = {x: i for i, x in enumerate(sorted(train[col].unique()))}
        frame[col + "_code"] = frame[col].map(mapping).fillna(-1).astype(int)
    for col in [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean",
        "rolling_std",
        "price_lag",
    ]:
        frame[col] = np.log1p(frame[col])
    return frame[FEATURES]


def market_evaluation(cfg):
    """Replay daily batches and reveal labels only after recording each interval."""
    panel = pd.read_parquet(ROOT / "data/processed/panel.parquet")
    frame = panel.loc[
        panel.observed & panel.rolling_mean.notna() & (panel.rolling_mean > 0)
    ].copy()
    frame["scale"] = frame.rolling_mean.clip(lower=1.0)
    train, cal, validation, test = chronological_splits(
        frame, cfg, cut_dates=cfg.get("market_split_dates")
    )
    model = forecaster(cfg["seed"])
    model.fit(encode_market(train, train), np.log1p(train.arrivals))
    model_dir = ROOT / "experiments/models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_dir / "market_lightgbm.txt"))
    save_json(
        model_dir / "market_feature_encoding.json",
        {
            "features": FEATURES,
            "categorical_vocabulary": {
                col: sorted(train[col].unique())
                for col in ["state", "market", "commodity"]
            },
            "unknown_category_code": -1,
            "log1p_columns": [
                "lag_1",
                "lag_7",
                "lag_14",
                "lag_28",
                "rolling_mean",
                "rolling_std",
                "price_lag",
            ],
            "target": "log1p(arrivals); invert with expm1 and clip below at zero",
            "scope": "Frozen historical predictor; construct past-only features with scrc.data.panel before use.",
        },
    )
    for part in [cal, validation, test]:
        part["prediction"] = np.maximum(
            0, np.expm1(model.predict(encode_market(part, train)))
        )
        part["residual"] = abs(part.arrivals - part.prediction) / part.scale
    fixed = {}
    adaptive = {}
    histories = {}
    recent_features = {}
    references = {}
    drift_cols = ["lag_1", "price_lag"]
    for key, group in cal.groupby(["state", "commodity"]):
        scores = group.residual.to_numpy()
        fixed[key] = conformal_quantile(scores, cfg["alpha"])
        adaptive[key] = AdaptiveConformal(
            scores,
            cfg["alpha"],
            cfg["adaptive_gamma"],
            window=cfg["online_residual_window"],
        )
        histories[key] = []
        recent_features[key] = []
        ref = train.loc[
            (train.state == key[0]) & (train.commodity == key[1]), drift_cols
        ]
        references[key] = np.log1p(ref.fillna(0).to_numpy())
    records = []
    for split, part in [("validation", validation), ("test", test)]:
        for date, day in part.groupby("date", sort=True):
            for key, batch in day.groupby(["state", "commodity"]):
                if key not in adaptive:
                    raise ValueError(f"No calibration support for {key}")
                q = adaptive[key].quantile()
                lower = np.maximum(0, batch.prediction - q * batch.scale)
                upper = batch.prediction + q * batch.scale
                covered = (batch.arrivals >= lower) & (batch.arrivals <= upper)
                recent = np.asarray(recent_features[key][-cfg["drift_recent_rows"] :])
                hist = histories[key][-cfg["online_residual_window"] :]
                drift = (
                    shift_score(
                        references[key],
                        recent,
                        np.mean(hist) if hist else 1 - cfg["alpha"],
                    )
                    if len(recent) >= 20
                    else 0.0
                )
                b = batch.copy()
                b["split"] = split
                b["lower_adaptive"] = lower
                b["upper_adaptive"] = upper
                b["lower_split"] = np.maximum(0, b.prediction - fixed[key] * b.scale)
                b["upper_split"] = b.prediction + fixed[key] * b.scale
                b["shift_score"] = drift
                b["width_ratio"] = (upper - lower) / np.maximum(b.prediction, 1.0)
                b["covered_adaptive"] = covered
                records.append(b)
                # Feedback arrives only after every same-day prediction in this group.
                adaptive[key].update(batch.residual.to_numpy(), (~covered).to_numpy())
                histories[key].extend(covered.tolist())
                recent_features[key].extend(
                    np.log1p(batch[drift_cols].fillna(0).to_numpy()).tolist()
                )
    replay = pd.concat(records, ignore_index=True)
    v = replay.loc[replay.split == "validation"]
    sweep = []
    for width in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
        for shift in [0.15, 0.25, 0.4, 0.6, 1.0]:
            mask = (v.width_ratio <= width) & (v.shift_score <= shift)
            n = int(mask.sum())
            coverage = float(v.loc[mask, "covered_adaptive"].mean()) if n else None
            sweep.append(
                {
                    "tau_width": width,
                    "tau_shift": shift,
                    "n": n,
                    "acceptance": float(mask.mean()),
                    "coverage": coverage,
                    "feasible": bool(
                        n >= 30 and coverage >= cfg["green_coverage_target"]
                    ),
                }
            )
    pd.DataFrame(sweep).to_csv(
        ROOT / "experiments/threshold_sweep_validation.csv", index=False
    )
    feasible = [x for x in sweep if x["feasible"]]
    if feasible:
        tuned = max(
            feasible, key=lambda x: (x["acceptance"], -x["tau_width"], -x["tau_shift"])
        )
    else:
        tuned = {
            "tau_width": 0.0,
            "tau_shift": 0.0,
            "acceptance": 0.0,
            "coverage": None,
            "n": 0,
            "feasible": False,
        }
    save_json(ROOT / "experiments/tuned_predictive_gate.json", tuned)
    replay["predictive_eligible"] = (replay.width_ratio <= tuned["tau_width"]) & (
        replay.shift_score <= tuned["tau_shift"]
    )
    replay["eligible_without_shift"] = replay.width_ratio <= tuned["tau_width"]
    replay["eligible_without_width"] = replay.shift_score <= tuned["tau_shift"]
    replay["event"] = "outside_labelled_window"
    events = yaml.safe_load((ROOT / "configs/events.yaml").read_text())
    for event in events:
        match = (replay.date >= event["start"]) & (replay.date <= event["end"])
        if event["commodity"] != "all":
            match &= replay.commodity == event["commodity"]
        replay.loc[match, "event"] = event["name"]
    held = replay.loc[replay.split == "test"].copy()
    historical = train.groupby(["state", "market", "commodity"]).arrivals.mean()
    regional = train.groupby(["state", "commodity"]).arrivals.mean()
    held["historical_mean"] = [
        historical.get((s, m, c), regional.get((s, c), train.arrivals.mean()))
        for s, m, c in zip(held.state, held.market, held.commodity)
    ]
    results = []
    for label, subset in [("overall", held)] + list(held.groupby("event")):
        for name, pred, lo, hi, mask in [
            (
                "historical_mean",
                "historical_mean",
                None,
                None,
                np.ones(len(subset), dtype=bool),
            ),
            (
                "rolling_mean",
                "rolling_mean",
                None,
                None,
                np.ones(len(subset), dtype=bool),
            ),
            (
                "point_forecast",
                "prediction",
                None,
                None,
                np.ones(len(subset), dtype=bool),
            ),
            (
                "split_conformal",
                "prediction",
                "lower_split",
                "upper_split",
                np.ones(len(subset), dtype=bool),
            ),
            (
                "adaptive_conformal",
                "prediction",
                "lower_adaptive",
                "upper_adaptive",
                np.ones(len(subset), dtype=bool),
            ),
            (
                "predictive_gate",
                "prediction",
                "lower_adaptive",
                "upper_adaptive",
                subset.predictive_eligible.to_numpy(),
            ),
            (
                "gate_without_shift",
                "prediction",
                "lower_adaptive",
                "upper_adaptive",
                subset.eligible_without_shift.to_numpy(),
            ),
            (
                "gate_without_width",
                "prediction",
                "lower_adaptive",
                "upper_adaptive",
                subset.eligible_without_width.to_numpy(),
            ),
        ]:
            picked = subset.loc[mask]
            row = {
                "dataset": "AGMARKNET",
                "window": label,
                "method": name,
                "acceptance": float(np.mean(mask)),
            }
            if len(picked):
                row.update(
                    forecast_metrics(
                        picked.arrivals,
                        picked[pred],
                        picked[lo] if lo else None,
                        picked[hi] if hi else None,
                    )
                )
                if lo:
                    cover = (
                        (picked.arrivals >= picked[lo])
                        & (picked.arrivals <= picked[hi])
                    ).astype(float)
                    ci = clustered_interval(
                        cover, picked.date, cfg["bootstrap_repeats"], cfg["seed"]
                    )
                    row.update(coverage_ci_low=ci[0], coverage_ci_high=ci[1])
            else:
                row["n"] = 0
            results.append(row)
    held.to_parquet(ROOT / "experiments/market_replay.parquet", index=False)
    replay.loc[replay.split == "validation"].to_parquet(
        ROOT / "experiments/market_validation.parquet", index=False
    )
    pd.DataFrame(results).to_csv(ROOT / "experiments/market_results.csv", index=False)
    per_series = []
    for key, g in held.groupby(["state", "commodity"]):
        row = {"state": key[0], "commodity": key[1]}
        row.update(
            forecast_metrics(
                g.arrivals, g.prediction, g.lower_adaptive, g.upper_adaptive
            )
        )
        per_series.append(row)
    pd.DataFrame(per_series).to_csv(
        ROOT / "experiments/market_results_by_region.csv", index=False
    )
    detection = []
    for event in events:
        # Dates are event annotations, not exhaustive ground truth for all drift.
        event_rows = held.loc[
            (held.date >= event["start"]) & (held.date <= event["end"])
        ]
        if event["commodity"] != "all":
            event_rows = event_rows.loc[event_rows.commodity == event["commodity"]]
        if not len(event_rows):
            continue
        daily = event_rows.groupby("date").shift_score.mean()
        flags = daily[daily > tuned["tau_shift"]]
        outside = (
            held.loc[held.event == "outside_labelled_window"]
            .groupby("date")
            .shift_score.mean()
        )
        detection.append(
            {
                "event": event["name"],
                "detected": bool(len(flags)),
                "delay_days": (
                    int((flags.index[0] - pd.Timestamp(event["start"])).days)
                    if len(flags)
                    else None
                ),
                "threshold": tuned["tau_shift"],
                "outside_window_alert_rate": float(
                    (outside > tuned["tau_shift"]).mean()
                ),
                "caveat": "Outside-window alerts are not proven false alarms; event dates are not complete shift ground truth.",
            }
        )
    save_json(ROOT / "experiments/shift_detection.json", detection)
    save_json(
        ROOT / "experiments/market_splits.json",
        [
            {
                "name": name,
                "n": len(part),
                "start": str(part.date.min().date()),
                "end": str(part.date.max().date()),
            }
            for name, part in zip(
                ["train", "calibration", "validation", "test"],
                [train, cal, validation, test],
            )
        ],
    )
    return results


def logistics_evaluation(cfg):
    """Evaluate a fixed policy only on overlap; never imply deployment identification."""
    trips = pd.read_parquet(ROOT / "data/processed/trips.parquet")
    train, cal, validation, test = chronological_splits(trips, cfg)
    # Remove unresolved trips at each nuisance-training boundary.
    original_train = len(train)
    train = train.loc[train.end_time < cal.date.min()].copy()
    model = ObservationalEffects(cfg["seed"]).fit(train)
    model_dir = ROOT / "experiments/models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for arm, fitted in enumerate(model.models):
        fitted.booster_.save_model(str(model_dir / f"logistics_outcome_arm_{arm}.txt"))
    numeric = model.transform.named_transformers_["numeric"]
    categorical = model.transform.named_transformers_["categorical"]
    save_json(
        model_dir / "logistics_encoding_and_propensity.json",
        {
            "numeric_columns": ["log_distance", "weekday", "hour"],
            "numeric_mean": numeric.mean_.tolist(),
            "numeric_scale": numeric.scale_.tolist(),
            "categorical_columns": ["source_state", "destination_state"],
            "categories": [c.tolist() for c in categorical.categories_],
            "unknown_categories": "All-zero one-hot encoding",
            "propensity_coefficients": model.propensity.coef_.tolist(),
            "propensity_intercept": model.propensity.intercept_.tolist(),
            "missing_category": "Unknown",
            "scope": "Retrospective observational research models; causal deployment is unsupported.",
        },
    )
    all_target = pd.concat([validation, test], ignore_index=True)
    m0, m1, e = model.predict(all_target)
    print("Refitting date-cluster causal bootstrap models...", flush=True)
    lb, ub = model.interval(train, all_target, cfg)
    all_target["mu_carting"] = m0
    all_target["mu_ftl"] = m1
    all_target["propensity"] = e
    all_target["benefit_lower"] = lb
    all_target["benefit_upper"] = ub
    all_target["overlap"] = (e >= cfg["propensity_min"]) & (e <= cfg["propensity_max"])
    all_target["conservative_policy"] = select(lb, all_target.overlap)
    all_target["point_policy"] = ((m0 - m1) > cfg["action_cost"]).astype(int)
    resolved_cal = cal.loc[cal.end_time < validation.date.min()]
    c0, c1, _ = model.predict(resolved_cal)
    factual_cal = np.where(resolved_cal.treatment.to_numpy() == 1, c1, c0)
    q = conformal_quantile(
        abs(resolved_cal.delay_factor.to_numpy() - factual_cal), cfg["alpha"]
    )
    chosen_mu = np.where(all_target.conservative_policy == 1, m1, m0)
    all_target["lower_delay"] = np.maximum(0, chosen_mu - q)
    all_target["upper_delay"] = chosen_mu + q
    all_target["width_ratio"] = (
        all_target.upper_delay - all_target.lower_delay
    ) / np.maximum(chosen_mu, 0.01)
    all_target["shift_score"] = 0.0
    reference = train[["log_distance", "weekday", "hour"]].to_numpy()
    for date, batch in all_target.groupby("date", sort=True):
        past = (
            all_target.loc[all_target.date < date, ["log_distance", "weekday", "hour"]]
            .tail(cfg["drift_recent_rows"])
            .to_numpy()
        )
        # Coverage uses only trips whose recorded end time precedes the current day.
        resolved = all_target.loc[all_target.end_time < date].tail(
            cfg["online_residual_window"]
        )
        coverage = (
            float(
                (
                    (resolved.delay_factor >= resolved.lower_delay)
                    & (resolved.delay_factor <= resolved.upper_delay)
                ).mean()
            )
            if len(resolved)
            else 0.9
        )
        all_target.loc[batch.index, "shift_score"] = shift_score(
            reference, past, coverage
        )
    all_target["gate"] = [
        gate(w, s, l, bool(o and cfg["verified_predispatch_covariates"]), cfg)
        for w, s, l, o in zip(
            all_target.width_ratio, all_target.shift_score, lb, all_target.overlap
        )
    ]
    hold = all_target.loc[all_target.date >= test.date.min()].copy()
    diag = diagnostics(train, test, model, cfg)
    overlap = hold.loc[hold.overlap].copy()
    empirical_support = (
        len(overlap) >= cfg["min_overlap_rows"]
        and len(overlap) / len(hold) >= cfg["min_overlap_fraction"]
        and overlap.groupby("treatment").size().reindex([0, 1], fill_value=0).min()
        >= cfg["min_action_rows"]
    )
    diag.update(
        train_rows=len(train),
        purged_unresolved_train_trips=original_train - len(train),
        test_rows=len(hold),
        test_overlap_rows=len(overlap),
        test_overlap_fraction=float(hold.overlap.mean()),
        test_propensity_quantiles={
            str(q): float(hold.propensity.quantile(q))
            for q in [0, 0.05, 0.5, 0.95, 1.0]
        },
        empirical_support_pass=bool(empirical_support),
        verified_predispatch_covariates=cfg["verified_predispatch_covariates"],
        automation_supported=False,
        automation_reason="No verified pre-dispatch causal covariates, unmeasured shipment load, no intervention cost data; causal identification not established.",
        bootstrap_note="Model intervals from training-date cluster bootstrap; omit unmeasured-confounding uncertainty.",
        independent_test_dates=int(hold.date.nunique()),
        gate_counts=hold.gate.value_counts().to_dict(),
    )
    policies = policy_baselines(len(overlap))
    policies.update(
        point_effect_policy=overlap.point_policy.to_numpy(),
        conservative_lower_bound_policy=overlap.conservative_policy.to_numpy(),
    )
    results = []
    if empirical_support:
        baseline = doubly_robust_scores(
            overlap,
            overlap.mu_carting.to_numpy(),
            overlap.mu_ftl.to_numpy(),
            overlap.propensity.to_numpy(),
            policies["always_carting"],
        )
        for name, policy in policies.items():
            loss = (
                doubly_robust_scores(
                    overlap,
                    overlap.mu_carting.to_numpy(),
                    overlap.mu_ftl.to_numpy(),
                    overlap.propensity.to_numpy(),
                    policy,
                )
                + cfg["action_cost"] * policy
            )
            ci = clustered_interval(
                loss, overlap.date, cfg["bootstrap_repeats"], cfg["seed"]
            )
            gain = baseline - loss
            gain_ci = clustered_interval(
                gain, overlap.date, cfg["bootstrap_repeats"], cfg["seed"]
            )
            results.append(
                {
                    "dataset": "Delhivery",
                    "window": "overlap_test_only",
                    "method": name,
                    "n": len(overlap),
                    "mean_dr_delay_loss": float(np.mean(loss)),
                    "loss_ci_low": ci[0],
                    "loss_ci_high": ci[1],
                    "estimated_gain_vs_carting": float(np.mean(gain)),
                    "gain_ci_low": gain_ci[0],
                    "gain_ci_high": gain_ci[1],
                    "ftl_recommendation_fraction": float(np.mean(policy)),
                    "status": "exploratory; observational assumptions unverified",
                }
            )
    else:
        results.append(
            {
                "dataset": "Delhivery",
                "window": "test",
                "method": "policy_evaluation",
                "n": len(overlap),
                "status": "NOT ESTIMABLE: empirical overlap below frozen threshold",
            }
        )
    results.append(
        {
            "dataset": "Delhivery",
            "window": "test",
            "method": "reliability_gate",
            "n": len(hold),
            "automated_fraction": 0.0,
            "escalated_fraction": 1.0,
            "status": "All actions withheld: identification precondition not met. No counterfactual gate value claimed.",
        }
    )
    hold.to_parquet(ROOT / "experiments/logistics_replay.parquet", index=False)
    pd.DataFrame(results).to_csv(
        ROOT / "experiments/logistics_results.csv", index=False
    )
    save_json(ROOT / "experiments/causal_diagnostics.json", diag)
    save_json(
        ROOT / "experiments/logistics_splits.json",
        [
            {
                "name": name,
                "n": len(part),
                "start": str(part.date.min().date()),
                "end": str(part.date.max().date()),
            }
            for name, part in zip(
                ["train_purged", "calibration", "validation", "test"],
                [train, cal, validation, test],
            )
        ],
    )
    return results


def main():
    """Write reproducible metrics from real observations and honest unavailable fields."""
    cfg = config()
    (ROOT / "experiments").mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    print("Evaluating market forecasting with chronological feedback...", flush=True)
    market = market_evaluation(cfg)
    print("Evaluating separate observational logistics policies...", flush=True)
    logistics = logistics_evaluation(cfg)
    pd.DataFrame(market + logistics).to_csv(
        ROOT / "experiments/results.csv", index=False
    )
    official = cfg.get("market_source") == "official"
    from scrc.data.official import official_jobs

    files = (
        [
            ROOT / f"data/raw/official/{s}_{c}_{y}_{m:02d}.json.gz"
            for s, c, y, m in official_jobs(cfg)
        ]
        if official
        else list((ROOT / "data/raw").glob("agmarknet_*.csv"))
    )
    files += [ROOT / "data/raw/delhivery.csv"]
    save_json(
        ROOT / "experiments/run_manifest.json",
        {
            "config_file": os.environ.get("SCRC_CONFIG", "configs/default.yaml"),
            "config_hash": digest(
                ROOT / os.environ.get("SCRC_CONFIG", "configs/default.yaml")
            ),
            "events_hash": digest(ROOT / "configs/events.yaml"),
            "python": platform.python_version(),
            "seed": cfg["seed"],
            "raw_data_hashes": {str(p.relative_to(ROOT)): digest(p) for p in files},
            "source_code_hashes": {
                str(p.relative_to(ROOT)): digest(p)
                for p in sorted((ROOT / "scrc").rglob("*.py"))
            },
            "selected_markets_hash": digest(
                ROOT
                / (
                    "configs/"
                    + cfg.get("selected_market_file", "official_markets.json")
                    if official
                    else "configs/archive_markets.json"
                )
            ),
            "package_versions": {
                name: version(name)
                for name in [
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "lightgbm",
                    "pyarrow",
                    "PyYAML",
                    "matplotlib",
                    "requests",
                ]
            },
            "method": "Fixed historical model; rolling daily conformal feedback. Separate observational logistics holdout.",
            "no_claims": [
                "No achieved stock-out reduction, monetary cost savings, live execution, or guaranteed coverage under arbitrary shift."
            ],
        },
    )
    from scrc.eval.plots import main as plots

    plots()
    print(
        "Evaluation complete. See experiments/results.csv and reports/EVALUATION.md",
        flush=True,
    )


if __name__ == "__main__":
    main()
