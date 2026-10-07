"""Freeze commodity specialists using rolling validation, then acquire a fresh future test."""

import hashlib
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from scrc.common import ROOT, digest, save_json
from scrc.data.official import CROPS, fetch, rows
from scrc.models.conformal import conformal_quantile
from scrc.production.features import (
    CATEGORIES,
    FEATURES,
    build_features,
    eligible,
    encode,
    predict,
    vocabulary,
)
from scrc.production.train import metrics

OUT = ROOT / "artifacts/production"
SPECS = [
    ("log_l2", "regression", "log", 15, 400),
    ("raw_l1", "regression_l1", "raw", 31, 500),
    ("raw_l2", "regression", "raw", 31, 400),
    ("tweedie", "tweedie", "raw", 31, 500),
    ("poisson", "poisson", "raw", 15, 400),
    ("ratio_l1", "regression_l1", "ratio", 15, 400),
    ("raw_l1_small", "regression_l1", "raw", 15, 400),
    ("huber", "huber", "raw", 31, 500),
]


def fit(train, spec, vocab):
    _name, objective, target, leaves, estimators = spec
    y = train.arrivals.to_numpy()
    if target == "log":
        y = np.log1p(y)
    if target == "ratio":
        y = y / train.rolling_mean.clip(lower=1).to_numpy()
    model = LGBMRegressor(
        objective=objective,
        n_estimators=estimators,
        learning_rate=0.035,
        num_leaves=leaves,
        min_child_samples=50,
        reg_lambda=5,
        deterministic=True,
        force_col_wise=True,
        random_state=42,
        n_jobs=2,
        verbosity=-1,
    )
    model.fit(
        encode(train, vocab), y, categorical_feature=[c + "_code" for c in CATEGORIES]
    )
    return model


def predict_bundle(models, frame, meta):
    answer = np.empty(len(frame))
    positions = np.arange(len(frame))
    for crop, route in meta["routing"].items():
        mask = frame.commodity.to_numpy() == crop
        if mask.any():
            answer[positions[mask]] = predict(
                models[route["model_file"]], frame.loc[mask], {**meta, **route}
            )
    return answer


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    observations = pd.read_parquet(ROOT / "data/processed/market_observations.parquet")
    panel = build_features(observations)
    data = panel.loc[eligible(panel)].copy()
    folds = [
        ("2023-01-01", "2023-04-01"),
        ("2023-04-01", "2023-07-01"),
        ("2023-07-01", "2023-10-01"),
    ]
    scores = []
    for fold, (start, end) in enumerate(folds):
        train = data.loc[data.date < start]
        val = data.loc[(data.date >= start) & (data.date < end)]
        vocab = vocabulary(train)
        for spec in SPECS:
            print(f"Rolling validation fold {fold+1}/3: {spec[0]}", flush=True)
            model = fit(train, spec, vocab)
            for weight in [1.0, 0.75, 0.5]:
                p = predict(
                    model.booster_,
                    val,
                    {"vocabulary": vocab, "target": spec[2], "blend_weight": weight},
                )
                for crop in sorted(val.commodity.unique()):
                    mask = val.commodity.to_numpy() == crop
                    scores.append(
                        {
                            "fold": fold,
                            "commodity": crop,
                            "name": spec[0],
                            "target": spec[2],
                            "blend_weight": weight,
                            **metrics(val.loc[mask, "arrivals"], p[mask]),
                        }
                    )
    scores = pd.DataFrame(scores)
    scores.to_csv(OUT / "validation_search.csv", index=False)
    aggregate = (
        scores.groupby(["commodity", "name", "target", "blend_weight"])
        .agg(
            mean_wape=("wape", "mean"),
            wape_std=("wape", "std"),
            mean_rmse=("rmse", "mean"),
        )
        .reset_index()
    )
    aggregate["selection_score"] = aggregate.mean_wape + 0.25 * aggregate.wape_std
    routing = {}
    for crop, g in aggregate.groupby("commodity"):
        winner = g.sort_values(["selection_score", "mean_rmse", "name"]).iloc[0]
        routing[crop] = {
            "name": winner["name"],
            "model_file": f"model_{winner['name']}.txt",
            "target": winner["target"],
            "blend_weight": float(winner["blend_weight"]),
            "validation_score": float(winner["selection_score"]),
        }
    aggregate.to_csv(OUT / "validation_ranking.csv", index=False)
    train = data.loc[data.date < "2023-10-01"]
    cal = data.loc[data.date >= "2023-10-01"]
    vocab = vocabulary(train)
    models = {}
    selected_names = {route["name"] for route in routing.values()} | {"log_l2"}
    for spec in SPECS:
        if spec[0] in selected_names:
            print("Final pre-holdout fit: " + spec[0], flush=True)
            fitted = fit(train, spec, vocab)
            name = f"model_{spec[0]}.txt"
            fitted.booster_.save_model(str(OUT / name))
            models[name] = fitted.booster_
    meta = {
        "vocabulary": vocab,
        "features": FEATURES,
        "routing": routing,
        "nominal_coverage": 0.95,
        "training_end": "2023-09-30",
        "calibration_start": "2023-10-01",
        "calibration_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-03-31",
        "model_data_cutoff": "2023-12-31",
        "selection_rule": "Per commodity: minimum mean rolling-fold WAPE + 0.25 * fold WAPE standard deviation; ties use mean RMSE. Three expanding validation quarters in 2023. No 2024 records used for fitting or selection.",
        "scope": {
            "states": sorted(observations.state.unique()),
            "commodities": sorted(observations.commodity.unique()),
        },
        "frozen_before_forward_acquisition_utc": datetime.now(UTC).isoformat(),
    }
    cp = predict_bundle(models, cal, meta)
    score = cal[["state", "commodity"]].copy()
    score["residual"] = (
        np.abs(cal.arrivals.to_numpy() - cp) / cal.rolling_mean.clip(lower=1).to_numpy()
    )
    meta["conformal_quantiles"] = {
        f"{s}|{c}": conformal_quantile(g.residual, 0.05)
        for (s, c), g in score.groupby(["state", "commodity"])
    }
    meta["references"] = {
        f"{s}|{m}|{c}": {
            "median_arrivals": float(g.arrivals.median()),
            "median_price": float(g.modal_price.median()),
            "rows": len(g),
        }
        for (s, m, c), g in train.groupby(CATEGORIES)
    }
    save_json(OUT / "frozen_selection.json", meta)
    print(
        "Model and 95% calibration frozen. Acquiring January–March 2024 forward holdout.",
        flush=True,
    )
    sources = []
    future = []
    chosen = set(
        map(tuple, observations[["state", "market"]].drop_duplicates().to_numpy())
    )
    for crop in CROPS:
        for month in [1, 2, 3]:
            record = fetch((34, crop, 2024, month))
            sources.append(record)
            for row in rows(ROOT / record["file"]):
                if (row["state"], row["market"]) in chosen:
                    future.append(row)
            print(f"Fresh holdout: {len(sources)}/12 monthly sources", flush=True)
    save_json(OUT / "forward_source_manifest.json", sources)
    combined = pd.concat(
        [observations, pd.DataFrame(future)], ignore_index=True
    ).sort_values([*CATEGORIES, "date"])
    if combined.duplicated([*CATEGORIES, "date"]).any():
        raise ValueError("Duplicate source keys in forward evaluation.")
    combined.to_csv(ROOT / "data/processed/deployment_observations.csv", index=False)
    forward = build_features(combined)
    test = forward.loc[eligible(forward) & (forward.date >= "2024-01-01")].copy()
    tp = predict_bundle(models, test, meta)
    q = np.array(
        [
            meta["conformal_quantiles"][f"{s}|{c}"]
            for s, c in zip(test.state, test.commodity)
        ]
    )
    lo = np.maximum(0, tp - q * test.rolling_mean.clip(lower=1))
    hi = tp + q * test.rolling_mean.clip(lower=1)
    baseline = predict(
        models["model_log_l2.txt"], test, {**meta, "target": "log", "blend_weight": 1.0}
    )
    results = [
        {"method": name, **metrics(test.arrivals, p)}
        for name, p in [
            ("rolling_mean", test.rolling_mean.to_numpy()),
            (
                "weekly_seasonal",
                test.weekly_seasonal.fillna(test.rolling_mean).to_numpy(),
            ),
            ("log_lightgbm_same_training", baseline),
            ("deployment_champion", tp),
        ]
    ]
    results[-1].update(
        coverage=float(((test.arrivals >= lo) & (test.arrivals <= hi)).mean()),
        mean_width=float(np.mean(hi - lo)),
    )
    pd.DataFrame(results).to_csv(OUT / "test_metrics.csv", index=False)
    replay = test[
        ["date", "state", "market", "commodity", "arrivals", "rolling_mean"]
    ].copy()
    replay["prediction"] = tp
    replay["lower"] = lo
    replay["upper"] = hi
    replay.to_parquet(OUT / "test_replay.parquet", index=False)
    series = [
        {
            "state": s,
            "market": m,
            "commodity": c,
            **metrics(g.arrivals, g.prediction),
            "coverage": float(
                ((g.arrivals >= g.lower) & (g.arrivals <= g.upper)).mean()
            ),
        }
        for (s, m, c), g in replay.groupby(CATEGORIES)
    ]
    pd.DataFrame(series).to_csv(OUT / "series_metrics.csv", index=False)
    meta["data_cutoff"] = str(combined.date.max().date())
    meta["test_metrics"] = results[-1]
    meta["partitions"] = {
        "final_training_rows": len(train),
        "calibration_rows": len(cal),
        "forward_test_rows": len(test),
        "rolling_validation_folds": folds,
    }
    meta["version"] = hashlib.sha256(
        (
            "".join(digest(OUT / n) for n in sorted(models))
            + json.dumps(routing, sort_keys=True)
        ).encode()
    ).hexdigest()[:16]
    save_json(OUT / "model_card.json", meta)
    files = {
        p.name: digest(p)
        for p in OUT.iterdir()
        if p.is_file() and p.name != "manifest.json"
    }
    save_json(
        OUT / "manifest.json",
        {
            "model_version": meta["version"],
            "files": files,
            "seed_csv_sha256": digest(
                ROOT / "data/processed/deployment_observations.csv"
            ),
            "source_hashes": {
                str(p.relative_to(ROOT)): digest(p)
                for p in (ROOT / "scrc/production").glob("*.py")
            },
        },
    )
    summary = {
        "routing": routing,
        "test_metrics": results,
        "version": meta["version"],
        "protocol": meta["selection_rule"],
        "frozen_before_forward_acquisition_utc": meta[
            "frozen_before_forward_acquisition_utc"
        ],
    }
    save_json(ROOT / "reports/DEPLOYMENT_EVALUATION.json", summary)
    lines = [
        "# Deployment evaluation: fresh 2024 forward holdout",
        "",
        meta["selection_rule"],
        "",
        "Models and 95% split-conformal quantiles were frozen before fetching the 12 January–March 2024 AGMARKNET source files. Earlier 2023 exploratory results are preserved separately. The 95% nominal interval is a conservative planning choice; empirical coverage is reported without a shift guarantee.",
        "",
        "| Method | MAE (tonnes) | RMSE (tonnes) | WAPE | R² | Coverage |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['method']} | {r['mae']:.3f} | {r['rmse']:.3f} | {r['wape']:.2%} | {r['r2']:.3f} | {format(r['coverage'],'.2%') if 'coverage' in r else '—'} |"
        )
    lines += [
        "",
        "WAPE measures absolute error relative to total observed arrivals. It is not classification accuracy. Model selection never uses the forward test metrics. The same-training log model provides a stronger baseline than comparing only with the older research model.",
        "",
        "Scope: rice, wheat, onion and potato in 16 Uttar Pradesh markets. Model training ends September 2023; calibration ends December 2023. The new source observations extend the local history through March 2024, not to the present day. Live planning requires refreshed observations. The reliability gate never executes unverified causal interventions. No actual stock-out reductions or monetary savings are established.",
    ]
    (ROOT / "reports/DEPLOYMENT_EVALUATION.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
