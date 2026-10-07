"""Select on validation, calibrate separately, and benchmark a frozen deployment model."""

import hashlib
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from scrc.common import ROOT, digest, save_json
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

OUT = ROOT / "artifacts/production"


def metrics(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    err = y - p
    return {
        "n": len(y),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "wape": float(np.abs(err).sum() / np.maximum(np.abs(y).sum(), 1e-12)),
        "smape": float(
            np.mean(2 * np.abs(err) / np.maximum(np.abs(y) + np.abs(p), 1e-8))
        ),
        "r2": float(1 - np.sum(err**2) / np.sum((y - y.mean()) ** 2)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    observations = pd.read_parquet(ROOT / "data/processed/market_observations.parquet")
    panel = build_features(observations)
    data = panel.loc[eligible(panel)].copy()
    train = data.loc[data.date < "2023-01-01"]
    validation = data.loc[(data.date >= "2023-01-01") & (data.date < "2023-04-01")]
    calibration = data.loc[(data.date >= "2023-04-01") & (data.date < "2023-07-01")]
    test = data.loc[data.date >= "2023-07-01"]
    vocab = vocabulary(train)
    x = encode(train, vocab)
    candidates = []
    fitted = {}
    specs = [
        ("log_l2", "regression", "log", 15, 400),
        ("raw_l1", "regression_l1", "raw", 31, 500),
        ("raw_l2", "regression", "raw", 31, 400),
        ("tweedie", "tweedie", "raw", 31, 500),
        ("poisson", "poisson", "raw", 15, 400),
        ("ratio_l1", "regression_l1", "ratio", 15, 400),
        ("raw_l1_small", "regression_l1", "raw", 15, 400),
        ("huber", "huber", "raw", 31, 500),
    ]
    for name, objective, target, leaves, estimators in specs:
        print(f"Fitting validation candidate: {name}", flush=True)
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
        model.fit(x, y, categorical_feature=[c + "_code" for c in CATEGORIES])
        fitted[name] = model
        for weight in [1.0, 0.75, 0.5]:
            meta = {"vocabulary": vocab, "target": target, "blend_weight": weight}
            p = predict(model.booster_, validation, meta)
            candidates.append(
                {
                    "name": name,
                    "target": target,
                    "blend_weight": weight,
                    **metrics(validation.arrivals, p),
                }
            )
    baseline = metrics(validation.arrivals, validation.rolling_mean)
    feasible = [r for r in candidates if r["rmse"] <= baseline["rmse"] * 1.1]
    pool = feasible or candidates
    winner = min(pool, key=lambda r: (r["wape"], r["rmse"], r["name"]))
    pd.DataFrame(candidates).to_csv(OUT / "validation_search.csv", index=False)
    meta = {
        "vocabulary": vocab,
        "features": FEATURES,
        "target": winner["target"],
        "blend_weight": winner["blend_weight"],
        "candidate": winner["name"],
        "nominal_coverage": 0.9,
        "training_end": "2022-12-31",
        "validation_start": "2023-01-01",
        "validation_end": "2023-03-31",
        "calibration_start": "2023-04-01",
        "calibration_end": "2023-06-30",
        "test_start": "2023-07-01",
        "test_end": "2023-12-31",
        "selection_rule": "Minimum validation WAPE among candidates with RMSE <= 1.1 * validation rolling-mean RMSE. If none, minimum validation WAPE.",
        "scope": {
            "states": sorted(observations.state.unique()),
            "commodities": sorted(observations.commodity.unique()),
        },
        "data_cutoff": str(observations.date.max().date()),
        "created_utc": datetime.now(UTC).isoformat(),
    }
    selected = fitted[winner["name"]].booster_
    selected.save_model(str(OUT / "forecast_model.txt"))
    cp = predict(selected, calibration, meta)
    normalized = (
        np.abs(calibration.arrivals.to_numpy() - cp)
        / calibration.rolling_mean.clip(lower=1).to_numpy()
    )
    scores = calibration[["state", "commodity"]].copy()
    scores["residual"] = normalized
    meta["conformal_quantiles"] = {
        f"{s}|{c}": conformal_quantile(g.residual, 0.1)
        for (s, c), g in scores.groupby(["state", "commodity"])
    }
    meta["references"] = {
        f"{s}|{m}|{c}": {
            "median_arrivals": float(g.arrivals.median()),
            "median_price": float(g.modal_price.median()),
            "rows": len(g),
        }
        for (s, m, c), g in train.groupby(CATEGORIES)
    }
    result = []
    baseline_specs = [
        ("rolling_mean", test.rolling_mean.to_numpy()),
        ("weekly_seasonal", test.weekly_seasonal.fillna(test.rolling_mean).to_numpy()),
    ]
    log_meta = {**meta, "target": "log", "blend_weight": 1.0}
    baseline_specs.append(
        (
            "log_lightgbm_same_training",
            predict(fitted["log_l2"].booster_, test, log_meta),
        )
    )
    tp = predict(selected, test, meta)
    q = np.array(
        [
            meta["conformal_quantiles"][f"{s}|{c}"]
            for s, c in zip(test.state, test.commodity)
        ]
    )
    lo = np.maximum(0, tp - q * test.rolling_mean.clip(lower=1))
    hi = tp + q * test.rolling_mean.clip(lower=1)
    for name, p in [*baseline_specs, ("deployment_champion", tp)]:
        row = {"method": name, **metrics(test.arrivals, p)}
        if name == "deployment_champion":
            row.update(
                coverage=float(((test.arrivals >= lo) & (test.arrivals <= hi)).mean()),
                mean_width=float(np.mean(hi - lo)),
            )
        result.append(row)
    pd.DataFrame(result).to_csv(OUT / "test_metrics.csv", index=False)
    replay = test[
        ["date", "state", "market", "commodity", "arrivals", "rolling_mean"]
    ].copy()
    replay["prediction"] = tp
    replay["lower"] = lo
    replay["upper"] = hi
    replay.to_parquet(OUT / "test_replay.parquet", index=False)
    by_series = []
    for (s, m, c), g in replay.groupby(CATEGORIES):
        by_series.append(
            {
                "state": s,
                "market": m,
                "commodity": c,
                **metrics(g.arrivals, g.prediction),
                "coverage": float(
                    ((g.arrivals >= g.lower) & (g.arrivals <= g.upper)).mean()
                ),
            }
        )
    pd.DataFrame(by_series).to_csv(OUT / "series_metrics.csv", index=False)
    partitions = {
        name: {
            "rows": len(part),
            "first": str(part.date.min().date()),
            "last": str(part.date.max().date()),
        }
        for name, part in [
            ("train", train),
            ("validation", validation),
            ("calibration", calibration),
            ("test", test),
        ]
    }
    meta["partitions"] = partitions
    meta["version"] = hashlib.sha256(
        (
            digest(OUT / "forecast_model.txt") + json.dumps(winner, sort_keys=True)
        ).encode()
    ).hexdigest()[:16]
    meta["test_metrics"] = result[-1]
    save_json(OUT / "model_card.json", meta)
    hashes = {
        p.name: digest(p)
        for p in OUT.iterdir()
        if p.is_file() and p.name != "manifest.json"
    }
    save_json(
        OUT / "manifest.json",
        {
            "model_version": meta["version"],
            "files": hashes,
            "observation_sha256": digest(
                ROOT / "data/processed/market_observations.parquet"
            ),
            "source_hashes": {
                str(p.relative_to(ROOT)): digest(p)
                for p in (ROOT / "scrc/production").glob("*.py")
            },
        },
    )
    summary = {
        "winner": winner,
        "partitions": partitions,
        "test_metrics": result,
        "model_version": meta["version"],
    }
    save_json(ROOT / "reports/DEPLOYMENT_EVALUATION.json", summary)
    lines = [
        "# Deployment forecasting evaluation",
        "",
        f"Selected model: **{winner['name']}**, weekly-seasonal blend weight {winner['blend_weight']}.",
        "",
        "Selection uses January–March 2023 only. April–June is a separate, untouched calibration set. Models train through December 2022. July–December test results were previously observed in the earlier research experiment, so this is a controlled regression benchmark, not a newly blinded holdout. No candidate is selected from test performance.",
        "",
        "| Method | MAE (tonnes) | RMSE (tonnes) | WAPE | R² | Coverage |",
        "|---|---|---|---|---|---|",
    ]
    for row in result:
        lines.append(
            f"| {row['method']} | {row['mae']:.3f} | {row['rmse']:.3f} | {row['wape']:.2%} | {row['r2']:.3f} | {format(row['coverage'],'.2%') if 'coverage' in row else '—'} |"
        )
    lines += [
        "",
        "WAPE is absolute error divided by total observed arrivals; it is not classification accuracy. All predictions use earlier dates only. Calibration coverage is empirical and does not establish arbitrary-shift guarantees. The deployment experiment uses more recent training data and richer features than the original research experiment; the same-training log-LightGBM baseline is included for a fairer comparison.",
        "",
        "The market data end in December 2023 and cover Uttar Pradesh. Without refreshed operational inputs, this application supports historical demonstrations and user-supplied histories. No validated causal automation, actual stock-out reduction, or monetary savings is established.",
    ]
    (ROOT / "reports/DEPLOYMENT_EVALUATION.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
