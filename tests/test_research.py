"""Mathematical and real-record regression checks; no synthetic experiment data."""

import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest

from scrc.causal.effects import clustered_interval
from scrc.causal.selector import select
from scrc.common import ROOT, config, digest
from scrc.data.panel import chronological_splits, logistics_trips, market_features
from scrc.gate.gate import gate
from scrc.models.conformal import AdaptiveConformal, conformal_quantile
from scrc.models.shift_detector import shift_score


@pytest.fixture(scope="module")
def observations():
    return pd.read_parquet(ROOT / "data/processed/market_observations.parquet")


@pytest.fixture(scope="module")
def trips():
    return pd.read_parquet(ROOT / "data/processed/trips.parquet")


@pytest.fixture(scope="module")
def scans():
    return pd.read_csv(ROOT / "data/raw/delhivery.csv")


def test_real_data_provenance():
    official = config().get("market_source") == "official"
    manifest = json.loads(
        (
            ROOT
            / (
                "data/raw/official_manifest.json"
                if official
                else "data/raw/archive_manifest.json"
            )
        ).read_text()
    )
    from scrc.data.official import official_jobs

    assert len(manifest) == (len(official_jobs(config())) if official else 7)
    for source in manifest:
        assert digest(ROOT / source["file"]) == source["sha256"]
    if official:
        assert {
            tuple(
                s["parameters"][k] for k in ["stateId", "commodityId", "year", "month"]
            )
            for s in manifest
        } == set(official_jobs(config()))
        record = json.loads((ROOT / "data/raw/delhivery_manifest.json").read_text())
        assert digest(ROOT / record["file"]) == record["sha256"]


def test_official_varieties_do_not_duplicate_arrivals():
    import gzip

    from scrc.data.official import rows

    path = next((ROOT / "data/raw/official").glob("*.json.gz"))
    actual = list(rows(path))
    source = json.loads(gzip.decompress(path.read_bytes()))
    original = {
        (m["marketName"], d["arrivalDate"]): d
        for m in source["markets"]
        for d in m["dates"]
    }
    assert actual
    for row in actual:
        day = original[row["market"], row["date"].strftime("%d/%m/%Y")]
        assert row["arrivals"] == float(day["total_arrivals"])


def test_official_response_identity_and_month():
    if config().get("market_source") != "official":
        pytest.skip("Official-source experiment only")
    from scrc.data.official import fetch

    manifest = json.loads((ROOT / "data/raw/official_manifest.json").read_text())
    from scrc.data.official import official_jobs

    assert len(manifest) == len(official_jobs(config()))
    for source in manifest:
        p = source["parameters"]
        checked = fetch(
            tuple(p[k] for k in ["stateId", "commodityId", "year", "month"])
        )
        assert checked["response_sha256"] == source["response_sha256"]


def test_no_duplicate_market_keys(observations):
    assert not observations.duplicated(["state", "market", "commodity", "date"]).any()


def test_real_target_size_not_calendar_padding(observations):
    card = json.loads((ROOT / "reports/market_data_card.json").read_text())
    assert len(observations) == card["selected_real_observations"]
    assert observations.arrivals.notna().all()
    assert card["calendar_panel_rows"] >= len(observations)


def test_calendar_gaps_are_not_imputed():
    panel = pd.read_parquet(ROOT / "data/processed/panel.parquet")
    assert panel.loc[~panel.observed, "arrivals"].isna().all()
    assert panel.loc[~panel.observed, "shortage_proxy"].isna().all()


def test_split_dates_do_not_overlap(observations):
    parts = chronological_splits(observations, config())
    for left, right in itertools.pairwise(parts):
        assert left.date.max() < right.date.min()
    assert sum(map(len, parts)) == len(observations)


def test_features_ignore_current_and_future_targets(observations):
    counts = observations.groupby(["state", "market", "commodity"]).size()
    key = counts.idxmax()
    source = observations.loc[
        (observations.state == key[0])
        & (observations.market == key[1])
        & (observations.commodity == key[2])
    ].copy()
    cutoff = source.date.sort_values().iloc[len(source) // 2]
    altered = source.copy()
    altered.loc[altered.date >= cutoff, "arrivals"] *= 10
    altered.loc[altered.date >= cutoff, "modal_price"] *= 5
    before = market_features(source, config())
    after = market_features(altered, config())
    cols = [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean",
        "rolling_std",
        "price_lag",
    ]
    pd.testing.assert_frame_equal(
        before.loc[before.date <= cutoff, cols], after.loc[after.date <= cutoff, cols]
    )


def test_quantile_finite_sample_rank(observations):
    actual = observations.arrivals.head(100).to_numpy()
    alpha = 0.1
    expected = np.sort(actual)[math.ceil((len(actual) + 1) * (1 - alpha)) - 1]
    assert conformal_quantile(actual, alpha) == expected


def test_quantile_insufficient_calibration_is_infinite(observations):
    assert np.isinf(conformal_quantile(observations.arrivals.head(1), 0.1))


def test_empty_calibration_fails():
    with pytest.raises(ValueError, match="observed residuals"):
        conformal_quantile([], 0.1)


def test_adaptation_feedback_order(observations):
    scores = observations.arrivals.head(100).to_numpy()
    online = AdaptiveConformal(scores, alpha=0.1, gamma=0.01)
    first = online.quantile()
    alpha = online.alpha
    assert first == conformal_quantile(scores, alpha)
    online.update(scores[:1], [True])
    assert online.alpha < alpha


@pytest.mark.parametrize(
    "width,shift,benefit,supported,expected",
    [
        (1.0, 0.1, 1.0, True, "GREEN"),
        (2.0, 0.1, 1.0, True, "AMBER"),
        (1.0, 0.7, 1.0, True, "AMBER"),
        (1.0, 0.7, -1.0, True, "RED"),
        (1.0, 0.1, -1.0, True, "AMBER"),
        (1.0, 0.1, 1.0, False, "RED"),
        (1.0, 0.1, float("nan"), True, "RED"),
    ],
)
def test_gate_truth_table(width, shift, benefit, supported, expected):
    assert gate(width, shift, benefit, supported, config()) == expected


def test_selector_requires_overlap_and_positive_bound():
    assert select([-1, 0, 1, 1], [True, True, True, False]).tolist() == [0, 0, 1, 0]


def test_one_row_per_trip(trips, scans):
    assert trips.trip_uuid.is_unique
    assert len(trips) <= scans.trip_uuid.nunique()
    assert set(trips.trip_uuid) <= set(scans.trip_uuid)


def test_cumulative_scans_not_summed(trips, scans):
    trip = trips.iloc[0]
    records = scans.loc[scans.trip_uuid == trip.trip_uuid]
    expected = (
        records.groupby(["source_center", "destination_center"]).actual_time.max().sum()
    )
    assert trip.actual_time == expected
    assert trip.actual_time <= records.actual_time.sum()


def test_invalid_osrm_is_excluded(scans):
    trip_id = scans.trip_uuid.iloc[0]
    subset = scans.loc[scans.trip_uuid == trip_id].copy()
    subset["osrm_time"] = 0.0
    assert logistics_trips(subset).empty


def test_trip_treatment_is_constant(scans):
    assert scans.groupby("trip_uuid").route_type.nunique().max() == 1


def test_trip_endpoints_follow_leg_chronology(trips, scans):
    for _, trip in trips.loc[trips.legs > 1].head(20).iterrows():
        records = scans.loc[scans.trip_uuid == trip.trip_uuid]
        first = records.sort_values("od_start_time").iloc[0]
        last = records.sort_values("od_end_time").iloc[-1]
        assert trip.source_name == first.source_name
        assert trip.destination_name == last.destination_name


def test_bootstrap_reproducible(observations):
    data = observations.head(200)
    first = clustered_interval(data.arrivals, data.date, repeats=20, seed=42)
    assert first == clustered_interval(data.arrivals, data.date, repeats=20, seed=42)


def test_no_drift_against_identical_reference(trips):
    actual = trips[["log_distance", "weekday", "hour"]].head(100).to_numpy()
    assert shift_score(actual, actual, 0.9) == 0.0


def test_replay_artifacts_have_no_unsupported_green_actions():
    path = ROOT / "experiments/logistics_replay.parquet"
    if not path.exists():
        pytest.skip("Run evaluation first")
    data = pd.read_parquet(path)
    assert set(data.gate) == {"RED"}


def test_policy_estimates_use_overlap_population():
    path = ROOT / "experiments/logistics_replay.parquet"
    if not path.exists():
        pytest.skip("Run evaluation first")
    data = pd.read_parquet(path)
    selected = data.loc[data.overlap]
    e = selected.propensity.to_numpy()
    assert ((e >= config()["propensity_min"]) & (e <= config()["propensity_max"])).all()
    assert not data.loc[~data.overlap, "conservative_policy"].any()


def test_event_is_in_test_period():
    path = ROOT / "experiments/market_replay.parquet"
    if not path.exists():
        pytest.skip("Run evaluation first")
    data = pd.read_parquet(path)
    assert (
        "onion_export_prohibition_2023"
        if config().get("market_source") == "official"
        else "demonetisation_2016"
    ) in set(data.event)
    assert set(data.split) == {"test"}


def test_gate_tuning_uses_validation_only():
    path = ROOT / "experiments/market_validation.parquet"
    if not path.exists():
        pytest.skip("Run evaluation first")
    data = pd.read_parquet(path)
    assert set(data.split) == {"validation"}
    sweep = pd.read_csv(ROOT / "experiments/threshold_sweep_validation.csv")
    assert (sweep.n <= len(data)).all()


def test_exported_market_model_reproduces_replay():
    from lightgbm import Booster

    folder = ROOT / "experiments/models"
    if not (folder / "market_lightgbm.txt").exists():
        pytest.skip("Run evaluation to export the fitted model")
    meta = json.loads((folder / "market_feature_encoding.json").read_text())
    actual = pd.read_parquet(ROOT / "experiments/market_replay.parquet").head(100)
    features = actual.copy()
    for column, values in meta["categorical_vocabulary"].items():
        features[column + "_code"] = (
            features[column]
            .map({value: i for i, value in enumerate(values)})
            .fillna(-1)
        )
    for column in meta["log1p_columns"]:
        features[column] = np.log1p(features[column])
    loaded = Booster(model_file=str(folder / "market_lightgbm.txt"))
    predictions = np.maximum(0, np.expm1(loaded.predict(features[meta["features"]])))
    np.testing.assert_allclose(predictions, actual.prediction, rtol=1e-12, atol=1e-10)
