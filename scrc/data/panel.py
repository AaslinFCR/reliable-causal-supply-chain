"""Market features and an explicitly labelled shortage-pressure proxy."""

import numpy as np
import pandas as pd


def chronological_splits(frame, cfg, time="date", cut_dates=None):
    """Keep whole dates together in train/calibration/validation/test order."""
    dates = np.sort(frame[time].unique())
    if len(dates) < 12:
        raise ValueError("Insufficient distinct dates for four chronological splits.")
    ends = np.cumsum(
        [cfg["train_fraction"], cfg["calibration_fraction"], cfg["validation_fraction"]]
    )
    cuts = [dates[min(len(dates) - 1, int(len(dates) * x))] for x in ends]
    if cut_dates is not None:
        cuts = [pd.Timestamp(value) for value in cut_dates]
        if len(cuts) != 3 or not cuts[0] < cuts[1] < cuts[2]:
            raise ValueError("Exactly three increasing split dates are required.")
    parts = [
        frame.loc[frame[time] < cuts[0]].copy(),
        frame.loc[(frame[time] >= cuts[0]) & (frame[time] < cuts[1])].copy(),
        frame.loc[(frame[time] >= cuts[1]) & (frame[time] < cuts[2])].copy(),
        frame.loc[frame[time] >= cuts[2]].copy(),
    ]
    if any(part.empty for part in parts):
        raise ValueError(
            "Chronological partitions must all contain actual observations."
        )
    return parts


def market_features(raw, cfg):
    """Create past-only features; retain missing dates without inventing observations."""
    pieces = []
    for key, group in raw.groupby(["state", "market", "commodity"], observed=True):
        g = group.sort_values("date").set_index("date").asfreq("D")
        for column, value in zip(["state", "market", "commodity"], key):
            g[column] = value
        past = g["arrivals"].shift(1)
        for lag in [1, 7, 14, 28]:
            g[f"lag_{lag}"] = g["arrivals"].shift(lag)
        g["rolling_mean"] = past.rolling(
            cfg["feature_window"], min_periods=cfg["min_history"]
        ).mean()
        g["rolling_std"] = past.rolling(
            cfg["feature_window"], min_periods=cfg["min_history"]
        ).std()
        g["price_lag"] = g["modal_price"].shift(1)
        q = (
            g["modal_price"]
            .shift(1)
            .rolling(cfg["feature_window"], min_periods=cfg["min_history"])
            .quantile(cfg["stockout_price_quantile"])
        )
        valid = (
            g["arrivals"].notna()
            & g["modal_price"].notna()
            & q.notna()
            & g["rolling_mean"].notna()
        )
        g["shortage_proxy"] = np.where(
            valid,
            (
                (g["arrivals"] < cfg["stockout_k"] * g["rolling_mean"])
                & (g["modal_price"] > q)
            ).astype(float),
            np.nan,
        )
        g["weekday"] = g.index.dayofweek
        g["month"] = g.index.month
        g["year_day"] = g.index.dayofyear
        g["observed"] = g["arrivals"].notna()
        pieces.append(g.reset_index())
    result = pd.concat(pieces, ignore_index=True)
    return result.sort_values(["date", "state", "market", "commodity"]).reset_index(
        drop=True
    )


def logistics_trips(raw):
    """Collapse cumulative scan rows to OD legs, then trips; avoid scan pseudoreplication."""
    needed = [
        "trip_uuid",
        "trip_creation_time",
        "route_type",
        "source_center",
        "destination_center",
        "actual_time",
        "osrm_time",
        "osrm_distance",
        "source_name",
        "destination_name",
        "od_end_time",
        "od_start_time",
    ]
    missing = set(needed) - set(raw)
    if missing:
        raise ValueError(f"Missing Delhivery columns: {sorted(missing)}")
    if raw.groupby("trip_uuid")["route_type"].nunique().max() != 1:
        raise ValueError("Trip treatment changes within a trip.")
    legs = raw.groupby(
        ["trip_uuid", "source_center", "destination_center"], as_index=False
    ).agg(
        actual_time=("actual_time", "max"),
        osrm_time=("osrm_time", "max"),
        distance=("osrm_distance", "max"),
        source_name=("source_name", "first"),
        destination_name=("destination_name", "first"),
        end_time=("od_end_time", "max"),
        leg_start_time=("od_start_time", "min"),
        route_type=("route_type", "first"),
        trip_creation_time=("trip_creation_time", "first"),
    )
    legs = legs.sort_values(["trip_uuid", "leg_start_time", "end_time"], kind="stable")
    trips = legs.groupby("trip_uuid", as_index=False).agg(
        actual_time=("actual_time", "sum"),
        osrm_time=("osrm_time", "sum"),
        distance=("distance", "sum"),
        legs=("distance", "size"),
        source_name=("source_name", "first"),
        destination_name=("destination_name", "last"),
        route_type=("route_type", "first"),
        trip_creation_time=("trip_creation_time", "first"),
        end_time=("end_time", "max"),
    )
    trips["date"] = pd.to_datetime(
        trips.trip_creation_time, format="mixed"
    ).dt.normalize()
    trips["end_time"] = pd.to_datetime(trips.end_time, format="mixed")
    trips["source_state"] = trips.source_name.str.extract(r"\(([^)]+)\)$", expand=False)
    trips["destination_state"] = trips.destination_name.str.extract(
        r"\(([^)]+)\)$", expand=False
    )
    trips["delay_factor"] = trips.actual_time / trips.osrm_time
    trips["treatment"] = (trips.route_type == "FTL").astype(int)
    trips["weekday"] = trips.date.dt.dayofweek
    trips["hour"] = pd.to_datetime(trips.trip_creation_time, format="mixed").dt.hour
    # Distance is a retrospective OSRM route measurement, not a verified pre-dispatch field.
    trips["log_distance"] = np.log1p(trips.distance)
    good = (
        (trips.osrm_time > 0)
        & (trips.actual_time > 0)
        & (trips.distance > 0)
        & np.isfinite(trips.delay_factor)
    )
    return trips.loc[good].sort_values(["date", "trip_uuid"]).reset_index(drop=True)
