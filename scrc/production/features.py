"""Shared training and serving transformations using strictly past observations."""

import numpy as np
import pandas as pd

from scrc.common import config
from scrc.data.panel import market_features

CATEGORIES = ["state", "market", "commodity"]


def build_features(observations):
    panel = market_features(observations, config())
    pieces = []
    for _, group in panel.groupby(CATEGORIES, sort=True):
        g = group.sort_values("date").copy()
        past = g.arrivals.shift(1)
        for lag in [2, 3, 21]:
            g[f"lag_{lag}"] = g.arrivals.shift(lag)
        for window in [7, 14, 28]:
            roll = past.rolling(window, min_periods=max(3, window // 2))
            g[f"mean_{window}"] = roll.mean()
            g[f"median_{window}"] = roll.median()
            g[f"std_{window}"] = roll.std()
        g["weekly_seasonal"] = g[["lag_7", "lag_14", "lag_21", "lag_28"]].mean(axis=1)
        g["price_mean_7"] = g.modal_price.shift(1).rolling(7, min_periods=3).mean()
        g["price_mean_28"] = g.modal_price.shift(1).rolling(28, min_periods=14).mean()
        g["trend_ratio"] = g.mean_7 / g.rolling_mean.clip(lower=1)
        g["recent_ratio"] = g.lag_1 / g.rolling_mean.clip(lower=1)
        g["weekday_sin"] = np.sin(2 * np.pi * g.weekday / 7)
        g["weekday_cos"] = np.cos(2 * np.pi * g.weekday / 7)
        g["season_sin"] = np.sin(2 * np.pi * g.year_day / 365.25)
        g["season_cos"] = np.cos(2 * np.pi * g.year_day / 365.25)
        pieces.append(g)
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["date", *CATEGORIES])
        .reset_index(drop=True)
    )


FEATURES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",
    "rolling_mean",
    "rolling_std",
    "price_lag",
    "mean_7",
    "mean_14",
    "mean_28",
    "median_7",
    "median_14",
    "median_28",
    "std_7",
    "std_14",
    "std_28",
    "weekly_seasonal",
    "price_mean_7",
    "price_mean_28",
    "trend_ratio",
    "recent_ratio",
    "weekday",
    "month",
    "year_day",
    "weekday_sin",
    "weekday_cos",
    "season_sin",
    "season_cos",
    "state_code",
    "market_code",
    "commodity_code",
]


def vocabulary(train):
    return {c: sorted(train[c].unique()) for c in CATEGORIES}


def encode(frame, vocab):
    data = frame.copy()
    for c in CATEGORIES:
        data[c + "_code"] = (
            data[c].map({v: i for i, v in enumerate(vocab[c])}).fillna(-1).astype(int)
        )
    return data[FEATURES].replace([np.inf, -np.inf], np.nan)


def eligible(panel):
    return panel.observed & panel.rolling_mean.notna() & (panel.rolling_mean > 0)


def predict(booster, frame, meta):
    prediction = booster.predict(encode(frame, meta["vocabulary"]), num_threads=2)
    if meta["target"] == "log":
        prediction = np.expm1(prediction)
    elif meta["target"] == "ratio":
        prediction = prediction * frame.rolling_mean.clip(lower=1).to_numpy()
    prediction = np.maximum(0, prediction)
    weight = meta.get("blend_weight", 1.0)
    return (
        weight * prediction
        + (1 - weight) * frame.weekly_seasonal.fillna(frame.rolling_mean).to_numpy()
    )
