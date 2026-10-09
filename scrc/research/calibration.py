"""Date-batched, past-only conformal comparisons with auditable sparse fallback."""

import math
from collections import deque

import numpy as np
import pandas as pd


def quantile(scores, alpha):
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    if not len(values) or rank > len(values):
        return float("inf")
    return float(np.partition(values, max(0, rank - 1))[max(0, rank - 1)])


def interval_score(y, lower, upper, alpha=0.05):
    y, lower, upper = map(np.asarray, (y, lower, upper))
    return (upper - lower + 2 / alpha * (lower - y) * (y < lower)
            + 2 / alpha * (y - upper) * (y > upper))


def calibrate(calibration, test, method, window=500, gamma=0.005,
              minimum_count=100, alpha=0.05, alpha_clip=(0.005, 0.2)):
    """Each complete date batch gets intervals before any of its outcomes update state."""
    if calibration.date.max() >= test.date.min():
        raise ValueError("Calibration must strictly precede evaluation")
    initial = np.abs(calibration.actual - calibration.prediction) / calibration.scale
    pooled = quantile(initial, alpha)
    if not np.isfinite(pooled):
        raise ValueError("Pooled calibration lacks finite-sample support")
    histories = {crop: deque(initial.loc[calibration.commodity == crop].tolist(),
                             maxlen=window if method == "commodity_adaptive" else None)
                 for crop in sorted(calibration.commodity.unique())}
    alphas = {crop: alpha for crop in histories}
    rows, fallback, clips = [], 0, 0
    previous = calibration.date.max()
    for date, batch in test.sort_values(["date", "market", "commodity"]).groupby("date", sort=True):
        records = batch.copy()
        qs, flags, current_alphas = [], [], []
        for crop in records.commodity:
            local_alpha = alphas.get(crop, alpha) if method == "commodity_adaptive" else alpha
            history = histories.get(crop, [])
            q = pooled if method == "pooled_fixed" else quantile(history, local_alpha)
            use_fallback = method != "pooled_fixed" and (len(history) < minimum_count or not np.isfinite(q))
            if use_fallback:
                q = pooled
            qs.append(q)
            flags.append(use_fallback)
            current_alphas.append(local_alpha)
        records["lower"] = np.maximum(0, records.prediction - np.asarray(qs) * records.scale)
        records["upper"] = records.prediction + np.asarray(qs) * records.scale
        records["alpha_used"] = current_alphas
        records["fallback"] = flags
        records["latest_calibration_date"] = previous
        fallback += sum(flags)
        # Only now are the complete date batch's outcomes revealed.
        if method == "commodity_adaptive":
            residuals = np.abs(records.actual - records.prediction) / records.scale
            misses = (records.actual < records.lower) | (records.actual > records.upper)
            for crop in records.commodity.unique():
                mask = records.commodity == crop
                before = alphas.get(crop, alpha)
                raw = before + gamma * (alpha - float(misses.loc[mask].mean()))
                clipped = float(np.clip(raw, *alpha_clip))
                clips += int(clipped != raw)
                alphas[crop] = clipped
                histories.setdefault(crop, deque(maxlen=window)).extend(residuals.loc[mask].tolist())
            previous = date
        rows.append(records)
    return pd.concat(rows, ignore_index=True), {"fallback_rows": fallback,
        "alpha_clipping_updates": clips, "method": method, "window": window,
        "gamma": gamma, "minimum_group_count": minimum_count,
        "fallback": "Initial pooled fixed quantile at nominal coverage, never future outcomes"}


def interval_metrics(frame, nominal=0.95):
    inside = (frame.actual >= frame.lower) & (frame.actual <= frame.upper)
    coverage = float(inside.mean())
    return {"n": len(frame), "coverage": coverage,
            "coverage_gap": coverage - nominal,
            "mean_width": float((frame.upper - frame.lower).mean()),
            "mean_interval_score": float(interval_score(frame.actual, frame.lower, frame.upper, 1-nominal).mean())}
