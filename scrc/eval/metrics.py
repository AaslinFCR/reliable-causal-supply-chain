"""Forecast metrics that do not imply unobserved inventory outcomes."""

import numpy as np


def forecast_metrics(y, pred, lower=None, upper=None):
    """Summarise actual observed targets; interval metrics only when available."""
    y = np.asarray(y)
    pred = np.asarray(pred)
    result = {
        "n": len(y),
        "mae": float(np.mean(np.abs(y - pred))),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
    }
    if lower is not None:
        width = np.asarray(upper) - np.asarray(lower)
        result.update(
            coverage=float(np.mean((y >= lower) & (y <= upper))),
            mean_width=float(np.mean(width)),
            infinite_intervals=int((~np.isfinite(width)).sum()),
        )
    return result
