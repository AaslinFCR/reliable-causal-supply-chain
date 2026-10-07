"""Finite-sample split quantiles and chronological adaptive calibration."""

import math

import numpy as np


def conformal_quantile(scores, alpha):
    """Use the finite-sample corrected order statistic; infinity if unavailable."""
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if not len(scores):
        raise ValueError("Calibration requires observed residuals.")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    if rank > len(scores) or alpha <= 0:
        return float("inf")
    if rank <= 0:
        return 0.0
    return float(np.partition(scores, rank - 1)[rank - 1])


class AdaptiveConformal:
    """Update only after the current target is observed, one time batch at a time."""

    def __init__(self, scores, alpha=0.1, gamma=0.01, window=500):
        self.scores = list(scores)[-window:]
        self.target = alpha
        self.alpha = alpha
        self.gamma = gamma
        self.window = window

    def quantile(self):
        return conformal_quantile(self.scores, self.alpha)

    def update(self, scores, misses):
        self.alpha = float(
            np.clip(
                self.alpha + self.gamma * (self.target - np.mean(misses)), 0.001, 0.999
            )
        )
        self.scores = (self.scores + list(scores))[-self.window :]
