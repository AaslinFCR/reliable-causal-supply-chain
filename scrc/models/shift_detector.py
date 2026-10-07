"""Observable feature drift combined with past interval miscoverage."""

import numpy as np
from scipy.stats import ks_2samp


def shift_score(reference, recent, past_coverage, target=0.9):
    """Return effect-size KS drift, avoiding p-value dependence on sample size."""
    if len(recent) < 20:
        return 0.0
    ks = max(
        ks_2samp(np.asarray(reference)[:, i], np.asarray(recent)[:, i]).statistic
        for i in range(np.asarray(reference).shape[1])
    )
    drop = max(0.0, target - past_coverage) / target
    return float(np.clip(0.5 * ks + 0.5 * drop, 0, 1))
