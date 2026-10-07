"""Reference policy definitions."""

import numpy as np


def policy_baselines(size):
    """Constant route policies evaluated only on the same overlap population."""
    return {
        "always_carting": np.zeros(size, dtype=int),
        "always_ftl": np.ones(size, dtype=int),
    }
