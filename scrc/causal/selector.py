"""Conservative action selection relative to an explicit Carting reference."""

import numpy as np


def select(benefit_lower, overlap):
    """Choose FTL only for supported positive lower-bound benefit; otherwise Carting."""
    return (np.asarray(overlap) & (np.asarray(benefit_lower) > 0)).astype(int)
