"""Reliability gate with an explicit causal-identification precondition."""

import numpy as np


def gate(width_ratio, shift, benefit_lower, supported, cfg):
    """Do not automate unsupported interventions, even with narrow forecasts."""
    if not supported or not np.isfinite(benefit_lower):
        return "RED"
    if (
        width_ratio <= cfg["tau_width"]
        and shift <= cfg["tau_shift"]
        and benefit_lower > 0
    ):
        return "GREEN"
    if benefit_lower > 0 or shift <= cfg["tau_shift_high"]:
        return "AMBER"
    return "RED"
