"""Falsification and specification checks; these cannot prove identification."""

import numpy as np

from scrc.causal.effects import ObservationalEffects


def diagnostics(train, test, model, cfg):
    """Run fixed-seed placebo and subset checks using observed data only."""
    m0, m1, _e = model.predict(test)
    result = {
        "interpretation": "Exploratory diagnostics, not proof of no unmeasured confounding.",
        "unadjusted_delay_contrast_carting_minus_ftl": float(
            train.groupby("treatment").delay_factor.mean().loc[0]
            - train.groupby("treatment").delay_factor.mean().loc[1]
        ),
        "adjusted_mean_contrast": float(np.mean(m0 - m1)),
    }
    # Permuting existing labels is a placebo refutation, not a synthetic research dataset.
    placebo = train.copy()
    rng = np.random.default_rng(cfg["seed"])
    placebo["treatment"] = rng.permutation(placebo.treatment.to_numpy())
    p0, p1, _ = ObservationalEffects(cfg["seed"]).fit(placebo).predict(test)
    result["permuted_treatment_mean_contrast"] = float(np.mean(p0 - p1))
    early = train.loc[
        train.date <= train.date.sort_values().iloc[int(len(train) * 0.75)]
    ]
    s0, s1, _ = ObservationalEffects(cfg["seed"]).fit(early).predict(test)
    result["earlier_subset_mean_contrast"] = float(np.mean(s0 - s1))
    # Omit potentially treatment-dependent retrospective distance, reusing actual records.
    no_distance = train.copy()
    no_distance["log_distance"] = 0.0
    test_no_distance = test.copy()
    test_no_distance["log_distance"] = 0.0
    d0, d1, _ = (
        ObservationalEffects(cfg["seed"]).fit(no_distance).predict(test_no_distance)
    )
    result["omit_retrospective_distance_mean_contrast"] = float(np.mean(d0 - d1))
    result["unmeasured_bias_sweep"] = [
        {
            "additive_bias_delay_factor": float(b),
            "mean_contrast_after_bias": float(np.mean(m0 - m1) - b),
        }
        for b in [0, 0.1, 0.25, 0.5, 1.0]
    ]
    result["bias_sweep_caveat"] = (
        "Illustrative additive-bias scenario, not a formal identification bound or validated confounding model."
    )
    result["random_common_cause"] = (
        "Not used: adds invented covariates and cannot validate this observational design."
    )
    return result
