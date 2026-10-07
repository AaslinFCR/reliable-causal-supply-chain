"""Time-separated boosted-tree forecaster."""

from lightgbm import LGBMRegressor


def forecaster(seed=42):
    """Return fixed model settings, without test-period tuning."""
    return LGBMRegressor(
        n_estimators=180,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=40,
        reg_lambda=2,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
