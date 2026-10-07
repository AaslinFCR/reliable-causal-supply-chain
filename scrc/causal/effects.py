"""T-learner outcome models and an independent propensity model."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scrc.models.forecaster import forecaster

NUMERIC = ["log_distance", "weekday", "hour"]
CATEGORICAL = ["source_state", "destination_state"]
FEATURES = NUMERIC + CATEGORICAL


class ObservationalEffects:
    """Estimate delay contrast for FTL versus Carting; identification is conditional."""

    def __init__(self, seed=42):
        self.seed = seed
        self.transform = ColumnTransformer(
            [
                ("numeric", StandardScaler(), NUMERIC),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    CATEGORICAL,
                ),
            ]
        )

    def fit(self, train):
        """Fit nuisance models using historical observations only."""
        x = self.transform.fit_transform(train[FEATURES].fillna("Unknown"))
        a = train.treatment.to_numpy()
        y = train.delay_factor.to_numpy()
        if set(a) != {0, 1}:
            raise ValueError("Both observed route types are required.")
        self.models = []
        for arm in [0, 1]:
            model = forecaster(self.seed + arm)
            model.fit(x[a == arm], y[a == arm])
            self.models.append(model)
        self.propensity = LogisticRegression(
            max_iter=1500, C=1, random_state=self.seed
        ).fit(x, a)
        self.feature_bounds = {
            col: (float(train[col].min()), float(train[col].max())) for col in NUMERIC
        }
        return self

    def predict(self, frame):
        """Return two potential-outcome regressions and treatment probabilities."""
        x = self.transform.transform(frame[FEATURES].fillna("Unknown"))
        return (
            self.models[0].predict(x),
            self.models[1].predict(x),
            self.propensity.predict_proba(x)[:, 1],
        )

    def interval(self, train, target, cfg):
        """Date-cluster bootstrap refits estimate conditional benefit uncertainty."""
        rng = np.random.default_rng(cfg["seed"])
        dates = train.date.unique()
        draws = []
        for b in range(cfg["causal_bootstrap_models"]):
            chosen = rng.choice(dates, size=len(dates), replace=True)
            sample = __import__("pandas").concat(
                [train.loc[train.date == d] for d in chosen], ignore_index=True
            )
            if sample.treatment.nunique() < 2:
                continue
            model = ObservationalEffects(cfg["seed"] + b + 10).fit(sample)
            m0, m1, _ = model.predict(target)
            draws.append(m0 - m1 - cfg["action_cost"])
        if len(draws) < 10:
            raise ValueError("Insufficient successful date-cluster bootstrap fits.")
        return np.quantile(draws, [0.025, 0.975], axis=0)


def doubly_robust_scores(frame, m0, m1, e, policy):
    """Evaluate held-out loss under a fixed policy on a prespecified overlap set."""
    a = frame.treatment.to_numpy()
    y = frame.delay_factor.to_numpy()
    if np.any((e <= 0) | (e >= 1)):
        raise ValueError("Propensities must be strictly between zero and one.")
    v0 = m0 + (a == 0) * (y - m0) / (1 - e)
    v1 = m1 + (a == 1) * (y - m1) / e
    return np.where(np.asarray(policy) == 1, v1, v0)


def clustered_interval(values, dates, repeats=300, seed=42):
    """Bootstrap whole observed dates, retaining within-date dependence."""
    import pandas as pd

    tmp = pd.DataFrame({"v": values, "date": np.asarray(dates)})
    grouped = tmp.groupby("date").v.agg(["sum", "count"])
    if len(grouped) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(grouped), size=(repeats, len(grouped)))
    draws = grouped["sum"].to_numpy()[idx].sum(axis=1) / grouped["count"].to_numpy()[
        idx
    ].sum(axis=1)
    return np.quantile(draws, [0.025, 0.975]).tolist()
