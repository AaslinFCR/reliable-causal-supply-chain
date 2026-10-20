"""Effect estimation for discrete supply-chain interventions from confounded planner logs.

Outcome: 7-day cost, normalised by (unit price x 7-day forecast) so that series of
different size share one model. Lower is better; effect tau_a(x) = E[Y(a) - Y(none) | x].

Learners: naive arm means, S-learner, T-learner, DR-learner (Kennedy 2023) with
cross-fitted propensities (Kunzel et al. 2019 for meta-learners). Uncertainty:
bootstrap ensembles (agreement) and weighted split-conformal counterfactual intervals
(Lei & Candes 2021) using estimated propensity weights.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from . import config as C

K = len(C.ACTIONS)
XCOLS = ["log_fc", "cover", "cover3", "ip_cover", "q_rel", "gap_hi", "gap_lo", "dem_flag", "cusum",
         "transit", "supply", "fill_obs", "fest", "promo", "donor", "exp_lt", "log_price", "region", "product"]


def to_matrix(F):
    return np.column_stack([F[c] for c in XCOLS])


def _reg(seed=0, it=300):
    return HistGradientBoostingRegressor(max_iter=it, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=20, l2_regularization=1.0, random_state=seed)


def _clf(seed=0):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=30, l2_regularization=1.0, random_state=seed)


class Naive:
    name = "Naive"

    def fit(self, X, A, Y):
        self.m = np.array([Y[A == a].mean() for a in range(K)]); return self

    def mu(self, X):
        return np.tile(self.m, (len(X), 1))


class SLearner:
    name = "S-learner"

    def __init__(self, seed=0): self.seed = seed

    def fit(self, X, A, Y):
        self.m = _reg(self.seed).fit(np.column_stack([X, np.eye(K)[A]]), Y); return self

    def mu(self, X):
        return np.column_stack([self.m.predict(np.column_stack([X, np.tile(np.eye(K)[a], (len(X), 1))]))
                                for a in range(K)])


class TLearner:
    name = "T-learner"

    def __init__(self, seed=0, it=300): self.seed, self.it = seed, it

    def fit(self, X, A, Y):
        self.ms = [_reg(self.seed + a, self.it).fit(X[A == a], Y[A == a]) for a in range(K)]; return self

    def mu(self, X):
        return np.column_stack([m.predict(X) for m in self.ms])


class Propensity:
    def __init__(self, seed=0): self.seed = seed

    def fit(self, X, A):
        self.m = _clf(self.seed).fit(X, A); return self

    def predict(self, X):
        return np.clip(self.m.predict_proba(X), 1e-3, 1)


class DRLearner:
    """DR-learner: cross-fitted nuisances, pseudo-outcomes regressed on X for each a != none."""
    name = "DR-learner"

    def __init__(self, seed=0, folds=2, clip=0.02, it=300):
        self.seed, self.folds, self.clip, self.it = seed, folds, clip, it

    def fit(self, X, A, Y):
        n = len(Y); mu = np.zeros((n, K)); pi = np.zeros((n, K))
        for tr, te in KFold(self.folds, shuffle=True, random_state=self.seed).split(X):
            mu[te] = TLearner(self.seed, self.it).fit(X[tr], A[tr], Y[tr]).mu(X[te])
            pi[te] = Propensity(self.seed).fit(X[tr], A[tr]).predict(X[te])
        pi = np.clip(pi, self.clip, 1)
        onehot = np.eye(K)[A]
        phi = mu + onehot * (Y[:, None] - mu) / pi            # AIPW pseudo-outcomes per arm
        self.tau = [None] + [_reg(self.seed + 10 + a, self.it).fit(X, phi[:, a] - phi[:, 0]) for a in range(1, K)]
        self.base = TLearner(self.seed, self.it).fit(X, A, Y)  # level for Y(none)
        self.prop = Propensity(self.seed).fit(X, A)
        return self

    def mu(self, X):
        m0 = self.base.mu(X)[:, 0]
        return np.column_stack([m0] + [m0 + self.tau[a].predict(X) for a in range(1, K)])


class Ensemble:
    """Bootstrap ensemble of a learner; gives argmin agreement."""

    def __init__(self, maker, B=8, seed=0):
        self.maker, self.B, self.seed = maker, B, seed

    def fit(self, X, A, Y):
        rng = np.random.default_rng(self.seed); self.models = []
        for b in range(self.B):
            idx = rng.integers(0, len(Y), len(Y))
            self.models.append(self.maker(b).fit(X[idx], A[idx], Y[idx]))
        return self

    def agreement(self, X, choice):
        votes = np.stack([m.mu(X).argmin(1) for m in self.models])
        return (votes == choice[None, :]).mean(0)


class CounterfactualConformal:
    """Weighted split-conformal intervals for Y(a) (Lei & Candes 2021).

    Calibration points with A=a have score |Y - mu_a(X)|; weight w(x) = 1/pi_a(x)
    corrects the covariate shift between the treated-with-a subpopulation and the
    decision population. Returns per-arm lower/upper bounds."""

    def __init__(self, base, prop, alpha=0.2):
        self.base, self.prop, self.alpha = base, prop, alpha

    def fit(self, X, A, Y):
        mu = self.base.mu(X); pi = self.prop.predict(X)
        self.s = [np.abs(Y[A == a] - mu[A == a, a]) for a in range(K)]
        self.w = [1 / np.clip(pi[A == a, a], 0.02, 1) for a in range(K)]
        return self

    def interval(self, X):
        mu = self.base.mu(X); pi = self.prop.predict(X)
        lo = np.zeros_like(mu); hi = np.zeros_like(mu)
        for a in range(K):
            o = np.argsort(self.s[a]); s = self.s[a][o]; w = self.w[a][o]
            wt = 1 / np.clip(pi[:, a], 0.02, 1)
            cw = np.cumsum(w)[None, :] / (w.sum() + wt[:, None])
            idx = (cw < 1 - self.alpha).sum(1)
            qa = np.where(idx >= len(s), np.inf, s[np.minimum(idx, len(s) - 1)])
            lo[:, a] = mu[:, a] - qa; hi[:, a] = mu[:, a] + qa
        return lo, hi


def ope_value(choice, X, A, Y, mu_model, prop):
    """Doubly-robust off-policy estimate (Dudik et al. 2011) of the mean normalised cost of a
    deterministic policy, using only logged (factual) outcomes."""
    n = len(A); idx = np.arange(n)
    mu = mu_model.mu(X)[idx, choice]
    p = np.clip(prop.predict(X)[idx, choice], 0.02, 1)
    return float(np.mean(mu + (A == choice) * (Y - mu) / p))
