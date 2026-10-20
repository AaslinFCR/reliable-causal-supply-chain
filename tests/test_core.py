"""Fast sanity tests: python -m pytest -q  (or python tests/test_core.py)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rcd import conformal as CF, causal as CA, simulator as SIM, config as C


def test_split_conformal_coverage_exchangeable():
    rng = np.random.default_rng(0)
    cal = np.abs(rng.normal(size=2000)); test = np.abs(rng.normal(size=20000))
    q = CF._q(cal, 0.9)
    assert 0.88 < (test <= q).mean() < 0.92


def test_aci_recovers_coverage_after_shift():
    rng = np.random.default_rng(1)
    n = 3000; yhat = np.full(n, 100.0)
    y = yhat + rng.normal(0, 10, n); y[1000:] = yhat[1000:] + rng.normal(0, 25, 2000)   # variance shift
    cal = np.abs(rng.normal(0, 10, 500)) / 100
    q = CF.intervals("aci", yhat, y, cal, alpha=0.1)
    cov = (np.abs(y - yhat) / 100 <= q)[2000:].mean()
    assert cov > 0.85


def test_weighted_quantile_matches_unweighted():
    s = np.arange(1, 101, dtype=float)
    assert CF._wq(s, np.ones(100), 0.9) == CF._q(s, 0.9)


def test_simulator_shapes_and_truth():
    w = SIM.generate(7)
    S, T = w["demand"].shape
    assert S == len(C.REGIONS) * len(C.PRODUCTS) and T == len(C.dates())
    assert (w["demand"] >= 0).all() and SIM.shift_truth(w).any()


def test_tlearner_recovers_effect_sign():
    rng = np.random.default_rng(3)
    n = 4000; X = rng.normal(size=(n, len(CA.XCOLS))); A = rng.integers(0, 4, n)
    Y = 1 + X[:, 0] + np.where(A == 1, -0.5 * (X[:, 1] > 0), 0) + 0.1 * rng.normal(size=n)
    mu = CA.TLearner(0).fit(X, A, Y).mu(X)
    tau = mu[:, 1] - mu[:, 0]
    assert tau[X[:, 1] > 0.5].mean() < -0.3 and abs(tau[X[:, 1] < -0.5].mean()) < 0.15


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
