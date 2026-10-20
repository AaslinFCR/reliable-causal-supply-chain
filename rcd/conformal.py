"""Conformal intervals for 7-day demand with delayed (7-day) label feedback.

Score: relative absolute residual r = |y - yhat| / max(yhat, 1).
Methods
  split    : one pooled split-conformal quantile (Vovk et al.; Lei et al. 2018)
  mondrian : per-series split-conformal quantile
  aci      : adaptive conformal inference (Gibbs & Candes 2021) per series
  waci     : ACI on a recency-weighted score distribution
             (non-exchangeable conformal weights, Barber et al. 2023)
  sacp     : proposed shift-conditional ACI - separate (Mondrian) score pools and ACI
             levels for origins flagged / not flagged by the online shift monitor.
             The flag uses only past data, so it is a valid Mondrian taxonomy.
At origin t only origins o <= t-7 have fully observed targets, so feedback is delayed.
"""
import numpy as np


def _q(scores, level):
    n = len(scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * level))
    if k > n:
        return np.inf
    return np.partition(scores, k - 1)[k - 1]


def _wq(scores, weights, level):
    """Weighted quantile with an extra test-point mass of weight 1 at +inf (Barber et al. 2023)."""
    o = np.argsort(scores); s = scores[o]; w = weights[o]
    tot = w.sum() + 1.0
    cw = np.cumsum(w) / tot
    i = np.searchsorted(cw, level)
    return np.inf if i >= len(s) else s[i]


def intervals(method, yhat, y, cal_scores, alpha=0.1, gamma=0.01, rho=0.99, delay=7, pooled=None):
    """yhat, y: 1-D arrays over consecutive origins of ONE series in the test period.
    cal_scores: that series' calibration scores. Returns q (relative half-width) per origin."""
    n = len(yhat)
    sc_test = np.abs(y - yhat) / np.maximum(yhat, 1.0)
    q = np.zeros(n)
    if method == "split":
        q[:] = _q(pooled, 1 - alpha); return q
    if method == "mondrian":
        q[:] = _q(cal_scores, 1 - alpha); return q
    a_t = alpha
    hist = list(cal_scores)
    ages_cal = np.arange(len(cal_scores))[::-1] + delay  # older calibration points
    for t in range(n):
        # reveal origin t-delay
        o = t - delay
        if o >= 0 and not np.isnan(sc_test[o]):
            err = float(sc_test[o] > q[o])
            a_t = float(np.clip(a_t + gamma * (alpha - err), 0.005, 0.5))
            hist.append(sc_test[o])
        level = 1 - a_t
        if method == "aci":
            q[t] = _q(np.asarray(hist[-500:]), level)
        elif method == "waci":
            h = np.asarray(hist)
            n_rev = len(hist) - len(cal_scores)
            ages = np.concatenate([ages_cal + t, np.arange(n_rev)[::-1] + delay])
            w = rho ** ages
            q[t] = _wq(h, w, level)
        if not np.isfinite(q[t]):
            q[t] = np.max(hist) if len(hist) else 1.0
    return q


def interval_score(y, lo, hi, alpha):
    return (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)


def shift_conditional(yhat, y, cal_scores, cal_flag, test_flag, alpha=0.1, gamma=0.01, delay=7,
                      window=500, pool1=None):
    """Shift-conditional ACI (SC-ACI): one ACI recursion per shift state g in {0,1}.
    pool1: flagged calibration scores pooled over all series, used when this series has
    fewer than 30 flagged calibration points."""
    n = len(yhat)
    sc = np.abs(y - yhat) / np.maximum(yhat, 1.0)
    hist = {g: list(cal_scores[cal_flag == g]) for g in (0, 1)}
    if len(hist[1]) < 30 and pool1 is not None:
        hist[1] = list(pool1)
    a = {0: alpha, 1: alpha}
    q = np.zeros(n)
    for t in range(n):
        o = t - delay
        if o >= 0 and not np.isnan(sc[o]):
            g = int(test_flag[o])
            err = float(sc[o] > q[o])
            a[g] = float(np.clip(a[g] + gamma * (alpha - err), 0.005, 0.5))
            hist[g].append(sc[o])
        g = int(test_flag[t])
        q[t] = _q(np.asarray(hist[g][-window:]), 1 - a[g])
        if not np.isfinite(q[t]):
            q[t] = max(hist[g])
    return q
