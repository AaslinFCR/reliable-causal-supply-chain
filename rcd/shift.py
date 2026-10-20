"""Online distribution-shift monitors (all use information available at the start of day t).

demand : two-sided CUSUM (Page 1954) on standardised daily forecast residuals
ks     : two-sample Kolmogorov-Smirnov statistic, last 14 residuals vs calibration residuals
transit: overdue-shipment telemetry per region (shipment dispatched >= 6 days ago, not arrived)
supply : main-warehouse fill rate for the product below 80% (observed on dispatch)
"""
import numpy as np
from scipy.stats import ks_2samp
from . import config as C


def daily_residuals(world, fc, dowfac):
    """z[s,t] uses forecast made at origin t (start of day t) and sales of day t (known end of t).
    It is therefore usable from day t+1 onward."""
    D = world["demand"].astype(float)
    dow = world["dates"].dayofweek.values
    yhat_d = fc / 7.0 * dowfac[:, dow]
    return (D - yhat_d) / np.maximum(yhat_d, 1.0)


def cusum(z, k=0.5, h=None):
    S, T = z.shape
    sp = np.zeros((S, T)); sm = np.zeros((S, T))
    for t in range(1, T):
        zt = np.nan_to_num(z[:, t - 1])  # residual of day t-1 is known at start of t
        sp[:, t] = np.maximum(0, sp[:, t - 1] + zt - k)
        sm[:, t] = np.maximum(0, sm[:, t - 1] - zt - k)
    return np.maximum(sp, sm)


def ks_stat(z, cal_z, win=14):
    S, T = z.shape
    out = np.zeros((S, T))
    for s in range(S):
        for t in range(win + 1, T):
            w = z[s, t - win - 1:t - 1]
            if np.isnan(w).any():
                continue
            out[s, t] = ks_2samp(w, cal_z[s]).statistic
    return out


def transit_flag(world):
    lt = world["lt_route"]; R, T = lt.shape
    flag = np.zeros((R, T), dtype=bool)
    for r in range(R):
        for t in range(T):
            for d in range(max(0, t - 12), t - 5):     # dispatched >= 6 days ago
                if d + lt[r, d] > t - 1:               # still not arrived
                    flag[r, t] = True; break
    return flag


def supply_flag(world):
    sf = world["supply_frac"]
    f = np.zeros_like(sf, dtype=bool)
    f[:, 1:] = sf[:, :-1] < 0.8   # yesterday's dispatch fill rate observed
    return f


def combine(world, dem_flag, tr_flag, sup_flag):
    S = len(world["series"])
    out = np.zeros_like(dem_flag)
    for s, (r, p) in enumerate(world["series"]):
        out[s] = dem_flag[s] | tr_flag[C.REGIONS.index(r)] | sup_flag[C.PRODUCTS.index(p)]
    return out


def event_detection(flag, truth_events, world, horizon_after=0):
    """Per ground-truth demand event (series-level): detected? delay in days."""
    res = []
    for _, e in truth_events.iterrows():
        if e.kind not in ("surge", "flood", "regime"):
            continue
        k0 = (e.start - world["dates"][0]).days
        k1 = min(flag.shape[1], k0 + min(e.days, 60))
        for s, (r, p) in enumerate(world["series"]):
            if r != e.region:
                continue
            if e.kind == "regime" and p not in e["product"]:
                continue
            hit = np.where(flag[s, k0:k1])[0]
            res.append(dict(kind=e.kind, s=s, start=e.start, detected=len(hit) > 0,
                            delay=hit[0] if len(hit) else np.nan))
    return res
