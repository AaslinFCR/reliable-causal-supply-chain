"""Multi-echelon network simulator: 1 main warehouse -> 4 regional warehouses x 10 SKUs.

Generates (i) daily demand with festivals, promotions, trend, seasonality and
over-dispersed noise, (ii) a ground-truth schedule of disruption events that cause
distribution shift (demand surges, floods, transport strikes, supplier failures and
permanent regime changes), and (iii) realised lead times. Everything that does not
depend on the decision policy is pre-generated so that every policy and every
counterfactual action is evaluated on exactly the same random world
(common random numbers).
"""
import numpy as np
import pandas as pd
from . import config as C


def _festival_profile(dts, region, product):
    f = np.ones(len(dts))
    known = np.zeros(len(dts))          # calendar intensity known to the forecaster
    days_to = np.full(len(dts), 60.0)
    idx = pd.Series(np.arange(len(dts)), index=dts)
    for name, (fdates, regions, uplift, ramp) in C.FESTIVALS.items():
        if region not in regions:
            continue
        up = uplift.get(product, 1.05)
        for fd in pd.to_datetime(fdates):
            if fd not in idx.index:
                continue
            k = idx[fd]
            for j in range(-ramp, 2):
                t = k + j
                if 0 <= t < len(dts):
                    w = (ramp + j + 1) / (ramp + 1) if j < 0 else 1.0
                    f[t] = max(f[t], 1 + (up - 1) * w)
                    known[t] = max(known[t], w)
            for t in range(max(0, k - 60), k + 1):
                days_to[t] = min(days_to[t], k - t)
    return f, known, days_to


def make_events(rng, dts):
    """Ground-truth disruption schedule. Unknown to forecaster features."""
    ev = []
    years = sorted(set(dts.year))
    for y in years:
        test = (y == C.TEST_YEAR)
        n = dict(surge=3 if test else 2, flood=2 if test else 1, strike=2 if test else 1,
                 supplier=3 if test else 2)
        for kind, cnt in n.items():
            for _ in range(cnt):
                start = pd.Timestamp(f"{y}-01-20") + pd.Timedelta(days=int(rng.integers(0, 320)))
                if kind == "surge":
                    ev.append(dict(kind=kind, region=str(rng.choice(C.REGIONS)), product=None,
                                   start=start, days=int(rng.integers(8, 15)), mag=float(rng.uniform(1.45, 1.85))))
                elif kind == "flood":
                    ev.append(dict(kind=kind, region=str(rng.choice(C.REGIONS)), product=None,
                                   start=start, days=int(rng.integers(10, 15)), mag=5.0))
                elif kind == "strike":
                    ev.append(dict(kind=kind, region=None, product=None, start=start,
                                   days=int(rng.integers(6, 11)), mag=4.0))
                else:
                    ev.append(dict(kind=kind, region=None, product=str(rng.choice(C.PRODUCTS)),
                                   start=start, days=int(rng.integers(14, 22)), mag=0.35))
    # permanent regime changes in the test year (new quick-commerce channel / channel loss)
    ev.append(dict(kind="regime", region="Mumbai", product=("biscuits", "tea", "detergent", "soap"),
                   start=pd.Timestamp("2025-04-01"), days=9999, mag=1.30))
    ev.append(dict(kind="regime", region="Delhi", product=("rice", "dal", "salt"),
                   start=pd.Timestamp("2025-07-01"), days=9999, mag=0.80))
    return pd.DataFrame(ev)


def generate(seed=C.SEED):
    rng = np.random.default_rng(seed)
    dts = C.dates()
    T = len(dts)
    events = make_events(rng, dts)
    series = [(r, p) for r in C.REGIONS for p in C.PRODUCTS]
    S = len(series)
    lam = np.zeros((S, T)); demand = np.zeros((S, T), dtype=np.int64)
    promo = np.zeros((S, T)); fest_known = np.zeros((S, T)); fest_days = np.zeros((S, T))
    shift_dem = np.zeros((S, T), dtype=bool)
    t_years = (dts - dts[0]).days.values / 365.25
    doy = dts.dayofyear.values; dow = dts.dayofweek.values
    week_prof = np.array([0.95, 0.93, 0.96, 0.98, 1.04, 1.12, 1.02])
    for s, (r, p) in enumerate(series):
        base = C.BASE_UNITS[p] * C.REGION_SCALE[r] * C.TASTE.get((r, p), 1.0)
        g = rng.uniform(0.02, 0.08)
        amp = rng.uniform(0.04, 0.12); ph = rng.uniform(0, 365)
        seas = 1 + amp * np.sin(2 * np.pi * (doy - ph) / 365.25)
        if p == "tea":
            seas *= 1 + 0.12 * np.cos(2 * np.pi * (doy - 15) / 365.25)
        f, known, days_to = _festival_profile(dts, r, p)
        pr = np.ones(T); t = int(rng.integers(5, 40))
        while t < T:
            L = 4; up = rng.uniform(1.25, 1.5)
            pr[t:t + L] = up; promo[s, t:t + L] = 1
            t += int(rng.integers(30, 60))
        # latent AR(1) log-noise (unforecastable week-level variation)
        e = np.zeros(T); eps = rng.normal(0, 0.06, T)
        for k in range(1, T):
            e[k] = 0.75 * e[k - 1] + eps[k]
        lam[s] = base * (1 + g * t_years) * week_prof[dow] * seas * f * pr * np.exp(e)
        fest_known[s] = known; fest_days[s] = days_to
    # apply events to demand
    lt_extra = np.zeros((len(C.REGIONS), T)); exp_extra = np.zeros((len(C.REGIONS), T))
    supply_frac = np.ones((len(C.PRODUCTS), T))
    shift_lt = np.zeros((len(C.REGIONS), T), dtype=bool)
    shift_sup = np.zeros((len(C.PRODUCTS), T), dtype=bool)
    staples = {"rice", "atta", "dal", "salt", "sugar", "cooking_oil"}
    for _, e in events.iterrows():
        k0 = (e.start - dts[0]).days
        k1 = min(T, k0 + e.days)
        if k0 >= T:
            continue
        if e.kind == "surge":
            for s, (r, p) in enumerate(series):
                if r == e.region:
                    m = e.mag if p in staples else 1 + (e.mag - 1) * 0.5
                    lam[s, k0:k1] *= m; shift_dem[s, k0:k1] = True
        elif e.kind == "flood":
            ri = C.REGIONS.index(e.region)
            lt_extra[ri, k0:k1] += e.mag; exp_extra[ri, k0:k1] += 2; shift_lt[ri, k0:k1] = True
            for s, (r, p) in enumerate(series):
                if r == e.region:
                    prof = np.where(np.arange(k1 - k0) < 4, 1.5 if p in staples else 1.2, 0.75)
                    lam[s, k0:k1] *= prof; shift_dem[s, k0:k1] = True
        elif e.kind == "strike":
            lt_extra[:, k0:k1] += e.mag; exp_extra[:, k0:k1] += 2; shift_lt[:, k0:k1] = True
        elif e.kind == "supplier":
            pi = C.PRODUCTS.index(e["product"])
            supply_frac[pi, k0:k1] = e.mag; shift_sup[pi, k0:k1] = True
        elif e.kind == "regime":
            for s, (r, p) in enumerate(series):
                if r == e.region and p in e["product"]:
                    lam[s, k0:] *= e.mag; shift_dem[s, k0:] = True
    k = 25.0  # NB dispersion
    for s in range(S):
        pnb = k / (k + lam[s])
        demand[s] = rng.negative_binomial(k, pnb)
    base_lt = rng.choice([2, 3, 4], size=(len(C.REGIONS), T), p=[0.2, 0.6, 0.2])
    world = dict(dates=dts, series=series, demand=demand, lam=lam, promo=promo,
                 fest_known=fest_known, fest_days=fest_days, events=events,
                 lt_route=base_lt + lt_extra.astype(int), exp_extra=exp_extra.astype(int),
                 supply_frac=supply_frac, shift_dem=shift_dem, shift_lt=shift_lt, shift_sup=shift_sup)
    return world


def shift_truth(world):
    """Series x day boolean: any ground-truth disruption affecting that series."""
    S = len(world["series"])
    out = world["shift_dem"].copy()
    for s, (r, p) in enumerate(world["series"]):
        out[s] |= world["shift_lt"][C.REGIONS.index(r)]
        out[s] |= world["shift_sup"][C.PRODUCTS.index(p)]
    return out
