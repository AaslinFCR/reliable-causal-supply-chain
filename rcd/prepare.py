"""Stage 1: forecasts, conformal intervals and shift monitors for 2023-2025 (policy independent)."""
import numpy as np
import pandas as pd
from . import config as C, forecasting as F, conformal as CF, shift as SH, simulator as SIM


def _arr(df, col, S, T, mask=None):
    out = np.full((S, T), np.nan)
    sub = df if mask is None else df[mask]
    out[sub.s.values, sub.t.values] = sub[col].values
    return out


def prepare(world, years=(2023, 2024, 2025), eval_year=C.TEST_YEAR, extra_models=True, verbose=True):
    S, T = world["demand"].shape
    dts = world["dates"]; D = world["demand"].astype(float)
    df = F.build_frame(world)
    pre = {k: np.full((S, T), np.nan) for k in ["fc", "fc_ma", "ss", "ss_ma", "q", "cusum"]}
    pre["dem_flag"] = np.zeros((S, T), dtype=bool)
    pre["fest_next7"] = _arr(df, "fest_next7", S, T); pre["promo_next7"] = _arr(df, "promo_next7", S, T)
    report = {}
    truth = SIM.shift_truth(world)
    tr_flag = SH.transit_flag(world); sup_flag = SH.supply_flag(world)
    pre["tr_flag"], pre["sup_flag"] = tr_flag, sup_flag
    for Y in years:
        models = ("GBDT", "RF") if (Y == eval_year and extra_models) else ("GBDT",)
        out = F.fit_year(df, Y, models=models)
        tr, ca, te = out["masks"]
        gb = out["GBDT"]["model"]
        win = (df.date >= pd.Timestamp(f"{Y-1}-07-01")) & (df.date.dt.year <= Y) & df.ma28x7.notna()
        fcw = np.full((S, T), np.nan)
        fcw[df.s[win].values, df.t[win].values] = gb.predict(df[win])
        yfull = _arr(df, "y", S, T)
        tY = np.where(dts.year == Y)[0]; tC = np.where(ca.groupby(df.t).any().values)[0]
        # --- safety stocks from calibration residuals; z from the newsvendor critical fractile
        cr = C.COSTS.short_rate / (C.COSTS.short_rate + C.COSTS.hold_rate * C.REVIEW)
        from scipy.stats import norm as _n
        zcr = float(_n.ppf(cr))
        for name, fc_arr, key in [("gbdt", fcw, "ss"), ("ma", _arr(df, "ma28x7", S, T), "ss_ma")]:
            res = yfull[:, tC] - fc_arr[:, tC]
            sd = np.nanstd(res, axis=1)
            pre[key][:, tY] = (zcr * sd * np.sqrt((C.LEAD + C.REVIEW) / 7))[:, None]
        pre["fc"][:, tY] = fcw[:, tY]
        pre["fc_ma"][:, tY] = _arr(df, "ma28x7", S, T)[:, tY]
        # --- conformal (deployed: WACI) with delayed feedback
        cal_scores = np.abs(yfull[:, tC] - fcw[:, tC]) / np.maximum(fcw[:, tC], 1)
        pooled = cal_scores[~np.isnan(cal_scores)]
        # --- shift monitors
        trd = df[tr]
        dowfac = np.ones((S, 7))
        for s in range(S):
            g = trd[trd.s == s]
            m = pd.Series(D[s, g.t.values]).groupby(dts[g.t.values].dayofweek).mean()
            dowfac[s] = (m / m.mean()).reindex(range(7)).fillna(1).values
        rz = SH.daily_residuals(world, fcw, dowfac)
        mu = np.nanmean(rz[:, tC], axis=1, keepdims=True); sd = np.nanstd(rz[:, tC], axis=1, keepdims=True)
        z = (rz - mu) / sd
        t0 = tC[0]
        cs_stat = np.zeros((S, T)); cs_stat[:, t0:] = SH.cusum(z[:, t0:])
        h = np.nanpercentile(cs_stat[:, tC], 95)
        pre["cusum"][:, tY] = cs_stat[:, tY] / h
        pre["dem_flag"][:, tY] = cs_stat[:, tY] > h
        # --- conformal intervals (deployed: shift-conditional ACI) with 7-day delayed feedback
        cflag = cs_stat > h
        methods = ["split", "mondrian", "aci", "waci", "sacp"] if Y == eval_year else ["sacp"]
        qs = {m: np.zeros((S, len(tY))) for m in methods}
        fl_c = cflag[:, tC] & ~np.isnan(cal_scores)
        pool1 = cal_scores[fl_c]
        for s in range(S):
            okc = ~np.isnan(cal_scores[s]); cs = cal_scores[s][okc]
            for m in methods:
                if m == "sacp":
                    qs[m][s] = CF.shift_conditional(fcw[s, tY], yfull[s, tY], cs,
                                                    cflag[s, tC][okc].astype(int), cflag[s, tY].astype(int),
                                                    alpha=C.ALPHA, pool1=pool1)
                else:
                    qs[m][s] = CF.intervals(m, fcw[s, tY], yfull[s, tY], cs, alpha=C.ALPHA, pooled=pooled)
        pre["q"][:, tY] = qs["sacp"]
        if verbose:
            print(f"[prepare] year {Y}: fitted forecaster, h={h:.2f}")
        if Y == eval_year:
            rep = {}
            yte = yfull[:, tY]
            m_ok = ~np.isnan(yte)
            for name in ["naive", "ma28", "RF", "GBDT"]:
                if name not in out:
                    continue
                p = np.full((S, T), np.nan)
                p[df.s[te].values, df.t[te].values] = out[name]["test"]
                rep[name] = F.point_metrics(yte[m_ok], p[:, tY][m_ok])
            report["point"] = rep
            # conformal metrics, overall / under shift (any truth in target window) / normal
            tw = np.zeros((S, len(tY)), dtype=bool)
            for k in range(7):
                idx = np.clip(tY + k, 0, T - 1); tw |= truth[:, idx]
            cr = {}
            for m in methods:
                lo = np.maximum(0, fcw[:, tY] * (1 - qs[m])); hi = fcw[:, tY] * (1 + qs[m])
                cov = (yte >= lo) & (yte <= hi)
                isc = CF.interval_score(yte, lo, hi, C.ALPHA)
                wid = (hi - lo) / np.maximum(fcw[:, tY], 1)
                ent = {}
                for lab, msk in [("all", m_ok), ("shift", m_ok & tw), ("normal", m_ok & ~tw)]:
                    ent[lab] = dict(coverage=100 * cov[msk].mean(), rel_width=100 * np.median(wid[msk]),
                                    width=(hi - lo)[msk].mean(), int_score=isc[msk].mean(), n=int(msk.sum()))
                # worst series coverage
                ent["min_series_cov"] = 100 * np.nanmin([cov[s][m_ok[s]].mean() for s in range(S)])
                cr[m] = ent
            report["conformal"] = cr
            # rolling coverage series for plots
            report["cov_roll"] = {m: pd.DataFrame(
                ((yte >= np.maximum(0, fcw[:, tY] * (1 - qs[m]))) & (yte <= fcw[:, tY] * (1 + qs[m])) | ~m_ok
                 ).astype(float)).where(m_ok).mean(0).rolling(28, min_periods=7).mean().values for m in methods}
            report["dates_eval"] = dts[tY]
            # shift detection: CUSUM vs KS (KS threshold = 95th pct on calibration, label free)
            ks = np.zeros((S, T)); ks[:, t0:] = SH.ks_stat(z[:, t0:], z[:, tC][:, ~np.isnan(z[:, tC]).any(0)])
            hk = np.nanpercentile(ks[:, tC], 95)
            flags = {"CUSUM": cs_stat > h, "KS-14": ks > hk}
            dr = {}
            norm = ~truth
            for k2 in range(1, 8):   # exclude a week after any disruption from false-alarm base
                norm[:, k2:] &= ~truth[:, :-k2]
            for nm, fl in flags.items():
                ev = pd.DataFrame(SH.event_detection(fl, world["events"][world["events"].start.dt.year == Y], world))
                dr[nm] = dict(detect_rate=100 * ev.detected.mean(), median_delay=float(ev.delay.median()),
                              mean_delay=float(ev.delay.mean()),
                              false_alarm=100 * fl[:, tY][norm[:, tY]].mean(), n_events=len(ev),
                              by_kind=ev.groupby("kind").detected.mean().mul(100).round(1).to_dict())
            # transit monitor vs logistics truth
            tl = world["shift_lt"][:, tY]; trf = tr_flag[:, tY]
            dr["Transit-overdue"] = dict(recall_days=100 * trf[tl].mean(), false_alarm=100 * trf[~tl].mean())
            report["detection"] = dr
            report["cusum_example"] = (cs_stat / h, truth)
    return pre, report, df
