"""End-to-end experiment for one simulated world (one seed)."""
import time
import numpy as np
import pandas as pd
from . import config as C, simulator as SIM, prepare as PR, engine as EN, causal as CA, policies as PO

A = C.ACTIONS


def _stack(dec, net, world):
    rows = []
    for k in dec:
        F = k["F"]; n = len(F["cover"])
        X = CA.to_matrix(F)
        norm = np.exp(F["log_price"]) * np.expm1(F["log_fc"])
        rows.append(dict(d=np.full(n, k["d"]), s=np.arange(n), X=X, A=k["a"], Y=k["Y"], norm=norm,
                         F=F, info=k.get("info", {})))
    out = dict(d=np.concatenate([r["d"] for r in rows]), s=np.concatenate([r["s"] for r in rows]),
               X=np.vstack([r["X"] for r in rows]), A=np.concatenate([r["A"] for r in rows]),
               Y=np.vstack([r["Y"] for r in rows]), norm=np.concatenate([r["norm"] for r in rows]))
    out["Yn"] = out["Y"] / out["norm"][:, None]
    out["Yf"] = out["Yn"][np.arange(len(out["A"])), out["A"]]
    truth = SIM.shift_truth(world)
    out["shift_true"] = np.array([truth[s, d:d + 7].any() for d, s in zip(out["d"], out["s"])])
    out["dem_flag"] = np.concatenate([r["F"]["dem_flag"] for r in rows])
    out["q_rel"] = np.concatenate([r["F"]["q_rel"] for r in rows])
    out["Fall"] = {c: np.concatenate([r["F"][c] for r in rows]) for c in rows[0]["F"]}
    return out


def _sub(D, m):
    return {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(m) else
                ({c: x[m] for c, x in v.items()} if isinstance(v, dict) else v)) for k, v in D.items()}


def decision_quality(mu, D):
    """mu: predicted normalised outcomes (n x K). Truth in D['Yn'] / D['Y']."""
    choice = mu.argmin(1); n = len(choice); idx = np.arange(n)
    best = D["Y"].argmin(1)
    regret = D["Y"][idx, choice] - D["Y"][idx, best]
    tau_true = D["Yn"][:, 1:] - D["Yn"][:, [0]]; tau_hat = mu[:, 1:] - mu[:, [0]]
    return dict(pehe=float(np.sqrt(((tau_true - tau_hat) ** 2).mean())),
                decision_acc=100 * float((choice == best).mean()),
                regret=float(regret.mean()),
                harmful=100 * float((D["Y"][idx, choice] > D["Y"][:, 0] + 1e-6).mean()),
                value=float(D["Y"][idx, choice].mean()))


def run(seed=C.SEED, verbose=True, B=8):
    t0 = time.time()
    world = SIM.generate(seed)
    pre, rep, _ = PR.prepare(world, verbose=verbose)
    net = EN.Network(world, pre)
    dts = world["dates"]
    d_log0 = int(np.where(dts == pd.Timestamp("2023-01-02"))[0][0])
    d_test0 = int(np.where(dts == pd.Timestamp(f"{C.TEST_YEAR}-01-01"))[0][0])
    d_end = len(dts) - 7   # leave room for the last 7-day outcome window
    # ---------------- historical planner logs 2023-2024 and offline test logs 2025
    st = net.init_state(d_log0)
    logpol = PO.logging_policy(seed)
    _, dec_hist = net.run(st, d_log0, d_test0 - 1, logpol)
    snap = net.copy_state(st)
    st_off = net.copy_state(st)
    _, dec_off = net.run(st_off, d_test0, d_end, logpol)
    H = _stack(dec_hist, net, world); O = _stack(dec_off, net, world)
    cut = int(np.where(dts == pd.Timestamp(C.CAUSAL_TRAIN_END))[0][0])
    Tr = _sub(H, H["d"] <= cut); Va = _sub(H, H["d"] > cut)
    if verbose:
        print(f"[seed {seed}] logs: train {len(Tr['A'])}, val {len(Va['A'])}, offline-test {len(O['A'])};"
              f" action shares {np.bincount(Tr['A'], minlength=4) / len(Tr['A'])}  ({time.time()-t0:.0f}s)")
    # ---------------- effect learners
    learners = {"Naive": CA.Naive(), "S-learner": CA.SLearner(seed), "T-learner": CA.TLearner(seed),
                "DR-learner": CA.DRLearner(seed)}
    for m in learners.values():
        m.fit(Tr["X"], Tr["A"], Tr["Yf"])
    dq = {k: decision_quality(m.mu(O["X"]), O) for k, m in learners.items()}
    dq["Logged planner"] = dict(pehe=np.nan, decision_acc=100 * float((O["A"] == O["Y"].argmin(1)).mean()),
                                regret=float((O["Y"][np.arange(len(O["A"])), O["A"]] - O["Y"].min(1)).mean()),
                                harmful=100 * float((O["Y"][np.arange(len(O["A"])), O["A"]] > O["Y"][:, 0] + 1e-6).mean()),
                                value=float(O["Y"][np.arange(len(O["A"])), O["A"]].mean()))
    dq["Always none"] = dict(pehe=np.nan, decision_acc=100 * float((O["Y"].argmin(1) == 0).mean()),
                             regret=float((O["Y"][:, 0] - O["Y"].min(1)).mean()), harmful=0.0,
                             value=float(O["Y"][:, 0].mean()))
    dq["Oracle"] = dict(pehe=0.0, decision_acc=100.0, regret=0.0, harmful=0.0, value=float(O["Y"].min(1).mean()))
    # ---------------- learner selection on validation logs by doubly-robust off-policy value
    nuis_mu = CA.TLearner(seed + 7).fit(Tr["X"], Tr["A"], Tr["Yf"])
    nuis_pi = CA.Propensity(seed + 7).fit(Tr["X"], Tr["A"])
    ope = {k: CA.ope_value(m.mu(Va["X"]).argmin(1), Va["X"], Va["A"], Va["Yf"], nuis_mu, nuis_pi)
           for k, m in learners.items()}
    ope["Always none"] = CA.ope_value(np.zeros(len(Va["A"]), dtype=int), Va["X"], Va["A"], Va["Yf"], nuis_mu, nuis_pi)
    sel = min([k for k in learners if k != "Naive"], key=lambda k: ope[k])
    makers = {"S-learner": lambda b: CA.SLearner(seed * 100 + b), "T-learner": lambda b: CA.TLearner(seed * 100 + b),
              "DR-learner": lambda b: CA.DRLearner(seed * 100 + b, it=150)}
    best = learners[sel]
    prop = CA.Propensity(seed).fit(Tr["X"], Tr["A"])
    best.prop = prop
    dr = best
    ens = CA.Ensemble(makers[sel], B=B, seed=seed).fit(Tr["X"], Tr["A"], Tr["Yf"])
    ccp = CA.CounterfactualConformal(dr, prop, alpha=0.2).fit(Va["X"], Va["A"], Va["Yf"])
    lo, hi = ccp.interval(O["X"])
    cf_cov = {A[a]: 100 * float(((O["Yn"][:, a] >= lo[:, a]) & (O["Yn"][:, a] <= hi[:, a])).mean()) for a in range(4)}
    cf_cov_unweighted = None
    # ---------------- gate thresholds from validation logs (label free)
    q_hi = float(np.quantile(Va["q_rel"], 0.80)); q_max = float(np.quantile(Va["q_rel"], 0.98))
    gate = PO.Gate(dr, ens, ccp, q_hi=q_hi, q_max=q_max)
    # ---------------- offline gate ablation and risk-coverage on 2025 offline logs
    g = gate.assess(O["Fall"]); n = len(O["A"]); idx = np.arange(n)
    reg = O["Y"][idx, g["rec"]] - O["Y"].min(1)
    harm = O["Y"][idx, g["rec"]] > O["Y"][:, 0] + 1e-6
    c_ag = g["agree"] >= gate.agree; c_ov = g["pi"] >= gate.overlap
    comps = dict(agree=c_ag, overlap=c_ov, shift=~g["shift"], width=~g["wide"], cert=g["cert"])
    abl = {}
    for drop in [None] + list(comps):
        m = np.ones(n, dtype=bool)
        for k, v in comps.items():
            if k != drop:
                m &= v
        abl["Full gate" if drop is None else f"without {drop}"] = dict(
            green_share=100 * float(m.mean()), green_regret=float(reg[m].mean()) if m.any() else np.nan,
            green_harmful=100 * float(harm[m].mean()) if m.any() else np.nan,
            rest_regret=float(reg[~m].mean()) if (~m).any() else np.nan)
    order = np.lexsort((-g["pi"], -g["agree"] - 1.0 * (~g["shift"]) - 1.0 * g["cert"]))
    cov_grid = np.linspace(0.05, 1, 20)
    rc = [float(reg[order[:max(1, int(c * n))]].mean()) for c in cov_grid]
    rc_rand = float(reg.mean())
    if verbose:
        print(f"[seed {seed}] learners fitted ({time.time()-t0:.0f}s); selected {sel} by validation OPE "
              f"{ {k: round(v, 4) for k, v in ope.items()} }")
    # ---------------- closed-loop 2025
    pols = {
        "Static safety stock (MA)": (PO.none_policy, "ma"),
        "ML point forecast": (PO.none_policy, "gbdt"),
        "Rule-based planner": (PO.rule_planner, "gbdt"),
        "Conformal rule (no causal)": (PO.conformal_rule, "gbdt"),
        "Causal, no gate": (PO.causal_policy(dr), "gbdt"),
        "Proposed: gated (amber approved)": (PO.gated_policy(gate, "approve"), "gbdt"),
        "Proposed: gated (strict)": (PO.gated_policy(gate, "fallback"), "gbdt"),
        "Oracle (myopic)": (PO.oracle, "gbdt"),
    }
    closed = {}; zones = None
    for name, (pol, routine) in pols.items():
        st = net.copy_state(snap)
        daily, dec = net.run(st, d_test0, d_end, pol, routine=routine)
        D = _stack(dec, net, world)
        idx = np.arange(len(D["A"]))
        yc = D["Y"][idx, D["A"]]
        res = dict(total_cost=float(sum(daily[k].sum() for k in ("hold", "short", "act"))),
                   holding=float(daily["hold"].sum()), shortage=float(daily["short"].sum()),
                   action=float(daily["act"].sum()),
                   fill_rate=100 * float(daily["sales"].sum() / daily["dem"].sum()),
                   stockout_days=100 * float(daily["stockout"].mean()),
                   regret=float((yc - D["Y"].min(1)).mean()),
                   harmful=100 * float((yc > D["Y"][:, 0] + 1e-6).mean()),
                   act_share={A[a]: 100 * float((D["A"] == a).mean()) for a in range(4)},
                   regret_shift=float((yc - D["Y"].min(1))[D["shift_true"]].mean()),
                   regret_normal=float((yc - D["Y"].min(1))[~D["shift_true"]].mean()),
                   decision_acc=100 * float((D["A"] == D["Y"].argmin(1)).mean()))
        if name.startswith("Proposed"):
            z = np.concatenate([k["info"]["zone"] for k in dec])
            rec = np.concatenate([k["info"]["rec"] for k in dec])
            yr = D["Y"][idx, rec]; reg_rec = yr - D["Y"].min(1)
            fb = np.concatenate([PO.conformal_rule_actions(k["F"]) for k in dec])
            reg_fb = D["Y"][idx, fb] - D["Y"].min(1)
            zs = {}
            for zi, zn in enumerate(["green", "amber", "red"]):
                m = z == zi
                zs[zn] = dict(share=100 * float(m.mean()),
                              rec_regret=float(reg_rec[m].mean()) if m.any() else np.nan,
                              fb_regret=float(reg_fb[m].mean()) if m.any() else np.nan,
                              rec_harmful=100 * float((yr[m] > D["Y"][m, 0] + 1e-6).mean()) if m.any() else np.nan,
                              rec_acc=100 * float((rec[m] == D["Y"][m].argmin(1)).mean()) if m.any() else np.nan,
                              shift_true=100 * float(D["shift_true"][m].mean()) if m.any() else np.nan)
            ga = (z == 0) & (rec != 0)
            res["zones"] = zs
            res["auto_action_precision"] = 100 * float((yr[ga] <= D["Y"][ga, 0] + 1e-6).mean()) if ga.any() else np.nan
            res["auto_actions"] = int(ga.sum())
            res["escalated"] = 100 * float((z != 0).mean())
            if name.endswith("(amber approved)"):
                zones = dict(z=z, reg_rec=reg_rec, reg_fb=reg_fb, agree=np.concatenate([k["info"]["agree"] for k in dec]))
        closed[name] = res
        if verbose:
            print(f"  {name:36s} cost {res['total_cost']/1e5:9.2f} lakh  fill {res['fill_rate']:.2f}%  regret {res['regret']:.0f}")
    if verbose:
        print(f"[seed {seed}] done in {time.time()-t0:.0f}s")
    return dict(seed=seed, prepare=rep, selected=sel, ope=ope, gate_ablation=abl,
                risk_coverage=dict(coverage=cov_grid.tolist(), regret=rc, all_regret=rc_rand), decision_quality=dq, cf_coverage=cf_cov, closed=closed,
                gate_thresholds=dict(q_hi=q_hi, q_max=q_max), zones=zones,
                n_logs=dict(train=len(Tr["A"]), val=len(Va["A"]), test=len(O["A"])),
                logged_shares=(np.bincount(Tr["A"], minlength=4) / len(Tr["A"])).tolist())
