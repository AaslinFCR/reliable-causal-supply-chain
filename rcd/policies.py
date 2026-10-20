"""Decision policies: historical planner, baselines, the reliability-gated causal policy and oracle."""
import numpy as np
from . import config as C, causal as CA

A = C.ACTIONS
GREEN, AMBER, RED = 0, 1, 2


def _utilities(F):
    """Hand-crafted 'gut-feel' utilities used by historical planners (confounding mechanism)."""
    u = np.zeros((len(F["cover"]), 4))
    u[:, 0] = 1.2
    u[:, 1] = 2.2 * (F["cover3"] < 3) + 1.0 * F["transit"] + 0.4 * (F["fest"] > 0.3)
    u[:, 2] = 1.6 * (F["cover3"] < 4) * (F["donor"] > 0.3) + 0.6 * F["supply"]
    u[:, 3] = 1.2 * (F["ip_cover"] < 9) + 1.0 * (F["fest"] > 0.3) + 0.8 * F["supply"]
    return u


def logging_policy(seed, temp=0.7, eps=0.10):
    rng = np.random.default_rng(seed)

    def pol(d, F, st, Y):
        u = _utilities(F) / temp
        p = np.exp(u - u.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        p = (1 - eps) * p + eps / 4
        a = np.array([rng.choice(4, p=pi) for pi in p])
        return [A[i] for i in a], dict(p=p)
    return pol


def none_policy(d, F, st, Y):
    return ["none"] * len(F["cover"]), {}


def rule_planner(d, F, st, Y):
    return [A[i] for i in _utilities(F).argmax(1)], {}


def conformal_rule_actions(F):
    """Safe fallback: act on conformal bounds only, no causal model."""
    q = F["q_rel"]
    a = np.zeros(len(q), dtype=int)
    a[F["gap_hi"] > 0] = 3                         # upper-bound demand not covered -> advance order
    a[F["cover3"] < 3 * (1 - q)] = 1               # even lower-bound demand stocks out before routine arrival
    return a


def conformal_rule(d, F, st, Y):
    return [A[i] for i in conformal_rule_actions(F)], {}


def oracle(d, F, st, Y):
    return [A[i] for i in Y.argmin(1)], {}


def _norm(F):
    return np.exp(F["log_price"]) * np.expm1(F["log_fc"])


def causal_policy(model):
    def pol(d, F, st, Y):
        mu = model.mu(CA.to_matrix(F))
        return [A[i] for i in mu.argmin(1)], dict(rec=mu.argmin(1))
    return pol


class Gate:
    """Reliability gate. Thresholds are fixed from held-out planner logs (label free)."""

    def __init__(self, model, ens, ccp, q_hi, q_max, agree=0.75, overlap=0.05, red_overlap=0.02):
        self.model, self.ens, self.ccp = model, ens, ccp
        self.q_hi, self.q_max, self.agree, self.overlap, self.red_overlap = q_hi, q_max, agree, overlap, red_overlap

    def assess(self, F):
        X = CA.to_matrix(F)
        mu = self.model.mu(X); rec = mu.argmin(1)
        agree = self.ens.agreement(X, rec)
        pi = self.model.prop.predict(X)[np.arange(len(rec)), rec]
        lo, hi = self.ccp.interval(X)
        cert = hi[np.arange(len(rec)), rec] <= hi[:, 0] + 1e-12     # worst case of action <= worst case of none
        shift = (F["dem_flag"] > 0) | (F["transit"] > 0) | (F["supply"] > 0)
        wide = F["q_rel"] > self.q_hi
        green = (agree >= self.agree) & (pi >= self.overlap) & ~shift & ~wide & cert
        red = (pi < self.red_overlap) | (shift & (agree < 0.6)) | (F["q_rel"] > self.q_max) | ~cert
        zone = np.where(green, GREEN, np.where(red, RED, AMBER))
        return dict(rec=rec, zone=zone, agree=agree, pi=pi, cert=cert, shift=shift, wide=wide)


def gated_policy(gate, amber="approve"):
    """amber='approve': planner approves the recommendation (human-in-the-loop);
    amber='fallback': amber treated like red (strict mode)."""
    def pol(d, F, st, Y):
        g = gate.assess(F)
        fb = conformal_rule_actions(F)
        use_rec = (g["zone"] == GREEN) | ((g["zone"] == AMBER) & (amber == "approve"))
        a = np.where(use_rec, g["rec"], fb)
        return [A[i] for i in a], g
    return pol
