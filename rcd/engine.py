"""Inventory engine for the main -> regional network with weekly decisions.

Day order: (1) receive arrivals, (2) Monday: routine order-up-to replenishment from the
main warehouse, (3) Monday: intervention decision, (4) demand is served (lost sales).
Every Monday we also compute, by exact replay, the 7-day cost of every candidate action
for every series (the potential outcomes Y(a)). The 7-day window ends before the next
Monday, so no other decision can interfere: Y(a) is the true counterfactual outcome
under common random numbers. Replays are used only for evaluation and to produce the
factual outcome of the action actually taken; learners only ever see the factual one.
"""
import numpy as np
from . import config as C

A = C.ACTIONS


class Network:
    def __init__(self, world, pre):
        self.w, self.pre = world, pre
        self.S = len(world["series"])
        self.reg = np.array([C.REGIONS.index(r) for r, _ in world["series"]])
        self.prod = np.array([C.PRODUCTS.index(p) for _, p in world["series"]])
        self.price = np.array([C.PRICE[p] for _, p in world["series"]], dtype=float)
        # same-product peers for transfers
        self.peers = {s: [j for j in range(self.S) if self.prod[j] == self.prod[s] and j != s]
                      for s in range(self.S)}

    # ------------------------------------------------------------------ state
    def init_state(self, d, routine="gbdt"):
        fc = self.pre["fc" if routine == "gbdt" else "fc_ma"][:, d]
        oh = fc * (C.LEAD + C.REVIEW) / 7 * 0.7 + self.pre["ss"][:, d]
        return dict(oh=oh.astype(float), pipe=[[] for _ in range(self.S)])

    @staticmethod
    def copy_state(st):
        return dict(oh=st["oh"].copy(), pipe=[list(p) for p in st["pipe"]])

    # ------------------------------------------------------------------ helpers
    def _pipe_sum(self, st, s, upto=None):
        return sum(q for a, q in st["pipe"][s] if upto is None or a <= upto)

    def _order_up_to(self, s, d, routine):
        if routine == "gbdt":
            return self.pre["fc"][s, d] * (C.LEAD + C.REVIEW) / 7 + self.pre["ss"][s, d]
        return self.pre["fc_ma"][s, d] * (C.LEAD + C.REVIEW) / 7 + self.pre["ss_ma"][s, d]

    def _donor(self, st, s, d):
        best, surplus = None, -np.inf
        for j in self.peers[s]:
            sj = st["oh"][j] - self.pre["fc"][j, d]          # keep one week of cover at donor
            if sj > surplus:
                best, surplus = j, sj
        return best, surplus

    def action_spec(self, st, s, d, a):
        """Returns list of (series, arrival_day, qty) arrivals, immediate removals and action cost."""
        fc = self.pre["fc"][s, d]; r = self.reg[s]; p = self.prod[s]
        base_q = C.ACTION_FRAC * fc
        fill = self.w["supply_frac"][p, d]
        price = self.price[s]; c = C.COSTS
        if a == "none":
            return [], [], 0.0
        if a == "expedite":
            q = base_q * fill
            return [(s, d + 1 + self.w["exp_extra"][r, d], q)], [], c.exp_rate * price * q + c.exp_fixed
        if a == "advance":
            q = base_q * fill
            return [(s, d + self.w["lt_route"][r, d], q)], [], c.adv_fixed
        if a == "transfer":
            j, surplus = self._donor(st, s, d)
            q = float(np.clip(min(base_q, surplus), 0, None))
            if q <= 0:
                return [], [], 0.0
            lt = C.TRANSFER_LT[frozenset([C.REGIONS[r], C.REGIONS[self.reg[j]]])] + self.w["exp_extra"][r, d]
            return [(s, d + lt, q)], [(j, q)], c.tr_rate * price * q + c.tr_fixed
        raise ValueError(a)

    def _replay(self, oh0, pipe, s, d, extra=(), days=7):
        """Cost of series s over days d..d+days-1 with no further decisions."""
        D = self.w["demand"][s]; price = self.price[s]; c = C.COSTS
        oh = oh0; cost = 0.0
        arr = list(pipe) + list(extra)
        for k in range(d, min(d + days, D.shape[0])):
            if k > d:
                oh += sum(q for a, q in arr if a == k)
            sales = min(oh, D[k]); oh -= sales
            cost += c.short_rate * price * (D[k] - sales) + c.hold_rate * price * oh
        return cost

    def potential_outcomes(self, st, d):
        """Y[s, a] = 7-day cost of series s (+ donor's incremental cost for transfers)."""
        Y = np.zeros((self.S, len(A)))
        for s in range(self.S):
            base = self._replay(st["oh"][s], st["pipe"][s], s, d)
            for ai, a in enumerate(A):
                if a == "none":
                    Y[s, ai] = base; continue
                arrs, rem, cost = self.action_spec(st, s, d, a)
                y = self._replay(st["oh"][s], st["pipe"][s], s, d, [(t, q) for _, t, q in arrs]) + cost
                for j, q in rem:
                    y += (self._replay(st["oh"][j] - q, st["pipe"][j], j, d)
                          - self._replay(st["oh"][j], st["pipe"][j], j, d))
                Y[s, ai] = y
        return Y

    # ------------------------------------------------------------------ features
    def features(self, st, d):
        pre = self.pre; S = self.S
        fc = pre["fc"][:, d]; dpd = np.maximum(fc / 7, 1e-6); q = pre["q"][:, d]
        oh = st["oh"]
        arr3 = np.array([self._pipe_sum(st, s, d + 3) for s in range(S)])
        arr7 = np.array([self._pipe_sum(st, s, d + 6) for s in range(S)])
        allp = np.array([self._pipe_sum(st, s) for s in range(S)])
        donor = np.array([self._donor(st, s, d)[1] for s in range(S)]) / np.maximum(fc, 1)
        F = dict(
            log_fc=np.log1p(fc), cover=oh / dpd, cover3=(oh + arr3) / dpd, ip_cover=(oh + allp) / dpd,
            q_rel=q, gap_hi=(fc * (1 + q) - oh - arr7) / np.maximum(fc, 1),
            gap_lo=(fc * (1 - q) - oh - arr7) / np.maximum(fc, 1),
            dem_flag=pre["dem_flag"][:, d].astype(float), cusum=pre["cusum"][:, d],
            transit=pre["tr_flag"][self.reg, d].astype(float), supply=pre["sup_flag"][self.prod, d].astype(float),
            fill_obs=self.w["supply_frac"][self.prod, max(d - 1, 0)],
            fest=pre["fest_next7"][:, d], promo=pre["promo_next7"][:, d],
            donor=donor, exp_lt=1 + self.w["exp_extra"][self.reg, max(d - 1, 0)],
            log_price=np.log(self.price), region=self.reg.astype(float), product=self.prod.astype(float))
        return F

    # ------------------------------------------------------------------ main loop
    def run(self, st, d0, d1, policy, routine="gbdt"):
        """policy(d, F, st, Y) -> (list of action names, info); Y is given only to the oracle. Returns daily cost log and decision log."""
        S = self.S; D = self.w["demand"]; c = C.COSTS; dts = self.w["dates"]
        daily = dict(hold=np.zeros((S, d1 - d0 + 1)), short=np.zeros((S, d1 - d0 + 1)),
                     act=np.zeros((S, d1 - d0 + 1)), sales=np.zeros((S, d1 - d0 + 1)),
                     dem=np.zeros((S, d1 - d0 + 1)), stockout=np.zeros((S, d1 - d0 + 1)))
        dec = []
        for d in range(d0, d1 + 1):
            i = d - d0
            for s in range(S):  # arrivals
                if st["pipe"][s]:
                    st["oh"][s] += sum(q for a, q in st["pipe"][s] if a <= d)
                    st["pipe"][s] = [(a, q) for a, q in st["pipe"][s] if a > d]
            if dts[d].dayofweek == 0:
                for s in range(S):  # routine replenishment
                    need = self._order_up_to(s, d, routine) - st["oh"][s] - self._pipe_sum(st, s)
                    if need > 0:
                        q = need * self.w["supply_frac"][self.prod[s], d]
                        st["pipe"][s].append((d + self.w["lt_route"][self.reg[s], d], q))
                F = self.features(st, d)
                Y = self.potential_outcomes(st, d)
                acts, info = policy(d, F, st, Y)
                snap = self.copy_state(st)
                for s, a in enumerate(acts):
                    if a == "none":
                        continue
                    arrs, rem, cost = self.action_spec(snap if a != "transfer" else st, s, d, a)
                    for ss, t, q in arrs:
                        st["pipe"][ss].append((t, q))
                    for j, q in rem:
                        st["oh"][j] -= min(q, st["oh"][j])
                    daily["act"][s, i] += cost
                dec.append(dict(d=d, F=F, Y=Y, a=np.array([A.index(a) for a in acts]), info=info))
            dem = D[:, d]
            sales = np.minimum(st["oh"], dem)
            st["oh"] = st["oh"] - sales
            daily["short"][:, i] = c.short_rate * self.price * (dem - sales)
            daily["hold"][:, i] = c.hold_rate * self.price * st["oh"]
            daily["sales"][:, i] = sales; daily["dem"][:, i] = dem
            daily["stockout"][:, i] = (dem > sales + 1e-9)
        return daily, dec
