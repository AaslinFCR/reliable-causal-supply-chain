# Reliable Causal Decisions for Supply Chains Under Shift

Executable research code for a **conformal, shift-aware, reliability-gated causal decision
system** for a two-echelon supply network (one main warehouse supplying four regional
warehouses — Bengaluru, Chennai, Mumbai, Delhi — with ten FMCG SKUs).

The system answers three questions every week for every warehouse–SKU pair:

1. **What will demand be, and how sure are we?** – gradient-boosted forecaster + conformal
   interval (shift-conditional adaptive conformal inference, SC-ACI).
2. **Has the world changed?** – online CUSUM on forecast residuals, overdue-shipment telemetry,
   supplier fill-rate monitor.
3. **Which intervention actually helps, and can we trust that answer?** – meta-learners
   (S/T/DR) learned from confounded historical planner logs, selected by doubly-robust
   off-policy evaluation, with weighted conformal counterfactual intervals (Lei & Candès 2021)
   and a GREEN / AMBER / RED reliability gate.

Actions: `none`, `expedite` (main → regional, 1 day, premium freight), `transfer`
(lateral, from the regional warehouse with most surplus), `advance` (extra order at normal
lead time, i.e. temporarily raised safety stock).

## Run

```bash
pip install -r requirements.txt          # numpy, pandas, scipy, scikit-learn, matplotlib (lightgbm optional)
python tests/test_core.py                # ~20 s sanity tests
python run_all.py --seeds 5              # full study, ~3-4 min per seed on 2 CPU cores
```

Outputs: `results/summary.json`, `results/table_*.csv`, `figures/fig_*.pdf|png`.
Quick single-world run: `python run_all.py --seeds 1`.

If `lightgbm` is installed it is used as the forecaster; otherwise scikit-learn's
`HistGradientBoostingRegressor` (same histogram GBDT family) is used. The reported results
were produced with the scikit-learn model.

## Layout

| File | Role |
|---|---|
| `rcd/config.py` | network, SKUs, prices, costs, festival calendar, splits |
| `rcd/simulator.py` | demand generator + ground-truth disruption schedule (surges, floods, strikes, supplier failures, permanent regime changes) and lead times |
| `rcd/forecasting.py` | past-only features, ratio-target GBDT, Random Forest, naive and moving-average baselines |
| `rcd/conformal.py` | split, Mondrian, ACI, weighted ACI and proposed SC-ACI with 7-day delayed feedback |
| `rcd/shift.py` | CUSUM, KS window test, transit and supply monitors, event-level detection metrics |
| `rcd/prepare.py` | yearly refits (2023, 2024, 2025), intervals and monitors for every day |
| `rcd/engine.py` | inventory engine; exact counterfactual replay of every action (common random numbers) |
| `rcd/causal.py` | Naive, S-, T-, DR-learners, propensity model, bootstrap ensembles, weighted conformal counterfactual intervals, DR off-policy value |
| `rcd/policies.py` | historical planner (logging policy), baselines, reliability gate, gated policy, oracle |
| `rcd/experiment.py` | one complete world: logs → learners → gate → closed-loop 2025 evaluation |
| `rcd/report.py` | multi-seed tables and figures |

## Using your own data

Replace `simulator.generate()` with a loader that returns the same dictionary:
`dates`, `series` (region, sku), `demand` [series × day], `promo`, `fest_known`, `fest_days`,
`lt_route` [region × day], `exp_extra`, `supply_frac` [sku × day]. Ground-truth arrays
(`shift_dem`, `shift_lt`, `shift_sup`, `events`) are only used for evaluation. With real
data the counterfactual replay in `engine.py` is not available, so effect estimates must be
judged with the off-policy estimator in `causal.ope_value` and by prospective (shadow-mode)
trials.

## Honest scope

All data are simulated; costs are illustrative planning units, not company costs. Because
the simulator knows every counterfactual outcome, it can grade causal estimates exactly —
which real logs cannot — but the numbers are evidence about the method under the stated
simulation assumptions, not measured savings in a real company.

## Results web page

After a run, run_all.py writes report.html (built by rcd/webpage.py): a self-contained page with headline indicators, result tables and embedded figures, which opens in the browser automatically. Add --no-browser to skip opening it.

The figures/ folder holds the five-world figures (seeds 2026-2030) used in the paper and the Phase I report; run_all.py regenerates them.
