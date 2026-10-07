# Conformal Reliability-Gated Causal Intervention Selection for Supply Chain Resilience Under Distribution Shift

**Complete Final-Phase Project File: Idea, Real-Data Design, Method, Execution (Codex), Evaluation, Paper and Viva**
M.Tech Research (Data Science) | Updated: 07 Oct 2026

> **Data rule for this project:** only real Indian datasets (about 50,000 records). No synthetic data anywhere in the main results.

---

## 1. One-paragraph summary

Supply chains fail when demand or lead time suddenly changes (festivals, strikes, floods, policy shocks) because forecasting models trained on normal periods become confidently wrong. This project builds a decision system for a **main warehouse → regional warehouse** network that (1) forecasts with **conformal prediction intervals** that carry a coverage guarantee, (2) **detects distribution shift** in real time, (3) estimates the **causal effect of candidate interventions** and selects the one with the best *conservative* benefit, and (4) passes every decision through a **reliability gate** that auto-executes only when the evidence is trustworthy, otherwise asks a human or falls back to a safe rule. It is built and evaluated on **real Indian data** using a rolling real-time replay.

**Simplified one-line pitch:** *"Predict with honest uncertainty, notice when the world changes, choose the intervention that causally helps, and only act automatically when it is safe."*

---

## 2. Real-life scenario (main warehouse → regional warehouses)

An Indian FMCG/food distributor has one **main (central) warehouse** supplying four **regional warehouses** (North, South, East, West). Common products: rice, wheat, onion, potato (steady demand, visible stock-out risk).

```
Suppliers → Main Warehouse → Regional Warehouses → Retailers → Customers
```

**What goes wrong today**
- Festival demand, strikes, floods, fuel price jumps or export bans change the data pattern (distribution shift).
- Forecasts give one number with no trustworthy uncertainty.
- Planners choose fixes (expedite, change transport mode, raise safety stock) by experience, with no estimate of what truly works.
- No system says when its own recommendation should *not* be trusted.

**Worked example (Diwali week):** A regional warehouse expects 1,500–2,100 units of demand (conformal 90% interval) against 1,200 on hand. The shift detector flags a festival regime. The causal model compares: expedite (largest risk reduction), inter-warehouse transfer (moderate, risks the donor region), raise safety stock (small, high holding cost). The gate checks that the evidence for "expedite" is reliable and recommends it, with planner approval in the amber zone.

---

## 3. Problem statement

> In multi-echelon supply chains, disruptions change the data distribution, so forecasting and decision models fail without warning. Existing systems predict *what will happen* but do not reliably tell managers *which intervention will actually help*, and they cannot signal *when their own recommendation should not be trusted*. This causes stock-outs at regional warehouses, overstock at the main warehouse, and costly wrong interventions.

## 4. Objectives

1. Forecast demand/supply and transit delay at each region with **calibrated uncertainty** (conformal intervals with target coverage of 90%).
2. **Detect distribution shift** in real time and quantify detection delay and false alarms.
3. Estimate the **causal effect** of each *observable* intervention, with confidence intervals.
4. Select interventions **conservatively** (best lower bound of net benefit).
5. Add a **reliability gate** (green / amber / red) that controls automation.
6. Show on **real Indian data** that the full pipeline beats static safety stock, point forecasts, conformal-only, and causal-without-gate baselines on reliability and cost-aware metrics.

## 5. Contributions (for the paper)

- A unified decision loop combining conformal uncertainty, shift awareness, causal intervention selection and a reliability gate.
- A conservative selection rule (lower-bound benefit) tied to a gate with an explicit safety-vs-automation trade-off.
- A reproducible real-data evaluation on Indian wholesale-market and logistics data, with real shift events labelled from public information.
- Honest reporting of where the method does and does not win.

---

## 6. Real data design (about 50,000 records, all real)

### 6.1 Dataset stack

| Role | Dataset | What it provides |
|---|---|---|
| Demand/supply signal (primary) | **AGMARKNET** daily arrivals and prices (Ministry of Agriculture; available via data.gov.in, the India Data Portal, and the CEDA Ashoka processed version) | Daily price (min/max/modal) and arrivals for 300+ commodities from wholesale markets across India |
| Lead time and delay (secondary) | **Delhivery trip dataset** (public logistics dataset) | Source/destination centers, route type, `actual_time` vs `osrm_time`, distances; delay factor = actual_time / osrm_time |
| Shift labels | Real events with public sources (festival dates, COVID lockdown, flood periods, export bans, fuel price changes) | Labels real distribution shifts; nothing is injected |

> Verify before use: exact row counts, coverage gaps, file licences and citation requirements of each source (the Delhivery file's size and licence could not be confirmed while preparing this plan).

### 6.2 Reaching about 50,000 rows

Take a slice, not the whole dataset. Example:

**12 markets × 4 commodities × 3 years of daily data ≈ 52,000 rows.**

Pick the 12 markets across 4 regions (3 per region) so the regional structure is real. Confirm the final count after download; mandi data often has gaps.

### 6.3 Mapping real data to the model

| Model concept | Real-data equivalent |
|---|---|
| Regional demand/supply | Daily arrivals plus price pressure at a regional market |
| Main → region lead time | Delhivery hub-to-hub transit time and delay factor for the matching route/state |
| Stock-out event | **Proxy:** arrivals below k × rolling baseline AND modal price above a rolling quantile |
| Distribution shift | Labelled real events from `events.yaml` |
| Intervention | Only *observed* actions: e.g., transport/route type (FTL vs carting) and dispatch timing from Delhivery data |

### 6.4 Honest limitations (state these in the paper)

- Public data has no inventory levels, so the **stock-out is a proxy**. Define and freeze it before looking at test results.
- Real data never shows the counterfactual ("what if we had expedited?"). Causal claims are **estimated, not experimentally verified**.
- Only interventions observable in the data are evaluated. Transfers between warehouses and supplier switching are discussed as future work unless a dataset records them.
- Mandi markets stand in for regional distribution nodes; they are an analogy to warehouses, not warehouses.

---

## 7. Method (final, locked)

**Notation:** region r, product p, time t. ŷ = forecast, α = 0.1 (90% coverage).

**Step 1: Forecast with conformal interval**
- Base model: LightGBM (lags, rolling means, week-of-year, festival flag, delay features). Time-based train / calibration / test split only.
- Split conformal: residuals on the calibration set, quantile q at level ⌈(n+1)(1−α)⌉/n, interval [ŷ − q, ŷ + q].
- Shift robustness: **adaptive conformal inference** (online α update from recent miscoverage) and/or weighted conformal.

**Step 2: Shift score S_t ∈ [0,1]**
- Combine feature drift (PSI or KS vs training window) and rolling coverage drop. Flag when above a configured threshold.

**Step 3: Causal effect of each observable intervention**
- Candidate set A = {no action, observable interventions such as transport mode}.
- Estimate effect τ(a) on the outcome (delay factor / stock-out proxy risk) with bootstrap confidence intervals using EconML (T-learner or DR-learner) and DoWhy.
- Net benefit B(a) = benefit (cost avoided) − cost of action.

**Step 4: Conservative selection**
- a* = argmax over a of the **lower bound** of B(a).

**Step 5: Reliability gate**
```
width_ratio = interval_width / mean_demand
if width_ratio <= tau_width and S_t <= tau_shift and LB(B(a*)) > 0:
    GREEN  -> auto-execute a*
elif LB(B(a*)) > 0 or S_t <= tau_shift_high:
    AMBER  -> recommend, planner approves
else:
    RED    -> hold, fall back to static safety stock, alert human
```
Tune thresholds on a **validation period only**, never on test data.

**Step 6: Feedback**
- Log predicted vs. realised outcome, recalibrate, retrain on schedule or when drift is flagged.

---

## 8. Real-time deployment plan

| Layer | Function | Real-life equivalent |
|---|---|---|
| Data ingestion | Stream stock/orders/shipments/market arrivals/delays | WMS, ERP, TMS, market feeds |
| Forecast + conformal | Range forecasts with coverage | Planner sees a range, not one number |
| Shift detector | Flags regime change | Festival, strike, flood |
| Causal module | Effect of each action | "What happens if we do X?" |
| Reliability gate | Green / amber / red | Auto vs approval vs hold |
| Execution | Transfer/purchase orders to ERP/TMS | Truck dispatched, PO raised |
| Feedback | Compare predicted vs actual | Continuous recalibration |

**Rollout phases**
1. **Offline build:** train and calibrate on historical real data.
2. **Shadow mode:** run live beside the existing process; recommend only.
3. **Human-in-the-loop:** planners approve or reject; log overrides.
4. **Gated autonomy:** auto-execute green-zone actions on low-risk products, then expand.

**Suggested stack:** Kafka (streaming), PostgreSQL/TimescaleDB (storage), Python with LightGBM, MAPIE, EconML, DoWhy, FastAPI + Docker (serving), Grafana (monitoring of coverage, shift score, KPIs), REST integration with ERP/WMS/TMS.

**How this is evaluated without a live system:** a **rolling-origin replay** on real data. At each step the system only sees data up to time t, makes its decision, and is scored on what actually happened afterwards. This emulates real-time operation.

---

## 9. Experimental design

**Baselines**
- B1: static safety stock
- B2: ML point forecast, no uncertainty
- B3: forecast + conformal, no causal selection
- B4: causal selection without the gate
- **Proposed:** conformal + shift detection + causal selection + gate

**Evaluation windows:** the test period must contain real shift events (e.g., festival season, lockdown period, a flood or export-ban window). Report results **per event** as well as overall.

**Metrics**
- Conformal coverage (target 90%) and interval width
- Stock-out proxy rate / proxy fill rate
- Delay and transport-cost proxy
- Shift detection delay and false-alarm rate
- Auto-action precision (green zone) and % escalated to humans
- Policy value of the selected interventions, estimated with doubly-robust off-policy evaluation on logged data

**Causal validity checks (required):** placebo treatment test, random common cause, data-subset refutation, sensitivity to unmeasured confounding, temporal backtest.

**Ablations:** remove conformal; remove shift detector; remove gate; threshold sweep (safety vs automation curve).

**Reporting:** mean ± standard deviation over bootstrap or rolling windows; fixed seeds.

**Results table template**

| Method | Coverage | Proxy fill rate | Cost proxy | Policy value | Auto-precision |
|---|---|---|---|---|---|
| B1 … Proposed | – | – | – | – | – |

*(Fill only from actual code runs. Never write numbers by hand.)*

---

## 10. Codex execution plan (real-data version)

### Working rules
1. One phase per task; small scoped prompts.
2. Rules live in `AGENTS.md`; every prompt has acceptance criteria.
3. Review the diff, run `pytest -q` yourself, commit after each phase (`phase-N: name`).
4. Fix seeds and configs early.

### Repo structure
```
scrc/
  data/       loaders.py, clean.py, panel.py, build.py
  models/     forecaster.py, conformal.py, shift_detector.py
  causal/     effects.py, selector.py, refutation.py
  gate/       gate.py
  baselines/  static_ss.py, point_forecast.py, conformal_only.py, causal_nogate.py
  eval/       metrics.py, replay.py, ablations.py, plots.py
configs/      default.yaml, region_map.yaml, events.yaml
data/raw/     (never modified)   data/processed/
tests/  experiments/  README.md  AGENTS.md  pyproject.toml  Makefile
```

### Phase 0: AGENTS.md
```
# Project: Conformal Reliability-Gated Causal Intervention Selection for Supply Chain Resilience

## Goal
Research prototype on REAL Indian data (AGMARKNET arrivals/prices + Delhivery trip data).
Pipeline: forecast -> conformal interval -> shift detection -> causal effect of observable
interventions -> conservative selection -> reliability gate -> action.

## Rules
- NEVER generate synthetic data. If a dataset is missing or a column is absent, stop and ask.
- Python 3.11, type hints, docstrings, black + ruff.
- All randomness via the seed in configs/default.yaml. Results must be reproducible.
- No hard-coded paths or thresholds: use YAML configs.
- Time-based splits only. No random splits. No leakage from the future.
- Every module has pytest tests. Run `pytest -q` before finishing a task.
- Do not fabricate results. Numbers in outputs must come from code runs.
- After each task, summarize changes and how to verify.
```

### Phase 1: Scaffold
```
Task: Scaffold the repo with the structure above (scrc package with data, models, causal,
gate, baselines, eval), pyproject.toml (numpy, pandas, pyarrow, scikit-learn, lightgbm,
scipy, pyyaml, matplotlib, pytest, econml, dowhy, mapie, pandera), configs/default.yaml
(seed, alpha=0.1, split dates, tau_width, tau_shift, tau_shift_high, stockout_k,
stockout_price_quantile), empty region_map.yaml and events.yaml, and a smoke test importing
every subpackage.
Acceptance: `pip install -e .` and `pytest -q` pass.
```

### Phase 2: Real-data pipeline
```
Task: Build the real-data pipeline in scrc/data.
Inputs in data/raw/ (never modified): AGMARKNET daily arrivals/prices CSVs
(12 markets x 4 commodities x ~3 years) and the Delhivery trip dataset CSV.
1. Loaders with schema validation (dates, market, commodity, arrivals, min/max/modal price).
2. Cleaning: duplicates, unit normalization, impossible values; documented missing-value
   policy; log every imputation (no silent fills).
3. Build a panel indexed by (region, commodity, date).
4. From Delhivery: per-route transit time and delay factor = actual_time/osrm_time; cap
   outliers with a documented rule; join to regions via configs/region_map.yaml.
5. Stock-out PROXY from config: arrivals < k * rolling baseline AND modal price above a
   rolling quantile. Freeze the definition.
6. Label real shift events from configs/events.yaml (date ranges + source citations).
7. Time-based train / calibration / test splits; test must include at least one real event.
8. Save data/processed/panel.parquet and a data card (rows, date range, missingness).
Tests: no leakage across splits, no duplicate keys.
Acceptance: `python -m scrc.data.build` prints the final row count (target ~50,000) and a summary.
```

### Phase 3: Forecaster and conformal
```
Task: Implement models/forecaster.py and conformal.py.
- LightGBM with lag, rolling and calendar/festival features; time-based splits.
- Split conformal intervals; adaptive conformal inference (online alpha update);
  functions for coverage and mean interval width.
- Evaluate on normal periods and on each labelled real event window.
Tests: coverage near 1-alpha on stable periods; adaptive version recovers coverage after
a real shift better than split conformal.
Acceptance: table of coverage and width per event window.
```

### Phase 4: Shift detector
```
Task: Implement models/shift_detector.py.
Shift score S_t per region from (a) PSI/KS of recent features vs training window and
(b) rolling coverage drop. Normalized to [0,1] with a configurable threshold.
Plot S_t over time with real event dates marked.
Acceptance: detection delay and false-alarm rate reported per labelled event.
```

### Phase 5: Causal effects (observational)
```
Task: Implement causal/effects.py, selector.py, refutation.py.
- Estimate effect of each OBSERVABLE intervention (e.g., route type / dispatch timing from
  Delhivery data) on delay factor and stock-out-proxy risk with EconML (T-learner or
  DR-learner), bootstrap CIs, and covariate adjustment (distance, region, season, load).
- selector.py: a* = argmax of the LOWER bound of net benefit B(a); return all candidates.
- refutation.py: placebo treatment, random common cause, data-subset refutation, and a
  sensitivity analysis for unmeasured confounding.
- Doubly-robust off-policy evaluation of the selection policy on logged data.
Tests: selector never picks an action with a negative lower bound over "none";
refutations run without errors.
Acceptance: report effect estimates, CIs and refutation outcomes.
```

### Phase 6: Reliability gate
```
Task: Implement gate/gate.py with GREEN/AMBER/RED logic exactly as in Section 7, with
thresholds read from config. Add a threshold tuning function that uses the VALIDATION
period only (maximize proxy fill rate subject to auto-action precision >= target).
Tests: truth-table tests for all branches.
Acceptance: threshold sweep produces a safety-vs-automation trade-off table.
```

### Phase 7: Baselines and replay runner
```
Task: Implement baselines and eval/replay.py.
Baselines B1-B4 plus the proposed pipeline. replay.py performs a rolling-origin replay:
at each step use only data up to t, decide, then score against what actually happened.
Log coverage, width, proxy fill rate, cost proxy, detection delay, auto-precision,
% escalated, policy value. Save experiments/results.csv with config hash and git commit.
Acceptance: `python -m scrc.eval.replay --config configs/default.yaml` writes results.csv.
```

### Phase 8: Ablations, tables, figures
```
Task: Implement eval/ablations.py and plots.py.
Ablations: no conformal, no shift detector, no gate, threshold sweep.
Produce: main results table (mean +/- std, CSV and LaTeX), coverage-under-shift plot,
cost vs automation curve, gate decision distribution per event. 300 dpi, IEEE column widths.
Acceptance: all outputs regenerate from results.csv with one command.
```

### Phase 9: Quality and reproducibility
```
Task: Finalize the repo. Run ruff, black, pytest; write README (install, data download
instructions, how to reproduce each table/figure, config explanation, demo command);
Makefile targets: setup, data, test, experiments, figures, demo. Verify a clean-environment
run reproduces results.csv for the same seed.
```

### Phase 10: Paper support
```
Task: Using experiments/results.csv and generated figures, draft the Experimental Setup and
Results/Discussion sections in LaTeX (IEEEtran). Use ONLY numbers present in results.csv.
State where the proposed method does NOT win, and describe limitations honestly.
```

### Per-phase workflow
Start a fresh task with the phase prompt → read the diff → run `pytest -q` and the acceptance command → if it fails, give back the error output → commit.

---

## 11. Finishing plan (final phase)

### Deliverables
| # | Deliverable | Done when |
|---|---|---|
| 1 | Working prototype | Runs end to end on real data via replay |
| 2 | Experiments and results | All baselines, event windows and ablations complete |
| 3 | IEEE paper | Full draft, proofread, similarity-checked |
| 4 | Final report/thesis chapter | Problem → method → results → conclusion |
| 5 | Presentation and demo | Slides, 3–5 min demo, Q&A sheet |
| 6 | Code repository | README, requirements, seeds, data instructions |

### Timeline (about 4 weeks)
| Week | Focus |
|---|---|
| 1 | Data pipeline (Phase 2), freeze the stock-out proxy and event labels, first forecasts |
| 2 | Conformal, shift detector, causal estimates and refutations (Phases 3–5) |
| 3 | Gate, baselines, replay runs, ablations, figures (Phases 6–8) |
| 4 | Paper writing, repo cleanup, slides, demo rehearsal, Q&A drill (Phases 9–10) |

### Final checklist
- [ ] Raw data untouched; every dataset cited with source, URL, download date, licence
- [ ] Exact final row count reported (target ~50,000)
- [ ] Stock-out proxy and event labels frozen before test evaluation
- [ ] No synthetic data anywhere in main results
- [ ] Thresholds tuned on validation only
- [ ] Code runs from a clean setup with one command; seeds fixed
- [ ] Every claim in the paper backed by a table or figure
- [ ] Causal claims worded as *estimated*, with refutation results shown
- [ ] Limitations stated honestly (proxy outcome, observable interventions only)
- [ ] References checked in IEEE format; similarity check done
- [ ] Slides, demo and Q&A rehearsed twice

---

## 12. IEEE paper structure

1. **Abstract** (150–200 words): problem, method, key real-data result, contribution
2. **Introduction:** supply chain fragility in India, why shift breaks models, gap, 3–4 contributions
3. **Related work:** demand forecasting, conformal prediction (incl. under shift), causal inference in operations, supply chain resilience
4. **Problem formulation:** notation, network, observable interventions
5. **Proposed method:** Steps 1–6, system diagram, gate algorithm
6. **Data and experimental setup:** datasets, mapping, proxy definition, event labels, baselines, metrics
7. **Results and discussion:** main table, per-event plots, ablations, threshold trade-off, causal refutations
8. **Limitations and deployment:** proxy outcome, causal assumptions, shadow → human-in-the-loop → gated autonomy
9. **Conclusion and future work**
10. **References:** 25–35 solid ones

Figures: system architecture, gate flowchart, coverage under real shifts, cost vs automation curve, per-event gate decisions.

---

## 13. Review / viva preparation

**Demo flow (3–5 min):** normal week with automatic action → replay into a real festival/shock window → interval widens and shift alert fires → intervention comparison with causal estimates → gate decision and outcome → final results table.

**Likely questions and answers**
- *Why conformal instead of Bayesian intervals?* Distribution-free coverage guarantee, simple, works with any model.
- *Conformal assumes exchangeability; how do you handle shift?* Adaptive/weighted conformal, plus the shift detector and the gate.
- *How do you validate causal effects on real data?* Observational estimators with covariate adjustment, refutation tests, sensitivity analysis, temporal backtest, and conservative lower-bound selection; claims are worded as estimates.
- *Stock levels are not in public data; what is your outcome?* A frozen, config-defined stock-out proxy from arrivals and price pressure; stated as a limitation.
- *What is novel?* Combining calibrated uncertainty, shift awareness and causal action choice inside one reliability-gated loop, evaluated on real Indian data.
- *What if the gate is too strict?* The threshold sweep shows the safety-vs-automation trade-off explicitly.
- *How would this be deployed?* Shadow mode → human-in-the-loop → gated autonomy.

---

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Poor or gappy data | Validation checks, documented imputation, fall back to the safe rule |
| Proxy outcome criticised | Pre-register definition, sensitivity analysis over k and quantile |
| Causal assumptions fail | Refutation tests, sensitivity analysis, conservative gate |
| Planner distrust | Human approval and explanations in early phases |
| Few real shift events in the data | Use several event types and report per-event results; do not over-claim |

## 15. Limitations and future work

- Results rely on a proxy for stock-outs and on observable interventions only.
- Causal estimates rest on assumptions such as no unmeasured confounding.
- Future work: warehouse-level inventory data from a real partner, inter-warehouse transfer effects, multi-product interactions, reinforcement-learning-based interventions, and a live pilot with a real distributor.

---

## 16. Data and citation checklist

| Item | To record |
|---|---|
| AGMARKNET (data.gov.in / India Data Portal / CEDA Ashoka) | Exact source URL, download date, licence, markets and commodities chosen, row count |
| Delhivery trip dataset | Source page, licence, row count, columns used, outlier rule |
| Event labels | Public source for each event date range |

*Cite every dataset in the paper's data section and reference list.*
