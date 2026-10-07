# Reliable Causal Decisions for Supply Chains Under Distribution Shift

Implementation and evaluation based on the supplied M.Tech plan. The download, preparation, forecasting, calibration, monitoring, conservative intervention estimation, gate, baselines, ablations, tables and figures are implemented. Operational stock-out reduction is not identifiable from these public datasets.

Measured result: split-conformal coverage 90.98%; adaptive coverage 89.60%, with 9.38% narrower mean intervals. Adaptive calibration fell below the nominal 90% target by 0.40 percentage points. The gate accepted every held-out prediction, so its ablations are identical here; this experiment shows no selective-prediction benefit from the gate. No causal automation benefit is established.

## Data actually used

**AGMARKNET:** 59,648 real market-day-commodity observations; 16 markets in Uttar Pradesh; rice, wheat, onion and potato; 2021-01-01–2023-12-31. Source: [Government AGMARKNET](https://agmarknet.gov.in/). All 144 monthly responses required by this cohort and their checksums are retained locally. Source columns define arrivals in metric tonnes and prices in INR/quintal. Published daily totals are counted once rather than repeated for every variety. Calendar gaps add 10,374 missing rows; these are never treated as observations or scored as zero.

**Delhivery:** 144,867 scan records aggregated to 14,817 distinct usable trips, September–October 2018. Source: [documented educational mirror](https://github.com/shekshavalipattan/Delhivery-Logistics-Data-Pipeline-and-Feature-Engineering). Cumulative scans are aggregated by OD-leg maxima, with legs ordered chronologically before trip-level endpoints are selected.

The original four-state rice/wheat archive experiment is preserved in `runs/archive_2014_2016/`; it is not mixed into this official-source experiment. The attempted four-state 2021–2023 download stopped at 184/576 monthly files because the government service continued returning HTTP 429 after extended cooldowns. The complete Uttar Pradesh cohort has 144/144 files. Incomplete Karnataka files and unavailable West Bengal/Maharashtra files are excluded. India Data Portal's public page was readable but its data service required a signed-in session. This is a documented regional limitation relative to the four-region plan, not a claim of four-region validation. Partial CEDA downloads are also excluded.

## Experimental protocol

Market selection uses January 2021–June 2022 completeness only. Training ends June 2022, calibration covers July–December 2022, validation covers January–June 2023, and the held-out test covers July–December 2023. Calendar lags and rolling means use strictly earlier dates. Test outcomes update online conformal calibration only after their predictions are logged; they never tune model or gate settings. Actual eligible counts appear in `experiments/market_splits.json`.

The forecaster is fixed LightGBM on log arrivals. Residuals are normalized by a prior rolling mean and calibrated by state/commodity. Adaptive calibration is a clipped, batched ACI-inspired implementation; original ACI theoretical guarantees do not automatically transfer. Drift combines feature KS effect sizes and past miscoverage. The predictive gate is selected using validation data only. Selective prediction acceptance is not successful-action precision.

Logistics evaluation has separate chronological partitions. Training trips unresolved at the next boundary are purged. T-learner outcome regressions and a propensity model support a conservative lower-bound route selector. Model uncertainty uses 100 training-date bootstrap refits; held-out policy loss intervals use 300 date-cluster resamples. Only 4 independent held-out dates are available. Carting is an observed reference route, not a verified no-action option.

## Overall measured forecast results

| method | n | mae | rmse | coverage | mean_width | acceptance |
|---|---|---|---|---|---|---|
| historical_mean | 9955 | 116.1459 | 319.0815 | — | — | 1.0000 |
| rolling_mean | 9955 | 38.9210 | 217.0976 | — | — | 1.0000 |
| point_forecast | 9955 | 30.8866 | 222.8334 | — | — | 1.0000 |
| split_conformal | 9955 | 30.8866 | 222.8334 | 0.9098 | 125.5868 | 1.0000 |
| adaptive_conformal | 9955 | 30.8866 | 222.8334 | 0.8960 | 113.8058 | 1.0000 |
| predictive_gate | 9955 | 30.8866 | 222.8334 | 0.8960 | 113.8058 | 1.0000 |
| gate_without_shift | 9955 | 30.8866 | 222.8334 | 0.8960 | 113.8058 | 1.0000 |
| gate_without_width | 9955 | 30.8866 | 222.8334 | 0.8960 | 113.8058 | 1.0000 |

Adaptive coverage is **89.60%** on 9,955 held-out observations. Its mean width is 113.8058 tonnes versus 125.5868 for split conformal. The rolling-mean baseline has MAE 38.9210 and RMSE 217.0976; LightGBM has MAE 30.8866 and RMSE 222.8334. Comparisons may favour different methods on different metrics; no universal superiority is asserted. State/commodity coverage ranges from 89.25% to 90.02%.

The validation-selected predictive gate accepts 100.00% of test predictions. Its no-shift and no-width ablations are included in the table; changes in accepted-subset metrics must be interpreted alongside acceptance rates.

## Real event windows

### rice_export_restriction_2023

Source: [rice_export_restriction_2023](https://www.pib.gov.in/PressReleasePage.aspx?PRID=1941139). Policy date verified; 30-day analysis window is a protocol choice, not a proven shift. Rice category may include unaffected varieties.

| method | n | mae | rmse | coverage | mean_width | acceptance |
|---|---|---|---|---|---|---|
| historical_mean | 400 | 93.5276 | 155.8670 | — | — | 1.0000 |
| rolling_mean | 400 | 20.7133 | 40.4777 | — | — | 1.0000 |
| point_forecast | 400 | 20.4531 | 40.2876 | — | — | 1.0000 |
| split_conformal | 400 | 20.4531 | 40.2876 | 0.9525 | 116.5815 | 1.0000 |
| adaptive_conformal | 400 | 20.4531 | 40.2876 | 0.8975 | 84.9149 | 1.0000 |
| predictive_gate | 400 | 20.4531 | 40.2876 | 0.8975 | 84.9149 | 1.0000 |
| gate_without_shift | 400 | 20.4531 | 40.2876 | 0.8975 | 84.9149 | 1.0000 |
| gate_without_width | 400 | 20.4531 | 40.2876 | 0.8975 | 84.9149 | 1.0000 |
### onion_minimum_export_price_2023

Source: [onion_minimum_export_price_2023](https://www.pib.gov.in/Pressreleaseshare.aspx?PRID=1972618). Policy date verified; 30-day analysis window fixed before results.

| method | n | mae | rmse | coverage | mean_width | acceptance |
|---|---|---|---|---|---|---|
| historical_mean | 413 | 22.1846 | 35.5930 | — | — | 1.0000 |
| rolling_mean | 413 | 16.7404 | 30.8418 | — | — | 1.0000 |
| point_forecast | 413 | 9.5711 | 16.9301 | — | — | 1.0000 |
| split_conformal | 413 | 9.5711 | 16.9301 | 0.8257 | 40.5527 | 1.0000 |
| adaptive_conformal | 413 | 9.5711 | 16.9301 | 0.9056 | 48.4475 | 1.0000 |
| predictive_gate | 413 | 9.5711 | 16.9301 | 0.9056 | 48.4475 | 1.0000 |
| gate_without_shift | 413 | 9.5711 | 16.9301 | 0.9056 | 48.4475 | 1.0000 |
| gate_without_width | 413 | 9.5711 | 16.9301 | 0.9056 | 48.4475 | 1.0000 |
### onion_export_prohibition_2023

Source: [onion_export_prohibition_2023](https://www.pib.gov.in/PressReleasePage.aspx?PRID=1985229). End truncated to data availability; event date does not prove distribution shift.

| method | n | mae | rmse | coverage | mean_width | acceptance |
|---|---|---|---|---|---|---|
| historical_mean | 326 | 26.3201 | 40.8478 | — | — | 1.0000 |
| rolling_mean | 326 | 12.1592 | 21.2770 | — | — | 1.0000 |
| point_forecast | 326 | 8.2266 | 14.7562 | — | — | 1.0000 |
| split_conformal | 326 | 8.2266 | 14.7562 | 0.8405 | 39.9941 | 1.0000 |
| adaptive_conformal | 326 | 8.2266 | 14.7562 | 0.8896 | 48.3759 | 1.0000 |
| predictive_gate | 326 | 8.2266 | 14.7562 | 0.8896 | 48.3759 | 1.0000 |
| gate_without_shift | 326 | 8.2266 | 14.7562 | 0.8896 | 48.3759 | 1.0000 |
| gate_without_width | 326 | 8.2266 | 14.7562 | 0.8896 | 48.3759 | 1.0000 |

| event | detected | delay_days | outside_window_alert_rate |
|---|---|---|---|
| rice_export_restriction_2023 | False | — | 0.0000 |
| onion_minimum_export_price_2023 | False | — | 0.0000 |
| onion_export_prohibition_2023 | False | — | 0.0000 |

The frozen validation-selected threshold produced alerts in 0/3 annotated event windows. Missing detection delays mean that no alert occurred in that window; they are not zero-day detections.

Policy dates annotate possible changes; they are not exhaustive drift ground truth or proof of a causal effect. `experiments/shift_detection.json` reports event alerts and delays. Outside-window alerts cannot automatically be labelled false alarms. Generic Rice includes varieties potentially unaffected by a specific export restriction.

## Observational logistics evaluation

| method | n | mean_dr_delay_loss | estimated_gain_vs_carting | gain_ci_low | gain_ci_high | ftl_recommendation_fraction | status |
|---|---|---|---|---|---|---|---|
| always_carting | 749 | 2.6873 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | exploratory; observational assumptions unverified |
| always_ftl | 749 | 2.8596 | -0.1723 | -0.6132 | 0.1711 | 1.0000 | exploratory; observational assumptions unverified |
| point_effect_policy | 749 | 2.3944 | 0.2929 | 0.0672 | 0.4837 | 0.5741 | exploratory; observational assumptions unverified |
| conservative_lower_bound_policy | 749 | 2.6158 | 0.0715 | -0.0511 | 0.1697 | 0.1509 | exploratory; observational assumptions unverified |
| reliability_gate | 2296 | — | — | — | — | — | All actions withheld: identification precondition not met. No counterfactual gate value claimed. |

Conservative policy estimated gain: 0.0715 delay-factor units; date-cluster interval [-0.0511, 0.1697]. The interval includes zero. Overlap includes 749/2,296 held-out trips (32.62%). These estimates condition on unverified observational assumptions. Aggregate OSRM distance may be treatment-dependent, and shipment load/fleet constraints are unobserved. Placebo, subset, omitted-distance and additive-bias diagnostics are in `experiments/causal_diagnostics.json`; they do not prove identification.

The causal gate withholds automatic actions because verified pre-dispatch confounders and causal identification are absent. The all-red result is a legitimate evidence rejection, not proof of improved supply-chain outcomes. Zero action cost is a declared delay-factor scenario; monetary savings are not inferred.

## Shortage-pressure proxy

| event | observed_rows | valid_proxy_rows | shortage_pressure_rate |
|---|---|---|---|
| onion_export_prohibition_2023 | 326 | 326 | 0.0000 |
| onion_minimum_export_price_2023 | 413 | 413 | 0.0993 |
| outside_labelled_window | 8816 | 8816 | 0.0200 |
| rice_export_restriction_2023 | 400 | 400 | 0.0025 |

This descriptive arrivals/price proxy is not a measured stock-out or fill rate. No counterfactual improvement in this proxy can be attributed to Delhivery route choices.

## Limits on the original plan

Public AGMARKNET markets and Delhivery shipments have no shared shipment identifiers and cover different years. They cannot be validly joined into one causal warehouse simulation. Static safety-stock costs, actual fill rates, transfers, supplier switching, inventory savings and green-action success are unobservable. They are marked unavailable rather than fabricated. There is no live ERP/WMS execution.

The implementation uses transparent scikit-learn/LightGBM estimators and explicit doubly robust scores rather than EconML/DoWhy. This software choice does not change the need for identification assumptions. The source supports Python 3.11+; the final isolated environment version is recorded in `experiments/run_manifest.json`.

## Reproduction

`python run.py scrc.pipeline` performs source acquisition (using immutable cache), preparation, evaluation, tests and delivery packaging. `run.ps1` launches it using the project environment. `scripts/verify_reproducibility.py` repeats the numerical evaluation; its actual result is in `reports/reproducibility.json`. Raw/processed inputs stay local; the code ZIP includes source, aggregate metrics and figures. See `README.md` for environment setup.

## Method references

[Gibbs and Candès, ACI, NeurIPS 2021](https://arxiv.org/abs/2106.00170); [Dudík, Langford and Li, doubly robust policy evaluation, ICML 2011](https://arxiv.org/abs/1103.4601); [Künzel et al., metalearners, PNAS 2019](https://doi.org/10.1073/pnas.1804597116).
