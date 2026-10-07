# Reliable Causal Decisions for Supply Chains Under Distribution Shift

## Evaluation status

This is a completed real-data prototype evaluation with a narrower, explicitly documented scope than the supplied project plan. It is not evidence of operational stock-out reduction or deployable causal automation.

AGMARKNET: **51,655 observed market-day-commodity records**, 48 markets, rice and wheat, 2014-01-01–2016-12-31. Calendar padding adds 47,107 missing rows; these are not observations and are not scored. Raw target values were never generated or imputed.

Delhivery: **144,867 scan records**, aggregated to **14,817 usable trips**, 2018-09-12 00:00:00–2018-10-03 00:00:00. OD-leg cumulative maxima are summed before trip aggregation.

## Findings

Adaptive interval coverage is 90.09% on 5,641 held-out market observations, with 9.30% narrower intervals than split conformal. This is observed coverage, not a universal guarantee under shift. The forecasting model has MAE 13.3290 versus 13.4250 for the rolling mean, but RMSE 77.0483 versus 69.5910; it does not dominate that baseline.

The validation-selected predictive gate accepts 100.0% of test predictions. At this operating point, the gate and its ablations do not demonstrate an additional reliability benefit. The gate sweep is a prediction-acceptance study, not a cost-savings curve.

Coverage varies across state/commodity groups from 87.54% to 93.09%. Overall coverage therefore does not establish conditional reliability for every region or product. At the selected drift threshold, no alert is raised in the demonetisation window; this experiment does not demonstrate successful event detection.

The conservative policy's estimated gain over Carting is 0.0715 delay-factor units, with a date-cluster interval [-0.0511, 0.1697]. This interval includes zero, so the evaluation does not establish an intervention benefit.

## Forecast results on held-out real observations

| method | n | mae | rmse | coverage | mean_width | acceptance |
|---|---|---|---|---|---|---|
| historical_mean | 5641 | 31.3129 | 90.4373 | — | — | 1.0000 |
| rolling_mean | 5641 | 13.4250 | 69.5910 | — | — | 1.0000 |
| point_forecast | 5641 | 13.3290 | 77.0483 | — | — | 1.0000 |
| split_conformal | 5641 | 13.3290 | 77.0483 | 0.9239 | 56.4543 | 1.0000 |
| adaptive_conformal | 5641 | 13.3290 | 77.0483 | 0.9009 | 51.2031 | 1.0000 |
| predictive_gate | 5641 | 13.3290 | 77.0483 | 0.9009 | 51.2031 | 1.0000 |
| gate_without_shift | 5641 | 13.3290 | 77.0483 | 0.9009 | 51.2031 | 1.0000 |
| gate_without_width | 5641 | 13.3290 | 77.0483 | 0.9009 | 51.2031 | 1.0000 |

MAE and width use tonnes of arrivals. Forecasting arrivals is supply-flow forecasting, not measured customer demand. Historical mean and rolling mean are forecast references; they are not experimentally evaluated inventory policies. Predictive gate acceptance is selective interval reporting, not successful automatic intervention precision. Accepted subsets differ; compare acceptance and coverage together.

## Event-window results

| method | n | mae | rmse | coverage | mean_width | acceptance |
|---|---|---|---|---|---|---|
| historical_mean | 1053 | 28.3969 | 60.5702 | — | — | 1.0000 |
| rolling_mean | 1053 | 13.5904 | 48.4086 | — | — | 1.0000 |
| point_forecast | 1053 | 11.9825 | 44.9625 | — | — | 1.0000 |
| split_conformal | 1053 | 11.9825 | 44.9625 | 0.9136 | 52.7472 | 1.0000 |
| adaptive_conformal | 1053 | 11.9825 | 44.9625 | 0.9098 | 50.1443 | 1.0000 |
| predictive_gate | 1053 | 11.9825 | 44.9625 | 0.9098 | 50.1443 | 1.0000 |
| gate_without_shift | 1053 | 11.9825 | 44.9625 | 0.9098 | 50.1443 | 1.0000 |
| gate_without_width | 1053 | 11.9825 | 44.9625 | 0.9098 | 50.1443 | 1.0000 |

The legal tender withdrawal took effect on 9 November 2016. The fixed 30-day event window is an evaluation annotation; timing alone does not establish a causal market effect. Outside-window alerts are reported as alerts, not verified false positives. See `experiments/shift_detection.json`.

## Observational logistics policy estimates

| method | n | mean_dr_delay_loss | loss_ci_low | loss_ci_high | estimated_gain_vs_carting | gain_ci_low | gain_ci_high | ftl_recommendation_fraction | status |
|---|---|---|---|---|---|---|---|---|---|
| always_carting | 749 | 2.6873 | 2.5520 | 2.7727 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | exploratory; observational assumptions unverified |
| always_ftl | 749 | 2.8596 | 2.5977 | 3.1652 | -0.1723 | -0.6132 | 0.1711 | 1.0000 | exploratory; observational assumptions unverified |
| point_effect_policy | 749 | 2.3944 | 2.2978 | 2.4848 | 0.2929 | 0.0672 | 0.4837 | 0.5741 | exploratory; observational assumptions unverified |
| conservative_lower_bound_policy | 749 | 2.6158 | 2.5902 | 2.6423 | 0.0715 | -0.0511 | 0.1697 | 0.1509 | exploratory; observational assumptions unverified |
| reliability_gate | 2296 | — | — | — | — | — | — | — | All actions withheld: identification precondition not met. No counterfactual gate value claimed. |

Test overlap includes 749/2,296 trips (32.6%); empirical support check: True. Only 4 held-out calendar days underpin the cluster intervals. Lower estimated delay loss is better; intervals are exploratory and do not account for unmeasured confounding. They are computed for the overlap subset, not the full logistics population. Carting is a reference route type, not a documented no-action treatment. Action cost is set to zero in delay-factor units as a declared scenario; no monetary savings are inferred.

The causal gate withholds every intervention because verified pre-dispatch adjustment variables and identification are absent. Retrospectively aggregated OSRM distance may be affected by route choice; shipment load, available fleet and dispatch constraints are not observed. Narrow model intervals and placebo checks cannot fix these identification limits. No realised green-zone action precision or gate policy value can be measured from the logs.

## Method and leakage controls

- Whole dates are separated into training, calibration, validation and test. Features use strictly prior calendar observations. Categorical vocabularies are frozen on training data. Observations without sufficient past history are excluded from forecasting; see `experiments/market_splits.json` for exact eligible counts. Historical-mean forecasts for new market/commodity combinations fall back to a training-only state/commodity mean.
- One fixed LightGBM model predicts log arrivals. Absolute residuals are normalized by a prior 28-day mean, calibrated by state and commodity, then returned to original units.
- Split conformal uses the finite-sample corrected order statistic. Adaptive calibration updates after each same-day batch of outcomes; this clipped, batched ACI-inspired implementation is an empirical adaptation. Original ACI theoretical results do not automatically transfer to this implementation or establish conditional guarantees for dependent panels.
- Predictive thresholds maximize validation acceptance subject to empirical 90% accepted coverage and at least 30 accepted records. Test labels never select thresholds.
- Logistics models exclude training trips unresolved at the next split boundary. Potential outcome models and propensities fit on historical trips; date-cluster bootstrap refits estimate model uncertainty. Held-out doubly robust estimates use a frozen propensity overlap rule.
- No synthetic source records are used. Placebo analysis permutes existing treatment labels as a falsification check only.

## Deviations and unavailable claims

The CEDA public endpoint returned capped 1,000-row downloads and rate limiting; the government API returned HTTP 403. Complete rice/wheat data came from the documented Ian Covert AGMARKNET archive. Onion/potato and 2023 event experiments are not completed in this run because the archive does not contain them. The true observation count is reported rather than padded to 50,000.

AGMARKNET and Delhivery have no verified common shipment keys and cover different years. They are evaluated as separate evidence streams; no artificial geographic/time merge was made. Actual inventory, demand fulfilment, stock-outs, intervention costs, warehouse transfers and supplier switching are unavailable. The arrivals/price shortage proxy is descriptive, not a validated stock-out label. Static safety-stock costs, proxy fill-rate improvements under interventions, end-to-end causal policy superiority, and real-world automatic action precision are not estimable from these sources.

The prototype uses a transparent scikit-learn/LightGBM T-learner and manual doubly robust scoring rather than EconML/DoWhy. The available interpreter is Python 3.13; source supports Python 3.11+. No results are fabricated to satisfy planned performance tests.

## Reproduction and artifacts

See `README.md`. Results are in `experiments/results.csv`; per-date predictions in `experiments/market_replay.parquet`; policy diagnostics in `experiments/causal_diagnostics.json`; source checksums in `data/raw/archive_manifest.json`; configuration hashes in `experiments/run_manifest.json`. Figures and tables regenerate from these outputs.

## Sources

- AGMARKNET original government records and archive collection documentation: https://github.com/iancovert/Agmarknet
- Delhivery public educational dataset mirror: https://github.com/shekshavalipattan/Delhivery-Logistics-Data-Pipeline-and-Feature-Engineering
- RBI event notification: https://www.rbi.org.in/commonman/english/Scripts/Notification.aspx?Id=1938
- CEDA data portal: https://ceda.ashoka.edu.in/data-portal/

Method references: Gibbs and Candès, *Adaptive Conformal Inference Under Distribution Shift*, NeurIPS 2021 ([paper](https://arxiv.org/abs/2106.00170)); Dudík, Langford and Li, *Doubly Robust Policy Evaluation and Learning*, ICML 2011 ([paper](https://arxiv.org/abs/1103.4601)); Künzel et al., *Metalearners for estimating heterogeneous treatment effects using machine learning*, PNAS 2019 ([paper](https://doi.org/10.1073/pnas.1804597116)). These references motivate the methods; they do not validate the causal assumptions in these datasets.

Dataset redistribution licences were not verified for the archives. Raw data is kept locally and excluded from the code delivery ZIP.
