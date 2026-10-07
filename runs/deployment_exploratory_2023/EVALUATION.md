# Deployment forecasting evaluation

Selected model: **raw_l1_small**, weekly-seasonal blend weight 1.0.

Selection uses January–March 2023 only. April–June is a separate, untouched calibration set. Models train through December 2022. July–December test results were previously observed in the earlier research experiment, so this is a controlled regression benchmark, not a newly blinded holdout. No candidate is selected from test performance.

| Method | MAE (tonnes) | RMSE (tonnes) | WAPE | R² | Coverage |
|---|---|---|---|---|---|
| rolling_mean | 38.921 | 217.098 | 26.01% | 0.465 | — |
| weekly_seasonal | 44.027 | 237.399 | 29.42% | 0.361 | — |
| log_lightgbm_same_training | 28.015 | 216.868 | 18.72% | 0.466 | — |
| deployment_champion | 28.472 | 219.870 | 19.03% | 0.451 | 87.71% |

WAPE is absolute error divided by total observed arrivals; it is not classification accuracy. All predictions use earlier dates only. Calibration coverage is empirical and does not establish arbitrary-shift guarantees. The deployment experiment uses more recent training data and richer features than the original research experiment; the same-training log-LightGBM baseline is included for a fairer comparison.

The market data end in December 2023 and cover Uttar Pradesh. Without refreshed operational inputs, this application supports historical demonstrations and user-supplied histories. No validated causal automation, actual stock-out reduction, or monetary savings is established.