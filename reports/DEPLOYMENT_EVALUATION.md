# Deployment evaluation: fresh 2024 forward holdout

Per commodity: minimum mean rolling-fold WAPE + 0.25 * fold WAPE standard deviation; ties use mean RMSE. Three expanding validation quarters in 2023. No 2024 records used for fitting or selection.

Models and 95% split-conformal quantiles were frozen before fetching the 12 January–March 2024 AGMARKNET source files. Earlier 2023 exploratory results are preserved separately. The 95% nominal interval is a conservative planning choice; empirical coverage is reported without a shift guarantee.

| Method | MAE (tonnes) | RMSE (tonnes) | WAPE | R² | Coverage |
|---|---|---|---|---|---|
| rolling_mean | 37.960 | 150.312 | 27.71% | 0.644 | — |
| weekly_seasonal | 43.493 | 161.742 | 31.75% | 0.588 | — |
| log_lightgbm_same_training | 29.946 | 149.437 | 21.86% | 0.648 | — |
| deployment_champion | 31.242 | 155.374 | 22.81% | 0.620 | 91.94% |

WAPE measures absolute error relative to total observed arrivals. It is not classification accuracy. Model selection never uses the forward test metrics. The same-training log model provides a stronger baseline than comparing only with the older research model.

Scope: rice, wheat, onion and potato in 16 Uttar Pradesh markets. Model training ends September 2023; calibration ends December 2023. The new source observations extend the local history through March 2024, not to the present day. Live planning requires refreshed observations. The reliability gate never executes unverified causal interventions. No actual stock-out reductions or monetary savings are established.