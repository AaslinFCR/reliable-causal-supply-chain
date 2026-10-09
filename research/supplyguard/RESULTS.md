# SupplyGuard - executed research benchmark

On the later April-June benchmark, Reselected LGBM bundle has the lowest WAPE among the recorded main and preserved-reference forecasts (19.80%). For identical log-LightGBM point predictions, commodity fixed coverage is 90.79% with mean width 194.135 tonnes; commodity adaptive coverage is 94.88% with width 344.446 tonnes. Their mean interval scores are 495.119 and 475.210, respectively. These are empirical tradeoffs, not a universal superiority or causal claim. The later-period interval-score difference has a descriptive paired interval that includes zero.

A later contiguous April--June 2024 cohort was newly acquired only after model and calibration settings were frozen.

## Protocol
Rolling one-day-ahead wholesale market-arrival forecasts. Identical eligible rows and source information across learned models. Training-only preprocessing. Three expanding validation folds, each with a preceding 90-day calibration segment held out of proper training. Final fitting through September 2023; October-December calibration. Main nominal coverage 95%.

## Actual results

### Previously inspected January-March 2024
| Model | MAE (t) | RMSE (t) | WAPE | R2 | Fit seconds | Inference seconds |
| --- | --- | --- | --- | --- | --- | --- |
| Rolling mean | 37.960 | 150.312 | 27.71% | 0.644 | 0.000 | 0.0003 |
| Weekly seasonal | 43.493 | 161.742 | 31.75% | 0.588 | 0.000 | 0.0007 |
| Random Forest | 31.970 | 149.687 | 23.34% | 0.647 | 102.821 | 0.0907 |
| Log-LightGBM | 29.810 | 149.830 | 21.76% | 0.646 | 4.307 | 0.5628 |
| Reselected LGBM bundle | 30.945 | 154.259 | 22.59% | 0.625 | 9.292 | 1.2314 |
| Original log-LGBM / full | 29.946 | 149.437 | 21.86% | 0.648 | 3.008 | 0.4302 |
| Existing LGBM bundle | 31.242 | 155.374 | 22.81% | 0.620 | 11.559 | 1.7675 |

#### Calibration around the same log-LightGBM predictions
| Calibration | Coverage | Gap (percentage points) | Width (t) | Interval score |
| --- | --- | --- | --- | --- |
| Commodity adaptive | 94.72% | -0.28 | 194.471 | 374.341 |
| Commodity fixed | 91.94% | -3.06 | 148.822 | 392.924 |
| Pooled fixed | 92.02% | -2.98 | 148.772 | 391.396 |

#### Commodity calibration breakdown
| Commodity | Calibration | n | Coverage | Width (t) | Interval score |
| --- | --- | --- | --- | --- | --- |
| Onion | Commodity adaptive | 1288 | 94.80% | 70.807 | 103.631 |
| Potato | Commodity adaptive | 1289 | 94.80% | 178.090 | 228.890 |
| Rice | Commodity adaptive | 1193 | 94.22% | 242.538 | 486.571 |
| Wheat | Commodity adaptive | 1195 | 95.06% | 297.443 | 710.968 |
| Onion | Commodity fixed | 1288 | 93.17% | 60.555 | 99.197 |
| Potato | Commodity fixed | 1289 | 93.72% | 160.762 | 228.072 |
| Rice | Commodity fixed | 1193 | 89.94% | 169.505 | 545.552 |
| Wheat | Commodity fixed | 1195 | 90.71% | 210.429 | 734.959 |
| Onion | Pooled fixed | 1288 | 92.78% | 58.200 | 98.354 |
| Potato | Pooled fixed | 1289 | 93.87% | 165.922 | 229.166 |
| Rice | Pooled fixed | 1193 | 91.03% | 179.894 | 531.668 |
| Wheat | Pooled fixed | 1195 | 90.21% | 196.822 | 742.200 |

#### Feature ablations: prespecified log-LightGBM; fixed commodity calibration
| Variant | MAE (t) | RMSE (t) | WAPE | Coverage | Width (t) |
| --- | --- | --- | --- | --- | --- |
| Without price | 30.031 | 149.127 | 21.93% | 92.06% | 150.912 |
| Without calendar | 29.747 | 148.927 | 21.72% | 92.23% | 151.583 |
| Original log-LGBM / full | 29.946 | 149.437 | 21.86% | 91.92% | 150.124 |

#### Paired descriptive differences
| Comparison | Metric | Adaptive minus reference | Date-block 95% interval |
| --- | --- | --- | --- |
| adaptive-minus-pooled_fixed | coverage | 0.0270 | [0.0213, 0.0331] |
| adaptive-minus-pooled_fixed | width | 45.6996 | [42.3307, 49.5597] |
| adaptive-minus-pooled_fixed | interval_score | -17.0553 | [-34.9816, 0.4542] |
| adaptive-minus-commodity_fixed | coverage | 0.0278 | [0.0226, 0.0336] |
| adaptive-minus-commodity_fixed | width | 45.6498 | [42.3252, 49.5678] |
| adaptive-minus-commodity_fixed | interval_score | -18.5835 | [-33.4814, -3.7106] |

Date blocks preserve markets within a date, but independent date resampling ignores serial dependence. These are descriptive uncertainty estimates, not rigorous dependent-time-series significance claims.

### Later April-June 2024
| Model | MAE (t) | RMSE (t) | WAPE | R2 | Fit seconds | Inference seconds |
| --- | --- | --- | --- | --- | --- | --- |
| Rolling mean | 63.317 | 179.693 | 33.51% | 0.728 | 0.000 | 0.0003 |
| Weekly seasonal | 72.675 | 203.804 | 38.46% | 0.650 | 0.000 | 0.0004 |
| Random Forest | 38.712 | 112.172 | 20.49% | 0.894 | 102.821 | 0.0889 |
| Log-LightGBM | 38.272 | 118.213 | 20.25% | 0.882 | 4.307 | 0.5303 |
| Reselected LGBM bundle | 37.424 | 116.869 | 19.80% | 0.885 | 9.292 | 1.1507 |
| Original log-LGBM / full | 38.538 | 120.909 | 20.39% | 0.877 | 3.008 | 0.4007 |
| Existing LGBM bundle | 37.977 | 117.973 | 20.10% | 0.883 | 11.559 | 1.6593 |

#### Calibration around the same log-LightGBM predictions
| Calibration | Coverage | Gap (percentage points) | Width (t) | Interval score |
| --- | --- | --- | --- | --- |
| Commodity adaptive | 94.88% | -0.12 | 344.446 | 475.210 |
| Commodity fixed | 90.79% | -4.21 | 194.135 | 495.119 |
| Pooled fixed | 90.65% | -4.35 | 188.484 | 500.412 |

#### Commodity calibration breakdown
| Commodity | Calibration | n | Coverage | Width (t) | Interval score |
| --- | --- | --- | --- | --- | --- |
| Onion | Commodity adaptive | 1253 | 95.37% | 75.904 | 86.836 |
| Potato | Commodity adaptive | 1254 | 95.30% | 154.403 | 231.841 |
| Rice | Commodity adaptive | 1147 | 93.98% | 144.153 | 293.309 |
| Wheat | Commodity adaptive | 1147 | 94.77% | 1045.870 | 1347.450 |
| Onion | Commodity fixed | 1253 | 91.94% | 55.954 | 73.745 |
| Potato | Commodity fixed | 1254 | 93.86% | 129.672 | 221.545 |
| Rice | Commodity fixed | 1147 | 91.19% | 109.278 | 305.240 |
| Wheat | Commodity fixed | 1147 | 85.79% | 500.418 | 1444.410 |
| Onion | Pooled fixed | 1253 | 91.22% | 53.782 | 72.853 |
| Potato | Pooled fixed | 1254 | 94.34% | 133.821 | 222.280 |
| Rice | Pooled fixed | 1147 | 92.07% | 115.908 | 300.017 |
| Wheat | Pooled fixed | 1147 | 84.57% | 467.971 | 1471.958 |

#### Feature ablations: prespecified log-LightGBM; fixed commodity calibration
| Variant | MAE (t) | RMSE (t) | WAPE | Coverage | Width (t) |
| --- | --- | --- | --- | --- | --- |
| Without price | 38.576 | 120.738 | 20.41% | 90.84% | 194.975 |
| Without calendar | 38.664 | 121.501 | 20.46% | 91.06% | 199.220 |
| Original log-LGBM / full | 38.538 | 120.909 | 20.39% | 90.92% | 194.067 |

#### Paired descriptive differences
| Comparison | Metric | Adaptive minus reference | Date-block 95% interval |
| --- | --- | --- | --- |
| adaptive-minus-pooled_fixed | coverage | 0.0423 | [0.0357, 0.0490] |
| adaptive-minus-pooled_fixed | width | 155.9624 | [133.0650, 183.4572] |
| adaptive-minus-pooled_fixed | interval_score | -25.2021 | [-87.9286, 34.5279] |
| adaptive-minus-commodity_fixed | coverage | 0.0408 | [0.0350, 0.0486] |
| adaptive-minus-commodity_fixed | width | 150.3115 | [127.4359, 173.7172] |
| adaptive-minus-commodity_fixed | interval_score | -19.9093 | [-76.6260, 36.1274] |

Date blocks preserve markets within a date, but independent date resampling ignores serial dependence. These are descriptive uncertainty estimates, not rigorous dependent-time-series significance claims.

## Calibration choice and audit
Adaptive choices are selected on validation interval scores subject to a 94% coverage screening rule; if none qualify, nearest nominal coverage then interval score. Final settings and full search appear in frozen_selection.json. Sparse groups fall back to initial pooled fixed calibration. No method sees its own target before interval formation.

## Feature removals
{
  "full": [],
  "without_price": [
    "price_lag",
    "price_mean_7",
    "price_mean_28"
  ],
  "without_seasonality": [
    "weekday",
    "month",
    "year_day",
    "weekday_sin",
    "weekday_cos",
    "season_sin",
    "season_cos"
  ]
}

## Limitations
One state and four crops; reporting latency is not verified; market arrivals are not demand; all operational crises are synthetic; logistics effects remain exploratory; no stock-out reduction, savings or causal improvement is inferred. Price and calendar removal do not remove arrival lags that indirectly retain seasonal information. The existing production routing is also independently refitted as a fixed reference. The reselected bundle reuses the existing selection approach/configurations but is refitted under the common new fold protocol, so it need not reproduce the old bundle routing. Existing 100%-acceptance gate ablations remain unchanged and do not prove gate effectiveness. No EnbPI comparison or new theoretical guarantee is implemented. Novelty remains uncertain.

## Execution evidence
89 tests passed. Existing production artifacts are hash-checked in preservation_check.json. Numerical model runs are completed; paper is a rough draft. Native compiler and rendering checks are reported separately.