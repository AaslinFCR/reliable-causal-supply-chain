# Beginner staff-review guide - updated executed benchmark



## A short explanation

We predict tomorrow's wholesale market arrivals from historical arrivals, prices and calendar patterns. We then compare three ways of forming prediction intervals, which express uncertainty. Our research question is whether crop-specific adaptive calibration improves the balance between coverage and width. Warehouse crisis displays remain a synthetic supporting demonstration.



## Real and synthetic data

AGMARKNET is real historical government market data. Delhivery comes from a historical educational mirror and is analysed separately. Main/regional stock, demand, fuel, strikes and floods are synthetic. Arrivals are not customer demand.



## What each model does

Rolling mean averages recent arrivals. Weekly seasonal uses observations from the same weekday in earlier weeks. Random Forest averages many trees; the new implementation trains median imputation and categorical encoding only on training data. LightGBM builds trees sequentially to reduce prediction error. The routed bundle chooses a validated global LightGBM configuration for each crop; its eight candidates are configurations of one algorithm.



## Why LightGBM, and what the benchmark says

LightGBM was originally a practical choice for nonlinear tabular histories, missing values and categories. Random Forest was not previously implemented; it is now an executed benchmark, not an invented past result.

On the later April-June benchmark, Reselected LGBM bundle has the lowest WAPE among the recorded main and preserved-reference forecasts (19.80%). For identical log-LightGBM point predictions, commodity fixed coverage is 90.79% with mean width 194.135 tonnes; commodity adaptive coverage is 94.88% with width 344.446 tonnes. Their mean interval scores are 495.119 and 475.210, respectively. These are empirical tradeoffs, not a universal superiority or causal claim. The later-period interval-score difference has a descriptive paired interval that includes zero.



## Features

34 source features: seven arrival lags; rolling mean/std; historical price features; arrival mean/median/std windows; weekly seasonal average; trend/recent ratios; calendar/cyclical fields; state/market/crop codes. Fuel, strikes and inventory do not enter this forecaster. Exact names are in config.json and scrc/production/features.py.



## Intervals and calibration

A prediction is one expected number. An interval gives a lower and upper range. Coverage is the fraction of actual arrivals inside that range. Width shows how broad the range is. Pooled calibration shares errors across crops; commodity fixed uses separate crop errors; commodity adaptive updates crop error histories after each whole day. We aim for 95%, but empirical results may miss it. Narrower alone is not better.



## Actual tables

See RESULTS.md for all actual metrics, timing and ablations. Detailed market and monthly results are in the CSV files.



## Honest answers

**Is Random Forest used?** Yes, now in the separate new research benchmark; the production app remains unchanged LightGBM.

**What is the target?** Market arrivals in tonnes, not demand or stock-out probability.

**Is adaptive calibration new?** No; we reuse existing methods and evaluate them on our cohort. No new theorem is claimed.

**Was fixed commodity calibration already implemented?** Yes; the original state/crop grouping becomes crop-only in this one-state cohort.

**Is the test untouched?** January-March 2024 was already inspected. A later contiguous April--June 2024 cohort was newly acquired only after model and calibration settings were frozen.

**Can I call WAPE accuracy?** No. MAE, RMSE, WAPE and R2 are regression metrics.

**Does green guarantee a solution?** No; it passes synthetic scenario checks, not a measured real-world success probability.

**Does this prove causal savings?** No. Logistics is observational and warehouse activity is simulated.

**What did ablations prove?** They measure error changes after removing specified feature groups under the same data/settings; they do not establish causality.

**What is unfinished?** Broader external validation, real business feeds, measured solution outcomes, a deeper novelty review and publication polishing.



## Review opening

“I strengthened the project with an executed fair benchmark, including Random Forest, controlled feature ablations and three calibration methods. I use historical government market arrivals, preserve chronological timing, and report coverage with interval width and score. I do not claim guaranteed coverage, algorithmic novelty or real causal benefit. The paper is an honest rough draft.”



## Checks
89 tests passed; see TEST_RESULTS.json. Existing artifacts are hash-preserved.
Random Forest achieved the best later-period RMSE and R2; the reselected LightGBM bundle achieved the best WAPE and MAE. A winner depends on the evaluation metric.
