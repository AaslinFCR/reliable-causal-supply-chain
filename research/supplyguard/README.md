# SupplyGuard research benchmark

This directory contains separate research experiments. Existing production artifacts and the working application are preserved.

## Reproduce
From the project root, with the project's installed Python dependencies and existing processed datasets:

```powershell
.venv\Scripts\python.exe run.py scrc.research.benchmark
```

On another platform: `python run.py scrc.research.benchmark`.

The complete experiment configuration is `config.json`; dependency versions and seed are in `environment.json`. The input is `data/processed/deployment_observations.csv`. All models use the same eligible market-day rows, 34-feature information, chronological folds and final partitions. Training-only vocabulary and imputation are refitted within each fold. Random Forest one-hot encodes the three categories; LightGBM uses native category handling. Category representation differs, but source information is identical.

Each expanding validation quarter has a preceding 90-day calibration segment removed from that fold's proper training segment. The final models train through September 2023; October-December calibrates intervals. January-March 2024 is a previously inspected benchmark. April-June 2024 acquisition is attempted only after point-model and adaptive settings are frozen. A rate limit ends acquisition immediately. Incomplete acquisition produces no claimed later test. Acquisition has the existing downloader's finite retry bound for other request failures.

The task is rolling one-day-ahead prediction. Earlier actual arrivals and prices may be used for later target dates. This assumes prior-day reporting is available; source reporting latency is not verified. It is not a multi-day forecast made from one frozen origin.

## Calibration comparison
All methods use `abs(actual-prediction)/max(past_28_day_mean,1)`, the same point forecasts and 95% nominal coverage. Fixed pooled uses one residual quantile; fixed commodity uses one quantile per crop; rolling adaptive uses the most recent selected number of crop residuals and a clipped ACI-inspired level update. Whole date batches receive intervals before any outcome in that batch updates the state. At fewer than 100 group residuals, or an unsupported quantile rank, fallback is the original pooled 95% quantile. Adaptive histories for each evaluation period begin from the October-December calibration pool; January-March outcome residuals are not used to warm the later test. Prior outcomes still enter next-day forecast features.

The window is measured in individual residual observations, not days. Adaptive level is clipped to [0.005,0.2]. No original ACI guarantee is asserted for this batched/windowed/clipped implementation. Fixed state/commodity calibration already existed; with one state it equals commodity calibration and is reused transparently.

Interval score at miscoverage alpha is `(upper-lower) + (2/alpha)*(lower-y)*I(y<lower) + (2/alpha)*(y-upper)*I(y>upper)`. Lower is better. Coverage gap is empirical coverage minus 0.95; a negative gap is undercoverage. Coverage and width must be considered jointly.

## Ablations
A prespecified log-LightGBM configuration (15 leaves, 400 trees) is retrained with all 34 features, without price_lag/price_mean_7/price_mean_28, and without weekday/month/year_day/weekday_sin/weekday_cos/season_sin/season_cos. No other rows or hyperparameters change. Fixed commodity calibration is the primary ablation uncertainty comparison; adaptive settings are inherited, not retuned for each ablation.

## Outputs and boundaries
- `frozen_selection.json`: model and adaptive choices frozen before later acquisition.
- `validation_search.csv`: actual validation configurations, scores and timings.
- `point_metrics.csv`: point metrics overall and by crop, market and month.
- `interval_metrics.csv`: coverage, signed gap, width and interval score.
- `paired_differences.csv`: descriptive date-block bootstrap differences.
- `dataset_provenance.json`, `later_source_manifest.json`: real source information and missingness.
- `*_calibration_audit.json`: fallback and clipping counts.
- `plots/`: actual result figures.
- `models/`, `*_point.parquet`, `*_intervals.parquet`: local fitted research objects and row-level predictions; not committed publicly.
- `logs/run.log`, `failure.json` if present: executed progress and failures.
- `preservation_check.json`: hash comparison of existing production/research files.
- `RELATED_WORK.md`: verified primary-source comparison.

Timing is measured on this machine and is not a general speed guarantee. Final-fit and test-inference timing exclude common feature construction; Random Forest fit includes its train-only transformation. A bundle sums the compute times of the distinct required global models; training shared models is counted within each method, not across the whole project.

Whole-date resampling retains all market observations within each date but treats date blocks as independent. It does not preserve serial dependence across days; intervals for paired differences are descriptive, not rigorous dependent-time-series significance claims.

Synthetic warehouse scenarios and exploratory Delhivery effects remain supporting components. No forecast metric proves customer demand, stock-out prevention, monetary saving, disaster prediction or causal improvement. Existing gate ablations accepted every historical prediction and demonstrated no selective benefit. Software tests establish specified invariants, not model accuracy.

## Preserved reference configurations
The original 15-leaf log-LightGBM is also reported as `prespecified_log_full`, and the original production routing as `existing_lightgbm_bundle`. These existing choices were fixed before the new later period. They are refitted on the identical final training observations; original production files remain unchanged. The original routing and feature ablations inherit the validation-selected log-LightGBM adaptive settings rather than receiving separate calibration searches. `selected_lightgbm_bundle` denotes the new common-fold reselected routing. See reference_and_timing_audit.json for this distinction.
