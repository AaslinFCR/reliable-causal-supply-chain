# Research data dictionary

The two datasets support separate experiments. They have no verified shipment-level join.

## AGMARKNET observations

Local export: `data/processed/agmarknet_observations.csv`. One row is one real market, commodity and reporting date.

| Column | Meaning |
|---|---|
| state | Government state name |
| market | Government APMC market name; a market is not a warehouse |
| commodity | Rice, Wheat, Onion or Potato |
| date | Reporting calendar date |
| arrivals | Published daily total market inflow, metric tonnes, counted once across varieties |
| modal_price | Arrival-weighted modal price over valid variety rows, INR/quintal; median if all weights are zero |

Records require nonnegative reported arrivals and at least one positive reported modal price. Original monthly JSON responses retain all source fields for audit. The data card records published and usable market-day counts. Market selection uses only training-era reporting coverage.

The model panel in `data/processed/panel.parquet` expands each series to daily calendar dates. Missing arrivals and prices remain missing. `observed` identifies actual reporting dates. `lag_1`, `lag_7`, `lag_14`, `lag_28`, `rolling_mean`, `rolling_std` and `price_lag` use strictly previous dates. The rolling window is 28 calendar days with at least 14 observed values. `shortage_proxy` is arrivals below half the prior rolling mean and price above the prior rolling 80th percentile. It is descriptive pressure, not a measured stock-out.

## Delhivery trips

Local export: `data/processed/delhivery_trips.csv`. One row is one distinct trip UUID. Raw repeated scans are in `data/raw/delhivery.csv`.

| Column | Meaning |
|---|---|
| trip_uuid | Source trip identifier |
| actual_time | Sum of OD-leg maximum cumulative actual transit times, minutes |
| osrm_time | Sum of OD-leg maximum OSRM expected times, minutes |
| distance | Sum of OD-leg maximum OSRM distances, kilometres |
| legs | Number of distinct source/destination centre pairs |
| source_name, destination_name | First and last centres after chronological leg ordering |
| trip_creation_time, end_time | Source creation time and final observed leg completion time |
| date | Creation calendar date |
| source_state, destination_state | State text extracted from centre names; missing values are retained |
| route_type | Observed Carting or FTL route label |
| treatment | FTL=1; Carting=0 |
| delay_factor | actual_time / osrm_time; dimensionless |
| weekday, hour | Dispatch calendar fields |
| log_distance | log(1 + distance), a retrospective measurement |

Trips with nonpositive times/distances or nonfinite delay ratios are excluded. The outcome is delay factor, not a monetary cost. Shipment load, verified pre-dispatch route covariates, inventory and intervention prices are not observed. These omissions prevent verified causal automation.
