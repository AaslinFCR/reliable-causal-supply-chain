# Implementation and evaluation completion

| Plan phase | Delivered evidence |
|---|---|
| Scaffold | Python package, YAML settings, isolated runtime |
| Real data | Government AGMARKNET monthly records, Delhivery CSV, checksums and data cards |
| Preparation | Units, chronological splits, missing dates and descriptive shortage proxy |
| Forecast and conformal | LightGBM, split and adaptive intervals, held-out results |
| Shift monitoring | Past-only scores, real event windows, alert/delay report |
| Observational causal estimates | T-learner, overlap, date bootstrap, placebo/subset/specification checks, DR scoring |
| Selection and gate | Conservative route policy; reject unsupported causal automation |
| Baselines and replay | Historical/rolling means, point forecasts, conformal and policy references |
| Ablations and figures | No-uncertainty/no-gate/no-shift/no-width comparisons; validation threshold sweep |
| Quality | Tests, lint, formatting, pipeline log, dependency lock and repeatability script |
| Paper support | Generated experimental setup/results fragment and measured tables |

Data-limited items: actual fill rates, operational stock-out reduction, inventory-policy costs, warehouse transfers, intervention costs, a joined market-shipment causal network, and live ERP execution cannot be evaluated from these public datasets. They are not represented as achieved results. Negative or inconclusive research findings are retained.
