"""Describe observed coverage and exclusions without filling source gaps."""

import pandas as pd

from scrc.common import ROOT, save_json


def main():
    observations = pd.read_parquet(ROOT / "data/processed/market_observations.parquet")
    panel = pd.read_parquet(ROOT / "data/processed/panel.parquet")
    keys = ["state", "market", "commodity"]
    counts = (
        observations.groupby(keys)
        .agg(
            observed_days=("date", "nunique"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            zero_arrivals=("arrivals", lambda x: int((x == 0).sum())),
            median_arrivals=("arrivals", "median"),
            median_price=("modal_price", "median"),
        )
        .reset_index()
    )
    counts["calendar_span_days"] = (counts.last_date - counts.first_date).dt.days + 1
    counts["reporting_fraction_within_span"] = (
        counts.observed_days / counts.calendar_span_days
    )
    counts.to_csv(ROOT / "reports/market_coverage_by_series.csv", index=False)
    eligible = panel.observed & panel.rolling_mean.notna() & (panel.rolling_mean > 0)
    save_json(
        ROOT / "reports/data_quality.json",
        {
            "observed_rows": len(observations),
            "model_eligible_rows": int(eligible.sum()),
            "observations_excluded_for_insufficient_past_history_or_zero_baseline": int(
                panel.observed.sum() - eligible.sum()
            ),
            "missing_calendar_rows": int((~panel.observed).sum()),
            "duplicate_market_date_keys": int(
                observations.duplicated([*keys, "date"]).sum()
            ),
            "negative_arrivals": int((observations.arrivals < 0).sum()),
            "nonpositive_prices": int((observations.modal_price <= 0).sum()),
            "interpretation": "Reporting completeness is descriptive; unavailable dates are never zero-filled or scored as observations.",
        },
    )


if __name__ == "__main__":
    main()
