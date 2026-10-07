"""Prepare auditable real AGMARKNET market panels and Delhivery trips."""

import json

import pandas as pd

from scrc.common import ROOT, config, save_json
from scrc.data.panel import logistics_trips, market_features


def market_panel(cfg):
    """Aggregate price varieties once, and join quantities by genuine market keys."""
    prices = []
    quantities = []
    paths = sorted((ROOT / "data/raw").glob("ceda_*.json"))
    expected = len(cfg["states"]) * len(cfg["commodities"]) * 3 * 2
    if len(paths) != expected:
        raise FileNotFoundError(
            f"Need all {expected} original CEDA responses; found {len(paths)}. Run download first."
        )
    for path in paths:
        records = json.loads(path.read_text(encoding="utf-8"))["data"]
        if records:
            (prices if path.stem.endswith("prices") else quantities).append(
                pd.DataFrame(records)
            )
    p = pd.concat(prices, ignore_index=True).drop_duplicates()
    q = pd.concat(quantities, ignore_index=True).drop_duplicates()
    keys = ["t", "state_id", "market_id", "cmdty"]
    required_p = set(keys + ["p_min", "p_max", "p_modal", "market_name", "state_name"])
    required_q = set(keys + ["qty"])
    if required_p - set(p) or required_q - set(q):
        raise ValueError("CEDA response schema changed.")
    for col in ["p_min", "p_max", "p_modal"]:
        p[col] = pd.to_numeric(p[col], errors="coerce")
    q["qty"] = pd.to_numeric(q["qty"], errors="coerce")
    invalid_price = (
        (p.p_modal <= 0)
        | (p.p_min < 0)
        | (p.p_max < 0)
        | ((p.p_max > 0) & (p.p_modal > p.p_max))
        | ((p.p_min > 0) & (p.p_modal < p.p_min))
    )
    invalid_quantity = (q.qty < 0) | q.qty.isna()
    audit = {
        "invalid_price_rows_dropped": int(invalid_price.sum()),
        "invalid_quantity_rows_dropped": int(invalid_quantity.sum()),
        "price_rows_after_exact_dedup": len(p),
        "quantity_rows_after_exact_dedup": len(q),
        "imputation": "None. Missing dates retained as NaN; unobserved targets excluded from scoring.",
        "price_varieties": "Median modal/min/max across varieties within market-date-commodity; not a volume-weighted price.",
        "quantities": "Median duplicate market-date-commodity reports, never repeated/summed across price varieties.",
    }
    p = (
        p.loc[~invalid_price]
        .groupby(keys, as_index=False)
        .agg(
            modal_price=("p_modal", "median"),
            min_price=("p_min", "median"),
            max_price=("p_max", "median"),
            market=("market_name", "first"),
            state=("state_name", "first"),
        )
    )
    q = (
        q.loc[~invalid_quantity]
        .groupby(keys, as_index=False)
        .agg(arrivals=("qty", "median"))
    )
    joined = p.merge(q, on=keys, how="inner", validate="one_to_one").rename(
        columns={"t": "date", "cmdty": "commodity"}
    )
    joined["date"] = pd.to_datetime(joined.date)
    # Freeze market selection on training-era records only. No test performance or future completeness.
    selection = ROOT / "configs/selected_markets.json"
    if selection.exists():
        selected = json.loads(selection.read_text())
    else:
        early = joined.loc[joined.date < "2022-07-01"]
        counts = (
            early.groupby(["state_id", "market_id", "commodity"])
            .size()
            .unstack(fill_value=0)
        )
        counts = counts.reindex(
            columns=["Wheat", "Rice", "Onion", "Potato"], fill_value=0
        )
        counts["score"] = counts.min(axis=1) * 10 + counts.sum(axis=1)
        selected = []
        for state in cfg["states"]:
            ranked = counts.loc[state].sort_values(
                ["score"], ascending=False, kind="stable"
            )
            selected.extend(
                [
                    [int(state), int(m)]
                    for m in ranked.head(cfg["markets_per_state"]).index
                ]
            )
        save_json(selection, selected)
    selected = set(map(tuple, selected))
    keep = [
        (int(s), int(m)) in selected for s, m in zip(joined.state_id, joined.market_id)
    ]
    joined = (
        joined.loc[keep]
        .sort_values(["date", "state_id", "market_id", "commodity"])
        .reset_index(drop=True)
    )
    audit["joined_rows_before_market_selection"] = len(keep)
    audit["selected_markets"] = len(selected)
    audit["selected_rows"] = len(joined)
    audit["date_min"] = str(joined.date.min().date())
    audit["date_max"] = str(joined.date.max().date())
    audit["market_selection"] = (
        "Four markets per state ranked by training-era commodity completeness; selection frozen to JSON."
    )
    audit["units"] = (
        "Raw CEDA arrivals and prices; portal unit documentation must be verified before economic interpretation. No invented conversions."
    )
    audit["missingness"] = joined.isna().sum().to_dict()
    audit["by_state_commodity"] = (
        joined.groupby(["state", "commodity"]).size().to_dict()
    )
    audit["by_state_commodity"] = {
        str(k): int(v) for k, v in audit["by_state_commodity"].items()
    }
    save_json(ROOT / "reports/market_data_card.json", audit)
    joined.to_parquet(ROOT / "data/processed/market_observations.parquet", index=False)
    return market_features(joined, cfg)


def main():
    """Build both independent datasets, preserving every raw file."""
    cfg = config()
    (ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
    raw_path = ROOT / "data/raw/delhivery.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            "Real Delhivery CSV missing. No generated substitute is allowed."
        )
    raw = pd.read_csv(raw_path)
    trips = logistics_trips(raw)
    trips.to_parquet(ROOT / "data/processed/trips.parquet", index=False)
    trips.to_csv(ROOT / "data/processed/delhivery_trips.csv", index=False)
    save_json(
        ROOT / "reports/logistics_data_card.json",
        {
            "raw_rows": len(raw),
            "raw_trip_ids": int(raw.trip_uuid.nunique()),
            "usable_trips": len(trips),
            "start": str(trips.date.min()),
            "end": str(trips.date.max()),
            "route_types": trips.route_type.value_counts().to_dict(),
            "aggregation": "OD-leg maximum of cumulative times/distance, summed over distinct legs to one trip.",
            "missingness": trips.isna().sum().to_dict(),
            "limitations": [
                "22 calendar days only; cannot evaluate festivals or policy shocks in 2023.",
                "OSRM distance and aggregated endpoints are retrospective, not verified pre-dispatch variables.",
                "No verified shipment join to AGMARKNET, action costs, shipment loads or inventory records.",
                "Route type is observational and strongly related to trip geography/distance.",
            ],
        },
    )
    if cfg.get("market_source") == "official":
        from scrc.data.official import market_panel as official_panel

        panel = official_panel(cfg)
    elif cfg.get("market_source") == "historical_archive":
        from scrc.data.archive import market_panel as archive_panel

        panel = archive_panel(cfg)
    else:
        panel = market_panel(cfg)
    panel.to_parquet(ROOT / "data/processed/panel.parquet", index=False)
    from scrc.data.quality import main as quality

    quality()
    pd.read_parquet(ROOT / "data/processed/market_observations.parquet").to_csv(
        ROOT / "data/processed/agmarknet_observations.csv", index=False
    )
    print(
        f"Market panel: {len(panel):,} calendar rows; {panel.observed.sum():,} actual observations. Logistics: {len(trips):,} trips."
    )


if __name__ == "__main__":
    main()
