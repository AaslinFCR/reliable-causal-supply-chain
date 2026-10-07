"""Normalize original AGMARKNET historical CSVs without a fabricated join."""

import json

import pandas as pd

from scrc.common import ROOT, save_json
from scrc.data.panel import market_features


def market_panel(cfg):
    """Choose market scope using training records, preserving raw source files."""
    pieces = []
    columns = {
        "State Name": "state",
        "Market Name": "market",
        "Reported Date": "date",
        "Arrivals (Tonnes)": "arrivals",
        "Min Price (Rs./Quintal)": "min_price",
        "Max Price (Rs./Quintal)": "max_price",
        "Modal Price (Rs./Quintal)": "modal_price",
    }
    for crop in ["Wheat", "Rice"]:
        for year in [2014, 2015, 2016]:
            path = ROOT / f"data/raw/agmarknet_{crop}_{year}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Real AGMARKNET file missing: {path.name}")
            d = pd.read_csv(
                path,
                header=None,
                names=[
                    "State Name",
                    "District Name",
                    "Market Name",
                    "Variety",
                    "Group",
                    "Arrivals (Tonnes)",
                    "Min Price (Rs./Quintal)",
                    "Max Price (Rs./Quintal)",
                    "Modal Price (Rs./Quintal)",
                    "Reported Date",
                ],
            ).rename(columns=columns)
            missing = set(columns.values()) - set(d)
            if missing:
                raise ValueError(
                    f"{path.name}: missing expected columns {missing}; observed {list(d)}"
                )
            d["commodity"] = crop
            pieces.append(d)
    raw = pd.concat(pieces, ignore_index=True)
    initial = len(raw)
    raw = raw.drop_duplicates()
    raw["date"] = pd.to_datetime(raw.date, format="%d %b %Y", errors="raise")
    for col in ["arrivals", "min_price", "max_price", "modal_price"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    good = (
        (raw.arrivals >= 0)
        & (raw.modal_price > 0)
        & (raw.min_price >= 0)
        & (raw.max_price >= 0)
    )
    good &= ~((raw.max_price > 0) & (raw.modal_price > raw.max_price))
    good &= ~((raw.min_price > 0) & (raw.modal_price < raw.min_price))
    removed = int((~good).sum())
    raw = raw.loc[good]
    raw = raw.loc[(raw.date >= cfg["start_date"]) & (raw.date <= cfg["end_date"])]
    names = ["Uttar Pradesh", "Karnataka", "West Bengal", "Maharashtra"]
    raw = raw.loc[raw.state.isin(names)]
    # The archive states each row is a market-day sale summary. Multiple varieties are
    # aggregated by distinct variety; quantity summed across varieties; weighted price.
    keys = ["state", "market", "commodity", "date"]
    audit_duplicates = int(raw.duplicated(keys).sum())
    raw["price_x_arrivals"] = raw.modal_price * raw.arrivals
    grouped = raw.groupby(keys, as_index=False).agg(
        arrivals=("arrivals", "sum"),
        weighted=("price_x_arrivals", "sum"),
        modal_median=("modal_price", "median"),
        min_price=("min_price", "min"),
        max_price=("max_price", "max"),
    )
    grouped["modal_price"] = grouped.weighted.div(
        grouped.arrivals.where(grouped.arrivals > 0)
    ).fillna(grouped.modal_median)
    grouped = grouped.drop(columns=["weighted", "modal_median"])
    selection = ROOT / "configs/archive_markets.json"
    if selection.exists():
        selected = json.loads(selection.read_text())
    else:
        early = grouped.loc[grouped.date < "2015-07-01"]
        counts = (
            early.groupby(["state", "market", "commodity"]).size().unstack(fill_value=0)
        )
        counts = counts.reindex(columns=["Wheat", "Rice"], fill_value=0)
        counts["score"] = 10 * counts.min(axis=1) + counts.sum(axis=1)
        selected = []
        for state in names:
            rank = counts.loc[state].sort_values(
                "score", ascending=False, kind="stable"
            )
            selected.extend(
                [[state, str(m)] for m in rank.head(cfg["markets_per_state"]).index]
            )
        save_json(selection, selected)
    selected_set = set(map(tuple, selected))
    keep = [(s, m) in selected_set for s, m in zip(grouped.state, grouped.market)]
    observed = grouped.loc[keep].sort_values(keys).reset_index(drop=True)
    observed.to_parquet(
        ROOT / "data/processed/market_observations.parquet", index=False
    )
    panel = market_features(observed, cfg)
    card = {
        "source": "AGMARKNET historical archive by Ian Covert, original government data",
        "documentation": "https://github.com/iancovert/Agmarknet",
        "raw_archive_rows": initial,
        "invalid_rows_removed_all_states": removed,
        "multi_variety_extra_rows": audit_duplicates,
        "selected_markets": len(selected),
        "selected_real_observations": len(observed),
        "calendar_panel_rows": len(panel),
        "unobserved_calendar_rows": int((~panel.observed).sum()),
        "date_min": str(observed.date.min().date()),
        "date_max": str(observed.date.max().date()),
        "units": {"arrivals": "tonnes", "prices": "INR per quintal"},
        "imputation": "No source targets or prices imputed. Calendar gaps remain NaN.",
        "aggregation": "Sum arrivals and arrivals-weighted modal price across reported varieties within each market-day; median price if arrivals zero.",
        "selection": "Up to twelve markets per state, ranked only on January 2014–June 2015 completeness; frozen JSON.",
        "coverage_by_state_commodity": {
            str(k): int(v)
            for k, v in observed.groupby(["state", "commodity"]).size().items()
        },
        "missingness": observed.isna().sum().to_dict(),
        "deviations": [
            "Rice and wheat only: onion/potato absent from this documented archive.",
            "2014–2016 instead of proposed recent years because live sources are restricted/capped.",
            "Actual observed row count reported; no padding to the approximate 50,000 target.",
            "Warehouse inventory, actual stock-outs and action costs absent.",
        ],
    }
    save_json(ROOT / "reports/market_data_card.json", card)
    return panel
