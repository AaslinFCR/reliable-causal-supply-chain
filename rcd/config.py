"""Global configuration for the RCD (Reliable Causal Decisions) study.

All costs are illustrative planning units (INR-like), not measured company costs.
"""
from dataclasses import dataclass, field
import pandas as pd

SEED = 2026

REGIONS = ["Bengaluru", "Chennai", "Mumbai", "Delhi"]
PRODUCTS = ["rice", "atta", "sugar", "cooking_oil", "detergent",
            "biscuits", "tea", "soap", "dal", "salt"]

# unit price (INR) used to scale costs
PRICE = {"rice": 55, "atta": 42, "sugar": 44, "cooking_oil": 140, "detergent": 110,
         "biscuits": 25, "tea": 120, "soap": 35, "dal": 110, "salt": 22}
# base daily demand (units) before regional scaling
BASE_UNITS = {"rice": 420, "atta": 360, "sugar": 300, "cooking_oil": 260, "detergent": 200,
              "biscuits": 620, "tea": 160, "soap": 300, "dal": 240, "salt": 280}
REGION_SCALE = {"Bengaluru": 1.0, "Chennai": 0.9, "Mumbai": 1.2, "Delhi": 1.1}
# regional taste multipliers (rice-heavy south, wheat-heavy north)
TASTE = {("Chennai", "rice"): 1.35, ("Bengaluru", "rice"): 1.2, ("Delhi", "rice"): 0.75,
         ("Delhi", "atta"): 1.45, ("Mumbai", "atta"): 1.1, ("Chennai", "atta"): 0.7,
         ("Bengaluru", "atta"): 0.85, ("Delhi", "tea"): 1.3, ("Chennai", "dal"): 1.2}

START, END = "2021-01-01", "2025-12-31"
TEST_YEAR = 2025
LOG_YEARS = (2023, 2024)          # historical planner logs used to learn effects
CAUSAL_TRAIN_END = "2024-09-30"   # logs after this date are held out for calibration

H = 7            # forecast horizon (days of demand summed)
LEAD = 3         # nominal main->regional lead time (days)
REVIEW = 7       # routine review period (Mondays)
ALPHA = 0.10     # nominal miscoverage (90% intervals)

ACTIONS = ["none", "expedite", "transfer", "advance"]
ACTION_FRAC = 0.5   # every intervention moves 0.5 x the 7-day point forecast

# Transfer lead times between regional warehouses (days)
TRANSFER_LT = {frozenset(["Bengaluru", "Chennai"]): 1, frozenset(["Mumbai", "Bengaluru"]): 2,
               frozenset(["Mumbai", "Delhi"]): 2, frozenset(["Chennai", "Mumbai"]): 2,
               frozenset(["Delhi", "Bengaluru"]): 3, frozenset(["Delhi", "Chennai"]): 3}


@dataclass
class Costs:
    hold_rate: float = 0.0015     # x price per unit per day
    short_rate: float = 0.40      # x price per unit of unmet demand
    exp_rate: float = 0.08        # x price per expedited unit
    exp_fixed: float = 2500.0
    tr_rate: float = 0.035
    tr_fixed: float = 1500.0
    adv_fixed: float = 300.0      # admin cost of an advance order


COSTS = Costs()

FESTIVALS = {
    # name: (dates, regions, product uplift dict, ramp days)
    "diwali": (["2021-11-04", "2022-10-24", "2023-11-12", "2024-11-01", "2025-10-20"], REGIONS,
               {"cooking_oil": 1.9, "sugar": 1.8, "biscuits": 1.7, "atta": 1.3, "dal": 1.3,
                "detergent": 1.4, "soap": 1.3, "tea": 1.2}, 10),
    "pongal": (["2021-01-14", "2022-01-14", "2023-01-15", "2024-01-15", "2025-01-14"],
               ["Chennai", "Bengaluru"], {"rice": 1.6, "sugar": 1.5, "dal": 1.3, "cooking_oil": 1.3}, 6),
    "ganesh": (["2021-09-10", "2022-08-31", "2023-09-19", "2024-09-07", "2025-08-27"],
               ["Mumbai", "Bengaluru"], {"sugar": 1.6, "cooking_oil": 1.4, "atta": 1.2}, 6),
    "holi": (["2021-03-29", "2022-03-18", "2023-03-08", "2024-03-25", "2025-03-14"],
             ["Delhi", "Mumbai"], {"sugar": 1.5, "cooking_oil": 1.4, "atta": 1.3, "biscuits": 1.3}, 5),
    "dussehra": (["2021-10-15", "2022-10-05", "2023-10-24", "2024-10-12", "2025-10-02"],
                 ["Delhi", "Bengaluru"], {"sugar": 1.3, "cooking_oil": 1.3, "biscuits": 1.2}, 5),
}


def dates():
    return pd.date_range(START, END, freq="D")
