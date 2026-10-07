"""Read-only console demonstration of completed historical replay results."""

import pandas as pd

from scrc.common import ROOT


def main():
    """Display real event observations and the evidence-gated logistics result."""
    market = pd.read_parquet(ROOT / "experiments/market_replay.parquet")
    logistics = pd.read_parquet(ROOT / "experiments/logistics_replay.parquet")
    print("HISTORICAL MARKET REPLAY (not live warehouse demand)")
    columns = [
        "date",
        "state",
        "market",
        "commodity",
        "arrivals",
        "prediction",
        "lower_adaptive",
        "upper_adaptive",
        "shift_score",
    ]
    for name in sorted(market.event.unique()):
        print("\n" + name)
        print(market.loc[market.event == name, columns].head(3).to_string(index=False))
    print("\nSEPARATE OBSERVATIONAL LOGISTICS REPLAY")
    print(
        logistics[
            [
                "date",
                "route_type",
                "delay_factor",
                "propensity",
                "benefit_lower",
                "gate",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )
    print(
        "\nAll automatic actions withheld: causal identification has not been established."
    )
    print("Detailed findings: reports/EVALUATION.md")


if __name__ == "__main__":
    main()
