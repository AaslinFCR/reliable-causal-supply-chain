"""Verified model loading and identical training/serving feature construction."""

import hashlib
import json
import threading
from datetime import date
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from lightgbm import Booster

from scrc.common import digest
from scrc.production.features import CATEGORIES, build_features
from scrc.production.train_forward import predict_bundle


class ForecastService:
    def __init__(self, folder, store):
        self.folder = Path(folder)
        self.store = store
        manifest = json.loads((self.folder / "manifest.json").read_text())
        self.meta = json.loads((self.folder / "model_card.json").read_text())
        names = {r["model_file"] for r in self.meta["routing"].values()}
        for name in [
            "model_card.json",
            "test_metrics.csv",
            "series_metrics.csv",
            *names,
        ]:
            if digest(self.folder / name) != manifest["files"][name]:
                raise ValueError(f"Model artifact checksum mismatch: {name}")
        self.meta = json.loads((self.folder / "model_card.json").read_text())
        self.models = {
            name: Booster(model_file=str(self.folder / name)) for name in names
        }
        self.lock = threading.Lock()

    def forecast(self, state, market, commodity, forecast_date, history=None):
        for column, value in zip(CATEGORIES, [state, market, commodity]):
            if value not in self.meta["vocabulary"][column]:
                raise ValueError(
                    f"Unsupported {column}; choose a trained series from the catalog."
                )
        key = f"{state}|{market}|{commodity}"
        if key not in self.meta["references"]:
            raise ValueError("This market/commodity series has no training support.")
        if pd.Timestamp(forecast_date).date() <= date.fromisoformat(
            self.meta["calibration_end"]
        ):
            raise ValueError("Forecast date must follow the model calibration period.")
        if history is None:
            raw = self.store.history(state, market, commodity, forecast_date)
        else:
            raw = pd.DataFrame(history)
            if raw.empty:
                raise ValueError("Provide actual historical observations.")
            if raw.date.duplicated().any():
                raise ValueError("Duplicate history dates are not allowed.")
            if (pd.to_datetime(raw.date) >= pd.Timestamp(forecast_date)).any():
                raise ValueError(
                    "History must precede the forecast date; current/future outcomes are forbidden."
                )
            for c, v in zip(CATEGORIES, [state, market, commodity]):
                raw[c] = v
        if raw.empty:
            raise ValueError(
                "No recent history exists for this forecast date. Supply refreshed observations."
            )
        raw["date"] = pd.to_datetime(raw.date)
        last = raw.date.max()
        age = (pd.Timestamp(forecast_date) - last).days
        if age > 7:
            raise ValueError(
                "History is stale by more than seven days. Ingest recent observations or provide history."
            )
        target = pd.DataFrame(
            [
                {
                    "state": state,
                    "market": market,
                    "commodity": commodity,
                    "date": pd.Timestamp(forecast_date),
                    "arrivals": np.nan,
                    "modal_price": np.nan,
                }
            ]
        )
        frame = build_features(pd.concat([raw, target], ignore_index=True))
        row = frame.loc[frame.date == pd.Timestamp(forecast_date)].copy()
        if row.empty or row.rolling_mean.isna().any() or row.rolling_mean.iloc[0] <= 0:
            raise ValueError(
                "At least 14 actual observations in the previous 28 calendar days and a positive baseline are required."
            )
        with self.lock:
            point = float(predict_bundle(self.models, row, self.meta)[0])
        scale = max(1, float(row.rolling_mean.iloc[0]))
        q = self.meta["conformal_quantiles"][f"{state}|{commodity}"]
        lower = max(0, point - q * scale)
        upper = point + q * scale
        reference = self.meta["references"][key]["median_arrivals"]
        drift = min(1.0, abs(scale - reference) / max(reference, 1.0))
        width = (upper - lower) / max(point, 1)
        reasons = []
        selected = self.meta["test_metrics"]
        if selected["coverage"] < self.meta["nominal_coverage"]:
            reasons.append("Forward-test interval coverage is below the nominal target")
        if age > 1:
            reasons.append("Recent reporting gap")
        if width > 1.5:
            reasons.append("Wide prediction interval")
        if drift > 0.5:
            reasons.append("Recent inflow differs from training baseline")
        if (
            pd.Timestamp(forecast_date) - pd.Timestamp(self.meta["model_data_cutoff"])
        ).days > 365:
            reasons.append(
                "Model evaluation is over a year older than the forecast date"
            )
        status = "AMBER" if reasons else "GREEN"
        fingerprint = hashlib.sha256(
            raw.sort_values("date")
            .to_json(date_format="iso", orient="records")
            .encode()
        ).hexdigest()
        result = {
            "forecast_id": str(uuid4()),
            "model_version": self.meta["version"],
            "state": state,
            "market": market,
            "commodity": commodity,
            "forecast_date": str(forecast_date),
            "prediction": point,
            "unit": "metric tonnes of market arrivals",
            "interval": {
                "lower": lower,
                "upper": upper,
                "nominal_coverage": self.meta["nominal_coverage"],
                "method": "Frozen state/commodity scaled split conformal",
            },
            "reliability": {
                "status": status,
                "width_ratio": width,
                "drift_score": drift,
                "drift_definition": "Clipped relative change in prior rolling mean versus training median; descriptive, not the research KS detector.",
                "reasons": reasons
                or ["Supported historical series with recent observations"],
                "automation_allowed": False,
                "causal_gate": "RED",
                "action": "Forecast for planner review; no causal intervention is executed.",
            },
            "history_last_date": str(last.date()),
            "history_rows": len(raw),
            "data_cutoff": self.meta["data_cutoff"],
            "actual_if_recorded": self.store.actual(
                state, market, commodity, forecast_date
            ),
            "scope_note": "Predicts wholesale market inflow, not inventory demand or stock-outs. Coverage under future shift is not guaranteed.",
        }
        self.store.save_forecast(result, fingerprint)
        return result
