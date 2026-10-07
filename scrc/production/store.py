"""Transactional observation, prediction and feedback storage."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


class Store:
    def __init__(self, path, seed=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS observations (
                    state TEXT NOT NULL, market TEXT NOT NULL, commodity TEXT NOT NULL,
                    date TEXT NOT NULL, arrivals REAL NOT NULL CHECK(arrivals>=0),
                    modal_price REAL NOT NULL CHECK(modal_price>0),
                    PRIMARY KEY(state,market,commodity,date));
                CREATE TABLE IF NOT EXISTS forecasts (
                    id TEXT PRIMARY KEY, created_utc TEXT NOT NULL, model_version TEXT NOT NULL,
                    state TEXT NOT NULL,market TEXT NOT NULL,commodity TEXT NOT NULL,
                    date TEXT NOT NULL,input_hash TEXT NOT NULL,result TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS feedback (
                    forecast_id TEXT PRIMARY KEY REFERENCES forecasts(id),
                    actual REAL NOT NULL CHECK(actual>=0),covered INTEGER NOT NULL,
                    absolute_error REAL NOT NULL,recorded_utc TEXT NOT NULL);
            """)
            count = db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        if not count and seed and Path(seed).exists():
            frame = pd.read_csv(seed)
            self.ingest(
                frame[
                    ["state", "market", "commodity", "date", "arrivals", "modal_price"]
                ].to_dict("records")
            )

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def ingest(self, records):
        with self.connect() as db:
            before = db.total_changes
            for r in records:
                values = tuple(
                    str(r[k]) for k in ["state", "market", "commodity", "date"]
                ) + (float(r["arrivals"]), float(r["modal_price"]))
                old = db.execute(
                    "SELECT arrivals,modal_price FROM observations WHERE state=? AND market=? AND commodity=? AND date=?",
                    values[:4],
                ).fetchone()
                if old and (
                    old["arrivals"] != values[4] or old["modal_price"] != values[5]
                ):
                    raise ValueError(
                        "An existing observation differs; historical records cannot be silently overwritten."
                    )
                db.execute(
                    "INSERT OR IGNORE INTO observations VALUES (?,?,?,?,?,?)", values
                )
            return db.total_changes - before

    def catalog(self):
        with self.connect() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT state,market,commodity,MIN(date) AS first_date,MAX(date) AS last_date,COUNT(*) AS rows FROM observations GROUP BY state,market,commodity ORDER BY state,market,commodity"
                )
            ]

    def history(self, state, market, commodity, date, days=120):
        start = str((pd.Timestamp(date) - pd.Timedelta(days=days)).date())
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM observations WHERE state=? AND market=? AND commodity=? AND date>=? AND date<? ORDER BY date",
                (state, market, commodity, start, str(date)),
            ).fetchall()
        return pd.DataFrame(
            [dict(r) for r in rows],
            columns=["state", "market", "commodity", "date", "arrivals", "modal_price"],
        )

    def actual(self, state, market, commodity, date):
        with self.connect() as db:
            r = db.execute(
                "SELECT arrivals FROM observations WHERE state=? AND market=? AND commodity=? AND date=?",
                (state, market, commodity, str(date)),
            ).fetchone()
        return r[0] if r else None

    def save_forecast(self, result, input_hash):
        with self.connect() as db:
            db.execute(
                "INSERT INTO forecasts VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    result["forecast_id"],
                    datetime.now(UTC).isoformat(),
                    result["model_version"],
                    result["state"],
                    result["market"],
                    result["commodity"],
                    result["forecast_date"],
                    input_hash,
                    json.dumps(result, allow_nan=False),
                ),
            )

    def record_feedback(self, forecast_id, actual):
        with self.connect() as db:
            r = db.execute(
                "SELECT result FROM forecasts WHERE id=?", (forecast_id,)
            ).fetchone()
            if r is None:
                raise KeyError("Forecast not found.")
            result = json.loads(r[0])
            if pd.Timestamp(result["forecast_date"]).date() > datetime.now(UTC).date():
                raise ValueError(
                    "Future outcomes cannot be recorded before the forecast date."
                )
            old = db.execute(
                "SELECT actual FROM feedback WHERE forecast_id=?", (forecast_id,)
            ).fetchone()
            if old and old[0] != actual:
                raise ValueError(
                    "Feedback already recorded with a different actual outcome."
                )
            covered = (
                result["interval"]["lower"] <= actual <= result["interval"]["upper"]
            )
            db.execute(
                "INSERT OR IGNORE INTO feedback VALUES (?,?,?,?,?)",
                (
                    forecast_id,
                    actual,
                    int(covered),
                    abs(actual - result["prediction"]),
                    datetime.now(UTC).isoformat(),
                ),
            )
            return {
                "forecast_id": forecast_id,
                "covered": covered,
                "absolute_error": abs(actual - result["prediction"]),
                "idempotent": bool(old),
            }

    def monitoring(self):
        with self.connect() as db:
            counts = db.execute(
                "SELECT COUNT(*) AS forecast_count FROM forecasts"
            ).fetchone()
            feedback = db.execute(
                "SELECT COUNT(*) AS feedback_count,AVG(covered) AS empirical_coverage,AVG(absolute_error) AS mae FROM feedback"
            ).fetchone()
            recent = [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT result FROM forecasts ORDER BY created_utc DESC LIMIT 20"
                )
            ]
        return {
            **dict(counts),
            **dict(feedback),
            "recent_forecasts": recent,
            "note": "Feedback statistics describe submitted outcomes and may be selection-biased; they are not an independent benchmark.",
        }
