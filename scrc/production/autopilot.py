"""Autonomous, explicitly synthetic operations with historical dataset context."""

import csv
import json
from pathlib import Path

from scrc.common import ROOT
from scrc.production.crisis import CrisisPlanner
from scrc.production.fulfillment import now


class Autopilot:
    def __init__(self, engine, enabled=True, source=None, forecast_service=None):
        self.engine = engine
        self.enabled = enabled
        self.forecast_service = forecast_service
        self.crisis = CrisisPlanner(engine)
        self.source = Path(
            source or ROOT / "artifacts/production/demo_market_replay.csv"
        )
        self.rows = []
        if self.source.exists():
            with self.source.open(encoding="utf-8-sig", newline="") as file:
                self.rows = list(csv.DictReader(file))
        with engine.store.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS autopilot_cursor(id INTEGER PRIMARY KEY CHECK(id=1),step INTEGER NOT NULL);
                INSERT OR IGNORE INTO autopilot_cursor VALUES(1,0);
                CREATE TABLE IF NOT EXISTS autopilot_history(step INTEGER PRIMARY KEY,payload TEXT NOT NULL);
            """)
        if enabled:
            engine.seed()
            engine.configure(
                {
                    "running": True,
                    "generate_orders": False,
                    "reorder_point": 0,
                    "target_stock": 0,
                }
            )
            history = ROOT / "artifacts/production/demo_forecast_history.csv"
            if history.exists():
                with history.open(encoding="utf-8-sig", newline="") as file:
                    observations = [
                        {
                            **r,
                            "arrivals": float(r["arrivals"]),
                            "modal_price": float(r["modal_price"]),
                        }
                        for r in csv.DictReader(file)
                    ]
                engine.store.ingest(observations)

    def step(self):
        if not self.enabled or not self.rows:
            return
        with self.engine.store.connect() as db:
            if not self.engine.settings(db)["running"]:
                return
            step = (
                db.execute("SELECT step FROM autopilot_cursor WHERE id=1").fetchone()[0]
                + 1
            )
        row = dict(self.rows[(step - 1) % len(self.rows)])
        if self.forecast_service:
            try:
                row["forecast"] = self.forecast_service.forecast(
                    row["state"], row["market"], row["commodity"], row["date"]
                )
            except ValueError as error:
                row["forecast_error"] = str(error)
        # Synthetic disruptions/fuel variation make the operational behavior visible.
        # They are not historical strike/fuel measurements from either dataset.
        self.engine.configure(
            {
                "strike_fraction": 1.0 if step % 24 in {8, 9, 10} else 0.0,
                "fuel_price": float(95 + (step % 12) * 2),
            }
        )
        self.crisis.advance(step)
        # Simulate supplier receipts through a separate ledger. Actual warehouse
        # stock is never touched; automatic opening balances are clearly synthetic.
        with self.engine.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if step % 12 == 0:
                for crop in ["Rice", "Wheat", "Onion", "Potato"]:
                    total = db.execute(
                        "SELECT COALESCE(SUM(tonnes),0) FROM stock WHERE warehouse_id='DEMO-MAIN'"
                    ).fetchone()[0]
                    capacity = db.execute(
                        "SELECT capacity_tonnes FROM warehouses WHERE id='DEMO-MAIN'"
                    ).fetchone()[0]
                    amount = min(30.0, max(0.0, capacity - total))
                    identifier = f"autopilot-receipt-{step}-{crop}"
                    if (
                        amount
                        and not db.execute(
                            "SELECT 1 FROM movements WHERE request_id=?", (identifier,)
                        ).fetchone()
                    ):
                        self.engine.set_balance(
                            db,
                            "DEMO-MAIN",
                            crop,
                            self.engine.balance(db, "DEMO-MAIN", crop) + amount,
                        )
                        db.execute(
                            "INSERT INTO movements VALUES (?,?,?)",
                            (
                                identifier,
                                now(),
                                json.dumps(
                                    {
                                        "kind": "receipt",
                                        "commodity": crop,
                                        "tonnes": amount,
                                        "warehouse_id": "DEMO-MAIN",
                                        "reference": "SYNTHETIC AUTOPILOT SUPPLIER RECEIPT",
                                    }
                                ),
                            ),
                        )
                        self.engine.event(
                            db,
                            None,
                            "SUPPLIER_RECEIPT",
                            f"SYNTHETIC {amount:g} tonnes {crop}; no real purchase or receipt.",
                        )
            db.execute(
                "INSERT OR IGNORE INTO autopilot_history VALUES (?,?)",
                (step, json.dumps(row)),
            )
            db.execute("UPDATE autopilot_cursor SET step=? WHERE id=1", (step,))
            db.execute("DELETE FROM autopilot_history WHERE step<?", (step - 40,))

    def snapshot(self):
        with self.engine.store.connect() as db:
            step = db.execute(
                "SELECT step FROM autopilot_cursor WHERE id=1"
            ).fetchone()[0]
            history = [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT payload FROM autopilot_history ORDER BY step DESC"
                )
            ]
        logistics_path = ROOT / "artifacts/production/demo_logistics_context.json"
        logistics = (
            json.loads(logistics_path.read_text()) if logistics_path.exists() else {}
        )
        return {
            "enabled": self.enabled,
            "source_available": bool(self.rows),
            "source_rows": len(self.rows),
            "step": step,
            "crisis": self.crisis.snapshot(),
            "historical_market_rows": history,
            "logistics_context": logistics,
            "basis": "Historical dataset replay with isolated SYNTHETIC inventory, regional consumption, main-to-regional transfers, supplier receipts, strikes, fuel variation and flood scenarios. No live business or disaster feed.",
            "replay_loops": step // max(1, len(self.rows)),
            "research_model_version": "18e5692d481b0f84",
            "research_results": {
                "forward_rows": 4965,
                "wape_pct": 22.809,
                "mae_tonnes": 31.242,
                "coverage_pct": 91.944,
                "nominal_coverage_pct": 95,
                "predictive_gate": "AMBER",
                "causal_gate": "RED",
            },
        }
