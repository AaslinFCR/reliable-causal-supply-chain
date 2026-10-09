"""Constrained warehouse planning with explicitly synthetic crisis scenarios."""

import json
import math
from statistics import mean

from scrc.production.decision_options import evaluate_options
from scrc.production.fulfillment import now


class CrisisPlanner:
    def __init__(self, engine):
        self.engine = engine
        with engine.store.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS crisis_demand (
                    step INTEGER,warehouse_id TEXT,commodity TEXT,requested REAL NOT NULL,
                    served REAL NOT NULL,PRIMARY KEY(step,warehouse_id,commodity));
                CREATE TABLE IF NOT EXISTS crisis_latest(id INTEGER PRIMARY KEY CHECK(id=1),payload TEXT NOT NULL);
            """)

    @staticmethod
    def hazards(step):
        day = step % 24
        return {
            "fuel_price": float(95 + step % 12 * 2),
            "baseline_fuel": 95.0,
            "strike_active": day in {8, 9, 10},
            "disaster": {
                "type": "Flood scenario",
                "region": "North",
                "active": day in {12, 13, 14},
                "early_warning": day in {10, 11},
                "days_until_onset": 12 - day if day in {10, 11} else None,
                "basis": "Injected synthetic scenario, not a weather forecast or government warning",
            },
            "demand_surge": day in {4, 5, 6, 7},
            "basis": "Synthetic disruption inputs; not measured fuel, strike or disaster data",
        }

    def advance(self, step):
        engine = self.engine
        hazards = self.hazards(step)
        with engine.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            regions = list(
                db.execute("SELECT * FROM warehouses WHERE kind='regional' ORDER BY id")
            )
            # One replay step represents one synthetic business day. Every regional
            # commodity has an explicit simulated daily demand record.
            for warehouse in regions:
                for i, crop in enumerate(["Rice", "Wheat", "Onion", "Potato"]):
                    if db.execute(
                        "SELECT 1 FROM crisis_demand WHERE step=? AND warehouse_id=? AND commodity=?",
                        (step, warehouse["id"], crop),
                    ).fetchone():
                        continue
                    base = (
                        0.8 + i * 0.1 + (0.2 if warehouse["region"] == "South" else 0)
                    )
                    demand = round(
                        (base + (step % 3 - 1) * 0.1)
                        * (
                            2.5
                            if hazards["demand_surge"]
                            and warehouse["region"] == "South"
                            else 1
                        ),
                        3,
                    )
                    served = min(demand, engine.available(db, warehouse["id"], crop))
                    engine.set_balance(
                        db,
                        warehouse["id"],
                        crop,
                        engine.balance(db, warehouse["id"], crop) - served,
                    )
                    db.execute(
                        "INSERT INTO crisis_demand VALUES (?,?,?,?,?)",
                        (step, warehouse["id"], crop, demand, served),
                    )
                    db.execute(
                        "INSERT INTO movements VALUES (?,?,?)",
                        (
                            f"crisis-demand-{step}-{warehouse['id']}-{crop}",
                            now(),
                            json.dumps(
                                {
                                    "kind": "dispatch",
                                    "warehouse_id": warehouse["id"],
                                    "commodity": crop,
                                    "tonnes": served,
                                    "reference": "SYNTHETIC REGIONAL DEMAND CONSUMPTION",
                                }
                            ),
                        ),
                    )
            plans = []
            for warehouse in regions:
                for crop in ["Rice", "Wheat", "Onion", "Potato"]:
                    past = list(
                        db.execute(
                            "SELECT requested,served FROM crisis_demand WHERE warehouse_id=? AND commodity=? AND step<=? ORDER BY step DESC LIMIT 7",
                            (warehouse["id"], crop, step),
                        )
                    )
                    daily = mean(r["requested"] for r in past)
                    # This is a transparent scenario stress band, not a calibrated
                    # probabilistic interval or a learned customer-demand model.
                    high = daily * 1.25
                    low = daily * 0.75
                    available = engine.available(db, warehouse["id"], crop)
                    cover = available / daily if daily else None
                    shortage = max(0.0, 3 * high - available)
                    blocked = hazards["strike_active"] or (
                        hazards["disaster"]["active"] and warehouse["region"] == "North"
                    )
                    warning = (
                        hazards["disaster"]["early_warning"]
                        and warehouse["region"] == "North"
                    )
                    if warning:
                        shortage = max(shortage, 5 * high - available)
                    free_capacity = max(
                        0.0,
                        warehouse["capacity_tonnes"]
                        - db.execute(
                            "SELECT COALESCE(SUM(tonnes),0) FROM stock WHERE warehouse_id=?",
                            (warehouse["id"],),
                        ).fetchone()[0],
                    )
                    main_budget = max(
                        0.0, engine.available(db, warehouse["parent_id"], crop) - 10.0
                    )
                    feasible = (
                        min(shortage, free_capacity, main_budget)
                        if not blocked
                        else 0.0
                    )
                    action = "Monitor inventory; no transfer required"
                    risk = "LOW"
                    if blocked:
                        risk = "HIGH" if shortage > 0 else "WATCH"
                        action = "Hold affected-route dispatches; protect local stock and verify a safe alternative route with the manager"
                    elif shortage > 0:
                        risk = (
                            "HIGH"
                            if cover is not None and cover < 2
                            else (
                                "WATCH"
                                if warning or (cover is not None and cover < 3)
                                else "LOW"
                            )
                        )
                        action = (
                            "Pre-position stock before the simulated flood"
                            if warning
                            else "Replenish from the main warehouse"
                        )
                        if feasible + 1e-8 < shortage:
                            action += "; capacity or protected main-stock limits prevent full replenishment"
                    elif hazards["fuel_price"] >= 104.5:
                        risk = "WATCH"
                        action = "Consolidate non-urgent loads; keep the three-day inventory buffer before deferring trips"
                    peer_budget = 0.0
                    for peer in regions:
                        if peer["id"] == warehouse["id"]:
                            continue
                        peer_history = list(db.execute(
                            "SELECT requested FROM crisis_demand WHERE warehouse_id=? AND commodity=? AND step<=? ORDER BY step DESC LIMIT 7",
                            (peer["id"], crop, step),
                        ))
                        peer_daily = mean(r["requested"] for r in peer_history) if peer_history else 0
                        if peer_history:
                            peer_budget += max(0.0, engine.available(db, peer["id"], crop) - 3 * peer_daily * 1.25)
                    options = evaluate_options(
                        shortage, main_budget, free_capacity, blocked, len(past),
                        peer_budget, hazards["fuel_price"] >= 104.5,
                    )
                    # Only the original main transfer can execute in this simulation.
                    # No red/grey option passes the automatic action gate.
                    main_option = next(o for o in options if o["id"] == "main_transfer")
                    if main_option["gate"] != "PASS":
                        feasible = 0.0
                    if feasible <= 1e-8 and shortage > 0 and not blocked:
                        action += "; automatic transfer withheld by reliability gate"
                    selected = next((o for o in options if o["status"] == "GREEN"), None)
                    transfer_id = f"crisis-transfer-{step}-{warehouse['id']}-{crop}"
                    moved = 0.0
                    if (
                        feasible > 1e-8
                        and not db.execute(
                            "SELECT 1 FROM movements WHERE request_id=?", (transfer_id,)
                        ).fetchone()
                    ):
                        engine.set_balance(
                            db,
                            warehouse["parent_id"],
                            crop,
                            engine.balance(db, warehouse["parent_id"], crop) - feasible,
                        )
                        engine.set_balance(
                            db,
                            warehouse["id"],
                            crop,
                            engine.balance(db, warehouse["id"], crop) + feasible,
                        )
                        moved = feasible
                        db.execute(
                            "INSERT INTO movements VALUES (?,?,?)",
                            (
                                transfer_id,
                                now(),
                                json.dumps(
                                    {
                                        "kind": "transfer",
                                        "warehouse_id": warehouse["parent_id"],
                                        "destination_id": warehouse["id"],
                                        "commodity": crop,
                                        "tonnes": feasible,
                                        "reference": "SYNTHETIC CRISIS RESPONSE; NO REAL DISPATCH",
                                    }
                                ),
                            ),
                        )
                        engine.event(
                            db,
                            None,
                            "CRISIS_TRANSFER",
                            f"SYNTHETIC {feasible:g} tonnes {crop}: main → {warehouse['id']}; constrained response, not a real dispatch.",
                        )
                    residual = max(0.0, shortage - feasible)
                    distance = 120.0 if warehouse["region"] == "North" else 180.0
                    fuel_cost = (
                        math.ceil(feasible / 10)
                        * distance
                        * 2
                        * 0.3
                        * hazards["fuel_price"]
                        if feasible
                        else 0.0
                    )
                    plans.append(
                        {
                            "warehouse_id": warehouse["id"],
                            "main_available_before_response": engine.available(db, warehouse["parent_id"], crop) + moved,
                            "main_stock_out": engine.available(db, warehouse["parent_id"], crop) + moved <= 1e-8,
                            "main_backup_below_reserve": main_budget <= 1e-8,
                            "demand_surge": hazards["demand_surge"] and warehouse["region"] == "South",
                            "response_options": options,
                            "selected_option_id": selected["id"] if selected else None,
                            "decision_gate": "PASS" if selected else "NO_RELIABLE_OPTION",
                            "region": warehouse["region"],
                            "commodity": crop,
                            "daily_demand_forecast_tonnes": daily,
                            "stress_band_low": low,
                            "stress_band_high": high,
                            "horizon_days": 3,
                            "available_before_response": available,
                            "cover_days_before_response": cover,
                            "shortfall_tonnes": shortage,
                            "proposed_transfer_tonnes": feasible,
                            "simulated_transfer_tonnes": moved,
                            "uncovered_shortfall_tonnes": residual,
                            "risk": risk,
                            "recommendation": action,
                            "route_blocked": blocked,
                            "main_reserve_tonnes": 10,
                            "estimated_round_trip_fuel_cost_inr": fuel_cost,
                            "history_days": len(past),
                            "evidence": "Seven-day mean of synthetic regional demand"
                            if len(past) >= 7
                            else "Limited synthetic history; fewer than seven simulated days",
                            "reliability": "Constraint-checked scenario response; manager review required for real use",
                            "uncertainty": "±25% scenario stress band, not a validated probability interval",
                            "basis": "SYNTHETIC DEMONSTRATION",
                        }
                    )
            latest = {
                "simulation_day": step,
                "hazards": hazards,
                "plans": plans,
                "updated_utc": now(),
                "scope": "Main-to-regional warehouse decision support. No prediction of real disasters or guaranteed crisis prevention. Synthetic transfer execution only.",
            }
            db.execute(
                "INSERT INTO crisis_latest VALUES (1,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                (json.dumps(latest),),
            )
        return latest

    def snapshot(self):
        with self.engine.store.connect() as db:
            row = db.execute("SELECT payload FROM crisis_latest WHERE id=1").fetchone()
        return (
            json.loads(row[0])
            if row
            else {"plans": [], "hazards": {}, "simulation_day": 0}
        )
