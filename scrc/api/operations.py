"""Recorded warehouse ledger and explicit, non-causal disruption scenarios."""

import json
from datetime import UTC, date, datetime, timedelta
from typing import Literal
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field, model_validator


def install_operations(app, authorize, base):
    class Warehouse(base):
        id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,40}$")
        name: str = Field(min_length=1, max_length=120)
        kind: Literal["main", "regional"]
        region: str = Field(min_length=1, max_length=120)
        capacity_tonnes: float = Field(gt=0, le=1e9)
        parent_id: str | None = Field(default=None, max_length=40)

        @model_validator(mode="after")
        def hierarchy(self):
            if self.kind == "main" and self.parent_id:
                raise ValueError("Main warehouses cannot have a parent.")
            if self.kind == "regional" and not self.parent_id:
                raise ValueError("Regional warehouses require a main warehouse parent.")
            return self

    class Movement(base):
        request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
        warehouse_id: str = Field(min_length=1, max_length=40)
        commodity: Literal["Rice", "Wheat", "Onion", "Potato"]
        kind: Literal["receipt", "dispatch", "transfer"]
        tonnes: float = Field(gt=0, le=1e9)
        destination_id: str | None = Field(default=None, max_length=40)
        reference: str = Field(min_length=1, max_length=200)

    class Signal(base):
        kind: Literal["demand", "fuel", "strike"]
        region: str = Field(min_length=1, max_length=120)
        commodity: Literal["Rice", "Wheat", "Onion", "Potato"] | None = None
        start: date
        end: date
        value: float = Field(ge=0, le=1e9)
        source: str = Field(min_length=1, max_length=300)
        basis: Literal["recorded", "scenario"]

        @model_validator(mode="after")
        def validity(self):
            if self.end < self.start:
                raise ValueError("End date precedes start date.")
            if self.kind == "strike" and self.value > 1:
                raise ValueError("Strike disruption fraction must be between 0 and 1.")
            if self.kind == "demand" and self.commodity is None:
                raise ValueError("Demand requires a commodity.")
            if self.kind == "fuel" and self.value <= 0:
                raise ValueError("Fuel price must be positive.")
            return self

    class Plan(base):
        warehouse_id: str = Field(min_length=1, max_length=40)
        commodity: Literal["Rice", "Wheat", "Onion", "Potato"]
        as_of: date
        horizon_days: int = Field(ge=1, le=90)
        demand_tonnes_daily: float = Field(ge=0, le=1e7)
        demand_change_pct: float = Field(ge=-100, le=1000)
        strike_days: int = Field(ge=0, le=90)
        strike_fraction: float = Field(ge=0, le=1)
        distance_km: float = Field(ge=0, le=100000)
        litres_per_100km: float = Field(gt=0, le=200)
        fuel_price_inr_litre: float = Field(gt=0, le=10000)
        baseline_fuel_price: float = Field(gt=0, le=10000)
        truck_capacity_tonnes: float = Field(gt=0, le=1000)

    router = APIRouter(prefix="/v1/operations", dependencies=[Depends(authorize)])

    def initialize():
        with app.state.store.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS warehouses (
                    id TEXT PRIMARY KEY,name TEXT NOT NULL,kind TEXT NOT NULL,
                    region TEXT NOT NULL,capacity_tonnes REAL NOT NULL,parent_id TEXT REFERENCES warehouses(id));
                CREATE TABLE IF NOT EXISTS stock (
                    warehouse_id TEXT REFERENCES warehouses(id),commodity TEXT NOT NULL,
                    tonnes REAL NOT NULL CHECK(tonnes>=0),PRIMARY KEY(warehouse_id,commodity));
                CREATE TABLE IF NOT EXISTS movements (
                    request_id TEXT PRIMARY KEY,created_utc TEXT NOT NULL,payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS operational_signals (
                    id TEXT PRIMARY KEY,created_utc TEXT NOT NULL,payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS planning_runs (
                    id TEXT PRIMARY KEY,created_utc TEXT NOT NULL,payload TEXT NOT NULL,result TEXT NOT NULL);
            """)

    app.state.initialize_operations = initialize

    def warehouse(db, identifier):
        row = db.execute(
            "SELECT * FROM warehouses WHERE id=?", (identifier,)
        ).fetchone()
        if row is None:
            raise HTTPException(404, "Warehouse not found.")
        return dict(row)

    def quantity(db, identifier, crop):
        row = db.execute(
            "SELECT tonnes FROM stock WHERE warehouse_id=? AND commodity=?",
            (identifier, crop),
        ).fetchone()
        return float(row[0]) if row else 0.0

    def set_quantity(db, identifier, crop, value):
        db.execute(
            "INSERT INTO stock VALUES (?,?,?) ON CONFLICT(warehouse_id,commodity) DO UPDATE SET tonnes=excluded.tonnes",
            (identifier, crop, value),
        )

    @router.post("/warehouses")
    def create_warehouse(payload: Warehouse):
        values = payload.model_dump()
        with app.state.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute(
                "SELECT * FROM warehouses WHERE id=?", (payload.id,)
            ).fetchone()
            if old:
                if dict(old) != values:
                    raise HTTPException(
                        409, "Warehouse ID already exists with different details."
                    )
                return {"warehouse": values, "unchanged": True}
            if payload.parent_id and warehouse(db, payload.parent_id)["kind"] != "main":
                raise HTTPException(422, "Parent must be a main warehouse.")
            db.execute(
                "INSERT INTO warehouses VALUES (?,?,?,?,?,?)", tuple(values.values())
            )
        return {"warehouse": values, "unchanged": False}

    @router.get("/inventory")
    def inventory():
        with app.state.store.connect() as db:
            warehouses = [
                dict(r) for r in db.execute("SELECT * FROM warehouses ORDER BY kind,id")
            ]
            for w in warehouses:
                w["stock"] = [
                    dict(r)
                    for r in db.execute(
                        "SELECT commodity,tonnes FROM stock WHERE warehouse_id=? ORDER BY commodity",
                        (w["id"],),
                    )
                ]
                w["total_tonnes"] = sum(r["tonnes"] for r in w["stock"])
                w["utilization"] = w["total_tonnes"] / w["capacity_tonnes"]
            ledger = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM movements ORDER BY created_utc DESC LIMIT 30"
                )
            ]
        return {
            "warehouses": warehouses,
            "recent_movements": ledger,
            "basis": "User-recorded inventory; no warehouse stock is inferred from market arrivals.",
        }

    @router.post("/movements")
    def movement(payload: Movement):
        data = payload.model_dump(mode="json")
        serialized = json.dumps(data, sort_keys=True)
        with app.state.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute(
                "SELECT payload FROM movements WHERE request_id=?",
                (payload.request_id,),
            ).fetchone()
            if old:
                if old[0] != serialized:
                    raise HTTPException(
                        409, "Request ID already used for a different movement."
                    )
                return {"recorded": True, "unchanged": True}
            origin = warehouse(db, payload.warehouse_id)
            current = quantity(db, payload.warehouse_id, payload.commodity)
            change = payload.tonnes if payload.kind == "receipt" else -payload.tonnes
            if (
                change < 0
                and db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='wf_orders'"
                ).fetchone()
            ):
                reserved = db.execute(
                    "SELECT COALESCE(SUM(tonnes),0) FROM wf_orders WHERE warehouse_id=? AND commodity=? AND status='RESERVED'",
                    (payload.warehouse_id, payload.commodity),
                ).fetchone()[0]
                if current + change < reserved - 1e-9:
                    raise HTTPException(
                        409,
                        "Movement would consume inventory reserved for customer orders.",
                    )
            if current + change < -1e-9:
                raise HTTPException(409, "Insufficient inventory.")
            total = db.execute(
                "SELECT COALESCE(SUM(tonnes),0) FROM stock WHERE warehouse_id=?",
                (payload.warehouse_id,),
            ).fetchone()[0]
            if total + change > origin["capacity_tonnes"] + 1e-9:
                raise HTTPException(409, "Warehouse capacity exceeded.")
            if payload.kind == "transfer":
                if (
                    not payload.destination_id
                    or payload.destination_id == payload.warehouse_id
                ):
                    raise HTTPException(
                        422, "Transfer requires a different destination warehouse."
                    )
                destination = warehouse(db, payload.destination_id)
                destination_total = db.execute(
                    "SELECT COALESCE(SUM(tonnes),0) FROM stock WHERE warehouse_id=?",
                    (payload.destination_id,),
                ).fetchone()[0]
                if (
                    destination_total + payload.tonnes
                    > destination["capacity_tonnes"] + 1e-9
                ):
                    raise HTTPException(409, "Destination capacity exceeded.")
                set_quantity(
                    db,
                    payload.destination_id,
                    payload.commodity,
                    quantity(db, payload.destination_id, payload.commodity)
                    + payload.tonnes,
                )
            elif payload.destination_id:
                raise HTTPException(422, "Destination is only allowed for transfers.")
            set_quantity(
                db, payload.warehouse_id, payload.commodity, max(0, current + change)
            )
            db.execute(
                "INSERT INTO movements VALUES (?,?,?)",
                (payload.request_id, datetime.now(UTC).isoformat(), serialized),
            )
        return {"recorded": True, "unchanged": False}

    @router.post("/signals")
    def signal(payload: Signal):
        identifier = str(uuid4())
        with app.state.store.connect() as db:
            db.execute(
                "INSERT INTO operational_signals VALUES (?,?,?)",
                (
                    identifier,
                    datetime.now(UTC).isoformat(),
                    json.dumps(payload.model_dump(mode="json")),
                ),
            )
        return {
            "id": identifier,
            **payload.model_dump(mode="json"),
            "units": {
                "demand": "tonnes/day",
                "fuel": "INR/litre",
                "strike": "disrupted transport fraction",
            }[payload.kind],
        }

    @router.get("/signals")
    def signals():
        with app.state.store.connect() as db:
            return {
                "signals": [
                    {
                        "id": r["id"],
                        "created_utc": r["created_utc"],
                        **json.loads(r["payload"]),
                    }
                    for r in db.execute(
                        "SELECT * FROM operational_signals ORDER BY created_utc DESC LIMIT 100"
                    )
                ]
            }

    @router.get("/market")
    def market(state: str, market: str, commodity: str, as_of: date):
        frame = app.state.store.history(
            state, market, commodity, as_of + timedelta(days=1), 28
        )
        if frame.empty:
            raise HTTPException(404, "No recorded market observations for this period.")
        dates = pd.to_datetime(frame.date)
        recent = frame.loc[dates > pd.Timestamp(as_of) - pd.Timedelta(days=7)]
        previous = frame.loc[
            (dates <= pd.Timestamp(as_of) - pd.Timedelta(days=7))
            & (dates > pd.Timestamp(as_of) - pd.Timedelta(days=14))
        ]

        def change(column):
            a = float(recent[column].mean()) if len(recent) else None
            b = float(previous[column].mean()) if len(previous) else None
            return {
                "recent_mean": a,
                "previous_mean": b,
                "change_pct": (
                    100 * (a / b - 1)
                    if a is not None and b is not None and b > 0
                    else None
                ),
            }

        return {
            "as_of": str(as_of),
            "last_recorded_date": str(frame.date.max()),
            "recent_reports": len(recent),
            "previous_reports": len(previous),
            "price_inr_quintal": change("modal_price"),
            "arrivals_tonnes": change("arrivals"),
            "basis": "Recorded AGMARKNET observations; two seven-calendar-day reporting windows. Arrivals are not demand.",
        }

    @router.post("/plan")
    def plan(payload: Plan):
        import math

        if payload.strike_days > payload.horizon_days:
            raise HTTPException(422, "Strike days cannot exceed the planning horizon.")
        with app.state.store.connect() as db:
            w = warehouse(db, payload.warehouse_id)
            stock = quantity(db, w["id"], payload.commodity)
            parent_stock = (
                quantity(db, w["parent_id"], payload.commodity)
                if w["parent_id"]
                else 0.0
            )
            demand = payload.demand_tonnes_daily * (1 + payload.demand_change_pct / 100)
            required = demand * payload.horizon_days
            gap = max(0, required - stock)
            transport_factor = (
                1 - payload.strike_fraction * payload.strike_days / payload.horizon_days
            )
            capacity_left = max(
                0,
                w["capacity_tonnes"]
                - sum(
                    r[0]
                    for r in db.execute(
                        "SELECT tonnes FROM stock WHERE warehouse_id=?", (w["id"],)
                    )
                ),
            )
            transferable = min(parent_stock * transport_factor, gap, capacity_left)
            trips = (
                math.ceil(transferable / payload.truck_capacity_tonnes)
                if transferable
                else 0
            )
            litres = trips * payload.distance_km * 2 * payload.litres_per_100km / 100
            result = {
                "as_of": str(payload.as_of),
                "inventory_snapshot_utc": datetime.now(UTC).isoformat(),
                "id": str(uuid4()),
                "basis": "User-entered deterministic scenario; not a fitted demand model or identified causal effect.",
                "warehouse_id": w["id"],
                "commodity": payload.commodity,
                "inventory_tonnes": stock,
                "parent_available_tonnes": parent_stock,
                "adjusted_daily_demand_tonnes": demand,
                "horizon_demand_tonnes": required,
                "days_of_cover": stock / demand if demand > 0 else None,
                "initial_gap_tonnes": gap,
                "suggested_transfer_tonnes": transferable,
                "residual_gap_tonnes": max(0, gap - transferable),
                "truck_trips": trips,
                "round_trip_fuel_litres": litres,
                "fuel_cost_inr": litres * payload.fuel_price_inr_litre,
                "fuel_cost_change_inr": litres
                * (payload.fuel_price_inr_litre - payload.baseline_fuel_price),
                "transport_availability_factor": transport_factor,
                "automation_allowed": False,
                "assumptions": [
                    "Parent stock is unreserved and usable.",
                    "Strike proportion reduces available parent supply linearly; this is a scenario assumption.",
                    "Transfer is capped by current warehouse free capacity.",
                    "Distance is one-way; fuel cost includes a return trip.",
                    "No route optimization, lead time, spoilage or non-fuel transport costs are estimated.",
                    "Inventory reflects current ledger, not a historical stock reconstruction for as_of.",
                ],
            }
            db.execute(
                "INSERT INTO planning_runs VALUES (?,?,?,?)",
                (
                    result["id"],
                    datetime.now(UTC).isoformat(),
                    json.dumps(payload.model_dump(mode="json")),
                    json.dumps(result),
                ),
            )
        return result

    @router.get("/plans")
    def saved_plans():
        with app.state.store.connect() as db:
            return {
                "plans": [
                    {
                        "created_utc": r["created_utc"],
                        "inputs": json.loads(r["payload"]),
                        "result": json.loads(r["result"]),
                    }
                    for r in db.execute(
                        "SELECT * FROM planning_runs ORDER BY created_utc DESC LIMIT 30"
                    )
                ]
            }

    app.include_router(router)
