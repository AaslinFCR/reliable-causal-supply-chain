"""Durable order state machine; isolated simulation and externally confirmed live events."""

import json
import math
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import HTTPException


def now():
    return datetime.now(UTC).isoformat()


class Fulfillment:
    def __init__(self, store, mode="demo"):
        if mode not in {"demo", "live"}:
            raise ValueError("FULFILLMENT_MODE must be demo or live.")
        self.store = store
        self.mode = mode
        with store.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS warehouses (
                    id TEXT PRIMARY KEY,name TEXT NOT NULL,kind TEXT NOT NULL,
                    region TEXT NOT NULL,capacity_tonnes REAL NOT NULL,parent_id TEXT REFERENCES warehouses(id));
                CREATE TABLE IF NOT EXISTS stock (
                    warehouse_id TEXT REFERENCES warehouses(id),commodity TEXT NOT NULL,
                    tonnes REAL NOT NULL CHECK(tonnes>=0),PRIMARY KEY(warehouse_id,commodity));
                CREATE TABLE IF NOT EXISTS movements (
                    request_id TEXT PRIMARY KEY,created_utc TEXT NOT NULL,payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS wf_settings (id INTEGER PRIMARY KEY CHECK(id=1),payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS wf_orders (
                    id TEXT PRIMARY KEY,created_utc TEXT NOT NULL,warehouse_id TEXT NOT NULL REFERENCES warehouses(id),
                    commodity TEXT NOT NULL,tonnes REAL NOT NULL CHECK(tonnes>0),destination TEXT NOT NULL,
                    status TEXT NOT NULL,tracking_reference TEXT,updated_utc TEXT NOT NULL,
                    input TEXT NOT NULL,shipped_tick INTEGER,fuel_cost_inr REAL NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS wf_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,created_utc TEXT NOT NULL,
                    order_id TEXT,kind TEXT NOT NULL,detail TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS wf_callbacks (event_id TEXT PRIMARY KEY,payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS wf_jobs (
                    id TEXT PRIMARY KEY,order_id TEXT NOT NULL UNIQUE REFERENCES wf_orders(id),
                    state TEXT NOT NULL,created_utc TEXT NOT NULL);
            """)
            old = db.execute("SELECT payload FROM wf_settings WHERE id=1").fetchone()
            if old and json.loads(old[0])["mode"] != mode:
                raise ValueError(
                    "Workflow database belongs to another mode; use a separate database."
                )
            if not old:
                settings = {
                    "mode": mode,
                    "running": mode == "live",
                    "tick": 0,
                    "generate_orders": False,
                    "strike_fraction": 0.0,
                    "fuel_price": 95.0,
                    "distance_km": 100.0,
                    "litres_per_100km": 30.0,
                    "truck_capacity": 20.0,
                    "reorder_point": 12.0,
                    "target_stock": 30.0,
                    "last_tick_utc": None,
                    "last_error": None,
                }
                db.execute(
                    "INSERT INTO wf_settings VALUES (1,?)", (json.dumps(settings),)
                )

    def settings(self, db):
        return json.loads(
            db.execute("SELECT payload FROM wf_settings WHERE id=1").fetchone()[0]
        )

    def configure(self, changes):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            s = self.settings(db)
            if self.mode == "live" and changes.get("generate_orders"):
                raise HTTPException(
                    422, "Synthetic order generation is disabled in live mode."
                )
            if self.mode == "live" and any(k not in {"running"} for k in changes):
                raise HTTPException(
                    422, "Simulation settings cannot be applied to live operations."
                )
            s.update(changes)
            if s["target_stock"] < s["reorder_point"]:
                raise HTTPException(
                    422, "Target stock must be at least the reorder point."
                )
            db.execute("UPDATE wf_settings SET payload=? WHERE id=1", (json.dumps(s),))
        return s

    def event(self, db, order_id, kind, detail):
        db.execute(
            "INSERT INTO wf_events(created_utc,order_id,kind,detail) VALUES (?,?,?,?)",
            (now(), order_id, kind, detail),
        )

    @staticmethod
    def balance(db, warehouse, crop):
        r = db.execute(
            "SELECT tonnes FROM stock WHERE warehouse_id=? AND commodity=?",
            (warehouse, crop),
        ).fetchone()
        return float(r[0]) if r else 0.0

    @staticmethod
    def reserved(db, warehouse, crop):
        return float(
            db.execute(
                "SELECT COALESCE(SUM(tonnes),0) FROM wf_orders WHERE warehouse_id=? AND commodity=? AND status='RESERVED'",
                (warehouse, crop),
            ).fetchone()[0]
        )

    def available(self, db, warehouse, crop):
        return max(
            0, self.balance(db, warehouse, crop) - self.reserved(db, warehouse, crop)
        )

    @staticmethod
    def set_balance(db, warehouse, crop, value):
        if value < -1e-8:
            raise ValueError("Stock invariant violated.")
        db.execute(
            "INSERT INTO stock VALUES (?,?,?) ON CONFLICT(warehouse_id,commodity) DO UPDATE SET tonnes=excluded.tonnes",
            (warehouse, crop, max(0, value)),
        )

    def seed(self):
        if self.mode != "demo":
            raise HTTPException(409, "Synthetic inventory is forbidden in live mode.")
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT COUNT(*) FROM warehouses").fetchone()[0]:
                return {"seeded": False, "reason": "Demo already initialized."}
            for values in [
                ("DEMO-MAIN", "DEMO Main Warehouse", "main", "Central", 1000.0, None),
                (
                    "DEMO-NORTH",
                    "DEMO Northern Warehouse",
                    "regional",
                    "North",
                    100.0,
                    "DEMO-MAIN",
                ),
                (
                    "DEMO-SOUTH",
                    "DEMO Southern Warehouse",
                    "regional",
                    "South",
                    100.0,
                    "DEMO-MAIN",
                ),
            ]:
                db.execute("INSERT INTO warehouses VALUES (?,?,?,?,?,?)", values)
                for crop in ["Rice", "Wheat", "Onion", "Potato"]:
                    tonnes = 100.0 if values[2] == "main" else 15.0
                    self.set_balance(db, values[0], crop, tonnes)
                    db.execute(
                        "INSERT INTO movements VALUES (?,?,?)",
                        (
                            f"opening-{values[0]}-{crop}",
                            now(),
                            json.dumps(
                                {
                                    "kind": "receipt",
                                    "warehouse_id": values[0],
                                    "commodity": crop,
                                    "tonnes": tonnes,
                                    "reference": "SYNTHETIC DEMONSTRATION OPENING BALANCE",
                                }
                            ),
                        ),
                    )
            self.event(
                db,
                None,
                "DEMO_INITIALIZED",
                "Synthetic stock is isolated from recorded warehouse inventory and research datasets.",
            )
        return {"seeded": True, "basis": "Synthetic demonstration only."}

    def create_order(self, payload):
        serialized = json.dumps(payload, sort_keys=True)
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute(
                "SELECT * FROM wf_orders WHERE id=?", (payload["id"],)
            ).fetchone()
            if old:
                if old["input"] != serialized:
                    raise HTTPException(
                        409, "Order ID already exists with different inputs."
                    )
                return {**dict(old), "unchanged": True}
            w = db.execute(
                "SELECT * FROM warehouses WHERE id=?", (payload["warehouse_id"],)
            ).fetchone()
            if w is None:
                raise HTTPException(
                    404, "Warehouse not found in the current workflow mode."
                )
            db.execute(
                "INSERT INTO wf_orders(id,created_utc,warehouse_id,commodity,tonnes,destination,status,updated_utc,input) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    payload["id"],
                    now(),
                    payload["warehouse_id"],
                    payload["commodity"],
                    payload["tonnes"],
                    payload["destination"],
                    "NEW",
                    now(),
                    serialized,
                ),
            )
            self.event(
                db,
                payload["id"],
                "ORDER_CREATED",
                f"{self.mode}: order persisted; no stock has left the warehouse.",
            )
        return {"id": payload["id"], "status": "NEW", "unchanged": False}

    def cancel(self, order_id):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            r = db.execute("SELECT * FROM wf_orders WHERE id=?", (order_id,)).fetchone()
            if r is None:
                raise HTTPException(404, "Order not found.")
            if r["status"] == "CANCELLED":
                return {"status": "CANCELLED", "unchanged": True}
            if r["status"] not in {"NEW", "BACKORDER", "RESERVED"}:
                raise HTTPException(
                    409,
                    "Dispatched orders require a separate return workflow and cannot be cancelled here.",
                )
            db.execute(
                "UPDATE wf_orders SET status='CANCELLED',updated_utc=? WHERE id=?",
                (now(), order_id),
            )
            db.execute(
                "UPDATE wf_jobs SET state='CANCELLED' WHERE order_id=?", (order_id,)
            )
            self.event(
                db,
                order_id,
                "CANCELLED",
                "Reservation released; no on-hand stock change.",
            )
        return {"status": "CANCELLED", "unchanged": False}

    def transfer(self, db, warehouse, crop, s):
        w = db.execute("SELECT * FROM warehouses WHERE id=?", (warehouse,)).fetchone()
        if not w["parent_id"] or s["strike_fraction"] >= 1:
            return
        available = self.available(db, warehouse, crop)
        if available > s["reorder_point"]:
            return
        free = (
            w["capacity_tonnes"]
            - db.execute(
                "SELECT COALESCE(SUM(tonnes),0) FROM stock WHERE warehouse_id=?",
                (warehouse,),
            ).fetchone()[0]
        )
        amount = min(
            s["target_stock"] - available,
            self.available(db, w["parent_id"], crop),
            max(0, free),
        )
        if amount <= 1e-9:
            return
        self.set_balance(
            db, w["parent_id"], crop, self.balance(db, w["parent_id"], crop) - amount
        )
        self.set_balance(
            db, warehouse, crop, self.balance(db, warehouse, crop) + amount
        )
        db.execute(
            "INSERT INTO movements VALUES (?,?,?)",
            (
                str(uuid4()),
                now(),
                json.dumps(
                    {
                        "kind": "transfer",
                        "warehouse_id": w["parent_id"],
                        "destination_id": warehouse,
                        "commodity": crop,
                        "tonnes": amount,
                        "reference": "SYNTHETIC DEMO reorder-point rule",
                    }
                ),
            ),
        )
        self.event(
            db,
            None,
            "REPLENISHED",
            f"DEMO {amount:g} tonnes {crop}: {w['parent_id']} → {warehouse}",
        )

    def dispatch(self, db, order, reference, tick, s):
        if (
            self.balance(db, order["warehouse_id"], order["commodity"]) + 1e-9
            < order["tonnes"]
        ):
            raise HTTPException(409, "On-hand stock is below the reserved quantity.")
        self.set_balance(
            db,
            order["warehouse_id"],
            order["commodity"],
            self.balance(db, order["warehouse_id"], order["commodity"])
            - order["tonnes"],
        )
        trips = math.ceil(order["tonnes"] / s["truck_capacity"])
        fuel = (
            trips * s["distance_km"] * 2 * s["litres_per_100km"] / 100 * s["fuel_price"]
            if self.mode == "demo"
            else 0.0
        )
        db.execute(
            "UPDATE wf_orders SET status='DISPATCHED',tracking_reference=?,updated_utc=?,shipped_tick=?,fuel_cost_inr=? WHERE id=?",
            (reference, now(), tick, fuel, order["id"]),
        )
        db.execute(
            "INSERT INTO movements VALUES (?,?,?)",
            (
                f"fulfillment-{uuid4()}",
                now(),
                json.dumps(
                    {
                        "kind": "dispatch",
                        "warehouse_id": order["warehouse_id"],
                        "commodity": order["commodity"],
                        "tonnes": order["tonnes"],
                        "reference": reference,
                    }
                ),
            ),
        )
        db.execute(
            "UPDATE wf_jobs SET state='CONFIRMED' WHERE order_id=?", (order["id"],)
        )
        self.event(db, order["id"], "DISPATCHED", reference)

    def tick(self, force=False):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            s = self.settings(db)
            if not s["running"] and not force:
                return {"advanced": False}
            if force and self.mode != "demo":
                raise HTTPException(
                    409, "Manual simulated ticks are disabled in live mode."
                )
            s["tick"] += 1
            s["last_tick_utc"] = now()
            s["last_error"] = None
            if self.mode == "demo" and s["generate_orders"] and s["tick"] % 4 == 0:
                total = db.execute("SELECT COUNT(*) FROM wf_orders").fetchone()[0]
                if total < 1000:
                    crop = ["Rice", "Wheat", "Onion", "Potato"][(s["tick"] // 4) % 4]
                    warehouse = ["DEMO-NORTH", "DEMO-SOUTH"][(s["tick"] // 4) % 2]
                    if db.execute(
                        "SELECT 1 FROM warehouses WHERE id=?", (warehouse,)
                    ).fetchone():
                        payload = {
                            "id": f"DEMO-AUTO-{s['tick']}",
                            "warehouse_id": warehouse,
                            "commodity": crop,
                            "tonnes": float(2 + s["tick"] % 5),
                            "destination": "Synthetic customer region",
                        }
                        db.execute(
                            "INSERT INTO wf_orders(id,created_utc,warehouse_id,commodity,tonnes,destination,status,updated_utc,input) VALUES (?,?,?,?,?,?,?,?,?)",
                            (
                                payload["id"],
                                now(),
                                warehouse,
                                crop,
                                payload["tonnes"],
                                payload["destination"],
                                "NEW",
                                now(),
                                json.dumps(payload, sort_keys=True),
                            ),
                        )
                        self.event(
                            db,
                            payload["id"],
                            "ORDER_CREATED",
                            "SYNTHETIC automatic demonstration order",
                        )
            if self.mode == "demo":
                for row in db.execute(
                    "SELECT id FROM warehouses WHERE kind='regional'"
                ).fetchall():
                    for crop in ["Rice", "Wheat", "Onion", "Potato"]:
                        self.transfer(db, row["id"], crop, s)
            orders = db.execute(
                "SELECT * FROM wf_orders WHERE status NOT IN ('DELIVERED','CANCELLED') ORDER BY created_utc,id"
            ).fetchall()
            for order in orders:
                if order["status"] in {"NEW", "BACKORDER"}:
                    if (
                        self.available(db, order["warehouse_id"], order["commodity"])
                        + 1e-9
                        >= order["tonnes"]
                    ):
                        db.execute(
                            "UPDATE wf_orders SET status='RESERVED',updated_utc=? WHERE id=?",
                            (now(), order["id"]),
                        )
                        self.event(
                            db,
                            order["id"],
                            "RESERVED",
                            "Available stock reserved atomically.",
                        )
                        if self.mode == "live":
                            db.execute(
                                "INSERT OR IGNORE INTO wf_jobs VALUES (?,?,?,?)",
                                (str(uuid4()), order["id"], "AWAITING_PROVIDER", now()),
                            )
                    elif order["status"] != "BACKORDER":
                        db.execute(
                            "UPDATE wf_orders SET status='BACKORDER',updated_utc=? WHERE id=?",
                            (now(), order["id"]),
                        )
                        self.event(
                            db,
                            order["id"],
                            "BACKORDER",
                            "Insufficient available inventory; order retained for retry.",
                        )
                elif self.mode == "demo" and order["status"] == "RESERVED":
                    stride = math.ceil(1 / max(0.01, 1 - s["strike_fraction"]))
                    if s["strike_fraction"] < 1 and s["tick"] % stride == 0:
                        self.dispatch(
                            db, order, "DEMO-TRACK-" + order["id"], s["tick"], s
                        )
                elif self.mode == "demo" and order["status"] == "DISPATCHED":
                    db.execute(
                        "UPDATE wf_orders SET status='IN_TRANSIT',updated_utc=? WHERE id=?",
                        (now(), order["id"]),
                    )
                    self.event(
                        db,
                        order["id"],
                        "IN_TRANSIT",
                        "SIMULATED carrier event; not a physical shipment.",
                    )
                elif (
                    self.mode == "demo"
                    and order["status"] == "IN_TRANSIT"
                    and s["tick"] - order["shipped_tick"] >= 3
                    and s["strike_fraction"] < 1
                ):
                    db.execute(
                        "UPDATE wf_orders SET status='DELIVERED',updated_utc=? WHERE id=?",
                        (now(), order["id"]),
                    )
                    self.event(
                        db,
                        order["id"],
                        "DELIVERED",
                        "SIMULATED proof of delivery; not a real delivery.",
                    )
            db.execute("UPDATE wf_settings SET payload=? WHERE id=1", (json.dumps(s),))
        return {"advanced": True, "tick": s["tick"], "mode": self.mode}

    def callback(self, payload):
        if self.mode != "live":
            raise HTTPException(
                409, "Real provider callbacks are disabled in demo mode."
            )
        serialized = json.dumps(payload, sort_keys=True)
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute(
                "SELECT payload FROM wf_callbacks WHERE event_id=?",
                (payload["event_id"],),
            ).fetchone()
            if old:
                if old[0] != serialized:
                    raise HTTPException(
                        409, "Event ID reused with a different payload."
                    )
                return {"accepted": True, "unchanged": True}
            order = db.execute(
                "SELECT * FROM wf_orders WHERE id=?", (payload["order_id"],)
            ).fetchone()
            if order is None:
                raise HTTPException(404, "Order not found.")
            permitted = {
                "RESERVED": "DISPATCHED",
                "DISPATCHED": "IN_TRANSIT",
                "IN_TRANSIT": "DELIVERED",
            }
            if permitted.get(order["status"]) != payload["status"]:
                raise HTTPException(409, "Out-of-order or invalid shipment transition.")
            if (
                order["tracking_reference"]
                and order["tracking_reference"] != payload["tracking_reference"]
            ):
                raise HTTPException(
                    409, "Tracking reference differs from the dispatched shipment."
                )
            if payload["status"] == "DISPATCHED":
                self.dispatch(
                    db, order, payload["tracking_reference"], 0, self.settings(db)
                )
                self.event(
                    db, order["id"], "DISPATCH_EVIDENCE", payload["evidence_reference"]
                )
            else:
                db.execute(
                    "UPDATE wf_orders SET status=?,updated_utc=? WHERE id=?",
                    (payload["status"], now(), order["id"]),
                )
                self.event(
                    db, order["id"], payload["status"], payload["evidence_reference"]
                )
            db.execute(
                "INSERT INTO wf_callbacks VALUES (?,?)",
                (payload["event_id"], serialized),
            )
        return {"accepted": True, "unchanged": False}

    def snapshot(self):
        with self.store.connect() as db:
            s = self.settings(db)
            inventory = []
            for w in db.execute("SELECT * FROM warehouses ORDER BY kind,id").fetchall():
                for r in db.execute(
                    "SELECT * FROM stock WHERE warehouse_id=? ORDER BY commodity",
                    (w["id"],),
                ).fetchall():
                    reserved = self.reserved(db, w["id"], r["commodity"])
                    inventory.append(
                        {
                            "warehouse_id": w["id"],
                            "warehouse_name": w["name"],
                            "kind": w["kind"],
                            "commodity": r["commodity"],
                            "on_hand_tonnes": r["tonnes"],
                            "reserved_tonnes": reserved,
                            "available_tonnes": max(0, r["tonnes"] - reserved),
                        }
                    )
            orders = [
                dict(r)
                for r in db.execute(
                    "SELECT id,created_utc,warehouse_id,commodity,tonnes,destination,status,tracking_reference,updated_utc,fuel_cost_inr FROM wf_orders ORDER BY created_utc DESC LIMIT 100"
                )
            ]
            counts = {
                r["status"]: r["n"]
                for r in db.execute(
                    "SELECT status,COUNT(*) AS n FROM wf_orders GROUP BY status"
                )
            }
            events = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM wf_events ORDER BY id DESC LIMIT 100"
                )
            ]
            jobs = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM wf_jobs ORDER BY created_utc DESC LIMIT 100"
                )
            ]
            demand = [
                dict(r)
                for r in db.execute(
                    "SELECT commodity,COUNT(*) AS orders,SUM(tonnes) AS ordered_tonnes FROM wf_orders WHERE status!='CANCELLED' AND julianday(created_utc)>=julianday('now','-7 days') GROUP BY commodity"
                )
            ]
        return {
            "mode": self.mode,
            "basis": (
                "SYNTHETIC isolated demonstration"
                if self.mode == "demo"
                else "Recorded orders and confirmed warehouse/carrier events"
            ),
            "settings": s,
            "inventory": inventory,
            "orders": orders,
            "counts": counts,
            "events": events,
            "shipment_jobs": jobs,
            "seven_day_order_demand": demand,
            "integrations": {
                "erp": "Not connected; use the authenticated order/warehouse APIs.",
                "carrier": (
                    "Simulation only"
                    if self.mode == "demo"
                    else "No carrier booking connector configured; authenticated confirmation events required."
                ),
            },
            "scope": "Rule-based operational workflow, separate from the research causal gate and forecast accuracy evaluation.",
        }
