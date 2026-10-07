"""Persistent operational alerts and an explicitly enabled TLS email outbox."""

import json
import os
import re
import smtplib
import ssl
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage
from uuid import uuid4

from fastapi import HTTPException

EMAIL = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def stamp():
    return datetime.now(UTC).isoformat()


def smtp_ready():
    return all(
        os.environ.get(k)
        for k in ("SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_FROM")
    ) and bool(EMAIL.fullmatch(os.environ.get("SMTP_FROM", "")))


def send_email(recipient, subject, body, identifier):
    """No plaintext transport or credential-bearing logs; SMTP acceptance is not receipt."""
    if not smtp_ready():
        raise RuntimeError("Email service is not configured.")
    message = EmailMessage()
    message["From"] = os.environ["SMTP_FROM"]
    message["To"] = recipient
    message["Subject"] = subject
    message["Message-ID"] = f"<{identifier}@{os.environ['SMTP_FROM'].split('@')[1]}>"
    message.set_content(body)
    with smtplib.SMTP(
        os.environ["SMTP_HOST"], int(os.environ.get("SMTP_PORT", "587")), timeout=10
    ) as server:
        server.ehlo()
        server.starttls(context=ssl.create_default_context())
        server.ehlo()
        server.login(os.environ["SMTP_USERNAME"], os.environ["SMTP_PASSWORD"])
        refused = server.send_message(message)
        if refused:
            raise RuntimeError("Email recipient was refused.")


class Alerts:
    def __init__(self, engine):
        self.engine = engine
        self.store = engine.store
        with self.store.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS alert_settings (id INTEGER PRIMARY KEY CHECK(id=1),payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,incident_key TEXT NOT NULL,status TEXT NOT NULL,
                    created_utc TEXT NOT NULL,updated_utc TEXT NOT NULL,acknowledged_utc TEXT,payload TEXT NOT NULL);
                CREATE UNIQUE INDEX IF NOT EXISTS active_alert_key ON alerts(incident_key) WHERE status='ACTIVE';
                CREATE TABLE IF NOT EXISTS alert_outbox (
                    id TEXT PRIMARY KEY,alert_id TEXT NOT NULL REFERENCES alerts(id),recipient TEXT NOT NULL,
                    state TEXT NOT NULL,attempts INTEGER NOT NULL DEFAULT 0,next_attempt_utc TEXT NOT NULL,
                    error TEXT,accepted_utc TEXT,UNIQUE(alert_id,recipient));
            """)
            db.execute(
                "INSERT OR IGNORE INTO alert_settings VALUES (1,?)",
                (
                    json.dumps(
                        {
                            "enabled": True,
                            "email_enabled": False,
                            "owner_email": "",
                            "warehouse_emails": {},
                            "low_stock_tonnes": 12.0,
                            "stalled_minutes": 60,
                            "fuel_increase_pct": 10.0,
                            "demand_increase_pct": 30.0,
                            "last_scan_utc": None,
                            "last_error": None,
                        }
                    ),
                ),
            )

    @staticmethod
    def settings(db):
        return json.loads(
            db.execute("SELECT payload FROM alert_settings WHERE id=1").fetchone()[0]
        )

    def configure(self, changes):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            settings = self.settings(db)
            if self.engine.mode == "demo" and changes.get("email_enabled"):
                raise HTTPException(422, "Synthetic alerts never send external email.")
            addresses = [
                changes.get("owner_email", ""),
                *changes.get("warehouse_emails", {}).values(),
            ]
            if any(a and (len(a) > 254 or not EMAIL.fullmatch(a)) for a in addresses):
                raise HTTPException(
                    422, "Use a single valid email address for each recipient."
                )
            if any(not a for a in changes.get("warehouse_emails", {}).values()):
                raise HTTPException(
                    422, "Warehouse recipient addresses cannot be empty."
                )
            for warehouse in changes.get("warehouse_emails", {}):
                if not db.execute(
                    "SELECT 1 FROM warehouses WHERE id=?", (warehouse,)
                ).fetchone():
                    raise HTTPException(
                        422, "Recipient mapping refers to an unknown warehouse."
                    )
            settings.update(changes)
            if settings["email_enabled"] and (
                not smtp_ready()
                or not (settings["owner_email"] or settings["warehouse_emails"])
            ):
                raise HTTPException(
                    422,
                    "Configure server SMTP secrets and at least one recipient before enabling email.",
                )
            db.execute(
                "UPDATE alert_settings SET payload=? WHERE id=1",
                (json.dumps(settings),),
            )
        return settings

    def conditions(self, db, settings):
        conditions = {}

        def add(key, kind, title, detail, warehouse=None, severity="WARNING"):
            conditions[key] = {
                "kind": kind,
                "title": title,
                "detail": detail,
                "warehouse_id": warehouse,
                "severity": severity,
                "basis": "SYNTHETIC DEMONSTRATION"
                if self.engine.mode == "demo"
                else "Recorded business data",
            }

        warehouses = list(db.execute("SELECT * FROM warehouses"))
        for w in warehouses:
            stock = list(
                db.execute("SELECT * FROM stock WHERE warehouse_id=?", (w["id"],))
            )
            if not stock:
                add(
                    f"opening:{w['id']}",
                    "MISSING_INVENTORY",
                    "Opening inventory is missing",
                    "Record a verified opening stock count before accepting orders.",
                    w["id"],
                )
            for row in stock:
                available = self.engine.available(db, w["id"], row["commodity"])
                if available <= settings["low_stock_tonnes"]:
                    add(
                        f"stock:{w['id']}:{row['commodity']}",
                        "LOW_STOCK",
                        f"Low available {row['commodity']} stock",
                        f"Available: {available:g} tonnes. Threshold: {settings['low_stock_tonnes']:g} tonnes. Check replenishment; no purchase has been placed.",
                        w["id"],
                        "CRITICAL" if available == 0 else "WARNING",
                    )
        for order in db.execute(
            "SELECT * FROM wf_orders WHERE status NOT IN ('DELIVERED','CANCELLED')"
        ):
            if order["status"] == "BACKORDER":
                add(
                    f"backorder:{order['id']}",
                    "BACKORDER",
                    f"Order {order['id']} cannot be allocated",
                    "Insufficient available stock. The worker will retry allocation automatically after inventory changes.",
                    order["warehouse_id"],
                    "CRITICAL",
                )
            age = (
                datetime.now(UTC) - datetime.fromisoformat(order["updated_utc"])
            ).total_seconds() / 60
            if age >= settings["stalled_minutes"]:
                add(
                    f"stalled:{order['id']}",
                    "STALLED_ORDER",
                    f"Order {order['id']} is stalled",
                    f"No state transition for at least {settings['stalled_minutes']} minutes; current state {order['status']}. Verify dispatch/provider events.",
                    order["warehouse_id"],
                    "CRITICAL",
                )
        workflow = self.engine.settings(db)
        if db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='crisis_latest'"
        ).fetchone():
            saved = db.execute(
                "SELECT payload FROM crisis_latest WHERE id=1"
            ).fetchone()
            if saved:
                crisis = json.loads(saved[0])
                disaster = crisis["hazards"]["disaster"]
                if disaster["active"] or disaster["early_warning"]:
                    for warehouse in warehouses:
                        if warehouse["region"] == disaster["region"]:
                            add(
                                f"disaster:{warehouse['id']}",
                                "DISASTER_SCENARIO",
                                "Synthetic flood warning for North",
                                "Scenario active: hold affected-route dispatches."
                                if disaster["active"]
                                else f"Injected scenario begins in {disaster['days_until_onset']} simulated days; pre-position feasible stock before route closure. Not a real weather forecast.",
                                warehouse["id"],
                                "CRITICAL" if disaster["active"] else "WARNING",
                            )
                for warehouse in warehouses:
                    affected = [
                        p
                        for p in crisis["plans"]
                        if p["warehouse_id"] == warehouse["id"] and p["risk"] != "LOW"
                    ]
                    if affected:
                        worst = max(
                            affected,
                            key=lambda p: (
                                p["uncovered_shortfall_tonnes"] + p["shortfall_tonnes"]
                            ),
                        )
                        add(
                            f"crisis:{warehouse['id']}",
                            "WAREHOUSE_CRISIS",
                            f"Crisis response for {warehouse['name']}",
                            f"{worst['commodity']}: {worst['recommendation']}. Proposed {worst['proposed_transfer_tonnes']:.2f} tonnes; residual shortfall {worst['uncovered_shortfall_tonnes']:.2f} tonnes. Synthetic scenario; constraint checks do not guarantee real outcomes.",
                            warehouse["id"],
                            "CRITICAL"
                            if any(p["risk"] == "HIGH" for p in affected)
                            else "WARNING",
                        )
        if workflow.get("last_error"):
            add(
                "worker:error",
                "WORKER_ERROR",
                "Order processing needs attention",
                "The background worker reported an error. Inspect server logs and the workflow state.",
                severity="CRITICAL",
            )
        if self.engine.mode == "demo":
            if workflow["strike_fraction"] > 0:
                add(
                    "demo:strike",
                    "STRIKE",
                    "Simulated transport disruption",
                    f"Synthetic disrupted fraction: {workflow['strike_fraction']:g}.",
                )
            if workflow["fuel_price"] >= 95 * (1 + settings["fuel_increase_pct"] / 100):
                add(
                    "demo:fuel",
                    "FUEL_INCREASE",
                    "Simulated fuel-price increase",
                    f"Synthetic fuel price {workflow['fuel_price']:g} INR/litre against fixed demo reference 95 INR/litre.",
                )
        else:
            today = (
                (datetime.now(UTC) + timedelta(hours=5, minutes=30)).date().isoformat()
            )
            signals = [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT payload FROM operational_signals ORDER BY created_utc,id"
                )
            ]
            fuels = {}
            for signal in signals:
                if signal["basis"] != "recorded":
                    continue
                if signal["kind"] == "fuel":
                    fuels.setdefault(signal["region"], []).append(signal)
                if (
                    signal["kind"] == "strike"
                    and signal["value"] > 0
                    and signal["start"] <= today <= signal["end"]
                ):
                    affected = [
                        w["id"]
                        for w in warehouses
                        if w["region"].casefold() == signal["region"].casefold()
                    ]
                    for warehouse in affected or [None]:
                        add(
                            f"strike:{signal['region']}:{warehouse}",
                            "STRIKE",
                            f"Recorded disruption in {signal['region']}",
                            f"Disrupted transport fraction {signal['value']:g}. Source: {signal['source']}",
                            warehouse,
                        )
            for region, records in fuels.items():
                if len(records) < 2:
                    continue
                previous, current = records[-2:]
                if current["start"] <= today <= current["end"] and current[
                    "value"
                ] >= previous["value"] * (1 + settings["fuel_increase_pct"] / 100):
                    add(
                        f"fuel:{region}",
                        "FUEL_INCREASE",
                        f"Recorded fuel-price increase in {region}",
                        f"Latest {current['value']:g} vs previous recorded {previous['value']:g} INR/litre. Verify transporter costs.",
                    )
        for row in db.execute("""SELECT warehouse_id,commodity,
            SUM(CASE WHEN julianday(created_utc)>=julianday('now','-7 days') THEN tonnes ELSE 0 END) AS recent,
            SUM(CASE WHEN julianday(created_utc)<julianday('now','-7 days') THEN tonnes ELSE 0 END) AS previous
            FROM wf_orders WHERE status!='CANCELLED' AND julianday(created_utc)>=julianday('now','-14 days') GROUP BY warehouse_id,commodity"""):
            if row["previous"] > 0 and row["recent"] >= row["previous"] * (
                1 + settings["demand_increase_pct"] / 100
            ):
                add(
                    f"demand:{row['warehouse_id']}:{row['commodity']}",
                    "DEMAND_INCREASE",
                    f"{row['commodity']} orders increased",
                    f"Last seven days {row['recent']:g} vs preceding seven days {row['previous']:g} tonnes. Compare business context before replenishment.",
                    row["warehouse_id"],
                )
        return conditions

    def scan(self):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            settings = self.settings(db)
            if not settings["enabled"]:
                return
            conditions = self.conditions(db, settings)
            active = {
                r["incident_key"]: r
                for r in db.execute("SELECT * FROM alerts WHERE status='ACTIVE'")
            }
            for key, row in active.items():
                if key not in conditions:
                    db.execute(
                        "UPDATE alerts SET status='RESOLVED',updated_utc=? WHERE id=?",
                        (stamp(), row["id"]),
                    )
                    db.execute(
                        "UPDATE alert_outbox SET state='CANCELLED' WHERE alert_id=? AND state IN ('PENDING','FAILED')",
                        (row["id"],),
                    )
            for key, payload in conditions.items():
                identifier = active[key]["id"] if key in active else str(uuid4())
                if key not in active:
                    db.execute(
                        "INSERT INTO alerts VALUES (?,?,?,?,?,NULL,?)",
                        (
                            identifier,
                            key,
                            "ACTIVE",
                            stamp(),
                            stamp(),
                            json.dumps(payload),
                        ),
                    )
                else:
                    db.execute(
                        "UPDATE alerts SET payload=?,updated_utc=? WHERE id=?",
                        (json.dumps(payload), stamp(), identifier),
                    )
                if (
                    self.engine.mode == "live"
                    and settings["email_enabled"]
                    and not (key in active and active[key]["acknowledged_utc"])
                ):
                    recipients = {
                        settings["owner_email"],
                        settings["warehouse_emails"].get(payload["warehouse_id"], ""),
                    } - {""}
                    for recipient in recipients:
                        db.execute(
                            "INSERT OR IGNORE INTO alert_outbox(id,alert_id,recipient,state,next_attempt_utc) VALUES (?,?,?,'PENDING',?)",
                            (str(uuid4()), identifier, recipient, stamp()),
                        )
            settings["last_scan_utc"] = stamp()
            settings["last_error"] = None
            db.execute(
                "UPDATE alert_settings SET payload=? WHERE id=1",
                (json.dumps(settings),),
            )

    def deliver(self):
        if self.engine.mode != "live":
            return
        for _ in range(5):
            with self.store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                settings = self.settings(db)
                if (
                    not settings["enabled"]
                    or not settings["email_enabled"]
                    or not smtp_ready()
                ):
                    return
                row = db.execute(
                    """SELECT o.*,a.payload FROM alert_outbox o JOIN alerts a ON a.id=o.alert_id
                    WHERE a.status='ACTIVE' AND a.acknowledged_utc IS NULL
                    AND o.state IN ('PENDING','SENDING') AND o.next_attempt_utc<=? ORDER BY o.next_attempt_utc LIMIT 1""",
                    (stamp(),),
                ).fetchone()
                if not row:
                    return
                payload = json.loads(row["payload"])
                allowed = {
                    settings["owner_email"],
                    settings["warehouse_emails"].get(payload["warehouse_id"], ""),
                } - {""}
                if row["recipient"] not in allowed:
                    db.execute(
                        "UPDATE alert_outbox SET state='CANCELLED' WHERE id=?",
                        (row["id"],),
                    )
                    continue
                db.execute(
                    "UPDATE alert_outbox SET state='SENDING',attempts=attempts+1,next_attempt_utc=? WHERE id=?",
                    ((datetime.now(UTC) + timedelta(minutes=5)).isoformat(), row["id"]),
                )
            try:
                send_email(
                    row["recipient"],
                    f"SupplyChain {payload['severity']}: {payload['title']}",
                    f"{payload['basis']}\nWarehouse: {payload['warehouse_id'] or 'Business-wide'}\n{payload['detail']}\nAlert ID: {row['alert_id']}\nReview the application alert inbox. This alert does not execute a purchase or dispatch.",
                    row["id"],
                )
            except (OSError, ValueError, RuntimeError) as error:
                attempts = row["attempts"] + 1
                with self.store.connect() as db:
                    db.execute(
                        "UPDATE alert_outbox SET state=?,error=?,next_attempt_utc=? WHERE id=?",
                        (
                            "FAILED" if attempts >= 5 else "PENDING",
                            type(error).__name__,
                            (
                                datetime.now(UTC)
                                + timedelta(seconds=min(3600, 60 * 2**attempts))
                            ).isoformat(),
                            row["id"],
                        ),
                    )
            else:
                with self.store.connect() as db:
                    db.execute(
                        "UPDATE alert_outbox SET state='SMTP_ACCEPTED',error=NULL,accepted_utc=? WHERE id=?",
                        (stamp(), row["id"]),
                    )

    def poll(self):
        self.scan()
        self.deliver()

    def acknowledge(self, identifier):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute(
                "SELECT 1 FROM alerts WHERE id=?", (identifier,)
            ).fetchone():
                raise HTTPException(404, "Alert not found.")
            db.execute(
                "UPDATE alerts SET acknowledged_utc=COALESCE(acknowledged_utc,?) WHERE id=?",
                (stamp(), identifier),
            )
            db.execute(
                "UPDATE alert_outbox SET state='CANCELLED' WHERE alert_id=? AND state='PENDING'",
                (identifier,),
            )
        return {"acknowledged": True}

    def snapshot(self):
        with self.store.connect() as db:
            settings = self.settings(db)
            alerts = [
                {**dict(r), **json.loads(r["payload"])}
                for r in db.execute(
                    "SELECT * FROM alerts ORDER BY created_utc DESC LIMIT 100"
                )
            ]
            counts = dict(
                db.execute("SELECT status,COUNT(*) FROM alerts GROUP BY status")
            )
            outbox = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM alert_outbox ORDER BY next_attempt_utc DESC LIMIT 100"
                )
            ]
        return {
            "mode": self.engine.mode,
            "settings": settings,
            "alerts": alerts,
            "counts": counts,
            "email_outbox": outbox,
            "smtp_configured": smtp_ready(),
            "external_delivery": "Disabled for synthetic demo"
            if self.engine.mode == "demo"
            else "SMTP acceptance is not proof of inbox receipt",
            "scan_interval_seconds": 5,
        }
