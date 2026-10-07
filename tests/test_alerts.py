"""Durable exception handling, routing, privacy and autonomous replay behavior."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from scrc.api.app import create_app
from scrc.production.alerts import Alerts


@pytest.fixture
def client(tmp_path):
    with TestClient(
        create_app(database_path=tmp_path / "alerts.sqlite", seed_path="")
    ) as c:
        with c.app.state.store.connect() as db:
            db.execute(
                "INSERT INTO warehouses VALUES ('TEST','Artificial test warehouse','main','Test',100,NULL)"
            )
            db.execute("INSERT INTO stock VALUES ('TEST','Rice',10)")
        yield c


def test_dedup_resolution_recurrence_and_acknowledgement(client):
    monitor = client.app.state.alerts["live"]
    monitor.scan()
    first = monitor.snapshot()["alerts"][0]
    for _ in range(3):
        monitor.scan()
    assert len(monitor.snapshot()["alerts"]) == 1
    monitor.acknowledge(first["id"])
    restored = Alerts(client.app.state.fulfillment["live"])
    assert restored.snapshot()["alerts"][0]["acknowledged_utc"]
    with monitor.store.connect() as db:
        db.execute("UPDATE stock SET tonnes=30")
    monitor.scan()
    assert monitor.snapshot()["counts"] == {"RESOLVED": 1}
    with monitor.store.connect() as db:
        db.execute("UPDATE stock SET tonnes=5")
    monitor.scan()
    assert monitor.snapshot()["counts"] == {"ACTIVE": 1, "RESOLVED": 1}
    assert sum(bool(r["acknowledged_utc"]) for r in monitor.snapshot()["alerts"]) == 1


def smtp_config(monkeypatch):
    for key, value in {
        "SMTP_HOST": "smtp.example.test",
        "SMTP_USERNAME": "test",
        "SMTP_PASSWORD": "artificial-secret",
        "SMTP_FROM": "sender@example.test",
    }.items():
        monkeypatch.setenv(key, value)


def test_email_routing_and_no_duplicates(client, monkeypatch):
    smtp_config(monkeypatch)
    sent = []
    monkeypatch.setattr(
        "scrc.production.alerts.send_email",
        lambda recipient, *args: sent.append(recipient),
    )
    monitor = client.app.state.alerts["live"]
    monitor.configure(
        {
            "owner_email": "owner@example.test",
            "warehouse_emails": {"TEST": "manager@example.test"},
            "email_enabled": True,
        }
    )
    monitor.poll()
    monitor.poll()
    assert sorted(sent) == ["manager@example.test", "owner@example.test"]
    assert all(
        r["state"] == "SMTP_ACCEPTED" for r in monitor.snapshot()["email_outbox"]
    )


def test_delivery_failures_back_off_and_persist(client, monkeypatch):
    smtp_config(monkeypatch)

    def fail(*_):
        raise OSError("Artificial transport failure with private details")

    monkeypatch.setattr("scrc.production.alerts.send_email", fail)
    monitor = client.app.state.alerts["live"]
    monitor.configure({"owner_email": "owner@example.test", "email_enabled": True})
    monitor.poll()
    first = monitor.snapshot()["email_outbox"][0]
    assert first["attempts"] == 1 and first["state"] == "PENDING"
    assert first["error"] == "OSError"
    monitor.deliver()
    assert monitor.snapshot()["email_outbox"][0]["attempts"] == 1
    for _ in range(4):
        with monitor.store.connect() as db:
            db.execute("UPDATE alert_outbox SET next_attempt_utc='2000-01-01'")
        monitor.deliver()
    saved = Alerts(client.app.state.fulfillment["live"]).snapshot()["email_outbox"][0]
    assert saved["state"] == "FAILED" and saved["attempts"] == 5


def test_resolution_cancels_unsent_email(client, monkeypatch):
    smtp_config(monkeypatch)
    monitor = client.app.state.alerts["live"]
    monitor.configure({"owner_email": "owner@example.test", "email_enabled": True})
    monitor.scan()
    with monitor.store.connect() as db:
        db.execute("UPDATE stock SET tonnes=30")
    monitor.scan()
    assert monitor.snapshot()["email_outbox"][0]["state"] == "CANCELLED"


def test_synthetic_mode_never_emails_and_validates_recipients(client, monkeypatch):
    smtp_config(monkeypatch)
    with pytest.raises(Exception, match="Synthetic alerts"):
        client.app.state.alerts["demo"].configure({"email_enabled": True})
    for payload in [
        {"owner_email": "bad\nBcc: victim@example.test"},
        {"warehouse_emails": {"UNKNOWN": "manager@example.test"}},
    ]:
        assert (
            client.post("/v1/fulfillment/live/alert-settings", json=payload).status_code
            == 422
        )
    assert (
        client.post(
            "/v1/fulfillment/live/alert-settings", json={"low_stock_tonnes": -1}
        ).status_code
        == 422
    )


def test_backorders_stalls_and_recorded_strikes_only(client):
    engine = client.app.state.fulfillment["live"]
    engine.create_order(
        {
            "id": "TEST-ORDER",
            "warehouse_id": "TEST",
            "commodity": "Rice",
            "tonnes": 11.0,
            "destination": "Artificial test",
        }
    )
    engine.tick()
    monitor = client.app.state.alerts["live"]
    today = datetime.now(UTC).date()
    with monitor.store.connect() as db:
        db.execute(
            "UPDATE wf_orders SET updated_utc=?",
            ((datetime.now(UTC) - timedelta(hours=2)).isoformat(),),
        )
        for basis in ["scenario", "recorded"]:
            db.execute(
                "INSERT INTO operational_signals VALUES (?,?,?)",
                (
                    basis,
                    datetime.now(UTC).isoformat(),
                    json.dumps(
                        {
                            "kind": "strike",
                            "region": "Test",
                            "value": 1,
                            "start": (today - timedelta(days=1)).isoformat(),
                            "end": (today + timedelta(days=1)).isoformat(),
                            "basis": basis,
                            "source": "Artificial test",
                        }
                    ),
                ),
            )
    monitor.scan()
    kinds = [r["kind"] for r in monitor.snapshot()["alerts"]]
    assert "BACKORDER" in kinds and "STALLED_ORDER" in kinds
    assert kinds.count("STRIKE") == 1


def test_autonomous_site_has_no_forms_and_preserves_live_privacy(tmp_path):
    with TestClient(
        create_app(
            database_path=tmp_path / "auto.sqlite", seed_path="", autopilot_enabled=True
        )
    ) as c:
        c.app.state.alerts["live"].configure(
            {"owner_email": "private-owner@example.test"}
        )
        for path in ["/", "/operations", "/fulfillment"]:
            page = c.get(path).text
            assert "<form" not in page and "<input" not in page
            assert "autopilot.js" in page
        pilot = c.app.state.autopilot
        for _ in range(6):
            pilot.step()
            c.app.state.fulfillment["demo"].tick()
        c.app.state.alerts["demo"].scan()
        response = c.get("/demo/autopilot")
        assert response.json()["autopilot"]["step"] == 6
        assert len(response.json()["autopilot"]["crisis"]["plans"]) == 8
        assert response.json()["autopilot"]["historical_market_rows"][0].get("forecast")
        assert response.json()["workflow"]["orders"] == []
        assert "private-owner@example.test" not in response.text
        assert c.app.state.fulfillment["live"].snapshot()["inventory"] == []


def test_smtp_transport_requires_tls(monkeypatch):
    from scrc.production.alerts import send_email

    smtp_config(monkeypatch)
    calls = []

    class Transport:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def ehlo(self):
            calls.append("ehlo")

        def starttls(self, **_):
            calls.append("tls")

        def login(self, *_):
            calls.append("login")

        def send_message(self, message):
            calls.append("send")
            assert message["To"] == "owner@example.test"
            return {}

    monkeypatch.setattr("scrc.production.alerts.smtplib.SMTP", Transport)
    send_email(
        "owner@example.test", "Artificial test", "Artificial test", "stable-test-id"
    )
    assert calls == ["ehlo", "tls", "ehlo", "login", "send"]
