"""Operational invariants using explicitly artificial, temporary business records."""

import base64
import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from scrc.api.app import create_app
from scrc.production.fulfillment import Fulfillment


@pytest.fixture
def client(tmp_path):
    with TestClient(
        create_app(database_path=tmp_path / "orders.db", seed_path="")
    ) as c:
        c.post(
            "/v1/operations/warehouses",
            json={
                "id": "TEST",
                "name": "Artificial test warehouse",
                "kind": "main",
                "region": "Test",
                "capacity_tonnes": 100,
                "parent_id": None,
            },
        ).raise_for_status()
        c.post(
            "/v1/operations/movements",
            json={
                "request_id": "opening",
                "warehouse_id": "TEST",
                "kind": "receipt",
                "commodity": "Rice",
                "tonnes": 10,
                "destination_id": None,
                "reference": "ARTIFICIAL TEST DATA",
            },
        ).raise_for_status()
        yield c


def order(c, identifier="ORDER-1", tonnes=6, mode="live", warehouse="TEST"):
    return c.post(
        f"/v1/fulfillment/{mode}/orders",
        json={
            "id": identifier,
            "warehouse_id": warehouse,
            "commodity": "Rice",
            "tonnes": tonnes,
            "destination": "Artificial test destination",
        },
    )


def state(c, mode="live"):
    return c.get(f"/v1/fulfillment/{mode}/state").json()


def event(c, status, event_id=None, tracking="TEST-TRACK"):
    return c.post(
        "/v1/fulfillment/live/carrier-events",
        json={
            "event_id": event_id or status,
            "order_id": "ORDER-1",
            "status": status,
            "tracking_reference": tracking,
            "evidence_reference": "Artificial test confirmation",
        },
    )


def test_reservation_prevents_overselling_and_manual_dispatch(client):
    order(client).raise_for_status()
    order(client, "ORDER-2").raise_for_status()
    client.app.state.fulfillment["live"].tick()
    s = state(client)
    assert s["counts"] == {"BACKORDER": 1, "RESERVED": 1}
    assert s["inventory"][0]["available_tonnes"] == 4
    r = client.post(
        "/v1/operations/movements",
        json={
            "request_id": "conflict",
            "warehouse_id": "TEST",
            "kind": "dispatch",
            "commodity": "Rice",
            "tonnes": 5,
            "destination_id": None,
            "reference": "ARTIFICIAL TEST",
        },
    )
    assert r.status_code == 409
    client.post(
        "/v1/fulfillment/live/orders/ORDER-1/cancel", json={}
    ).raise_for_status()
    client.app.state.fulfillment["live"].tick()
    assert state(client)["counts"] == {"CANCELLED": 1, "RESERVED": 1}


def test_confirmation_sequence_idempotency_and_stock(client):
    order(client).raise_for_status()
    assert order(client).json()["unchanged"] is True
    assert order(client, tonnes=7).status_code == 409
    client.app.state.fulfillment["live"].tick()
    assert event(client, "DELIVERED").status_code == 409
    event(client, "DISPATCHED").raise_for_status()
    assert event(client, "DISPATCHED").json()["unchanged"] is True
    assert event(client, "DISPATCHED", tracking="OTHER").status_code == 409
    assert state(client)["inventory"][0]["on_hand_tonnes"] == 4
    assert event(client, "IN_TRANSIT", tracking="OTHER").status_code == 409
    event(client, "IN_TRANSIT").raise_for_status()
    event(client, "DELIVERED").raise_for_status()
    assert state(client)["counts"] == {"DELIVERED": 1}
    assert state(client)["inventory"][0]["on_hand_tonnes"] == 4
    assert (
        client.post("/v1/fulfillment/live/orders/ORDER-1/cancel", json={}).status_code
        == 409
    )


def test_live_cannot_generate_synthetic_data_and_survives_restart(client):
    assert (
        client.post(
            "/v1/fulfillment/live/controls", json={"generate_orders": True}
        ).status_code
        == 422
    )
    order(client).raise_for_status()
    live = client.app.state.fulfillment["live"]
    live.tick()
    for _ in range(4):
        live.tick()
    assert state(client)["counts"] == {"RESERVED": 1}
    assert Fulfillment(live.store, "live").snapshot()["counts"] == {"RESERVED": 1}


def test_demo_isolation_strike_and_automatic_delivery(client):
    client.post("/v1/fulfillment/demo/initialize", json={}).raise_for_status()
    assert (
        client.post("/v1/fulfillment/demo/initialize", json={}).json()["seeded"]
        is False
    )
    order(client, mode="demo", warehouse="DEMO-NORTH").raise_for_status()
    client.post(
        "/v1/fulfillment/demo/controls", json={"strike_fraction": 1}
    ).raise_for_status()
    for _ in range(4):
        client.post("/v1/fulfillment/demo/advance", json={}).raise_for_status()
    assert state(client, "demo")["counts"] == {"RESERVED": 1}
    client.post(
        "/v1/fulfillment/demo/controls", json={"strike_fraction": 0}
    ).raise_for_status()
    for _ in range(5):
        client.post("/v1/fulfillment/demo/advance", json={}).raise_for_status()
    s = state(client, "demo")
    assert s["counts"] == {"DELIVERED": 1}
    assert s["orders"][0]["fuel_cost_inr"] > 0
    assert sum(r["on_hand_tonnes"] for r in s["inventory"]) == 520 - 6
    assert state(client)["inventory"][0]["on_hand_tonnes"] == 10
    assert state(client)["orders"] == []


def test_signed_erp_order_intake(client, monkeypatch):
    raw = json.dumps(
        {
            "id": "ERP-TEST",
            "warehouse_id": "TEST",
            "commodity": "Rice",
            "tonnes": 2,
            "destination": "Artificial destination",
        }
    ).encode()
    url = "/v1/fulfillment/live/erpnext-order"
    assert client.post(url, content=raw).status_code == 503
    secret = "test-secret-" * 4
    monkeypatch.setenv("ERPNEXT_WEBHOOK_SECRET", secret)
    assert client.post(url, content=raw).status_code == 401
    signature = base64.b64encode(
        hmac.new(secret.encode(), raw, hashlib.sha256).digest()
    ).decode()
    headers = {
        "X-Frappe-Webhook-Signature": signature,
        "Content-Type": "application/json",
    }
    client.post(url, content=raw, headers=headers).raise_for_status()
    assert client.post(url, content=raw, headers=headers).json()["unchanged"] is True
    assert client.post(url, content=raw + b" ", headers=headers).status_code == 401


@pytest.mark.parametrize("quantity", [0, -1, 100001])
def test_invalid_order_quantity(client, quantity):
    assert order(client, tonnes=quantity).status_code == 422
