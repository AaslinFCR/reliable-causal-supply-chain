"""Warehouse ledger invariants and transparent operational scenario calculations."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from scrc.api.app import create_app
from scrc.production.store import Store


@pytest.fixture
def client(tmp_path):
    with TestClient(
        create_app(database_path=tmp_path / "operations.db", seed_path="")
    ) as c:
        yield c


def setup_network(c):
    main = {
        "id": "MAIN",
        "name": "Main warehouse",
        "kind": "main",
        "region": "Central",
        "capacity_tonnes": 1000,
        "parent_id": None,
    }
    regional = {
        "id": "REG",
        "name": "Regional warehouse",
        "kind": "regional",
        "region": "North",
        "capacity_tonnes": 100,
        "parent_id": "MAIN",
    }
    for w in [main, regional]:
        assert c.post("/v1/operations/warehouses", json=w).status_code == 200
    return main, regional


def move(c, request_id, warehouse="MAIN", kind="receipt", tonnes=100, destination=None):
    return c.post(
        "/v1/operations/movements",
        json={
            "request_id": request_id,
            "warehouse_id": warehouse,
            "kind": kind,
            "commodity": "Rice",
            "tonnes": tonnes,
            "destination_id": destination,
            "reference": "Explicit test fixture; not real warehouse data",
        },
    )


def stock(c):
    return {
        w["id"]: w["total_tonnes"]
        for w in c.get("/v1/operations/inventory").json()["warehouses"]
    }


def plan_body():
    return {
        "warehouse_id": "REG",
        "commodity": "Rice",
        "as_of": "2026-10-07",
        "horizon_days": 7,
        "demand_tonnes_daily": 10,
        "demand_change_pct": 20,
        "strike_days": 2,
        "strike_fraction": 0.5,
        "distance_km": 100,
        "litres_per_100km": 30,
        "fuel_price_inr_litre": 100,
        "baseline_fuel_price": 90,
        "truck_capacity_tonnes": 20,
    }


def test_network_and_persistent_transfer(client):
    main, _ = setup_network(client)
    assert client.post("/v1/operations/warehouses", json=main).json()["unchanged"]
    assert move(client, "opening").status_code == 200
    assert move(client, "opening").json()["unchanged"]
    assert move(client, "opening", tonnes=110).status_code == 409
    assert (
        move(
            client, "transfer", kind="transfer", tonnes=40, destination="REG"
        ).status_code
        == 200
    )
    assert stock(client) == {"MAIN": 60, "REG": 40}
    assert sum(stock(client).values()) == 100
    reopened = Store(client.app.state.store.path)
    with reopened.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM movements").fetchone()[0] == 2


def test_capacity_and_insufficient_stock_rollback(client):
    setup_network(client)
    move(client, "opening", tonnes=200)
    assert (
        move(
            client, "too-large", kind="transfer", tonnes=101, destination="REG"
        ).status_code
        == 409
    )
    assert stock(client) == {"MAIN": 200, "REG": 0}
    assert move(client, "overdraw", kind="dispatch", tonnes=201).status_code == 409
    assert move(client, "overflow", tonnes=801).status_code == 409
    assert move(client, "same", kind="transfer", destination="MAIN").status_code == 422
    assert stock(client) == {"MAIN": 200, "REG": 0}


def test_concurrent_dispatch_cannot_overdraw(client):
    setup_network(client)
    move(client, "opening", tonnes=100)
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(
            pool.map(
                lambda i: move(
                    client, "dispatch-" + str(i), kind="dispatch", tonnes=80
                ),
                range(2),
            )
        )
    assert sorted(r.status_code for r in responses) == [200, 409]
    assert stock(client)["MAIN"] == 20


def test_hierarchy_validation(client):
    main, regional = setup_network(client)
    assert (
        client.post(
            "/v1/operations/warehouses", json={**main, "parent_id": "REG"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/operations/warehouses",
            json={**regional, "id": "INVALID", "parent_id": "REG"},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/operations/warehouses", json={**main, "name": "Changed"}
        ).status_code
        == 409
    )


def test_scenario_calculation_and_no_stock_execution(client):
    setup_network(client)
    move(client, "opening", tonnes=100)
    move(client, "regional", "REG", tonnes=20)
    result = client.post("/v1/operations/plan", json=plan_body())
    assert result.status_code == 200, result.text
    r = result.json()
    assert r["horizon_demand_tonnes"] == 84
    assert r["days_of_cover"] == pytest.approx(20 / 12)
    assert r["suggested_transfer_tonnes"] == 64
    assert r["residual_gap_tonnes"] == 0
    assert r["truck_trips"] == 4
    assert r["round_trip_fuel_litres"] == 240
    assert r["fuel_cost_inr"] == 24000
    assert r["fuel_cost_change_inr"] == 2400
    assert r["automation_allowed"] is False
    saved = client.get("/v1/operations/plans").json()["plans"]
    assert saved[0]["result"]["id"] == r["id"]
    assert saved[0]["inputs"]["demand_change_pct"] == 20
    assert stock(client) == {"MAIN": 100, "REG": 20}


def test_full_strike_blocks_supply_and_zero_demand(client):
    setup_network(client)
    move(client, "opening")
    r = client.post(
        "/v1/operations/plan",
        json={**plan_body(), "strike_days": 7, "strike_fraction": 1},
    ).json()
    assert r["suggested_transfer_tonnes"] == 0
    assert r["residual_gap_tonnes"] == 84
    r = client.post(
        "/v1/operations/plan", json={**plan_body(), "demand_change_pct": -100}
    ).json()
    assert r["days_of_cover"] is None
    assert r["truck_trips"] == 0
    assert (
        client.post(
            "/v1/operations/plan", json={**plan_body(), "strike_days": 8}
        ).status_code
        == 422
    )


@pytest.mark.parametrize(
    "kind,value,crop,status",
    [
        ("strike", 1.1, None, 422),
        ("strike", 0.5, None, 200),
        ("fuel", 0, None, 422),
        ("fuel", 95, None, 200),
        ("demand", 20, None, 422),
        ("demand", 20, "Rice", 200),
    ],
)
def test_signal_units_and_provenance(client, kind, value, crop, status):
    body = {
        "kind": kind,
        "value": value,
        "commodity": crop,
        "start": "2026-10-07",
        "end": "2026-10-09",
        "region": "North",
        "source": "User scenario assumption",
        "basis": "scenario",
    }
    r = client.post("/v1/operations/signals", json=body)
    assert r.status_code == status, r.text
    if status == 200:
        assert (
            client.get("/v1/operations/signals").json()["signals"][0]["basis"]
            == "scenario"
        )


def test_operations_authentication(tmp_path):
    with TestClient(
        create_app(
            database_path=tmp_path / "secure.db",
            seed_path="",
            api_key="x" * 40,
            environment="production",
        )
    ) as c:
        assert c.get("/operations").status_code == 200
        assert c.get("/v1/operations/inventory").status_code == 401
        assert (
            c.get(
                "/v1/operations/inventory", headers={"X-API-Key": "x" * 40}
            ).status_code
            == 200
        )
