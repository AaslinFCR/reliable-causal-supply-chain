"""Warehouse response feasibility and honest scenario boundaries."""

import pytest

from scrc.production.crisis import CrisisPlanner
from scrc.production.fulfillment import Fulfillment
from scrc.production.store import Store


@pytest.fixture
def network(tmp_path):
    engine = Fulfillment(Store(tmp_path / "synthetic.sqlite"), "demo")
    engine.seed()
    return engine, CrisisPlanner(engine)


def total(engine):
    return sum(r["on_hand_tonnes"] for r in engine.snapshot()["inventory"])


def test_responses_conserve_network_stock_and_respect_capacity(network):
    engine, planner = network
    for day in range(1, 41):
        before = total(engine)
        result = planner.advance(day)
        with engine.store.connect() as db:
            served = db.execute(
                "SELECT SUM(served) FROM crisis_demand WHERE step=?", (day,)
            ).fetchone()[0]
            for warehouse in db.execute("SELECT * FROM warehouses"):
                stock = db.execute(
                    "SELECT SUM(tonnes) FROM stock WHERE warehouse_id=?",
                    (warehouse["id"],),
                ).fetchone()[0]
                assert stock <= warehouse["capacity_tonnes"] + 1e-8
            assert all(
                r[0] >= 10 - 1e-8
                for r in db.execute(
                    "SELECT tonnes FROM stock WHERE warehouse_id='DEMO-MAIN'"
                )
            )
        assert total(engine) == pytest.approx(before - served)
        assert all(p["proposed_transfer_tonnes"] >= 0 for p in result["plans"])


def test_route_closures_block_transfers_and_early_warning_is_labelled(network):
    _, planner = network
    for day in range(1, 15):
        result = planner.advance(day)
        if day in {8, 9, 10}:
            assert all(p["proposed_transfer_tonnes"] == 0 for p in result["plans"])
        if day in {12, 13, 14}:
            assert all(
                p["proposed_transfer_tonnes"] == 0
                for p in result["plans"]
                if p["region"] == "North"
            )
        if day == 11:
            assert result["hazards"]["disaster"]["days_until_onset"] == 1
            assert "not a weather forecast" in result["hazards"]["disaster"]["basis"]


def test_main_reservations_are_protected_and_shortfall_is_explicit(network):
    engine, planner = network
    engine.create_order(
        {
            "id": "TEST-RESERVATION",
            "warehouse_id": "DEMO-MAIN",
            "commodity": "Rice",
            "tonnes": 95.0,
            "destination": "Artificial reservation test",
        }
    )
    engine.tick(force=True)
    with engine.store.connect() as db:
        db.execute(
            "UPDATE stock SET tonnes=0 WHERE warehouse_id='DEMO-NORTH' AND commodity='Rice'"
        )
    result = planner.advance(1)
    rice = next(
        p
        for p in result["plans"]
        if p["warehouse_id"] == "DEMO-NORTH" and p["commodity"] == "Rice"
    )
    assert rice["proposed_transfer_tonnes"] == 0
    assert rice["uncovered_shortfall_tonnes"] > 0
    assert "limits prevent full replenishment" in rice["recommendation"]


def test_replayed_day_does_not_consume_inventory_twice(network):
    engine, planner = network
    planner.advance(1)
    first = total(engine)
    planner.advance(1)
    assert total(engine) == first


def test_main_stockout_with_surge_produces_four_gated_options(network):
    engine, planner = network
    with engine.store.connect() as db:
        db.execute("UPDATE stock SET tonnes=0 WHERE warehouse_id='DEMO-MAIN' AND commodity='Rice'")
        db.execute("UPDATE stock SET tonnes=0 WHERE warehouse_id='DEMO-SOUTH' AND commodity='Rice'")
    result = planner.advance(4)
    plan = next(p for p in result["plans"] if p["region"] == "South" and p["commodity"] == "Rice")
    assert plan["main_stock_out"] and plan["demand_surge"]
    assert plan["main_backup_below_reserve"]
    assert len(plan["response_options"]) == 4
    assert plan["decision_gate"] == "NO_RELIABLE_OPTION"
    assert plan["simulated_transfer_tonnes"] == 0
    assert plan["uncovered_shortfall_tonnes"] > 0
