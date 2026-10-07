"""Integration checks against the real frozen model and recorded forward outcomes."""

import shutil
from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from scrc.api.app import create_app
from scrc.common import ROOT
from scrc.production.service import ForecastService
from scrc.production.store import Store


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    path = tmp_path_factory.mktemp("api") / "app.sqlite"
    with TestClient(create_app(database_path=path)) as c:
        yield c


def payload():
    return {
        "state": "Uttar Pradesh",
        "market": "Achalda APMC",
        "commodity": "Onion",
        "forecast_date": "2024-01-01",
    }


def test_health_dashboard_and_evidence(client):
    assert client.get("/health/ready").json()["status"] == "ready"
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/v1/catalog").json()["seeded"]
    assert client.get("/v1/model").json()["nominal_coverage"] == 0.95
    assert len(client.get("/v1/benchmark").json()["rows"]) == 4
    assert client.get("/v1/logistics/evidence").json()["automation_allowed"] is False
    m = client.get(
        "/v1/operations/market",
        params={
            "state": "Uttar Pradesh",
            "market": "Achalda APMC",
            "commodity": "Onion",
            "as_of": "2024-03-15",
        },
    )
    assert m.status_code == 200, m.text
    assert m.json()["recent_reports"] > 0
    assert m.json()["last_recorded_date"] <= "2024-03-15"


def test_real_forward_replay_parity(client):
    replay = pd.read_parquet(ROOT / "artifacts/production/test_replay.parquet")
    for index in [0, len(replay) // 2, len(replay) - 1]:
        row = replay.iloc[index]
        body = {k: row[k] for k in ["state", "market", "commodity"]}
        body["forecast_date"] = str(row.date.date())
        response = client.post("/v1/forecast", json=body)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["prediction"] == pytest.approx(row.prediction, abs=1e-8)
        assert result["interval"]["lower"] == pytest.approx(row.lower, abs=1e-8)
        assert result["interval"]["upper"] == pytest.approx(row.upper, abs=1e-8)
        assert result["reliability"]["automation_allowed"] is False
        assert result["reliability"]["causal_gate"] == "RED"
        assert result["reliability"]["status"] == "AMBER"


@pytest.mark.parametrize(
    "change",
    [
        {"market": "Unknown"},
        {"forecast_date": "2023-12-31"},
        {"forecast_date": "2026-10-07"},
        {"extra": "forbidden"},
        {"history": [{"date": "2024-01-01", "arrivals": -1, "modal_price": 10}] * 14},
        {"history": [{"date": "2024-01-01", "arrivals": 1, "modal_price": 10}] * 14},
    ],
)
def test_invalid_or_stale_requests(client, change):
    assert client.post("/v1/forecast", json={**payload(), **change}).status_code == 422


def test_feedback_idempotency_and_persistence(client):
    r = client.post("/v1/forecast", json=payload()).json()
    body = {"forecast_id": r["forecast_id"], "actual": r["actual_if_recorded"]}
    assert client.post("/v1/feedback", json=body).status_code == 200
    assert client.post("/v1/feedback", json=body).json()["idempotent"]
    assert (
        client.post(
            "/v1/feedback", json={**body, "actual": body["actual"] + 1}
        ).status_code
        == 409
    )
    assert client.get("/v1/monitoring").json()["feedback_count"] >= 1
    store = Store(client.app.state.store.path)
    assert store.monitoring()["feedback_count"] >= 1


def test_cloud_auth_and_body_limit(tmp_path):
    with pytest.raises(RuntimeError, match="32 characters"):
        create_app(environment="production", api_key="short")
    key = "a" * 40
    with TestClient(
        create_app(
            database_path=tmp_path / "secure.db",
            seed_path="",
            environment="production",
            api_key=key,
        )
    ) as c:
        assert c.get("/health/ready").status_code == 200
        assert c.get("/v1/catalog").status_code == 401
        r = c.get("/v1/catalog", headers={"X-API-Key": key})
        assert r.status_code == 200
        assert r.headers["cache-control"] == "no-store"
        assert c.get("/docs").status_code == 404
        assert c.post("/v1/forecast", content="x" * 2_000_001).status_code == 413


def test_tampered_model_rejected(tmp_path):
    shutil.copytree(ROOT / "artifacts/production", tmp_path / "model")
    card = tmp_path / "model/model_card.json"
    card.write_text(card.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        ForecastService(tmp_path / "model", Store(tmp_path / "store.db"))


def test_ingest_conflict_is_atomic_and_future_rejected(client):
    p = {**payload(), "date": "2024-01-01", "arrivals": 5.0, "modal_price": 1000.0}
    del p["forecast_date"]
    store = client.app.state.store
    old = store.history(p["state"], p["market"], p["commodity"], date(2024, 1, 2)).iloc[
        -1
    ]
    p["modal_price"] = float(old.modal_price)
    assert (
        client.post("/v1/observations", json={"observations": [p]}).json()["inserted"]
        == 0
    )
    before = sum(r["rows"] for r in store.catalog())
    with pytest.raises(ValueError):
        store.ingest([{**p, "date": "2020-01-01"}, {**p, "arrivals": 6.0}])
    assert sum(r["rows"] for r in store.catalog()) == before
    assert (
        client.post(
            "/v1/observations", json={"observations": [{**p, "date": "2099-01-01"}]}
        ).status_code
        == 422
    )


def test_future_outcomes_cannot_change_prediction(client):
    r = client.post("/v1/forecast", json=payload()).json()
    store = client.app.state.store
    raw = store.history("Uttar Pradesh", "Achalda APMC", "Onion", date(2024, 1, 1))
    history = raw[["date", "arrivals", "modal_price"]].to_dict("records")
    s = client.post("/v1/forecast", json={**payload(), "history": history})
    assert s.status_code == 200, s.text
    assert s.json()["prediction"] == pytest.approx(r["prediction"])
    history.append({"date": "2024-01-01", "arrivals": 99999.0, "modal_price": 1000.0})
    assert (
        client.post("/v1/forecast", json={**payload(), "history": history}).status_code
        == 422
    )
