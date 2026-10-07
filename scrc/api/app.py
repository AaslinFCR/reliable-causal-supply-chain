"""Private FastAPI service with persistent audit trails and a local dashboard."""

import asyncio
import copy
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from scrc.api.fulfillment import install_fulfillment
from scrc.api.operations import install_operations
from scrc.common import ROOT
from scrc.production.alerts import Alerts
from scrc.production.autopilot import Autopilot
from scrc.production.fulfillment import Fulfillment
from scrc.production.service import ForecastService
from scrc.production.store import Store

LOG = logging.getLogger("supplychain.api")
STATIC = ROOT / "web"


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, str_strip_whitespace=True
    )


class HistoryPoint(StrictModel):
    date: date
    arrivals: float = Field(ge=0, le=1e9)
    modal_price: float = Field(gt=0, le=1e9)


class ForecastRequest(StrictModel):
    state: str = Field(min_length=1, max_length=120)
    market: str = Field(min_length=1, max_length=120)
    commodity: str = Field(min_length=1, max_length=120)
    forecast_date: date
    history: list[HistoryPoint] | None = Field(
        default=None, min_length=14, max_length=400
    )


class Observation(HistoryPoint):
    state: str = Field(min_length=1, max_length=120)
    market: str = Field(min_length=1, max_length=120)
    commodity: str = Field(min_length=1, max_length=120)


class IngestRequest(StrictModel):
    observations: list[Observation] = Field(min_length=1, max_length=5000)


class FeedbackRequest(StrictModel):
    forecast_id: UUID
    actual: float = Field(ge=0, le=1e9)


def create_app(
    artifact_dir=None,
    database_path=None,
    seed_path=None,
    api_key=None,
    environment=None,
    autopilot_enabled=None,
):
    folder = Path(
        artifact_dir or os.environ.get("MODEL_DIR", ROOT / "artifacts/production")
    )
    db = Path(
        database_path
        or os.environ.get("DATABASE_PATH", ROOT / "var/application.sqlite")
    )
    seed = (
        seed_path
        if seed_path is not None
        else os.environ.get(
            "SEED_CSV", str(ROOT / "data/processed/deployment_observations.csv")
        )
    )
    key = api_key if api_key is not None else os.environ.get("SUPPLYCHAIN_API_KEY", "")
    env = environment or os.environ.get("DEPLOY_ENV", "local")
    automated = (
        autopilot_enabled
        if autopilot_enabled is not None
        else (
            database_path is None
            and os.environ.get("AUTOPILOT_ENABLED", "true").lower() == "true"
        )
    )
    if env == "production" and len(key) < 32:
        raise RuntimeError(
            "Production requires SUPPLYCHAIN_API_KEY with at least 32 characters."
        )
    counters = {"requests": 0, "errors": 0, "seconds": 0.0}
    mutex = threading.Lock()
    limits = defaultdict(deque)

    @asynccontextmanager
    async def lifespan(app):
        app.state.store = Store(db, seed)
        app.state.initialize_operations()
        app.state.service = ForecastService(folder, app.state.store)
        app.state.fulfillment = {
            "live": Fulfillment(app.state.store, "live"),
            "demo": Fulfillment(Store(db.parent / (db.stem + "-demo.sqlite")), "demo"),
        }
        app.state.alerts = {
            mode: Alerts(engine) for mode, engine in app.state.fulfillment.items()
        }
        replay_service = copy.copy(app.state.service)
        replay_service.store = app.state.fulfillment["demo"].store
        app.state.autopilot = Autopilot(
            app.state.fulfillment["demo"],
            enabled=automated,
            forecast_service=replay_service,
        )
        if automated:
            app.state.alerts["demo"].configure({"low_stock_tonnes": 3.0})

        async def alert_worker():
            while True:
                await asyncio.sleep(5)
                for monitor in app.state.alerts.values():
                    try:
                        await asyncio.to_thread(monitor.poll)
                    except (sqlite3.Error, ValueError, KeyError, TypeError) as error:
                        LOG.error("Alert monitor failed: %s", type(error).__name__)
                        with monitor.store.connect() as connection:
                            settings = monitor.settings(connection)
                            settings["last_error"] = type(error).__name__
                            connection.execute(
                                "UPDATE alert_settings SET payload=? WHERE id=1",
                                (json.dumps(settings),),
                            )

        async def worker():
            while True:
                await asyncio.sleep(5)
                if automated:
                    try:
                        await asyncio.to_thread(app.state.autopilot.step)
                    except (
                        sqlite3.Error,
                        ValueError,
                        HTTPException,
                        KeyError,
                        TypeError,
                    ) as error:
                        LOG.error("Dataset replay failed: %s", type(error).__name__)
                for engine in app.state.fulfillment.values():
                    try:
                        await asyncio.to_thread(engine.tick)
                    except (
                        sqlite3.Error,
                        ValueError,
                        HTTPException,
                        KeyError,
                        TypeError,
                    ) as error:
                        LOG.error("Fulfillment worker failed: %s", type(error).__name__)
                        with engine.store.connect() as connection:
                            settings = engine.settings(connection)
                            settings["last_error"] = type(error).__name__
                            connection.execute(
                                "UPDATE wf_settings SET payload=? WHERE id=1",
                                (json.dumps(settings),),
                            )

        task = asyncio.create_task(worker())
        notifications = asyncio.create_task(alert_worker())
        try:
            yield
        finally:
            task.cancel()
            notifications.cancel()
            await asyncio.gather(task, notifications, return_exceptions=True)

    app = FastAPI(
        title="Reliable Supply Chain Decisions",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if env == "local" else None,
        redoc_url=None,
        openapi_url="/openapi.json" if env == "local" else None,
    )

    def authorize(x_api_key: str | None = Header(default=None)):
        if key and not secrets.compare_digest(x_api_key or "", key):
            raise HTTPException(status_code=401, detail="A valid API key is required.")

    @app.middleware("http")
    async def guard(request: Request, call_next):
        request_id = str(uuid4())
        started = time.perf_counter()
        if request.method == "POST":
            length = request.headers.get("content-length")
            if length is None:
                return JSONResponse(
                    {"detail": "Content-Length is required."}, status_code=411
                )
            if not length.isdigit():
                return JSONResponse(
                    {"detail": "Invalid Content-Length."}, status_code=400
                )
            if int(length) > 2_000_000:
                return JSONResponse(
                    {"detail": "Request body exceeds 2 MB."}, status_code=413
                )
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with mutex:
            if len(limits) > 1000:
                for host in list(limits):
                    if not limits[host] or limits[host][-1] < now - 60:
                        del limits[host]
            bucket = limits[client]
            while bucket and bucket[0] < now - 60:
                bucket.popleft()
            limited = request.url.path.startswith("/v1/") and len(bucket) >= 120
            if not limited:
                bucket.append(now)
        if limited:
            return JSONResponse(
                {"detail": "Request limit reached; retry in 60 seconds."},
                status_code=429,
                headers={"Retry-After": "60"},
            )
        response = await call_next(request)
        elapsed = time.perf_counter() - started
        with mutex:
            counters["requests"] += 1
            counters["seconds"] += elapsed
            counters["errors"] += int(response.status_code >= 400)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'"
        )
        if request.url.path.startswith("/v1/"):
            response.headers["Cache-Control"] = "no-store"
        LOG.info(
            json.dumps(
                {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "milliseconds": round(elapsed * 1000, 2),
                }
            )
        )
        return response

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/demo/autopilot")
    def demo_autopilot():
        # Only historical context and isolated synthetic data are public.
        monitor = app.state.alerts["demo"].snapshot()
        return {
            "autopilot": app.state.autopilot.snapshot(),
            "workflow": app.state.fulfillment["demo"].snapshot(),
            "alerts": monitor["alerts"],
            "alert_counts": monitor["counts"],
            "last_alert_scan": monitor["settings"]["last_scan_utc"],
            "email_configured": monitor["smtp_configured"],
            "email_note": "Synthetic alerts are displayed here; external email is disabled in demo mode.",
        }

    @app.get("/health/ready")
    def ready():
        if not hasattr(app.state, "service"):
            raise HTTPException(503, "Model is not loaded.")
        with app.state.store.connect() as connection:
            connection.execute("SELECT 1")
        return {"status": "ready", "model_version": app.state.service.meta["version"]}

    @app.get(
        "/metrics", dependencies=[Depends(authorize)], response_class=PlainTextResponse
    )
    def metrics():
        with mutex:
            values = dict(counters)
        return (
            "\n".join(
                [
                    f"supplychain_requests_total {values['requests']}",
                    f"supplychain_errors_total {values['errors']}",
                    f"supplychain_request_seconds_sum {values['seconds']}",
                ]
            )
            + "\n"
        )

    @app.get("/v1/catalog", dependencies=[Depends(authorize)])
    def catalog():
        available = app.state.store.catalog()
        return {
            "series": available,
            "model_version": app.state.service.meta["version"],
            "data_cutoff": max((r["last_date"] for r in available), default=None),
            "model_data_cutoff": app.state.service.meta["model_data_cutoff"],
            "seeded": bool(available),
        }

    @app.get("/v1/model", dependencies=[Depends(authorize)])
    def model():
        meta = app.state.service.meta
        return {
            k: meta[k]
            for k in [
                "version",
                "routing",
                "nominal_coverage",
                "training_end",
                "calibration_start",
                "calibration_end",
                "test_start",
                "test_end",
                "model_data_cutoff",
                "data_cutoff",
                "selection_rule",
                "scope",
                "test_metrics",
            ]
        }

    @app.get("/v1/benchmark", dependencies=[Depends(authorize)])
    def benchmark():
        table = pd.read_csv(folder / "test_metrics.csv").astype(object)
        return {
            "period": "2024-01-01 to 2024-03-31",
            "rows": table.where(pd.notna(table), None).to_dict("records"),
            "selection": "Rolling validation in 2023; selection and calibration frozen before acquiring the 2024 holdout.",
        }

    @app.post("/v1/forecast", dependencies=[Depends(authorize)])
    def forecast(payload: ForecastRequest):
        try:
            return app.state.service.forecast(
                payload.state,
                payload.market,
                payload.commodity,
                payload.forecast_date,
                (
                    [p.model_dump(mode="json") for p in payload.history]
                    if payload.history is not None
                    else None
                ),
            )
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/v1/observations", dependencies=[Depends(authorize)])
    def ingest(payload: IngestRequest):
        for r in payload.observations:
            if r.date > datetime.now(UTC).date():
                raise HTTPException(422, "Future observations are not allowed.")
            if (
                f"{r.state}|{r.market}|{r.commodity}"
                not in app.state.service.meta["references"]
            ):
                raise HTTPException(
                    422, "An observation series is outside the trained model scope."
                )
        try:
            inserted = app.state.store.ingest(
                [r.model_dump(mode="json") for r in payload.observations]
            )
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        return {
            "received": len(payload.observations),
            "inserted": inserted,
            "unchanged": len(payload.observations) - inserted,
        }

    @app.post("/v1/feedback", dependencies=[Depends(authorize)])
    def feedback(payload: FeedbackRequest):
        try:
            return app.state.store.record_feedback(
                str(payload.forecast_id), payload.actual
            )
        except KeyError as error:
            raise HTTPException(404, "Forecast not found.") from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @app.get("/v1/monitoring", dependencies=[Depends(authorize)])
    def monitoring():
        return app.state.store.monitoring()

    @app.get("/v1/series", dependencies=[Depends(authorize)])
    def series(
        end: date,
        state: str = Query(max_length=120),
        market: str = Query(max_length=120),
        commodity: str = Query(max_length=120),
        days: int = Query(default=60, ge=7, le=120),
    ):
        history = app.state.store.history(
            state, market, commodity, end + timedelta(days=1), days
        )
        return {"observations": history.to_dict("records"), "unit": "metric tonnes"}

    @app.get("/v1/logistics/evidence", dependencies=[Depends(authorize)])
    def logistics():
        p = ROOT / "experiments/causal_diagnostics.json"
        return {
            "automation_allowed": False,
            "gate": "RED",
            "reason": "Unverified pre-dispatch confounders and no causal identification; planner review is required.",
            "diagnostics": json.loads(p.read_text()) if p.exists() else {},
            "scope": "Separate 2018 observational logistics study; not joined to market observations.",
        }

    install_operations(app, authorize, StrictModel)
    install_fulfillment(app, authorize, StrictModel)

    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

        @app.get("/", include_in_schema=False)
        def dashboard():
            return FileResponse(
                STATIC / ("autopilot.html" if automated else "index.html")
            )

        @app.get("/operations", include_in_schema=False)
        def operations_dashboard():
            return FileResponse(
                STATIC / ("autopilot.html" if automated else "operations.html")
            )

        @app.get("/fulfillment", include_in_schema=False)
        def fulfillment_dashboard():
            return FileResponse(
                STATIC / ("autopilot.html" if automated else "fulfillment.html")
            )

    return app


app = create_app()
