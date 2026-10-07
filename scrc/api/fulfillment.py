"""Authenticated fulfillment controls and signed ERP order intake."""

import base64
import hashlib
import hmac
import os
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import Field, ValidationError


def install_fulfillment(app, authorize, base):
    class Order(base):
        id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
        warehouse_id: str = Field(min_length=1, max_length=40)
        commodity: Literal["Rice", "Wheat", "Onion", "Potato"]
        tonnes: float = Field(gt=0, le=100000)
        destination: str = Field(min_length=1, max_length=200)

    class Controls(base):
        running: bool | None = None
        generate_orders: bool | None = None
        strike_fraction: float | None = Field(default=None, ge=0, le=1)
        fuel_price: float | None = Field(default=None, gt=0, le=10000)
        distance_km: float | None = Field(default=None, ge=0, le=100000)
        litres_per_100km: float | None = Field(default=None, gt=0, le=200)
        truck_capacity: float | None = Field(default=None, gt=0, le=1000)
        reorder_point: float | None = Field(default=None, ge=0, le=1e6)
        target_stock: float | None = Field(default=None, ge=0, le=1e6)

    class CarrierEvent(base):
        event_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")
        order_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
        status: Literal["DISPATCHED", "IN_TRANSIT", "DELIVERED"]
        tracking_reference: str = Field(min_length=1, max_length=120)
        evidence_reference: str = Field(min_length=1, max_length=200)

    router = APIRouter(prefix="/v1/fulfillment", dependencies=[Depends(authorize)])

    def engine(mode):
        return app.state.fulfillment[mode]

    @router.get("/{mode}/state")
    def state(mode: Literal["live", "demo"]):
        return engine(mode).snapshot()

    @router.post("/{mode}/controls")
    def controls(mode: Literal["live", "demo"], payload: Controls):
        changes = payload.model_dump(exclude_none=True)
        if not changes:
            raise HTTPException(422, "At least one control is required.")
        return engine(mode).configure(changes)

    @router.post("/{mode}/orders")
    def order(mode: Literal["live", "demo"], payload: Order):
        return engine(mode).create_order(payload.model_dump())

    @router.post("/{mode}/orders/{order_id}/cancel")
    def cancel(mode: Literal["live", "demo"], order_id: str):
        return engine(mode).cancel(order_id)

    @router.post("/demo/initialize")
    def seed():
        return engine("demo").seed()

    @router.post("/demo/advance")
    def advance():
        return engine("demo").tick(force=True)

    @router.post("/live/carrier-events")
    def carrier_event(payload: CarrierEvent):
        return engine("live").callback(payload.model_dump())

    @router.post("/live/erpnext-order")
    async def erp_order(request: Request):
        secret = os.environ.get("ERPNEXT_WEBHOOK_SECRET", "")
        if len(secret) < 32:
            raise HTTPException(503, "ERPNext webhook secret is not configured.")
        raw = await request.body()
        expected = base64.b64encode(
            hmac.new(secret.encode(), raw, hashlib.sha256).digest()
        ).decode()
        if not hmac.compare_digest(
            expected, request.headers.get("X-Frappe-Webhook-Signature", "")
        ):
            raise HTTPException(401, "Invalid ERPNext webhook signature.")
        try:
            payload = Order.model_validate_json(raw)
        except ValidationError as error:
            raise HTTPException(
                422, "Webhook body must match the documented normalized order contract."
            ) from error
        return engine("live").create_order(payload.model_dump())

    app.include_router(router)
