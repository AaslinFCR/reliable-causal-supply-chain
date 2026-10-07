# Inventory to delivery: business onboarding

The `/fulfillment` control tower defaults to recorded business operations. A five-second worker allocates available stock, creates reservations and queues shipments. Confirmed dispatch deducts stock once; confirmed transit and delivery events advance tracking. All state, event IDs and reservations persist in SQLite. Order IDs and event IDs are idempotent; conflicting reuse is rejected. Cancellation releases reservations before dispatch. Warehouse withdrawals and transfers cannot consume reserved stock.

**Current connection status:** no ERPNext instance, carrier account or cloud host is connected. Shipment jobs wait for a provider. The application does not yet book vehicles, purchase postage, issue invoices, or collect payments. Real dispatch and delivery require verified confirmations through the form or authenticated event API. This is an integration-ready operational foundation, not an activated carrier service or a complete ERP replacement.

## Selected systems and data sources

| Information | Source | Initial setup |
|---|---|---|
| Warehouse inventory | Physical stock count and your ERPNext stock ledger | Create main and regional warehouses in `/operations`; record verified opening receipts, then actual receipts/transfers/dispatches. Quantities are tonnes. |
| Customer demand | Actual customer sales orders in ERPNext | Start with the order form; configure signed order intake after ERPNext is installed. Orders are different from agricultural market arrivals. |
| Bulk delivery | Delhivery Part Truckload, subject to serviceability and commercial approval | Obtain a business account and the applicable B2B API contract. A provider adapter must map bookings, credentials, errors, retries and status evidence to this application's contracts. |
| Agricultural market context | Government AGMARKNET/data.gov.in published mandi prices | Research history is available locally; it is not a current live feed. Supply new verified daily observations to `/v1/observations`. Check `/docs` for the exact contract. |
| Fuel | PPAC/IndianOil published prices and actual transporter invoices | Record current price and route inputs in the operations scenario. No authenticated live fuel connector is configured. |
| Strikes/disruptions | Carrier notices and verified local operational information | Record the affected region, duration and capacity reduction in a scenario. No universal verified strike feed is connected. |

Sources: [ERPNext stock transactions](https://docs.frappe.io/erpnext/stock-transactions), [ERPNext sales orders](https://docs.frappe.io/erpnext/sales-order), [Delhivery bulk freight](https://www.delhivery.com/services/part-truckload), [Delhivery developer access](https://help.delhivery.com/home/docs/client-developer-portal-1), [AGMARKNET](https://www.data.gov.in/catalog/current-daily-price-various-commodities-various-markets-mandi), [PPAC](https://ppac.gov.in/), [IndianOil prices](https://iocl.com/petrol-diesel-price).

Public research datasets cannot provide your actual opening stock, customer orders or proof of delivery. Do not import sample records as business inventory.

## ERPNext order intake

`POST /v1/fulfillment/live/erpnext-order` accepts one normalized commodity/warehouse allocation per request:

```json
{"id":"ERP-SALESORDER-LINE-UNIQUE-ID","warehouse_id":"YOUR-WAREHOUSE-ID","commodity":"Rice","tonnes":2.5,"destination":"YOUR-CUSTOMER-REFERENCE"}
```

This is a contract example, not a real order. Supported commodities currently are Rice, Wheat, Onion and Potato. Multi-line ERPNext sales orders require an adapter that sends one stable ID per allocated line and converts ERP units to tonnes. It must map ERP warehouse identifiers to the registered warehouse IDs. Native ERPNext Sales Order JSON cannot be sent directly.

Set `ERPNEXT_WEBHOOK_SECRET` to a separately generated random secret of at least 32 characters on the server. Send `X-API-Key` and `X-Frappe-Webhook-Signature`. The signature is base64 HMAC-SHA256 of the exact raw request body, using that secret, following [Frappe webhook signing](https://docs.frappe.io/framework/user/en/guides/integration/webhooks). Never put secrets in Git, browser URLs or customer-visible fields. A custom Frappe integration may normalize sales-order lines before signing; test the integration in staging before connecting actual orders. Stock sync is currently through the authenticated warehouse APIs, not automatic ERP reconciliation.

## Carrier confirmation contract

`POST /v1/fulfillment/live/carrier-events`, with `X-API-Key`:

```json
{"event_id":"UNIQUE-PROVIDER-EVENT-ID","order_id":"ERP-SALESORDER-LINE-UNIQUE-ID","status":"DISPATCHED","tracking_reference":"ACTUAL-WAYBILL","evidence_reference":"ACTUAL-DISPATCH-RECORD"}
```

Accepted sequence: `RESERVED → DISPATCHED → IN_TRANSIT → DELIVERED`. Out-of-order events return 409; the provider adapter must buffer/retry them after earlier events arrive. Reuse an event ID on retries with the same body. The tracking reference must remain unchanged. A trusted adapter must validate the carrier's own authentication/signature and translate events; this endpoint is not a native Delhivery webhook. Retain proof of dispatch/delivery in your business system and refer to it in the event. The form is for recording actual verified confirmations while the adapter is being connected.

## Demonstration and controls

Select **Isolated synthetic demonstration**, initialize it, and start processing. Synthetic orders can be generated every four ticks. Demo replenishment conserves stock and respects warehouse capacity; strikes can block dispatch and delivery. Fuel values are planning estimates. The demo uses a separate `*-demo.sqlite` database and cannot change real inventory. Real operations prohibit synthetic order generation and simulation settings.

## Production activation checklist

1. Deploy ERPNext and complete the warehouse/item/customer master data and physical opening-stock reconciliation.
2. Obtain carrier B2B access, confirm service areas and product requirements, then implement and stage-test the provider adapter. Do not mark delivery from an estimated travel time.
3. Connect the private GitHub repository to the chosen cloud account using `render.yaml`; configure application and webhook secrets and persistent storage. Existing GitHub checks build and test the application; they do not create a hosting account.
4. Back up both databases, test restoration and reconciliation, and restrict network access to trusted operators/integrations. Current API-key access is shared administration, not per-user roles. SQLite requires one application instance; multi-instance production needs a database/queue migration.
5. Validate returns, invoicing, taxes, product lots/expiry, cold-chain needs and customer-data handling for your business before operational launch. These workflows are not implemented in this release.

The operational worker is rule-based. It does not bypass the research causal gate, make purchases, or guarantee forecast accuracy. Existing research evaluation and its limitations remain unchanged.
