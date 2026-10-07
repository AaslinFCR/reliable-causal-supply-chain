# Automatic warehouse crisis monitor

The default website is a read-only, automatic main-to-regional warehouse demonstration. `/`, `/operations` and `/fulfillment` show the same crisis dashboard. No entry forms, API-key input or start buttons are needed. The background worker runs while the application process is running, including when the browser is closed. It resumes its persisted replay cursor after restart.

## Data and simulation boundaries

- Actual historical AGMARKNET observations provide market prices/arrivals and automatic frozen-model historical forecasts. Each forecast uses strictly earlier observations. A 320-row March 2024 excerpt is replayed cyclically; 4,965 January–March 2024 observations provide prior history. These exports are drawn from the previously verified project inputs. They are not a current market feed.
- Historical Delhivery trip aggregates provide separate logistics context. They are not joined to individual market observations or simulated transfers.
- Warehouse stock, regional consumption, supplier receipts, fuel prices, strikes and flood warnings are synthetic. No disaster forecast, verified weather warning or current fuel API is connected.
- Regional demand projections use the last seven synthetic daily consumption records. The ±25% stress band is a stated scenario assumption. It has no claimed probability coverage or measured real-business forecasting accuracy.
- Published research results remain unchanged: 22.81% WAPE and 91.94% empirical coverage for 95% nominal intervals in the separate frozen forward test. Causal interventions remain blocked.

## Automatic response logic

One replay step represents one simulated day, every five seconds. Two regional warehouses consume four commodities. The planner estimates three-day stock pressure and creates eight regional commodity plans. It checks available main stock after reservations, a protected ten-tonne main reserve per commodity, regional capacity, and route closure before simulating any transfer. Shared stock/capacity budgets update within one transaction so plans cannot allocate the same stock twice.

The response shows days of cover before action, projected daily demand, proposed transfer, uncovered shortfall, evidence and a readable recommendation. Unsupported full replenishment remains an explicit shortfall. Responses include replenishment, pre-positioning before a synthetic flood, holding blocked-route dispatches, and consolidating non-urgent loads during a synthetic fuel spike. Real dispatches, purchases and claims of optimal routes or guaranteed cost savings are outside scope. Transport time, perishability and complete freight pricing still require business-specific inputs.

Synthetic supplier receipts occur every twelve steps, subject to main warehouse capacity. Supplier stock is generated explicitly for the demonstration. No purchase is made.

## Demonstration cycle

The scenario repeats every 24 simulated days (about two minutes, excluding processing time):

| Simulated day modulo 24 | Scenario |
|---|---|
| 4–7 | Synthetic Southern regional demand surge |
| 8–10 | Synthetic transport strike; affected dispatches held |
| 10–11 | Synthetic North flood early warning; feasible stock pre-positioning |
| 12–14 | Synthetic North flood closure; transfers to North blocked |
| Every day | Fuel scenario varies against a fixed 95 INR/litre reference |

The flood warning is an injected demonstration scenario, not a learned natural-disaster prediction. In a real deployment, authoritative warnings and verified warehouse/route locations must replace it.

## Manager alerts

An independent five-second monitor creates durable incidents for low stock, backorders, stalled orders, fuel changes, strikes and warehouse crisis plans. It automatically resolves incidents when conditions clear. Incident keys suppress duplicates; recurrence creates a new incident. Owner recipients receive all business incidents; warehouse recipients receive their warehouse's incidents. Acknowledgement stops pending email but does not fix the underlying condition.

Synthetic alerts stay in the app and cannot send external email. Recorded-business email has a persistent outbox with retries, recipient checks and exponential backoff. It uses verified STARTTLS with authenticated SMTP. `SMTP_ACCEPTED` means the server accepted a message, not proof of inbox receipt. Delivery is at least once: a process failure after SMTP acceptance and before database commit can cause a duplicate, using the same Message-ID. Five failed attempts produce a visible FAILED record for operator investigation.

Configure these server environment variables securely, never through a browser URL or Git:

```text
SMTP_HOST=
SMTP_PORT=587
SMTP_USERNAME=
SMTP_PASSWORD=
SMTP_FROM=
```

Then configure owner and warehouse recipients through authenticated `POST /v1/fulfillment/live/alert-settings` and explicitly enable `email_enabled`. Recipients are stored only in the runtime database. Actual SMTP delivery has not been tested against a real provider; automated tests use an artificial transport. The email implementation follows the [Python SMTP documentation](https://docs.python.org/3/library/smtplib.html).

## Deployment

Keep a single application instance with persistent storage. `AUTOPILOT_ENABLED=true` is the normal default. Set it to false only when switching to the existing recorded-business operator interface. A configured production API key still protects every `/v1` endpoint. Public `/demo/autopilot` exposes historical context and isolated synthetic records only, excluding live stock, recipient addresses and credentials.

The private GitHub pipeline checks Python, JavaScript syntax, warehouse/alert/crisis behavior and the production Docker container. Hosting must still be connected to activate a public cloud URL. Real business automation also requires verified inventory/sales feeds, real transport disruptions and carrier confirmations.
