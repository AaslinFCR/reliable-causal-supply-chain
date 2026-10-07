# Local and cloud application

## Warehouse operations and automatic deployment

Open `/operations` for main and regional warehouse inventories. Create a main warehouse first, then regional warehouses linked to its ID. Record opening stock as a receipt with an actual reference. Receipts, dispatches and transfers form a persistent ledger; concurrent transactions cannot overdraw stock or exceed warehouse capacity. Transfers update both warehouses atomically. Repeated request IDs are idempotent, and conflicting reuse is rejected.

Demand, strike and fuel conditions can be entered with a region, effective dates, reference and recorded/scenario label. No live feed is connected. Demand is tonnes/day, fuel INR/litre, and strike intensity a fraction from zero to one. Saved signals are available for review; users explicitly enter the assumptions for each planning run. The system does not silently apply signals to scenarios.

Market fluctuations compare actual AGMARKNET mean prices and arrivals between two seven-calendar-day reporting windows ending at the selected date. Missing reports are not filled. Price is INR/quintal; arrivals remain inflow, not demand.

The planner calculates horizon demand, days of stock cover, gaps, parent transfers capped by available stock and regional free space, truck counts, round-trip fuel cost and change versus an entered baseline. Strike effects are an explicit linear availability assumption, not a learned causal estimate. Stock reflects the current ledger; the plan reference date does not reconstruct historical inventory. Every planning run is saved with inputs and assumptions and can be retrieved from `/v1/operations/plans`. It never executes a movement automatically.

Automatic deployment is configured in `.github/workflows/deploy-checks.yaml` and `render.yaml`: pushes to `main` trigger lint, warehouse/scenario integration tests, a Docker build and a production-mode health/authentication smoke test. Render's `autoDeployTrigger: checksPass` waits for linked branch checks before deploying. These are configuration files, not an activated cloud pipeline: there is no repository remote or hosting account connected on this machine. Linux container checks will run once the project is pushed to GitHub. Historical research/API replay tests additionally require local datasets and are run locally, rather than being represented as dataset-free CI checks.

To activate, provide the GitHub repository URL and connect that repository in the chosen Render account. Configure the Blueprint once and review paid resources before creation. Subsequent changes to the linked `main` branch can deploy automatically after checks pass. Protect the branch and require the `application` check before merging. Provider behavior is documented in [Render's Blueprint reference](https://render.com/docs/blueprint-spec) and [GitHub's Python CI guide](https://docs.github.com/en/actions/tutorials/build-and-test-code/python).

The working application includes a FastAPI API, browser dashboard, frozen commodity-specific LightGBM forecasts, 95% nominal conformal intervals, SQLite observation and audit storage, CSV import, and outcome feedback. It serves the real-data model evaluated in `reports/DEPLOYMENT_EVALUATION.md`.

## Start locally

In PowerShell, open this project directory and run:

```powershell
./start-app.ps1
```

Open http://127.0.0.1:8000. Local API documentation is at `/docs`. The existing environment and downloaded observations are already available on this machine. For another machine, install Python 3.12, create `.venv`, install `requirements-runtime.txt`, and download/rebuild the research datasets before starting. Do not use the unrelated parent project's environment.

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements-runtime.txt
./.venv/Scripts/python.exe -m pip install pytest==8.3.4 httpx==0.28.1
```

Select a market and date during January–March 2024 to replay a historical forecast. Defaults use 15 March 2024. Forecast inputs stop strictly before the requested date. Outcome labels are returned separately and never enter the prediction. Dates after the available history are rejected when inputs are stale. Import recent genuine observations for later dates; older model dates trigger review warnings.

Local data: `data/processed/deployment_observations.csv`; application state: `var/application.sqlite`. Importing duplicate identical observations is safe. Conflicts are rejected; each CSV batch is atomic, while previously completed batches remain committed if a later batch fails.

## Docker locally

Install Docker Desktop, then run:

```powershell
docker compose up --build -d
docker compose logs -f app
```

The browser address remains http://127.0.0.1:8000. Compose mounts the downloaded observations read-only and preserves application state in `app-data`. `docker compose down` preserves the volume; do not add `-v` unless you intend to delete stored outcomes.

Docker is unavailable on the current machine, so the image build has not been executed here. The application itself is tested on Python 3.12 under Windows. Linux image execution and resource limits require verification on the target host.

## Cloud deployment

`render.yaml` defines a single Docker web service, health endpoint, generated API key and persistent 1 GB disk. It requests a paid starter service; review the provider's current price before creating resources. No cloud account, paid service or public deployment was created during this implementation.

1. Place the project in a private Git repository accessible to your Render account. Keep `.env`, `var`, raw data and processed datasets out of Git.
2. In Render, create a Blueprint using this repository. Review the service and disk charges, then deploy.
3. Retrieve `SUPPLYCHAIN_API_KEY` from the service's private environment settings. Enter it into the dashboard's API key field; it stays in page memory. Do not paste it into URLs or commit it.
4. Import the locally downloaded `data/processed/deployment_observations.csv` through the protected dashboard. Raw datasets are deliberately absent from the image and delivery ZIP; redistribution terms are not established.
5. Verify `/health/ready`, generate a historical forecast, record an outcome, restart the service and verify that monitoring persists. Validate disk permissions for UID 10001 on the target host.

The blueprint follows the provider's [Blueprint reference](https://render.com/docs/blueprint-spec) and [persistent disk documentation](https://render.com/docs/disks). You can deploy the same Docker image to a VPS or another container host: set `DEPLOY_ENV=production`, a random API key of at least 32 characters, `PORT`, and a writable persistent `DATABASE_PATH`; terminate HTTPS at the hosting proxy. Production startup refuses a missing or short key.

This SQLite deployment uses one process and one instance. Rate limiting is per directly connected IP, so a shared proxy can share its quota. Multi-instance operation requires a shared database and gateway rate limiting. It does not include tenant accounts, role-based access, external alert delivery or an SLA.

## Operations and validation

`/health/live` and `/health/ready` are public; data endpoints and `/metrics` require the configured API key. Production disables interactive API documentation. Request logs omit bodies and keys. Request bodies are limited to 2 MB; observation batches accept at most 5,000 rows. The dashboard imports batches of 2,000. Model checksums are verified before startup.

Use SQLite's backup API to take a consistent backup while the application runs:

```powershell
./.venv/Scripts/python.exe -c "import sqlite3; a=sqlite3.connect('var/application.sqlite'); b=sqlite3.connect('var/backup.sqlite'); a.backup(b); b.close(); a.close()"
```

Restore into a separate path with the service stopped, set `DATABASE_PATH` to that copy, then verify health and monitoring before switching traffic. Keep backups private. Rotate a cloud key through the hosting environment and restart the service.

Run the checks locally:

```powershell
./.venv/Scripts/python.exe run.py pytest -q
./.venv/Scripts/python.exe run.py ruff check scrc tests scripts run.py
```

API integration tests use the actual local datasets and frozen holdout replay; they need those files. The package contains source and fitted models, not full datasets. Reproduce training with `python run.py scrc.production.train_forward` after the primary research pipeline has built observations. The original `frozen_selection.json` records selection before the first 2024 acquisition; a later rerun is reproducibility evidence, not a new unseen holdout.

## Accuracy and use limits

The independently acquired January–March 2024 holdout has 4,965 observations. The validation-selected deployment model has MAE 31.242 tonnes and WAPE 22.81%, compared with rolling-mean MAE 37.960 and WAPE 27.71%. The same-training log-LightGBM baseline performs better: MAE 29.946 and WAPE 21.86%. No superiority over that baseline is claimed.

The 95% nominal intervals achieve 91.94% observed coverage, below their target. This is explicitly surfaced as an AMBER review reason. It does not establish reliable 95% coverage under future shift. Current data end March 2024; no live feed is configured. Forecasts concern wholesale arrivals, not demand, inventory or stock-outs. Causal automation remains RED because identification assumptions in the separate logistics study are not verified. This is a deployable research application requiring validation before operational reliance.
