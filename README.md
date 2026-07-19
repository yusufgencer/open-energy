# OpenEnergy

Wind and solar production forecasting platform. Combines historical data, weather features, and AutoML to generate quantile forecasts (p10/p50/p90) for energy production.

## Architecture Overview

OpenEnergy is built across five services (SP1–SP5):

- **SP1: Data & Ingestion** — Load production CSVs, fetch weather/NWP data
- **SP2: Feature Engineering** — Extract temporal/spatial features from raw data
- **SP3: AutoML Pipeline** — Hyperparameter search, model selection, training
- **SP4: Serving & Retraining** — FastAPI backend with job queue for experiments
- **SP5: Frontend & UX** — Next.js web app for plant/point management and forecast visualization

## Quick Start

### Backend

Prerequisites: Python 3.10+, `uv` package manager

```bash
cd backend
uv sync
```

**Start the server** (with auto-reload):

```bash
uvicorn "openenergy.api.app:create_app" --factory \
  --host 0.0.0.0 --port 8000 --reload
```

The `--factory` flag tells uvicorn to call `create_app()` to instantiate the app.

**Run tests:**

```bash
cd backend
uv run pytest -q
```

**API Documentation:**
Once running, visit `http://localhost:8000/docs` for interactive Swagger UI.

### Frontend

Prerequisites: Node.js 18+, npm 9+

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Open `http://localhost:3000` in your browser.

**Run tests:**

```bash
cd frontend
npm run test
```

## End-to-End Flow

### 1. Create a Plant
```
POST /plants
{
  "id": "wind-farm-1",
  "name": "North Wind Farm",
  "kind": "wind",
  "capacity_mw": 100.0,
  "latitude": 52.5,
  "longitude": 13.4
}
```

### 2. Add Measurement Points
Add weather stations, turbines, or solar panels on the interactive map:

```
POST /plants/{plant_id}/points
{
  "name": "Turbine A",
  "kind": "turbine",
  "latitude": 52.51,
  "longitude": 13.41
}
```

### 3. Upload Production Data
Upload a CSV with columns: `timestamp`, `production_mwh` (or similar).

```bash
curl -F "file=@production.csv" \
  http://localhost:8000/plants/{plant_id}/production
```

### 4. Run an Experiment
Trigger AutoML. This runs in the background and:
- Fetches historical weather/NWP data for the location
- Extracts features (lag, rolling stats, temporal features)
- Trains multiple models (XGBoost, Random Forest, etc.)
- Selects the best model per forecasting horizon

```json
POST /experiments
{
  "plant_id": "wind-farm-1",
  "point_id": 1,
  "horizons": [24, 48],
  "kind": "wind",
  "capacity_mw": 100.0,
  "nwp_sources": ["icon"],
  "n_trials": 10
}
```

Response:
```json
{
  "job_id": 42
}
```

Poll the job status:
```
GET /jobs/42
```

### 5. View Forecasts
Once the experiment completes, fetch forecasts (p10/p50/p90):

```
GET /plants/{plant_id}/forecasts?horizon=24
```

The frontend displays these as interactive charts.

## Project Structure

```
openenergy/
├── backend/
│   ├── openenergy/
│   │   ├── api/          # FastAPI app (create_app factory)
│   │   ├── assets/       # Plant/point models & repository
│   │   ├── experiments/  # AutoML runner
│   │   ├── features/     # Feature extraction
│   │   ├── ingestion/    # CSV/weather loading
│   │   ├── jobs/         # Job queue & runner
│   │   ├── storage.py    # DuckDB init & connection
│   │   └── config.py     # Settings (data_dir, etc.)
│   ├── tests/            # Unit & integration tests
│   ├── pyproject.toml    # Python dependencies (uv)
│   └── README.md         # Backend documentation
├── frontend/
│   ├── app/              # Next.js App Router pages & layouts
│   ├── components/       # React components (map, charts, forms)
│   ├── lib/              # API client, utilities
│   ├── test/             # Vitest tests
│   ├── public/           # Static assets
│   ├── package.json      # Node dependencies
│   ├── tsconfig.json     # TypeScript config
│   ├── next.config.js    # Next.js configuration
│   └── README.md         # Frontend documentation
└── docs/                 # Architecture & design docs
```

## Configuration

**Backend** (`.env` or `openenergy/config.py`):
- `OPENENERGY_DATA_DIR` — Where to store models, databases (default: `~/.openenergy/data`)

**Frontend** (`.env.local`):
- `NEXT_PUBLIC_API_URL` — Backend API URL (default: `http://localhost:8000`)

## Database

Uses **DuckDB** for all data storage:
- Plant metadata & geometry
- Production time series
- Experiment results & model metadata
- Forecasts (p10/p50/p90)

Database file location: `$OPENENERGY_DATA_DIR/openenergy.duckdb`

## Testing

### Backend
```bash
cd backend && uv run pytest -q
```

### Frontend
```bash
cd frontend && npm run test
```

## Deployment

Both services can run standalone:

1. **Backend** — Serves HTTP API (FastAPI/Uvicorn)
2. **Frontend** — Static Next.js build + server (optional for client-only SSG)

For production, use:
- `uvicorn "openenergy.api.app:create_app" --factory` (or gunicorn)
- `npm run build && npm start` for the frontend

## Contributing

Before submitting a PR:
1. Backend: `uv run pytest -q`
2. Frontend: `npm run test && npm run build`
3. Both: TypeScript strict mode must pass (no `any` without justification)

## License

[Add license info if applicable]
