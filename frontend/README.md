# OpenEnergy Frontend

Web-based interface for wind/solar production forecasting. Build plants, add measurement points on an interactive map, upload production data, and run AutoML experiments to generate forecasts.

## Prerequisites

- **Node.js** 18 or higher
- **npm** 9 or higher
- Backend API running at `http://localhost:8000`

## Setup

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Create environment configuration:**
   ```bash
   cp .env.local.example .env.local
   ```

3. **Start development server:**
   ```bash
   npm run dev
   ```

   The app will be available at `http://localhost:3000`

## Environment Variables

- `NEXT_PUBLIC_API_URL` — Backend API base URL (default: `http://localhost:8000`)

## Stack

- **Framework:** Next.js 15 with App Router
- **Language:** TypeScript with strict mode
- **Styling:** Tailwind CSS 4
- **UI Components:** React 19
- **Charts:** Recharts 3
- **Maps:** Maplibre GL 5
- **Testing:** Vitest

## Available Scripts

- `npm run dev` — Start development server (with hot reload)
- `npm run build` — Build for production
- `npm run test` — Run unit tests
- `npm run lint` — Run ESLint

## User Flow

1. **Create a Plant** — Use the UI to create a new wind/solar plant with metadata (capacity, location)
2. **Add Measurement Points** — Mark weather stations, turbines, or solar panels on the interactive map
3. **Upload Production Data** — Upload historical CSV with timestamps and production values
4. **Run Experiment** — Trigger AutoML (selects, trains, and tunes models for different forecasting horizons)
5. **View Forecast** — Charts show p10/p50/p90 quantiles for each horizon window (24h, 48h, etc.)

## API Reference

The backend exposes interactive API documentation at `http://localhost:8000/docs` (Swagger UI).

Key endpoints used by this frontend:

- `POST /plants` — Create a new plant
- `GET /plants/{plant_id}` — Retrieve plant and its measurement points
- `POST /plants/{plant_id}/points` — Add a point to a plant
- `POST /plants/{plant_id}/production` — Upload production CSV
- `POST /experiments` — Start an AutoML experiment (background job)
- `GET /jobs/{job_id}` — Poll job status
- `GET /plants/{plant_id}/forecasts` — Fetch forecasts (p10/p50/p90)

## Testing

```bash
npm run test
```

Tests use Vitest with jsdom and focus on API client integration (mocked fetch).

## Troubleshooting

**Backend not responding:**
```bash
# Verify backend is running
curl http://localhost:8000/health
```

**Port 3000 already in use:**
```bash
# Use a different port
npm run dev -- -p 3001
```
