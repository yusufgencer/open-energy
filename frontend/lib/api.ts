import type {
  Plant,
  PlantPoint,
  Job,
  TrialResult,
  ForecastRow,
  BacktestRow,
  ExperimentRequest,
  EpiasStatus,
  EpiasPowerplant,
  EpiasIngestBody,
  ScenarioLeaderboardRow,
  StrategyProfileRow,
  ScenarioRunRow,
  WeatherGridRequest,
  WeatherGridResponse,
  WeatherIngestBody,
  ChampionRow,
  PolicyLeaderboardRow,
} from "./types";

const apiBase = (): string =>
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  // Canlı/değişken veri: sunucu-tarafı RSC fetch'lerinin Next.js tarafından
  // cache'lenmesini engelle (aksi halde eski 404/veri servis edilir).
  const res = await fetch(url, { cache: "no-store", ...init });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`HTTP ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export async function createPlant(p: Plant): Promise<Plant> {
  return request<Plant>(`${apiBase()}/plants`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(p),
  });
}

export async function getPlant(
  id: string
): Promise<{ plant: Plant; points: PlantPoint[] }> {
  return request<{ plant: Plant; points: PlantPoint[] }>(
    `${apiBase()}/plants/${id}`
  );
}

export async function addPoint(
  plantId: string,
  pt: PlantPoint
): Promise<PlantPoint> {
  return request<PlantPoint>(`${apiBase()}/plants/${plantId}/points`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(pt),
  });
}

export async function uploadProduction(
  plantId: string,
  file: File
): Promise<{ rows: number }> {
  const fd = new FormData();
  fd.append("file", file);
  return request<{ rows: number }>(
    `${apiBase()}/plants/${plantId}/production`,
    { method: "POST", body: fd }
  );
}

export async function startExperiment(
  body: ExperimentRequest
): Promise<{ job_id: number }> {
  return request<{ job_id: number }>(`${apiBase()}/experiments`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function getJob(id: number): Promise<Job> {
  return request<Job>(`${apiBase()}/jobs/${id}`);
}

export async function getResults(experimentId: number): Promise<TrialResult[]> {
  return request<TrialResult[]>(
    `${apiBase()}/experiments/${experimentId}/results`
  );
}

export async function getExperimentChampion(
  experimentId: number
): Promise<ChampionRow[]> {
  return request<ChampionRow[]>(
    `${apiBase()}/experiments/${experimentId}/champion`
  );
}

// ── Weather grid (B4-t1/t3) ────────────────────────────────────────────────────

export async function createWeatherGrid(
  plantId: string,
  body: WeatherGridRequest
): Promise<WeatherGridResponse> {
  return request<WeatherGridResponse>(
    `${apiBase()}/plants/${plantId}/weather-grid`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }
  );
}

export async function ingestWeather(
  plantId: string,
  body: WeatherIngestBody
): Promise<{ job_id: number }> {
  return request<{ job_id: number }>(
    `${apiBase()}/plants/${plantId}/ingest-weather`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }
  );
}

export async function getForecasts(
  plantId: string,
  horizon: number
): Promise<ForecastRow[]> {
  return request<ForecastRow[]>(
    `${apiBase()}/plants/${plantId}/forecasts?horizon=${horizon}`
  );
}

export async function getBacktest(
  plantId: string,
  horizon: number
): Promise<BacktestRow[]> {
  return request<BacktestRow[]>(
    `${apiBase()}/plants/${plantId}/backtest?horizon=${horizon}`
  );
}

export async function health(): Promise<{ status: string }> {
  return request<{ status: string }>(`${apiBase()}/health`);
}

export async function listPlants(): Promise<Plant[]> {
  return request<Plant[]>(`${apiBase()}/plants`);
}

// ── EPİAŞ entegrasyonu ────────────────────────────────────────────────────────

export async function epiasStatus(): Promise<EpiasStatus> {
  return request<EpiasStatus>(`${apiBase()}/integrations/epias/status`);
}

export async function epiasConnect(
  username: string,
  password: string,
  persist = true
): Promise<EpiasStatus> {
  return request<EpiasStatus>(`${apiBase()}/integrations/epias/connect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password, persist }),
  });
}

export async function epiasDisconnect(): Promise<EpiasStatus> {
  return request<EpiasStatus>(`${apiBase()}/integrations/epias/disconnect`, {
    method: "POST",
  });
}

export async function epiasSearchPlants(q: string): Promise<EpiasPowerplant[]> {
  return request<EpiasPowerplant[]>(
    `${apiBase()}/integrations/epias/powerplants?q=${encodeURIComponent(q)}`
  );
}

export async function ingestEpias(
  plantId: string,
  body: EpiasIngestBody
): Promise<{ job_id: number }> {
  return request<{ job_id: number }>(
    `${apiBase()}/plants/${plantId}/ingest-epias`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }
  );
}

// ── Scenario Memory ──────────────────────────────────────────────────────────

export async function getScenarioLeaderboard(
  kind?: string,
  horizon?: number
): Promise<ScenarioLeaderboardRow[]> {
  const params = new URLSearchParams();
  if (kind) params.set("kind", kind);
  if (horizon != null) params.set("horizon", String(horizon));
  const query = params.toString();
  return request<ScenarioLeaderboardRow[]>(
    `${apiBase()}/scenario-memory/leaderboard${query ? `?${query}` : ""}`
  );
}

export async function getStrategyProfile(
  plantId: string
): Promise<StrategyProfileRow[]> {
  return request<StrategyProfileRow[]>(
    `${apiBase()}/plants/${plantId}/strategy-profile`
  );
}

export async function getScenarioRuns(
  plantId: string
): Promise<ScenarioRunRow[]> {
  return request<ScenarioRunRow[]>(
    `${apiBase()}/plants/${plantId}/scenario-runs`
  );
}

// Per-plant {point_policy × source_policy} wins — the "which spatial/source
// combination won for this plant" memory (B3-t2 / B4-t3).
export async function getPolicyLeaderboard(
  plantId: string,
  kind?: string,
  horizon?: number
): Promise<PolicyLeaderboardRow[]> {
  const params = new URLSearchParams();
  if (kind) params.set("kind", kind);
  if (horizon != null) params.set("horizon", String(horizon));
  const query = params.toString();
  return request<PolicyLeaderboardRow[]>(
    `${apiBase()}/plants/${plantId}/policy-leaderboard${query ? `?${query}` : ""}`
  );
}
