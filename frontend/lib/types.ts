export type PlantKind = "wind" | "solar";
export type PointType = "weather" | "turbine" | "panel_array";

export interface Plant {
  plant_id: string;
  name: string;
  kind: PlantKind;
  capacity_mw: number;
  timezone: string;
}

export interface PlantPoint {
  point_id?: number | null;
  plant_id: string;
  point_type: PointType;
  latitude: number;
  longitude: number;
  hub_height_m?: number | null;
  tilt?: number | null;
  azimuth?: number | null;
}

export interface Job {
  job_id: number;
  kind: string;
  status: string;
  progress: number;
  detail: string | null;
  created_at?: string;
}

export interface TrialResult {
  horizon_hours: number;
  trial_id?: number;
  model_family: string;
  nwp_source: string;
  feature_blocks: string; // JSON-encoded array — parse in UI
  params?: string;        // JSON-encoded dict — parse in UI
  cv_nrmse: number | null;
  test_nrmse: number | null;
  test_nmae: number | null;
  test_bias: number | null;
  test_pinball: number | null;
  test_coverage: number | null;
  test_interval_width: number | null;
  test_crps: number | null;
  test_peak_bias: number | null;
  test_peak_mae: number | null;
  skill_score: number;
  is_champion: boolean;
  // B4-t2: winning spatial/source policy for this horizon (null on legacy runs).
  point_policy?: string | null;
  source_policy?: string | null;
}

export interface ForecastRow {
  plant_id: string;
  point_id: number;
  horizon_hours: number;
  issue_time: string;
  valid_time: string;
  p10: number;
  p50: number;
  p90: number;
  model_id?: number | null;
}

/** One point of the champion's held-out test-window backtest (MW). */
export interface BacktestRow {
  valid_time: string;
  actual: number;
  p50: number;
  p10: number | null;
  p90: number | null;
}

export interface ExperimentRequest {
  plant_id: string;
  point_id: number;
  horizons: number[];
  kind: string;
  capacity_mw: number;
  nwp_sources: string[];
  n_trials: number;
  top_k?: number;
  search_mode?: "legacy" | "strategy";
  preset?: "quick" | "balanced" | "deep" | "research";
}

export interface EpiasStatus {
  connected: boolean;
  username: string | null;
}

export interface EpiasPowerplant {
  id: number;
  name: string | null;
  eic: string | null;
}

export interface EpiasIngestBody {
  power_plant_id?: number;
  plant_name?: string;
  start_date?: string;
  end_date?: string;
  field?: string;
}

// ── B4-t1/t3: weather grid setup + multi-source ingest ─────────────────────────

export interface WeatherGridRequest {
  source: string;              // NWP source (icon_eu, ecmwf_ifs025, gfs_global, …)
  size?: number;               // NxN grid side (default 5 → 25 cells)
  upwind_deg?: number | null;  // wind-origin bearing; grid leans upwind
}

export interface WeatherGridResponse {
  grid_id: string;
  source: string;
  size: number;
  count: number;
  points: PlantPoint[];
}

export interface WeatherIngestBody {
  point_id?: number;           // legacy single-point path
  grid_id?: string;            // grid path
  sources?: string[];          // grid mode: explicit NWP source list (source_policy axis)
  past_days?: number;          // Open-Meteo previous-runs history (≤ ~92 days)
  previous_days?: number;      // lead 0/24/48 → horizons 24 & 48
}

// ── B4-t2/t3: winning point/source policy display ──────────────────────────────

export interface ChampionRow {
  horizon_hours: number;
  point_policy: string | null;
  source_policy: string | null;
  strategy_family: string;
  nrmse: number | null;
  skill_score: number | null;
  baseline_skills: Record<string, number>;
}

export interface PolicyLeaderboardRow {
  point_policy: string;
  source_policy: string;
  trials: number;
  wins: number;
  win_rate: number;
  median_skill_score: number | null;
  median_nrmse: number | null;
  median_pinball: number | null;
}

export interface ScenarioLeaderboardRow {
  asset_kind: PlantKind;
  horizon_hours: number;
  strategy_family: string;
  trials: number;
  wins: number;
  win_rate: number;
  median_skill_score: number | null;
  median_nrmse: number | null;
  median_pinball: number | null;
  last_updated_at?: string;
}

export interface StrategyProfileRow {
  horizon_hours: number;
  strategy_family: string;
  trials: number;
  wins: number;
  median_nrmse: number | null;
  median_skill_score: number | null;
  updated_at?: string;
}

export interface ScenarioRunRow {
  scenario_run_id: number;
  experiment_id: number;
  plant_id: string;
  asset_kind: PlantKind;
  horizon_hours: number;
  strategy_trial_id: number | null;
  strategy_family: string;
  feature_signature: string | null;
  model_signature: string | null;
  train_policy: string | null;
  point_policy: string | null;
  ensemble_policy: string | null;
  quantile_policy: string | null;
  nrmse: number | null;
  nmae: number | null;
  bias: number | null;
  pinball: number | null;
  coverage: number | null;
  crps: number | null;
  peak_bias: number | null;
  peak_mae: number | null;
  skill_score: number | null;
  runtime_seconds: number | null;
  is_champion: boolean;
  created_at?: string;
}
