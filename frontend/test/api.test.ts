import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  createPlant,
  getForecasts,
  getBacktest,
  getScenarioLeaderboard,
  getScenarioRuns,
  getStrategyProfile,
  listPlants,
  createWeatherGrid,
  ingestWeather,
  getExperimentChampion,
  getPolicyLeaderboard,
  getDrift,
  getFleetHealth,
  listJobs,
  startDriftCheck,
  startRetrain,
  startReforecast,
  startPostprocessorFit,
} from "../lib/api";

beforeEach(() => { (global as any).fetch = vi.fn(); });

it("createPlant POSTs to /plants", async () => {
  (global.fetch as any).mockResolvedValue({ ok: true, json: async () => ({ plant_id: "wf1" }) });
  await createPlant({ plant_id: "wf1", name: "WF", kind: "wind", capacity_mw: 10, timezone: "UTC" });
  const [url, opts] = (global.fetch as any).mock.calls[0];
  expect(url).toContain("/plants");
  expect(opts.method).toBe("POST");
});

it("getForecasts hits forecasts with horizon query", async () => {
  (global.fetch as any).mockResolvedValue({ ok: true, json: async () => [] });
  await getForecasts("wf1", 24);
  expect((global.fetch as any).mock.calls[0][0]).toContain("/plants/wf1/forecasts?horizon=24");
});

it("getBacktest hits backtest with horizon query", async () => {
  (global.fetch as any).mockResolvedValue({ ok: true, json: async () => [] });
  await getBacktest("wf1", 48);
  expect((global.fetch as any).mock.calls[0][0]).toContain("/plants/wf1/backtest?horizon=48");
});

it("throws on non-2xx", async () => {
  (global.fetch as any).mockResolvedValue({ ok: false, status: 400, text: async () => "bad" });
  await expect(createPlant({} as any)).rejects.toThrow();
});

it("listPlants GETs /plants", async () => {
  (global.fetch as any).mockResolvedValue({
    ok: true,
    json: async () => [{ plant_id: "wf1", name: "WF", kind: "wind", capacity_mw: 10, timezone: "UTC" }],
  });
  const plants = await listPlants();
  const [url] = (global.fetch as any).mock.calls[0];
  expect(url).toContain("/plants");
  expect(Array.isArray(plants)).toBe(true);
});

it("getScenarioLeaderboard sends optional filters", async () => {
  (global.fetch as any).mockResolvedValue({ ok: true, json: async () => [] });
  await getScenarioLeaderboard("wind", 24);
  const [url] = (global.fetch as any).mock.calls[0];
  expect(url).toContain("/scenario-memory/leaderboard?kind=wind&horizon=24");
});

it("gets strategy profile and scenario runs for a plant", async () => {
  (global.fetch as any).mockResolvedValue({ ok: true, json: async () => [] });
  await getStrategyProfile("wf1");
  await getScenarioRuns("wf1");
  expect((global.fetch as any).mock.calls[0][0]).toContain("/plants/wf1/strategy-profile");
  expect((global.fetch as any).mock.calls[1][0]).toContain("/plants/wf1/scenario-runs");
});

it("createWeatherGrid POSTs the grid spec to the plant", async () => {
  (global.fetch as any).mockResolvedValue({
    ok: true,
    json: async () => ({ grid_id: "icon_eu_3x3", source: "icon_eu", size: 3, count: 9, points: [] }),
  });
  const res = await createWeatherGrid("wf1", { source: "icon_eu", size: 3, upwind_deg: 270 });
  const [url, opts] = (global.fetch as any).mock.calls[0];
  expect(url).toContain("/plants/wf1/weather-grid");
  expect(opts.method).toBe("POST");
  expect(JSON.parse(opts.body)).toEqual({ source: "icon_eu", size: 3, upwind_deg: 270 });
  expect(res.grid_id).toBe("icon_eu_3x3");
  expect(res.count).toBe(9);
});

it("ingestWeather POSTs grid + sources and returns a job id", async () => {
  (global.fetch as any).mockResolvedValue({ ok: true, json: async () => ({ job_id: 12 }) });
  const res = await ingestWeather("wf1", {
    grid_id: "icon_eu_3x3",
    sources: ["icon_eu", "ecmwf_ifs025"],
    past_days: 30,
  });
  const [url, opts] = (global.fetch as any).mock.calls[0];
  expect(url).toContain("/plants/wf1/ingest-weather");
  expect(opts.method).toBe("POST");
  expect(JSON.parse(opts.body).sources).toEqual(["icon_eu", "ecmwf_ifs025"]);
  expect(res.job_id).toBe(12);
});

it("getExperimentChampion GETs the champion payload", async () => {
  (global.fetch as any).mockResolvedValue({
    ok: true,
    json: async () => [
      {
        horizon_hours: 24,
        point_policy: "spatial_mean",
        source_policy: "icon+gfs",
        strategy_family: "wind_topk_mean",
        nrmse: 0.2,
        skill_score: 0.3,
        baseline_skills: { best: 0.3 },
      },
    ],
  });
  const champs = await getExperimentChampion(5);
  expect((global.fetch as any).mock.calls[0][0]).toContain("/experiments/5/champion");
  expect(champs[0].point_policy).toBe("spatial_mean");
  expect(champs[0].source_policy).toBe("icon+gfs");
  expect(champs[0].baseline_skills.best).toBe(0.3);
});

it("getPolicyLeaderboard scopes to a plant with optional filters", async () => {
  (global.fetch as any).mockResolvedValue({ ok: true, json: async () => [] });
  await getPolicyLeaderboard("wf1", "wind", 24);
  const [url] = (global.fetch as any).mock.calls[0];
  expect(url).toContain("/plants/wf1/policy-leaderboard?kind=wind&horizon=24");
});

it("getPolicyLeaderboard omits filters when not given", async () => {
  (global.fetch as any).mockResolvedValue({ ok: true, json: async () => [] });
  await getPolicyLeaderboard("wf1");
  const [url] = (global.fetch as any).mock.calls[0];
  expect(url).toContain("/plants/wf1/policy-leaderboard");
  expect(url).not.toContain("?");
});

describe("operational API clients", () => {
  it("gets fleet health, plant drift, and the bounded job center", async () => {
    (global.fetch as any).mockResolvedValue({ ok: true, json: async () => [] });

    await getFleetHealth();
    await getDrift("wf1");
    await listJobs(25);

    expect((global.fetch as any).mock.calls[0][0]).toContain("/fleet/health");
    expect((global.fetch as any).mock.calls[1][0]).toContain("/plants/wf1/drift");
    expect((global.fetch as any).mock.calls[2][0]).toContain("/jobs?limit=25");
  });

  it.each([
    ["startDriftCheck", startDriftCheck, "/plants/wf1/drift/check", { point_id: 7, horizons: [24], kind: "wind" }],
    ["startRetrain", startRetrain, "/plants/wf1/retrain", { plant_id: "wf1", point_id: 7, horizons: [24], kind: "wind", capacity_mw: 10 }],
    ["startReforecast", startReforecast, "/plants/wf1/reforecast", { point_id: 7, horizon_hours: 24, kind: "wind" }],
    ["startPostprocessorFit", startPostprocessorFit, "/plants/wf1/postprocessor/fit", { horizons: [24] }],
  ])("%s POSTs its body to the operational endpoint", async (_name, fn, path, body) => {
    (global.fetch as any).mockResolvedValue({ ok: true, json: async () => ({ job_id: 9, status: "pending" }) });

    const result = await fn("wf1", body as never);
    const [url, opts] = (global.fetch as any).mock.calls[0];

    expect(url).toContain(path);
    expect(opts.method).toBe("POST");
    expect(JSON.parse(opts.body)).toEqual(body);
    expect(result.job_id).toBe(9);
  });
});
