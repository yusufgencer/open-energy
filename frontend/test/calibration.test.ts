import { expect, it } from "vitest";
import { reliabilityPoints } from "../lib/calibration";
import type { BacktestRow } from "../lib/types";

it("reliabilityPoints estimates empirical p10-p90 consistency within ±0.03", () => {
  const rows: BacktestRow[] = Array.from({ length: 100 }, (_, i) => ({
    valid_time: new Date(2026, 0, 1, i).toISOString(),
    actual: i,
    p10: i < 10 ? i + 1 : i - 1,
    p50: i,
    p90: i < 90 ? i + 1 : i - 1,
  }));

  const points = reliabilityPoints(rows);

  expect(points.find((point) => point.nominal === 0.1)?.observed).toBeCloseTo(0.1, 2);
  expect(points.find((point) => point.nominal === 0.9)?.observed).toBeCloseTo(0.9, 2);
  expect(points.find((point) => point.nominal === 0.8)?.observed).toBeCloseTo(0.8, 2);
  expect(Math.abs((points.find((point) => point.nominal === 0.8)?.observed ?? 0) - 0.8)).toBeLessThanOrEqual(0.03);
});
