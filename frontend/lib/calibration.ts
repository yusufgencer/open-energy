import type { BacktestRow } from "./types";

export interface ReliabilityPoint {
  label: string;
  nominal: number;
  observed: number;
}

/** Empirical quantile and central-interval calibration from held-out rows. */
export function reliabilityPoints(rows: BacktestRow[]): ReliabilityPoint[] {
  const withBands = rows.filter(
    (row): row is BacktestRow & { p10: number; p90: number } =>
      row.p10 != null && row.p90 != null
  );
  if (withBands.length === 0) return [];

  const share = (predicate: (row: (typeof withBands)[number]) => boolean) =>
    withBands.filter(predicate).length / withBands.length;

  return [
    { label: "p10", nominal: 0.1, observed: share((row) => row.actual <= row.p10) },
    { label: "p90", nominal: 0.9, observed: share((row) => row.actual <= row.p90) },
    {
      label: "p10–p90",
      nominal: 0.8,
      observed: share((row) => row.actual >= row.p10 && row.actual <= row.p90),
    },
  ];
}
