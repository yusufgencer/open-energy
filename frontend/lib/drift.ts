import type { DriftStatus } from "./types";

export type OperationalSeverity = "healthy" | "warning" | "critical" | "unknown";

export function driftSeverity(
  drift?: Pick<DriftStatus, "detector_status" | "breaching">
): OperationalSeverity {
  if (!drift) return "unknown";
  const status = drift.detector_status?.toLowerCase();
  if (
    drift.breaching ||
    status === "breach" ||
    status === "drift" ||
    status === "stale" ||
    status === "missing" ||
    status === "nwp_stale" ||
    status === "nwp_missing"
  ) {
    return "critical";
  }
  if (status === "warning" || status === "insufficient_data") return "warning";
  if (status === "ok" || status === "healthy") return "healthy";
  return "unknown";
}
