import { expect, it } from "vitest";
import { driftSeverity } from "../lib/drift";

it("driftSeverity prioritises stale and breaching operational states", () => {
  expect(driftSeverity({ detector_status: "stale", breaching: false })).toBe("critical");
  expect(driftSeverity({ detector_status: "breach", breaching: true })).toBe("critical");
  expect(driftSeverity({ detector_status: "warning", breaching: false })).toBe("warning");
  expect(driftSeverity({ detector_status: "ok", breaching: false })).toBe("healthy");
  expect(driftSeverity(undefined)).toBe("unknown");
});
