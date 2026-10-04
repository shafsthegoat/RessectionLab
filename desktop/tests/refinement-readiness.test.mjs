import assert from "node:assert/strict";
import { test } from "node:test";
import {
  validateRefinementReadiness,
  readinessReason,
} from "../src/refinement-readiness.ts";
const hash = `sha256:${"a".repeat(64)}`;
function fixture(frame = "RAS+") {
  const source = { caseHash: hash, frame };
  const route = {
    case_hash: hash,
    route_id: "native-axis-test",
    entry_mm: [10, 20, 30],
    target_mm: [11, 21, 31],
    tool_id: "native",
    tool: {
      tool_id: "native",
      tip_radius_mm: 2,
      shaft_radius_mm: 1,
      working_length_mm: 80,
      tip_length_mm: 4,
      max_access_angle_deg: 20,
    },
    window: {
      center_mm: [10, 20, 30],
      normal_inward: [0, 1, 0],
      radius_mm: 5,
      window_id: "research",
    },
  };
  const ras = (p) => p.map((n, i) => (frame === "LPS+" && i < 2 ? -n : n));
  const access = {
    ...route.window,
    center_mm: ras(route.window.center_mm),
    normal_inward: ras(route.window.normal_inward),
  };
  const binding = {
    mode: "exact_selected_route",
    geometry_frame: "RAS+",
    binding_hash: hash,
    decision_model_hash: hash,
    requested_geometry: {
      geometry_frame: "RAS+",
      selected_entry_mm: ras(route.entry_mm),
      selected_target_mm: ras(route.target_mm),
      access,
      tools: [route.tool],
    },
    access,
    tools: [route.tool],
    candidate_entries_mm: [ras(route.entry_mm)],
    candidate_targets_mm: [ras(route.target_mm)],
  };
  return {
    source,
    route,
    result: {
      caseHash: hash,
      routeId: route.route_id,
      status: "ready",
      legalNonStopActions: 1,
      reasons: [],
      decision_model_hash: hash,
      route_binding: binding,
      optimizationChoiceScope: "STOP_or_declared_native_stroke",
    },
  };
}
test("preflight accepts only exact selected geometry, with explicit fixed-stroke scope", () => {
  const f = fixture();
  assert.equal(
    validateRefinementReadiness(f.result, f.source, f.route),
    f.result,
  );
  const lps = fixture("LPS+");
  assert.doesNotThrow(() =>
    validateRefinementReadiness(lps.result, lps.source, lps.route),
  );
});
test("unactionable geometry is a valid negative, not readiness to train", () => {
  const f = fixture();
  f.result.status = "no_actionable_moves";
  f.result.legalNonStopActions = 0;
  f.result.reasons = ["SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE"];
  assert.equal(
    validateRefinementReadiness(f.result, f.source, f.route).status,
    "no_actionable_moves",
  );
  assert.match(readinessReason(f.result.reasons[0]), /shaft intersects/);
});
test("stale route/case, contradictory counts and missing scope are rejected", () => {
  for (const mutate of [
    (r) => (r.routeId = "old"),
    (r) => (r.caseHash = "other"),
    (r) => (r.legalNonStopActions = 0),
    (r) => (r.optimizationChoiceScope = undefined),
    (r) => (r.decision_model_hash = "unbound"),
  ]) {
    const f = fixture();
    mutate(f.result);
    assert.throws(() =>
      validateRefinementReadiness(f.result, f.source, f.route),
    );
  }
});
test("entry, target, access and tool substitution are rejected even with same route ID", () => {
  for (const mutate of [
    (b) => b.candidate_entries_mm[0][0]++,
    (b) => b.requested_geometry.selected_target_mm[1]++,
    (b) => (b.tools = [{ ...b.tools[0], shaft_radius_mm: 3 }]),
    (b) => (b.access = { ...b.access, radius_mm: 9 }),
    (b) => (b.requested_geometry.tools = []),
  ]) {
    const f = fixture();
    mutate(f.result.route_binding);
    assert.throws(() =>
      validateRefinementReadiness(f.result, f.source, f.route),
    );
  }
});
