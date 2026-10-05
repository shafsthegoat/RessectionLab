import assert from "node:assert/strict";
import { test } from "node:test";
import {
  neighboringPathsRequest,
  validateNeighboringPaths,
} from "../src/neighboring-paths.ts";
import { digest, neighboringFixture } from "./neighboring-paths-fixture.mjs";
const validate = (f) =>
  validateNeighboringPaths(
    f.result,
    neighboringPathsRequest(f.context, f.choices),
    f.context,
  );

test("neighboring paths require explicit instruments and distinct model/support acknowledgments", () => {
  const { context, choices } = neighboringFixture();
  for (const changed of [
    { ...choices, toolIds: [] },
    { ...choices, neighboringColumns: false },
    { ...choices, estimatedSupport: false },
  ])
    assert.equal(neighboringPathsRequest(context, changed), null);
  assert.equal(
    neighboringPathsRequest({ ...context, support: "blocked" }, choices),
    null,
  );
  assert.equal(neighboringPathsRequest(null, choices), null);
  assert.equal(
    neighboringPathsRequest(context, {
      ...choices,
      toolIds: ["generic_suction"],
    }),
    null,
  );
  const request = neighboringPathsRequest(context, choices);
  assert.equal(request.acknowledgeNeighboringColumns, true);
  assert.equal(request.acknowledgeEstimatedSupport, true);
  assert.equal(request.routePlanningModelHash, "c".repeat(64));
  assert.equal("anchorWindowRas" in request, false);
  request.toolIds.push("mutated");
  assert.deepEqual(choices.toolIds, ["native-fine-aspiration"]);
});

test("complete bounded inventory preserves rejected primary, accepted fallback, rejected and omitted slots", () => {
  const f = neighboringFixture(),
    before = structuredClone(f.result);
  const view = validate(f);
  assert.deepEqual(
    [
      view.total,
      view.passed,
      view.rejected,
      view.omitted,
      view.previewAttempts,
    ],
    [13, 1, 1, 11, 3],
  );
  assert.deepEqual(
    view.rows[0].attempts.map((a) => [a.phase, a.passed]),
    [
      ["primary", false],
      ["fallback", true],
    ],
  );
  assert.deepEqual(view.rows[0].entryMm, [3, 3, -0.5]);
  assert.equal(view.rows[12].entryMm, null);
  assert.deepEqual(view.rows[12].attempts, []);
  view.binding.case_id = "changed copy";
  view.rows[0].entryMm[0] = 99;
  assert.deepEqual(f.result, before);
});

test("display rejects stale case/model/tools, promoted authority and nonzero execution", () => {
  const mutations = [
    (r) => (r.caseHash = digest("4")),
    (r) => (r.planningHash = digest("4")),
    (r) => (r.routeId = "other"),
    (r) => (r.routePlanningModelHash = "4".repeat(64)),
    (r) => (r.requestedToolIds = ["native-wide-aspiration"]),
    (r) => (r.inspection.candidate_eligible = true),
    (r) => (r.inspection.removal_authorized = true),
    (r) => (r.inspection.clinical_deficit_probability = 0.1),
    (r) => (r.inspection.accounting.simulated_removed_volume_mm3 = 2),
    (r) => (r.inspection.accounting.executed_transitions = 1),
    (r) => (r.inspection.binding.functional_evidence_available.motor = true),
  ];
  for (const mutate of mutations) {
    const f = neighboringFixture();
    mutate(f.result);
    assert.throws(() => validate(f));
  }
});

test("window and complete instrument binding reject substitution, with only disclosed float64 normalization", () => {
  for (const mutate of [
    (f) => f.result.anchorWindowRas.center_mm[0]++,
    (f) => f.result.inspection.binding.requested_access_ras.radius_mm++,
    (f) => f.result.inspection.binding.tools[0].shaft_radius_mm++,
    (f) => (f.result.inspection.binding.access.normal_inward[0] = 1e-12),
  ]) {
    const f = neighboringFixture();
    mutate(f);
    assert.throws(() => validate(f));
  }
  const f = neighboringFixture(),
    delta = Number.EPSILON;
  f.result.inspection.binding.requested_access_ras.normal_inward[0] = delta;
  f.result.inspection.binding.access.normal_inward[0] = delta;
  f.result.anchorWindowNormalization.normalMaximumDifference = delta;
  assert.equal(validate(f).total, 13);
  f.result.inspection.binding.source_frame = "LPS+";
  f.result.inspection.binding.access.normal_inward[0] += delta;
  assert.equal(validate(f).total, 13);
  f.result.inspection.binding.source_frame = "RAS+";
  assert.throws(() => validate(f), /access direction/);
});

test("source/model/cavity cross-bindings cannot be replaced even when previews repeat a wrong hash", () => {
  for (const key of [
    "source_hash",
    "engine_model_hash",
    "rule_hash",
    "proposal_model_hash",
    "cavity_state_hash",
  ]) {
    const f = neighboringFixture();
    f.result.inspection.inventory.batch[key] = digest("4");
    if (key === "source_hash")
      f.result.inspection.actions[1].native_preview.source_hash = digest("4");
    assert.throws(() => validate(f), /identities disagree/);
  }
});

test("missing slots, hidden attempts, contradictory dispositions and changed preset fail closed", () => {
  const mutations = [
    (r) => r.inventory.batch.ledger.pop(),
    (r) => r.inventory.attempts.shift(),
    (r) => r.inventory.attempts.push(r.inventory.attempts[0]),
    (r) => r.inventory.batch.counts.NO_REMAINING_TARGET_IN_COLUMN--,
    (r) => (r.inventory.batch.ledger[12].proposal_id = "hidden"),
    (r) => (r.inventory.attempts[0].feasible = true),
    (r) => r.inventory.batch.proposals[0].fallback_target_mm[2]++,
    (r) => (r.inventory_complete = false),
    (r) =>
      (r.inventory.batch.unsupported_reason = "UNSUPPORTED_NONAXIAL_ACCESS"),
    (r) => (r.binding.input_profile = "FEATURE_UNITS"),
    (r) => (r.binding.max_steps = 4),
    (r) => (r.binding.proposal_rule.offsets_source_voxels[1][0] = -4),
    (r) => r.actions[1].geometry.tip_mm[2]++,
    (r) => (r.actions[1].native_preview.independent_history_checked = true),
    (r) => r.unknowns.pop(),
  ];
  for (const mutate of mutations) {
    const f = neighboringFixture();
    mutate(f.result.inspection);
    assert.throws(() => validate(f));
  }
});

test("a genuinely complete all-rejected sample retains its denominator rather than manufacturing success", () => {
  const f = neighboringFixture(),
    r = f.result.inspection;
  r.inventory.attempts[1].feasible = false;
  r.inventory.attempts[1].reason = "SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE";
  r.inventory.certified_action_ids = [];
  r.actions = [r.actions[0]];
  r.legal_non_stop_actions = 0;
  r.status = "no_actionable_moves";
  const view = validate(f);
  assert.deepEqual(
    [view.total, view.passed, view.rejected, view.omitted],
    [13, 0, 2, 11],
  );
});

test("two explicitly selected instruments retain all 26 slots and separate instrument checks on the same thirteen paths", () => {
  const f = neighboringFixture(),
    r = f.result.inspection;
  const wide = {
    ...f.context.toolCatalog[0],
    tool_id: "native-wide-aspiration",
    tip_radius_mm: 2.25,
    shaft_radius_mm: 1.1,
    tip_length_mm: 3,
  };
  f.context.toolCatalog.push(wide);
  f.choices.toolIds = [wide.tool_id, "native-fine-aspiration"];
  r.binding.tools.push(structuredClone(wide));
  f.result.requestedToolIds.push(wide.tool_id);
  const batch = r.inventory.batch;
  batch.ledger = r.binding.proposal_rule.offsets_source_voxels.flatMap(
    (offset, column) =>
      r.binding.tools.map((tool) => ({
        column_index: column,
        offset_source_voxels: offset,
        tool_id: tool.tool_id,
        reason: "NO_REMAINING_TARGET_IN_COLUMN",
        proposal_id: null,
      })),
  );
  batch.slot_count = 26;
  batch.counts = { NO_REMAINING_TARGET_IN_COLUMN: 26 };
  batch.proposals = [];
  r.inventory.attempts = [];
  r.inventory.certified_action_ids = [];
  r.actions = [r.actions[0]];
  r.legal_non_stop_actions = 0;
  r.status = "no_actionable_moves";
  r.proposal_accounting.preview_calls = 0;
  const view = validate(f);
  assert.deepEqual(
    [
      view.total,
      view.passed,
      view.rejected,
      view.omitted,
      view.previewAttempts,
    ],
    [26, 0, 0, 26, 0],
  );
  assert.deepEqual(
    view.rows.slice(0, 4).map((row) => row.pathNumber),
    [1, 1, 2, 2],
  );
});
