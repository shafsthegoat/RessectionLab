/** Display boundary for the separate, unexecuted neighboring-path inspection. */
import type { InspectAxisPlanningRequest } from "./types";
export type NeighboringPathsRequest = InspectAxisPlanningRequest;
export interface NeighboringToolGeometry {
  tool_id: string;
  tip_radius_mm: number;
  shaft_radius_mm: number;
  working_length_mm: number;
  max_access_angle_deg: number;
  tip_length_mm: number;
  parameter_source: string;
}
export interface NeighboringAccessWindow {
  center_mm: [number, number, number];
  normal_inward: [number, number, number];
  radius_mm: number;
  window_id: string;
}
export interface NeighboringPathsContext {
  caseHash: string;
  planningHash: string;
  routeId: string;
  routePlanningModelHash: string;
  routeLabel: string;
  support: "not_required" | "acknowledgment_required" | "blocked";
  supportReason?: string;
  anchorWindowRas: NeighboringAccessWindow;
  toolCatalog: NeighboringToolGeometry[];
}

export interface NeighboringPathsChoices {
  toolIds: string[];
  neighboringColumns: boolean;
  estimatedSupport: boolean;
}

export const neighboringToolChoices = [
  { id: "native-fine-aspiration", label: "Fine aspiration" },
  { id: "native-wide-aspiration", label: "Wide aspiration" },
] as const;

type Point = [number, number, number];
type RecordValue = Record<string, unknown>;
export interface NeighboringPathAttempt {
  phase: "primary" | "fallback";
  passed: boolean;
  reason: string;
  endpointMm: Point;
}
export interface NeighboringPathRow {
  ordinal: number;
  pathNumber: number;
  toolId: string;
  status: "preview_passed" | "rejected" | "omitted";
  reason: string;
  entryMm: Point | null;
  attempts: NeighboringPathAttempt[];
}
export interface NeighboringPathsView {
  bindingHash: string;
  inspectionHash: string;
  status: "ready" | "no_actionable_moves";
  total: number;
  passed: number;
  rejected: number;
  omitted: number;
  previewAttempts: number;
  rows: NeighboringPathRow[];
  binding: RecordValue;
  unknowns: string[];
}

const hash = /^sha256:[a-f0-9]{64}$/;
const declaredOffsets = [
  [0, 0],
  [-2, 0],
  [2, 0],
  [0, -2],
  [0, 2],
  [-2, -2],
  [-2, 2],
  [2, -2],
  [2, 2],
  [-3, 0],
  [3, 0],
  [0, -3],
  [0, 3],
];
function requireValue(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}
function record(value: unknown, name: string): RecordValue {
  requireValue(
    value && typeof value === "object" && !Array.isArray(value),
    `Missing ${name}.`,
  );
  return value as RecordValue;
}
function list(value: unknown, name: string): unknown[] {
  requireValue(Array.isArray(value), `Missing ${name}.`);
  return value;
}
function text(value: unknown, name: string): string {
  requireValue(
    typeof value === "string" && value.length > 0,
    `Missing ${name}.`,
  );
  return value;
}
function point(value: unknown): Point {
  const array = list(value, "physical coordinates");
  requireValue(
    array.length === 3 &&
      array.every((n) => typeof n === "number" && Number.isFinite(n)),
    "Invalid physical coordinates.",
  );
  return [...array] as Point;
}
function same(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}
function toolsMatch(actual: unknown, expected: string[]): actual is string[] {
  return (
    Array.isArray(actual) &&
    actual.length === expected.length &&
    new Set(actual).size === actual.length &&
    actual.every((id) => typeof id === "string" && expected.includes(id))
  );
}

export function neighboringPathsRequest(
  context: NeighboringPathsContext | null,
  choices: NeighboringPathsChoices,
): NeighboringPathsRequest | null {
  if (
    !context ||
    context.support === "blocked" ||
    !choices.neighboringColumns ||
    !["not_required", "acknowledgment_required"].includes(context.support) ||
    (context.support === "acknowledgment_required" &&
      !choices.estimatedSupport) ||
    ![context.caseHash, context.planningHash].every((value) =>
      hash.test(value),
    ) ||
    !/^[a-f0-9]{64}$/.test(context.routePlanningModelHash) ||
    !context.routeId ||
    !choices.toolIds.length ||
    choices.toolIds.length > 2 ||
    new Set(choices.toolIds).size !== choices.toolIds.length ||
    choices.toolIds.some(
      (id) =>
        !neighboringToolChoices.some((tool) => tool.id === id) ||
        context.toolCatalog.filter((tool) => tool.tool_id === id).length !== 1,
    )
  )
    return null;
  return {
    caseHash: context.caseHash,
    planningHash: context.planningHash,
    routeId: context.routeId,
    routePlanningModelHash: context.routePlanningModelHash,
    toolIds: [...choices.toolIds],
    acknowledgeNeighboringColumns: true,
    acknowledgeEstimatedSupport:
      context.support === "acknowledgment_required" && choices.estimatedSupport,
  };
}

/** Checks display accounting, not geometric safety or independent surgical validity. */
export function validateNeighboringPaths(
  value: unknown,
  expected: NeighboringPathsRequest,
  context: NeighboringPathsContext,
): NeighboringPathsView {
  const response = record(value, "inspection response");
  requireValue(
    response.schemaVersion === 1 &&
      response.accessSource === "selected_route_window_only",
    "Unsupported neighboring-path inspection.",
  );
  for (const key of [
    "caseHash",
    "planningHash",
    "routeId",
    "routePlanningModelHash",
  ] as const)
    requireValue(
      response[key] === expected[key] && expected[key] === context[key],
      "Inspection belongs to another case or route model.",
    );
  requireValue(
    toolsMatch(response.requestedToolIds, expected.toolIds),
    "Inspection uses different instruments.",
  );
  const report = record(response.inspection, "complete inspection");
  const binding = record(report.binding, "inspection binding");
  requireValue(
    report.version === "native-axis-inspection-v1" &&
      report.role === "inspection" &&
      report.inventory_complete === true &&
      binding.mode === "experimental_axis_columns" &&
      binding.geometry_frame === "RAS+",
    "A complete initial inspection is required.",
  );
  requireValue(
    binding.case_hash === expected.caseHash &&
      binding.planning_hash === expected.planningHash &&
      binding.neighboring_columns_acknowledged === true,
    "Inspection source binding is inconsistent.",
  );
  const bindingHash = text(binding.binding_hash, "binding hash"),
    inspectionHash = text(report.inspection_hash, "inspection hash");
  requireValue(
    hash.test(bindingHash) &&
      hash.test(inspectionHash) &&
      (!expected.expectedBindingHash ||
        expected.expectedBindingHash === bindingHash),
    "Stale or invalid inspection binding.",
  );
  requireValue(
    report.candidate_eligible === false &&
      report.removal_authorized === false &&
      report.clinical_deficit_probability === null,
    "Inspection cannot grant route, removal or clinical authority.",
  );
  const accounting = record(report.accounting, "inspection accounting");
  requireValue(
    [
      "gradient_steps",
      "executed_transitions",
      "native_commits",
      "simulated_removed_volume_mm3",
    ].every((key) => accounting[key] === 0),
    "Inspection unexpectedly reports execution or removal.",
  );
  const evidence = record(
    binding.functional_evidence_available,
    "functional evidence status",
  );
  requireValue(
    evidence.motor === false &&
      evidence.language === false &&
      binding.vascular_evidence_status === "unassessed" &&
      binding.population_priors_used === false &&
      binding.world_partitions_created === false &&
      binding.world_role === null,
    "Inspection evidence or evaluation scope is inconsistent.",
  );
  const support = record(binding.source_support, "support assumptions");
  requireValue(
    support.cortical_access_permitted === false &&
      binding.access_status === "hypothetical_unverified_cortical_access" &&
      support.estimated_support_acknowledged ===
        expected.acknowledgeEstimatedSupport,
    "Inspection changed its access or support assumptions.",
  );
  const anchor = record(response.anchorWindowRas, "selected access window"),
    requestedAccess = record(
      binding.requested_access_ras,
      "requested access window",
    );
  for (const key of [
    "center_mm",
    "normal_inward",
    "radius_mm",
    "window_id",
  ] as const)
    requireValue(
      same(anchor[key], context.anchorWindowRas[key]),
      "Inspection substituted the selected access window.",
    );
  for (const key of ["center_mm", "radius_mm", "window_id"])
    requireValue(
      same(requestedAccess[key], anchor[key]),
      "Inspection changed the requested access window.",
    );
  const requestedNormal = point(requestedAccess.normal_inward),
    anchorNormal = point(anchor.normal_inward);
  const normalization = record(
    response.anchorWindowNormalization,
    "anchor direction convention",
  );
  const difference = Math.max(
    ...requestedNormal.map((value, i) => Math.abs(value - anchorNormal[i])),
  );
  requireValue(
    normalization.normalAbsoluteTolerance === 4 * Number.EPSILON &&
      normalization.normalMaximumDifference === difference &&
      difference <= 4 * Number.EPSILON,
    "Inspection changed the selected window direction beyond float64 normalization.",
  );
  const actualAccess = record(binding.access, "actual access window");
  for (const key of ["center_mm", "radius_mm", "window_id"])
    requireValue(
      same(actualAccess[key], anchor[key]),
      "Inspection changed the access window.",
    );
  const actualNormal = point(actualAccess.normal_inward);
  const tolerance = binding.source_frame === "LPS+" ? 4 * Number.EPSILON : 0;
  requireValue(
    (binding.source_frame === "RAS+" || binding.source_frame === "LPS+") &&
      actualNormal.every(
        (value, i) => Math.abs(value - requestedNormal[i]) <= tolerance,
      ),
    "Inspection changed the access direction.",
  );
  const tools = list(binding.tools, "bound tools").map((tool) =>
    record(tool, "bound tool"),
  );
  const toolIds = tools.map((tool) => text(tool.tool_id, "tool identity"));
  requireValue(
    same(toolIds, response.requestedToolIds),
    "Bound instrument order disagrees with the request receipt.",
  );
  for (const tool of tools) {
    const expectedTool = context.toolCatalog.find(
      (item) => item.tool_id === tool.tool_id,
    );
    requireValue(
      expectedTool &&
        Object.entries(expectedTool).every(
          ([key, value]) => tool[key] === value,
        ),
      "Inspection substituted the instrument geometry.",
    );
  }
  const rule = record(binding.proposal_rule, "proposal rule");
  const offsets = list(
    rule.offsets_source_voxels,
    "declared neighboring paths",
  );
  requireValue(
    offsets.length > 0 &&
      offsets.length <= 128 &&
      offsets.every(
        (offset) =>
          Array.isArray(offset) &&
          offset.length === 2 &&
          offset.every(Number.isSafeInteger),
      ),
    "Invalid neighboring-path declaration.",
  );
  requireValue(
    new Set(offsets.map((offset) => JSON.stringify(offset))).size ===
      offsets.length,
    "Duplicate neighboring paths.",
  );
  requireValue(
    same(offsets, declaredOffsets) &&
      rule.max_primary_rays === 26 &&
      rule.version === "experimental-residual-axis-columns-v1" &&
      binding.input_profile === "RAW" &&
      binding.max_steps === 3 &&
      binding.max_actions === 27 &&
      binding.partial_contact_weight === 0.05 &&
      binding.max_tip_step_mm === 0.25 &&
      binding.fallback_policy === "only_after_primary_preview_rejection" &&
      binding.ordering === "STOP_then_provider_column_tool_order",
    "Inspection differs from the declared neighboring-path preset.",
  );
  const inventory = record(report.inventory, "initial inventory");
  const batch = record(inventory.batch, "proposal inventory");
  const ledger = list(batch.ledger, "complete path ledger");
  requireValue(
    inventory.status === "complete" &&
      inventory.terminated === false &&
      batch.unsupported_reason === null &&
      batch.geometry_certified === false &&
      batch.removal_authorized === false,
    "Partial or unsupported inventory withheld.",
  );
  requireValue(
    batch.slot_count === offsets.length * tools.length &&
      ledger.length === batch.slot_count,
    "Initial path denominator is incomplete.",
  );
  requireValue(
    batch.source_hash === expected.caseHash &&
      batch.engine_model_hash === binding.native_config_hash &&
      batch.rule_hash === binding.proposal_rule_hash &&
      batch.proposal_model_hash === binding.proposal_model_hash &&
      batch.cavity_state_hash === report.initial_cavity_state_hash,
    "Inspection source, model or initial cavity identities disagree.",
  );
  requireValue(
    [
      batch.source_hash,
      batch.engine_model_hash,
      batch.rule_hash,
      batch.proposal_model_hash,
      batch.cavity_state_hash,
    ].every((value) => typeof value === "string" && hash.test(value)),
    "Inspection source and model fingerprints are missing.",
  );
  const unknowns = list(report.unknowns, "unassessed evidence");
  requireValue(
    unknowns.every((item) => typeof item === "string") &&
      [
        "motor_function_unassessed",
        "language_function_unassessed",
        "vascular_anatomy_unassessed",
        "cortical_access_unverified",
      ].every((item) => unknowns.includes(item)),
    "Missing evidence must remain unassessed.",
  );
  const proposals = list(batch.proposals, "proposed endpoints").map((item) =>
    record(item, "proposal"),
  );
  requireValue(
    new Set(
      proposals.map((proposal) =>
        text(proposal.proposal_id, "proposal identity"),
      ),
    ).size === proposals.length,
    "Duplicate proposal identities.",
  );
  const attempts = list(inventory.attempts, "preview attempts").map((item) =>
    record(item, "attempt"),
  );
  const rows: NeighboringPathRow[] = [],
    accepted: RecordValue[] = [];
  const reasonCounts: Record<string, number> = {};
  let proposalIndex = 0,
    attemptIndex = 0;
  for (const [index, item] of ledger.entries()) {
    const slot = record(item, "path slot");
    const column = Math.floor(index / tools.length),
      toolId = toolIds[index % tools.length];
    const reason = text(slot.reason, "path disposition");
    requireValue(
      slot.column_index === column &&
        slot.tool_id === toolId &&
        same(slot.offset_source_voxels, offsets[column]),
      "Initial path ordering or tool identity changed.",
    );
    reasonCounts[reason] = (reasonCounts[reason] ?? 0) + 1;
    const row: NeighboringPathRow = {
      ordinal: index + 1,
      pathNumber: column + 1,
      toolId,
      status: "omitted",
      reason,
      entryMm: null,
      attempts: [],
    };
    if (slot.proposal_id !== null) {
      requireValue(
        reason === "PROPOSED_UNCERTIFIED",
        "A proposed path has an omitted disposition.",
      );
      const proposal = proposals[proposalIndex++];
      requireValue(
        proposal &&
          proposal.proposal_id === slot.proposal_id &&
          proposal.column_index === column &&
          proposal.tool_id === toolId &&
          same(proposal.offset_source_voxels, offsets[column]),
        "Proposed path is missing from the complete ledger.",
      );
      requireValue(
        proposal.fallback_condition === "primary_preview_rejected",
        "Fallback may only follow a rejected primary endpoint.",
      );
      row.entryMm = point(proposal.entry_mm);
      const endpoints = [point(proposal.primary_target_mm)];
      if (proposal.fallback_target_mm !== null)
        endpoints.push(point(proposal.fallback_target_mm));
      for (const [phaseIndex, endpoint] of endpoints.entries()) {
        const attempt = attempts[attemptIndex++],
          phase = phaseIndex === 0 ? "primary" : "fallback";
        requireValue(
          attempt &&
            attempt.status === "complete" &&
            typeof attempt.feasible === "boolean" &&
            attempt.proposal_id === slot.proposal_id &&
            attempt.phase === phase &&
            attempt.tool_id === toolId &&
            same(point(attempt.entry_mm), row.entryMm) &&
            same(point(attempt.tip_mm), endpoint),
          "Preview ledger is incomplete or changed geometry.",
        );
        row.attempts.push({
          phase,
          passed: attempt.feasible,
          reason: text(attempt.reason, "preview reason"),
          endpointMm: endpoint,
        });
        if (attempt.feasible) {
          accepted.push(attempt);
          break;
        }
      }
      row.status = row.attempts.at(-1)!.passed ? "preview_passed" : "rejected";
      row.reason = row.attempts.at(-1)!.reason;
    } else
      requireValue(
        reason !== "PROPOSED_UNCERTIFIED",
        "Proposed path has no endpoint identity.",
      );
    rows.push(row);
  }
  requireValue(
    proposalIndex === proposals.length && attemptIndex === attempts.length,
    "Hidden or extra preview records were withheld.",
  );
  const counts = record(batch.counts, "proposal disposition counts");
  requireValue(
    Object.keys(counts).length === Object.keys(reasonCounts).length &&
      Object.entries(reasonCounts).every(
        ([reason, count]) => counts[reason] === count,
      ),
    "Omitted path counts do not reconcile.",
  );
  const actions = list(report.actions, "unexecuted actions").map((item) =>
    record(item, "action"),
  );
  const ids = list(inventory.certified_action_ids, "preview action identities");
  requireValue(
    actions.length === accepted.length + 1 &&
      ids.length === accepted.length &&
      report.legal_non_stop_actions === accepted.length,
    "Preview action counts do not reconcile.",
  );
  requireValue(
    actions[0].action_id === "STOP" &&
      actions[0].kind === "stop" &&
      actions[0].geometry === null &&
      actions[0].native_preview === null,
    "Missing unexecuted STOP record.",
  );
  const actionIds = new Set<unknown>();
  for (const [index, attempt] of accepted.entries()) {
    const action = actions[index + 1],
      geometry = record(action.geometry, "preview geometry"),
      preview = record(action.native_preview, "native preview scope");
    requireValue(
      action.kind === "native_stroke" &&
        action.action_id === ids[index] &&
        typeof action.action_id === "string" &&
        !actionIds.has(action.action_id),
      "Preview action identities disagree.",
    );
    actionIds.add(action.action_id);
    requireValue(
      geometry.frame === "RAS+" &&
        geometry.tool_id === attempt.tool_id &&
        same(point(geometry.entry_mm), attempt.entry_mm) &&
        same(point(geometry.tip_mm), attempt.tip_mm),
      "Preview action changed its entry, endpoint or instrument.",
    );
    requireValue(
      preview.scope === "native_engine_preview_only" &&
        preview.independent_history_checked === false &&
        preview.proposal_id === attempt.proposal_id &&
        preview.phase === attempt.phase &&
        preview.reason === attempt.reason &&
        preview.source_hash === batch.source_hash &&
        preview.source_state_hash === batch.cavity_state_hash &&
        preview.native_config_hash === binding.native_config_hash,
      "Preview has unsupported certification or stale source geometry.",
    );
    requireValue(
      list(preview.geometry_unknowns, "preview uncertainty").every(
        (item) => typeof item === "string" && unknowns.includes(item),
      ),
      "A preview uncertainty is missing from the inspection summary.",
    );
    for (const key of [
      "unexecuted_contained_cell_count",
      "unexecuted_contact_cell_count",
    ])
      requireValue(
        Number.isSafeInteger(preview[key]) && (preview[key] as number) >= 0,
        "Invalid unexecuted preview cell count.",
      );
  }
  requireValue(
    report.status === (accepted.length ? "ready" : "no_actionable_moves"),
    "Inspection status disagrees with the complete inventory.",
  );
  requireValue(
    record(report.proposal_accounting, "preview accounting").preview_calls ===
      attempts.length,
    "Preview-check denominator is inconsistent.",
  );
  return {
    bindingHash,
    inspectionHash,
    status: report.status as NeighboringPathsView["status"],
    total: rows.length,
    passed: accepted.length,
    rejected: rows.filter((row) => row.status === "rejected").length,
    omitted: rows.filter((row) => row.status === "omitted").length,
    previewAttempts: attempts.length,
    rows,
    binding: structuredClone(binding),
    unknowns: [...unknowns] as string[],
  };
}

export function neighboringPathReason(reason: string): string {
  const descriptions: Record<string, string> = {
    COLUMN_OUT_OF_IMAGE: "Neighboring path is outside the source image.",
    NO_REMAINING_TARGET_IN_COLUMN:
      "No target annotation lies on this neighboring path.",
    FULL_TOOL_APERTURE_PREFILTER:
      "The complete instrument does not fit the declared access opening.",
    PRIMARY_CANDIDATE_CAP: "Omitted by the declared candidate limit.",
    SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE:
      "The instrument shaft intersects remaining tissue.",
    NO_NEW_FULLY_CONTAINED_SURFACE_CELLS:
      "No new connected source cells fit fully within the modeled brush.",
    NATIVE_CONNECTED_STROKE: "Passed the initial modeled geometry preview.",
    MICROSTEP_BUDGET: "The declared geometry-check limit was reached.",
    UNSUPPORTED_NONAXIAL_ACCESS:
      "This access window is not aligned with the source image axes. Choose another research window; this window will not be changed automatically.",
    UNSUPPORTED_FRACTIONAL_TRANSVERSE_ORIGIN:
      "This access window does not start on a supported source-grid position. Choose another research window; it will not be moved automatically.",
  };
  return descriptions[reason] ?? reason.toLowerCase().replace(/_/g, " ");
}
