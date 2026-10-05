import { inverseAffine, rasAffine } from "./coordinates.ts";
import type { Point3 } from "./coordinates.ts";
import type { ViewerInspectionTool, ViewerVolume } from "./contracts.ts";

export const INSPECTION_TOOL_COLOR = "#80b9ef";
export interface InstrumentCapsuleDisplay {
  readonly shaftStart: readonly number[];
  readonly shaftEnd: readonly number[];
  readonly tip: readonly number[];
  readonly shaftRadius: number;
  readonly tipRadius: number;
  readonly color: string;
}
export interface InspectionToolDisplay extends InstrumentCapsuleDisplay {
  readonly identity: string;
  readonly caseHash: string;
  readonly actionId: string;
  readonly toolId: string;
  readonly entry: readonly number[];
  readonly axis: readonly number[];
  readonly unknowns: readonly string[];
}
type RecordValue = Record<string, unknown>;
const HASH = /^sha256:[a-f0-9]{64}$/;
function requireValue(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(`Inspection tool withheld: ${message}`);
}
function record(value: unknown): RecordValue {
  requireValue(value && typeof value === "object" && !Array.isArray(value), "missing report record.");
  return value as RecordValue;
}
function list(value: unknown): unknown[] {
  requireValue(Array.isArray(value), "missing report list.");
  return value;
}
function point(value: unknown): Point3 {
  const coordinates = list(value);
  requireValue(coordinates.length === 3 && coordinates.every((v) => typeof v === "number" && Number.isFinite(v)), "invalid physical coordinates.");
  return [...coordinates] as Point3;
}
function positive(value: unknown): number {
  requireValue(typeof value === "number" && Number.isFinite(value) && value > 0, "invalid instrument dimensions.");
  return value;
}
function same(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}
function validHash(value: unknown): boolean { return typeof value === "string" && HASH.test(value); }

/** Display-integrity gate only. The host authenticates the complete response;
 * this gate joins the selected initial preview to that report and source grid.
 * It grants no trajectory, removal, or independent clinical authority. */
export function validateInspectionTool(volume: ViewerVolume, input: ViewerInspectionTool): InspectionToolDisplay {
  requireValue(input.scope === "unexecuted-native-axis-inspection" && input.caseHash === volume.caseHash &&
    validHash(volume.planningHash) && input.planningHash === volume.planningHash, "stale or unavailable source planning identity.");
  requireValue([input.caseHash, input.planningHash, input.bindingHash, input.inspectionHash].every(validHash), "invalid source or inspection identity.");
  const report = record(input.report), binding = record(report.binding), inventory = record(report.inventory), batch = record(inventory.batch);
  requireValue(report.version === "native-axis-inspection-v1" && report.role === "inspection" && report.status === "ready" &&
    report.inventory_complete === true && inventory.status === "complete" && inventory.terminated === false && batch.unsupported_reason === null,
  "a complete initial inspection is required.");
  requireValue(report.candidate_eligible === false && report.removal_authorized === false && report.clinical_deficit_probability === null &&
    batch.geometry_certified === false && batch.removal_authorized === false, "unsupported execution or clinical authority.");
  const accounting = record(report.accounting);
  requireValue(["gradient_steps", "executed_transitions", "native_commits", "simulated_removed_volume_mm3"].every((key) => accounting[key] === 0), "inspection already reports execution.");
  requireValue(binding.case_hash === input.caseHash && binding.planning_hash === input.planningHash && binding.binding_hash === input.bindingHash &&
    report.inspection_hash === input.inspectionHash && binding.mode === "experimental_axis_columns" && binding.geometry_frame === "RAS+", "inspection binding mismatch.");
  requireValue([binding.native_config_hash, binding.decision_model_hash, binding.proposal_model_hash, binding.proposal_rule_hash, report.initial_cavity_state_hash].every(validHash), "missing model or cavity identity.");
  requireValue(batch.source_hash === input.caseHash && batch.engine_model_hash === binding.native_config_hash &&
    batch.proposal_model_hash === binding.proposal_model_hash && batch.rule_hash === binding.proposal_rule_hash && batch.cavity_state_hash === report.initial_cavity_state_hash,
  "stale initial inventory.");
  requireValue(same(binding.source_shape, volume.shape), "source grid changed.");
  const sourceAffine = rasAffine(volume.affine, volume.frame);
  inverseAffine(sourceAffine);
  requireValue(same(binding.native_affine_ras_mm, sourceAffine), "source physical frame changed.");
  const ledger = list(batch.ledger).map(record), proposals = list(batch.proposals).map(record), attempts = list(inventory.attempts).map(record);
  requireValue(Number.isSafeInteger(batch.slot_count) && batch.slot_count === ledger.length && ledger.length > 0 &&
    attempts.every((attempt) => attempt.status === "complete" && typeof attempt.feasible === "boolean"), "incomplete preview ledger.");
  const actions = list(report.actions).map(record), ids = list(inventory.certified_action_ids);
  requireValue(actions.length === ids.length + 1 && report.legal_non_stop_actions === ids.length && new Set(ids).size === ids.length &&
    actions[0].action_id === "STOP" && actions[0].kind === "stop" && actions[0].geometry === null && actions[0].native_preview === null &&
    actions.slice(1).every((action, index) => action.action_id === ids[index] && action.kind === "native_stroke"), "action membership mismatch.");
  requireValue(input.actionId !== "STOP" && ids.includes(input.actionId), "only an accepted, unexecuted tool preview can be displayed.");
  const action = actions.find((item) => item.action_id === input.actionId)!;
  const geometry = record(action.geometry), preview = record(action.native_preview);
  requireValue(geometry.frame === "RAS+" && preview.scope === "native_engine_preview_only" && preview.independent_history_checked === false &&
    preview.source_hash === input.caseHash && preview.source_state_hash === report.initial_cavity_state_hash && preview.native_config_hash === binding.native_config_hash,
  "preview source or authority mismatch.");
  requireValue(preview.phase === "primary" || preview.phase === "fallback", "invalid preview phase.");
  const matching = attempts.filter((attempt) => attempt.proposal_id === preview.proposal_id && attempt.phase === preview.phase);
  requireValue(matching.length === 1 && matching[0].feasible === true && matching[0].tool_id === geometry.tool_id && matching[0].reason === preview.reason &&
    same(matching[0].entry_mm, geometry.entry_mm) && same(matching[0].tip_mm, geometry.tip_mm), "selected action differs from its accepted preview.");
  const proposal = proposals.filter((item) => item.proposal_id === preview.proposal_id);
  const slot = ledger.filter((item) => item.proposal_id === preview.proposal_id);
  requireValue(proposal.length === 1 && slot.length === 1 && slot[0].reason === "PROPOSED_UNCERTIFIED" &&
    slot[0].tool_id === geometry.tool_id && proposal[0].tool_id === geometry.tool_id && same(proposal[0].entry_mm, geometry.entry_mm) &&
    same(proposal[0][preview.phase === "primary" ? "primary_target_mm" : "fallback_target_mm"], geometry.tip_mm), "selected preview differs from its source-bound proposal.");
  if (preview.phase === "fallback") {
    const primary = attempts.filter((attempt) => attempt.proposal_id === preview.proposal_id && attempt.phase === "primary");
    requireValue(proposal[0].fallback_condition === "primary_preview_rejected" && primary.length === 1 && primary[0].feasible === false,
      "fallback did not follow a rejected primary preview.");
  }
  const catalog = list(binding.tools).map(record), toolIds = catalog.map((tool) => tool.tool_id);
  requireValue(new Set(toolIds).size === toolIds.length && typeof geometry.tool_id === "string", "ambiguous tool catalog.");
  const tool = catalog.find((item) => item.tool_id === geometry.tool_id);
  requireValue(tool, "selected instrument is absent from the bound catalog.");
  const length = positive(tool.working_length_mm), tipLength = positive(tool.tip_length_mm), shaftRadius = positive(tool.shaft_radius_mm), tipRadius = positive(tool.tip_radius_mm);
  requireValue(tipLength < length && typeof tool.parameter_source === "string" && tool.parameter_source.length > 0, "invalid bound instrument.");
  const entry = point(geometry.entry_mm), tip = point(geometry.tip_mm), axis = point(geometry.axis_unit), distance = positive(geometry.insertion_distance_mm);
  const measuredDistance = Math.hypot(...tip.map((value, index) => value - entry[index]));
  requireValue(measuredDistance > 0 && Math.abs(measuredDistance - distance) <= 1e-10 * Math.max(1, distance) &&
    Math.abs(Math.hypot(...axis) - 1) <= 1e-12 && axis.every((value, index) => Math.abs(value - (tip[index] - entry[index]) / measuredDistance) <= 1e-12),
  "tool axis and insertion endpoints disagree.");
  const unknowns = list(report.unknowns), previewUnknowns = list(preview.geometry_unknowns);
  requireValue(unknowns.every((value) => typeof value === "string") && previewUnknowns.every((value) => typeof value === "string" && unknowns.includes(value)), "preview uncertainty was lost.");
  requireValue(typeof preview.native_footprint === "string" && preview.native_footprint.length > 0 &&
    [preview.unexecuted_contained_cell_count, preview.unexecuted_contact_cell_count].every((value) => Number.isSafeInteger(value) && (value as number) >= 0), "invalid unexecuted footprint metadata.");
  const shaftStart = tip.map((value, index) => value - axis[index] * length), shaftEnd = tip.map((value, index) => value - axis[index] * tipLength);
  requireValue([...shaftStart, ...shaftEnd].every(Number.isFinite), "tool envelope exceeds display coordinates.");
  return Object.freeze({ identity: `${input.inspectionHash}:${input.bindingHash}:${input.actionId}`, caseHash: input.caseHash,
    actionId: input.actionId, toolId: geometry.tool_id, entry: Object.freeze(entry), axis: Object.freeze(axis), tip: Object.freeze(tip),
    shaftStart: Object.freeze(shaftStart), shaftEnd: Object.freeze(shaftEnd), shaftRadius, tipRadius, color: INSPECTION_TOOL_COLOR,
    unknowns: Object.freeze([...previewUnknowns] as string[]) });
}

/** A/B data remains intact during inspection; replay clears the unexecuted tool.
 * Used by both mesh visibility and linked MRI capsule uniforms. */
export class InstrumentDisplayState {
  private routes: readonly InstrumentCapsuleDisplay[] = [];
  private inspection: InspectionToolDisplay | null = null;
  private replay = false;
  setRoutes(routes: readonly InstrumentCapsuleDisplay[]): void { this.routes = routes.slice(0, 2); }
  setInspection(volume: ViewerVolume, input: ViewerInspectionTool | null): InspectionToolDisplay | null {
    this.inspection = null; // Invalid replacements must never retain the prior tool.
    if (input && !this.replay) this.inspection = validateInspectionTool(volume, input);
    return this.inspection;
  }
  setReplay(active: boolean): void { this.replay = active; if (active) this.inspection = null; }
  get inspected(): InspectionToolDisplay | null { return this.inspection; }
  get routeVisible(): boolean { return !this.replay && !this.inspection; }
  get displayed(): readonly InstrumentCapsuleDisplay[] { return this.replay ? [] : this.inspection ? [this.inspection] : this.routes; }
}

function selectionIdentity(input: ViewerInspectionTool): string {
  return JSON.stringify([input.caseHash, input.planningHash, input.bindingHash, input.inspectionHash, input.actionId]);
}

/** Suppression survives equivalent prop wrappers. A null selection is the explicit
 * clear boundary before the same action can be selected again. */
export class InspectionSelectionGate {
  private suppressedIdentity: string | null = null;
  clear(input: ViewerInspectionTool | null): boolean {
    const identity = input ? selectionIdentity(input) : null;
    const changed = identity !== this.suppressedIdentity;
    this.suppressedIdentity = identity;
    return changed;
  }
  select(input: ViewerInspectionTool | null, blocked: boolean): ViewerInspectionTool | null {
    if (!input) { this.suppressedIdentity = null; return null; }
    if (blocked) this.clear(input);
    return selectionIdentity(input) === this.suppressedIdentity ? null : input;
  }
}
