import type {
  CasePayload,
  RefinementReadiness,
  RouteCandidate,
  Vec3,
} from "./types";

const hashPattern = /^sha256:[a-f0-9]{64}$/;
function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("Missing frozen route record.");
  return value as Record<string, unknown>;
}
function samePoint(value: unknown, expected: number[]): boolean {
  return (
    Array.isArray(value) &&
    value.length === 3 &&
    value.every(
      (n, i) =>
        typeof n === "number" &&
        Number.isFinite(n) &&
        Math.abs(n - expected[i]) < 1e-7,
    )
  );
}
function sameTool(value: unknown, expected: RouteCandidate["tool"]): boolean {
  const candidate = record(value);
  return Object.entries(expected).every(
    ([key, item]) => candidate[key] === item,
  );
}

/** Check the UI selection against the engine's frozen native action geometry. */
export function validateRefinementReadiness(
  result: RefinementReadiness,
  source: Pick<CasePayload, "caseHash" | "frame">,
  route: RouteCandidate,
): RefinementReadiness {
  if (
    result.caseHash !== source.caseHash ||
    route.case_hash !== source.caseHash ||
    result.routeId !== route.route_id
  )
    throw new Error("Readiness belongs to another case or route.");
  if (
    !Number.isInteger(result.legalNonStopActions) ||
    result.legalNonStopActions < 0 ||
    (result.status === "ready"
      ? result.legalNonStopActions < 1
      : result.status !== "no_actionable_moves" ||
        result.legalNonStopActions !== 0)
  )
    throw new Error("Readiness action count is inconsistent.");
  if (
    !Array.isArray(result.reasons) ||
    !result.reasons.every((reason) => typeof reason === "string")
  )
    throw new Error("Readiness reasons are malformed.");
  const binding = record(result.route_binding),
    requested = record(binding.requested_geometry);
  if (
    !hashPattern.test(result.decision_model_hash) ||
    !hashPattern.test(String(binding.binding_hash)) ||
    binding.decision_model_hash !== result.decision_model_hash ||
    binding.mode !== "exact_selected_route" ||
    binding.geometry_frame !== "RAS+" ||
    requested.geometry_frame !== "RAS+"
  )
    throw new Error("Readiness has no valid frozen route binding.");
  if (result.optimizationChoiceScope !== "STOP_or_declared_native_stroke")
    throw new Error("The fixed-stroke optimization scope is missing.");
  const ras = (point: Vec3) =>
    point.map((n, i) => (source.frame === "LPS+" && i < 2 ? -n : n));
  const entry = ras(route.entry_mm),
    target = ras(route.target_mm);
  if (
    !samePoint(requested.selected_entry_mm, entry) ||
    !samePoint(requested.selected_target_mm, target) ||
    !Array.isArray(binding.candidate_entries_mm) ||
    binding.candidate_entries_mm.length !== 1 ||
    !samePoint(binding.candidate_entries_mm[0], entry) ||
    !Array.isArray(binding.candidate_targets_mm) ||
    binding.candidate_targets_mm.length !== 1 ||
    !samePoint(binding.candidate_targets_mm[0], target)
  )
    throw new Error("The native action changed the selected entry or target.");
  for (const geometry of [requested, binding]) {
    if (
      !Array.isArray(geometry.tools) ||
      geometry.tools.length !== 1 ||
      !sameTool(geometry.tools[0], route.tool)
    )
      throw new Error("The native action changed the selected instrument.");
    const access = record(geometry.access);
    if (
      !samePoint(access.center_mm, ras(route.window.center_mm)) ||
      !samePoint(access.normal_inward, ras(route.window.normal_inward)) ||
      access.radius_mm !== route.window.radius_mm ||
      access.window_id !== route.window.window_id
    )
      throw new Error("The native action changed the selected access window.");
  }
  return result;
}

export function readinessReason(reason: string): string {
  const known: Record<string, string> = {
    SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE:
      "The initial shaft intersects remaining tissue.",
    NO_FULLY_CONTAINED_NATIVE_TISSUE:
      "The brush fully contains no removable source cells.",
    NO_LEGAL_NATIVE_NONSTOP_ACTION:
      "No legal initial cutting action is available.",
  };
  return known[reason.toUpperCase()] ?? reason.toLowerCase().replace(/_/g, " ");
}
