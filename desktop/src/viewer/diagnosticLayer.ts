/** Display-only Case4 diagnostic layer admission; never creates planning data. */
import type { UnreviewedDiagnosticLayerView } from "../scan-diagnostic-layer";
import { inverseAffine, rasAffine, transformPoint } from "./coordinates.ts";
import type { Affine, Point3 } from "./coordinates";
import type { ViewerVolume } from "./contracts";

export interface LoadedDiagnosticLayer extends UnreviewedDiagnosticLayerView {
  /** `stateXYZ` and `coverageXYZ` must be decoded into XYZ C order
   * (flat index `(x * Y + y) * Z + z`), not raw NIfTI on-disk order. */
  /** Computed by the local loader over the exact saved NIfTI bytes. */
  loadedStateSha256: string;
  loadedCoverageSha256: string;
  /** Actual authoritative sforms and shapes parsed independently from each file. */
  stateGrid: { shape: [number, number, number]; affineRAS: Affine; sformCode: number };
  coverageGrid: { shape: [number, number, number]; affineRAS: Affine; sformCode: number };
}

export interface CheckedDiagnosticLayer {
  layer: LoadedDiagnosticLayer;
  worldToVoxel: Affine;
  shape: [number, number, number];
  /** Routes/replay/planning remain disabled while inspecting this layer. */
  primaryOverlaysPermitted: false;
  planningInteractionPermitted: false;
}

const HEX64 = /^[0-9a-f]{64}$/;

function sameShape(a: readonly number[], b: readonly number[]): boolean {
  return a.length === 3 && b.length === 3 && a.every((n, i) => n === b[i]);
}

function sameAffine(a: Affine, b: Affine): boolean {
  return a.every((row, i) => row.every((n, j) =>
    Number.isFinite(n) && Number.isFinite(b[i][j]) && Math.abs(n - b[i][j]) <= 1e-5));
}

/** The host supplies a SHA over the *selected displayed atlas T1c file*.
 * A native T1, FLAIR or unregistered scan cannot borrow this overlay. */
export function validateDiagnosticLayer(
  layer: LoadedDiagnosticLayer,
  volume: ViewerVolume,
  selectedSourceSha256: string,
): CheckedDiagnosticLayer {
  const d = layer.descriptor;
  if (d.schema !== "case4_unreviewed_diagnostic_display_layer_v1" ||
      d.scope !== "display_only_atlas_native_grid" ||
      d.output_origin !== "model_generated_from_observed_case4_preoperative_scans" ||
      d.model_output_verified !== true || d.same_grid_only !== true ||
      d.anatomical_qc !== "unreviewed_inferior_mask_omission" ||
      d.training_overlap_status !== "unknown" ||
      d.planning_eligible !== false || d.evaluation_eligible !== false ||
      d.clinical_evidence !== false ||
      layer.primaryOverlaysPermitted !== false ||
      layer.planningInteractionPermitted !== false ||
      !HEX64.test(selectedSourceSha256) ||
      selectedSourceSha256 !== d.source_sha256.t1c ||
      !HEX64.test(d.geometry_sha256) ||
      !HEX64.test(d.input_receipt_sha256) ||
      !HEX64.test(d.forward_receipt_sha256) ||
      !HEX64.test(d.outputs.state.sha256) ||
      !HEX64.test(d.outputs.coverage.sha256) ||
      layer.loadedStateSha256 !== d.outputs.state.sha256 ||
      layer.loadedCoverageSha256 !== d.outputs.coverage.sha256 ||
      volume.compartments.length !== 0 ||
      d.outputs.state.dtype !== "int8" || d.outputs.coverage.dtype !== "uint8" ||
      layer.stateGrid.sformCode === 0 || layer.coverageGrid.sformCode === 0 ||
      !sameShape(d.shape_xyz, volume.shape) ||
      !sameShape(d.shape_xyz, layer.stateGrid.shape) ||
      !sameShape(d.shape_xyz, layer.coverageGrid.shape)) {
    throw new Error("Diagnostic layer source, role, receipt or grid is not accepted for display.");
  }
  const sourceRAS = rasAffine(volume.affine, volume.frame);
  if (!sameAffine(d.affine_ras_mm as Affine, sourceRAS) ||
      !sameAffine(d.affine_ras_mm as Affine, layer.stateGrid.affineRAS) ||
      !sameAffine(d.affine_ras_mm as Affine, layer.coverageGrid.affineRAS)) {
    throw new Error("Diagnostic layer sform differs from selected image frame.");
  }
  const count = volume.shape.reduce((n, size) => n * size, 1);
  if (!Number.isSafeInteger(count) || count <= 0 ||
      layer.stateXYZ.length !== count || layer.coverageXYZ.length !== count) {
    throw new Error("Diagnostic state/coverage arrays do not match the selected grid.");
  }
  let covered = 0, unknown = 0, positive = 0;
  for (let i = 0; i < count; i++) {
    const state = layer.stateXYZ[i], coverage = layer.coverageXYZ[i];
    if (coverage !== 0 && coverage !== 1) {
      throw new Error("Diagnostic coverage contains a nonbinary value.");
    }
    if ((coverage === 0 && state !== -1) ||
        (coverage === 1 && state !== 0 && state !== 1)) {
      throw new Error("Diagnostic unknown, negative and positive states disagree with coverage.");
    }
    covered += coverage;
    unknown += state === -1 ? 1 : 0;
    positive += state === 1 ? 1 : 0;
  }
  if (covered !== d.predicted_voxels || unknown !== d.unknown_voxels ||
      d.excluded_positive_voxels < 0 || positive > covered) {
    throw new Error("Diagnostic count receipt disagrees with loaded state/coverage.");
  }
  return {
    layer, worldToVoxel: inverseAffine(d.affine_ras_mm as Affine),
    shape: [...d.shape_xyz], primaryOverlaysPermitted: false,
    planningInteractionPermitted: false,
  };
}

export function sampleDiagnosticLayer(checked: CheckedDiagnosticLayer, world: Point3):
  "unknown" | "candidate_negative" | "candidate_positive" {
  const v = transformPoint(checked.worldToVoxel, world);
  const index = v.map((x) => Math.floor(x + 0.5));
  if (index.some((x, i) => x < 0 || x >= checked.shape[i])) return "unknown";
  const flat = (index[0] * checked.shape[1] + index[1]) * checked.shape[2] + index[2];
  const state = checked.layer.stateXYZ[flat];
  return state === 1 ? "candidate_positive" : state === 0 ? "candidate_negative" : "unknown";
}
