import { validateCaseDescriptor } from "./case-data.ts";
import {
  arrayDigest,
  sha256Bytes,
  sourceFrameDigest,
  sourceImageDigest,
  throwIfAborted,
} from "./source-integrity.ts";
import { inverseAffine, transformPoint } from "./viewer/coordinates.ts";
import { samplePriorVoxel } from "./viewer/priorLayer.ts";
import type {
  ArrayDescriptor,
  CasePayload,
  Mat4,
  PriorLayerView,
  PriorProposal,
  ResectionApi,
  Vec3,
  ViewerCase,
} from "./types";

const hashPattern = /^sha256:[a-f0-9]{64}$/;
const componentNames = {
  motor: "Motor",
  phonology: "Phonology",
  semantics: "Semantics",
  speech_articulation: "Speech arrest / articulation",
};
export function priorLabel(
  item: Pick<PriorProposal, "component" | "mapKind">,
): string {
  return `${componentNames[item.component]} · ${item.mapKind === "functional_concordance" ? "functional network" : "structural network mask"}`;
}
function grid(affine: Mat4): void {
  if (
    !Array.isArray(affine) ||
    affine.length !== 4 ||
    affine.some(
      (row) =>
        !Array.isArray(row) ||
        row.length !== 4 ||
        row.some((value) => !Number.isFinite(value)),
    ) ||
    affine[3].some(
      (value, index) => Math.abs(value - (index === 3 ? 1 : 0)) > 1e-12,
    )
  )
    throw new Error("Prior has an invalid physical affine.");
  inverseAffine(affine);
}
function matchesGrid(a: Mat4, b: Mat4, shape: Vec3): boolean {
  grid(a);
  grid(b);
  for (const x of [0, shape[0] - 1])
    for (const y of [0, shape[1] - 1])
      for (const z of [0, shape[2] - 1]) {
        const p = transformPoint(a, [x, y, z]),
          q = transformPoint(b, [x, y, z]);
        if (Math.hypot(...p.map((value, index) => value - q[index])) > 0.01)
          return false;
      }
  return true;
}
function descriptor(
  value: ArrayDescriptor,
  dtype: string,
  source: CasePayload,
  count: number,
): void {
  if (
    !value ||
    value.dtype !== dtype ||
    value.byteOrder !== "little" ||
    value.order !== "C" ||
    !value.assetId ||
    !/^[a-f0-9]{64}$/.test(value.sha256) ||
    value.byteLength !== count * (dtype === "float32" ? 4 : 1) ||
    !Array.isArray(value.shape) ||
    value.shape.length !== 3 ||
    value.shape.some((n, i) => n !== source.shape[i])
  )
    throw new Error(
      "Prior values or coverage do not match their declared source grid.",
    );
}
export async function hydratePriorProposal(
  source: CasePayload,
  viewer: ViewerCase,
  proposalId: string,
  api: Pick<ResectionApi, "readAsset">,
  signal?: AbortSignal,
): Promise<PriorLayerView> {
  throwIfAborted(signal);
  const { voxelCount } = validateCaseDescriptor(source);
  const sourceRas = source.affine.map((row, r) =>
    row.map((n) => (source.frame === "LPS+" && r < 2 ? -n : n)),
  );
  if (
    viewer.caseHash !== source.caseHash ||
    viewer.frame !== "RAS+" ||
    viewer.shape.length !== 3 ||
    viewer.shape.some((n, i) => n !== source.shape[i]) ||
    !matchesGrid(viewer.affine, sourceRas, source.shape)
  )
    throw new Error("Prior source is no longer the visible MRI.");
  const matches = (source.priorProposals ?? []).filter(
    (item) => item.proposalId === proposalId,
  );
  if (matches.length !== 1)
    throw new Error("Choose one prior from the current case inventory.");
  const item = matches[0],
    record = item.provenanceRecord;
  if (
    item.provenance !== "prior" ||
    item.reviewStatus !== "alignment_review_required" ||
    item.viewOnly !== true ||
    item.planningEligible !== false ||
    item.patientSpecificFunction !== false ||
    item.clinicalDeficitProbability !== null ||
    item.frame !== "RAS+" ||
    item.spatialUnits !== "mm" ||
    item.valueUnits !== "unitless" ||
    item.coverageMeaning !==
      "atlas field of view, not patient functional coverage"
  )
    throw new Error(
      "Prior evidence cannot be promoted to patient function or planning evidence.",
    );
  if (
    !Object.hasOwn(componentNames, item.component) ||
    !["functional_concordance", "structural_mask"].includes(item.mapKind) ||
    item.mapId !== `${item.component}_${item.mapKind}` ||
    (item.component === "motor" && item.mapKind !== "functional_concordance") ||
    item.interpolation !==
      (item.mapKind === "structural_mask" ? "nearest_neighbor" : "linear")
  )
    throw new Error(
      "Prior component, map kind or interpolation is inconsistent.",
    );
  if (
    !Array.isArray(item.shape) ||
    item.shape.length !== 3 ||
    item.shape.some((n, i) => n !== source.shape[i]) ||
    !matchesGrid(item.affine, sourceRas, source.shape)
  )
    throw new Error(
      "Prior physical sampling grid differs from the source case.",
    );
  for (const hash of [
    item.evidenceHash,
    item.registrationHash,
    item.sourceImageHash,
    item.sourceFrameHash,
    item.sourcePlanningHash,
    item.registrationCaseHash,
  ])
    if (!hashPattern.test(hash))
      throw new Error("Prior provenance is missing a valid identity.");
  if (item.sourcePlanningHash !== source.planningHash)
    throw new Error("Prior registration belongs to different planning inputs.");
  if (
    !record ||
    record.evidence_hash !== item.evidenceHash ||
    record.registration_hash !== item.registrationHash ||
    record.source_image_hash !== item.sourceImageHash ||
    record.source_frame_hash !== item.sourceFrameHash ||
    record.source_case_planning_hash !== item.sourcePlanningHash ||
    record.registration_case_hash !== item.registrationCaseHash ||
    record.proposal_id !== item.proposalId ||
    record.map_id !== item.mapId ||
    record.review !== null ||
    record.view_only !== true ||
    record.planning_eligible !== false ||
    record.patient_specific_function !== false ||
    record.clinical_deficit_probability !== null ||
    !hashPattern.test(String(record.data_hash)) ||
    !hashPattern.test(String(record.sampling_coverage_hash))
  )
    throw new Error("Prior provenance does not match the displayed proposal.");
  if (
    item.source.provenance !== "prior" ||
    item.source.native_frame !== "FSL_MNI152_RAS_mm" ||
    !/^[a-f0-9]{64}$/.test(item.source.sha256) ||
    !source.sourceRefs?.some(
      (ref) =>
        ref.sha256?.replace(/^sha256:/, "") ===
        String(record.source_file_sha256).replace(/^sha256:/, ""),
    )
  )
    throw new Error("Prior source or patient-image provenance is incomplete.");
  descriptor(item.data, "float32", source, voxelCount);
  descriptor(item.samplingCoverage, "uint8", source, voxelCount);
  const [imageHash, physicalHash] = await Promise.all([
    sourceImageDigest(viewer, source.shape),
    sourceFrameDigest(source),
  ]);
  throwIfAborted(signal);
  if (
    imageHash !== item.sourceImageHash ||
    physicalHash !== item.sourceFrameHash
  )
    throw new Error("Prior belongs to another source MRI or physical frame.");
  const [rawValues, rawCoverage] = await Promise.all([
    api.readAsset(item.data.assetId),
    api.readAsset(item.samplingCoverage.assetId),
  ]);
  throwIfAborted(signal);
  if (
    rawValues.byteLength !== voxelCount * 4 ||
    rawCoverage.byteLength !== voxelCount
  )
    throw new Error("Prior value or coverage transfer is incomplete.");
  const values = new Float32Array(rawValues.slice().buffer),
    coverage = rawCoverage.slice();
  const [rawValueHash, rawCoverageHash, valueHash, coverageHash] =
    await Promise.all([
      sha256Bytes(rawValues),
      sha256Bytes(coverage),
      arrayDigest(new Uint8Array(values.buffer), source.shape, "<f4"),
      arrayDigest(coverage, source.shape, "|b1"),
    ]);
  throwIfAborted(signal);
  if (
    rawValueHash !== item.data.sha256 ||
    rawCoverageHash !== item.samplingCoverage.sha256 ||
    valueHash !== record.data_hash ||
    coverageHash !== record.sampling_coverage_hash
  )
    throw new Error(
      "Prior values or coverage differ from the recorded evidence.",
    );
  let covered = 0;
  for (let index = 0; index < voxelCount; index++) {
    const value = values[index],
      inside = coverage[index];
    if (
      !Number.isFinite(value) ||
      value < 0 ||
      value > 1 ||
      (inside !== 0 && inside !== 1) ||
      (item.mapKind === "structural_mask" && value !== 0 && value !== 1) ||
      (!inside && value !== 0)
    )
      throw new Error("Prior values or atlas coverage have invalid semantics.");
    covered += inside;
    if (index > 0 && index % 262144 === 0) {
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
      throwIfAborted(signal);
    }
  }
  if (
    !Number.isFinite(item.samplingCoverageFraction) ||
    Math.abs(covered / voxelCount - item.samplingCoverageFraction) > 1e-10
  )
    throw new Error(
      "Atlas field-of-view coverage count disagrees with its record.",
    );
  return {
    caseHash: source.caseHash,
    proposalId: item.proposalId,
    mapId: item.mapId,
    title: priorLabel(item),
    component: item.component,
    mapKind: item.mapKind,
    values,
    coverage,
    shape: [...source.shape],
    affine: item.affine.map((row) => [...row]),
    frame: "RAS+",
    scope: "view-only-population-prior",
    provenance: "prior",
    reviewStatus: "alignment_review_required",
    planningEligible: false,
    patientSpecificFunction: false,
    valueUnits: "unitless",
    spatialUnits: "mm",
    proposal: item,
  };
}
export interface PriorCursorSample {
  covered: boolean;
  value: number | null;
  voxel: Vec3 | null;
  reason: string | null;
}
/** Use the same prior-affine sample and coverage support as the MRI renderer. */
export function samplePriorAtCursor(
  layer: PriorLayerView,
  cursor: Vec3 | null,
): PriorCursorSample {
  if (!cursor || cursor.some((n) => !Number.isFinite(n)))
    return {
      covered: false,
      value: null,
      voxel: null,
      reason: "outside-atlas-coverage",
    };
  const position = transformPoint(inverseAffine(layer.affine), cursor);
  if (position.some((n, i) => n < -0.5 || n >= layer.shape[i] - 0.5))
    return {
      covered: false,
      value: null,
      voxel: null,
      reason: "outside-atlas-coverage",
    };
  const voxel = position.map((n, i) =>
    Math.max(0, Math.min(layer.shape[i] - 1, Math.round(n))),
  ) as Vec3;
  return { ...samplePriorVoxel(layer, position), voxel };
}
