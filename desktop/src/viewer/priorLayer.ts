import type { ViewerPriorLayer, ViewerVolume } from "./contracts";
import { inverseAffine, transformPoint } from "./coordinates.ts";

export const PRIOR_COLORS = ["#415577", "#749daf", "#e1bc82"] as const;

type PriorGrid = Pick<ViewerPriorLayer, "affine" | "shape">;
type VoxelTolerance = readonly [number, number, number];
const precisionCache = new WeakMap<
  PriorGrid,
  {
    gridSignature: string;
    tolerance: VoxelTolerance | null;
  }
>();
const MAX_VOXEL_TOLERANCE = 0.001;

function gridSignature(layer: PriorGrid): string {
  return [...layer.shape, ...layer.affine.flat()].join(",");
}

function computeSamplingTolerance(layer: PriorGrid): VoxelTolerance | null {
  const inverse = inverseAffine(layer.affine);
  const maxWorld = [0, 0, 0];
  for (const x of [-0.5, layer.shape[0] - 0.5])
    for (const y of [-0.5, layer.shape[1] - 0.5])
      for (const z of [-0.5, layer.shape[2] - 0.5]) {
        const world = transformPoint(layer.affine, [x, y, z]);
        world.forEach((value, axis) => {
          maxWorld[axis] = Math.max(maxWorld[axis], Math.abs(value));
        });
      }
  // Allowance for float32 world/coefficient quantization and matrix products.
  // Store float32 values so the CPU and uniform use the same boundary band.
  const tolerance = inverse
    .slice(0, 3)
    .map((row) =>
      Math.fround(
        Math.max(
          2 ** -20,
          16 *
            2 ** -24 *
            (1 +
              Math.abs(row[3]) +
              maxWorld.reduce(
                (sum, value, axis) => sum + value * Math.abs(row[axis]),
                0,
              )),
        ),
      ),
    );
  return tolerance.some(
    (value) => !Number.isFinite(value) || value > MAX_VOXEL_TOLERANCE,
  )
    ? null
    : (Object.freeze(tolerance) as VoxelTolerance);
}

/** Reuse precision preparation while all 19 grid numbers match; null means abstain. */
export function priorSamplingTolerance(
  layer: PriorGrid,
): VoxelTolerance | null {
  const signature = gridSignature(layer),
    cached = precisionCache.get(layer);
  if (cached?.gridSignature === signature) return cached.tolerance;
  const tolerance = computeSamplingTolerance(layer);
  precisionCache.set(layer, { gridSignature: signature, tolerance });
  return tolerance;
}

/** A positive atlas sample must never be rounded into a displayed zero. */
export function formatPriorValue(value: number): string {
  return value > 0 && value < 0.0001
    ? value.toExponential(2)
    : value.toFixed(4);
}

/** Registered source-grid previews retain their supplied RAS affine. */
export function validatePriorLayer(
  volume: ViewerVolume,
  layer: ViewerPriorLayer,
): void {
  if (
    layer.scope !== "view-only-population-prior" ||
    layer.provenance !== "prior" ||
    layer.planningEligible !== false ||
    layer.patientSpecificFunction !== false ||
    layer.reviewStatus !== "alignment_review_required"
  )
    throw new Error(
      "Population prior requires view-only, alignment-review-required status.",
    );
  if (
    layer.caseHash !== volume.caseHash ||
    typeof layer.proposalId !== "string" ||
    !layer.proposalId.trim()
  )
    throw new Error(
      "Population prior belongs to a different source case version.",
    );
  if (
    layer.frame !== "RAS+" ||
    !["RAS+", "RAS", "LPS+", "LPS"].includes(volume.frame) ||
    layer.spatialUnits !== "mm" ||
    layer.valueUnits !== "unitless"
  )
    throw new Error(
      "Population prior has unresolved physical frame or value units.",
    );
  if (!["functional_concordance", "structural_mask"].includes(layer.mapKind))
    throw new Error("Population prior map kind is unsupported.");
  const size = volume.shape.reduce((a, b) => a * b, 1);
  if (
    !(layer.values instanceof Float32Array) ||
    !(layer.coverage instanceof Uint8Array) ||
    layer.values.length !== size ||
    layer.coverage.length !== size ||
    layer.shape.length !== 3 ||
    layer.shape.some((value, index) => value !== volume.shape[index])
  )
    throw new Error(
      "Population prior values and coverage must match the native source grid.",
    );
  const affine = volume.affine.map((row, r) =>
    row.map((value) =>
      volume.frame.startsWith("LPS") && r < 2 ? -value : value,
    ),
  );
  if (
    layer.affine.length !== 4 ||
    layer.affine.some(
      (row) => row.length !== 4 || row.some((value) => !Number.isFinite(value)),
    ) ||
    layer.affine[3].some((value, index) => value !== (index === 3 ? 1 : 0))
  )
    throw new Error("Population prior affine is invalid.");
  for (const x of [0, volume.shape[0] - 1])
    for (const y of [0, volume.shape[1] - 1])
      for (const z of [0, volume.shape[2] - 1]) {
        const p = [x, y, z, 1];
        const delta = [0, 1, 2].map((axis) =>
          p.reduce(
            (sum, value, col) =>
              sum + value * (layer.affine[axis][col] - affine[axis][col]),
            0,
          ),
        );
        if (Math.hypot(...delta) > 0.01 + 1e-10)
          throw new Error(
            "Population prior affine is outside the registered source-grid tolerance.",
          );
      }
  for (let i = 0; i < size; i++) {
    const value = layer.values[i],
      coverage = layer.coverage[i];
    if (
      !Number.isFinite(value) ||
      value < 0 ||
      value > 1 ||
      (layer.mapKind === "structural_mask" && value !== 0 && value !== 1)
    )
      throw new Error(
        "Population prior values do not match their declared 0–1 semantics.",
      );
    if (coverage !== 0 && coverage !== 1)
      throw new Error("Population prior coverage must be binary.");
  }
  // Revalidating a proposal refreshes its budget even if a caller replaced its grid.
  const tolerance = computeSamplingTolerance(layer);
  precisionCache.set(layer, { gridSignature: gridSignature(layer), tolerance });
  if (!tolerance)
    throw new Error(
      "Population prior coordinates exceed the 0.001-voxel display precision limit.",
    );
}

/** CPU reference for the rendered sample. Missing support is never a zero value. */
export function samplePriorVoxel(
  layer: ViewerPriorLayer,
  point: readonly number[],
): { covered: boolean; value: number | null; reason: string | null } {
  const absent = {
    covered: false,
    value: null,
    reason: "outside-atlas-coverage",
  };
  if (point.length !== 3 || point.some((value) => !Number.isFinite(value)))
    return absent;
  const tolerance = priorSamplingTolerance(layer);
  if (!tolerance)
    return { ...absent, reason: "numerical-precision-unavailable" };
  if (
    point.some(
      (value, axis) =>
        Math.abs(value + 0.5) <= tolerance[axis] ||
        Math.abs(value - (layer.shape[axis] - 0.5)) <= tolerance[axis],
    )
  )
    return { ...absent, reason: "numerical-boundary-uncertainty" };
  if (
    point.some(
      (value, axis) => value < -0.5 || value >= layer.shape[axis] - 0.5,
    )
  )
    return absent;
  const voxel = point.map((value, axis) => {
    const grid = layer.mapKind === "structural_mask" ? 2 : 1;
    const anchor = Math.floor(value * grid + 0.5) / grid;
    const snapped =
      Math.abs(value - anchor) <= tolerance[axis] ? anchor : value;
    return Math.max(0, Math.min(layer.shape[axis] - 1, snapped));
  });
  const index = (x: number, y: number, z: number) =>
    (x * layer.shape[1] + y) * layer.shape[2] + z;
  if (layer.mapKind === "structural_mask") {
    const [x, y, z] = voxel.map(Math.round),
      i = index(x, y, z);
    return layer.coverage[i]
      ? { covered: true, value: layer.values[i], reason: null }
      : absent;
  }
  const base = voxel.map(Math.floor),
    fraction = voxel.map((value, axis) => value - base[axis]);
  let value = 0;
  for (let dx = 0; dx <= 1; dx++)
    for (let dy = 0; dy <= 1; dy++)
      for (let dz = 0; dz <= 1; dz++) {
        const weight =
          (dx ? fraction[0] : 1 - fraction[0]) *
          (dy ? fraction[1] : 1 - fraction[1]) *
          (dz ? fraction[2] : 1 - fraction[2]);
        if (weight <= 0) continue;
        const i = index(
          Math.min(base[0] + dx, layer.shape[0] - 1),
          Math.min(base[1] + dy, layer.shape[1] - 1),
          Math.min(base[2] + dz, layer.shape[2] - 1),
        );
        if (!layer.coverage[i]) {
          const nearest = index(
            ...(voxel.map(Math.round) as [number, number, number]),
          );
          return {
            ...absent,
            reason: layer.coverage[nearest]
              ? "incomplete-interpolation-support"
              : "outside-atlas-coverage",
          };
        }
        value += weight * layer.values[i];
      }
  return { covered: true, value, reason: null };
}
