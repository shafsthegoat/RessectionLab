import type { ViewerReplay, ViewerVolume } from "./contracts";

/** Display only a separately accepted effect in the unchanged source voxel grid.
 * This check catches stale arrays, frame errors and conflicting cell accounting;
 * it does not replace the independent backend trajectory/interaction checker.
 */
export function validateReplay(
  volume: ViewerVolume,
  replay: ViewerReplay,
): void {
  if (
    replay.independentlyAccepted !== true ||
    replay.scope !== "native-source-grid"
  )
    throw new Error(
      "Modeled removal requires independent source-grid geometry acceptance.",
    );
  if (replay.caseHash !== volume.caseHash)
    throw new Error("Modeled removal belongs to a different case version.");
  if (
    !Number.isInteger(replay.step) ||
    !Number.isInteger(replay.stepCount) ||
    replay.step < 0 ||
    replay.step > replay.stepCount
  )
    throw new Error("Modeled replay step is invalid.");
  const size = volume.shape.reduce((a, b) => a * b, 1);
  if (
    !(replay.removedMask instanceof Uint8Array) ||
    replay.shape.length !== 3 ||
    replay.shape.some((value, index) => value !== volume.shape[index]) ||
    replay.removedMask.length !== size
  )
    throw new Error("Modeled removal shape differs from the source grid.");
  if (!["RAS+", "RAS", "LPS+", "LPS"].includes(volume.frame))
    throw new Error("Source frame is unresolved.");
  const affine = volume.affine.map((row, r) =>
    row.map((value) =>
      volume.frame.startsWith("LPS") && r < 2 ? -value : value,
    ),
  );
  if (
    replay.affine.length !== 4 ||
    replay.affine.some(
      (row, r) =>
        row.length !== 4 ||
        row.some(
          (value, c) =>
            !Number.isFinite(value) || Math.abs(value - affine[r][c]) > 1e-7,
        ),
    )
  )
    throw new Error(
      "Modeled removal affine differs from source RAS millimeters.",
    );
  const [a, b, c] = affine;
  const voxelVolume = Math.abs(
    a[0] * (b[1] * c[2] - b[2] * c[1]) -
      a[1] * (b[0] * c[2] - b[2] * c[0]) +
      a[2] * (b[0] * c[1] - b[1] * c[0]),
  );
  if (
    !Number.isFinite(voxelVolume) ||
    voxelVolume <= 0 ||
    volume.compartments.some((layer) => layer.mask.length !== size)
  )
    throw new Error("Source geometry or annotation grid is invalid.");
  let removed = 0,
    removedTarget = 0,
    totalTarget = 0;
  for (let i = 0; i < size; i++) {
    const value = replay.removedMask[i];
    if (value !== 0 && value !== 1)
      throw new Error("Modeled removal mask is not binary.");
    const target = volume.compartments.some((layer) => layer.mask[i] !== 0);
    removed += value;
    if (target) {
      totalTarget++;
      removedTarget += value;
    }
  }
  const actual = [
    removedTarget * voxelVolume,
    (removed - removedTarget) * voxelVolume,
    (totalTarget - removedTarget) * voxelVolume,
  ];
  const reported = [
    replay.removedTargetVolumeMm3,
    replay.removedNormalVolumeMm3,
    replay.residualTargetVolumeMm3,
  ];
  if (
    reported.some(
      (value, i) =>
        !Number.isFinite(value) ||
        value < 0 ||
        Math.abs(value - actual[i]) > 1e-5 * Math.max(1, actual[i]),
    )
  )
    throw new Error(
      "Modeled removed/residual volumes do not match the source cells.",
    );
}

/** A new display array only; the supplied source annotation is never edited. */
export function residualMask(
  source: Uint8Array,
  removed: Uint8Array,
): Uint8Array {
  if (source.length !== removed.length)
    throw new Error("Residual mask requires identical source grids.");
  const result = new Uint8Array(source.length);
  for (let i = 0; i < result.length; i++)
    result[i] = source[i] && !removed[i] ? 1 : 0;
  return result;
}
