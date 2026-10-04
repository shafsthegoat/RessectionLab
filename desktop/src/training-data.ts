import type {
  CasePayload,
  CertifiedReplay,
  ReplayResult,
  ResectionApi,
} from "./types";

export async function hydrateCertifiedReplay(
  result: ReplayResult,
  source: CasePayload,
  api: ResectionApi,
): Promise<CertifiedReplay> {
  if (
    result.caseHash !== source.caseHash ||
    result.role !== "selection" ||
    result.finalEvaluation !== false ||
    result.clinicalDeficitProbability !== null ||
    result.frame !== "RAS+"
  )
    throw new Error("Replay case identity or evidence role is invalid.");
  const expected = source.affine.map((row, index) =>
    row.map((value) => (source.frame === "LPS+" && index < 2 ? -value : value)),
  );
  if (
    result.affine.length !== 4 ||
    result.affine.some(
      (row, r) =>
        row.length !== 4 ||
        row.some(
          (value, c) =>
            !Number.isFinite(value) || Math.abs(value - expected[r][c]) > 1e-7,
        ),
    )
  )
    throw new Error("Replay physical frame differs from source imaging.");
  const descriptor = result.removedMask,
    count = source.shape.reduce((a, b) => a * b, 1);
  if (
    !/^[a-f0-9]{64}$/.test(descriptor.sha256) ||
    !descriptor.assetId ||
    descriptor.dtype !== "uint8" ||
    descriptor.byteOrder !== "little" ||
    descriptor.order !== "C" ||
    descriptor.byteLength !== count ||
    descriptor.shape.length !== 3 ||
    descriptor.shape.some((value, index) => value !== source.shape[index])
  )
    throw new Error("Replay removal mask differs from the source grid.");
  const certificate = result.training.replay?.native_certificate;
  if (
    result.training.replay_status !== "accepted_independent_geometry" ||
    !certificate ||
    certificate.feasible !== true ||
    certificate.complete_tool_checked !== true ||
    certificate.frontier_checked !== true ||
    certificate.source_case_hash !== source.caseHash
  )
    throw new Error("Replay lacks independent native geometry acceptance.");
  if (
    !Number.isInteger(result.step) ||
    !Number.isInteger(result.stepCount) ||
    result.step < 0 ||
    result.step > result.stepCount
  )
    throw new Error("Replay step is invalid.");
  const volumes = [
    result.simulatedRemovedTargetVolumeMm3,
    result.simulatedRemovedNormalVolumeMm3,
    result.modeledResidualTargetVolumeMm3,
  ];
  if (volumes.some((value) => !Number.isFinite(value) || value < 0))
    throw new Error("Replay volume accounting is invalid.");
  const mask = await api.readAsset(descriptor.assetId);
  if (mask.length !== count)
    throw new Error("Replay mask transfer is incomplete.");
  let removed = 0;
  for (const value of mask) {
    if (value !== 0 && value !== 1)
      throw new Error("Replay mask is not binary.");
    removed += value;
  }
  const [a, b, c] = expected;
  const voxelVolume = Math.abs(
    a[0] * (b[1] * c[2] - b[2] * c[1]) -
      a[1] * (b[0] * c[2] - b[2] * c[0]) +
      a[2] * (b[0] * c[1] - b[1] * c[0]),
  );
  if (
    Math.abs(removed * voxelVolume - volumes[0] - volumes[1]) >
    1e-5 * Math.max(1, removed * voxelVolume)
  )
    throw new Error(
      "Replay displayed volumes do not match removed source cells.",
    );
  return { result, mask };
}
