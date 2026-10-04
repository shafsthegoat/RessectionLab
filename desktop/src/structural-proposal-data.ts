import { validateCaseDescriptor } from "./case-data.ts";
import type {
  CasePayload,
  ResectionApi,
  StructuralProposalView,
  Vec3,
  ViewerCase,
} from "./types";

import {
  sha256Bytes,
  arrayDigest,
  sourceFrameDigest,
  sourceImageDigest,
  throwIfAborted,
} from "./source-integrity.ts";
export { pythonAffineFloat } from "./source-integrity.ts";
const hashPattern = /^sha256:[a-f0-9]{64}$/;

export async function hydrateStructuralProposal(
  source: CasePayload,
  viewer: ViewerCase,
  evidenceId: string,
  api: Pick<ResectionApi, "readAsset">,
  signal?: AbortSignal,
): Promise<StructuralProposalView> {
  throwIfAborted(signal);
  const { voxelCount } = validateCaseDescriptor(source);
  if (
    viewer.caseHash !== source.caseHash ||
    viewer.frame !== "RAS+" ||
    viewer.shape.length !== 3 ||
    viewer.shape.some((value, index) => value !== source.shape[index])
  )
    throw new Error("Proposal source is no longer the visible case.");
  const expected = source.affine.map((row, index) =>
    row.map((value) => (source.frame === "LPS+" && index < 2 ? -value : value)),
  );
  if (
    viewer.affine.length !== 4 ||
    viewer.affine.some(
      (row, r) =>
        row.length !== 4 ||
        row.some(
          (value, c) =>
            !Number.isFinite(value) || Math.abs(value - expected[r][c]) > 1e-7,
        ),
    )
  )
    throw new Error(
      "Proposal source physical frame differs from the visible MRI.",
    );
  const matches = (source.structuralEvidence ?? []).filter(
    (item) => item.evidenceId === evidenceId,
  );
  if (matches.length !== 1)
    throw new Error("Choose a proposal from the current case inventory.");
  const item = matches[0],
    descriptor = item.array;
  if (
    item.kind !== "whole_brain_envelope" ||
    item.provenance !== "estimated" ||
    item.corticalAccessPermitted !== false
  )
    throw new Error(
      "This view requires a separate estimated brain-envelope record.",
    );
  if (
    !["review_required", "accepted", "rejected"].includes(item.reviewStatus) ||
    item.reviewRequired !== (item.reviewStatus === "review_required")
  )
    throw new Error("Proposal review status is inconsistent.");
  const review = item.review as Record<string, unknown> | null;
  if (
    item.reviewRequired
      ? review !== null
      : !review ||
        typeof review !== "object" ||
        review.evidence_hash !== item.evidenceHash ||
        review.decision !== item.reviewStatus ||
        review.scope !== "research_brain_envelope_only"
  )
    throw new Error("Proposal review is not bound to this exact evidence.");
  for (const identity of [
    item.sourceHash,
    item.sourceFrameHash,
    item.maskHash,
    item.runHash,
    item.evidenceHash,
  ])
    if (!hashPattern.test(identity))
      throw new Error("Proposal provenance has an invalid identity.");
  for (const identity of [item.modelHash, item.sourceFileHash])
    if (identity != null && !hashPattern.test(identity))
      throw new Error("Proposal source or model identity is invalid.");
  if (
    item.sourceFileHash &&
    !source.sourceRefs?.some(
      (ref) =>
        ref.sha256?.replace(/^sha256:/, "") === item.sourceFileHash!.slice(7),
    )
  )
    throw new Error("Proposal source file is absent from this case.");
  if (
    !descriptor ||
    descriptor.dtype !== "uint8" ||
    descriptor.byteOrder !== "little" ||
    descriptor.order !== "C" ||
    descriptor.byteLength !== voxelCount ||
    !Array.isArray(descriptor.shape) ||
    descriptor.shape.length !== 3 ||
    descriptor.shape.some((size, index) => size !== source.shape[index]) ||
    !descriptor.assetId ||
    !/^[a-f0-9]{64}$/.test(descriptor.sha256)
  )
    throw new Error("Proposal mask does not match the original MRI grid.");
  if (
    viewer.mri.length !== voxelCount ||
    viewer.compartments.some((layer) => layer.mask.length !== voxelCount)
  )
    throw new Error("Visible source arrays no longer match their grid.");
  const [imageHash, physicalHash] = await Promise.all([
    sourceImageDigest(viewer, source.shape),
    sourceFrameDigest(source),
  ]);
  throwIfAborted(signal);
  if (imageHash !== item.sourceHash || physicalHash !== item.sourceFrameHash)
    throw new Error(
      "Proposal belongs to another source image or physical frame.",
    );
  const transferred = await api.readAsset(descriptor.assetId);
  throwIfAborted(signal);
  if (transferred.byteLength !== voxelCount)
    throw new Error("Proposal mask transfer is incomplete.");
  // Own the display buffer, so source masks and later IPC buffers cannot be changed through it.
  const mask = transferred.slice();
  const [transferHash, maskHash] = await Promise.all([
    sha256Bytes(mask),
    arrayDigest(mask, source.shape, "|b1"),
  ]);
  throwIfAborted(signal);
  if (transferHash !== descriptor.sha256 || maskHash !== item.maskHash)
    throw new Error(
      "Proposal mask checksum does not match its recorded evidence.",
    );
  let count = 0,
    outsideCount = 0,
    firstOutside = -1;
  const min: Vec3 = [Infinity, Infinity, Infinity],
    max: Vec3 = [-1, -1, -1];
  const [, sy, sz] = source.shape;
  for (let index = 0; index < mask.length; index++) {
    const value = mask[index];
    if (value !== 0 && value !== 1)
      throw new Error("Proposal mask contains non-binary source cells.");
    if (value) {
      count++;
      const point = [
        Math.floor(index / (sy * sz)),
        Math.floor(index / sz) % sy,
        index % sz,
      ];
      for (let axis = 0; axis < 3; axis++) {
        min[axis] = Math.min(min[axis], point[axis]);
        max[axis] = Math.max(max[axis], point[axis]);
      }
    } else if (viewer.compartments.some((layer) => layer.mask[index] === 1)) {
      outsideCount++;
      if (firstOutside < 0) firstOutside = index;
    }
    if (index > 0 && index % 262144 === 0) {
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
      throwIfAborted(signal);
    }
  }
  if (!count) throw new Error("Proposal mask has no estimated source cells.");
  const point =
    firstOutside < 0
      ? null
      : [
          Math.floor(firstOutside / (sy * sz)),
          Math.floor(firstOutside / sz) % sy,
          firstOutside % sz,
        ];
  const outsidePoint = point
    ? (expected
        .slice(0, 3)
        .map(
          (row) =>
            row[0] * point[0] + row[1] * point[1] + row[2] * point[2] + row[3],
        ) as Vec3)
    : null;
  return {
    caseHash: source.caseHash,
    evidenceId: item.evidenceId,
    mask,
    shape: [...source.shape],
    affine: expected,
    frame: "RAS+",
    reviewStatus: item.reviewStatus,
    provenance: "estimated",
    scope: "display-only-estimate",
    label:
      item.metadata.variant === "nocsf"
        ? "Estimated envelope · no CSF"
        : "Estimated brain envelope",
    annotationOutsideVoxelCount: outsideCount,
    outsideAnnotationPointMm: outsidePoint,
    voxelBounds: { min, max },
    estimatedVoxelCount: count,
  };
}
