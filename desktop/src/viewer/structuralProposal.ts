import type { ViewerStructuralProposal, ViewerVolume } from "./contracts";

export const STRUCTURAL_PROPOSAL_COLOR = "#d9a7ff";

/** Status comes from the bound evidence record, never from a free-form label. */
export function structuralProposalReviewLabel(status: string): string {
  if (status === "review_required") return "Review required";
  if (status === "accepted") return "Reviewed envelope, display only";
  if (status === "rejected") return "Rejected estimate";
  throw new Error("Estimated envelope has an unresolved review status.");
}

/** Display gate only. It cannot promote an estimate to anatomy or planning input. */
export function validateStructuralProposal(
  volume: ViewerVolume,
  proposal: ViewerStructuralProposal,
): void {
  if (
    proposal.scope !== "display-only-estimate" ||
    proposal.provenance !== "estimated"
  )
    throw new Error(
      "Envelope display requires explicitly estimated, view-only evidence.",
    );
  if (
    proposal.caseHash !== volume.caseHash ||
    typeof proposal.evidenceId !== "string" ||
    !proposal.evidenceId.trim()
  )
    throw new Error(
      "Estimated envelope is not bound to this source case version.",
    );
  structuralProposalReviewLabel(proposal.reviewStatus);
  if (
    proposal.frame !== "RAS+" ||
    !["RAS+", "RAS", "LPS+", "LPS"].includes(volume.frame)
  )
    throw new Error(
      "Estimated envelope requires a resolved source frame in RAS millimeters.",
    );
  const size = volume.shape.reduce((a, b) => a * b, 1);
  if (
    !(proposal.mask instanceof Uint8Array) ||
    proposal.mask.length !== size ||
    proposal.shape.length !== 3 ||
    proposal.shape.some((value, index) => value !== volume.shape[index])
  )
    throw new Error(
      "Estimated envelope dimensions differ from the original MRI grid.",
    );
  const affine = volume.affine.map((row, r) =>
    row.map((value) =>
      volume.frame.startsWith("LPS") && r < 2 ? -value : value,
    ),
  );
  if (
    affine.length !== 4 ||
    affine.some(
      (row) => row.length !== 4 || row.some((value) => !Number.isFinite(value)),
    ) ||
    proposal.affine.length !== 4 ||
    proposal.affine.some(
      (row, r) =>
        row.length !== 4 ||
        row.some(
          (value, c) =>
            !Number.isFinite(value) || Math.abs(value - affine[r][c]) > 1e-7,
        ),
    )
  )
    throw new Error(
      "Estimated envelope affine differs from the original MRI in RAS millimeters.",
    );
  let occupied = 0;
  for (let i = 0; i < proposal.mask.length; i++) {
    const value = proposal.mask[i];
    if (value !== 0 && value !== 1)
      throw new Error("Estimated envelope mask must be binary.");
    occupied += value;
  }
  if (!occupied) throw new Error("Estimated envelope mask is empty.");
}
