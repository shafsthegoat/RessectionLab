import type { CasePayload } from "./types";

/** Display-only structural proposals never become tissue support by inference. */
export function researchSupportGate(source: CasePayload | null): {
  blocked: boolean;
  requiresEstimatedSupport: boolean;
  reason: string | null;
} {
  if (!source)
    return { blocked: true, requiresEstimatedSupport: false, reason: null };
  if (source.brainMask) {
    if (
      source.brainSupport?.usableForResearchSimulation === true &&
      source.brainSupport.corticalAccessPermitted === false
    )
      return { blocked: false, requiresEstimatedSupport: false, reason: null };
    return {
      blocked: true,
      requiresEstimatedSupport: false,
      reason:
        "The supplied brain envelope needs source-bound review or an explicit research assumption.",
    };
  }
  if (
    source.metadata.structural_coverage === "full_head" ||
    source.metadata.allow_nonzero_mri_access_support === false
  )
    return {
      blocked: true,
      requiresEstimatedSupport: false,
      reason:
        "Full-head MRI requires reviewed research support before hypothetical route generation.",
    };
  return { blocked: false, requiresEstimatedSupport: true, reason: null };
}
