import assert from "node:assert/strict";
import { test } from "node:test";
import { researchSupportGate } from "../src/case-support.ts";
test("unreviewed estimated whole-brain proposals cannot unlock full-head access", () => {
  const source = {
    brainMask: null,
    brainSupport: { usableForResearchSimulation: false },
    metadata: {
      structural_coverage: "full_head",
      allow_nonzero_mri_access_support: false,
    },
    structuralEvidence: [
      { reviewStatus: "review_required", provenance: "estimated" },
    ],
  };
  assert.equal(researchSupportGate(source).blocked, true);
  assert.equal(researchSupportGate(source).requiresEstimatedSupport, false);
});
test("supplied masks without bound review remain blocked, including legacy payloads", () => {
  for (const source of [
    { brainMask: {}, metadata: {} },
    {
      brainMask: {},
      brainSupport: { usableForResearchSimulation: false },
      metadata: {},
    },
  ])
    assert.equal(researchSupportGate(source).blocked, true);
});
test("backend-approved hypothetical support permits research but does not infer cortex permission", () => {
  const source = {
    brainMask: {},
    brainSupport: {
      usableForResearchSimulation: true,
      corticalAccessPermitted: false,
    },
    metadata: { structural_coverage: "full_head" },
  };
  assert.deepEqual(researchSupportGate(source), {
    blocked: false,
    requiresEstimatedSupport: false,
    reason: null,
  });
});
test("intracranial structural mode keeps an explicit estimated-support opt-in", () => {
  assert.deepEqual(
    researchSupportGate({
      brainMask: null,
      metadata: { structural_coverage: "intracranial" },
    }),
    { blocked: false, requiresEstimatedSupport: true, reason: null },
  );
});

test("a contradictory cortical permission claim cannot unlock research support", () => {
  assert.equal(
    researchSupportGate({
      brainMask: {},
      brainSupport: {
        usableForResearchSimulation: true,
        corticalAccessPermitted: true,
      },
      metadata: {},
    }).blocked,
    true,
  );
});
