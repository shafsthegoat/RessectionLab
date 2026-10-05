import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
  validateStructuralProposal,
  structuralProposalReviewLabel,
} from "./structuralProposal.ts";
import {
  cIndex,
  inverseAffine,
  rasAffine,
  transformPoint,
} from "./coordinates.ts";

function fixture(frame = "RAS+") {
  const c = Math.cos(Math.PI / 6),
    s = Math.sin(Math.PI / 6);
  const volume = {
    caseId: "source-grid-landmark",
    caseHash: "source-version-a",
    frame,
    affine: [
      [2 * c, -3 * s, 0, 10],
      [2 * s, 3 * c, 0, -20],
      [0, 0, 4, 7],
      [0, 0, 0, 1],
    ],
    shape: [3, 4, 5],
    mri: new Float32Array(60).map((_, i) => i),
    compartments: [
      {
        name: "Target",
        color: "#abcdef",
        mask: new Uint8Array(60),
        volumeMm3: 24,
      },
    ],
  };
  volume.compartments[0].mask[cIndex(volume.shape, [1, 2, 3])] = 1;
  const proposal = {
    caseHash: volume.caseHash,
    evidenceId: "estimated-envelope-v1",
    frame: "RAS+",
    affine: rasAffine(volume.affine, volume.frame),
    shape: [...volume.shape],
    mask: new Uint8Array(60),
    reviewStatus: "review_required",
    provenance: "estimated",
    label: "Envelope proposal",
    scope: "display-only-estimate",
  };
  proposal.mask[cIndex(volume.shape, [2, 1, 4])] = 1;
  return { volume, proposal };
}

test("Estimated envelope keeps the native oblique source-grid landmark in RAS and LPS", () => {
  for (const frame of ["RAS+", "LPS+"]) {
    const { volume, proposal } = fixture(frame);
    validateStructuralProposal(volume, proposal);
    const rasPoint = transformPoint(proposal.affine, [2, 1, 4]);
    const sampledVoxel = transformPoint(
      inverseAffine(rasAffine(volume.affine, volume.frame)),
      rasPoint,
    );
    sampledVoxel.forEach((value, axis) =>
      assert.ok(Math.abs(value - [2, 1, 4][axis]) < 1e-10),
    );
    assert.equal(
      proposal.mask[cIndex(volume.shape, sampledVoxel.map(Math.round))],
      1,
    );
    assert.equal(
      volume.compartments[0].mask[
        cIndex(volume.shape, sampledVoxel.map(Math.round))
      ],
      0,
    );
  }
});

test("Inspecting an estimate does not alter source MRI, target masks, or target metrics", () => {
  const { volume, proposal } = fixture();
  const bytes = (array) =>
    new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
  const digest = (array) =>
    createHash("sha256").update(bytes(array)).digest("hex");
  const before = {
    mri: digest(volume.mri),
    target: digest(volume.compartments[0].mask),
    proposal: digest(proposal.mask),
    volume: volume.compartments[0].volumeMm3,
    count: volume.compartments.length,
  };
  for (const status of ["review_required", "accepted", "rejected"]) {
    proposal.reviewStatus = status;
    validateStructuralProposal(volume, proposal);
  }
  assert.deepEqual(
    {
      mri: digest(volume.mri),
      target: digest(volume.compartments[0].mask),
      proposal: digest(proposal.mask),
      volume: volume.compartments[0].volumeMm3,
      count: volume.compartments.length,
    },
    before,
  );
});

test("Estimate display rejects stale, misaligned, nonbinary, or promoted evidence", () => {
  const invalid = [
    { caseHash: "different-case" },
    { evidenceId: "" },
    { frame: "LPS+" },
    { scope: "working-anatomy" },
    { provenance: "supplied" },
    { reviewStatus: "approved_for_surgery" },
    { shape: [5, 4, 3] },
    { mask: new Uint8Array(59) },
    { mask: new Float32Array(60) },
    { mask: new Uint8Array(60) },
    { mask: new Uint8Array(60).fill(2) },
  ];
  for (const change of invalid) {
    const { volume, proposal } = fixture();
    assert.throws(() =>
      validateStructuralProposal(volume, { ...proposal, ...change }),
    );
  }
  const { volume, proposal } = fixture("LPS+");
  proposal.affine = volume.affine.map((row) => [...row]);
  assert.throws(
    () => validateStructuralProposal(volume, proposal),
    /affine differs/,
  );
  proposal.affine = rasAffine(volume.affine, volume.frame);
  proposal.affine[0][3] += 0.001;
  assert.throws(
    () => validateStructuralProposal(volume, proposal),
    /affine differs/,
  );
});

test("Canonical estimate labels keep accepted and rejected review states distinct", () => {
  assert.equal(
    structuralProposalReviewLabel("review_required"),
    "Review required",
  );
  assert.equal(structuralProposalReviewLabel("rejected"), "Rejected estimate");
  assert.equal(
    structuralProposalReviewLabel("accepted"),
    "Reviewed envelope, display only",
  );
  const { volume, proposal } = fixture();
  proposal.label = "Accepted for cortical access";
  validateStructuralProposal(volume, proposal);
  assert.equal(
    structuralProposalReviewLabel(proposal.reviewStatus),
    "Review required",
    "Free-form descriptions cannot relabel evidence status",
  );
  assert.throws(
    () => structuralProposalReviewLabel("accepted_for_cortical_access"),
    /unresolved/,
  );
});
