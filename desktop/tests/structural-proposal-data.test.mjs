import assert from "node:assert/strict";
import { test } from "node:test";
import { createHash } from "node:crypto";
import {
  hydrateStructuralProposal,
  pythonAffineFloat,
} from "../src/structural-proposal-data.ts";
const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const semantic = (bytes, dtype) =>
  "sha256:" +
  sha(
    Buffer.concat([
      Buffer.from(`{"dtype": "${dtype}", "shape": [2, 2, 2]}`),
      Buffer.from(bytes),
    ]),
  );
function fixture(frame = "RAS+") {
  const mri = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7]);
  const mriBytes = new Uint8Array(mri.buffer),
    mask = new Uint8Array([1, 1, 1, 1, 1, 1, 1, 0]);
  const affine = [
    [2, 0, 0, 10],
    [0, 3, 0, -5],
    [0, 0, 4, 7],
    [0, 0, 0, 1],
  ];
  const frameJson = `{"affine":[[2.0,0.0,0.0,10.0],[0.0,3.0,0.0,-5.0],[0.0,0.0,4.0,7.0],[0.0,0.0,0.0,1.0]],"frame":"${frame}","physical_units":"mm","shape":[2,2,2]}`;
  const descriptor = (bytes, dtype, id) => ({
    assetId: id,
    dtype,
    shape: [2, 2, 2],
    byteOrder: "little",
    order: "C",
    sha256: sha(bytes),
    byteLength: bytes.length,
  });
  const hash = "sha256:" + "a".repeat(64);
  const proposal = {
    evidenceId: "main",
    kind: "whole_brain_envelope",
    provenance: "estimated",
    reviewStatus: "review_required",
    reviewRequired: true,
    corticalAccessPermitted: false,
    sourceHash: semantic(mriBytes, "<f4"),
    sourceFrameHash: "sha256:" + sha(frameJson),
    sourceFileHash: null,
    maskHash: semantic(mask, "|b1"),
    modelHash: hash,
    runHash: hash,
    evidenceHash: hash,
    method: "fixture",
    array: descriptor(mask, "uint8", "proposal"),
    metadata: { variant: "main" },
    review: null,
  };
  const target = new Uint8Array([0, 0, 0, 0, 0, 0, 0, 1]);
  const source = {
    caseId: "fixture",
    caseHash: hash,
    frame,
    affine,
    shape: [2, 2, 2],
    spacingMm: [2, 3, 4],
    mri: descriptor(mriBytes, "float32", "mri"),
    compartments: [
      {
        name: "target",
        volumeMm3: 24,
        array: descriptor(target, "uint8", "target"),
      },
    ],
    unknowns: [],
    metadata: {},
    structuralEvidence: [proposal],
  };
  const viewer = {
    caseId: "fixture",
    caseHash: hash,
    frame: "RAS+",
    affine: affine.map((row, r) =>
      row.map((n) => (frame === "LPS+" && r < 2 ? -n : n)),
    ),
    shape: [2, 2, 2],
    mri,
    compartments: [
      { name: "target", volumeMm3: 24, mask: target, color: "orange" },
      { name: "overlapping", mask: target.slice() },
    ],
  };
  let reads = 0;
  const api = {
    readAsset: async (id) => {
      reads++;
      assert.equal(id, "proposal");
      return mask;
    },
  };
  return { source, viewer, proposal, mask, api, reads: () => reads };
}
test("chosen proposal is transferred lazily, source-bound, binary and display-only; outside annotations use source union", async () => {
  const f = fixture(),
    original = f.viewer.compartments[0].mask.slice();
  const result = await hydrateStructuralProposal(
    f.source,
    f.viewer,
    "main",
    f.api,
  );
  assert.equal(f.reads(), 1);
  assert.equal(result.scope, "display-only-estimate");
  assert.equal(result.estimatedVoxelCount, 7);
  assert.equal(result.annotationOutsideVoxelCount, 1);
  assert.deepEqual(result.outsideAnnotationPointMm, [12, -2, 11]);
  assert.deepEqual(result.voxelBounds, { min: [0, 0, 0], max: [1, 1, 1] });
  result.mask[0] = 0;
  assert.equal(f.mask[0], 1);
  assert.deepEqual(f.viewer.compartments[0].mask, original);
  assert.equal(f.source.brainMask, undefined);
});
test("LPS proposal is bound in its original frame and displayed in canonical RAS without resampling", async () => {
  const f = fixture("LPS+"),
    result = await hydrateStructuralProposal(f.source, f.viewer, "main", f.api);
  assert.deepEqual(result.outsideAnnotationPointMm, [-12, 2, 11]);
  assert.deepEqual(result.mask, f.mask);
  assert.equal(result.frame, "RAS+");
});
test("stale case, frame, source image and unsupported permission claims fail before mask transfer", async () => {
  for (const mutate of [
    (f) => (f.viewer.caseHash = "sha256:" + "b".repeat(64)),
    (f) => f.viewer.affine[0][3]++,
    (f) => (f.proposal.sourceHash = "sha256:" + "b".repeat(64)),
    (f) => (f.proposal.sourceFrameHash = "sha256:" + "b".repeat(64)),
    (f) => (f.proposal.corticalAccessPermitted = true),
    (f) => (f.proposal.reviewStatus = "unverified"),
    (f) => (f.proposal.array.shape = [8, 1, 1]),
    (f) => (f.proposal.array.dtype = "float32"),
  ]) {
    const f = fixture();
    mutate(f);
    await assert.rejects(
      hydrateStructuralProposal(f.source, f.viewer, "main", f.api),
    );
    assert.equal(f.reads(), 0);
  }
});
test("unknown or duplicate proposal identity is never fetched", async () => {
  const f = fixture();
  await assert.rejects(
    hydrateStructuralProposal(f.source, f.viewer, "other", f.api),
  );
  f.source.structuralEvidence.push(f.proposal);
  await assert.rejects(
    hydrateStructuralProposal(f.source, f.viewer, "main", f.api),
  );
  assert.equal(f.reads(), 0);
});
test("same-length corruption and a mismatched mask evidence digest are rejected", async () => {
  const f = fixture();
  f.mask[0] = 0;
  await assert.rejects(
    hydrateStructuralProposal(f.source, f.viewer, "main", f.api),
    /checksum/,
  );
  const g = fixture();
  g.proposal.maskHash = "sha256:" + "b".repeat(64);
  await assert.rejects(
    hydrateStructuralProposal(g.source, g.viewer, "main", g.api),
    /checksum/,
  );
});
test("even checksummed empty or non-binary proposals fail the cell gate", async () => {
  for (const bytes of [
    new Uint8Array(8),
    new Uint8Array([2, 0, 0, 0, 0, 0, 0, 0]),
  ]) {
    const f = fixture();
    f.proposal.array.sha256 = sha(bytes);
    f.proposal.maskHash = semantic(bytes, "|b1");
    await assert.rejects(
      hydrateStructuralProposal(f.source, f.viewer, "main", {
        readAsset: async () => bytes,
      }),
      /no estimated|non-binary/,
    );
  }
});
test("cancellation before transfer and while waiting cannot publish a proposal", async () => {
  const f = fixture(),
    cancelled = new AbortController();
  cancelled.abort();
  await assert.rejects(
    hydrateStructuralProposal(
      f.source,
      f.viewer,
      "main",
      f.api,
      cancelled.signal,
    ),
    { name: "AbortError" },
  );
  assert.equal(f.reads(), 0);
  const controller = new AbortController();
  let started;
  const ready = new Promise((resolve) => {
    started = resolve;
  });
  let release;
  const pending = hydrateStructuralProposal(
    f.source,
    f.viewer,
    "main",
    {
      readAsset: async () => {
        started();
        return new Promise((resolve) => {
          release = resolve;
        });
      },
    },
    controller.signal,
  );
  await ready;
  controller.abort();
  release(f.mask);
  await assert.rejects(pending, { name: "AbortError" });
});
test("Python affine serialization preserves negative zero and scientific exponent conventions", () => {
  assert.deepEqual(
    [0, -0, 1, 1e-5, 1e16, 1e-4, -12.25].map(pythonAffineFloat),
    ["0.0", "-0.0", "1.0", "1e-05", "1e+16", "0.0001", "-12.25"],
  );
});

test("a reviewed or rejected legend requires a matching limited-scope evidence record", async () => {
  const f = fixture();
  f.proposal.reviewRequired = false;
  f.proposal.reviewStatus = "accepted";
  await assert.rejects(
    hydrateStructuralProposal(f.source, f.viewer, "main", f.api),
    /review is not bound/,
  );
  f.proposal.review = {
    evidence_hash: f.proposal.evidenceHash,
    decision: "accepted",
    scope: "research_brain_envelope_only",
  };
  assert.equal(
    (await hydrateStructuralProposal(f.source, f.viewer, "main", f.api))
      .reviewStatus,
    "accepted",
  );
  f.proposal.review.scope = "clinical_clearance";
  await assert.rejects(
    hydrateStructuralProposal(f.source, f.viewer, "main", f.api),
    /review is not bound/,
  );
});

for (const mode of ["in-place", "replacement"]) {
  test(`structural estimate rechecks current MRI bytes after ${mode} mutation, before mask transfer`, async () => {
    const f = fixture();
    await hydrateStructuralProposal(f.source, f.viewer, "main", f.api);
    if (mode === "replacement") f.viewer.mri = f.viewer.mri.slice();
    f.viewer.mri[0] = 50;
    await assert.rejects(
      hydrateStructuralProposal(f.source, f.viewer, "main", f.api),
      /another source image/,
    );
    assert.equal(f.reads(), 1);
  });
}
