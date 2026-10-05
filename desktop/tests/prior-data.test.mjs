import assert from "node:assert/strict";
import { test } from "node:test";
import { createHash } from "node:crypto";
import {
  hydratePriorProposal,
  priorLabel,
  samplePriorAtCursor,
} from "../src/prior-data.ts";
const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const semantic = (bytes, dtype) =>
  "sha256:" +
  sha(
    Buffer.concat([
      Buffer.from(`{"dtype": "${dtype}", "shape": [2, 2, 2]}`),
      Buffer.from(bytes),
    ]),
  );
const hash = "sha256:" + "a".repeat(64),
  historical = "sha256:" + "b".repeat(64);
function fixture(kind = "functional_concordance") {
  const mri = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7]),
    values = new Float32Array(
      kind === "functional_concordance"
        ? [0, 0.25, 1, 0, 0, 0, 0, 0]
        : [0, 1, 1, 0, 0, 0, 0, 0],
    ),
    coverage = new Uint8Array([1, 1, 1, 1, 0, 0, 0, 0]);
  const bytes = (typed) =>
    new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength);
  const descriptor = (typed, dtype, id) => ({
    assetId: id,
    dtype,
    shape: [2, 2, 2],
    byteOrder: "little",
    order: "C",
    sha256: sha(bytes(typed)),
    byteLength: typed.byteLength,
  });
  const affine = [
    [2, 0, 0, 10],
    [0, 3, 0, -5],
    [0, 0, 4, 7],
    [0, 0, 0, 1],
  ];
  const frameHash =
    "sha256:" +
    sha(
      '{"affine":[[2.0,0.0,0.0,10.0],[0.0,3.0,0.0,-5.0],[0.0,0.0,4.0,7.0],[0.0,0.0,0.0,1.0]],"frame":"RAS+","physical_units":"mm","shape":[2,2,2]}',
    );
  const proposal = {
    proposalId: "prior",
    mapId: "phonology_" + kind,
    component: "phonology",
    mapKind: kind,
    title: "untrusted generic title",
    provenance: "prior",
    reviewStatus: "alignment_review_required",
    viewOnly: true,
    planningEligible: false,
    patientSpecificFunction: false,
    clinicalDeficitProbability: null,
    clinicalRiskReason: "no_validated_clinical_outcome_model",
    frame: "RAS+",
    spatialUnits: "mm",
    valueUnits: "unitless",
    affine: structuredClone(affine),
    shape: [2, 2, 2],
    data: descriptor(values, "float32", "values"),
    samplingCoverage: descriptor(coverage, "uint8", "coverage"),
    samplingCoverageFraction: 0.5,
    coverageMeaning: "atlas field of view, not patient functional coverage",
    interpolation: kind === "structural_mask" ? "nearest_neighbor" : "linear",
    evidenceHash: hash,
    registrationHash: hash,
    sourceImageHash: semantic(bytes(mri), "<f4"),
    sourceFrameHash: frameHash,
    sourcePlanningHash: hash,
    registrationCaseHash: historical,
    source: {
      source_id: "atlas-fixture",
      uri: "fixture",
      sha256: "c".repeat(64),
      native_frame: "FSL_MNI152_RAS_mm",
      provenance: "prior",
    },
    metadata: {},
    provenanceRecord: {},
  };
  const record = {
    evidence_hash: hash,
    registration_hash: hash,
    source_image_hash: proposal.sourceImageHash,
    source_frame_hash: frameHash,
    source_case_planning_hash: hash,
    registration_case_hash: historical,
    proposal_id: "prior",
    map_id: proposal.mapId,
    review: null,
    view_only: true,
    planning_eligible: false,
    patient_specific_function: false,
    clinical_deficit_probability: null,
    source_file_sha256: hash,
  };
  proposal.provenanceRecord = record;
  const source = {
    caseId: "fixture",
    caseHash: hash,
    planningHash: hash,
    frame: "RAS+",
    affine,
    shape: [2, 2, 2],
    spacingMm: [2, 3, 4],
    mri: descriptor(mri, "float32", "mri"),
    compartments: [],
    unknowns: [],
    metadata: { is_synthetic: true },
    sourceRefs: [{ source_id: "mri", uri: "fixture", sha256: "a".repeat(64) }],
    priorProposals: [proposal],
  };
  const viewer = {
    caseId: "fixture",
    caseHash: hash,
    frame: "RAS+",
    affine: structuredClone(affine),
    shape: [2, 2, 2],
    mri,
    compartments: [],
  };
  let reads = 0;
  const api = {
    readAsset: async (id) => {
      reads++;
      return bytes(id === "values" ? values : coverage);
    },
  };
  const reseal = () => {
    proposal.data.sha256 = sha(bytes(values));
    proposal.samplingCoverage.sha256 = sha(coverage);
    record.data_hash = semantic(bytes(values), "<f4");
    record.sampling_coverage_hash = semantic(coverage, "|b1");
  };
  reseal();
  return {
    source,
    viewer,
    proposal,
    record,
    values,
    coverage,
    api,
    reads: () => reads,
    reseal,
  };
}
test("one chosen prior loads values plus separate coverage, preserving historical registration identity and original arrays", async () => {
  const f = fixture(),
    result = await hydratePriorProposal(f.source, f.viewer, "prior", f.api);
  assert.equal(f.reads(), 2);
  assert.equal(result.title, "Phonology · functional network");
  assert.equal(result.proposal.registrationCaseHash, historical);
  assert.equal(result.scope, "view-only-population-prior");
  assert.equal(result.planningEligible, false);
  result.values[0] = 0.5;
  result.coverage[0] = 0;
  assert.equal(f.values[0], 0);
  assert.equal(f.coverage[0], 1);
  assert.deepEqual([...f.viewer.mri], [0, 1, 2, 3, 4, 5, 6, 7]);
});
test("covered zero, nonzero atlas concordance and uncovered unknown remain distinct at the physical cursor", async () => {
  const f = fixture(),
    view = await hydratePriorProposal(f.source, f.viewer, "prior", f.api);
  assert.deepEqual(samplePriorAtCursor(view, [10, -5, 7]), {
    covered: true,
    reason: null,
    value: 0,
    voxel: [0, 0, 0],
  });
  assert.deepEqual(samplePriorAtCursor(view, [10, -5, 11]), {
    covered: true,
    reason: null,
    value: 0.25,
    voxel: [0, 0, 1],
  });
  assert.deepEqual(samplePriorAtCursor(view, [12, -5, 7]), {
    covered: false,
    reason: "outside-atlas-coverage",
    value: null,
    voxel: [1, 0, 0],
  });
  assert.deepEqual(samplePriorAtCursor(view, [1000, 0, 0]), {
    covered: false,
    reason: "outside-atlas-coverage",
    value: null,
    voxel: null,
  });
});
test("explicit within-tolerance prior affine is preserved; mismatched physical grids are rejected", async () => {
  const f = fixture();
  f.proposal.affine[0][3] += 0.005;
  const view = await hydratePriorProposal(f.source, f.viewer, "prior", f.api);
  assert.equal(view.affine[0][3], 10.005);
  assert.equal(samplePriorAtCursor(view, [10.005, -5, 11]).value, 0.25);
  const bad = fixture();
  bad.proposal.affine[0][3] += 0.02;
  await assert.rejects(
    hydratePriorProposal(bad.source, bad.viewer, "prior", bad.api),
    /physical sampling grid/,
  );
  assert.equal(bad.reads(), 0);
});
test("stale planning identity and clinical or planning promotion fail before transfer", async () => {
  for (const mutate of [
    (f) => (f.proposal.sourcePlanningHash = historical),
    (f) => (f.proposal.planningEligible = true),
    (f) => (f.proposal.patientSpecificFunction = true),
    (f) => (f.proposal.clinicalDeficitProbability = 0.2),
    (f) => (f.proposal.reviewStatus = "accepted"),
    (f) => (f.record.review = { accepted: true }),
    (f) => (f.proposal.samplingCoverage.dtype = "float32"),
    (f) => (f.proposal.mapId = "merged_language"),
    (f) => (f.proposal.valueUnits = "probability"),
  ]) {
    const f = fixture();
    mutate(f);
    await assert.rejects(
      hydratePriorProposal(f.source, f.viewer, "prior", f.api),
    );
    assert.equal(f.reads(), 0);
  }
});
test("source or checksum corruption fails without producing an overlay", async () => {
  const f = fixture();
  f.proposal.sourceImageHash = historical;
  f.record.source_image_hash = historical;
  await assert.rejects(
    hydratePriorProposal(f.source, f.viewer, "prior", f.api),
    /another source MRI/,
  );
  assert.equal(f.reads(), 0);
  const g = fixture();
  g.values[1] = 0.3;
  await assert.rejects(
    hydratePriorProposal(g.source, g.viewer, "prior", g.api),
    /recorded evidence/,
  );
});
test("even rehashed malformed scalar/coverage combinations cannot masquerade as released evidence", async () => {
  for (const mutate of [
    (f) => (f.values[0] = NaN),
    (f) => (f.values[0] = 1.1),
    (f) => (f.coverage[0] = 2),
    (f) => (f.values[7] = 0.4),
  ]) {
    const f = fixture();
    mutate(f);
    f.reseal();
    await assert.rejects(
      hydratePriorProposal(f.source, f.viewer, "prior", f.api),
      /invalid semantics/,
    );
  }
  const f = fixture("structural_mask");
  f.values[0] = 0.5;
  f.reseal();
  await assert.rejects(
    hydratePriorProposal(f.source, f.viewer, "prior", f.api),
    /invalid semantics/,
  );
  const g = fixture();
  g.proposal.samplingCoverageFraction = 0.8;
  await assert.rejects(
    hydratePriorProposal(g.source, g.viewer, "prior", g.api),
    /coverage count/,
  );
});
test("cancelled prior selection cannot publish transferred data", async () => {
  const f = fixture(),
    controller = new AbortController();
  controller.abort();
  await assert.rejects(
    hydratePriorProposal(f.source, f.viewer, "prior", f.api, controller.signal),
    { name: "AbortError" },
  );
  assert.equal(f.reads(), 0);
  const pendingController = new AbortController();
  let starts = 0,
    started;
  const ready = new Promise((resolve) => (started = resolve)),
    releases = [];
  const pending = hydratePriorProposal(
    f.source,
    f.viewer,
    "prior",
    {
      readAsset: async (id) => {
        if (++starts === 2) started();
        return new Promise((resolve) =>
          releases.push(() =>
            resolve(
              new Uint8Array(
                id === "values" ? f.values.buffer : f.coverage.buffer,
              ),
            ),
          ),
        );
      },
    },
    pendingController.signal,
  );
  await ready;
  pendingController.abort();
  releases.forEach((release) => release());
  await assert.rejects(pending, { name: "AbortError" });
});
test("functional versus structural and speech-arrest labels remain distinct", () => {
  assert.equal(
    priorLabel({
      component: "speech_articulation",
      mapKind: "structural_mask",
    }),
    "Speech arrest / articulation · structural network mask",
  );
  assert.equal(
    priorLabel({ component: "semantics", mapKind: "functional_concordance" }),
    "Semantics · functional network",
  );
});

test("cursor concordance interpolates only with complete contributing atlas support", async () => {
  const f = fixture(),
    view = await hydratePriorProposal(f.source, f.viewer, "prior", f.api);
  assert.equal(samplePriorAtCursor(view, [10, -5, 9]).value, 0.125);
  assert.equal(samplePriorAtCursor(view, [11, -5, 9]).covered, false);
  assert.equal(samplePriorAtCursor(view, [11, -5, 9]).value, null);
  assert.equal(
    samplePriorAtCursor(view, [10.5, -5, 9]).reason,
    "incomplete-interpolation-support",
  );
  assert.equal(samplePriorAtCursor(view, [13, -5, 7]).voxel, null);
});
test("released structural membership uses nearest sampling without creating fractional tract values", async () => {
  const f = fixture("structural_mask"),
    view = await hydratePriorProposal(f.source, f.viewer, "prior", f.api);
  assert.equal(samplePriorAtCursor(view, [10, -5, 8]).value, 0);
  assert.equal(samplePriorAtCursor(view, [10, -5, 10]).value, 1);
});

for (const mode of ["in-place", "replacement"]) {
  test(`prior rechecks current MRI bytes after ${mode} mutation, before fetching another layer`, async () => {
    const f = fixture();
    await hydratePriorProposal(f.source, f.viewer, "prior", f.api);
    if (mode === "replacement") f.viewer.mri = f.viewer.mri.slice();
    f.viewer.mri[0] = 50;
    await assert.rejects(
      hydratePriorProposal(f.source, f.viewer, "prior", f.api),
      /another source MRI/,
    );
    assert.equal(f.reads(), 2);
  });
}

test("physical cursor delegates both sides of an outer atlas face to the shared precision convention", async () => {
  const f = fixture(),
    view = await hydratePriorProposal(f.source, f.viewer, "prior", f.api);
  for (const x of [13 - 1e-7, 13, 13 + 1e-7]) {
    const sample = samplePriorAtCursor(view, [x, -5, 7]);
    assert.equal(sample.covered, false);
    assert.equal(sample.value, null);
    assert.equal(sample.reason, "numerical-boundary-uncertainty");
  }
  assert.equal(
    samplePriorAtCursor(view, [13.1, -5, 7]).reason,
    "outside-atlas-coverage",
  );
});
