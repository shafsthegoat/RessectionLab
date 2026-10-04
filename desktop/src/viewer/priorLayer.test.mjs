import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
  validatePriorLayer,
  samplePriorVoxel,
  formatPriorValue,
  priorSamplingTolerance,
} from "./priorLayer.ts";
import { inverseAffine, rasAffine, transformPoint } from "./coordinates.ts";

function fixture(frame = "RAS+") {
  const c = Math.cos(Math.PI / 6),
    s = Math.sin(Math.PI / 6);
  const volume = {
    caseId: "atlas-source-grid",
    caseHash: "source-version-a",
    frame,
    affine: [
      [2 * c, -3 * s, 0, 10],
      [2 * s, 3 * c, 0, -20],
      [0, 0, 4, 7],
      [0, 0, 0, 1],
    ],
    shape: [2, 2, 2],
    mri: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]),
    compartments: [
      {
        name: "Source target",
        color: "#abcdef",
        mask: new Uint8Array([0, 0, 0, 1, 0, 0, 0, 0]),
        volumeMm3: 24,
      },
    ],
  };
  const layer = {
    caseHash: volume.caseHash,
    proposalId: "registered-prior-v1",
    mapId: "public-atlas",
    title: "Population motor concordance",
    component: "motor",
    mapKind: "functional_concordance",
    values: new Float32Array([0, 1, 0, 1, 0, 1, 0, 1]),
    coverage: new Uint8Array(8).fill(1),
    shape: [...volume.shape],
    affine: rasAffine(volume.affine, frame),
    frame: "RAS+",
    scope: "view-only-population-prior",
    provenance: "prior",
    reviewStatus: "alignment_review_required",
    planningEligible: false,
    patientSpecificFunction: false,
    valueUnits: "unitless",
    spatialUnits: "mm",
  };
  return { volume, layer };
}

test("tiny positive atlas values stay distinguishable from covered zero", () => {
  assert.equal(formatPriorValue(0), "0.0000");
  assert.equal(formatPriorValue(0.00014824711252003908), "0.0001");
  assert.equal(formatPriorValue(0.00001), "1.00e-5");
  assert.notEqual(Number(formatPriorValue(2 ** -149)), 0);
  assert.equal(formatPriorValue(0.5), "0.5000");
});

test("covered zero, interpolated atlas value and unavailable atlas support remain distinct", () => {
  const { layer } = fixture();
  assert.deepEqual(samplePriorVoxel(layer, [0, 0, 0]), {
    covered: true,
    value: 0,
    reason: null,
  });
  assert.deepEqual(samplePriorVoxel(layer, [0.5, 0.5, 0.5]), {
    covered: true,
    value: 0.5,
    reason: null,
  });
  // Only one positive-weight corner is uncovered; nearest cell is still known.
  layer.coverage[1] = 0;
  assert.deepEqual(samplePriorVoxel(layer, [0, 0, 0.25]), {
    covered: false,
    value: null,
    reason: "incomplete-interpolation-support",
  });
  assert.deepEqual(samplePriorVoxel(layer, [0, 0, 0]), {
    covered: true,
    value: 0,
    reason: null,
  });
  assert.deepEqual(samplePriorVoxel(layer, [0, 0, 1]), {
    covered: false,
    value: null,
    reason: "outside-atlas-coverage",
  });
  assert.deepEqual(samplePriorVoxel(layer, [0, 0, 1.5]), {
    covered: false,
    value: null,
    reason: "numerical-boundary-uncertainty",
  });
  assert.equal(samplePriorVoxel(layer, [NaN, 0, 0]).covered, false);
});

test("bounded float32 residue snaps to source centers without concealing genuinely missing support", () => {
  const { layer } = fixture();
  layer.coverage.set([1, 1, 1, 1, 0, 0, 0, 0]);
  const tolerance = priorSamplingTolerance(layer);
  assert.ok(
    tolerance.every(
      (value) => value > 0 && value <= 0.001 && Math.fround(value) === value,
    ),
  );
  assert.equal(
    priorSamplingTolerance(layer),
    tolerance,
    "Precision preparation is cached per immutable installed grid",
  );
  for (const x of [-tolerance[0] / 2, 0, tolerance[0] / 2]) {
    const sample = samplePriorVoxel(layer, [x, 0.3, 0.25]);
    assert.equal(sample.covered, true);
    assert.ok(Math.abs(sample.value - 0.25) < 1e-7);
  }
  assert.equal(
    samplePriorVoxel(layer, [2 * tolerance[0], 0.3, 0.25]).covered,
    false,
  );
  assert.equal(samplePriorVoxel(layer, [0.001, 0.3, 0.25]).covered, false);
  layer.coverage.fill(1);
  layer.coverage[6] = 0;
  // Positive diagonal support remains unknown even when its product is <1e-7.
  const point = [2 * tolerance[0], 2 * tolerance[1], 0];
  assert.ok(point[0] * point[1] < 1e-7);
  assert.equal(samplePriorVoxel(layer, point).covered, false);
});

test("binary half-cell ties are stable and outer-field precision bands abstain", () => {
  const { layer } = fixture();
  layer.mapKind = "structural_mask";
  const tolerance = priorSamplingTolerance(layer);
  for (const z of [0.5 - tolerance[2] / 2, 0.5, 0.5 + tolerance[2] / 2])
    assert.equal(samplePriorVoxel(layer, [0, 0, z]).value, 1);
  assert.equal(
    samplePriorVoxel(layer, [0, 0, 0.5 - 2 * tolerance[2]]).value,
    0,
  );
  for (const z of [
    -0.5 - tolerance[2] / 2,
    -0.5,
    -0.5 + tolerance[2] / 2,
    1.5 - tolerance[2] / 2,
    1.5,
    1.5 + tolerance[2] / 2,
  ])
    assert.deepEqual(samplePriorVoxel(layer, [0, 0, z]), {
      covered: false,
      value: null,
      reason: "numerical-boundary-uncertainty",
    });
  assert.equal(
    samplePriorVoxel(layer, [0, 0, -0.5 - 2 * tolerance[2]]).reason,
    "outside-atlas-coverage",
  );
  assert.equal(
    samplePriorVoxel(layer, [0, 0, -0.5 + 2 * tolerance[2]]).covered,
    true,
  );
});

test("precision budgets above the bound withhold the prior instead of increasing the snap band", () => {
  const { volume, layer } = fixture();
  layer.affine[0][3] = volume.affine[0][3] = 1e8;
  assert.equal(priorSamplingTolerance(layer), null);
  assert.throws(() => validatePriorLayer(volume, layer), /precision limit/);
  assert.deepEqual(samplePriorVoxel(layer, [0, 0, 0]), {
    covered: false,
    value: null,
    reason: "numerical-precision-unavailable",
  });
});

test("precision cache detects affine replacement, in-place edits and shape changes", () => {
  const { layer } = fixture();
  const original = layer.affine.map((row) => [...row]);
  assert.ok(priorSamplingTolerance(layer));
  layer.affine = original.map((row) => [...row]);
  layer.affine[0][3] = 1e8;
  assert.equal(
    samplePriorVoxel(layer, [0, 0, 0]).reason,
    "numerical-precision-unavailable",
  );
  layer.affine = original.map((row) => [...row]);
  assert.equal(samplePriorVoxel(layer, [0, 0, 0]).covered, true);
  layer.affine[0][3] = 1e8;
  assert.equal(
    samplePriorVoxel(layer, [0, 0, 0]).reason,
    "numerical-precision-unavailable",
  );
  layer.affine = original;
  assert.ok(priorSamplingTolerance(layer));
  layer.shape[0] = 1e6;
  assert.equal(priorSamplingTolerance(layer), null);
});

test("released binary atlas masks use nearest cells without creating fractional membership", () => {
  const { volume, layer } = fixture();
  layer.mapKind = "structural_mask";
  layer.coverage[1] = 0;
  validatePriorLayer(volume, layer);
  assert.deepEqual(samplePriorVoxel(layer, [0, 0, 0.49]), {
    covered: true,
    value: 0,
    reason: null,
  });
  assert.deepEqual(samplePriorVoxel(layer, [0, 0, 0.51]), {
    covered: false,
    value: null,
    reason: "outside-atlas-coverage",
  });
  layer.coverage[1] = 1;
  assert.deepEqual(samplePriorVoxel(layer, [0, 0, 0.51]), {
    covered: true,
    value: 1,
    reason: null,
  });
});

test("explicit registered affine preserves oblique and LPS physical landmarks", () => {
  for (const frame of ["RAS+", "LPS+"]) {
    const { volume, layer } = fixture(frame);
    layer.affine[2][3] += 0.009;
    validatePriorLayer(volume, layer);
    const registeredPoint = transformPoint(layer.affine, [0, 1, 0.25]);
    const priorVoxel = transformPoint(
      inverseAffine(layer.affine),
      registeredPoint,
    );
    const sourceVoxel = transformPoint(
      inverseAffine(rasAffine(volume.affine, frame)),
      registeredPoint,
    );
    assert.ok(Math.abs(priorVoxel[2] - 0.25) < 1e-12);
    assert.ok(Math.abs(sourceVoxel[2] - 0.25225) < 1e-12);
    assert.ok(
      Math.abs(samplePriorVoxel(layer, priorVoxel).value - 0.25) < 1e-12,
    );
    layer.affine[2][3] += 0.0011;
    assert.throws(() => validatePriorLayer(volume, layer), /tolerance/);
  }
});

test("display gate rejects stale grids, invalid values, and promotion to patient-specific planning evidence", () => {
  const changes = [
    { caseHash: "another-source" },
    { proposalId: "" },
    { frame: "LPS+" },
    { scope: "working-anatomy" },
    { provenance: "measured" },
    { reviewStatus: "accepted" },
    { planningEligible: true },
    { patientSpecificFunction: true },
    { valueUnits: "probability" },
    { spatialUnits: "cm" },
    { mapKind: "clinical_injury" },
    { shape: [4, 1, 2] },
    { values: new Float32Array(7) },
    { coverage: new Uint8Array(7) },
    { values: new Float64Array(8) },
    { coverage: new Uint8Array(8).fill(2) },
    { values: new Float32Array(8).fill(NaN) },
    { values: new Float32Array(8).fill(-0.01) },
    { values: new Float32Array(8).fill(1.01) },
    { mapKind: "structural_mask", values: new Float32Array(8).fill(0.5) },
  ];
  for (const change of changes) {
    const { volume, layer } = fixture();
    assert.throws(
      () => validatePriorLayer(volume, { ...layer, ...change }),
      JSON.stringify(change),
    );
  }
  const { volume, layer } = fixture("LPS+");
  layer.affine = volume.affine;
  assert.throws(() => validatePriorLayer(volume, layer), /tolerance/);
});

test("prior validation and sampling leave original MRI, targets, metrics and saved atlas bytes unchanged", () => {
  const { volume, layer } = fixture();
  const digest = (array) =>
    createHash("sha256")
      .update(new Uint8Array(array.buffer, array.byteOffset, array.byteLength))
      .digest("hex");
  const state = () => ({
    mri: digest(volume.mri),
    target: digest(volume.compartments[0].mask),
    targetVolume: volume.compartments[0].volumeMm3,
    compartments: volume.compartments.length,
    values: digest(layer.values),
    coverage: digest(layer.coverage),
    affine: JSON.stringify(layer.affine),
  });
  const before = state();
  validatePriorLayer(volume, layer);
  for (const point of [
    [0, 0, 0],
    [0.1, 0.5, 0.8],
    [1, 1, 1],
    [-10, 0, 0],
  ])
    samplePriorVoxel(layer, point);
  assert.deepEqual(state(), before);
});
