import assert from "node:assert/strict";
import { test } from "node:test";
import { hydrateCertifiedReplay } from "../src/training-data.ts";
const hash = `sha256:${"b".repeat(64)}`;
const source = {
  caseHash: hash,
  frame: "RAS+",
  shape: [2, 2, 2],
  affine: [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ],
};
function replay() {
  return {
    caseHash: hash,
    runId: "run",
    step: 1,
    stepCount: 1,
    role: "selection",
    finalEvaluation: false,
    frame: "RAS+",
    affine: source.affine.map((row) => [...row]),
    removedMask: {
      assetId: "mask",
      dtype: "uint8",
      shape: [2, 2, 2],
      byteOrder: "little",
      order: "C",
      sha256: "a".repeat(64),
      byteLength: 8,
    },
    clinicalDeficitProbability: null,
    simulatedRemovedTargetVolumeMm3: 1,
    simulatedRemovedNormalVolumeMm3: 0,
    modeledResidualTargetVolumeMm3: 1,
    training: {
      replay_status: "accepted_independent_geometry",
      replay: {
        native_certificate: {
          feasible: true,
          complete_tool_checked: true,
          frontier_checked: true,
          source_case_hash: hash,
        },
      },
    },
  };
}
const api = { readAsset: async () => new Uint8Array([0, 0, 0, 0, 0, 0, 0, 1]) };
test("accepted source-native selection mask hydrates without claiming final evidence", async () => {
  const result = await hydrateCertifiedReplay(replay(), source, api);
  assert.equal(result.mask[7], 1);
  assert.equal(result.result.finalEvaluation, false);
});
test("native replay rejects stale source, outcome claims and missing checks before transfer", async () => {
  for (const mutation of [
    (r) => (r.caseHash = "other"),
    (r) => (r.role = "final"),
    (r) => (r.finalEvaluation = true),
    (r) => (r.clinicalDeficitProbability = 0.2),
    (r) => (r.training.replay.native_certificate.feasible = false),
    (r) => (r.training.replay.native_certificate.frontier_checked = false),
    (r) => (r.training.replay_status = "pending"),
    (r) => (r.training.replay.native_certificate.source_case_hash = "other"),
  ]) {
    const result = replay();
    mutation(result);
    let reads = 0;
    await assert.rejects(
      hydrateCertifiedReplay(result, source, {
        readAsset: async () => {
          reads++;
          return new Uint8Array(8);
        },
      }),
    );
    assert.equal(reads, 0);
  }
});
test("native replay rejects spatial and volume-accounting corruption", async () => {
  for (const mutation of [
    (r) => (r.affine[0][3] = 1),
    (r) => (r.removedMask.order = "F"),
    (r) => (r.removedMask.shape = [1, 2, 4]),
    (r) => (r.removedMask.dtype = "int8"),
    (r) => (r.step = 2),
    (r) => (r.simulatedRemovedTargetVolumeMm3 = 1e9),
    (r) => (r.simulatedRemovedNormalVolumeMm3 = -1),
  ]) {
    const result = replay();
    mutation(result);
    await assert.rejects(hydrateCertifiedReplay(result, source, api));
  }
  await assert.rejects(
    hydrateCertifiedReplay(replay(), source, {
      readAsset: async () => new Uint8Array([0, 0, 0, 0, 0, 0, 0, 2]),
    }),
    /binary/,
  );
});
test("LPS source accepts correctly canonicalized RAS removal grid", async () => {
  const input = { ...source, frame: "LPS+" };
  const result = replay();
  result.affine[0][0] = -1;
  result.affine[1][1] = -1;
  assert.equal((await hydrateCertifiedReplay(result, input, api)).mask[7], 1);
});
