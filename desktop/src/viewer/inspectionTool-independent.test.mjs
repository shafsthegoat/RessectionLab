import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import * as THREE from "three";
import * as inspection from "./inspectionTool.ts";
import { inspectionToolMeshes } from "./inspectionToolGeometry.ts";
import { neighboringFixture, digest } from "../../tests/neighboring-paths-fixture.mjs";

// Synthetic contract + actual renderer methods with CPU THREE objects only.
// No WebGL context, patient data, numerical engine, or certificate generation.
function fixture() {
  const f = neighboringFixture(), report = f.result.inspection;
  report.binding.decision_model_hash = digest("5");
  report.actions[1].native_preview.native_footprint = "fully_contained_connected_cells_v1";
  const volume = { caseId: "synthetic-contract", caseHash: f.context.caseHash,
    planningHash: f.context.planningHash, frame: "RAS+",
    affine: structuredClone(report.binding.native_affine_ras_mm),
    shape: [...report.binding.source_shape], mri: new Float32Array(392), compartments: [] };
  const input = { scope: "unexecuted-native-axis-inspection", caseHash: volume.caseHash,
    planningHash: volume.planningHash, bindingHash: report.binding.binding_hash,
    inspectionHash: report.inspection_hash, actionId: report.actions[1].action_id, report };
  return { volume, input, report };
}

let server, cache, VolumeRenderer;
before(async () => {
  cache = await fs.mkdtemp(path.join(os.tmpdir(), "inspection-renderer-independent-"));
  server = await createServer({ root: fileURLToPath(new URL("../../", import.meta.url)),
    cacheDir: cache, logLevel: "error", server: { middlewareMode: true, hmr: false, ws: false }, appType: "custom" });
  VolumeRenderer = (await server.ssrLoadModule("/src/viewer/VolumeRenderer.ts")).VolumeRenderer;
});
after(async () => { await server?.close(); if (cache) await fs.rm(cache, { recursive: true, force: true }); });

function renderer(volume) {
  const r = Object.create(VolumeRenderer.prototype);
  const shaders = Array.from({ length: 4 }, () => ({ uniforms: {
    uRouteCount: { value: 0 }, uReplayActive: { value: 0 }, uRemoved: { value: null },
    ...Object.fromEntries(["uShaftStart", "uShaftEnd", "uTipEnd"].map((key) =>
      [key, { value: [new THREE.Vector3(), new THREE.Vector3()] }])),
    uRadii: { value: [new THREE.Vector2(), new THREE.Vector2()] },
    uRouteColors: { value: [new THREE.Color(), new THREE.Color()] },
  } }));
  Object.assign(r, { volume, instrumentDisplay: new inspection.InstrumentDisplayState(),
    tools: new THREE.Group(), inspectionTools: new THREE.Group(), anatomy: new THREE.Group(),
    replayGroup: new THREE.Group(), replayGeneration: 0, replayWorker: null,
    pendingReplayGroup: null, replayActive: false, disposed: false,
    removedTexture: new THREE.Data3DTexture(new Uint8Array(1), 1, 1, 1),
    materials: () => shaders, requestRender() {}, onSurfaceStatus() {}, onReplayError() {},
    setStructuralProposal() {}, setPriorLayer() {}, mode: "anatomy" });
  return { r, shaders };
}
const routeA = { shaftStart: [0, 0, -20], shaftEnd: [0, 0, 1], tip: [0, 0, 3], shaftRadius: .5, tipRadius: 1, color: "#a3e5d3" };
const routeB = { shaftStart: [8, 0, -20], shaftEnd: [8, 0, 1], tip: [8, 0, 3], shaftRadius: .75, tipRadius: 2, color: "#e5c598" };

test("inspection requires available planning identity even when case and grid agree", () => {
  const f = fixture(); delete f.volume.planningHash;
  assert.throws(() => inspection.validateInspectionTool(f.volume, f.input), /planning|source/);
});

test("same-identity cloned wrappers cannot resurrect cleared or replay-suppressed inspection", () => {
  const f = fixture(), gate = new inspection.InspectionSelectionGate();
  assert.equal(gate.select(f.input, false), f.input);
  gate.clear(f.input);
  assert.equal(gate.select(structuredClone(f.input), false), null);
  assert.equal(gate.select(null, false), null); // Explicit deselection permits a later deliberate reselection.
  assert.notEqual(gate.select(structuredClone(f.input), false), null);
  assert.equal(gate.select(f.input, true), null);
  assert.equal(gate.select(structuredClone(f.input), false), null);
  const distinct = structuredClone(f.input); distinct.inspectionHash = digest("6");
  assert.equal(gate.select(distinct, false), distinct);
});

test("actual renderer gives all MRI planes one inspection capsule pair, preserves A/B, and clears invalid replacement", () => {
  const f = fixture(), { r, shaders } = renderer(f.volume);
  r.instrumentDisplay.setRoutes([routeA, routeB]);
  const display = r.setInspectionTool(f.input);
  assert.equal(r.tools.visible, false); assert.equal(r.inspectionTools.visible, true);
  for (const shader of shaders) {
    assert.equal(shader.uniforms.uRouteCount.value, 1);
    assert.deepEqual(shader.uniforms.uShaftStart.value[0].toArray(), display.shaftStart);
    assert.deepEqual(shader.uniforms.uShaftEnd.value[0].toArray(), display.shaftEnd);
    assert.deepEqual(shader.uniforms.uTipEnd.value[0].toArray(), display.tip);
    assert.deepEqual(shader.uniforms.uRadii.value[0].toArray(), [.45, 1.25]);
  }
  // Comparison state can legitimately change while its meshes are hidden.
  r.instrumentDisplay.setRoutes([routeB, routeA]);
  assert.throws(() => r.setInspectionTool({ ...f.input, caseHash: digest("0") }), /withheld/);
  assert.equal(r.inspectionTools.children.length, 0); assert.equal(r.inspectionTools.visible, false);
  assert.equal(r.tools.visible, true);
  for (const shader of shaders) {
    assert.equal(shader.uniforms.uRouteCount.value, 2);
    assert.deepEqual(shader.uniforms.uTipEnd.value[0].toArray(), routeB.tip);
    assert.deepEqual(shader.uniforms.uTipEnd.value[1].toArray(), routeA.tip);
  }
});

test("actual replay methods suppress both mesh and MRI inspection and cannot restore it when replay clears", () => {
  const f = fixture(), { r, shaders } = renderer(f.volume);
  r.instrumentDisplay.setRoutes([routeA, routeB]); r.setInspectionTool(f.input);
  const savedWorker = globalThis.Worker;
  globalThis.Worker = class { postMessage() {} terminate() {} };
  try {
    r.setReplay({ independentlyAccepted: true, scope: "native-source-grid", caseHash: f.volume.caseHash,
      shape: f.volume.shape, affine: f.volume.affine, step: 0, stepCount: 0,
      removedMask: new Uint8Array(392), removedTargetVolumeMm3: 0,
      removedNormalVolumeMm3: 0, residualTargetVolumeMm3: 0 });
    assert.equal(r.inspectionTools.children.length, 0); assert.equal(r.inspectionTools.visible, false);
    assert.equal(r.tools.visible, false);
    assert.ok(shaders.every((shader) => shader.uniforms.uRouteCount.value === 0));
    assert.equal(r.setInspectionTool(structuredClone(f.input)), null);
    r.setReplay(null);
    assert.equal(r.instrumentDisplay.inspected, null); assert.equal(r.inspectionTools.visible, false);
    assert.ok(shaders.every((shader) => shader.uniforms.uRouteCount.value === 2));
  } finally { globalThis.Worker = savedWorker; r.removedTexture.dispose(); }
});

test("rejected replay replacement also clears the earlier unexecuted tool", () => {
  const f = fixture(), { r, shaders } = renderer(f.volume);
  r.instrumentDisplay.setRoutes([routeA]); r.setInspectionTool(f.input);
  assert.throws(() => r.setReplay({ independentlyAccepted: false }), /independent/);
  assert.equal(r.instrumentDisplay.inspected, null);
  assert.equal(r.inspectionTools.children.length, 0);
  assert.equal(r.inspectionTools.visible, false);
  assert.ok(shaders.every((shader) => shader.uniforms.uRouteCount.value === 1));
  r.removedTexture.dispose();
});

test("selected-tool geometry warning does not inherit another preview's report-level warning", () => {
  const f = fixture();
  f.report.actions[1].native_preview.geometry_unknowns = [];
  assert.ok(f.report.unknowns.includes("tool_geometry_outside_image_unassessed"));
  const display = inspection.validateInspectionTool(f.volume, f.input);
  assert.deepEqual(display.unknowns, []);
  assert.ok(f.report.unknowns.includes("tool_geometry_outside_image_unassessed"),
    "The full inspection's separate uncertainty record must remain unchanged");
});

test("oblique LPS fixture creates the full physical capsule envelope without a second frame conversion", () => {
  const f = fixture(), axis = [3 / 13, 4 / 13, 12 / 13], entry = [20, -10, 30], tip = [26, -2, 54];
  const affine = [[4 / 5, 36 / 65, axis[0], 17], [-3 / 5, 48 / 65, axis[1], -31], [0, -5 / 13, axis[2], 9], [0, 0, 0, 1]];
  f.report.binding.native_affine_ras_mm = affine;
  f.volume.frame = "LPS+";
  f.volume.affine = affine.map((row, i) => row.map((x) => i < 2 ? -x : x));
  const action = f.report.actions[1], proposal = f.report.inventory.batch.proposals[0];
  proposal.entry_mm = entry; proposal.fallback_target_mm = tip;
  Object.assign(f.report.inventory.attempts[1], { entry_mm: entry, tip_mm: tip });
  Object.assign(action.geometry, { entry_mm: entry, tip_mm: tip, axis_unit: axis, insertion_distance_mm: 26 });
  const display = inspection.validateInspectionTool(f.volume, f.input), group = inspectionToolMeshes(display);
  assert.deepEqual(display.tip, tip);
  group.updateMatrixWorld(true);
  for (const [index, mesh] of group.children.entries()) {
    const length = index === 0 ? 118 : 2, radius = index === 0 ? .45 : 1.25;
    const farEnd = index === 0 ? display.shaftEnd : display.tip;
    const localCap = new THREE.Vector3(0, length / 2 + radius, 0).applyMatrix4(mesh.matrixWorld);
    const expected = farEnd.map((value, i) => value + radius * axis[i]);
    assert.ok(localCap.distanceTo(new THREE.Vector3(...expected)) < 1e-12);
    mesh.geometry.dispose(); mesh.material.dispose();
  }
});
