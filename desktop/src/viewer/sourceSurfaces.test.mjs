import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {ExposureBodyTransport} from './exposureBodySurface.ts';
import {sourceCenter} from './exposureAnatomy.ts';
import {bindToBody, kerfWall, normalsToVoxel, PlanarLoop, rasToVoxel, splitSheet, transportVertices, voxelToRas} from './sourceSurfaceGeometry.ts';
import {CAVITY_LEVEL, cavityField, cavitySurface, distanceToRemovedBoundary, removalOccupancy, sampleCavity} from './sourceCavityField.ts';
import {cutOutlinePoints, OperativeSourceScene} from './operativeSourceScene.ts';
import {checkedDescriptor, validateSourceAsset} from '../operative-source-surfaces.ts';

// An oblique working grid like the host rehearsal's: rotated, 1 mm cells.
const angle = 0.6, c = Math.cos(angle), s = Math.sin(angle);
const affine = [[c, -s, 0, 12], [s, c, 0, -7], [0, 0, 1, 30], [0, 0, 0, 1]];
const shape = [24, 22, 20];
const id = (x, y, z) => (x * shape[1] + y) * shape[2] + z;

function sheet(n = 21, z = 10.4) {
  const positions = [], normals = [], triangles = [];
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { positions.push(1 + i, 1 + j, z); normals.push(0, 0, 1); }
  for (let i = 0; i + 1 < n; i++) for (let j = 0; j + 1 < n; j++) {
    const a = i * n + j, b = a + n;
    triangles.push(a, b, a + 1, a + 1, b, b + 1);
  }
  return {positions: Float32Array.from(positions), normals: Float32Array.from(normals), triangles: Uint32Array.from(triangles)};
}

test('RAS and voxel conversions round trip on an oblique grid and keep normals', () => {
  const voxel = Float32Array.from([3.2, 4.5, 6.1, 0, 0, 0]);
  const ras = voxelToRas(voxel, affine);
  assert.ok(rasToVoxel(ras, affine).every((v, i) => Math.abs(v - voxel[i]) < 1e-5));
  // A RAS normal maps to the voxel normal that the source-frame object matrix maps back.
  const n = normalsToVoxel(Float32Array.from([s, c, 0]), affine);
  const back = new THREE.Vector3(...n).applyMatrix3(new THREE.Matrix3().getNormalMatrix(new THREE.Matrix4().fromArray(affine.flat()).transpose())).normalize();
  assert.ok(Math.abs(back.x - s) < 1e-6 && Math.abs(back.y - c) < 1e-6);
});

test('a sheet split along its outline leaves a kerf of twice the gap and keeps a hinge joined', () => {
  const {positions, normals, triangles} = sheet();
  const value = Float32Array.from({length: positions.length / 3}, (_, v) => positions[3 * v] - 10.3);
  const {flap, fixed} = splitSheet(positions, normals, triangles, value, 0.5);
  for (let v = 0; v < flap.positions.length / 3; v++) assert.ok(flap.positions[3 * v] >= 10.8 - 1e-5);
  for (let v = 0; v < fixed.positions.length / 3; v++) assert.ok(fixed.positions[3 * v] <= 9.8 + 1e-5);
  assert.ok(flap.cutEdges.length > 0 && fixed.cutEdges.length > 0);
  for (const e of flap.cutEdges) assert.ok(Math.abs(flap.positions[3 * e] - 10.8) < 1e-4);
  // With every vertex near the outline marked as hinge, nothing is cut.
  const hinge = new Uint8Array(value.length).fill(1);
  const joined = splitSheet(positions, normals, triangles, value, 0.5, hinge);
  assert.equal(joined.flap.cutEdges.length + joined.fixed.cutEdges.length, 0);
  assert.equal((joined.flap.indices.length + joined.fixed.indices.length), triangles.length);
});

test('a kerf wall runs from the cut edge straight down to the inner outline', () => {
  const {positions, normals, triangles} = sheet();
  // A square outline 4 mm below the sheet; the sheet is cut at its inside offset.
  const loop = Float32Array.from([5, 5, 6.4, 17, 5, 6.4, 17, 17, 6.4, 5, 17, 6.4]);
  const inner = new PlanarLoop(loop, [0, 0, 1]);
  const value = Float32Array.from({length: positions.length / 3}, (_, v) => inner.query(positions.subarray(3 * v, 3 * v + 3)).distance);
  const {flap} = splitSheet(positions, normals, triangles, value, 1);
  const wall = kerfWall(flap, inner, [0, 0, 1]);
  assert.ok(wall.indices.length > 0);
  assert.ok(Math.max(...wall.indices) < wall.positions.length / 3);
  for (let v = 0; v < wall.positions.length / 6; v++) {
    const top = wall.positions.subarray(6 * v, 6 * v + 3), bottom = wall.positions.subarray(6 * v + 3, 6 * v + 6);
    assert.ok(Math.abs(top[0] - bottom[0]) < 1e-6 && Math.abs(top[1] - bottom[1]) < 1e-6);
    assert.ok(Math.abs(bottom[2] - 6.4) < 1e-4);
  }
});

test('source vertices follow the engine body cells exactly as the voxel flap does', () => {
  // A 6 x 6 x 2 slab body, rotated rigidly by 1.2 rad about a hinge along y at x = 4.
  const cells = [];
  for (let x = 4; x < 10; x++) for (let y = 5; y < 11; y++) for (let z = 9; z < 11; z++) cells.push(id(x, y, z));
  const rest = cells.map((cell) => sourceCenter(cell, shape, affine));
  const hinge = new THREE.Vector3(...voxelToRas(Float32Array.from([3.5, 0, 9.5]), affine));
  const axis = new THREE.Vector3(...voxelToRas(Float32Array.from([0, 1, 0]), affine)).sub(new THREE.Vector3(...voxelToRas(Float32Array.from([0, 0, 0]), affine))).normalize();
  const q = new THREE.Quaternion().setFromAxisAngle(axis, 1.2);
  const centersMm = rest.map((p) => new THREE.Vector3(...p).sub(hinge).applyQuaternion(q).add(hinge).toArray());
  const body = {cells, centersMm};
  const transport = new ExposureBodyTransport(body, {shape, affine});
  assert.equal(transport.update(centersMm), true);
  // Sample vertices at cell centres: they land on the engine's accepted centres.
  const pick = [0, 7, 30, 71];
  const voxel = Float32Array.from(pick.flatMap((i) => [...rasToVoxel(Float32Array.from(rest[i]), affine)]));
  const byId = new Map(cells.map((cell, i) => [cell, i]));
  const binding = bindToBody(voxel, () => true, byId, shape);
  const out = new Float32Array(voxel.length), restRas = voxelToRas(voxel, affine);
  transportVertices(binding, transport, restRas, null, affine, out, null);
  const moved = voxelToRas(out, affine);
  pick.forEach((cellIndex, k) => {
    for (let a = 0; a < 3; a++) assert.ok(Math.abs(moved[3 * k + a] - centersMm[cellIndex][a]) < 1e-4, `cell ${cellIndex} axis ${a}`);
  });
  // A vertex between cells (on the skin above the slab) moves rigidly with them.
  const between = Float32Array.from([6.3, 7.7, 10.4]);
  const b2 = bindToBody(between, () => true, byId, shape);
  const o2 = new Float32Array(3);
  transportVertices(b2, transport, voxelToRas(between, affine), null, affine, o2, null);
  const expected = new THREE.Vector3(...voxelToRas(between, affine)).sub(hinge).applyQuaternion(q).add(hinge);
  const got = voxelToRas(o2, affine);
  assert.ok(expected.distanceTo(new THREE.Vector3(...got)) < 1e-4);
  // Without a body cell within reach the vertex stays where it is.
  const far = bindToBody(Float32Array.from([20, 20, 2]), () => true, byId, shape, 2);
  assert.equal(far.unbound, 1);
});

function blob() {
  const removed = new Set();
  for (let x = 6; x < 14; x++) for (let y = 6; y < 13; y++) for (let z = 5; z < 12; z++)
    if ((x - 9.5) ** 2 + (y - 9) ** 2 + ((z - 8) * 1.3) ** 2 < 12) removed.add(id(x, y, z));
  removed.add(id(17, 17, 15)); // an isolated removed cell
  return removed;
}

test('the cavity field marks removed cell centres and keeps kept cells outside', () => {
  const removed = blob(), field = cavityField(removalOccupancy(removed), shape);
  for (let x = 4; x < 20; x++) for (let y = 4; y < 20; y++) for (let z = 3; z < 18; z++) {
    const inside = removed.has(id(x, y, z)), value = sampleCavity(field, [x, y, z]);
    if (inside) assert.ok(value > CAVITY_LEVEL, `removed ${x},${y},${z} ${value}`);
    else assert.ok(value < CAVITY_LEVEL, `kept ${x},${y},${z} ${value}`);
  }
});

test('the cavity wall stays within half a cell of the engine removed boundary', () => {
  const removed = blob(), field = cavityField(removalOccupancy(removed), shape);
  const wall = cavitySurface(field);
  assert.ok(wall.indices.length > 100);
  let worst = 0;
  for (let v = 0; v < wall.positions.length / 3; v++) worst = Math.max(worst, distanceToRemovedBoundary(wall.positions.subarray(3 * v, 3 * v + 3), removed, shape));
  assert.ok(worst <= 0.5, `worst ${worst}`);
  // Normals point out of the cavity into the remaining tissue.
  const v = 0, p = wall.positions.subarray(0, 3), n = wall.normals.subarray(0, 3);
  assert.ok(sampleCavity(field, [p[0] + n[0], p[1] + n[1], p[2] + n[2]]) < sampleCavity(field, [p[0] - n[0], p[1] - n[1], p[2] - n[2]]), String(v));
});

test('CUSA pending work recedes the wall into a partly worked cell', () => {
  const removed = new Set([id(10, 10, 10), id(11, 10, 10)]);
  const progress = {cells: [id(12, 10, 10)], fraction: [0.6]};
  const whole = cavityField(removalOccupancy(removed), shape), worked = cavityField(removalOccupancy(removed, progress), shape);
  const edge = (field) => { let x = 11; while (sampleCavity(field, [x, 10, 10]) > CAVITY_LEVEL) x += 0.05; return x; };
  const a = edge(whole), b = edge(worked);
  assert.ok(Math.abs(a - 11.5) < 0.1, String(a));
  assert.ok(b > a + 0.1 && b < 12.5, String(b));
  // A kept-out cell (filter) does not enter the field.
  assert.equal(removalOccupancy(removed, progress, (cell) => cell !== id(12, 10, 10)).size, 2);
});

test('outline points are marked cut only near the engine seam edges', () => {
  const points = [], hinge = [];
  for (let i = 0; i < 40; i++) { const t = (i / 40) * 2 * Math.PI; points.push(...voxelToRas(Float32Array.from([12 + 6 * Math.cos(t), 11 + 6 * Math.sin(t), 10]), affine)); hinge.push(i >= 30 ? 1 : 0); }
  const outline = {points: Float32Array.from(points), hinge: Uint8Array.from(hinge)};
  const seam = [[id(18, 11, 10), id(19, 11, 10)]];
  const flags = cutOutlinePoints(outline, seam, shape, affine);
  assert.ok(flags[0] === 1 && flags[20] === 0);
  assert.equal(flags.reduce((a, b) => a + b, 0) < 6, true);
});

test('transfer descriptors and cross-array checks refuse inconsistent assets', () => {
  const ok = {assetId: 'a', dtype: 'float32', shape: [4, 3], byteOrder: 'little', order: 'C', byteLength: 48, sha256: 'a'.repeat(64)};
  assert.equal(checkedDescriptor(ok, 'x', 'float32', 3).byteLength, 48);
  assert.throws(() => checkedDescriptor({...ok, dtype: 'float64'}, 'x', 'float32', 3));
  assert.throws(() => checkedDescriptor({...ok, byteLength: 40}, 'x', 'float32', 3));
  const {positions, normals, triangles} = sheet(3);
  const surface = {name: 'pial', layers: ['Cortex'], body: null, origin: 'estimated', positions, normals, triangles,
    cells: new Int32Array(9), ownerCells: new Int32Array(9).fill(-1)};
  const asset = {grid: {shape, affine}, surfaces: new Map([['pial', surface]]), vessels: null, tracts: []};
  validateSourceAsset(asset);
  assert.throws(() => validateSourceAsset({...asset, surfaces: new Map([['pial', {...surface, cells: Int32Array.from([0, 0, 0, 0, 0, 0, 0, 0, 1e9])}]])}));
  assert.throws(() => validateSourceAsset({...asset, surfaces: new Map([['pial', {...surface, triangles: Uint32Array.from([0, 1, 99])}]])}));
});

test('the source scene replaces the voxel layers, moves a flap and resolves picks to engine cells', () => {
  // Skin over a hinged scalp body; a removed brain cell clips the cortex.
  const skin = sheet(21, 12.4), cortex = sheet(21, 6.4);
  const ras = (p) => voxelToRas(p, affine);
  const rasNormals = (n) => { const out = new Float32Array(n.length); for (let i = 0; i < n.length; i += 3) for (let a = 0; a < 3; a++) out[i + a] = affine[a][0] * n[i] + affine[a][1] * n[i + 1] + affine[a][2] * n[i + 2]; return out; };
  const cellOf = (p) => Int32Array.from({length: p.length / 3}, (_, v) => id(...[0, 1, 2].map((a) => Math.round(p[3 * v + a]))));
  const edge = Float32Array.from({length: skin.positions.length / 3}, (_, v) => skin.positions[3 * v] - 10.3);
  const surfaces = new Map([
    ['skin', {name: 'skin', layers: ['Scalp skin'], body: 'scalp', origin: 'acquired', positions: ras(skin.positions), normals: rasNormals(skin.normals), triangles: skin.triangles,
      cells: cellOf(skin.positions), ownerCells: cellOf(skin.positions), bodyCode: Uint8Array.from(edge, (e) => (e > 0 ? 1 : 0)), bodyEdgeMm: edge}],
    ['pial', {name: 'pial', layers: ['Cerebral cortex'], body: null, origin: 'estimated', positions: ras(cortex.positions), normals: rasNormals(cortex.normals), triangles: cortex.triangles,
      cells: cellOf(cortex.positions), ownerCells: cellOf(cortex.positions)}],
  ]);
  const asset = {caseHash: 'h', assetName: 'a.npz', grid: {shape, affine}, surfaces, outlines: {}, vessels: null, tracts: [], loadMs: 0};
  const layers = [{name: 'Scalp skin', displayKind: 'scalp'}, {name: 'Subcutaneous tissue and galea', displayKind: 'galea'}, {name: 'Cerebral cortex', displayKind: 'grayMatter'}, {name: 'Deep gray matter'}];
  const scene = new OperativeSourceScene(asset, layers, [0, 0, 1], () => new THREE.MeshStandardMaterial());
  assert.deepEqual([...scene.replaced].sort(), ['Cerebral cortex', 'Scalp skin']);
  const cells = [];
  for (let x = 11; x < 21; x++) for (let y = 1; y < 22; y++) for (let z = 11; z < 13; z++) cells.push(id(x, y, z));
  const rest = cells.map((cell) => sourceCenter(cell, shape, affine));
  const lift = rest.map((p) => [p[0], p[1], p[2] + 5]);
  const body = (centersMm) => ({cells, centersMm, cutCells: [], seamEdges: [], cutCount: 1, edgeCount: 1});
  const owner = new Uint16Array(shape[0] * shape[1] * shape[2]);
  for (let x = 0; x < shape[0]; x++) for (let y = 0; y < shape[1]; y++) for (let z = 0; z < 8; z++) owner[id(x, y, z)] = 7;
  const view = (centersMm, removed = []) => ({scene: {shape, affine, regions: [{id: 7, name: 'Cerebral cortex', material: 'soft_tissue'}]}, removed,
    operativeExposureV1: {bodies: {scalp: body(centersMm), bone: body(rest), dura: body(rest)}}});
  scene.setOwner(owner, [{id: 7, name: 'Cerebral cortex', material: 'soft_tissue'}]);
  scene.update(view(rest));
  scene.appearance(() => true, () => {}, false, []);
  const names = scene.picks.map((m) => m.name);
  assert.ok(names.includes('source-surface:skin') && !names.some((n) => n.startsWith('source-flap')));
  const block = [];
  for (let x = 7; x < 10; x++) for (let y = 7; y < 10; y++) for (let z = 5; z < 8; z++) block.push(id(x, y, z));
  scene.update(view(lift, block));
  const flap = scene.group.getObjectByName('source-flap:skin');
  assert.ok(flap?.visible && !scene.group.getObjectByName('source-surface:skin').visible);
  const z = flap.geometry.getAttribute('position').array;
  // Flap vertices moved 5 mm outward (the slab is lifted rigidly).
  for (let v = 0; v < z.length / 3; v++) assert.ok(Math.abs(z[3 * v + 2] - 17.4) < 1e-3 || z[3 * v] < 11.5, `vertex ${v}`);
  assert.ok(scene.cavity.removed([8, 8, 6.4]) && !scene.cavity.removed([12, 8, 6.4]));
  // A hit on the cortex resolves to the engine cell of the nearest vertex and its owner layer.
  const mesh = scene.group.getObjectByName('source-surface:pial');
  mesh.updateMatrixWorld(true);
  const point = new THREE.Vector3(12.2, 9.1, 6.4).applyMatrix4(mesh.matrixWorld);
  const index = mesh.geometry.index.array;
  const hit = {object: mesh, point, face: {a: index[0], b: index[1], c: index[2]}};
  const near = (11 * 21 + 8) * 3;
  hit.face = {a: index[near * 2], b: index[near * 2 + 1], c: index[near * 2 + 2]};
  const resolved = scene.resolve(hit, (cell) => sourceCenter(cell, shape, affine));
  assert.ok(resolved && resolved.layer === 'Cerebral cortex');
  assert.equal(owner[resolved.cell], 7);
  // Brain-shift and injury displays act on the source cortex under the layer's name; head layers are not offered.
  const shifted = scene.scenarioSurfaces();
  assert.equal(shifted.get('Cerebral cortex').name, 'source-surface:pial');
  assert.equal(shifted.has('Scalp skin'), false);
  scene.dispose();
});

test('tip proximity on source surfaces measures to the tube wall and names an unknown vessel kind', async () => {
  const {sourceTipProximity} = await import('./tipProximity.ts');
  const vessels = {centrelinePoints: Float32Array.from([0, 0, 0, 0, 10, 0, 30, 0, 0, 30, 5, 0]), centrelineRadii: Float32Array.from([1, 1, 2, 2]),
    centrelineOffsets: Uint32Array.from([0, 2, 4]), segmentKind: Uint8Array.from([0, 1])};
  const tracts = [{bundle: 'CSTs', points: Float32Array.from([5, 0, 0, 9, 0, 0])}];
  const slots = {right: {instrument: 'cusa', position: [3, 0, 0], inserted: true}};
  const near = sourceTipProximity(slots, 'right', {vessels, tracts});
  assert.equal(near.vessel.name, 'Vessel (kind unknown)');
  assert.ok(Math.abs(near.vessel.distanceMm - 2) < 1e-6);
  assert.ok(Math.abs(near.tract.distanceMm - 2) < 1e-6);
  const artery = sourceTipProximity({right: {...slots.right, position: [26, 0, 0]}}, 'right', {vessels, tracts: []});
  assert.equal(artery.vessel.name, 'Artery');
  assert.equal(artery.tract, null);
});

test('small gaps in the assumed dura sheet are closed and its outer edge is left open', async () => {
  const {fillSmallHoles} = await import('./sourceSurfaceGeometry.ts');
  const {positions, normals, triangles} = sheet(9);
  // Remove the two triangles of one interior quad: a 4-edge hole.
  const n = 9, quad = 4 * (n - 1) + 4, kept = [];
  for (let t = 0; t < triangles.length / 3; t++) if (t !== 2 * quad && t !== 2 * quad + 1) kept.push(...triangles.subarray(3 * t, 3 * t + 3));
  const filled = fillSmallHoles(positions, normals, Uint32Array.from(kept), 10);
  assert.equal(filled.loops.length, 1);
  assert.equal(filled.loops[0].length, 4);
  assert.equal(filled.triangles.length, kept.length + 12);
  assert.equal(filled.positions.length, positions.length + 3);
});
