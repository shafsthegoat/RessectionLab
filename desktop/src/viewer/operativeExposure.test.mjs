import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {exteriorSkinSurface} from './exteriorSkinSurface.ts';
import {OperativeExposureScene} from './operativeExposureScene.ts';
import {bodyIsDisplaced, partitionExposureSurfaces, sourceCenter} from './exposureAnatomy.ts';
import {ExposureBodyTransport, buildExposureSurface} from './exposureBodySurface.ts';
import {exposureMicroscope, microscopeCamera} from './microscope.ts';

const shape = [16, 17, 18],
  affine = [
    [1, 0, 0, -5],
    [0, 1, 0, 3],
    [0, 0, 1, -7],
    [0, 0, 0, 1],
  ];
const id = (x, y, z) => (x * shape[1] + y) * shape[2] + z;
function model() {
  const scene = {shape, affine, resolutionMm: [1, 1, 1], regions: []};
  const bodies = {},
    owner = new Uint16Array(shape.reduce((a, b) => a * b)),
    compartments = [];
  for (const [name, region, z, thickness] of [
    ['scalp', 1, 13, 2],
    ['bone', 2, 9, 3],
    ['dura', 3, 5, 1],
  ]) {
    const cells = [];
    for (let x = 2; x <= 9; x++)
      for (let y = 3; y <= 10; y++) for (let k = 0; k < thickness; k++) cells.push(id(x, y, z + k));
    const bounds = cells.map((c) => sourceCenter(c, shape, affine));
    bodies[name] = {cells, centersMm: bounds, seamEdges: [], status: 'attached', cutCount: 0};
    const mask = new Uint8Array(owner.length);
    for (const c of [...cells, id(10, 8, z)]) {
      mask[c] = 1;
      owner[c] = region;
    }
    compartments.push({name, kind: name, mask, color: '#bbbbbb'});
    scene.regions.push({
      name,
      id: region,
      material: name === 'scalp' ? 'soft_tissue' : name === 'dura' ? 'membrane' : 'bone',
    });
  }
  return {
    owner,
    volume: {shape, affine, compartments},
    view: {scene, moved: [], removed: [], operativeExposureV1: {bodies, objects: []}},
  };
}

test('each reflected sheet and the rigid plate retain thickness at the accepted centers, including after a return to rest', () => {
  const {owner, view} = model();
  for (const [name, body] of Object.entries(view.operativeExposureV1.bodies)) {
    const transport = new ExposureBodyTransport(body, view.scene);
    const surface = buildExposureSurface(body.cells, owner, transport, view.scene, owner[body.cells[0]]);
    const rest = surface.geometry.attributes.position.array.slice();
    const restNormals = surface.geometry.attributes.normal.array.slice();
    const rotation = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0.8, 0.18, 0.42).normalize(), 1.63);
    const delta = new THREE.Vector3(2.1, -1.1, 2.7);
    const move = (p) => new THREE.Vector3(...p).applyQuaternion(rotation).add(delta);
    assert(transport.update(body.centersMm.map((p) => move(p).toArray())));
    surface.update();
    const current = surface.geometry.attributes.position.array;
    for (let i = 0; i < current.length; i += 3) {
      const point = move(rest.slice(i, i + 3));
      assert(point.distanceTo(new THREE.Vector3(...current.slice(i, i + 3))) < 2e-5, name);
      const movedNormal = new THREE.Vector3(...restNormals.slice(i, i + 3)).applyQuaternion(rotation);
      assert(
        movedNormal.distanceTo(new THREE.Vector3().fromBufferAttribute(surface.geometry.attributes.normal, i / 3)) <
          2e-5,
        name,
      );
    }
    assert(transport.update(body.centersMm));
    surface.update();
    assert.deepEqual(current, rest);
    surface.geometry.dispose();
  }
});

test('subvoxel pose yields one transported surface and leaves the exact numerical native labels intact', () => {
  const {owner, view, volume} = model(),
    body = view.operativeExposureV1.bodies.bone;
  const sourceMask = volume.compartments[1].mask.slice();
  body.centersMm = body.centersMm.map(([x, y, z]) => [x + 0.08, y, z + 0.11]);
  assert(bodyIsDisplaced(body, shape, affine));
  partitionExposureSurfaces(volume, view, owner, new Map());
  const mask = volume.compartments[1].surfaceMask;
  assert.equal(
    mask.reduce((a, b) => a + b),
    1,
  );
  assert.equal(mask[id(10, 8, 9)], 1);
  assert.deepEqual(volume.compartments[1].mask, sourceMask);
  const display = new OperativeExposureScene();
  display.update(view, volume, owner);
  assert.deepEqual(
    display.picks.map((m) => m.userData.compartment),
    ['bone'],
  );
  assert.equal(display.group.getObjectByName('accepted-seams:bone'), undefined);
  body.seamEdges = [[id(9, 8, 9), id(10, 8, 9)]];
  view.operativeExposureV1.objects.push({
    id: 'clip1',
    kind: 'raney-clip',
    radiusMm: 0.8,
    body: 'bone',
    positionMm: [4, 11, 2],
    status: 'placed',
  });
  display.update({...view}, volume, owner);
  assert(display.group.getObjectByName('accepted-seams:bone'));
  const clip = display.group.getObjectByName('exposure-object:clip1');
  assert.deepEqual(clip.position.toArray(), [4, 11, 2]);
  view.operativeExposureV1.objects[0].status = 'retrieved';
  body.centersMm = body.cells.map((c) => sourceCenter(c, shape, affine));
  partitionExposureSurfaces(volume, view, owner, new Map());
  display.update({...view}, volume, owner);
  assert.equal(volume.compartments[1].surfaceMask, undefined);
  assert.equal(display.picks.length, 0);
  assert.equal(display.group.getObjectByName('exposure-object:clip1'), undefined);
  const frame = exposureMicroscope(view),
    camera = microscopeCamera(frame);
  assert(frame.outward[2] > 0.99);
  assert(camera.position[2] > camera.target[2]);
  display.clear();
});

test('closure overlays bridge their two physical RAS anchors even when the seam direction is oblique', () => {
  const {owner, view, volume} = model();
  const display = new OperativeExposureScene(),
    start = [3, 14, 6.5],
    end = [4.2, 12.8, 6.5];
  const suture = {
    id: 's1',
    kind: 'suture',
    radiusMm: 0.45,
    status: 'intended-retained',
    body: 'scalp',
    positionMm: start,
    normal: [0, 0, 1],
    endpointsMm: [start, end],
  };
  view.operativeExposureV1.objects = [suture];
  display.update(view, volume, owner);
  const group = display.group.getObjectByName('exposure-object:s1');
  group.updateWorldMatrix(true, true);
  const points = group.children[0].geometry.attributes.position;
  const nearest = (anchor) => {
    let d = Infinity;
    for (let i = 0; i < points.count; i++) {
      const vertex = new THREE.Vector3().fromBufferAttribute(points, i).applyMatrix4(group.matrixWorld);
      d = Math.min(d, vertex.distanceTo(new THREE.Vector3(...anchor)));
    }
    return d;
  };
  assert(nearest(start) < 0.06);
  assert(nearest(end) < 0.06);
  display.clear();
});

test('blood color requires a generated empty-origin field with exclusively modeled wound input', async () => {
  const {SurgicalFieldResources} = await import('./surgicalFieldOptics.ts');
  const {view} = model();
  Object.assign(view, {
    fluid: {[id(8, 7, 15)]: 0.26},
    metrics: {fluidAddedMm3: 0.28, modeledVesselFluidGeneratedMm3: 0.28},
    sources: [{kind: 'modeled-vessel-wound'}],
  });
  const wet = () => {
    const resources = new SurgicalFieldResources(),
      origin = resources.update(view).group.userData.fieldSummary.fluidOrigin;
    resources.clear();
    return origin;
  };
  assert.equal(wet(), 'vascular sources only');
  view.metrics.fluidAddedMm3 = 0.29;
  assert.equal(wet(), 'mixed or unknown');
  view.metrics.fluidAddedMm3 = 0.28;
  view.sources.push({kind: 'external'});
  assert.equal(wet(), 'mixed or unknown');
  view.sources.pop();
  delete view.operativeExposureV1;
  assert.equal(wet(), undefined);
});

test('three declared fixed supports draw only the actual collision capsules and persist across engagement', () => {
  const {owner, view, volume} = model(),
    display = new OperativeExposureScene();
  const pads = [
    [2, 5, 7],
    [4, 6, 7],
    [5, 3, 7],
  ].map((p, i) => ({aMm: p, bMm: [p[0] + 0.1 * i, p[1] + 0.2, p[2] + 3 + i], radiusMm: 0.8 + 0.1 * i}));
  view.operativeExposureV1.boneRest = {pads, engaged: false};
  display.update(view, volume, owner);
  const rest = display.group.getObjectByName('exposure:bone-rest');
  assert.equal(rest.children.length, 3);
  rest.updateWorldMatrix(true, true);
  for (const [i, mesh] of rest.children.entries()) {
    const {aMm, bMm, radiusMm} = pads[i],
      a = new THREE.Vector3(...aMm),
      b = new THREE.Vector3(...bMm);
    const axis = b.clone().sub(a),
      len = axis.length();
    axis.normalize();
    const position = mesh.geometry.attributes.position;
    let maxR = 0,
      minT = Infinity,
      maxT = -Infinity;
    for (let j = 0; j < position.count; j++) {
      const q = new THREE.Vector3().fromBufferAttribute(position, j).applyMatrix4(mesh.matrixWorld).sub(a);
      const t = q.dot(axis);
      maxR = Math.max(maxR, q.addScaledVector(axis, -Math.min(len, Math.max(0, t))).length());
      minT = Math.min(minT, t);
      maxT = Math.max(maxT, t);
    }
    assert(Math.abs(maxR - radiusMm) < 1e-5);
    assert(Math.abs(minT + radiusMm) < 1e-5);
    assert(Math.abs(maxT - len - radiusMm) < 1e-5);
  }
  view.operativeExposureV1.boneRest.engaged = true;
  display.update({...view}, volume, owner);
  assert.equal(display.group.getObjectByName('exposure:bone-rest'), rest);
  assert(rest.userData.engaged);
  display.clear();
});

test('the two faces of a one-cell dural sheet remain separated during a 100 degree material bend', () => {
  const {owner, view} = model(),
    sheet = view.operativeExposureV1.bodies.dura;
  const transport = new ExposureBodyTransport(sheet, view.scene),
    surface = buildExposureSurface(sheet.cells, owner, transport, view.scene, 3);
  const raw = surface.sourceGeometry(),
    base = raw.attributes.position.array.slice(),
    hingeY = Math.min(...sheet.centersMm.map((p) => p[1])),
    baselineZ = sheet.centersMm[0][2];
  const length = Math.max(...sheet.centersMm.map((p) => p[1] - hingeY)),
    angle = (100 * Math.PI) / 180,
    radius = length / angle;
  const moved = sheet.centersMm.map(([x, y, z]) => [
    x,
    hingeY + radius * Math.sin((y - hingeY) / radius),
    z + radius * (1 - Math.cos((y - hingeY) / radius)),
  ]);
  transport.update(moved);
  surface.update();
  const pairs = new Map();
  for (let i = 0; i < base.length; i += 3) {
    const [x, y, z] = base.subarray(i, i + 3);
    if (!Number.isInteger(x) || !Number.isInteger(y) || Math.abs(Math.abs(z - baselineZ) - 0.5) > 1e-7) continue;
    const k = x + ',' + y,
      pair = pairs.get(k) ?? [];
    pair[z > baselineZ ? 1 : 0] = new THREE.Vector3().fromArray(raw.attributes.position.array, i);
    pairs.set(k, pair);
  }
  let checked = 0;
  for (const [lo, hi] of pairs.values())
    if (lo && hi) {
      assert(Math.abs(hi.distanceTo(lo) - 1) < 2e-5);
      checked++;
    }
  assert(checked >= 40);
  surface.disposeOther(raw);
  raw.dispose();
});

test('intact diagonal outer skin reconstruction requires nearby true faces and withdraws if its support changes', () => {
  const {view, volume, owner} = model(),
    [nx, ny, nz] = shape,
    steps = [ny * nz, nz, 1];
  owner.fill(0);
  for (let x = 0; x < nx; x++)
    for (let y = 0; y < ny; y++)
      for (let z = 0; z < nz; z++) {
        if (Math.hypot(x - 7, y - 8, z - 8) < 6.7) owner[id(x, y, z)] = 2;
      }
  const filled = owner.slice();
  for (let i = 0; i < owner.length; i++) {
    if (filled[i] && steps.some((k) => !filled[i + k] || !filled[i - k])) owner[i] = 1;
  }
  const before = owner.slice();
  const contour = exteriorSkinSurface(owner, shape, affine, 1, new Set());
  assert(contour.indices.length > 0);
  // Every emitted quarter-face triangle lies exactly in at least one real
  // rectangular native skin/air face in its registered cube.
  for (const [origin, [a, b]] of contour.cubes) {
    const base = [Math.floor(origin / (ny * nz)), Math.floor(origin / nz) % ny, origin % nz];
    for (let i = a; i < b; i += 3) {
      const pts = [0, 1, 2].map((c) =>
        contour.positions.slice(3 * contour.indices[i + c], 3 * contour.indices[i + c] + 3),
      );
      const axis = [0, 1, 2].find((k) => pts[0][k] === pts[1][k] && pts[1][k] === pts[2][k]);
      assert.notEqual(axis, undefined);
      const midpoint = pts[0].map((_, k) => pts.reduce((sum, point) => sum + point[k] / 3, 0)),
        p = Array.from(midpoint, Math.round);
      p[axis] = base[axis];
      const low = id(...p);
      p[axis]++;
      const high = id(...p);
      assert((owner[low] === 1 && owner[high] === 0) || (owner[high] === 1 && owner[low] === 0));
    }
  }
  assert.deepEqual(owner, before);
  assert.equal(contour.update(contour.sourceSupport).length, 0);
  assert.deepEqual(contour.update([]), contour.indices);
  // A genuine exterior galea face cannot be hidden as a diagonal artifact.
  const inside = [...contour.sourceSupport].find((c) => owner[c] === 2 && steps.some((k) => owner[c + k] === 1));
  const outer = steps.map((k) => inside + k).find((c) => owner[c] === 1);
  const withHole = owner.slice();
  withHole[outer] = 0;
  assert(!exteriorSkinSurface(withHole, shape, affine, 1, new Set())?.sourceSupport.has(inside));
  volume.compartments[0].displayKind = 'scalp';
  volume.compartments[1].displayKind = 'galea';
  const display = new OperativeExposureScene(),
    settings = {operativeView: 'surgeon'};
  // Move the test's editable sheet completely away from this static sphere.
  const scalp = view.operativeExposureV1.bodies.scalp;
  scalp.cells = [id(0, 0, 0)];
  scalp.centersMm = [sourceCenter(scalp.cells[0], shape, affine)];
  display.update(view, volume, owner);
  display.appearance(volume, settings, () => true);
  const mesh = display.group.getObjectByName('intact-exterior-skin-faces');
  assert(mesh.visible);
  display.setSmooth(false);
  assert(!mesh.visible);
  display.setSmooth(true);
  assert(mesh.visible);
  display.appearance(volume, {...settings, microscope: {tissueStyle: 'labels'}}, () => true);
  assert(!mesh.visible);
  display.appearance(volume, settings, () => true);
  const initial = mesh.geometry.drawRange.count;
  display.update({...view, removed: [inside]}, volume, owner);
  assert(mesh.visible);
  assert(mesh.geometry.drawRange.count < initial);
  assert(mesh.geometry.drawRange.count > 0);
  display.update(view, volume, owner);
  assert.equal(mesh.geometry.drawRange.count, initial);
  display.clear();
});

test('Raney backing and jaws use the actual accepted RAS capsules, including a retrieved clip', () => {
  const {owner, view, volume} = model();
  const a = [5, 7, 11],
    b = [5.4, 7.2, 9.6],
    o = {
      id: 'raney-clip-1',
      kind: 'raney-clip',
      status: 'placed',
      positionMm: a,
      sourceContactMm: b,
      normal: [1, 0, 0],
      radiusMm: 0.58,
      body: 'scalp',
      clipSegments: [
        {aMm: a, bMm: a, radiusMm: 0.58},
        {aMm: a, bMm: b, radiusMm: 0.16},
      ],
    };
  const display = new OperativeExposureScene();
  const current = (objects) => ({...view, operativeExposureV1: {...view.operativeExposureV1, objects}});
  display.update(current([o]), volume, owner);
  const group = display.group.children.flatMap((c) => c.children).find((c) => c.userData.operativeObject?.id === o.id);
  assert.equal(group.children.length, 2);
  group.updateWorldMatrix(true, true);
  assert.equal(group.children[0].geometry.parameters.radius, 0.58);
  assert(group.children[0].getWorldPosition(new THREE.Vector3()).distanceTo(new THREE.Vector3(...a)) < 1e-9);
  const jaw = group.children[1],
    dir = new THREE.Vector3().subVectors(new THREE.Vector3(...b), new THREE.Vector3(...a));
  assert.equal(jaw.geometry.parameters.radius, 0.16);
  assert(Math.abs(jaw.geometry.parameters.height - dir.length()) < 1e-9);
  assert(jaw.localToWorld(new THREE.Vector3(0, -dir.length() / 2, 0)).distanceTo(new THREE.Vector3(...a)) < 1e-9);
  assert(jaw.localToWorld(new THREE.Vector3(0, dir.length() / 2, 0)).distanceTo(new THREE.Vector3(...b)) < 1e-9);
  display.update(current([{...o, status: 'retrieved'}]), volume, owner);
  assert.equal(group.parent, null);
  display.clear();
});

test('a centerline tube vessel wall takes the wet finish only on a physical material', async () => {
  const THREE = await import('three');
  const {applyWetVesselFinish, wetVesselFinish} = await import('./surgicalFieldOptics.ts');
  const wall = new THREE.MeshPhysicalMaterial({roughness: .43, clearcoat: .22});
  assert.equal(applyWetVesselFinish(wall), true);
  assert.equal(wall.clearcoat, wetVesselFinish.clearcoat);
  // A broad soft coat band over a slightly tighter, faint wall glint.
  assert.ok(wall.clearcoatRoughness >= .3 && wall.roughness < wall.clearcoatRoughness && wall.specularIntensity < .5 && wall.userData.wetVessel);
  const plain = new THREE.MeshStandardMaterial({roughness: .52});
  assert.equal(applyWetVesselFinish(plain), false);
  assert.equal(plain.roughness, .52);
});

test('tube walls take the shared artery and vein colours from their vertex classes', async () => {
  const THREE = await import('three');
  const {applyTubeVesselColor, tubeVesselColors} = await import('./surgicalFieldOptics.ts');
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(12), 3));
  const mesh = new THREE.Mesh(geometry, new THREE.MeshPhysicalMaterial());
  applyTubeVesselColor(mesh, Uint8Array.of(0, 0, 0, 0));
  assert.equal('#' + mesh.material.color.getHexString(), tubeVesselColors.artery);
  assert.equal(mesh.material.vertexColors, false);
  applyTubeVesselColor(mesh, Uint8Array.of(0, 1, 1, 0));
  assert.equal(mesh.material.vertexColors, true);
  const colors = geometry.getAttribute('color').array;
  const vein = new THREE.Color(tubeVesselColors.vein);
  assert.deepEqual([...colors.slice(3, 6)].map(v => +v.toFixed(5)), vein.toArray().map(v => +v.toFixed(5)));
  applyTubeVesselColor(mesh, Uint8Array.of(1, 1, 1));  // Misaligned classes are never applied.
  assert.equal(mesh.material.vertexColors, false);
  assert.equal(geometry.hasAttribute('color'), false);
  assert.equal('#' + mesh.material.color.getHexString(), tubeVesselColors.artery);
  applyTubeVesselColor(mesh, Uint8Array.of(1, 1, 1, 1));
  assert.equal('#' + mesh.material.color.getHexString(), tubeVesselColors.vein);
});

test('a prepared vessel surface carries the tube-wall mark of its source surface', async () => {
  const {prepareOperativeVessel} = await import('./prepareSurface.ts');
  const wall = {mode: 'native-probability', region: 1, positions: new Float32Array(9), normals: new Float32Array(9),
    indices: Uint32Array.of(0, 1, 2), sourceCells: Uint32Array.of(0, 1, 2), sites: []};
  assert.equal(prepareOperativeVessel({...wall, tube: true}).tubeWall, true);
  assert.equal(prepareOperativeVessel(wall).tubeWall, false);
});
