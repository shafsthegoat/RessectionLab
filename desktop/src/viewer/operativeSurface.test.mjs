import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { operativeSurface, registerOperativeEdits, markOperativeEdits } from './operativeSurface.ts';

function sample(shape, spacingMm, fill) {
  const mask = new Uint8Array(shape.reduce((a, b) => a * b));
  for (let x = 0, i = 0; x < shape[0]; x++)
    for (let y = 0; y < shape[1]; y++)
      for (let z = 0; z < shape[2]; z++, i++) mask[i] = Number(fill(x, y, z));
  return {
    grid: { origin: [0, 0, 0], shape, spacingMm },
    mask,
    bounds: { low: [0, 0, 0], high: shape.map((n) => n - 1) },
  };
}
function topology(mesh) {
  const indices = mesh.index.array,
    vertices = mesh.getAttribute('position'),
    parents = Uint32Array.from({ length: vertices.count }, (_, i) => i);
  const root = (i) => {
    while (i !== parents[i]) i = parents[i];
    return i;
  };
  const edgeIncidence = new Map();
  for (let i = 0; i < indices.length; i += 3) {
    const points = [];
    for (let k = 0; k < 3; k++) {
      const a = indices[i + k],
        b = indices[i + ((k + 1) % 3)];
      parents[root(a)] = root(b);
      const key = Math.min(a, b) * vertices.count + Math.max(a, b);
      edgeIncidence.set(key, 1 + (edgeIncidence.get(key) ?? 0));
      points.push(new THREE.Vector3().fromBufferAttribute(vertices, a));
    }
    assert.ok(new THREE.Triangle(...points).getArea() > 1e-10);
  }
  assert.ok([...edgeIncidence.values()].every((v) => v === 2));
  return new Set(Array.from(indices, root)).size;
}

test('corner- and edge-touching vessel samples retain separate closed foreground components', () => {
  for (const [dx, dy, dz] of [
    [1, 0, 0],
    [1, 1, 0],
    [1, 1, 1],
  ]) {
    const { grid, mask, bounds } = sample(
      [11, 11, 11],
      0.3125,
      (x, y, z) => (x === 5 && y === 5 && z === 5) || (x === 5 + dx && y === 5 + dy && z === 5 + dz),
    );
    const mesh = operativeSurface(grid, mask, bounds, 0, 'vessel');
    assert.equal(topology(mesh), dx + dy + dz === 1 ? 1 : 2);
    mesh.dispose();
  }
});

test('one-cell vessel retains calibre and the inspection plane closes its interior', () => {
  const { grid, mask, bounds } = sample(
    [19, 23, 19],
    0.3125,
    (x, y, z) => x === 9 && y >= 2 && y <= 20 && z === 9,
  );
  const mesh = operativeSurface(grid, mask, bounds, 9, 'vessel');
  assert.equal(topology(mesh), 1);
  const position = mesh.getAttribute('position'),
    points = Array.from({ length: position.count }, (_, i) =>
      new THREE.Vector3().fromBufferAttribute(position, i),
    );
  assert.equal(Math.min(...points.map((p) => p.y)), 8.5 * 0.3125);
  const mid = points.filter((p) => p.y >= 12 * 0.3125 && p.y <= 17 * 0.3125),
    width = Math.max(...mid.map((p) => p.x)) - Math.min(...mid.map((p) => p.x));
  assert.ok(width >= 0.3125 * 0.35, 'bounded smoothing preserves the thin original surface-net profile');
  mesh.dispose();
});

test('native slice terraces are geometrically smoothed within one source slice', () => {
  const { grid, mask, bounds } = sample([79, 53, 51], 0.3125, (x, y, z) => {
    const X = x * 0.3125 - 12,
      Y = y * 0.3125 - 8,
      Z = z * 0.3125 - 7.5;
    const nx = Math.round((X * 0.9848 + Y * 0.17365) / 0.55) * 0.55,
      ny = Math.round((-X * 0.17365 + Y * 0.9848) / 0.55) * 0.55;
    return Math.round(Z) <= 0.27 * (nx * 0.9848 - ny * 0.17365) + 0.16 * (nx * 0.17365 + ny * 0.9848);
  });
  grid.origin = [-12, -8, -7.5];
  const original = mask.slice();
  const geometry = operativeSurface(grid, mask, bounds, 0, 'bone'),
    mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
  const ray = new THREE.Raycaster(),
    heights = [],
    angles = [],
    normal = new THREE.Vector3(-0.27, -0.16, 1).normalize();
  for (let i = 0; i < 80; i++) {
    const x = -8 + i * 0.2;
    ray.set(new THREE.Vector3(x, 0.55, 20), new THREE.Vector3(0, 0, -1));
    const hit = ray.intersectObject(mesh)[0];
    heights.push(hit.point.z);
    angles.push(hit.normal.angleTo(normal));
    assert.ok(Math.abs(hit.point.z - 0.27 * x - 0.16 * 0.55) < 0.5);
  }
  const rms = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0) / v.length);
  const slopes = heights.slice(1).map((z, i) => Math.atan((z - heights[i]) / 0.2) - Math.atan(0.27));
  assert.ok((rms(slopes) * 180) / Math.PI < 9);
  assert.ok((rms(angles) * 180) / Math.PI < 10);
  assert.deepEqual(mask, original);
  geometry.dispose();
  mesh.material.dispose();
});

function faceSignatures(g) {
  const values = [],
    p = g.getAttribute('position'),
    n = g.getAttribute('normal'),
    ix = g.index.array;
  for (let k = 0; k < ix.length; k += 3) {
    const row = [];
    for (let a = 0; a < 3; a++) {
      const i = ix[k + a];
      row.push(
        ...[p.getX(i), p.getY(i), p.getZ(i), n.getX(i), n.getY(i), n.getZ(i)].map((v) => Math.round(v * 1e5)),
      );
    }
    values.push(row.join(','));
  }
  return values.sort();
}

test('local cavity rebuild matches a fresh fit through forwards and backwards scrubs', () => {
  const { grid, mask, bounds } = sample(
    [101, 17, 19],
    1,
    (x, y, z) => x > 2 && x < 98 && y >= 3 && y <= 13 && z >= 4 && z <= 15,
  );
  const edit = { low: [48, 6, 12], high: [50, 10, 15] };
  registerOperativeEdits(mask, edit);
  operativeSurface(grid, mask, bounds, 7, 'tumor').dispose();
  for (const value of [0, 1]) {
    for (let x = 48; x <= 50; x++)
      for (let y = 6; y <= 10; y++) for (let z = 12; z <= 15; z++) mask[(x * 17 + y) * 19 + z] = value;
    markOperativeEdits(mask, edit);
    const cached = operativeSurface(grid, mask, bounds, 7, 'tumor'),
      fresh = operativeSurface(grid, mask.slice(), bounds, 7, 'tumor');
    assert.equal(topology(cached), 1);
    assert.deepEqual(faceSignatures(cached), faceSignatures(fresh));
    cached.dispose();
    fresh.dispose();
  }
});

test('thin-sheet fitting translates paired interfaces by at most 0.62 physical mm', () => {
  const { grid, mask, bounds } = sample(
    [45, 27, 30],
    0.3125,
    (x, y, z) =>
      z >= 7 + Math.floor(x / 7) + Math.floor(y / 15) && z <= 8 + Math.floor(x / 7) + Math.floor(y / 15),
  );
  const original = operativeSurface(grid, mask, bounds, 12, 'vessel'),
    fitted = operativeSurface(grid, mask, bounds, 12, 'dura');
  assert.equal(topology(original), 1);
  assert.equal(topology(fitted), 1);
  const p = original.getAttribute('position'),
    q = fitted.getAttribute('position');
  assert.equal(p.count, q.count);
  let largest = 0;
  for (let i = 0; i < p.count; i++) {
    assert.equal(p.getX(i), q.getX(i));
    assert.equal(p.getY(i), q.getY(i));
    largest = Math.max(largest, Math.abs(p.getZ(i) - q.getZ(i)));
  }
  assert.ok(largest > 0.035 && largest <= 0.62001);
  original.dispose();
  fitted.dispose();
});
