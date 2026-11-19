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

test('adjacent teaching-section labels share cap edges and retain an unlabeled gap after edits', async () => {
  const { registerOperativeSection } = await import('./operativeSectionContour.ts');
  const shape = [33, 13, 27],
    spacing = 0.3125,
    row = 5;
  const regions = (x, z) => (z === 13 ? 0 : x < 9 + Math.floor(z / 3) ? 1 : z < 19 ? 2 : 3);
  const layers = [1, 2, 3].map((label) => sample(shape, spacing, (x, _y, z) => regions(x, z) === label));
  const { grid, bounds } = layers[0],
    masks = layers.map((layer) => layer.mask);
  registerOperativeSection(grid, masks, row, [masks[1]]);
  function sectionEdges(mask) {
    const mesh = operativeSurface(grid, mask, bounds, row),
      p = mesh.attributes.position,
      n = mesh.attributes.normal;
    const uses = new Map(),
      ix = mesh.index.array,
      plane = (row - 0.5) * spacing;
    const key = (i) => [p.getX(i), p.getZ(i)].join(',');
    for (let i = 0; i < ix.length; i += 3) {
      const tri = [ix[i], ix[i + 1], ix[i + 2]];
      if (tri.some((id) => p.getY(id) !== plane)) continue;
      const signed =
        (p.getX(tri[1]) - p.getX(tri[0])) * (p.getZ(tri[2]) - p.getZ(tri[0])) -
        (p.getX(tri[2]) - p.getX(tri[0])) * (p.getZ(tri[1]) - p.getZ(tri[0]));
      assert.ok(signed > 0, 'the section cap has positive area and consistent direction');
      for (let j = 0; j < 3; j++) {
        const a = tri[j],
          b = tri[(j + 1) % 3],
          keyPair = [key(a), key(b)].sort().join('|');
        assert.equal(n.getY(a), -1);
        uses.set(keyPair, (uses.get(keyPair) ?? 0) + 1);
      }
    }
    const ray = new THREE.Raycaster(
      new THREE.Vector3(7 * spacing, -1, 13 * spacing),
      new THREE.Vector3(0, 1, 0),
    );
    assert.equal(ray.intersectObject(new THREE.Mesh(mesh)).length, 0, 'a thin empty source row stays open');
    mesh.dispose();
    return new Set([...uses].filter(([, count]) => count === 1).map(([edge]) => edge));
  }
  const contacts = (a, b) => [...a].filter((edge) => b.has(edge)).length;
  const first = sectionEdges(masks[0]),
    target = sectionEdges(masks[1]),
    third = sectionEdges(masks[2]);
  const expected = [0, 0, 0, 0];
  for (let x = 0; x < shape[0]; x++)
    for (let z = 0; z < shape[2]; z++) {
      if (regions(x, z) !== 1) continue;
      for (const [xx, zz] of [
        [x - 1, z],
        [x + 1, z],
        [x, z - 1],
        [x, z + 1],
      ])
        if (xx >= 0 && zz >= 0 && xx < shape[0] && zz < shape[2]) expected[regions(xx, zz)]++;
    }
  assert.equal(contacts(first, target), expected[2]);
  assert.equal(contacts(first, third), expected[3]);
  for (let x = 21; x < 26; x++) for (let z = 3; z < 9; z++) masks[1][(x * shape[1] + row) * shape[2] + z] = 0;
  assert.equal(contacts(first, sectionEdges(masks[1])), expected[2]);
});

test('a registered submillimetre cavity follows the accepted tip radius and restores on backward scrub', async () => {
  const { registerCavityField, setCavityStep } = await import('./operativeCavityField.ts');
  const { grid, mask, bounds } = sample([25, 25, 25], 0.3125, () => true),
    s = grid.spacingMm,
    r = 0.84;
  const center = 12,
    id = (x, y, z) => (x * 25 + y) * 25 + z;
  grid.origin = [-center * s, -center * s, -center * s];
  const pristine = mask.slice(),
    box = { low: [8, 8, 8], high: [16, 16, 16] };
  registerCavityField(grid, mask, box, [id(center, center, center)], r);
  for (let x = 8; x <= 16; x++)
    for (let y = 8; y <= 16; y++)
      for (let z = 8; z <= 16; z++)
        if (s * Math.hypot(x - center, y - center, z - center) <= r) mask[id(x, y, z)] = 0;
  const accepted = mask.slice();
  setCavityStep(mask, 1);
  const geometry = operativeSurface(grid, mask, bounds, 0, 'tumor');
  const material = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }),
    mesh = new THREE.Mesh(geometry, material);
  const ray = new THREE.Raycaster();
  let total = 0;
  for (let i = 0; i < 64; i++) {
    const nz = 1 - (2 * (i + 0.5)) / 64,
      a = 2.39996323 * i;
    ray.set(
      new THREE.Vector3(),
      new THREE.Vector3(Math.sqrt(1 - nz * nz) * Math.cos(a), Math.sqrt(1 - nz * nz) * Math.sin(a), nz),
    );
    total += Math.abs(ray.intersectObject(mesh)[0].distance - r);
  }
  assert.ok(total / 64 < 0.16, 'mean brush-radius error remains below half one display cell plus rounding');
  assert.deepEqual(mask, accepted);
  setCavityStep(mask, 0);
  mask.set(pristine);
  const restored = operativeSurface(grid, mask, bounds, 0, 'tumor'),
    fresh = operativeSurface(grid, pristine, bounds, 0, 'tumor');
  assert.deepEqual(restored.attributes.position.array, fresh.attributes.position.array);
  restored.dispose();
  fresh.dispose();
  geometry.dispose();
  material.dispose();
});

test('a protected source sample keeps clearance from the fitted target contour', async () => {
  const { registerOperativeSection, operativeSectionContour } = await import('./operativeSectionContour.ts');
  const { grid, mask } = sample([31, 7, 31], 0.3125, (x, _y, z) => x < 14 + Math.floor(z / 5));
  const [nx, ny, nz] = grid.shape,
    row = 3;
  grid.voxels = new Uint16Array(mask.length);
  const protectedCells = [
    [15, 5],
    [17, 15],
    [18, 20],
  ];
  for (const [x, z] of protectedCells) {
    const i = (x * ny + row) * nz + z;
    mask[i] = 0;
    grid.voxels[i] = 16;
  }
  registerOperativeSection(grid, [mask], row, [mask], undefined, mask);
  const fit = operativeSectionContour(grid, mask, row);
  for (const [x, z] of protectedCells)
    for (const dx of [0, 1])
      for (const dz of [0, 1]) {
        const k = (x + dx) * (nz + 1) + z + dz;
        assert.ok(
          Math.hypot(fit.x[k] - (x + dx - 0.5), fit.z[k] - (z + dz - 0.5)) <= 0.421,
          'protected center stays inside its source cell',
        );
      }
});
