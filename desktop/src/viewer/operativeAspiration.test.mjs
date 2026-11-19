import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import test from 'node:test';
import { aspiration } from './operativeAspiration.ts';

function slopedTarget() {
  const shape = [65, 65, 80];
  const spacingMm = 0.3125;
  const origin = [-10, -10, -12.5];
  const index = (x, y, z) => (x * shape[1] + y) * shape[2] + z;
  const voxels = new Uint16Array(shape[0] * shape[1] * shape[2]);
  for (let x = 12; x <= 52; x++)
    for (let y = 12; y <= 52; y++)
      for (let z = 12; z <= 56 - Math.floor(Math.abs(x - 32) / 4); z++) voxels[index(x, y, z)] = 8;
  return { shape, spacingMm, origin, radiusMm: 24, routeRadiusMm: 2, stations: { tumor: 5 }, voxels, index };
}

test('overlapping contacts leave an open pocket in the exposed target section', () => {
  const field = slopedTarget();
  const original = field.voxels.slice();
  const sweep = aspiration(field, 0.84, 'overlapping');
  const [, ny, nz] = field.shape;
  const coord = (i) => [Math.floor(i / (ny * nz)), Math.floor(i / nz) % ny, i % nz];
  // At each depth, the central 1.9 mm wide opening stays connected to the
  // original tumor surface. Untouched target below the floor is retained.
  for (let x = 29; x <= 35; x++) {
    for (let z = 44; z <= 56; z++) assert.ok(sweep.firstHit.has(field.index(x, 32, z)));
    assert.equal(sweep.firstHit.has(field.index(x, 32, 32)), false);
  }
  for (const [index, first] of sweep.firstHit) {
    assert.equal(original[index], 8);
    assert.ok(first >= 1 && first <= sweep.path.length);
    const tip = coord(sweep.path[first - 1]);
    const p = coord(index);
    assert.ok(Math.hypot(...p.map((value, a) => value - tip[a])) * field.spacingMm <= 0.84);
  }
  assert.deepEqual(field.voxels, original);
});

test('the original spiral retains its historical order of contacts', () => {
  const sweep = aspiration(slopedTarget(), 0.84, 'spiral');
  // Reference from the pre-change 23f114ff sweep on this sloped target.
  const digest = createHash('sha256')
    .update(JSON.stringify([sweep.path, [...sweep.firstHit]]))
    .digest('hex');
  assert.equal(digest, '2c2983438d1875c3fd3b892455832695588604ae507ed0dca30557381a8a11c4');
});
