import test from 'node:test';
import assert from 'node:assert/strict';
import {CAP_REGIONS, capAxes, capRegions, paintCap, planeSegments} from './sourceSectionCap.ts';

/** A flat open sheet at height z (normal +z), x and y from -half to +half, 1 mm triangles. */
function sheet(z, half = 30, step = 1) {
  const n = Math.round((2 * half) / step) + 1, positions = [], normals = [], triangles = [];
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { positions.push(-half + i * step, -half + j * step, z); normals.push(0, 0, 1); }
  for (let i = 0; i + 1 < n; i++) for (let j = 0; j + 1 < n; j++) {
    const a = i * n + j, b = a + n;
    triangles.push(a, b, a + 1, a + 1, b, b + 1);
  }
  return {positions: Float32Array.from(positions), normals: Float32Array.from(normals), triangles: Uint32Array.from(triangles)};
}

test('the cap columns run from outside inward and fill each layer between its smooth boundaries', () => {
  // Layers stacked along +z (outward): skin 10, outer table 8, inner table 6, pia 5, white 2.
  const heights = {skin: 10.3, skullOuter: 8.3, skullInner: 6.3, pial: 5.3, white: 2.3};
  const normal = [1, 0, 0];
  const {across, down} = capAxes(normal, [0, 0, 1]);
  assert.deepEqual(down.map((v) => Math.round(v * 1e6) / 1e6), [0, 0, -1].map((v) => v || 0));
  // The cap's corner: 20 mm above the skin, 10 mm to one side; 0.5 mm pixels, 40 x 60.
  const stepMm = 0.5, width = 40, height = 60;
  const corner = [0, -across[1] * 10, 20];
  const segments = new Map(Object.entries(heights).map(([name, z]) => [name, planeSegments({name, ...sheet(z)}, corner, normal, across, down)]));
  for (const list of segments.values()) assert.ok(list.length > 0 && list.every((s) => s.nv < 0), 'top faces point out of the column');
  const regions = capRegions(segments, {width, height, stepMm});
  const region = (y) => CAP_REGIONS[regions[y * width + 20]];
  const at = (z) => Math.floor((20 - z) / stepMm);
  assert.equal(region(at(15)), 'open');
  assert.equal(region(at(9.5)), 'scalp');
  assert.equal(region(at(7)), 'bone');
  assert.equal(region(at(5.8)), 'intracranial');
  assert.equal(region(at(4)), 'cortex');
  assert.equal(region(at(0)), 'white');
  // Boundaries are exact to a pixel: no cell-sized steps.
  assert.equal(region(at(10.3) - 1), 'open');
  assert.equal(region(at(10.3) + 1), 'scalp');
});

test('an open surface that ends inside the cap still encloses by the side its normal faces', () => {
  const normal = [1, 0, 0], {across, down} = capAxes(normal, [0, 0, 1]);
  const corner = [0, -across[1] * 20, 20];
  // A skin sheet only 20 mm wide (columns beyond it never cross it) over a whole-width outer table.
  const segments = new Map([['skin', planeSegments({name: 'skin', ...sheet(10, 10)}, corner, normal, across, down)],
    ['skullOuter', planeSegments({name: 'skullOuter', ...sheet(8)}, corner, normal, across, down)]]);
  const width = 80, regions = capRegions(segments, {width, height: 60, stepMm: 0.5});
  const row = (z) => Math.floor((20 - z) / 0.5);
  assert.equal(CAP_REGIONS[regions[row(9) * width + 40]], 'scalp');
  // Outside the skin sheet's columns, the table still starts bone below it.
  assert.equal(CAP_REGIONS[regions[row(9) * width + 2]], 'open');
  assert.equal(CAP_REGIONS[regions[row(7) * width + 2]], 'bone');
  // A column that starts inside a surface (its first crossing leaves the tissue) starts inside it.
  const flipped = sheet(8);
  flipped.normals = flipped.normals.map((v) => -v);
  const inverted = capRegions(new Map([['skullOuter', planeSegments({name: 'skullOuter', ...flipped}, corner, normal, across, down)]]), {width, height: 60, stepMm: 0.5});
  assert.equal(CAP_REGIONS[inverted[row(12) * width + 40]], 'bone');
  assert.equal(CAP_REGIONS[inverted[row(6) * width + 40]], 'open');
});

test('cap pixels carry the scan intensity and leave open, opened and removed tissue transparent', () => {
  const regions = Uint8Array.from([0, 1, 2, 4, 4]);
  const signal = Float32Array.from([5, 10, 20, 30, 30]);
  const pixels = paintCap(regions, signal, (i) => i !== 4);
  assert.equal(pixels[3], 0);
  assert.equal(pixels[4 * 4 + 3], 0, 'removed tissue is open');
  assert.equal(pixels[4 + 3], 255);
  // Brighter scan, brighter cap.
  assert.ok(pixels[4 * 3] > pixels[4]);
});
