import test from "node:test";
import assert from "node:assert/strict";
import {
  axisGridFrame,
  fieldGrid,
  fieldLocalToRas,
  formatLabelDistance,
  labelDistance,
  nearestCellDistance,
} from "./toolNavigation.ts";
import { rasAffine, transformPoint } from "./coordinates.ts";

// Brute force reference: exact point-to-box distance over every cell.
function bruteDistance(grid, accept, q) {
  const [nx, ny, nz] = grid.shape, [sx, sy, sz] = grid.spacingMm;
  let best = Infinity;
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) for (let k = 0; k < nz; k++) {
    if (!accept((i * ny + j) * nz + k)) continue;
    const gap = (v, c, s) => Math.max(0, Math.abs(v - c) - s / 2);
    best = Math.min(best, Math.hypot(gap(q[0], i * sx, sx), gap(q[1], j * sy, sy), gap(q[2], k * sz, sz)));
  }
  return best;
}
function boxMask(shape, low, high) {
  const mask = new Uint8Array(shape[0] * shape[1] * shape[2]);
  for (let i = low[0]; i <= high[0]; i++) for (let j = low[1]; j <= high[1]; j++) for (let k = low[2]; k <= high[2]; k++)
    mask[(i * shape[1] + j) * shape[2] + k] = 1;
  return mask;
}

test("Ring search matches brute-force point-to-cell distance on anisotropic grids", () => {
  const grid = { shape: [9, 7, 5], spacingMm: [0.5, 1.25, 3] };
  const mask = new Uint8Array(9 * 7 * 5);
  for (const index of [17, 101, 230, 300]) mask[index] = 1;
  const accept = i => mask[i] === 1;
  for (const q of [[0, 0, 0], [2.1, 4.4, 7.7], [-3, 9, 20], [4.0, 1.875, 6.0], [1.3, 2.6, 1.4]]) {
    assert.ok(Math.abs(nearestCellDistance(grid, q, accept, 1e6) - bruteDistance(grid, accept, q)) < 1e-12, JSON.stringify(q));
  }
  assert.equal(nearestCellDistance(grid, [0, 0, 0], () => false, 1e6), Infinity);
  assert.equal(nearestCellDistance(grid, [0, 0, 0], accept, 0.1), Infinity, "Farther cells than the limit are not reported");
});

test("Signed label distance is negative inside, positive outside and exact at the box faces", () => {
  const shape = [20, 20, 20], grid = { shape, spacingMm: [1, 1, 1] };
  const mask = boxMask(shape, [5, 5, 5], [14, 14, 14]); // label occupies 4.5 .. 14.5 mm
  const member = i => mask[i] === 1;
  const centre = labelDistance(grid, member, [9.5, 9.5, 9.5], { complete: true });
  assert.deepEqual(centre, { inside: true, distanceMm: 5, exact: true });
  const near = labelDistance(grid, member, [5.2, 9, 9], { complete: true });
  assert.equal(near.inside, true);
  assert.ok(Math.abs(near.distanceMm - 0.7) < 1e-12, "inside distance reaches the nearest label face");
  const outside = labelDistance(grid, member, [1.5, 9, 9], { complete: true });
  assert.deepEqual(outside, { inside: false, distanceMm: 3, exact: true });
  const corner = labelDistance(grid, member, [1.5, 1.5, 9], { complete: true });
  assert.ok(Math.abs(corner.distanceMm - Math.hypot(3, 3)) < 1e-12);
  assert.equal(formatLabelDistance(centre), "−5.0 mm");
  assert.equal(formatLabelDistance(outside), "3.0 mm");
});

test("A local crop reports only a lower bound when unsampled space is nearer", () => {
  const shape = [10, 10, 10], grid = { shape, spacingMm: [1, 1, 1] };
  const mask = boxMask(shape, [0, 0, 0], [9, 9, 9]); // label fills the crop
  const member = i => mask[i] === 1;
  const crop = labelDistance(grid, member, [1, 4.5, 4.5]);
  assert.deepEqual(crop, { inside: true, distanceMm: 1.5, exact: false }, "the label may continue beyond the crop");
  assert.equal(formatLabelDistance(crop), "≤ −1.5 mm");
  const whole = labelDistance(grid, member, [1, 4.5, 4.5], { complete: true });
  assert.deepEqual(whole, { inside: true, distanceMm: 1.5, exact: true }, "a complete acquisition ends the label at its edge");
  const empty = labelDistance(grid, () => false, [4.5, 4.5, 4.5]);
  assert.deepEqual(empty, { inside: false, distanceMm: 5, exact: false });
  assert.equal(formatLabelDistance(empty), "≥ 5.0 mm");
  assert.equal(labelDistance(grid, member, [-4, 4, 4]), null, "outside a crop the distance is unknown");
  const far = labelDistance({ shape: [200, 1, 1], spacingMm: [1, 1, 1] }, i => i === 199, [0, 0, 0], { complete: true, maxMm: 20 });
  assert.deepEqual(far, { inside: false, distanceMm: 20, exact: false });
});

test("Axis-aligned RAS frames preserve physical distance through flips, permutations and LPS sources", () => {
  const lps = [
    [0, 0, -2, 40],
    [1.5, 0, 0, -30],
    [0, 0.75, 0, 12],
    [0, 0, 0, 1],
  ];
  const affine = rasAffine(lps, "LPS+");
  const frame = axisGridFrame(affine, [8, 9, 10]);
  assert.ok(frame);
  assert.deepEqual(frame.grid.spacingMm, [1.5, 0.75, 2]);
  const a = [2, 3, 4], b = [5, 1, 7];
  const ra = transformPoint(affine, a), rb = transformPoint(affine, b);
  const ga = frame.fromRas(ra), gb = frame.fromRas(rb);
  ga.forEach((value, axis) => assert.ok(Math.abs(value - a[axis] * frame.grid.spacingMm[axis]) < 1e-12));
  assert.ok(Math.abs(Math.hypot(...ga.map((v, i) => v - gb[i])) - Math.hypot(...ra.map((v, i) => v - rb[i]))) < 1e-12);
  const angle = Math.PI / 8;
  const oblique = [
    [Math.cos(angle), -Math.sin(angle), 0, 0],
    [Math.sin(angle), Math.cos(angle), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
  assert.equal(axisGridFrame(oblique, [4, 4, 4]), null, "oblique cells are not axis-aligned boxes");
});

test("Operative field local coordinates map to RAS and onto the field grid", () => {
  const field = {
    targetRasMm: [10, -20, 30],
    basisRas: [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
    shape: [11, 11, 21], spacingMm: 0.5, origin: [-2.5, -2.5, -2],
  };
  assert.deepEqual(fieldLocalToRas(field, [0, 0, 0]), [10, -20, 30]);
  assert.deepEqual(fieldLocalToRas(field, [1, 2, 3]), [8, -19, 33]);
  const { grid, fromLocal } = fieldGrid(field);
  assert.deepEqual(grid.spacingMm, [0.5, 0.5, 0.5]);
  assert.deepEqual(fromLocal([-2.5, -2.5, -2]), [0, 0, 0]);
  const q = fromLocal([0, 0, 0]);
  const index = q.map(v => Math.round(v / 0.5));
  assert.deepEqual(index, [5, 5, 4], "the route target is cell (5, 5, 4) of this field");
});

test("A labeled-cell list gives the same outside distance as the ring search", async () => {
  const { labeledCells, nearestListedCellDistance } = await import("./toolNavigation.ts");
  const grid = { shape: [12, 9, 7], spacingMm: [0.8, 0.8, 1.6] };
  const mask = new Uint8Array(12 * 9 * 7);
  for (let i = 0; i < mask.length; i += 37) mask[i] = 1;
  const member = i => mask[i] === 1, cells = labeledCells(mask.length, member);
  assert.equal(cells.length, Math.ceil(mask.length / 37));
  for (const q of [[0.3, 0.2, 0.1], [5.1, 3.3, 4.9], [-6, 20, 3], [9.4, 6.2, 10.2]]) {
    const reference = bruteDistance(grid, member, q);
    assert.ok(Math.abs(nearestListedCellDistance(grid, cells, q) - reference) < 1e-12);
    const viaList = labelDistance(grid, member, q, { complete: true, cells, maxMm: 1e6 });
    const viaRings = labelDistance(grid, member, q, { complete: true, maxMm: 1e6 });
    assert.equal(viaList.inside, viaRings.inside);
    assert.equal(viaList.exact, viaRings.exact);
    assert.ok(Math.abs(viaList.distanceMm - viaRings.distanceMm) < 1e-12, JSON.stringify(q));
  }
});
