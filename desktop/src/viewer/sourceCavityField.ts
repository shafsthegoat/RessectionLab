/** Smooth display field of the engine's removed tissue and its cavity wall.
 *
 * The engine removes whole working-grid cells (and CUSA keeps exact pending
 * work on partly worked cells, `cusaProgressV1`). The display samples that
 * occupancy on a lattice of half a cell around the changed region, softens it
 * with a narrow Gaussian (σ = 0.25 cell) and takes its 0.5 level as the cavity boundary:
 * source surfaces are clipped where the field exceeds 0.5 and the cavity wall
 * is that level set. An isolated removed cell still rises above
 * 0.5 at its centre and a straight boundary stays on the cell face, so the
 * wall stays within half a cell of the engine's removed boundary. Counts,
 * cells and picking stay with the engine. */
import type {Shape3} from './coordinates';

export const CAVITY_STEP = 0.5;
export const CAVITY_SIGMA_CELLS = 0.25;
export const CAVITY_LEVEL = 0.5;

export interface CavityField {
  /** Lattice values in [0, 1], C order (x slowest); voxel position = origin + index·step. */
  values: Float32Array;
  size: [number, number, number];
  origin: [number, number, number];
  step: number;
}

/** Removal occupancy per cell: 1 for removed cells, the worked fraction for pending CUSA cells. */
export function removalOccupancy(removed: Iterable<number>, progress?: {cells: number[]; fraction: number[]} | null,
  include?: (cell: number) => boolean) {
  const occupancy = new Map<number, number>();
  for (const cell of removed) if (!include || include(cell)) occupancy.set(cell, 1);
  if (progress) progress.cells.forEach((cell, i) => {
    const f = Math.max(0, Math.min(1, progress.fraction[i] ?? 0));
    if (f > 0 && !occupancy.has(cell) && (!include || include(cell))) occupancy.set(cell, f);
  });
  return occupancy;
}

/** Sample the occupancy (each cell a unit cube) on the half-cell lattice and blur it. */
export function cavityField(occupancy: ReadonlyMap<number, number>, shape: Shape3, step = CAVITY_STEP, sigma = CAVITY_SIGMA_CELLS, margin = 2): CavityField | null {
  if (!occupancy.size) return null;
  const [, ny, nz] = shape;
  const low = [Infinity, Infinity, Infinity], high = [-Infinity, -Infinity, -Infinity];
  for (const cell of occupancy.keys()) {
    const p = [Math.floor(cell / (ny * nz)), Math.floor(cell / nz) % ny, cell % nz];
    for (let a = 0; a < 3; a++) { low[a] = Math.min(low[a], p[a]); high[a] = Math.max(high[a], p[a]); }
  }
  const origin = low.map((v) => v - margin) as [number, number, number];
  const size = high.map((v, a) => Math.round((v + margin - origin[a]) / step) + 1) as [number, number, number];
  const [sx, sy, sz] = size, values = new Float32Array(sx * sy * sz);
  // Each lattice point takes the occupancy of the cell containing it (cell faces at ±0.5).
  // A point exactly on a face takes half of each side, so the blurred level stays on the face.
  const per = Math.round(1 / step);
  for (const [cell, value] of occupancy) {
    const p = [Math.floor(cell / (ny * nz)), Math.floor(cell / nz) % ny, cell % nz];
    const first = p.map((v, a) => Math.round((v - 0.5 - origin[a]) / step));
    for (let i = 0; i <= per; i++) for (let j = 0; j <= per; j++) for (let k = 0; k <= per; k++) {
      const face = (i === 0 || i === per ? 0.5 : 1) * (j === 0 || j === per ? 0.5 : 1) * (k === 0 || k === per ? 0.5 : 1);
      const x = first[0] + i, y = first[1] + j, z = first[2] + k;
      if (x < 0 || y < 0 || z < 0 || x >= sx || y >= sy || z >= sz) continue;
      const index = (x * sy + y) * sz + z;
      // Faces and corners shared with a neighbour accumulate both halves.
      values[index] = Math.min(1, values[index] + value * face);
    }
  }
  blur(values, size, sigma / step);
  return {values, size, origin, step};
}

function blur(values: Float32Array, size: [number, number, number], sigma: number) {
  const radius = Math.max(1, Math.ceil(3 * sigma)), kernel = new Float32Array(2 * radius + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) sum += kernel[i + radius] = Math.exp(-(i * i) / (2 * sigma * sigma));
  for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;
  const [sx, sy, sz] = size, strides = [sy * sz, sz, 1], line = new Float32Array(Math.max(sx, sy, sz));
  for (let axis = 0; axis < 3; axis++) {
    const n = size[axis], stride = strides[axis];
    const others = [0, 1, 2].filter((a) => a !== axis);
    for (let u = 0; u < size[others[0]]; u++) for (let v = 0; v < size[others[1]]; v++) {
      const base = u * strides[others[0]] + v * strides[others[1]];
      for (let i = 0; i < n; i++) line[i] = values[base + i * stride];
      for (let i = 0; i < n; i++) {
        let acc = 0;
        for (let k = -radius; k <= radius; k++) {
          const j = i + k;
          if (j >= 0 && j < n) acc += kernel[k + radius] * line[j];
        }
        values[base + i * stride] = acc;
      }
    }
  }
}

/** Trilinear sample of the field at a voxel position; 0 outside. */
export function sampleCavity(field: CavityField, p: readonly number[]) {
  const [sx, sy, sz] = field.size;
  const g = [0, 1, 2].map((a) => (p[a] - field.origin[a]) / field.step);
  if (g.some((v, a) => v < 0 || v > field.size[a] - 1)) return 0;
  const b = g.map((v, a) => Math.min(Math.floor(v), field.size[a] - 2)), f = g.map((v, a) => v - b[a]);
  let out = 0;
  for (let dx = 0; dx <= 1; dx++) for (let dy = 0; dy <= 1; dy++) for (let dz = 0; dz <= 1; dz++) {
    const w = (dx ? f[0] : 1 - f[0]) * (dy ? f[1] : 1 - f[1]) * (dz ? f[2] : 1 - f[2]);
    out += w * field.values[((b[0] + dx) * sy + b[1] + dy) * sz + b[2] + dz];
  }
  return out;
}

/** Naive surface nets on the field's level set: one vertex per sign-changing
 * lattice cube (the mean of its edge crossings), quads across crossing edges.
 * Output positions are voxel coordinates; normals point out of the removed region. */
export function cavitySurface(field: CavityField, level = CAVITY_LEVEL, keepVertex?: (p: Point, normal: Point) => boolean) {
  const [sx, sy, sz] = field.size, values = field.values;
  const at = (x: number, y: number, z: number) => values[(x * sy + y) * sz + z];
  const vertexOf = new Int32Array((sx - 1) * (sy - 1) * (sz - 1)).fill(-1);
  const positions: number[] = [], normals: number[] = [];
  const corners = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
  const edges = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
  const keep: boolean[] = [];
  const above = new Uint8Array(values.length);
  for (let i = 0; i < values.length; i++) above[i] = values[i] > level ? 1 : 0;
  const yz = sy * sz;
  for (let x = 0; x < sx - 1; x++) for (let y = 0; y < sy - 1; y++) for (let z = 0; z < sz - 1; z++) {
    // Uniform cubes (almost all of them) are skipped without allocation.
    const o = x * yz + y * sz + z;
    const sum = above[o] + above[o + 1] + above[o + sz] + above[o + sz + 1] + above[o + yz] + above[o + yz + 1] + above[o + yz + sz] + above[o + yz + sz + 1];
    if (sum === 0 || sum === 8) continue;
    const v = corners.map(([i, j, k]) => at(x + i, y + j, z + k) - level);
    let inside = 0;
    for (const value of v) if (value > 0) inside++;
    if (inside === 0 || inside === 8) continue;
    let px = 0, py = 0, pz = 0, count = 0;
    for (const [a, b] of edges) {
      if ((v[a] > 0) === (v[b] > 0)) continue;
      const t = v[a] / (v[a] - v[b]);
      px += corners[a][0] + t * (corners[b][0] - corners[a][0]);
      py += corners[a][1] + t * (corners[b][1] - corners[a][1]);
      pz += corners[a][2] + t * (corners[b][2] - corners[a][2]);
      count++;
    }
    // Gradient of the occupancy points into the cavity; the wall normal points out of it.
    const gx = (v[1] + v[3] + v[5] + v[7]) - (v[0] + v[2] + v[4] + v[6]);
    const gy = (v[2] + v[3] + v[6] + v[7]) - (v[0] + v[1] + v[4] + v[5]);
    const gz = (v[4] + v[5] + v[6] + v[7]) - (v[0] + v[1] + v[2] + v[3]);
    const len = Math.hypot(gx, gy, gz) || 1;
    const p: Point = [field.origin[0] + (x + px / count) * field.step, field.origin[1] + (y + py / count) * field.step, field.origin[2] + (z + pz / count) * field.step];
    const n: Point = [-gx / len, -gy / len, -gz / len];
    vertexOf[(x * (sy - 1) + y) * (sz - 1) + z] = positions.length / 3;
    positions.push(...p);
    normals.push(...n);
    keep.push(!keepVertex || keepVertex(p, n));
  }
  const indices: number[] = [];
  const cube = (x: number, y: number, z: number) =>
    x < 0 || y < 0 || z < 0 || x >= sx - 1 || y >= sy - 1 || z >= sz - 1 ? -1 : vertexOf[(x * (sy - 1) + y) * (sz - 1) + z];
  const quad = (a: number, b: number, c: number, d: number, flip: boolean) => {
    if (a < 0 || b < 0 || c < 0 || d < 0) return;
    if (!keep[a] || !keep[b] || !keep[c] || !keep[d]) return;
    if (flip) indices.push(a, c, b, a, d, c);
    else indices.push(a, b, c, a, c, d);
  };
  for (let x = 0; x < sx; x++) for (let y = 0; y < sy; y++) for (let z = 0; z < sz; z++) {
    const o = x * yz + y * sz + z, here = above[o], inside = here === 1;
    if (x + 1 < sx && here !== above[o + yz]) quad(cube(x, y - 1, z - 1), cube(x, y, z - 1), cube(x, y, z), cube(x, y - 1, z), inside);
    if (y + 1 < sy && here !== above[o + sz]) quad(cube(x - 1, y, z - 1), cube(x - 1, y, z), cube(x, y, z), cube(x, y, z - 1), inside);
    if (z + 1 < sz && here !== above[o + 1]) quad(cube(x - 1, y - 1, z), cube(x, y - 1, z), cube(x, y, z), cube(x - 1, y, z), inside);
  }
  return {positions: Float32Array.from(positions), normals: Float32Array.from(normals), indices: Uint32Array.from(indices)};
}
type Point = [number, number, number];

/** Distance (cells) from a voxel point to the boundary of the removed cell set: the nearest face between a removed and a kept cell. */
export function distanceToRemovedBoundary(p: readonly number[], removed: ReadonlySet<number>, shape: Shape3, reach = 2) {
  const [nx, ny, nz] = shape;
  const id = (x: number, y: number, z: number) => (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz ? -1 : (x * ny + y) * nz + z);
  const c = p.map(Math.round);
  let best = Infinity;
  for (let dx = -reach; dx <= reach; dx++) for (let dy = -reach; dy <= reach; dy++) for (let dz = -reach; dz <= reach; dz++) {
    const x = c[0] + dx, y = c[1] + dy, z = c[2] + dz, here = removed.has(id(x, y, z));
    for (const [ax, ay, az] of [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) {
      if (here === removed.has(id(x + ax, y + ay, z + az))) continue;
      // Unit face between the two cells, centred at their midpoint.
      const f = [x + ax / 2, y + ay / 2, z + az / 2], axis = ax ? 0 : ay ? 1 : 2;
      const d = [0, 1, 2].map((a) => (a === axis ? Math.abs(p[a] - f[a]) : Math.max(0, Math.abs(p[a] - f[a]) - 0.5)));
      best = Math.min(best, Math.hypot(d[0], d[1], d[2]));
    }
  }
  return best;
}
