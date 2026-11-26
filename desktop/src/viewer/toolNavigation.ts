/** Instrument navigation geometry: tip-to-label distances on sampled grids.
 *
 * Every distance here is an exact point-to-cell computation on the sampled
 * label grid. A label cell is the axis-aligned box of one source cell; the
 * boundary of a label is the surface of the union of its cells. No smoothing,
 * interpolation or surface fitting is applied, so the result describes the
 * sampled label, not the true anatomy. */
import type {Affine,Point3} from './coordinates';

/** C-order grid (last axis fastest) whose cell (0,0,0) is centered at the
 * frame origin and whose axes are orthogonal with these physical spacings. */
export interface AxisGrid {
  shape: readonly [number, number, number];
  spacingMm: readonly [number, number, number];
}
export type Membership = (flatIndex: number) => boolean;

export interface LabelDistance {
  /** True when the point lies in a labeled cell. */
  inside: boolean;
  /** Unsigned distance in mm to the label boundary. */
  distanceMm: number;
  /** False when unsampled space or the search limit leaves only a lower
   * bound: the true distance is at least `distanceMm`. */
  exact: boolean;
}

export interface DistanceOptions {
  /** True when nothing outside the grid can belong to the label, as for a
   * whole acquisition. A local crop leaves the outside unknown. */
  complete?: boolean;
  /** Search limit in mm. Farther boundaries are reported as a lower bound. */
  maxMm?: number;
  /** Optional flat indices of every labeled cell. Outside the label the
   * nearest cell is then found by one pass over the list instead of a ring
   * search, which stays fast when the label is far away. */
  cells?: ArrayLike<number>;
}

const cellGap = (q: number, center: number, half: number) => Math.max(0, Math.abs(q - center) - half);

/** Smallest distance from `q` (grid frame, mm) to a cell satisfying `accept`,
 * searching Chebyshev index rings outward until no unvisited cell can be
 * closer, or until `maxMm` is exceeded. */
export function nearestCellDistance(grid: AxisGrid, q: readonly number[], accept: Membership, maxMm: number): number {
  const [nx, ny, nz] = grid.shape, [sx, sy, sz] = grid.spacingMm;
  const minSpacing = Math.min(sx, sy, sz);
  const clampIndex = (value: number, n: number) => Math.max(0, Math.min(n - 1, value));
  const cx = clampIndex(Math.round(q[0] / sx), nx), cy = clampIndex(Math.round(q[1] / sy), ny), cz = clampIndex(Math.round(q[2] / sz), nz);
  const maxRing = Math.max(nx, ny, nz);
  let best = Infinity;
  const visit = (i: number, j: number, k: number) => {
    if (!accept((i * ny + j) * nz + k)) return;
    const d = Math.hypot(cellGap(q[0], i * sx, sx / 2), cellGap(q[1], j * sy, sy / 2), cellGap(q[2], k * sz, sz / 2));
    if (d < best) best = d;
  };
  for (let ring = 0; ring <= maxRing; ring++) {
    // Cells on ring r are at least (r - 1) minimum spacings away; every
    // unvisited cell after this ring is at least r spacings away.
    if ((ring - 1) * minSpacing > Math.min(best, maxMm)) break;
    const i0 = Math.max(0, cx - ring), i1 = Math.min(nx - 1, cx + ring);
    const j0 = Math.max(0, cy - ring), j1 = Math.min(ny - 1, cy + ring);
    const k0 = Math.max(0, cz - ring), k1 = Math.min(nz - 1, cz + ring);
    for (let i = i0; i <= i1; i++) {
      const edgeI = Math.abs(i - cx) === ring;
      for (let j = j0; j <= j1; j++) {
        if (edgeI || Math.abs(j - cy) === ring) {
          for (let k = k0; k <= k1; k++) visit(i, j, k);
        } else {
          if (cz - ring >= 0) visit(i, j, cz - ring);
          if (ring > 0 && cz + ring < nz) visit(i, j, cz + ring);
        }
      }
    }
    if (i0 === 0 && j0 === 0 && k0 === 0 && i1 === nx - 1 && j1 === ny - 1 && k1 === nz - 1) break;
  }
  return best <= maxMm ? best : Infinity;
}

/** Smallest exact point-to-cell distance over a list of flat cell indices. */
export function nearestListedCellDistance(grid: AxisGrid, cells: ArrayLike<number>, q: readonly number[]): number {
  const [, ny, nz] = grid.shape, [sx, sy, sz] = grid.spacingMm, plane = ny * nz;
  let best = Infinity;
  for (let n = 0; n < cells.length; n++) {
    const flat = cells[n], i = Math.floor(flat / plane), j = Math.floor(flat / nz) % ny, k = flat % nz;
    const dx = cellGap(q[0], i * sx, sx / 2), dy = cellGap(q[1], j * sy, sy / 2), dz = cellGap(q[2], k * sz, sz / 2);
    const squared = dx * dx + dy * dy + dz * dz;
    if (squared < best) best = squared;
  }
  return Math.sqrt(best);
}

/** Flat indices of every labeled cell, for repeated distance queries. */
export function labeledCells(count: number, member: Membership): Int32Array {
  let total = 0;
  for (let i = 0; i < count; i++) if (member(i)) total++;
  const cells = new Int32Array(total);
  for (let i = 0, n = 0; i < count; i++) if (member(i)) cells[n++] = i;
  return cells;
}

/** Distance from a grid-frame point to the surface of the grid box. */
function edgeDistance(grid: AxisGrid, q: readonly number[]): number {
  let edge = Infinity;
  for (let a = 0; a < 3; a++) {
    const s = grid.spacingMm[a], low = -s / 2, high = (grid.shape[a] - .5) * s;
    if (q[a] < low || q[a] > high) return 0;
    edge = Math.min(edge, q[a] - low, high - q[a]);
  }
  return edge;
}

/** Signed distance from a point to the boundary of a sampled label. Negative
 * values are inside the label. Unknown space never shortens a reported
 * distance: it turns the value into a lower bound. */
export function labelDistance(grid: AxisGrid, member: Membership, q: readonly number[], options: DistanceOptions = {}): LabelDistance | null {
  const complete = options.complete ?? false, maxMm = options.maxMm ?? 60;
  const [nx, ny, nz] = grid.shape;
  if (!(nx >= 1 && ny >= 1 && nz >= 1) || !q.every(Number.isFinite)) return null;
  const edge = edgeDistance(grid, q);
  // Outside a local crop nothing bounds the distance in either direction.
  if (!complete && edge === 0) return null;
  const index = q.map((value, a) => Math.round(value / grid.spacingMm[a]));
  const inGrid = edge > 0 && index.every((value, a) => value >= 0 && value < grid.shape[a]);
  const inside = inGrid && member((index[0] * ny + index[1]) * nz + index[2]);
  if (inside) {
    const out = nearestCellDistance(grid, q, flat => !member(flat), maxMm);
    // Leaving the grid leaves the label in a complete grid; in a crop the
    // label may continue, so the edge only bounds the distance from below.
    if (complete) {
      const nearest = Math.min(out, edge);
      return nearest <= maxMm ? {inside, distanceMm: nearest, exact: true} : {inside, distanceMm: maxMm, exact: false};
    }
    if (out <= edge) return {inside, distanceMm: out, exact: true};
    return {inside, distanceMm: Math.min(edge, maxMm), exact: false};
  }
  const listed = options.cells ? nearestListedCellDistance(grid, options.cells, q) : null;
  const found = listed === null ? nearestCellDistance(grid, q, member, maxMm) : listed <= maxMm ? listed : Infinity;
  if (complete || found <= edge) return found <= maxMm ? {inside, distanceMm: found, exact: true} : {inside, distanceMm: maxMm, exact: false};
  return {inside, distanceMm: Math.min(edge, maxMm), exact: false};
}

/** Convert patient RAS to an axis grid frame when the voxel axes are parallel
 * to RAS axes (any permutation, flip or spacing). Oblique grids return null:
 * their cells are not axis-aligned boxes and point-to-box would be inexact. */
export function axisGridFrame(rasAffineMatrix: Affine, shape: readonly [number, number, number], tolerance = 1e-6) {
  const rows: number[] = [], spacing: number[] = [], scale: number[] = [];
  for (let j = 0; j < 3; j++) {
    const column = [0, 1, 2].map(r => rasAffineMatrix[r][j]);
    const norm = Math.hypot(...column);
    if (!(norm > 0)) return null;
    const row = column.findIndex(value => Math.abs(value) > norm * (1 - tolerance));
    if (row < 0 || rows.includes(row) || column.some((value, r) => r !== row && Math.abs(value) > norm * tolerance)) return null;
    rows.push(row); spacing.push(Math.abs(column[row])); scale.push(column[row]);
  }
  const translation = [0, 1, 2].map(r => rasAffineMatrix[r][3]);
  const grid: AxisGrid = {shape: [...shape] as [number, number, number], spacingMm: spacing as [number, number, number]};
  return {
    grid,
    /** Grid-frame millimetres: distance along each voxel axis from cell 0. */
    fromRas: (ras: Point3): Point3 => [0, 1, 2].map(j => (ras[rows[j]] - translation[rows[j]]) / scale[j] * spacing[j]) as Point3,
  };
}

/** Operative field local millimetres (x, y across, z along the route) to
 * patient RAS: target + basis^T local. */
export function fieldLocalToRas(field: {targetRasMm: Point3; basisRas: readonly Point3[]}, local: readonly number[]): Point3 {
  return field.targetRasMm.map((value, i) => value + field.basisRas.reduce((sum, axis, j) => sum + axis[i] * local[j], 0)) as Point3;
}

/** The operative label grid in its own local frame (isotropic spacing). */
export function fieldGrid(field: {shape: readonly number[]; spacingMm: number; origin: readonly number[]}) {
  const grid: AxisGrid = {shape: [field.shape[0], field.shape[1], field.shape[2]], spacingMm: [field.spacingMm, field.spacingMm, field.spacingMm]};
  return {grid, fromLocal: (local: readonly number[]): Point3 => [0, 1, 2].map(a => local[a] - field.origin[a]) as Point3};
}

/** Compact signed text, e.g. "−2.4 mm" inside or "≥ 8.0 mm" for a bound. */
export function formatLabelDistance(value: LabelDistance): string {
  const magnitude = value.distanceMm.toFixed(1);
  const signed = value.inside && value.distanceMm > 0 ? `−${magnitude}` : magnitude;
  return `${value.exact ? '' : value.inside ? '≤ ' : '≥ '}${signed} mm`;
}
