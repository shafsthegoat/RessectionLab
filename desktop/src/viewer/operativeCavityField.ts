import type { OperativeField } from '../operative-field';
import type { VoxelBounds } from './operativeSurface';

type Grid = Pick<OperativeField, 'shape' | 'spacingMm'>;
type Cavity = {
  step: number;
  path: number[];
  radius: number;
  ny: number;
  nz: number;
  x0: number;
  y0: number;
  z0: number;
  sy: number;
  sz: number;
  original: Uint8Array;
  distanceSquared: Float32Array;
};
const cavities = new WeakMap<Uint8Array, Cavity>();

/** Store only the local editable envelope. The original mask identifies which
 * side of a changed boundary was eligible target before the instrument moved. */
export function registerCavityField(
  grid: Grid,
  mask: Uint8Array,
  region: VoxelBounds,
  path: number[],
  radiusMm: number
) {
  if (!path.length) return;
  const [, ny, nz] = grid.shape;
  const [x0, y0, z0] = region.low.map((v) => Math.max(0, v - 2));
  const [x1, y1, z1] = region.high.map((v, i) => Math.min(grid.shape[i] - 1, v + 2));
  const sy = y1 - y0 + 1,
    sz = z1 - z0 + 1,
    original = new Uint8Array((x1 - x0 + 1) * sy * sz);
  for (let x = x0; x <= x1; x++)
    for (let y = y0; y <= y1; y++) {
      const i = (x * ny + y) * nz + z0;
      original.set(mask.subarray(i, i + sz), ((x - x0) * sy + y - y0) * sz);
    }
  cavities.set(mask, {
    step: 0,
    path,
    radius: radiusMm / grid.spacingMm,
    ny,
    nz,
    x0,
    y0,
    z0,
    sy,
    sz,
    original,
    distanceSquared: new Float32Array(original.length).fill(Infinity),
  });
}

/** The original brush visits face-adjacent integer centers. The nearest
 * position on their connecting segment, evaluated at any lattice sample,
 * is an endpoint. This incremental minimum therefore samples the swept tip
 * distance exactly. Backward scrubbing replays only this local buffer. */
export function setCavityStep(mask: Uint8Array, step: number, dirty?: VoxelBounds) {
  const c = cavities.get(mask);
  if (!c) return;
  step = Math.min(c.path.length, Math.max(0, step));
  if (step === c.step) return;
  const margin = Math.ceil(c.radius + 1);
  // A pass through already removed samples can shift the continuous wall
  // without flipping a center. Invalidate all changed brush footprints.
  if (dirty)
    for (let n = Math.min(step, c.step); n < Math.max(step, c.step); n++) {
      const k = c.path[n],
        center = [Math.floor(k / (c.ny * c.nz)), Math.floor(k / c.nz) % c.ny, k % c.nz];
      for (let axis = 0; axis < 3; axis++) {
        dirty.low[axis] = Math.min(dirty.low[axis], center[axis] - margin);
        dirty.high[axis] = Math.max(dirty.high[axis], center[axis] + margin);
      }
    }
  if (step < c.step) {
    c.distanceSquared.fill(Infinity);
    c.step = 0;
  }
  const sx = c.original.length / (c.sy * c.sz);
  for (; c.step < step; c.step++) {
    const k = c.path[c.step],
      xx = Math.floor(k / (c.ny * c.nz)) - c.x0,
      yy = (Math.floor(k / c.nz) % c.ny) - c.y0,
      zz = (k % c.nz) - c.z0;
    for (let x = Math.max(0, xx - margin); x <= Math.min(sx - 1, xx + margin); x++)
      for (let y = Math.max(0, yy - margin); y <= Math.min(c.sy - 1, yy + margin); y++) {
        const dxy = (x - xx) ** 2 + (y - yy) ** 2,
          base = (x * c.sy + y) * c.sz;
        for (let z = Math.max(0, zz - margin); z <= Math.min(c.sz - 1, zz + margin); z++) {
          const d = dxy + (z - zz) ** 2;
          if (d < c.distanceSquared[base + z]) c.distanceSquared[base + z] = d;
        }
      }
  }
}

type Samples = {
  mask: Uint8Array;
  smooth: Uint8Array;
  unresolved: Uint8Array;
  origin: number[];
  shape: number[];
  yz: number;
  sz: number;
  base: number;
};

/** Fit the displayed interior wall to the actual brush, preserving every
 * accepted sign and leaving adjacent protected, absent and intact support
 * alone. A mask-wide blur otherwise treats this submillimetre lumen like a
 * small anatomical feature and can close part of the accepted opening. */
export function conditionCavityField(source: Uint8Array, field: Samples) {
  const c = cavities.get(source);
  if (!c?.step) return;
  const sx = c.original.length / (c.sy * c.sz);
  const x0 = Math.max(c.x0, field.origin[0]),
    y0 = Math.max(c.y0, field.origin[1]),
    z0 = Math.max(c.z0, field.origin[2]);
  const x1 = Math.min(c.x0 + sx, field.origin[0] + field.shape[0]),
    y1 = Math.min(c.y0 + c.sy, field.origin[1] + field.shape[1]),
    z1 = Math.min(c.z0 + c.sz, field.origin[2] + field.shape[2]);
  const removed = (k: number, i: number) => !!c.original[k] && !source[i];
  const yz = c.ny * c.nz,
    localYZ = c.sy * c.sz;
  for (let x = x0; x < x1; x++)
    for (let y = y0; y < y1; y++)
      for (let z = z0; z < z1; z++) {
        const k = ((x - c.x0) * c.sy + y - c.y0) * c.sz + z - c.z0;
        if (!c.original[k] || c.distanceSquared[k] > (c.radius + 1) ** 2) continue;
        const i = x * yz + y * c.nz + z;
        if (
          source[i] &&
          !(x > c.x0 && removed(k - localYZ, i - yz)) &&
          !(x + 1 < c.x0 + sx && removed(k + localYZ, i + yz)) &&
          !(y > c.y0 && removed(k - c.sz, i - c.nz)) &&
          !(y + 1 < c.y0 + c.sy && removed(k + c.sz, i + c.nz)) &&
          !(z > c.z0 && removed(k - 1, i - 1)) &&
          !(z + 1 < c.z0 + c.sz && removed(k + 1, i + 1))
        )
          continue;
        const j = field.base + x * field.yz + y * field.sz + z;
        const distance = Math.sqrt(c.distanceSquared[k]) - c.radius;
        const scalar = Math.round(127.5 + 100 * distance);
        field.mask[j] = source[i];
        field.smooth[j] = source[i]
          ? Math.min(255, Math.max(129, scalar))
          : Math.max(0, Math.min(126, scalar));
        field.unresolved[j] = 0;
      }
}
