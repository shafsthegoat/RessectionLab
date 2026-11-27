import { conditionCavityField } from './operativeCavityField.ts';
import type { OperativeField } from '../operative-field';
import type { VoxelBounds } from './operativeSurface';

type Grid = Pick<OperativeField, 'shape' | 'spacingMm'>;
const clip = (v: number, min: number, max: number) => Math.max(min, Math.min(max, v));
const localIndex = (x: number, y: number, z: number) => (x * 3 + y) * 3 + z;
const near6: number[][] = [],
  near26: number[][] = [];
for (let x = 0; x < 3; x++)
  for (let y = 0; y < 3; y++)
    for (let z = 0; z < 3; z++) {
      const adjacent: number[] = [],
        all: number[] = [];
      for (let xx = Math.max(0, x - 1); xx <= Math.min(2, x + 1); xx++)
        for (let yy = Math.max(0, y - 1); yy <= Math.min(2, y + 1); yy++)
          for (let zz = Math.max(0, z - 1); zz <= Math.min(2, z + 1); zz++) {
            const j = localIndex(xx, yy, zz),
              distance = Math.abs(x - xx) + Math.abs(y - yy) + Math.abs(z - zz);
            if (j === 13 || !distance) continue;
            all.push(j);
            if (distance === 1) adjacent.push(j);
          }
      near6.push(adjacent);
      near26.push(all);
    }
/** Local digital-topology predicate. Test both the solid (6-connected) and
 * complement (26-connected) before each sequential boundary change. This
 * retains components, tunnels and enclosed cavities through the fairing. */
function simplePoint(neighborhood: Uint8Array, work: Uint8Array, queue: Uint8Array) {
  work.set(neighborhood);
  for (const bit of [1, 0]) {
    const adjacent = bit ? near6 : near26;
    let count = 0,
      seed = -1;
    for (let i = 0; i < 27; i++)
      if (i !== 13 && work[i] === bit) {
        seed = i;
        count++;
      }
    if (!count) return false;
    let end = 1;
    queue[0] = seed;
    work[seed] = 2;
    for (let cursor = 0; cursor < end; cursor++)
      for (const j of adjacent[queue[cursor]])
        if (work[j] === bit) {
          work[j] = 2;
          queue[end++] = j;
        }
    if (end !== count) return false;
  }
  return true;
}
function boxAxis(input: Uint8Array, output: Uint8Array, shape: number[], axis: number, radius: number) {
  const original = [shape[1] * shape[2], shape[2], 1],
    length = shape[axis],
    step = original[axis],
    size = 2 * radius + 1;
  const u = (axis + 1) % 3,
    v = (axis + 2) % 3;
  for (let a = 0; a < shape[u]; a++)
    for (let b = 0; b < shape[v]; b++) {
      const base = a * original[u] + b * original[v];
      let sum = 0;
      for (let d = -radius; d <= radius; d++) sum += input[base + clip(d, 0, length - 1) * step];
      for (let p = 0; p < length; p++) {
        output[base + p * step] = Math.round(sum / size);
        sum +=
          input[base + Math.min(p + radius + 1, length - 1) * step] -
          input[base + Math.max(p - radius, 0) * step];
      }
    }
}
export const operativeFilterRadius = (spacingMm: number, thin: boolean) =>
  Math.max(1, Math.min(3, Math.round((thin ? 0.6 : 0.87) / spacingMm)));

/** Display-only occupancy and edge fractions. The finite two-box filter has
 * a physical width set in mm. The selected source voxels remain untouched. */
export function operativeSurfaceField(grid: Grid, source: Uint8Array, bounds: VoxelBounds, thin: boolean, preserveOccupancy = false) {
  const radius = operativeFilterRadius(grid.spacingMm, thin),
    ghost = radius * 2 + 3;
  const origin = bounds.low.map((v) => v - ghost),
    shape = bounds.high.map((v, a) => v + ghost + 2 - origin[a]);
  const [sx, sy, sz] = shape,
    [nx, ny, nz] = grid.shape,
    yz = sy * sz,
    size = sx * yz;
  const mask = new Uint8Array(size),
    raw = new Uint8Array(size),
    scratch = new Uint8Array(size);
  // Extend acquired field edges continuously for conditioning, then close
  // them on their exact physical crop plane in the extractor.
  for (let x = 0, i = 0; x < sx; x++)
    for (let y = 0; y < sy; y++) {
      const row = (clip(x + origin[0], 0, nx - 1) * ny + clip(y + origin[1], 0, ny - 1)) * nz;
      for (let z = 0; z < sz; z++, i++) {
        const bit = source[row + clip(z + origin[2], 0, nz - 1)];
        mask[i] = bit;
        raw[i] = bit * 255;
      }
    }
  let smooth = raw,
    buffer = scratch;
  for (let pass = 0; pass < 2; pass++)
    for (let axis = 0; axis < 3; axis++) {
      boxAxis(smooth, buffer, shape, axis, radius);
      [smooth, buffer] = [buffer, smooth];
    }
  const offsets: number[] = [];
  for (let dx = -1; dx <= 1; dx++)
    for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) offsets.push(dx * yz + dy * sz + dz);
  const face = [-yz, yz, -sz, sz, -1, 1],
    window = new Uint8Array(27),
    work = new Uint8Array(27),
    queue = new Uint8Array(27);
  if (!thin && !preserveOccupancy) {
    // Consider only the original boundary shell. Move a sample only when a
    // two-voxel-deep solid and exterior support its locally dominant normal.
    // This rejects eroding a thin bridge or filling a narrow fluid gap.
    const candidates: number[][] = Array.from({ length: 8 }, () => []);
    for (let x = ghost - 1; x < sx - ghost + 1; x++)
      for (let y = ghost - 1; y < sy - ghost + 1; y++)
        for (let z = ghost - 1; z < sz - ghost + 1; z++) {
          const i = x * yz + y * sz + z,
            bit = mask[i];
          if (bit === Number(smooth[i] >= 128)) continue;
          if (!face.some((d) => mask[i + d] !== bit)) continue;
          const gx = smooth[i + yz] - smooth[i - yz],
            gy = smooth[i + sz] - smooth[i - sz],
            gz = smooth[i + 1] - smooth[i - 1];
          const ax = Math.abs(gx),
            ay = Math.abs(gy),
            az = Math.abs(gz),
            step = ax >= ay && ax >= az ? Math.sign(gx) * yz : ay >= az ? Math.sign(gy) * sz : Math.sign(gz);
          if (!step || !mask[i + 2 * step] || mask[i - 2 * step]) continue;
          const color = ((x + origin[0]) & 1) | (((y + origin[1]) & 1) << 1) | (((z + origin[2]) & 1) << 2);
          candidates[color].push(i);
        }
    for (const batch of candidates)
      for (const i of batch) {
        if (!face.some((d) => mask[i + d] === 1) || !face.some((d) => mask[i + d] === 0)) continue;
        for (let j = 0; j < 27; j++) window[j] = mask[i + offsets[j]];
        if (simplePoint(window, work, queue)) mask[i] ^= 1;
      }
  }
  // Retain sign and a finite edge margin for features below the filter width.
  // Their silhouette and closed interior survive even if the low-pass value
  // on both sides of a thin sheet is below the conventional 0.5 isovalue.
  for (let i = 0; i < size; i++) {
    const value = smooth[i],
      inside = mask[i];
    buffer[i] = Number(inside !== Number(value >= 128));
    if (inside && value < 128) smooth[i] = thin ? 176 : 144;
    if (!inside && value >= 128) smooth[i] = thin ? 79 : 111;
  }
  const field = {
    mask,
    smooth,
    unresolved: buffer,
    yz,
    sz,
    shape,
    origin,
    base: -origin[0] * yz - origin[1] * sz - origin[2],
  };
  conditionCavityField(source, field);
  return field;
}

export type ConditionedField = ReturnType<typeof operativeSurfaceField>;
export function copyConditionedRegion(to: ConditionedField, from: ConditionedField, region: VoxelBounds) {
  const [x0, y0, z0] = region.low,
    [x1, y1, z1] = region.high;
  if (x1 < x0 || y1 < y0 || z1 < z0) return;
  for (let x = x0; x <= x1; x++)
    for (let y = y0; y <= y1; y++) {
      const a = from.base + x * from.yz + y * from.sz + z0,
        b = to.base + x * to.yz + y * to.sz + z0;
      to.mask.set(from.mask.subarray(a, a + z1 - z0 + 1), b);
      to.smooth.set(from.smooth.subarray(a, a + z1 - z0 + 1), b);
      to.unresolved.set(from.unresolved.subarray(a, a + z1 - z0 + 1), b);
    }
}
