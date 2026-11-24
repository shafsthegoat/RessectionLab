import {guardCavityPatch} from './cavityPatchSafety.ts';
import type {Affine, Shape3} from './coordinates';

export const NATIVE_BRAIN_SURFACE_POLICY = 'engine-native-brain-free-surface-v1';
const MAX_TRIANGLES = 100_000;
const MAX_SUPPORT_SAMPLES = 40_000_000;
const MAX_MM = .4;
const TAPER_MM = .9;

/** Round the free brain surface of a checked native-wall crop. The accepted
 * owner grid and its existing cavity, interface and crop geometry stay fixed. */
export function fairNativeBrainSurface(raw:Float32Array, shown:Float32Array, unjoined:Float32Array, normals:Float32Array,
  indices:Uint32Array, owner:Uint16Array, mask:Uint8Array, region:number, shape:Shape3, affine:Affine,
  sourceDisplacementMm:number) {
  if (!indices.length || indices.length / 3 > MAX_TRIANGLES || sourceDisplacementMm > MAX_MM) return;
  const spacing = [0, 1, 2].map(axis => Math.hypot(affine[0][axis], affine[1][axis], affine[2][axis]));
  const limit = Math.min(MAX_MM, .75 * Math.min(...spacing));
  if (sourceDisplacementMm > limit) return;
  // Native angiography wall crops use an orthogonal physical lattice. In
  // these coordinates scaling by the three spacings preserves RAS distances.
  for (let axis = 0; axis < 3; axis++) for (let peer = axis + 1; peer < 3; peer++) {
    const dot = affine[0][axis] * affine[0][peer] + affine[1][axis] * affine[1][peer] + affine[2][axis] * affine[2][peer];
    if (Math.abs(dot) > 1e-5 * spacing[axis] * spacing[peer]) return;
  }
  const [nx, ny, nz] = shape, yz = ny * nz;
  const [sx, sy, sz] = spacing;
  const fixed = new Uint8Array(owner.length);
  for (let i = 0; i < owner.length; i++) {
    if (!!mask[i] !== (owner[i] === region)) fixed[i] = 2;
    else if (owner[i] && owner[i] !== region) fixed[i] = 1;
  }
  const interfaceMm = Math.max(1.5, Math.hypot(sx, sy, sz) + limit);
  const editMm = 2;
  const search = Math.max(interfaceMm, editMm) + TAPER_MM;
  const count = raw.length / 3, mobility = new Float32Array(count);
  const cavity = shown === unjoined ? undefined : new Uint8Array(shown.length / 3);
  if (cavity) {
    cavity.fill(1, count);
    for (let v = 0; v < count; v++) {
      const i = 3 * v;
      if (shown[i] !== unjoined[i] || shown[i + 1] !== unjoined[i + 1] || shown[i + 2] !== unjoined[i + 2]) cavity[v] = 1;
    }
    // A shared face with an inserted or refit vertex belongs to the exact
    // accepted cavity display. Only the free surface beyond it can relax.
    for (let f = 0; f < indices.length; f += 3) {
      const a = indices[f], b = indices[f + 1], c = indices[f + 2];
      if (cavity[a] === 1 || cavity[b] === 1 || cavity[c] === 1) {
        cavity[a] ||= 2; cavity[b] ||= 2; cavity[c] ||= 2;
      }
    }
  }
  let supportSamples = 0;
  for (let v = 0; v < count; v++) {
    if (cavity?.[v]) continue;
    const i = 3 * v, x = raw[i], y = raw[i + 1], z = raw[i + 2];
    const crop = Math.min(Math.min(x + .5, nx - .5 - x) * sx,
      Math.min(y + .5, ny - .5 - y) * sy, Math.min(z + .5, nz - .5 - z) * sz);
    let distance = Math.min(TAPER_MM, crop - .85);
    if (distance <= 0) continue;
    const xmin = Math.max(0, Math.ceil(x - search / sx)), xmax = Math.min(nx - 1, Math.floor(x + search / sx));
    const ymin = Math.max(0, Math.ceil(y - search / sy)), ymax = Math.min(ny - 1, Math.floor(y + search / sy));
    const zmin = Math.max(0, Math.ceil(z - search / sz)), zmax = Math.min(nz - 1, Math.floor(z + search / sz));
    supportSamples += (xmax - xmin + 1) * (ymax - ymin + 1) * (zmax - zmin + 1);
    if (supportSamples > MAX_SUPPORT_SAMPLES) return;
    nearby: for (let a = xmin; a <= xmax; a++) for (let b = ymin; b <= ymax; b++) {
      const transverse = ((a - x) * sx) ** 2 + ((b - y) * sy) ** 2;
      const row = a * yz + b * nz;
      for (let c = zmin; c <= zmax; c++) {
        const flag = fixed[row + c];
        if (!flag) continue;
        distance = Math.min(distance, Math.sqrt(transverse + ((c - z) * sz) ** 2) - (flag === 2 ? editMm : interfaceMm));
        if (distance <= 0) break nearby;
      }
    }
    if (distance > 0) {
      const t = distance / TAPER_MM;
      mobility[v] = t * t * (3 - 2 * t);
    }
  }

  const offsets = new Uint32Array(count + 1);
  for (const v of indices) if (mobility[v]) offsets[v + 1] += 2;
  for (let v = 1; v <= count; v++) offsets[v] += offsets[v - 1];
  if (!offsets[count]) return;
  const cursor = offsets.slice(), neighbors = new Uint32Array(offsets[count]);
  const weights = new Float32Array(neighbors.length), mass = new Float32Array(count);
  for (let f = 0; f < indices.length; f += 3) for (let c = 0; c < 3; c++) {
    const v = indices[f + c];
    if (!mobility[v]) continue;
    for (let k = 1; k <= 2; k++) {
      const j = indices[f + (c + k) % 3], edge = cursor[v]++;
      const length = Math.hypot((shown[3 * j] - shown[3 * v]) * sx,
        (shown[3 * j + 1] - shown[3 * v + 1]) * sy, (shown[3 * j + 2] - shown[3 * v + 2]) * sz);
      const weight = Math.max(1.5, Math.min(8, 1 / length));
      neighbors[edge] = 3 * j;
      weights[edge] = weight;
      mass[v] += weight;
    }
  }

  const active = Array.from(mobility.keys()).filter(v => mass[v] > 0);
  let current = shown.slice(), next = shown.slice();
  for (let pass = 0; pass < 14; pass++) {
    for (const v of active) {
      const i = 3 * v;
      let x = 0, y = 0, z = 0;
      for (let e = offsets[v]; e < offsets[v + 1]; e++) {
        const j = neighbors[e], w = weights[e];
        x += w * current[j]; y += w * current[j + 1]; z += w * current[j + 2];
      }
      const smooth = .60 * mobility[v], restore = .03 * mobility[v], mean = smooth / mass[v];
      const keep = 1 - smooth - restore;
      const dx = mean * x + keep * current[i] + restore * shown[i] - raw[i];
      const dy = mean * y + keep * current[i + 1] + restore * shown[i + 1] - raw[i + 1];
      const dz = mean * z + keep * current[i + 2] + restore * shown[i + 2] - raw[i + 2];
      const scale = Math.min(1, .998 * limit / Math.max(1e-12, Math.hypot(dx * sx, dy * sy, dz * sz)));
      next[i] = raw[i] + dx * scale;
      next[i + 1] = raw[i + 1] + dy * scale;
      next[i + 2] = raw[i + 2] + dz * scale;
    }
    [current, next] = [next, current];
  }
  const changed = active.filter(v => [0, 1, 2].some(axis => current[3 * v + axis] !== shown[3 * v + axis]));
  if (!changed.length) return;
  const smoothNormals = normals.slice();
  if (!guardCavityPatch(current, smoothNormals, indices, shown, [], changed, affine)) return;
  let maximum = sourceDisplacementMm;
  for (const v of changed) {
    const i = 3 * v;
    maximum = Math.max(maximum, Math.hypot((current[i] - raw[i]) * sx,
      (current[i + 1] - raw[i + 1]) * sy, (current[i + 2] - raw[i + 2]) * sz));
  }
  if (maximum > limit) return;
  return {positions:current, smoothNormals, maximum, limit};
}
