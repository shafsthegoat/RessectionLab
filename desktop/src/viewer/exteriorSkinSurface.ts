import type {Affine, Point3, Shape3} from './coordinates';
import {cIndex} from './coordinates.ts';
import {sourceVoxel} from './exposureAnatomy.ts';

export const EXTERIOR_AIR = 1,
  ORIGINAL_SKIN = 2,
  INTACT_FACE = 4,
  // Unowned in the undeformed partition: exterior air or enclosed free space
  // such as subarachnoid CSF. Later removal never sets it.
  FREE_SPACE = 8,
  // Free space or a declared layer outside the brain (scalp to dura); set by
  // the operative scene from its region kinds.
  OUTSIDE_BRAIN = 16;

/** Source-certified external skin faces around tetrahedral diagonal shortcuts.
 * Points lie exactly on native rectangular skin/air faces. Quarter faces meet
 * on the same grid as the local visibility test on the other tissue meshes. */
export function exteriorSkinSurface(
  owner: Uint16Array,
  shape: Shape3,
  affine: Affine,
  skin: number,
  protectedIds: ReadonlySet<number>,
) {
  const [nx, ny, nz] = shape,
    strides = [ny * nz, nz, 1];
  const flags = new Uint8Array(owner.length),
    queue = new Int32Array(owner.length);
  let front = 0,
    end = 0;
  const addAir = (id: number) => {
    if (!owner[id] && !(flags[id] & EXTERIOR_AIR)) {
      flags[id] |= EXTERIOR_AIR;
      queue[end++] = id;
    }
  };
  for (let x = 0; x < nx; x++)
    for (let y = 0; y < ny; y++)
      for (let z = 0; z < nz; z++) {
        const id = (x * ny + y) * nz + z;
        if (owner[id] === skin) flags[id] = ORIGINAL_SKIN;
        else if (!owner[id]) flags[id] |= FREE_SPACE;
        if (owner[id] === skin) continue;
        if (x === 0 || x === nx - 1 || y === 0 || y === ny - 1 || z === 0 || z === nz - 1) addAir(id);
      }
  while (front < end) {
    const id = queue[front++],
      [x, y, z] = sourceVoxel(id, shape);
    if (x > 0) addAir(id - strides[0]);
    if (x < nx - 1) addAir(id + strides[0]);
    if (y > 0) addAir(id - nz);
    if (y < ny - 1) addAir(id + nz);
    if (z > 0) addAir(id - 1);
    if (z < nz - 1) addAir(id + 1);
  }
  const corners: Point3[] = Array.from({length: 8}, (_, i) => [i & 1, (i >> 1) & 1, i >> 2]);
  const offsets = corners.map((p) => cIndex(shape, p)),
    affected = new Set<number>();
  for (let id = 0; id < owner.length; id++) {
    if (!owner[id] || owner[id] === skin) continue;
    const p = sourceVoxel(id, shape);
    if (p.some((n, a) => n < 2 || n > shape[a] - 3)) continue;
    if (offsets.some((k) => (flags[id + k] | flags[id - k]) & EXTERIOR_AIR)) {
      // Include the continuous one-cell collar on the same external boundary.
      for (let x = -2; x <= 1; x++)
        for (let y = -2; y <= 1; y++) for (let z = -2; z <= 1; z++) affected.add(id + cIndex(shape, [x, y, z]));
    }
  }
  const soup: number[] = [],
    normalValues: number[] = [],
    indices: number[] = [];
  const support = new Set<number>(),
    cubes = new Map<number, [number, number]>(),
    normalCache = new Map<number, number[]>();
  const normalAt = (v: Point3) => {
    const key = (Math.round(v[0] * 2) * ny * 2 + Math.round(v[1] * 2)) * nz * 2 + Math.round(v[2] * 2);
    const cached = normalCache.get(key);
    if (cached) return cached;
    const n = [0, 0, 0];
    for (let x = Math.max(0, Math.floor(v[0]) - 2); x <= Math.min(nx - 1, Math.ceil(v[0]) + 2); x++)
      for (let y = Math.max(0, Math.floor(v[1]) - 2); y <= Math.min(ny - 1, Math.ceil(v[1]) + 2); y++)
        for (let z = Math.max(0, Math.floor(v[2]) - 2); z <= Math.min(nz - 1, Math.ceil(v[2]) + 2); z++) {
          if (!(flags[(x * ny + y) * nz + z] & EXTERIOR_AIR)) continue;
          const d = [x - v[0], y - v[1], z - v[2]];
          const ras = affine.slice(0, 3).map((r) => r[0] * d[0] + r[1] * d[1] + r[2] * d[2]);
          const w = Math.exp(-0.5 * ras.reduce((s, q) => s + q * q, 0));
          for (let a = 0; a < 3; a++) n[a] += w * ras[a];
        }
    // Affine transpose converts a physical normal to Three's source-frame covector.
    const local = [0, 1, 2].map((a) => n.reduce((s, q, k) => s + q * affine[k][a], 0)),
      l = Math.hypot(...local) || 1;
    for (let a = 0; a < 3; a++) local[a] /= l;
    normalCache.set(key, local);
    return local;
  };
  for (const start of affected) {
    const ids = offsets.map((k) => start + k);
    if (!ids.some((id) => owner[id] === skin) || !ids.some((id) => flags[id] & EXTERIOR_AIR)) continue;
    if (ids.some((id) => protectedIds.has(owner[id]))) continue;
    // A real face-exposed interior layer is a genuine opening, never a
    // tetrahedral shortcut. Both sides stay on their ordinary native paths.
    if (
      ids.some(
        (id) => owner[id] && owner[id] !== skin && strides.some((k) => (flags[id - k] | flags[id + k]) & EXTERIOR_AIR),
      )
    )
      continue;
    const origin = sourceVoxel(start, shape),
      first = indices.length;
    for (let i = 0; i < 8; i++)
      if (owner[ids[i]] === skin)
        for (let axis = 0; axis < 3; axis++) {
          const j = i ^ (1 << axis);
          if (!(flags[ids[j]] & EXTERIOR_AIR)) continue;
          const b = (axis + 1) % 3,
            c = (axis + 2) % 3,
            n = corners[j][axis] - corners[i][axis];
          const p = origin.map((q, k) => q + corners[i][k]) as Point3;
          p[axis] += 0.5 * n;
          const vertex = soup.length / 3;
          for (const [u, v] of [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
          ]) {
            const point: Point3 = [...p];
            point[b] += (0.5 - corners[i][b]) * u;
            point[c] += (0.5 - corners[i][c]) * v;
            soup.push(...point);
            normalValues.push(...normalAt(point));
          }
          const positive = (0.5 - corners[i][b]) * (0.5 - corners[i][c]) * n > 0;
          indices.push(...(positive ? [0, 1, 2, 0, 2, 3] : [0, 2, 1, 0, 3, 2]).map((q) => q + vertex));
        }
    if (indices.length > first) {
      cubes.set(start, [first, indices.length]);
      flags[start] |= INTACT_FACE;
      for (const id of ids) support.add(id);
    }
  }
  if (!indices.length) return;
  const sourceIndices = Uint32Array.from(indices);
  let hidden = new Set<number>();
  return {
    positions: Float32Array.from(soup),
    normals: Float32Array.from(normalValues),
    indices: sourceIndices,
    flags,
    sourceSupport: support,
    cubes,
    update(changed: Iterable<number>) {
      const next = new Set<number>();
      for (const id of changed) if (support.has(id)) for (const k of offsets) if (cubes.has(id - k)) next.add(id - k);
      if (next.size === hidden.size && [...next].every((o) => hidden.has(o))) return;
      for (const o of hidden) flags[o] |= INTACT_FACE;
      for (const o of next) flags[o] &= ~INTACT_FACE;
      hidden = next;
      if (!hidden.size) return sourceIndices;
      const visible: number[] = [];
      for (const [o, [a, b]] of cubes) if (!hidden.has(o)) for (let j = a; j < b; j++) visible.push(sourceIndices[j]);
      return Uint32Array.from(visible);
    },
  };
}
