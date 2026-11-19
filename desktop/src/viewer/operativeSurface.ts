import * as THREE from 'three';
import { operativeSectionContour, carrySectionContour } from './operativeSectionContour.ts';
import { fitOperativeSheet } from './operativeSheetFit.ts';
import type { OperativeField } from '../operative-field';
import {
  operativeSurfaceField,
  operativeFilterRadius,
  copyConditionedRegion,
  type ConditionedField,
} from './operativeSurfaceField.ts';

type Grid = Pick<OperativeField, 'shape' | 'spacingMm' | 'origin'>;
type XYZ = [number, number, number];
export type VoxelBounds = { low: XYZ; high: XYZ };
export const emptyVoxelBounds = (): VoxelBounds => ({
  low: [Infinity, Infinity, Infinity],
  high: [-1, -1, -1],
});
export function extendVoxelBounds(b: VoxelBounds, x: number, y: number, z: number) {
  b.low[0] = Math.min(b.low[0], x);
  b.high[0] = Math.max(b.high[0], x);
  b.low[1] = Math.min(b.low[1], y);
  b.high[1] = Math.max(b.high[1], y);
  b.low[2] = Math.min(b.low[2], z);
  b.high[2] = Math.max(b.high[2], z);
}

// One dual vertex per boundary loop. Face checkerboards isolate each
// foreground corner, preserving distinct sheets touching at a corner or edge.
const edgeIds = Array.from({ length: 3 }, () => new Uint8Array(8));
const edges: Array<[number, number]> = [];
for (let axis = 0; axis < 3; axis++)
  for (let v = 0; v < 8; v++)
    if (!(v & (1 << axis))) {
      const w = v | (1 << axis);
      edgeIds[axis][v] = edgeIds[axis][w] = edges.length;
      edges.push([v, w]);
    }
const cellsTable = Array.from({ length: 256 }, (_, pattern) => {
  const inside = (v: number) => !!(pattern & (1 << v));
  const parent = edges.map((_, i) => i);
  const root = (i: number) => {
    while (parent[i] !== i) i = parent[i];
    return i;
  };
  const link = (a: number, b: number) => {
    parent[root(a)] = root(b);
  };
  for (let axis = 0; axis < 3; axis++)
    for (let side = 0; side < 2; side++) {
      const base = side << axis,
        u = 1 << (axis + 1) % 3,
        v = 1 << (axis + 2) % 3;
      const corners = [base, base | u, base | u | v, base | v];
      const loop = corners.map((c, i) => edgeIds[i % 2 ? (axis + 2) % 3 : (axis + 1) % 3][c]);
      const cut = loop.filter((_, i) => inside(corners[i]) !== inside(corners[(i + 1) % 4]));
      if (cut.length === 2) link(cut[0], cut[1]);
      if (cut.length === 4)
        for (let i = 0; i < 4; i++) if (inside(corners[i])) link(loop[(i + 3) % 4], loop[i]);
    }
  const active: number[] = [],
    slots = new Uint8Array(12),
    groups = new Map<number, number>(),
    counts: number[] = [],
    offsets: XYZ[] = [];
  for (let e = 0; e < edges.length; e++) {
    const [v, w] = edges[e];
    if (inside(v) === inside(w)) continue;
    const key = root(e);
    let slot = groups.get(key);
    if (slot === undefined) {
      slot = offsets.length;
      groups.set(key, slot);
      offsets.push([0, 0, 0]);
      counts.push(0);
    }
    active.push(e);
    slots[e] = slot;
    counts[slot]++;
    for (let a = 0; a < 3; a++) offsets[slot][a] += (((v >> a) & 1) + ((w >> a) & 1)) * 0.5;
  }
  offsets.forEach((offset, i) => {
    for (let a = 0; a < 3; a++) offset[a] /= counts[i];
  });
  return { slots, offsets, counts, active };
});

type Extracted = { positions: Float32Array; normals: Float32Array; indices: Uint32Array; keys: Uint32Array };
const empty = (): Extracted => ({
  positions: new Float32Array(),
  normals: new Float32Array(),
  indices: new Uint32Array(),
  keys: new Uint32Array(),
});
function pad(b: VoxelBounds, n: number): VoxelBounds {
  return { low: b.low.map((v) => v - n) as XYZ, high: b.high.map((v) => v + n) as XYZ };
}
function intersect(a: VoxelBounds, b: VoxelBounds): VoxelBounds {
  return {
    low: a.low.map((v, i) => Math.max(v, b.low[i])) as XYZ,
    high: a.high.map((v, i) => Math.min(v, b.high[i])) as XYZ,
  };
}
function unpack(key: number, grid: Grid): XYZ {
  const width = grid.shape[2] + 1,
    height = grid.shape[1] + 1,
    id = Math.floor(key / 4);
  return [Math.floor(id / (height * width)) - 1, (Math.floor(id / width) % height) - 1, (id % width) - 1];
}
function inBounds(x: number, y: number, z: number, b: VoxelBounds) {
  return (
    x >= b.low[0] && x <= b.high[0] && y >= b.low[1] && y <= b.high[1] && z >= b.low[2] && z <= b.high[2]
  );
}
function extract(
  grid: Grid,
  mask: Uint8Array,
  b: VoxelBounds,
  firstY: number,
  thin: boolean,
  iterations: number,
  sheet: boolean,
  conditioned?: ConditionedField
): Extracted {
  if (b.low.some((v, i) => v > b.high[i])) return empty();
  const [nx, ny, nz] = grid.shape;
  const field = conditioned ?? operativeSurfaceField(grid, mask, b, thin),
    yz = field.yz,
    sz = field.sz;
  const section = firstY > 0 ? operativeSectionContour(grid, mask, firstY) : undefined;
  const corners = [0, yz, sz, yz + sz, 1, yz + 1, sz + 1, yz + sz + 1];
  const shift = new Float64Array(12);
  const [x0, y0, z0] = b.low,
    [x1, y1, z1] = b.high,
    width = z1 - z0 + 1,
    slab = width * (y1 - y0 + 1);
  const bases = new Uint32Array(2 * slab),
    patterns = new Uint8Array(2 * slab);
  const coords: number[] = [],
    keys: number[] = [],
    flags: number[] = [],
    faces: number[] = [];
  const stride = [slab, width, 1];
  const value = (x: number, y: number, z: number) =>
    x >= 0 && x < nx && y >= firstY && y < ny && z >= 0 && z < nz
      ? section && y === firstY
        ? mask[(x * ny + y) * nz + z]
        : field.mask[field.base + x * yz + y * sz + z]
      : 0;
  for (let x = x0; x <= x1; x++) {
    const plane = ((x - x0) % 2) * slab;
    stride[0] = plane ? slab : -slab;
    for (let y = y0; y <= y1; y++) {
      let low =
        value(x, y, z0) |
        (value(x + 1, y, z0) << 1) |
        (value(x, y + 1, z0) << 2) |
        (value(x + 1, y + 1, z0) << 3);
      let cell = plane + (y - y0) * width;
      for (let z = z0; z <= z1; z++, cell++) {
        const high =
          value(x, y, z + 1) |
          (value(x + 1, y, z + 1) << 1) |
          (value(x, y + 1, z + 1) << 2) |
          (value(x + 1, y + 1, z + 1) << 3);
        const pattern = low | (high << 4);
        low = high;
        patterns[cell] = pattern;
        if (!pattern || pattern === 255) continue;
        const config = cellsTable[pattern],
          base = keys.length;
        bases[cell] = base;
        const key = 4 * (((x + 1) * (ny + 1) + y + 1) * (nz + 1) + z + 1);
        const pinned =
          (x === -1 || x === nx - 1 ? 1 : 0) |
          (y === firstY - 1 || y === ny - 1 ? 2 : 0) |
          (z === -1 || z === nz - 1 ? 4 : 0);
        const local = field.base + x * yz + y * sz + z;
        shift.fill(0);
        for (const edge of config.active) {
          const [v, w] = edges[edge],
            axis = Math.floor(edge / 4);
          if (
            pinned & (1 << axis) ||
            field.unresolved[local + corners[v]] ||
            field.unresolved[local + corners[w]]
          )
            continue;
          const lo = field.smooth[local + corners[v]],
            hi = field.smooth[local + corners[w]];
          const t = lo === hi ? 0.5 : (127.5 - lo) / (hi - lo),
            margin = thin ? 0.2 : 0.06;
          shift[config.slots[edge] * 3 + axis] += Math.max(margin, Math.min(1 - margin, t)) - 0.5;
        }
        for (let k = 0; k < config.offsets.length; k++) {
          const off = config.offsets[k],
            n = config.counts[k];
          keys.push(key + k);
          const cut = y === firstY - 1 ? section : undefined;
          flags.push((cut ? 7 : pinned) | (config.offsets.length > 1 ? 8 : 0));
          coords.push(
            cut ? x + 0.5 : x + (pinned & 1 ? 0.5 : off[0] + shift[k * 3] / n),
            y + (pinned & 2 ? 0.5 : off[1] + shift[k * 3 + 1] / n),
            cut ? z + 0.5 : z + (pinned & 4 ? 0.5 : off[2] + shift[k * 3 + 2] / n)
          );
        }
        for (let axis = 0; axis < 3; axis++) {
          if ((pattern & 1) === ((pattern >> (1 << axis)) & 1)) continue;
          const u = (axis + 1) % 3,
            v = (axis + 2) % 3;
          if ((axis !== 0 && x === x0) || (axis !== 1 && y === y0) || (axis !== 2 && z === z0)) continue;
          const cu = cell - stride[u],
            cv = cell - stride[v],
            cuv = cu - stride[v];
          const ids = edgeIds[axis],
            a = base + config.slots[ids[0]],
            b = bases[cu] + cellsTable[patterns[cu]].slots[ids[1 << u]],
            c = bases[cuv] + cellsTable[patterns[cuv]].slots[ids[(1 << u) | (1 << v)]],
            d = bases[cv] + cellsTable[patterns[cv]].slots[ids[1 << v]];
          if (pattern & 1) faces.push(a, b, c, a, c, d);
          else faces.push(a, d, c, a, c, b);
        }
      }
    }
  }
  if (!keys.length) return empty();
  const indices = Uint32Array.from(faces),
    source = Float32Array.from(coords),
    count = keys.length;
  // Quad edges give a four-neighbour stencil on a regular sheet, without
  // triangulation-diagonal bias. Opposing sheets have separate vertex rings.
  const offsets = new Uint32Array(count + 1);
  for (let i = 0; i < indices.length; i += 6) {
    offsets[indices[i] + 1]++;
    offsets[indices[i + 1] + 1]++;
    offsets[indices[i + 2] + 1]++;
    offsets[indices[i + 5] + 1]++;
  }
  for (let i = 1; i <= count; i++) offsets[i] += offsets[i - 1];
  const ends = offsets.slice(),
    neighbors = new Uint32Array(offsets[count]);
  for (let i = 0; i < indices.length; i += 6) {
    const a = indices[i],
      b = indices[i + 1],
      c = indices[i + 2],
      d = indices[i + 5];
    neighbors[ends[a]++] = b;
    neighbors[ends[b]++] = c;
    neighbors[ends[c]++] = d;
    neighbors[ends[d]++] = a;
  }
  let pos = source.slice(),
    scratch = new Float32Array(pos.length);
  const cap = thin ? 0.16 : 0.7;
  // Constrained elastic surface nets. Each position stays in its source
  // dual cell, with tighter displacement on thin or ambiguous sheets.
  // Section and field-boundary vertices stay on their physical plane.
  const lowClamp = new Float32Array(pos.length),
    highClamp = new Float32Array(pos.length);
  for (let i = 0; i < count; i++)
    for (let a = 0; a < 3; a++) {
      const p = i * 3 + a,
        base = source[p],
        cell = Math.floor(base),
        limit = flags[i] & 8 ? 0.13 : cap,
        pin = flags[i] & (1 << a);
      const margin = Math.max(0.04, Math.min(base - cell, cell + 1 - base) * (thin ? 0.7 : 0.4));
      lowClamp[p] = pin ? base : Math.max(cell + margin, base - limit);
      highClamp[p] = pin ? base : Math.min(cell + 1 - margin, base + limit);
    }
  for (let it = 0; it < iterations; it++) {
    for (let i = 0; i < count; i++) {
      const from = offsets[i],
        end = offsets[i + 1],
        p = i * 3;
      let sx = 0,
        sy = 0,
        sz = 0;
      for (let e = from; e < end; e++) {
        const j = neighbors[e] * 3;
        sx += pos[j];
        sy += pos[j + 1];
        sz += pos[j + 2];
      }
      // Alternating low-pass / expansion preserves the calibre of a thin
      // vessel and the separation between opposite membrane faces.
      const weight = thin ? (it % 2 ? -0.48 : 0.4) : 0.85;
      const alpha = weight / (end - from || 1),
        keep = end > from ? 1 - weight : 1;
      scratch[p] = Math.min(highClamp[p], Math.max(lowClamp[p], keep * pos[p] + alpha * sx));
      scratch[p + 1] = Math.min(highClamp[p + 1], Math.max(lowClamp[p + 1], keep * pos[p + 1] + alpha * sy));
      scratch[p + 2] = Math.min(highClamp[p + 2], Math.max(lowClamp[p + 2], keep * pos[p + 2] + alpha * sz));
    }
    [pos, scratch] = [scratch, pos];
  }
  if (sheet) fitOperativeSheet(grid, field, b, pos);
  if (section) carrySectionContour(section, firstY, grid.spacingMm, pos, source, sheet);
  let normals = new Float32Array(pos.length);
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i] * 3,
      b = indices[i + 1] * 3,
      c = indices[i + 2] * 3;
    const ux = pos[b] - pos[a],
      uy = pos[b + 1] - pos[a + 1],
      uz = pos[b + 2] - pos[a + 2];
    const vx = pos[c] - pos[a],
      vy = pos[c + 1] - pos[a + 1],
      vz = pos[c + 2] - pos[a + 2];
    const xx = uy * vz - uz * vy,
      yy = uz * vx - ux * vz,
      zz = ux * vy - uy * vx;
    for (let k = 0; k < 3; k++) {
      const j = indices[i + k] * 3;
      normals[j] += xx;
      normals[j + 1] += yy;
      normals[j + 2] += zz;
    }
  }
  for (let j = 0; j < normals.length; j += 3) {
    const r = 1 / (Math.sqrt(normals[j] ** 2 + normals[j + 1] ** 2 + normals[j + 2] ** 2) || 1);
    normals[j] *= r;
    normals[j + 1] *= r;
    normals[j + 2] *= r;
  }
  // Filter normals along the connected surface to temper highlights on
  // quantized slopes. Facing weights avoid cancellation on thin tubes.
  for (let it = 0; it < 4; it++) {
    for (let i = 0; i < count; i++) {
      const p = 3 * i,
        nx = normals[p],
        ny = normals[p + 1],
        nz = normals[p + 2];
      let sx = nx,
        sy = ny,
        sz = nz;
      for (let e = offsets[i]; e < offsets[i + 1]; e++) {
        const j = neighbors[e] * 3,
          w = Math.max(0, nx * normals[j] + ny * normals[j + 1] + nz * normals[j + 2]);
        sx += w * normals[j];
        sy += w * normals[j + 1];
        sz += w * normals[j + 2];
      }
      const r = 1 / (Math.sqrt(sx * sx + sy * sy + sz * sz) || 1);
      scratch[p] = sx * r;
      scratch[p + 1] = sy * r;
      scratch[p + 2] = sz * r;
    }
    [normals, scratch] = [scratch, normals];
  }
  for (let i = 0; i < pos.length; i += 3) {
    if (firstY > 0 && pos[i + 1] === firstY - 0.5) {
      normals[i] = 0;
      normals[i + 1] = -1;
      normals[i + 2] = 0;
    }
    pos[i] = grid.origin[0] + pos[i] * grid.spacingMm;
    pos[i + 1] = grid.origin[1] + pos[i + 1] * grid.spacingMm;
    pos[i + 2] = grid.origin[2] + pos[i + 2] * grid.spacingMm;
  }
  return { positions: pos, normals, indices, keys: Uint32Array.from(keys) };
}

type CachedExterior = {
  row: number;
  thin: boolean;
  field: ConditionedField;
  part: Extracted;
  seam: Map<number, number>;
};
const editable = new WeakMap<
  Uint8Array,
  { region: VoxelBounds; dirty?: VoxelBounds; cached?: CachedExterior }
>();
/** The model enumerates every cell mutable by a local operation once. A halo
 * around this box contains all possible changes in both mesh and normals. */
export function registerOperativeEdits(mask: Uint8Array, region: VoxelBounds) {
  if (region.high[0] >= region.low[0]) editable.set(mask, { region });
}
export function markOperativeEdits(mask: Uint8Array, dirty: VoxelBounds) {
  const change = editable.get(mask);
  if (!change?.cached || dirty.high[0] < dirty.low[0]) return;
  if (!change.dirty) change.dirty = dirty;
  else {
    extendVoxelBounds(change.dirty, ...dirty.low);
    extendVoxelBounds(change.dirty, ...dirty.high);
  }
}
function toGeometry(data: Extracted) {
  const g = new THREE.BufferGeometry();
  if (!data.indices.length) return g;
  g.setAttribute('position', new THREE.BufferAttribute(data.positions, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(data.normals, 3));
  g.setIndex(new THREE.BufferAttribute(data.indices, 1));
  return g;
}
// Each quad belongs to its emitting cell. Shared boundary vertices use the
// same dual-cell and loop IDs, including at an ambiguous mask configuration.
function partition(data: Extracted, grid: Grid, region: VoxelBounds, inside: boolean): Extracted {
  const remap = new Uint32Array(data.keys.length),
    ix: number[] = [],
    ids: number[] = [],
    positions: number[] = [],
    normals: number[] = [];
  const selected = new Uint8Array(data.keys.length);
  for (let i = 0; i < data.keys.length; i++) {
    const [x, y, z] = unpack(data.keys[i], grid);
    selected[i] = Number(inBounds(x, y, z, region) === inside);
  }
  for (let i = 0; i < data.indices.length; i += 6) {
    if (!selected[data.indices[i]]) continue;
    for (let k = 0; k < 6; k++) {
      const id = data.indices[i + k];
      let out = remap[id];
      if (!out) {
        out = ids.length + 1;
        remap[id] = out;
        ids.push(data.keys[id]);
        const p = id * 3;
        positions.push(data.positions[p], data.positions[p + 1], data.positions[p + 2]);
        normals.push(data.normals[p], data.normals[p + 1], data.normals[p + 2]);
      }
      ix.push(out - 1);
    }
  }
  return {
    positions: Float32Array.from(positions),
    normals: Float32Array.from(normals),
    indices: Uint32Array.from(ix),
    keys: Uint32Array.from(ids),
  };
}
function combine(fixed: CachedExterior, part: Extracted) {
  const ext = fixed.part,
    offset = ext.keys.length,
    positions = new Float32Array((offset + part.keys.length) * 3),
    normals = new Float32Array(positions.length);
  positions.set(ext.positions);
  positions.set(part.positions, 3 * offset);
  normals.set(ext.normals);
  normals.set(part.normals, 3 * offset);
  const remap = Uint32Array.from(part.keys, (key, i) => fixed.seam.get(key) ?? i + offset);
  const indices = new Uint32Array(ext.indices.length + part.indices.length);
  indices.set(ext.indices);
  for (let i = 0; i < part.indices.length; i++) indices[ext.indices.length + i] = remap[part.indices[i]];
  return toGeometry({ positions, normals, indices, keys: part.keys });
}
/** Closed, subvoxel, physically bounded display surface of one source label.
 * Editing a mask outside the registered change region requires rebuilding
 * the model. Switching section planes evicts the previous exterior cache. */
export function operativeSurface(
  grid: Grid,
  mask: Uint8Array,
  bounds: VoxelBounds,
  firstY = 0,
  kind = 'context'
) {
  const thin = kind === 'dura' || kind === 'duraFlap' || kind === 'vessel',
    sheet = ['bone', 'boneFlap', 'dura', 'duraFlap'].includes(kind);
  const iterations = Math.min(12, Math.max(5, Math.round(3 / grid.spacingMm)));
  const full: VoxelBounds = {
    low: [bounds.low[0] - 1, Math.max(bounds.low[1], firstY) - 1, bounds.low[2] - 1],
    high: [...bounds.high],
  };
  if (full.low.some((v, a) => v > full.high[a])) return new THREE.BufferGeometry();
  const change = editable.get(mask),
    halo = iterations + 7;
  if (!change) return toGeometry(extract(grid, mask, full, firstY, thin, iterations, sheet));
  // Eight independent checkerboard batches limit the topology predicate's
  // dependency radius. Update the scalar patch separately so the larger
  // mesh-normal halo never forces re-filtering the whole target on scrub.
  const filterHalo = 2 * operativeFilterRadius(grid.spacingMm, thin) + 10;
  const region = pad(change.region, halo + filterHalo + (sheet ? 8 : 0));
  if (sheet) {
    region.low[2] = full.low[2];
    region.high[2] = full.high[2];
  }

  if (!change.cached || change.cached.row !== firstY || change.cached.thin !== thin) {
    const field = operativeSurfaceField(grid, mask, full, thin);
    const data = extract(grid, mask, full, firstY, thin, iterations, sheet, field);
    const part = partition(data, grid, region, false),
      seam = new Map<number, number>();
    for (let i = 0; i < part.keys.length; i++) {
      const [x, y, z] = unpack(part.keys[i], grid);
      if (
        x >= region.low[0] - 1 &&
        x <= region.high[0] &&
        y >= region.low[1] - 1 &&
        y <= region.high[1] &&
        z >= region.low[2] - 1 &&
        z <= region.high[2]
      )
        seam.set(part.keys[i], i);
    }
    change.cached = { row: firstY, thin, field, part, seam };
    change.dirty = undefined;
    return toGeometry(data);
  }
  if (change.dirty) {
    const patch = intersect(
      { low: full.low, high: full.high.map((v) => v + 1) as XYZ },
      pad(change.dirty, filterHalo)
    );
    if (!patch.low.some((v, i) => v > patch.high[i])) {
      const updated = operativeSurfaceField(grid, mask, pad(patch, 9), thin);
      copyConditionedRegion(change.cached.field, updated, patch);
    }
    change.dirty = undefined;
  }
  const local = extract(
    grid,
    mask,
    intersect(full, pad(region, halo + (sheet ? 8 : 0))),
    firstY,
    thin,
    iterations,
    sheet,
    change.cached.field
  );
  if (
    !change.cached.part.indices.length &&
    full.low.every((v, a) => v >= region.low[a] && full.high[a] <= region.high[a])
  )
    return toGeometry(local);
  return combine(change.cached, partition(local, grid, region, true));
}
