/** Geometry helpers for drawing source-resolution surfaces under engine state.
 *
 * All positions here are continuous source-voxel coordinates of the engine
 * working grid (cell centres at integers), the frame the operative renderer
 * already draws in (`placeInSourceFrame`). Nothing here decides tissue state:
 * the engine's bodies, cuts and removed cells are inputs. */
import type {Affine, Point3, Shape3} from './coordinates';
import {inverseAffine} from './coordinates.ts';

export interface MeshPart {
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
  /** Source vertex of each output vertex, or -1 for a vertex made by clipping. */
  source: Int32Array;
  /** For clipping vertices: the two source vertices and the blend toward the second. */
  between: Float32Array;
  /** Edges of the clipped boundary, as output vertex pairs in triangle winding order. */
  cutEdges: Uint32Array;
}

/** RAS points to continuous voxel coordinates of `affine`. */
export function rasToVoxel(points: Float32Array, affine: Affine): Float32Array {
  const m = inverseAffine(affine), out = new Float32Array(points.length);
  for (let i = 0; i < points.length; i += 3) {
    const x = points[i], y = points[i + 1], z = points[i + 2];
    for (let a = 0; a < 3; a++) out[i + a] = m[a][0] * x + m[a][1] * y + m[a][2] * z + m[a][3];
  }
  return out;
}

/** RAS normals to the voxel-space normals that the source-frame object matrix maps back to them (Lᵀn, unit). */
export function normalsToVoxel(normals: Float32Array, affine: Affine): Float32Array {
  const out = new Float32Array(normals.length);
  for (let i = 0; i < normals.length; i += 3) {
    let len = 0;
    for (let a = 0; a < 3; a++) {
      const v = affine[0][a] * normals[i] + affine[1][a] * normals[i + 1] + affine[2][a] * normals[i + 2];
      out[i + a] = v;
      len += v * v;
    }
    len = Math.sqrt(len) || 1;
    for (let a = 0; a < 3; a++) out[i + a] /= len;
  }
  return out;
}

/** A RAS direction in voxel units (L⁻¹d), unit length. */
export function directionToVoxel(direction: readonly number[], affine: Affine): Point3 {
  const m = inverseAffine(affine);
  const v = [0, 1, 2].map((a) => m[a][0] * direction[0] + m[a][1] * direction[1] + m[a][2] * direction[2]);
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}

class PartBuilder {
  positions: number[] = [];
  normals: number[] = [];
  indices: number[] = [];
  source: number[] = [];
  between: number[] = [];
  cut: number[] = [];
  private map = new Map<number, number>();
  private edges = new Map<string, number>();
  private p: Float32Array;
  private n: Float32Array;
  constructor(p: Float32Array, n: Float32Array) {
    this.p = p;
    this.n = n;
  }
  vertex(v: number) {
    let out = this.map.get(v);
    if (out === undefined) {
      out = this.source.length;
      this.map.set(v, out);
      this.positions.push(this.p[3 * v], this.p[3 * v + 1], this.p[3 * v + 2]);
      this.normals.push(this.n[3 * v], this.n[3 * v + 1], this.n[3 * v + 2]);
      this.source.push(v);
      this.between.push(v, v, 0);
    }
    return out;
  }
  crossing(a: number, b: number, t: number) {
    const key = a < b ? `${a},${b}` : `${b},${a}`;
    let out = this.edges.get(key);
    if (out === undefined) {
      out = this.source.length;
      this.edges.set(key, out);
      const nx = [0, 1, 2].map((k) => this.n[3 * a + k] * (1 - t) + this.n[3 * b + k] * t), len = Math.hypot(nx[0], nx[1], nx[2]) || 1;
      this.positions.push(...[0, 1, 2].map((k) => this.p[3 * a + k] * (1 - t) + this.p[3 * b + k] * t));
      this.normals.push(nx[0] / len, nx[1] / len, nx[2] / len);
      this.source.push(-1);
      this.between.push(a, b, t);
    }
    return out;
  }
  finish(): MeshPart {
    return {positions: Float32Array.from(this.positions), normals: Float32Array.from(this.normals), indices: Uint32Array.from(this.indices),
      source: Int32Array.from(this.source), between: Float32Array.from(this.between), cutEdges: Uint32Array.from(this.cut)};
  }
}

/** Keep the part of one triangle where value ≥ level, recording the clipped boundary edge. */
function clipTriangle(out: PartBuilder, tri: [number, number, number], value: (v: number) => number, level: number) {
  const s = tri.map((v) => value(v) - level), inside = s.map((x) => x >= 0);
  const count = inside.filter(Boolean).length;
  if (count === 0) return;
  if (count === 3) {
    out.indices.push(out.vertex(tri[0]), out.vertex(tri[1]), out.vertex(tri[2]));
    return;
  }
  // Rotate so that the odd vertex is first, keeping the winding.
  const odd = count === 1 ? inside.indexOf(true) : inside.indexOf(false);
  const a = tri[odd], b = tri[(odd + 1) % 3], c = tri[(odd + 2) % 3];
  const sa = s[odd], sb = s[(odd + 1) % 3], sc = s[(odd + 2) % 3];
  const ab = out.crossing(a, b, sa / (sa - sb)), ac = out.crossing(a, c, sa / (sa - sc));
  if (count === 1) {
    out.indices.push(out.vertex(a), ab, ac);
    out.cut.push(ab, ac);
  } else {
    const vb = out.vertex(b), vc = out.vertex(c);
    out.indices.push(ab, vb, vc, ab, vc, ac);
    out.cut.push(ac, ab);
  }
}

/** Split a sheet along a cut: the flap keeps value ≥ +gap, the fixed side value ≤ −gap.
 * Triangles touching a hinge vertex are not cut; they go whole to the side of
 * their largest value, so a hinged flap stays joined to its base. */
export function splitSheet(positions: Float32Array, normals: Float32Array, triangles: Uint32Array, value: ArrayLike<number>,
  gapMm: number, hinge?: ArrayLike<number>, keep?: (triangle: number) => boolean) {
  const flap = new PartBuilder(positions, normals), fixed = new PartBuilder(positions, normals);
  const up = (v: number) => value[v], down = (v: number) => -value[v];
  for (let t = 0; t < triangles.length / 3; t++) {
    if (keep && !keep(t)) continue;
    const tri: [number, number, number] = [triangles[3 * t], triangles[3 * t + 1], triangles[3 * t + 2]];
    const high = Math.max(value[tri[0]], value[tri[1]], value[tri[2]]), low = Math.min(value[tri[0]], value[tri[1]], value[tri[2]]);
    if (low >= gapMm) { flap.indices.push(flap.vertex(tri[0]), flap.vertex(tri[1]), flap.vertex(tri[2])); continue; }
    if (high <= -gapMm) { fixed.indices.push(fixed.vertex(tri[0]), fixed.vertex(tri[1]), fixed.vertex(tri[2])); continue; }
    if (hinge && (hinge[tri[0]] || hinge[tri[1]] || hinge[tri[2]])) {
      const target = high > 0 ? flap : fixed;
      target.indices.push(target.vertex(tri[0]), target.vertex(tri[1]), target.vertex(tri[2]));
      continue;
    }
    clipTriangle(flap, tri, up, gapMm);
    clipTriangle(fixed, tri, down, gapMm);
  }
  return {flap: flap.finish(), fixed: fixed.finish()};
}

/** Interpolate a per-source-vertex attribute onto a clipped part. */
export function partAttribute(part: MeshPart, values: ArrayLike<number>): Float32Array {
  const out = new Float32Array(part.source.length);
  for (let i = 0; i < out.length; i++) {
    const a = part.between[3 * i], b = part.between[3 * i + 1], t = part.between[3 * i + 2];
    out[i] = values[a] * (1 - t) + values[b] * t;
  }
  return out;
}

/** Closed polyline helpers in the plane across `outward`. */
export function planeBasis(outward: readonly number[]) {
  const n = normalize(outward);
  const reference = Math.abs(n[2]) > 0.9 ? [1, 0, 0] : [0, 0, 1];
  const e1 = normalize(cross(reference, n)), e2 = cross(n, e1);
  return {n, e1, e2};
}
const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: readonly number[], b: readonly number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
function normalize(v: readonly number[]) {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}

/** In-plane nearest segment of a closed loop (voxel points), with its depth along the outward axis. */
export class PlanarLoop {
  readonly u: Float64Array;
  readonly v: Float64Array;
  readonly w: Float64Array;
  readonly basis: ReturnType<typeof planeBasis>;
  constructor(points: Float32Array, outward: readonly number[]) {
    this.basis = planeBasis(outward);
    const n = points.length / 3;
    this.u = new Float64Array(n);
    this.v = new Float64Array(n);
    this.w = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const p = [points[3 * i], points[3 * i + 1], points[3 * i + 2]];
      this.u[i] = dot(p, this.basis.e1);
      this.v[i] = dot(p, this.basis.e2);
      this.w[i] = dot(p, this.basis.n);
    }
  }
  /** Signed in-plane distance (positive inside the loop), the outward coordinate of the nearest loop point and its index. */
  query(p: ArrayLike<number>) {
    const u = dot(p, this.basis.e1), v = dot(p, this.basis.e2), n = this.u.length;
    let best = Infinity, depth = 0, index = 0, inside = false;
    for (let i = 0, j = n - 1; i < n; j = i++) {
      const ax = this.u[j], ay = this.v[j], bx = this.u[i], by = this.v[i];
      if ((ay > v) !== (by > v) && u < ((bx - ax) * (v - ay)) / (by - ay) + ax) inside = !inside;
      const dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy;
      const t = len2 > 0 ? Math.max(0, Math.min(1, ((u - ax) * dx + (v - ay) * dy) / len2)) : 0;
      const ex = ax + t * dx - u, ey = ay + t * dy - v, d = ex * ex + ey * ey;
      if (d < best) {
        best = d;
        depth = this.w[j] * (1 - t) + this.w[i] * t;
        index = t < 0.5 ? j : i;
      }
    }
    best = Math.sqrt(best);
    return {distance: inside ? best : -best, outward: depth, index};
  }
  get outward() { return this.basis.n; }
}

/** A smooth wall closing a clipped sheet: from each cut-boundary vertex straight
 * down the outward axis to the depth of the inner outline below it. */
export function kerfWall(part: MeshPart, inner: PlanarLoop, outwardVoxel: readonly number[], extraDepth = 0) {
  const top = new Map<number, number>(), positions: number[] = [], indices: number[] = [], depth: number[] = [];
  const n = normalize(outwardVoxel);
  const add = (v: number) => {
    let index = top.get(v);
    if (index !== undefined) return index;
    // Two wall vertices (top, bottom) per boundary vertex.
    index = positions.length / 6;
    top.set(v, index);
    const p = [part.positions[3 * v], part.positions[3 * v + 1], part.positions[3 * v + 2]];
    const below = Math.max(0.3, dot(p, inner.outward) - inner.query(p).outward + extraDepth);
    positions.push(...p, p[0] - n[0] * below, p[1] - n[1] * below, p[2] - n[2] * below);
    depth.push(below);
    return index;
  };
  for (let e = 0; e < part.cutEdges.length; e += 2) {
    const a = add(part.cutEdges[e]), b = add(part.cutEdges[e + 1]);
    // Top a→b follows the clipped surface's winding; the wall faces out of the kept region.
    indices.push(2 * a, 2 * a + 1, 2 * b, 2 * b, 2 * a + 1, 2 * b + 1);
  }
  return {positions: Float32Array.from(positions), indices: Uint32Array.from(indices), boundary: [...top.keys()]};
}

/** Engine body cells near each vertex: trilinear tent weights over the eight
 * surrounding cell centres that belong to the body, else the nearest body cell
 * within `reach` cells. Vertices with no body cell nearby stay unbound. */
export interface BodyBinding {offsets: Uint32Array; anchors: Uint32Array; weights: Float32Array; unbound: number}
export function bindToBody(voxel: Float32Array, bound: (vertex: number) => boolean, byId: ReadonlyMap<number, number>, shape: Shape3, reach = 4): BodyBinding {
  const [nx, ny, nz] = shape, count = voxel.length / 3;
  const offsets = new Uint32Array(count + 1), anchors: number[] = [], weights: number[] = [];
  let unbound = 0;
  const cell = (x: number, y: number, z: number) =>
    x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz ? undefined : byId.get((x * ny + y) * nz + z);
  for (let v = 0; v < count; v++) {
    offsets[v] = anchors.length;
    if (!bound(v)) continue;
    const p = [voxel[3 * v], voxel[3 * v + 1], voxel[3 * v + 2]];
    const base = p.map(Math.floor);
    let mass = 0;
    const start = anchors.length;
    for (let dx = 0; dx <= 1; dx++) for (let dy = 0; dy <= 1; dy++) for (let dz = 0; dz <= 1; dz++) {
      const x = base[0] + dx, y = base[1] + dy, z = base[2] + dz, anchor = cell(x, y, z);
      if (anchor === undefined) continue;
      const w = (1 - Math.abs(x - p[0])) * (1 - Math.abs(y - p[1])) * (1 - Math.abs(z - p[2]));
      if (w <= 1e-6) continue;
      anchors.push(anchor);
      weights.push(w);
      mass += w;
    }
    if (mass > 0) {
      for (let i = start; i < anchors.length; i++) weights[i] /= mass;
      continue;
    }
    let best = Infinity, bestAnchor = -1;
    const r = [0, 1, 2].map((a) => Math.round(p[a]));
    for (let dx = -reach; dx <= reach; dx++) for (let dy = -reach; dy <= reach; dy++) for (let dz = -reach; dz <= reach; dz++) {
      const x = r[0] + dx, y = r[1] + dy, z = r[2] + dz, anchor = cell(x, y, z);
      if (anchor === undefined) continue;
      const d = (x - p[0]) ** 2 + (y - p[1]) ** 2 + (z - p[2]) ** 2;
      if (d < best) { best = d; bestAnchor = anchor; }
    }
    if (bestAnchor >= 0) { anchors.push(bestAnchor); weights.push(1); }
    else unbound++;
  }
  offsets[count] = anchors.length;
  return {offsets, anchors: Uint32Array.from(anchors), weights: Float32Array.from(weights), unbound};
}

/** Per-cell rigid motion of an engine body (the same transform the voxel flap uses). */
export interface CellMotion {
  /** Rest RAS centre, current RAS centre and rotation (x, y, z, w) of each body cell. */
  rest: ArrayLike<{x: number; y: number; z: number}>;
  current: ArrayLike<{x: number; y: number; z: number}>;
  rotation: ArrayLike<{x: number; y: number; z: number; w: number}>;
}
function rotate(q: {x: number; y: number; z: number; w: number}, v: number[]) {
  // v' = v + 2w(q×v) + 2q×(q×v)
  const qx = q.x, qy = q.y, qz = q.z, w = q.w;
  const tx = 2 * (qy * v[2] - qz * v[1]), ty = 2 * (qz * v[0] - qx * v[2]), tz = 2 * (qx * v[1] - qy * v[0]);
  return [v[0] + w * tx + qy * tz - qz * ty, v[1] + w * ty + qz * tx - qx * tz, v[2] + w * tz + qx * ty - qy * tx];
}

/** Move bound vertices with their body cells. `restRas` are rest RAS positions;
 * outputs are written in voxel coordinates (and voxel-space normals) of `affine`.
 * Unbound vertices keep their rest position. */
export function transportVertices(binding: BodyBinding, motion: CellMotion, restRas: Float32Array, restNormalsRas: Float32Array | null,
  affine: Affine, outVoxel: Float32Array, outNormals: Float32Array | null) {
  const m = inverseAffine(affine);
  const count = restRas.length / 3;
  let moved = 0;
  for (let v = 0; v < count; v++) {
    const start = binding.offsets[v], end = binding.offsets[v + 1];
    let x = restRas[3 * v], y = restRas[3 * v + 1], z = restRas[3 * v + 2];
    let nx = restNormalsRas ? restNormalsRas[3 * v] : 0, ny = restNormalsRas ? restNormalsRas[3 * v + 1] : 0, nz = restNormalsRas ? restNormalsRas[3 * v + 2] : 0;
    if (end > start) {
      let sx = 0, sy = 0, sz = 0, gx = 0, gy = 0, gz = 0;
      for (let i = start; i < end; i++) {
        const a = binding.anchors[i], w = binding.weights[i], r = motion.rest[a], c = motion.current[a], q = motion.rotation[a];
        const d = rotate(q, [x - r.x, y - r.y, z - r.z]);
        sx += w * (d[0] + c.x); sy += w * (d[1] + c.y); sz += w * (d[2] + c.z);
        if (restNormalsRas) { const n = rotate(q, [nx, ny, nz]); gx += w * n[0]; gy += w * n[1]; gz += w * n[2]; }
      }
      if (Math.abs(sx - x) + Math.abs(sy - y) + Math.abs(sz - z) > 1e-6) moved++;
      x = sx; y = sy; z = sz;
      if (restNormalsRas) { const len = Math.hypot(gx, gy, gz) || 1; nx = gx / len; ny = gy / len; nz = gz / len; }
    }
    for (let a = 0; a < 3; a++) outVoxel[3 * v + a] = m[a][0] * x + m[a][1] * y + m[a][2] * z + m[a][3];
    if (outNormals && restNormalsRas) {
      let len = 0;
      const out = [0, 1, 2].map((a) => { const value = affine[0][a] * nx + affine[1][a] * ny + affine[2][a] * nz; len += value * value; return value; });
      len = Math.sqrt(len) || 1;
      for (let a = 0; a < 3; a++) outNormals[3 * v + a] = out[a] / len;
    }
  }
  return moved;
}

/** Voxel points to RAS. */
export function voxelToRas(points: Float32Array, affine: Affine): Float32Array {
  const out = new Float32Array(points.length);
  for (let i = 0; i < points.length; i += 3)
    for (let a = 0; a < 3; a++) out[i + a] = affine[a][0] * points[i] + affine[a][1] * points[i + 1] + affine[a][2] * points[i + 2] + affine[a][3];
  return out;
}

/** Close the small holes of an open sheet: every boundary loop shorter than
 * `maxPerimeter` (in the units of `positions`) gets a fan around its centroid.
 * Returns the extended vertex arrays, the added triangles and, for each added
 * vertex, the loop it closes (to carry per-vertex attributes). */
export function fillSmallHoles(positions: Float32Array, normals: Float32Array, triangles: Uint32Array, maxPerimeter: number) {
  const next = new Map<number, number>(), used = new Map<string, number>();
  for (let t = 0; t < triangles.length; t += 3)
    for (let k = 0; k < 3; k++) {
      const a = triangles[t + k], b = triangles[t + ((k + 1) % 3)], key = a < b ? `${a},${b}` : `${b},${a}`;
      used.set(key, (used.get(key) ?? 0) + 1);
    }
  for (let t = 0; t < triangles.length; t += 3)
    for (let k = 0; k < 3; k++) {
      const a = triangles[t + k], b = triangles[t + ((k + 1) % 3)], key = a < b ? `${a},${b}` : `${b},${a}`;
      // A boundary edge runs a→b in its triangle; the hole is traversed b→a.
      if (used.get(key) === 1) next.set(b, a);
    }
  const loops: number[][] = [], seen = new Set<number>();
  for (const start of next.keys()) {
    if (seen.has(start)) continue;
    const loop: number[] = [];
    let v: number | undefined = start, perimeter = 0, closed = false;
    while (v !== undefined && !seen.has(v) && loop.length < 4096) {
      seen.add(v);
      loop.push(v);
      const w = next.get(v);
      if (w === undefined) break;
      perimeter += Math.hypot(positions[3 * w] - positions[3 * v], positions[3 * w + 1] - positions[3 * v + 1], positions[3 * w + 2] - positions[3 * v + 2]);
      if (w === start) { closed = true; break; }
      v = w;
    }
    if (closed && loop.length >= 3 && perimeter <= maxPerimeter) loops.push(loop);
  }
  const extraPositions: number[] = [], extraNormals: number[] = [], extraTriangles: number[] = [];
  let index = positions.length / 3;
  for (const loop of loops) {
    const c = [0, 0, 0], n = [0, 0, 0];
    for (const v of loop) for (let a = 0; a < 3; a++) { c[a] += positions[3 * v + a] / loop.length; n[a] += normals[3 * v + a]; }
    const len = Math.hypot(n[0], n[1], n[2]) || 1;
    extraPositions.push(...c);
    extraNormals.push(n[0] / len, n[1] / len, n[2] / len);
    for (let i = 0; i < loop.length; i++) extraTriangles.push(loop[i], loop[(i + 1) % loop.length], index);
    index++;
  }
  const outPositions = new Float32Array(positions.length + extraPositions.length);
  outPositions.set(positions);
  outPositions.set(extraPositions, positions.length);
  const outNormals = new Float32Array(normals.length + extraNormals.length);
  outNormals.set(normals);
  outNormals.set(extraNormals, normals.length);
  const outTriangles = new Uint32Array(triangles.length + extraTriangles.length);
  outTriangles.set(triangles);
  outTriangles.set(extraTriangles, triangles.length);
  return {positions: outPositions, normals: outNormals, triangles: outTriangles, loops};
}
