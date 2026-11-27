/** The inspection cutaway's cap drawn from the source surfaces (display only).
 *
 * The plane cuts each smooth source surface (skin, outer and inner table,
 * pia, white matter, tumour) into segments. The cap image runs its columns
 * from outside the head inward (its `down` axis is the exterior normal turned
 * into the plane), and every surface crossing met along a column enters or
 * leaves that surface's tissue by the side its normal faces. Each pixel takes
 * the deepest tissue it lies in and is filled with the case scan intensity,
 * lightly tinted by that tissue; the surface segments themselves are the
 * layer boundaries, drawn as lines. Tissue the engine has opened (a moved
 * flap's footprint) or removed (the cavity) is left open. Nothing here is an
 * engine input. */
import type {Point3} from './coordinates';

export const CAP_REGIONS = ['open', 'scalp', 'bone', 'intracranial', 'cortex', 'white', 'tumourRim', 'tumourCore'] as const;
export type CapRegion = (typeof CAP_REGIONS)[number];
/** The surface that bounds each region from outside, in nesting order. */
export const CAP_SURFACES = ['skin', 'skullOuter', 'skullInner', 'pial', 'white', 'tumourRim', 'tumourCore'] as const;
export type CapSurfaceName = (typeof CAP_SURFACES)[number];

export interface CapSurface {name: CapSurfaceName; positions: Float32Array; normals: Float32Array; triangles: Uint32Array}
export interface CapGrid {origin: Point3; across: Point3; down: Point3; width: number; height: number; stepMm: number}
/** One plane crossing of a surface: end points in cap millimetres (u across, v down) and the in-plane normal. */
export interface CapSegment {u0: number; v0: number; u1: number; v1: number; nu: number; nv: number}

const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** The cap axes: `down` is `outward` turned into the plane and reversed (so columns run from outside
 * inward); `across` completes the frame. Falls back to a fixed reference when the plane faces `outward`. */
export function capAxes(normal: Point3, outward: Point3) {
  const n = unit(normal);
  let down = [-outward[0], -outward[1], -outward[2]];
  const along = dot(down, n);
  down = down.map((v, a) => v - along * n[a]);
  if (Math.hypot(down[0], down[1], down[2]) < 0.2) {
    const reference = Math.abs(n[2]) > 0.9 ? [1, 0, 0] : [0, 0, 1];
    down = cross(n, cross(reference, n));
  }
  const d = unit(down);
  const across = unit(cross(d, n));
  return {across: across as Point3, down: d as Point3};
}

/** Segments of one surface where the plane (point `origin`, unit `normal`) cuts it, in the cap frame. */
export function planeSegments(surface: CapSurface, origin: Point3, normal: Point3, across: Point3, down: Point3): CapSegment[] {
  const {positions: p, normals: m, triangles: t} = surface, out: CapSegment[] = [];
  const count = p.length / 3, side = new Float32Array(count);
  for (let i = 0; i < count; i++)
    side[i] = (p[3 * i] - origin[0]) * normal[0] + (p[3 * i + 1] - origin[1]) * normal[1] + (p[3 * i + 2] - origin[2]) * normal[2];
  const hit: number[] = [];
  for (let f = 0; f < t.length; f += 3) {
    const a = t[f], b = t[f + 1], c = t[f + 2], sa = side[a], sb = side[b], sc = side[c];
    if ((sa > 0 && sb > 0 && sc > 0) || (sa < 0 && sb < 0 && sc < 0)) continue;
    hit.length = 0;
    for (const [i, j, si, sj] of [[a, b, sa, sb], [b, c, sb, sc], [c, a, sc, sa]]) {
      if ((si < 0) === (sj < 0) || si === sj) continue;
      const s = si / (si - sj);
      for (let k = 0; k < 3; k++) hit.push(p[3 * i + k] + s * (p[3 * j + k] - p[3 * i + k]));
    }
    if (hit.length !== 6) continue;
    const g = [0, 1, 2].map((k) => m[3 * a + k] + m[3 * b + k] + m[3 * c + k]);
    const rel = (o: number) => [hit[o] - origin[0], hit[o + 1] - origin[1], hit[o + 2] - origin[2]];
    const q0 = rel(0), q1 = rel(3);
    out.push({u0: dot(q0, across), v0: dot(q0, down), u1: dot(q1, across), v1: dot(q1, down), nu: dot(g, across), nv: dot(g, down)});
  }
  return out;
}

/** Region of every cap pixel from the crossings met down its column. `segments` holds each named
 * surface's plane segments; a surface not given simply never encloses anything. */
export function capRegions(segments: ReadonlyMap<CapSurfaceName, CapSegment[]>, grid: Pick<CapGrid, 'width' | 'height' | 'stepMm'>): Uint8Array {
  const {width, height, stepMm} = grid, out = new Uint8Array(width * height);
  const columns: Array<Array<{v: number; s: number; enter: boolean}>> = Array.from({length: width}, () => []);
  CAP_SURFACES.forEach((name, s) => {
    for (const seg of segments.get(name) ?? []) {
      const lo = Math.min(seg.u0, seg.u1), hi = Math.max(seg.u0, seg.u1);
      if (hi - lo < 1e-9) continue;
      const first = Math.max(0, Math.ceil(lo / stepMm - 0.5)), last = Math.min(width - 1, Math.ceil(hi / stepMm - 0.5) - 1);
      // Going down the column (v increasing) enters the tissue where its outward normal faces up.
      const enter = seg.nv < 0;
      for (let x = first; x <= last; x++) {
        const u = (x + 0.5) * stepMm, f = (u - seg.u0) / (seg.u1 - seg.u0);
        columns[x].push({v: seg.v0 + f * (seg.v1 - seg.v0), s, enter});
      }
    }
  });
  const inside = new Uint8Array(CAP_SURFACES.length);
  for (let x = 0; x < width; x++) {
    const crossings = columns[x].sort((a, b) => a.v - b.v);
    inside.fill(0);
    // A surface whose first crossing in this column leaves its tissue starts inside it.
    const seen = new Uint8Array(CAP_SURFACES.length);
    for (const c of crossings) if (!seen[c.s]) { seen[c.s] = 1; inside[c.s] = c.enter ? 0 : 1; }
    let k = 0;
    for (let y = 0; y < height; y++) {
      const v = (y + 0.5) * stepMm;
      while (k < crossings.length && crossings[k].v <= v) { inside[crossings[k].s] = crossings[k].enter ? 1 : 0; k++; }
      // The deepest enclosing tissue, by nesting order.
      let region = 0;
      for (let s = CAP_SURFACES.length - 1; s >= 0; s--) if (inside[s]) { region = s + 1; break; }
      out[y * width + x] = region;
    }
  }
  return out;
}

/** Illustrative tint of each region on the cut (the scan intensity carries the detail). */
export const CAP_TINT: Record<CapRegion, [number, number, number]> = {
  open: [0, 0, 0], scalp: [1.0, 0.8, 0.72], bone: [1.0, 0.95, 0.82], intracranial: [0.78, 0.86, 0.97],
  cortex: [0.98, 0.82, 0.82], white: [1.0, 0.95, 0.88], tumourRim: [0.92, 0.72, 0.84], tumourCore: [0.84, 0.62, 0.76],
};
/** Boundary line colour of each surface. */
export const CAP_LINE: Record<CapSurfaceName | 'dura', string> = {
  skin: '#6b3a2e', skullOuter: '#8c7650', skullInner: '#8c7650', dura: '#7a6a8a', pial: '#7d4a52', white: '#a08d78',
  tumourRim: '#6e3a5a', tumourCore: '#5a2b4a',
};

/** RGBA cap pixels: scan intensity (windowed) times the region tint; open, opened or removed pixels transparent. */
export function paintCap(regions: Uint8Array, signal: Float32Array, closed: (i: number) => boolean) {
  const values: number[] = [];
  for (let i = 0; i < regions.length; i++) if (regions[i] && closed(i) && Number.isFinite(signal[i])) values.push(signal[i]);
  values.sort((a, b) => a - b);
  const low = values[Math.floor(values.length * 0.02)] ?? 0, high = values[Math.floor(values.length * 0.98)] ?? 1;
  const out = new Uint8ClampedArray(regions.length * 4), span = high > low ? high - low : 1;
  for (let i = 0; i < regions.length; i++) {
    const r = regions[i];
    if (!r || !closed(i) || !Number.isFinite(signal[i])) continue;
    const t = Math.max(0, Math.min(1, (signal[i] - low) / span)), g = 30 + 215 * t, tint = CAP_TINT[CAP_REGIONS[r]];
    out[4 * i] = g * tint[0]; out[4 * i + 1] = g * tint[1]; out[4 * i + 2] = g * tint[2]; out[4 * i + 3] = 255;
  }
  return out;
}

function unit(v: readonly number[]) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}
function cross(a: readonly number[], b: readonly number[]) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
