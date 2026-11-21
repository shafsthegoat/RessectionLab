import type {Affine, Point3, Shape3} from './coordinates';
import {cIndex} from './coordinates.ts';
import {sourceVoxel} from './exposureAnatomy.ts';
import {maskSurfaceRegion} from './surface.ts';
import {prepareSurfaceSoup} from './prepareSurface.ts';

/** When a generated skin is just one native cell thick, marching its skin and
 * galea separately can connect galea to outside air across a tetrahedral
 * diagonal. Recover only that small part of the external contour supported
 * by an intact skin/air voxel face. The flap and editable sites stay free. */
export function exteriorSkinSurface(
  owner: Uint16Array,
  shape: Shape3,
  affine: Affine,
  skin: number,
  galea: number,
  mobile: ReadonlySet<number>,
) {
  const stride = [shape[1] * shape[2], shape[2], 1];
  // Distance to a supporting face is evaluated in source-cell units. The
  // opt-in generated model is isotropic; leave other grids on their ordinary path.
  const cols = stride.map((_, a) => affine.slice(0, 3).map((row) => row[a]));
  const h2 = cols[0].reduce((s, v) => s + v * v, 0);
  if (
    !(h2 > 0) ||
    cols.some((u, a) =>
      cols.some((v, b) => Math.abs(u.reduce((s, n, i) => s + n * v[i], 0) - (a === b ? h2 : 0)) > 1e-7 * h2),
    )
  )
    return;
  const outside = (p: Point3) => p.some((n, a) => n < 0 || n >= shape[a]);
  const edgeStep = (p: Point3, axis: number, sign: number) => {
    const q: Point3 = [...p];
    q[axis] += sign;
    return q;
  };
  // An unobstructed axis ray certifies exterior air without presuming that
  // internal air pockets are accessible. A concave, ambiguous site is skipped.
  const openAir = new Map<number, boolean>();
  const open = (id: number) => {
    let result = openAir.get(id);
    if (result !== undefined) return result;
    const p = sourceVoxel(id, shape);
    result = stride.some((step, axis) =>
      [-1, 1].some((sign) => {
        let j = id;
        for (let i = p[axis] + sign; i >= 0 && i < shape[axis]; i += sign) {
          j += sign * step;
          if (owner[j]) return false;
        }
        return true;
      }),
    );
    openAir.set(id, result);
    return result;
  };
  const corners: Point3[] = Array.from({length: 8}, (_, i) => [i & 1, (i >> 1) & 1, (i >> 2) & 1]);
  const cornerOffset = corners.map((p) => cIndex(shape, p));
  const candidates = new Set<number>();
  for (let id = 0; id < owner.length; id++) {
    if (owner[id] !== galea || mobile.has(id)) continue;
    const p = sourceVoxel(id, shape);
    if (p.some((v, a) => v < 1 || v > shape[a] - 2)) continue;
    const diagonalAir = cornerOffset
      .slice(1)
      .some((offset) => [-1, 1].some((sign) => !owner[id + sign * offset] && open(id + sign * offset)));
    if (diagonalAir) for (const offset of cornerOffset) candidates.add(id - offset);
  }
  if (!candidates.size) return;
  const occupied = Uint8Array.from(owner, (id) => Number(id !== 0));
  const reference = Uint16Array.from(owner, (id) => (id === galea ? skin : id));
  type Face = {center: Point3; axis: number};
  const sourceSupport = new Set<number>(),
    soup: number[] = [],
    supportFaces: Face[][] = [];
  for (const start of candidates) {
    const cube = cornerOffset.map((offset) => start + offset);
    if (cube.some((id) => mobile.has(id) || (owner[id] !== 0 && owner[id] !== skin && owner[id] !== galea))) continue;
    if (cube.some((id) => !owner[id] && !open(id))) continue;
    const exposed = cube.some(
      (id) => owner[id] === galea && stride.some((step) => !owner[id - step] || !owner[id + step]),
    );
    if (exposed) continue;
    const origin = sourceVoxel(start, shape);
    const {positions} = maskSurfaceRegion(occupied, shape, origin, origin);
    const diagonalCrossing = (i: number) => {
      const p = [positions[i], positions[i + 1], positions[i + 2]];
      const a = owner[cIndex(shape, p.map(Math.floor) as Point3)],
        b = owner[cIndex(shape, p.map(Math.ceil) as Point3)];
      return (a === galea && !b) || (b === galea && !a);
    };
    const faces: Face[] = [];
    for (const id of cube) {
      if (owner[id] !== skin) continue;
      const p = sourceVoxel(id, shape);
      for (let axis = 0; axis < 3; axis++)
        for (const sign of [-1, 1]) {
          const q = edgeStep(p, axis, sign);
          if (outside(q) || !cube.includes(cIndex(shape, q)) || owner[cIndex(shape, q)]) continue;
          faces.push({center: edgeStep(p, axis, 0.5 * sign), axis});
        }
    }
    for (let i = 0; i < positions.length; i += 9) {
      if (!diagonalCrossing(i) && !diagonalCrossing(i + 3) && !diagonalCrossing(i + 6)) continue;
      if (faces.length) {
        supportFaces.push(faces);
        soup.push(...positions.subarray(i, i + 9));
        for (const id of cube) sourceSupport.add(id);
      }
    }
  }
  if (!soup.length) return;
  const surface = prepareSurfaceSoup(Float32Array.from(soup), occupied, shape, affine, 'soft_tissue', true, {
    owner: reference,
    protectedRegions: [],
    exteriorSkinRegion: skin,
  });
  const p = surface.smoothPositions!,
    keep = [];
  let maxFaceDistanceMm = 0;
  for (let i = 0; i < surface.indices.length; i += 3) {
    // All three conditioned vertices must lie within half a native cell of
    // the SAME true skin/air face. Convexity then bounds the entire triangle.
    const distance = Math.min(
      ...supportFaces[i / 3].map(({center, axis}) =>
        Math.max(
          ...[0, 1, 2].map((corner) => {
            const v = 3 * surface.indices[i + corner];
            return Math.hypot(...center.map((n, a) => Math.max(0, Math.abs(p[v + a] - n) - (axis === a ? 0 : 0.5))));
          }),
        ),
      ),
    );
    if (distance <= 0.500001) {
      maxFaceDistanceMm = Math.max(maxFaceDistanceMm, distance * Math.sqrt(h2));
      keep.push(...surface.indices.subarray(i, i + 3));
    }
  }
  if (!keep.length) return;
  surface.indices = Uint32Array.from(keep);
  return {surface, sourceSupport, maxFaceDistanceMm};
}
