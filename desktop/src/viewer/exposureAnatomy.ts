import type {OperativeExposure, OperativeExposureBody, ExposureBodyName} from '../operative-exposure-types';
import type {SurgicalView} from '../surgical-types';
import type {ViewerVolume} from './contracts';
import type {Affine, Point3, Shape3} from './coordinates';
import {transformPoint} from './coordinates.ts';

export const bodyNames: ExposureBodyName[] = ['scalp', 'bone', 'dura'];
export function sourceVoxel(id: number, shape: Shape3): Point3 {
  return [Math.floor(id / (shape[1] * shape[2])), Math.floor(id / shape[2]) % shape[1], id % shape[2]];
}
export function sourceCenter(id: number, shape: Shape3, affine: Affine): Point3 {
  return transformPoint(affine, sourceVoxel(id, shape));
}
export function bodyIsDisplaced(body: OperativeExposureBody, shape: Shape3, affine: Affine) {
  if (body.cells.length !== body.centersMm.length) throw Error('Exposure body has inconsistent source coordinates');
  return body.cells.some((id, i) => {
    const rest = sourceCenter(id, shape, affine),
      p = body.centersMm[i];
    return Math.abs(p[0] - rest[0]) > 0.0001 || Math.abs(p[1] - rest[1]) > 0.0001 || Math.abs(p[2] - rest[2]) > 0.0001;
  });
}
export function displacedBodies(exposure: OperativeExposure | undefined, shape: Shape3, affine: Affine) {
  return exposure ? bodyNames.filter((name) => bodyIsDisplaced(exposure.bodies[name], shape, affine)) : [];
}

/** The live labels continue to include each body's raster occupancy for MPR
 * and numerical accounting. Only the mesh extractor uses the remainder: a
 * separately transported, finite-thickness surface replaces that same body. */
const remainders = new WeakMap<Uint16Array, Map<number, {key: string; mask: Uint8Array}>>();
export function partitionExposureSurfaces(
  volume: ViewerVolume,
  view: SurgicalView,
  owner: Uint16Array,
  overrides: ReadonlyMap<number, number>,
) {
  const exposure = view.operativeExposureV1;
  if (!exposure) return;
  const names = displacedBodies(exposure, view.scene.shape, view.scene.affine);
  const moved = new Map(view.moved),
    bodySites = new Map<number, Set<number>>(),
    candidates = new Set(overrides.keys());
  const removed = new Set(view.removed);
  for (const name of names)
    for (const id of exposure.bodies[name].cells) {
      if (removed.has(id)) continue;
      const region = owner[id],
        site = moved.get(id) ?? id,
        sites = bodySites.get(region) ?? new Set<number>();
      sites.add(site);
      bodySites.set(region, sites);
      candidates.add(site);
    }
  let cache = remainders.get(owner);
  if (!cache) {
    cache = new Map();
    remainders.set(owner, cache);
  }
  volume.compartments = volume.compartments.map((layer, i) => {
    const region = view.scene.regions[i]?.id,
      sites = bodySites.get(region);
    if (!sites?.size) return layer.surfaceMask ? {...layer, surfaceMask: undefined} : layer;
    const different: number[] = [];
    for (const id of candidates)
      if (Boolean(owner[id] === region) !== Boolean(layer.mask[id] && !sites.has(id))) different.push(id);
    different.sort((a, b) => a - b);
    const key = different.join(','),
      old = cache!.get(region);
    if (old?.key === key) return {...layer, surfaceMask: old.mask};
    const mask = layer.mask.slice();
    for (const id of sites) mask[id] = 0;
    cache!.set(region, {key, mask});
    return {...layer, surfaceMask: mask};
  });
}

/** The ordinary 3D body mesh already draws the continuous lips of accepted
 * sheet/kerf edges. Suppress their duplicate raster diagnostic in 3D only.
 * All unrepresented cut, mesh and vascular evidence stays in the effect pass. */
export function effectsOutsideExposure(view: SurgicalView): SurgicalView {
  const exposure = view.operativeExposureV1;
  if (!exposure) return view;
  const key = ([a, b]: [number, number]) => (a < b ? a + ',' + b : b + ',' + a);
  const represented = new Set(bodyNames.flatMap((name) => exposure.bodies[name].seamEdges.map(key)));
  if (!represented.size) return view;
  const keep = (p: [number, number]) => !represented.has(key(p));
  return {
    ...view,
    cutEdges: view.cutEdges?.filter(keep),
    cuts: view.cuts.map((c) => (c.edges ? {...c, edges: c.edges.filter(keep)} : c)),
  };
}
