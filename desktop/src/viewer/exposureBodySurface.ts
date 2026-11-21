import * as THREE from 'three';
import type {OperativeExposureBody} from '../operative-exposure-types';
import type {SurgicalView} from '../surgical-types';
import type {Point3} from './coordinates';
import {transformPoint} from './coordinates.ts';
import {sourceCenter, sourceVoxel} from './exposureAnatomy.ts';
import {prepareSurface} from './prepareSurface.ts';
import {displaySurfaceNormals} from './displaySurfaceNormals.ts';

type Scene = SurgicalView['scene'];
const vec = (v: number[]) => new THREE.Vector3(v[0], v[1], v[2]);
function basis(a: THREE.Vector3, b: THREE.Vector3) {
  a.normalize();
  const normal = new THREE.Vector3().crossVectors(a, b);
  if (normal.lengthSq() < 1e-12) return null;
  normal.normalize();
  return new THREE.Matrix4().makeBasis(a, new THREE.Vector3().crossVectors(normal, a), normal);
}

/** A material point retains its source identity when a folded sheet occupies
 * less than one raster cell. Orient the finite-thickness cell from two
 * neighboring accepted center tangents, including on a one-cell membrane. */
export class ExposureBodyTransport {
  readonly rest: THREE.Vector3[];
  readonly byId: Map<number, number>;
  readonly current: THREE.Vector3[];
  readonly rotation: THREE.Quaternion[];
  private frames: Array<{ids: number[]; rest: THREE.Quaternion} | null>;
  constructor(body: OperativeExposureBody, scene: Scene) {
    this.byId = new Map(body.cells.map((id, i) => [id, i]));
    this.rest = body.cells.map((id) => vec(sourceCenter(id, scene.shape, scene.affine)));
    this.current = this.rest.map((p) => p.clone());
    this.rotation = this.rest.map(() => new THREE.Quaternion());
    const [nx, ny, nz] = scene.shape,
      steps = [ny * nz, nz, 1];
    this.frames = body.cells.map((id, i) => {
      const voxel = sourceVoxel(id, scene.shape),
        pairs: number[][] = [];
      for (let axis = 0; axis < 3; axis++) {
        const low = voxel[axis] > 0 ? this.byId.get(id - steps[axis]) : undefined;
        const high = voxel[axis] + 1 < [nx, ny, nz][axis] ? this.byId.get(id + steps[axis]) : undefined;
        if (low !== undefined || high !== undefined) pairs.push([low ?? i, high ?? i]);
      }
      for (let a = 0; a < pairs.length; a++)
        for (let b = a + 1; b < pairs.length; b++) {
          const [u, v] = pairs[a],
            [p, q] = pairs[b];
          const frame = basis(this.rest[v].clone().sub(this.rest[u]), this.rest[q].clone().sub(this.rest[p]));
          if (frame) return {ids: [u, v, p, q], rest: new THREE.Quaternion().setFromRotationMatrix(frame).invert()};
        }
      return null;
    });
    // At a one-voxel boundary tip use the nearest valid local frame. This
    // carries a rigid flap exactly without inventing a hinge trajectory.
    const valid = this.frames.flatMap((frame, i) => (frame ? [i] : []));
    for (let i = 0; i < this.frames.length; i++)
      if (!this.frames[i] && valid.length) {
        let best = valid[0],
          distance = Infinity;
        for (const j of valid) {
          const d = this.rest[i].distanceToSquared(this.rest[j]);
          if (d < distance) {
            distance = d;
            best = j;
          }
        }
        this.frames[i] = this.frames[best];
      }
  }
  update(centers: Point3[]) {
    if (centers.length !== this.rest.length) throw Error('Missing accepted body centers');
    let changed = false;
    for (let i = 0; i < centers.length; i++) {
      const p = centers[i],
        c = this.current[i];
      if (c.x !== p[0] || c.y !== p[1] || c.z !== p[2]) changed = true;
      c.fromArray(p);
    }
    if (!changed) return false;
    for (let i = 0; i < this.frames.length; i++) {
      const frame = this.frames[i];
      if (!frame) continue;
      const [u, v, p, q] = frame.ids;
      const current = basis(this.current[v].clone().sub(this.current[u]), this.current[q].clone().sub(this.current[p]));
      if (current) this.rotation[i].setFromRotationMatrix(current).multiply(frame.rest);
    }
    return true;
  }
  point(rest: THREE.Vector3, anchor: number, out = new THREE.Vector3()) {
    return out.copy(rest).sub(this.rest[anchor]).applyQuaternion(this.rotation[anchor]).add(this.current[anchor]);
  }
}

export function buildExposureSurface(
  ids: number[],
  owner: Uint16Array,
  transport: ExposureBodyTransport,
  scene: Scene,
  region: number,
  exteriorSkinRegion?: number,
) {
  const voxels = ids.map((id) => sourceVoxel(id, scene.shape));
  const low: Point3 = [...scene.shape],
    high = [0, 0, 0];
  for (const p of voxels)
    for (let k = 0; k < 3; k++) {
      low[k] = Math.min(low[k], Math.max(0, p[k] - 3));
      high[k] = Math.max(high[k], Math.min(scene.shape[k] - 1, p[k] + 3));
    }
  const shape = low.map((v, k) => high[k] - v + 1) as Point3,
    [nx, ny, nz] = shape;
  const mask = new Uint8Array(nx * ny * nz),
    labels = new Uint16Array(mask.length);
  const [, , sourceZ] = scene.shape,
    sourceYZ = scene.shape[1] * sourceZ;
  for (let x = 0; x < nx; x++)
    for (let y = 0; y < ny; y++) {
      const index = (x * ny + y) * nz,
        source = (x + low[0]) * sourceYZ + (y + low[1]) * sourceZ + low[2];
      labels.set(owner.subarray(source, source + nz), index);
    }
  for (const v of voxels) mask[((v[0] - low[0]) * ny + v[1] - low[1]) * nz + v[2] - low[2]] = 1;
  const prepared = prepareSurface(
    mask,
    shape,
    scene.affine,
    scene.regions.find((r) => r.id === region)?.material,
    true,
    {
      owner: labels,
      exteriorSkinRegion,
      protectedRegions: scene.regions
        .filter((r) => r.material === 'vessel' || r.material === 'membrane')
        .map((r) => r.id),
    },
  );
  const raw = prepared.positions;
  const normalMatrix = new THREE.Matrix3().getNormalMatrix(
    new THREE.Matrix4().fromArray(scene.affine.flat()).transpose(),
  );
  function bindSurface(local: Float32Array, sourceNormals?: Float32Array) {
    const restPoints: THREE.Vector3[] = [],
      bind: Array<Array<[number, number]>> = [];
    const attributes = new Float32Array(local.length);
    const restNormals = sourceNormals
      ? new THREE.BufferAttribute(sourceNormals.slice(), 3).applyNormalMatrix(normalMatrix)
      : undefined;
    const normals = restNormals
      ? new THREE.BufferAttribute(new Float32Array(local.length), 3).setUsage(THREE.DynamicDrawUsage)
      : undefined;
    const geometry = new THREE.BufferGeometry();
    for (let i = 0; i < local.length; i += 3) {
      const v = low.map((a, k) => a + local[i + k]) as Point3;
      restPoints.push(vec(transformPoint(scene.affine, v)));
      // Use the original extraction edge to bind the conditioned vertex. Even
      // an endpoint at the outer grid border has one occupied source neighbor.
      const cell = low.map((a, k) => a + raw[i + k]);
      const donors: Array<[number, number]> = [];
      let mass = 0;
      for (let x = Math.floor(cell[0]); x <= Math.ceil(cell[0]); x++)
        for (let y = Math.floor(cell[1]); y <= Math.ceil(cell[1]); y++)
          for (let z = Math.floor(cell[2]); z <= Math.ceil(cell[2]); z++) {
            if (x < 0 || y < 0 || z < 0 || x >= scene.shape[0] || y >= scene.shape[1] || z >= scene.shape[2]) continue;
            const index = transport.byId.get(x * sourceYZ + y * sourceZ + z);
            if (index === undefined) continue;
            const weight = Math.max(
              0.001,
              (1 - Math.abs(x - v[0])) * (1 - Math.abs(y - v[1])) * (1 - Math.abs(z - v[2])),
            );
            donors.push([index, weight]);
            mass += weight;
          }
      if (!donors.length) throw Error('Unbound body surface vertex');
      bind.push(donors.map(([index, w]) => [index, w / mass]));
    }
    geometry.setAttribute('position', new THREE.BufferAttribute(attributes, 3).setUsage(THREE.DynamicDrawUsage));
    geometry.setIndex(new THREE.BufferAttribute(prepared.indices, 1));
    if (normals) geometry.setAttribute('normal', normals);
    geometry.setAttribute(
      'exposureRest',
      new THREE.Float32BufferAttribute(
        restPoints.flatMap((p) => p.toArray()),
        3,
      ),
    );
    const normal = new THREE.Vector3(),
      scratch = new THREE.Vector3(),
      sum = new THREE.Vector3();
    function update() {
      for (let v = 0; v < restPoints.length; v++) {
        sum.set(0, 0, 0);
        for (const [anchor, w] of bind[v]) sum.addScaledVector(transport.point(restPoints[v], anchor, scratch), w);
        sum.toArray(attributes, 3 * v);
        if (normals && restNormals) {
          normal.fromBufferAttribute(restNormals, v);
          sum.set(0, 0, 0);
          for (const [anchor, w] of bind[v])
            sum.addScaledVector(scratch.copy(normal).applyQuaternion(transport.rotation[anchor]), w);
          sum.normalize();
          normals.setXYZ(v, sum.x, sum.y, sum.z);
        }
      }
      geometry.attributes.position.needsUpdate = true;
      if (normals) normals.needsUpdate = true;
      else geometry.computeVertexNormals();
      geometry.computeBoundingSphere();
      geometry.computeBoundingBox();
    }
    update();
    return {geometry, update};
  }
  const smooth = bindSurface(
    prepared.smoothPositions ?? raw,
    displaySurfaceNormals(prepared.smoothNormals ?? prepared.normals, prepared.indices, scene.affine),
  );
  let source: ReturnType<typeof bindSurface> | undefined;
  return {
    geometry: smooth.geometry,
    sourceGeometry() {
      source ??= bindSurface(raw);
      return source.geometry;
    },
    update() {
      smooth.update();
      source?.update();
    },
    disposeOther(displayed: THREE.BufferGeometry) {
      if (smooth.geometry !== displayed) smooth.geometry.dispose();
      if (source && source.geometry !== displayed) source.geometry.dispose();
    },
  };
}
