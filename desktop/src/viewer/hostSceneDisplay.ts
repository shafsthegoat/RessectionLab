import * as THREE from 'three';
import type {Affine, Point3, Shape3} from './coordinates';
import {inverseAffine} from './coordinates.ts';

/** Display helpers for an operative rehearsal on a host head: a real head
 * cropped to a working grid around the craniotomy. Nothing here changes the
 * engine grid, its cells or picking. */

export const CROP_SECTION_COLOR = '#8d8a86';

/** Faces lying on the working-grid crop are a section of the head, not anatomy:
 * draw them in one neutral section colour instead of each layer's skin, bone
 * or brain finish. Mesh positions are source voxel indices; the crop face of a
 * layer lies half a cell outside the first or last cell. */
export function installCropSection(material: THREE.Material, shape: Shape3) {
  if (!(material instanceof THREE.MeshStandardMaterial) || material.userData.cropSection) return false;
  const state = {shape: new THREE.Vector3(...shape), color: new THREE.Color(CROP_SECTION_COLOR)};
  material.userData.cropSection = state;
  const previous = material.onBeforeCompile,
    key = material.customProgramCacheKey();
  material.onBeforeCompile = (shader, renderer) => {
    previous.call(material, shader, renderer);
    shader.uniforms.uCropShape = {value: state.shape};
    shader.uniforms.uCropColor = {value: state.color};
    shader.vertexShader =
      'varying vec3 vCropVoxel;\n' +
      shader.vertexShader.replace('#include <begin_vertex>', '#include <begin_vertex>\nvCropVoxel=position;');
    shader.fragmentShader =
      'varying vec3 vCropVoxel;uniform vec3 uCropShape;uniform vec3 uCropColor;\n' +
      shader.fragmentShader.replace(
        '#include <color_fragment>',
        `#include <color_fragment>
        {
          vec3 inside=min(vCropVoxel+.5,uCropShape-.5-vCropVoxel);
          if(min(min(inside.x,inside.y),inside.z)<.08)diffuseColor.rgb=uCropColor;
        }`,
      );
  };
  material.customProgramCacheKey = () => key + '/crop-section';
  material.needsUpdate = true;
  return true;
}

/** A parked (not inserted) instrument stays in the scene for collision and
 * replay but is not drawn over the head: hidden from the microscope field and
 * from the external views (a faint ghost handpiece in the overview read as a
 * real instrument). Inserted tools are unchanged. `_surgeon` is kept for callers. */
export function presentParkedTool(group: THREE.Object3D, parked: boolean, _surgeon?: boolean) {
  group.visible = !parked;
}

/** The planned bone flap outline carried out to the skin along the exterior
 * normal: the line a surgeon marks before the incision. Ray-marches the
 * undeformed owner grid; returns RAS points of a closed loop, or nothing. */
export function craniotomyOutline(boneCells: readonly number[], owner: Uint16Array, shape: Shape3, affine: Affine,
  outward: Point3, bins = 96): Point3[] | undefined {
  if (boneCells.length < 8) return;
  const [nx, ny, nz] = shape, yz = ny * nz, inv = inverseAffine(affine);
  const n = new THREE.Vector3(...outward).normalize();
  const reference = Math.abs(n.z) > 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 0, 1);
  const e1 = new THREE.Vector3().crossVectors(reference, n).normalize(), e2 = new THREE.Vector3().crossVectors(n, e1);
  const ras = (id: number) => {
    const x = Math.floor(id / yz), y = Math.floor(id / nz) % ny, z = id % nz;
    return new THREE.Vector3(...[0, 1, 2].map((a) => affine[a][0] * x + affine[a][1] * y + affine[a][2] * z + affine[a][3]));
  };
  const centre = new THREE.Vector3();
  const points = boneCells.map(ras);
  for (const p of points) centre.add(p);
  centre.divideScalar(points.length);
  const radius = new Float64Array(bins).fill(NaN);
  for (const p of points) {
    const d = p.clone().sub(centre), u = d.dot(e1), v = d.dot(e2);
    const bin = Math.floor(((Math.atan2(v, u) / (2 * Math.PI)) + 1) % 1 * bins) % bins;
    const r = Math.hypot(u, v) + 0.5;
    if (!(r <= radius[bin])) radius[bin] = r;
  }
  // Fill empty bins and soften the digital rim with a circular moving average.
  const known = [...radius.keys()].filter((i) => Number.isFinite(radius[i]));
  if (known.length < bins / 3) return;
  for (let i = 0; i < bins; i++) if (!Number.isFinite(radius[i])) {
    let a = i, b = i;
    while (!Number.isFinite(radius[(a + bins) % bins])) a--;
    while (!Number.isFinite(radius[b % bins])) b++;
    const t = (i - a) / (b - a);
    radius[i] = radius[(a + bins) % bins] * (1 - t) + radius[b % bins] * t;
  }
  const smooth = Array.from(radius, (_, i) => {
    let sum = 0;
    for (let k = -5; k <= 5; k++) sum += radius[(i + k + bins) % bins];
    return sum / 11;
  });
  const sample = (p: THREE.Vector3) => {
    const x = Math.round(inv[0][0] * p.x + inv[0][1] * p.y + inv[0][2] * p.z + inv[0][3]);
    const y = Math.round(inv[1][0] * p.x + inv[1][1] * p.y + inv[1][2] * p.z + inv[1][3]);
    const z = Math.round(inv[2][0] * p.x + inv[2][1] * p.y + inv[2][2] * p.z + inv[2][3]);
    if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return -1;
    return owner[x * yz + y * nz + z];
  };
  // Height of the skin along the exterior normal at each rim point, softened
  // around the loop so the line follows the skin rather than its cell steps.
  const bases: THREE.Vector3[] = [], heights: number[] = [];
  for (let i = 0; i < bins; i++) {
    const angle = ((i + 0.5) / bins) * 2 * Math.PI;
    const base = centre.clone().addScaledVector(e1, smooth[i] * Math.cos(angle)).addScaledVector(e2, smooth[i] * Math.sin(angle));
    let height = NaN;
    for (let s = 40; s > -20; s -= 0.25) if (sample(base.clone().addScaledVector(n, s)) > 0) { height = s; break; }
    bases.push(base);
    heights.push(height);
  }
  const found = heights.filter(Number.isFinite).length;
  if (found < bins * 0.8) return;
  const loop: Point3[] = [];
  for (let i = 0; i < bins; i++) {
    let sum = 0, weight = 0;
    for (let k = -3; k <= 3; k++) {
      const h = heights[(i + k + bins) % bins];
      if (Number.isFinite(h)) { sum += h; weight++; }
    }
    loop.push(bases[i].clone().addScaledVector(n, sum / weight + 0.75).toArray() as Point3);
  }
  return loop;
}
