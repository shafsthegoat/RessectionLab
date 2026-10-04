import * as THREE from "three";
import type { Affine } from "./coordinates";

/** Explicitly invalidate world matrices after assigning an authoritative affine.
 * Three's updateWorldMatrix() need not recompute a manually assigned matrix
 * unless this flag is set. Otherwise a newly added surface can enlarge camera
 * bounds in voxel coordinates until its first render, mixing two frames.
 */
export function placeInSourceFrame(
  object: THREE.Object3D,
  affine: Affine,
): void {
  object.matrix.set(
    ...(affine.flat() as [
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
    ]),
  );
  object.matrixAutoUpdate = false;
  object.matrixWorldNeedsUpdate = true;
}

export function physicalBounds(object: THREE.Object3D): THREE.Box3 {
  object.updateWorldMatrix(true, true, true);
  return new THREE.Box3().setFromObject(object);
}
