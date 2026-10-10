import * as THREE from "three";
import type { InstrumentCapsuleDisplay, InspectionToolDisplay } from "./inspectionTool.ts";

/** RAS points are already canonical. Never apply the source-frame conversion here. */
export function inspectionToolMeshes(display: InspectionToolDisplay): THREE.Group {
  return instrumentCapsuleMeshes(display, "unexecuted-native-axis-inspection", display.identity);
}

/** Shared capsule geometry, with execution provenance supplied explicitly. */
export function instrumentCapsuleMeshes(display: InstrumentCapsuleDisplay, scope: string, identity: string): THREE.Group {
  const group = new THREE.Group();
  const vector = (point: readonly number[]) => new THREE.Vector3(point[0], point[1], point[2]);
  const capsule = (a: readonly number[], b: readonly number[], radius: number, active: boolean) => {
    const start = vector(a), end = vector(b);
    const mesh = new THREE.Mesh(
      new THREE.CapsuleGeometry(radius, start.distanceTo(end), 5, 16),
      new THREE.MeshStandardMaterial({ color: display.color, metalness: active ? 0.2 : 0.62,
        roughness: active ? 0.3 : 0.23, transparent: true, opacity: active ? 1 : 0.9 }),
    );
    mesh.position.copy(start).add(end).multiplyScalar(0.5);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), end.sub(start).normalize());
    if (scope === "unexecuted-native-axis-inspection") mesh.userData.inspectionIdentity = identity;
    else mesh.userData.replayIdentity = identity;
    mesh.userData.scope = scope;
    mesh.userData.part = active ? "active-tip" : "shaft";
    group.add(mesh);
  };
  capsule(display.shaftStart, display.shaftEnd, display.shaftRadius, false);
  capsule(display.shaftEnd, display.tip, display.tipRadius, true);
  return group;
}
