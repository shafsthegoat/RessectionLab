import * as THREE from 'three';

/** Fit unchanged surface vertices, avoiding empty corners of a world AABB.
 * The occupied fraction is an NDC limit: .92 leaves 4% of the pane per side.
 * This display-only pass allocates no per-vertex vectors and changes no mesh.
 */
export function surfaceCameraDistance(
  anatomy: THREE.Object3D,
  centre: THREE.Vector3,
  direction: THREE.Vector3,
  right: THREE.Vector3,
  up: THREE.Vector3,
  verticalFovRadians: number,
  aspect: number,
  occupiedFraction = .92,
): number | null {
  const vertical = Math.tan(verticalFovRadians / 2) * occupiedFraction;
  const horizontal = vertical * aspect;
  let distance = 40, count = 0;
  anatomy.updateWorldMatrix(true,true);
  anatomy.traverse(object=>{
    if(!(object instanceof THREE.Mesh))return;
    const positions=object.geometry.getAttribute('position');
    if(!positions)return;
    const m=object.matrixWorld.elements;
    const activeIndices=object.geometry.userData.sourceIndexing==='stable-source-ids'?object.geometry.getIndex():null;
    // Allocated only for an explicit fit. Stable snapshots cap positions plus
    // normals plus indices at 64 MiB, so this byte-per-slot scratch is <2.8 MB.
    const visited=activeIndices?new Uint8Array(positions.count):null;
    for(let slot=0;slot<(activeIndices?.count??positions.count);slot++){
      const i=activeIndices?activeIndices.getX(slot):slot;
      if(visited){if(visited[i])continue;visited[i]=1;}
      const x=positions.getX(i),y=positions.getY(i),z=positions.getZ(i);
      const dx=m[0]*x+m[4]*y+m[8]*z+m[12]-centre.x;
      const dy=m[1]*x+m[5]*y+m[9]*z+m[13]-centre.y;
      const dz=m[2]*x+m[6]*y+m[10]*z+m[14]-centre.z;
      const depth=dx*direction.x+dy*direction.y+dz*direction.z;
      distance=Math.max(distance,
        depth+Math.abs(dx*right.x+dy*right.y+dz*right.z)/horizontal,
        depth+Math.abs(dx*up.x+dy*up.y+dz*up.z)/vertical);
      count++;
    }
  });
  return count && Number.isFinite(distance) ? distance : null;
}
