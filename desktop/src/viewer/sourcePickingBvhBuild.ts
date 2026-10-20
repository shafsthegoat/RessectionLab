import * as THREE from 'three';
import {MeshBVH,type SerializedBVH} from 'three-mesh-bvh';

export interface SourceBvhInput {
  id:number; positions:Float32Array; index:Uint16Array|Uint32Array|null;
  groups:Array<{start:number;count:number;materialIndex?:number}>;
  drawRange:{start:number;count:number};
}
export interface SourceBvhResult {id:number;serialized:SerializedBVH;buildMs:number;rootBytes:number}
/** Runs in the dedicated worker; indirect construction preserves triangle IDs. */
export function buildSourcePickingBvh(input:SourceBvhInput):SourceBvhResult {
  const started=performance.now(),geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.BufferAttribute(input.positions,3));
  if(input.index)geometry.setIndex(new THREE.BufferAttribute(input.index,1));
  for(const group of input.groups)geometry.addGroup(group.start,group.count,group.materialIndex);
  geometry.setDrawRange(input.drawRange.start,input.drawRange.count);
  const tree=new MeshBVH(geometry,{indirect:true,setBoundingBox:false});
  const serialized=MeshBVH.serialize(tree,{cloneBuffers:false});
  // Source index ownership never crosses back. deserialize(setIndex:false)
  // retains the exact original source index and uses only the indirect map.
  serialized.index=null;
  const rootBytes=serialized.roots.reduce((n,root)=>n+root.byteLength,0)+(serialized.indirectBuffer?.byteLength??0);
  return {id:input.id,serialized,rootBytes,buildMs:performance.now()-started};
}
