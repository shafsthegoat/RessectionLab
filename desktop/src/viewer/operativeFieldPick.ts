import * as THREE from 'three';
import type {FieldLayer,OperativeField} from '../operative-field';
import {inverseAffine,transformPoint,type Point3} from './coordinates.ts';

type PickedSource=Pick<FieldLayer,'id'|'label'|'evidence'|'sourceSeries'|'coverageMeaning'|'reviewStatus'>;
export type OperativePick={layers:PickedSource[];ras:Point3;voxel:Point3|null;series:string|null};
export function pickOperativeStructure(scene:THREE.Scene,camera:THREE.Camera,ndc:THREE.Vector2,field:OperativeField):OperativePick|null{
  const ray=new THREE.Raycaster(),visible:THREE.Mesh[]=[];
  ray.setFromCamera(ndc,camera);
  scene.traverseVisible(o=>{if(o instanceof THREE.Mesh)visible.push(o)});
  for(const hit of ray.intersectObjects(visible,false)){
    const mesh=hit.object;
    if(!(mesh instanceof THREE.Mesh))continue;
    const identity=mesh.userData.sourceLayer;
    const section=mesh.userData.acquiredSection;
    if(section&&hit.point.x>=section.minX&&hit.point.x<=section.maxX&&hit.point.z>=section.minZ&&hit.point.z<=section.maxZ)continue;
    if(!identity)return null; // A visible instrument occludes the source surface.
    const {directId,role,detail,offset,openingLimit,kerf,patchRadius,exterior}=identity;
    const point=hit.point.clone();
    if(directId){
      if(Math.hypot(point.x,point.y)<=patchRadius&&point.z>=field.origin[2]-.5*field.spacingMm&&point.z<=field.origin[2]+(field.shape[2]-.5)*field.spacingMm)continue;
    }else{
      if(!hit.face||!hit.barycoord)continue;
      const p=mesh.geometry.getAttribute('position'),face=hit.face,weights=hit.barycoord.toArray();
      point.set(0,0,0);
      [face.a,face.b,face.c].forEach((v,i)=>point.addScaledVector(new THREE.Vector3().fromBufferAttribute(p,v),weights[i]));
      if(exterior?.value&&Math.hypot(point.x,point.y)>patchRadius)continue;
      if(openingLimit?.value>=0){
        const angle=THREE.MathUtils.euclideanModulo(Math.atan2(point.y,point.x),2*Math.PI);
        if((angle<openingLimit.value)!==kerf)continue;
      }
      if(offset)point.add(offset); // Rest-space fragment before raising or reflecting a flap.
    }
    const layers:PickedSource[]=identity.contextLayer?[identity.contextLayer]:field.layers.filter(layer=>{
      if(!layer.sampledVoxels)return false;
      if(directId)return layer.id===directId;
      if(role==='tumor')return layer.id===field.targetLayerId;
      if(detail)return layer.detail===detail;
      return layer.role===role&&!layer.detail;
    });
    if(!layers.length)return null;
    const local=point.toArray(),ras=field.targetRasMm.map((v,i)=>v+field.basisRas.reduce((sum,axis,j)=>sum+axis[i]*local[j],0)) as Point3;
    const owned=identity.appearance;
    const signal=directId?(owned?.sourceVoxelToPatientRasMm&&owned?.sourceShape
      ?{sourceSeries:owned.seriesId,sourceShape:owned.sourceShape,sourceVoxelToPatientRasMm:owned.sourceVoxelToPatientRasMm}
      :undefined):field.intensity;
    const grid=signal?transformPoint(inverseAffine(signal.sourceVoxelToPatientRasMm),ras):null;
    const voxel=grid?.map(Math.round) as Point3|undefined;
    return {layers,ras,voxel:voxel&&voxel.every((v,i)=>v>=0&&v<signal!.sourceShape[i])?voxel:null,series:signal?.sourceSeries??null};
  }
  return null;
}
