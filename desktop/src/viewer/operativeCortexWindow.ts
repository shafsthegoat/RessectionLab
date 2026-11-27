import * as THREE from 'three';
import type {FieldLayer, OperativeField} from '../operative-field';
import {detailBits} from '../operative-field.ts';
import {operativeDimensions} from './operativeFieldModel.ts';
import {installRetractionMorph} from './retractionMorph.ts';

/** Reuse native cortical triangles only superficial to the selected lesion and
 * all conflicts with the local tissue partition. The deeper measured interface
 * and cavity continue to use the route-grid geometry. Both limits use rest mm.
 */
export function corticalSourceWindow(field:OperativeField, layer:FieldLayer, radius:number, nativeDisplacementMm:number|undefined) {
  const s=field.spacingMm, [nx,ny,nz]=field.shape, [ox,oy,oz]=field.origin;
  const pitch=layer.labelSpacingMm??layer.sourceSpacingMm;
  if(!pitch||!pitch.every(n=>Number.isFinite(n)&&n>0)||nativeDisplacementMm===undefined||
    !Number.isFinite(nativeDisplacementMm)||nativeDisplacementMm<0)return null;
  const margin=.5*Math.hypot(...pitch)+nativeDisplacementMm+Math.sqrt(3)*s;
  let protectedZ=-1;
  for(let x=0;x<nx;x++)for(let y=0;y<ny;y++){
    if(Math.hypot(ox+x*s,oy+y*s)>radius+margin)continue;
    const start=(x*ny+y)*nz;
    for(let z=nz-1;z>protectedZ;z--){
      const bits=field.voxels[start+z];
      if(bits&8 || (bits&detailBits.grayMatter)&&bits!==(4|detailBits.grayMatter)){
        protectedZ=z;break;
      }
    }
  }
  const top=oz+(nz-.5)*s, floor=protectedZ<0?oz-.5*s:oz+protectedZ*s+margin;
  return {floor,top,margin,protectedZ:protectedZ<0?null:oz+protectedZ*s,activeFloor:{value:top+s}};
}

export type CortexWindow=NonNullable<ReturnType<typeof corticalSourceWindow>>;

/** The registered planning mesh is in native index space. Its morph must
 * represent the same physical shift as the local geometric illustration. */
export function corticalRetraction(mesh:THREE.Mesh,field:OperativeField,radius:number,window:CortexWindow){
  const p=mesh.geometry.getAttribute('position'),v=new THREE.Vector3(),m=mesh.matrixWorld;
  const {opening}=operativeDimensions(field);
  let exposedVertices=0;
  for(let i=0;i<p.count;i++){
    v.fromBufferAttribute(p,i).applyMatrix4(m);
    if(v.z>=window.floor&&v.z<=window.top&&v.x*v.x+v.y*v.y<=radius*radius)exposedVertices++;
  }
  if(exposedVertices&&field.stations.brain!==null){
    installRetractionMorph(mesh.geometry,opening,field.stations.brain,m);
    mesh.updateMorphTargets();
  }
  return exposedVertices;
}
