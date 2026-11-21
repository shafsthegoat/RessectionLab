import * as THREE from 'three';
import type {FieldLayer, OperativeField} from '../operative-field';
import {detailBits} from '../operative-field.ts';
import {operativeDimensions, retractionOffset} from './operativeFieldModel.ts';

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
  const localToNative=new THREE.Matrix3().setFromMatrix4(m).invert();
  const offsets=new Float32Array(p.count*3);
  const {opening}=operativeDimensions(field);
  let exposedVertices=0;
  for(let i=0;i<p.count;i++){
    v.fromBufferAttribute(p,i).applyMatrix4(m);
    if(v.z>=window.floor&&v.z<=window.top&&v.x*v.x+v.y*v.y<=radius*radius)exposedVertices++;
    // Include both ends of crossing triangles. Clipping uses rest positions,
    // so stopping the morph on the cutoff plane would tug the seam apart.
    const station=field.stations.brain;
    if(station===null||Math.abs(v.x)>4*opening||Math.abs(v.y)>4*opening||Math.abs(v.z-station)>4*opening)continue;
    const [dx,dy,dz]=retractionOffset(v.x,v.y,v.z,opening,station);
    v.set(dx,dy,dz).applyMatrix3(localToNative).toArray(offsets,3*i);
  }
  if(exposedVertices){
    mesh.geometry.morphAttributes.position=[new THREE.BufferAttribute(offsets,3)];
    mesh.geometry.morphTargetsRelative=true;
    mesh.updateMorphTargets();
  }
  return exposedVertices;
}
