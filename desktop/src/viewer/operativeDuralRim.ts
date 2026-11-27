import * as THREE from 'three';
import {mergeGeometries} from 'three/addons/utils/BufferGeometryUtils.js';
import type {OperativeModel} from './operativeFieldModel.ts';
import {emptyVoxelBounds,extendVoxelBounds,operativeSurface} from './operativeSurface.ts';
import {estimatedDuralSheet} from './operativeDuralFlap.ts';

/** Continue the illustrated dural midsurface across the stationary rim. End
 * this change beneath a complete ring of source-supported, uncut near-side
 * skull, preserving other dural runs and the original section presentation. */
export function prepareDuralRim(model:OperativeModel){
  const f=model.field,station=f.stations.dura;
  if(station===null||f.stations.bone===null)return null;
  const {spacingMm:s,origin:[ox,oy,oz],shape:[nx,ny,nz]}=f;
  const band=model.layers.dura,limit=model.opening+Math.max(2,4*s);
  if(limit+s>=f.radiusMm)return null;
  const depth=Math.max(8,model.opening*.9,4*s),near=emptyVoxelBounds();
  const cutEdge=model.duralOpening+Math.max(.7*s,model.opening*.065);
  const members:number[]=[];
  const remaining=band.mask.slice();
  for(let x=0;x<nx;x++)for(let y=0;y<ny;y++){
    const radius=Math.hypot(ox+x*s,oy+y*s);
    if(radius>limit+s)continue;
    const row=(x*ny+y)*nz;
    let outerBone=-1;
    for(let z=nz-1;z>=0;z--)if(f.voxels[row+z]&1){outerBone=z;break;}
    for(let z=0;z<nz;z++){
      if(!band.mask[row+z])continue;
      const low=z;while(z+1<nz&&band.mask[row+z+1])z++;
      const high=z;
      if(oz+high*s<station-depth||oz+low*s>station+depth)continue;
      // Keep whole occupied runs across a sloping band. A longitudinal
      // column thicker than the complete operative half-window provides
      // insufficient evidence for a representative transverse sheet.
      if((high-low+1)*s>depth){
        if(radius<=cutEdge)return null;
        continue; // Preserve unrelated, uncut thick runs as sampled volumes.
      }
      if(Math.abs(radius-limit)<=s&&outerBone<=high)return null;
      if(radius>limit)continue;
      for(let k=low;k<=high;k++){members.push(row+k);remaining[row+k]=0;}
      extendVoxelBounds(near,x,y,low);extendVoxelBounds(near,x,y,high);
    }
  }
  if(!members.length)return null;
  const outer=operativeSurface(f,remaining,band.bounds,0,'dura');
  // Only members inside the rim can be changed by this illustrated opening.
  // The original and kerfed states can reuse the same uncut remainder.
  const local=new Uint8Array(band.mask.length);
  return {
    geometry(){
      for(const k of members)local[k]=band.mask[k];
      const sheet=estimatedDuralSheet(f,local,near,0,new THREE.Vector3());
      if(!outer.getAttribute('position')?.count)return sheet;
      if(!sheet.getAttribute('position')?.count){sheet.dispose();return outer.clone();}
      const joined=mergeGeometries([outer,sheet]);
      sheet.dispose();
      return joined;
    },
    dispose(){outer.dispose();}
  };
}
