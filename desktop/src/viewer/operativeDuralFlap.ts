import * as THREE from 'three';
import type {OperativeField} from '../operative-field';
import type {VoxelBounds} from './operativeSurface';

type Band = {x:number;y:number;low:number;high:number;corners:number[]};
type Grid = Pick<OperativeField,'shape'|'spacingMm'|'origin'>;

// The flap mesh rotates around the same hinge as the raw label. These poses
// supply a short cylindrical bend near the attachment in that rotating frame.
const poses=6;
// Stop just short of vertical. The sheet stays within the access window,
// with positive projected width even at a steep change in its source relief.
export const duraReflectionAngle=80*Math.PI/180;

export function updateDuralFold(mesh:THREE.Mesh,lift:number) {
  const weights=mesh.morphTargetInfluences;
  if(!weights?.length)return;
  const step=THREE.MathUtils.clamp(lift,0,1)*poses;
  for(let i=0;i<poses;i++)weights[i]=Math.max(0,1-Math.abs(step-i-1));
}

/** A two-sided representative surface for the reflected *estimated* label.
 * Each face is inside one occupied column run. Distinct or missing bands
 * cannot be connected by a triangle crossing an unsupported cell. */
export function estimatedDuralFlap(grid:Grid,mask:Uint8Array,bounds:VoxelBounds,firstY:number,
  hinge:THREE.Vector3,radius:number) {
  const {shape:[,ny,nz],spacingMm:s,origin}=grid;
  const bands:Band[]=[];
  const corners=new Map<number,Band[]>();
  const key=(x:number,y:number)=>x*(ny+1)+y;
  const firstRow=Math.max(firstY,bounds.low[1]);
  for(let x=bounds.low[0];x<=bounds.high[0];x++){
    for(let y=firstRow;y<=bounds.high[1];y++){
      const base=(x*ny+y)*nz;
      for(let z=bounds.low[2];z<=bounds.high[2];z++){
        if(!mask[base+z])continue;
        const low=z-.5;
        while(z<bounds.high[2]&&mask[base+z+1])z++;
        const band={x,y,low,high:z+.5,corners:[]};
        bands.push(band);
        for(const offset of [key(x,y),key(x+1,y),key(x+1,y+1),key(x,y+1)]){
          const touching=corners.get(offset);
          if(touching)touching.push(band);else corners.set(offset,[band]);
        }
      }
    }
  }
  const position:number[]=[],indices:number[]=[];
  for(const [id,touching] of corners){
    const u=Math.floor(id/(ny+1)),v=id%(ny+1);
    // Intersect vertical intervals before welding a shared corner. A gap
    // between touching columns stays open even if they share a material ID.
    touching.sort((a,b)=>a.low-b.low);
    for(let i=0;i<touching.length;){
      const start=i;
      let low=touching[i].low,high=touching[i].high,total=0;
      do{
        const band=touching[i++];
        low=Math.max(low,band.low);high=Math.min(high,band.high);
        total+=(band.low+band.high)*.5;
      }while(i<touching.length&&touching[i].low<high);
      const z=THREE.MathUtils.clamp(total/(i-start),low+.0001,high-.0001);
      let x=u-.5,y=v-.5;
      if(i-start===1){
        // Round the exposed convex corner inward by a quarter cell. Preserve
        // the flat inspection section at its exact half-cell boundary.
        const band=touching[start];
        x+=.5*(band.x-x);
        if(!(firstY>0&&v===firstY))y+=.5*(band.y-y);
      }
      const index=position.length/3;
      position.push(origin[0]+x*s-hinge.x,origin[1]+y*s-hinge.y,origin[2]+z*s-hinge.z);
      for(let j=start;j<i;j++){
        const band=touching[j];
        const slot=v===band.y?(u===band.x?0:1):(u===band.x?3:2);
        band.corners[slot]=index;
      }
    }
  }
  for(const {corners:[a,b,c,d]} of bands)indices.push(a,b,c,a,c,d);
  const geometry=new THREE.BufferGeometry();
  const rest=new THREE.Float32BufferAttribute(position,3);
  geometry.setAttribute('position',rest);geometry.setIndex(indices);geometry.computeVertexNormals();
  if(!rest.count)return geometry;

  // Mark a bend only when the lifted label reaches the geometric hinge.
  // Carry each run's original z relief across the fold. Choosing a separate
  // rotation axis from the nearest vertical band tears adjoining triangles.
  let reachesHinge=false;
  for(let i=0;i<rest.count;i++)if(Math.abs(rest.getX(i))<=s){reachesHinge=true;break;}
  if(!reachesHinge)return geometry;
  const bendWidth=Math.max(s,Math.min(radius*.28,2.4));
  geometry.morphAttributes.position=[];
  geometry.morphAttributes.normal=[];
  const pose=new THREE.BufferGeometry();
  pose.setIndex(geometry.index);
  for(let k=1;k<=poses;k++){
    const angle=duraReflectionAngle*k/poses,co=Math.cos(angle),si=Math.sin(angle);
    const posed=new Float32Array(rest.array);
    for(let i=0;i<rest.count;i++){
      const x=rest.getX(i),z=rest.getZ(i),d=Math.max(0,-x);
      const localAngle=angle*Math.min(d/bendWidth,1);
      const excess=Math.max(0,d-bendWidth),bendRadius=bendWidth/angle;
      // Integrate the tangent across the bend. Its neutral line retains arc
      // length and meets the stationary hinge with its original tangent.
      const dx=Math.max(0,x)-bendRadius*Math.sin(localAngle)-excess*co;
      const dz=z+bendRadius*(1-Math.cos(localAngle))+excess*si;
      posed[3*i]=co*dx-si*dz;posed[3*i+2]=si*dx+co*dz;
    }
    const vertices=new THREE.BufferAttribute(posed,3);
    pose.setAttribute('position',vertices);pose.computeVertexNormals();
    geometry.morphAttributes.position.push(vertices);
    geometry.morphAttributes.normal.push(pose.getAttribute('normal').clone());
  }
  pose.dispose();
  return geometry;
}
