import * as THREE from 'three';
import type {Point3} from './coordinates';

/** A display envelope confined to the recorded free cells. Adjacent deposits
 * share a cross-section at their common face. The cell amounts, rather than
 * a subdivision of the envelope, remain the volume measurement. */
export type FieldSample = {cell:Point3; amount:number};

const cubeEdges = Array.from({length:8},(_,corner)=>
  [0,1,2].filter(axis=>(corner & (1<<axis))===0).map(axis=>[corner,corner|(1<<axis)] as const)).flat();

export function joinedFreeField(samples:FieldSample[], shape:Point3, capacity:number) {
  if(samples.length<2 || samples.length>2048)return null;
  const n=samples.length<=96?4:2;
  const stride=[shape[1]*shape[2],shape[2],1];
  const cellId=(p:Point3)=>p[0]*stride[0]+p[1]*stride[1]+p[2];
  const radii=new Map(samples.map(s=>[cellId(s.cell),.48*Math.cbrt(Math.min(1,s.amount/capacity))]));
  const joined=new Set<number>();
  const positions:number[]=[], indices:number[]=[];
  // Dual contour vertices live inside individual subcells. The shared key
  // connects facing patches across wet cell boundaries without bridging dry ones.
  const pitchZ=shape[2]*n+1, pitchX=(shape[1]*n+1)*pitchZ;
  const strides=[pitchX,pitchZ,1];
  const dual=new Map<number,number>();
  const crossings:Array<[number,number,boolean]>=[];
  const side=n+1, row=side*side;
  const corners=[0,row,side,row+side,1,row+1,side+1,row+side+1];
  const values=new Float64Array(side**3);
  for(const {cell} of samples){
    const id=cellId(cell), radius=radii.get(id)!;
    const reaches=[0,1,2].flatMap(axis=>[-1,1].map(sign=>{
      const next=cell[axis]+sign;
      return next>=0&&next<shape[axis]?radii.get(id+sign*stride[axis])??0:0;
    }));
    if(!reaches.some(r=>r>0))continue;
    joined.add(id);
    for(let x=0;x<=n;x++)for(let y=0;y<=n;y++)for(let z=0;z<=n;z++){
      const dx=x/n-.5,dy=y/n-.5,dz=z/n-.5;
      const dist2=dx*dx+dy*dy+dz*dz;
      let value=radius-Math.sqrt(dist2);
      for(let axis=0;axis<3;axis++){
        const distance=axis===0?dx:axis===1?dy:dz;
        const neighbor=reaches[2*axis+Number(distance>0)];
        if(!neighbor)continue;
        const t=2*Math.abs(distance);
        const neck=radius+(Math.min(radius,neighbor)-radius)*t*t*(3-2*t);
        value=Math.max(value,neck-Math.sqrt(Math.max(0,dist2-distance*distance)));
      }
      values[x*row+y*side+z]=value;
    }
    for(let x=0;x<n;x++)for(let y=0;y<n;y++)for(let z=0;z<n;z++){
      const local=x*row+y*side+z;
      const positive=values[local]>0;
      if(corners.every(offset=>(values[local+offset]>0)===positive))continue;
      const gx=cell[0]*n+x, gy=cell[1]*n+y, gz=cell[2]*n+z;
      const key=gx*pitchX+gy*pitchZ+gz;
      const vertex=[0,0,0];let count=0;
      for(const [a,b] of cubeEdges){
        const va=values[local+corners[a]],vb=values[local+corners[b]];
        if((va>0)===(vb>0))continue;
        const t=va/(va-vb);
        for(let axis=0;axis<3;axis++){
          const start=(a>>axis)&1, end=(b>>axis)&1;
          vertex[axis]+=start+t*(end-start);
        }
        count++;
      }
      dual.set(key,positions.length/3);
      positions.push((gx+vertex[0]/count)/n-.5,(gy+vertex[1]/count)/n-.5,(gz+vertex[2]/count)/n-.5);
      for(let axis=0;axis<3;axis++)if((values[local+corners[1<<axis]]>0)!==positive)
        crossings.push([key,axis,positive]);
    }
  }
  if(!positions.length)return null;
  for(const [key,axis,positive] of crossings){
    const u=strides[(axis+1)%3], v=strides[(axis+2)%3];
    const a=dual.get(key)!, b=dual.get(key-u), c=dual.get(key-u-v), d=dual.get(key-v);
    if(b===undefined||c===undefined||d===undefined)continue;
    if(positive)indices.push(a,b,c,a,c,d);
    else indices.push(a,d,c,a,c,b);
  }
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
  geometry.setIndex(indices);geometry.computeVertexNormals();
  return {geometry,joined};
}
