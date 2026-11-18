import * as THREE from 'three';
import type {OperativeField} from '../operative-field';
import type {VoxelBounds} from './operativeSurface';

/** Emit exposed faces at exactly half a grid spacing from their sample
 * centres, including at a teaching section. The sampled mask stays intact. */
export function operativeVoxelSurface(grid:Pick<OperativeField,'shape'|'origin'|'spacingMm'>,mask:Uint8Array,bounds:VoxelBounds,firstY:number){
  const [nx,ny,nz]=grid.shape,stride=[ny*nz,nz,1],size=[nx,ny,nz];
  const lower=[0,firstY,0];
  const lo=bounds.low.map((n,a)=>Math.max(lower[a],n)),hi=bounds.high;
  const visit=(face:(p:number[],axis:number,sign:number)=>void)=>{
    const p=[0,0,0];
    for(p[0]=lo[0];p[0]<=hi[0];p[0]++)for(p[1]=lo[1];p[1]<=hi[1];p[1]++){
      let id=p[0]*stride[0]+p[1]*nz+lo[2];
      for(p[2]=lo[2];p[2]<=hi[2];p[2]++,id++){
        if(!mask[id])continue;
        for(let a=0;a<3;a++){
          if(p[a]===lower[a]||!mask[id-stride[a]])face(p,a,-1);
          if(p[a]===size[a]-1||!mask[id+stride[a]])face(p,a,1);
        }
      }
    }
  };
  let faces=0;
  visit(()=>{faces++});
  const geometry=new THREE.BufferGeometry();
  if(!faces)return geometry;
  const positions=new Float32Array(faces*12),normals=new Float32Array(faces*12);
  const indices=new Uint32Array(faces*6);
  let index=0;
  const corners=[[-1,-1],[1,-1],[1,1],[-1,1]];
  visit((p,a,sign)=>{
    const u=(a+1)%3,v=(a+2)%3,b=index*4;
    for(let j=0;j<4;j++){
      const o=(b+j)*3;
      positions[o+a]=grid.origin[a]+(p[a]+.5*sign)*grid.spacingMm;
      positions[o+u]=grid.origin[u]+(p[u]+.5*corners[j][0])*grid.spacingMm;
      positions[o+v]=grid.origin[v]+(p[v]+.5*sign*corners[j][1])*grid.spacingMm;
      normals[o+a]=sign;
    }
    indices.set([b,b+1,b+2,b,b+2,b+3],index*6);
    index++;
  });
  geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));
  geometry.setAttribute('normal',new THREE.BufferAttribute(normals,3));
  geometry.setIndex(new THREE.BufferAttribute(indices,1));
  return geometry;
}
