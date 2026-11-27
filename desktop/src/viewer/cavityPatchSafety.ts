import type {Affine} from './coordinates';
import {intersectionVertices} from './surfaceSafety.ts';
import {indexedSurfaceNormals} from './surfaceNormals.ts';

/** Audit the patch and nearby opposing faces in physical coordinates. The
 * four-cell bucket halo exceeds two extraction tetrahedra plus the bounded
 * source error. Distant patient triangles cannot newly intersect this patch. */
export function guardCavityPatch(points:Float32Array,normals:Float32Array,indices:Uint32Array,original:Float32Array,
  extraOriginal:number[],affected:Iterable<number>,affine:Affine){
  const bucket=(x:number,y:number,z:number)=>(x*65536+y)*65536+z;
  const active=new Set<number>(),mutable=new Set(affected);
  const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
  for(const v of mutable){
    const xyz=[0,1,2].map(k=>Math.floor(points[3*v+k]/4));
    for(let k=0;k<3;k++){lo[k]=Math.min(lo[k],xyz[k]-1);hi[k]=Math.max(hi[k],xyz[k]+1);}
    for(let x=-1;x<=1;x++)for(let y=-1;y<=1;y++)for(let z=-1;z<=1;z++)active.add(bucket(xyz[0]+x,xyz[1]+y,xyz[2]+z));
  }
  const local=new Map<number,number>(),ids:number[]=[],faces:number[]=[];
  for(let f=0;f<indices.length;f+=3){
    const v=3*indices[f],x=Math.floor(points[v]/4),y=Math.floor(points[v+1]/4),z=Math.floor(points[v+2]/4);
    if(x<lo[0]||x>hi[0]||y<lo[1]||y>hi[1]||z<lo[2]||z>hi[2]||!active.has(bucket(x,y,z)))continue;
    if(faces.length>=120000)return false;
    for(let j=0;j<3;j++){
      const g=indices[f+j];let id=local.get(g);
      if(id===undefined){id=ids.length;ids.push(g);local.set(g,id);}
      faces.push(id);
    }
  }
  const base=new Float64Array(3*ids.length),current=new Float64Array(base.length);
  const source=(v:number,k:number)=>3*v<original.length?original[3*v+k]:extraOriginal[3*v-original.length+k];
  const copy=(output:Float64Array,read:(id:number,k:number)=>number)=>{
    for(let j=0;j<ids.length;j++)for(let k=0;k<3;k++){
      const row=affine[k],g=ids[j];output[3*j+k]=row[0]*read(g,0)+row[1]*read(g,1)+row[2]*read(g,2);
    }
  };
  const faceArray=Uint32Array.from(faces);
  const cross=(p:Float64Array,i:number)=>{
    const a=3*faces[i],b=3*faces[i+1],c=3*faces[i+2],x=p[b]-p[a],y=p[b+1]-p[a+1],z=p[b+2]-p[a+2],
      u=p[c]-p[a],v=p[c+1]-p[a+1],w=p[c+2]-p[a+2];
    return [y*w-z*v,z*u-x*w,x*v-y*u];
  };
  copy(base,source);
  for(let pass=0;pass<5;pass++){
    copy(current,(g,k)=>points[g*3+k]);
    const bad=new Set<number>();
    for(let i=0;i<faces.length;i+=3){
      const a=cross(base,i),b=cross(current,i),orientation=a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
      if(orientation<=.01*(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]))for(let k=0;k<3;k++)bad.add(faces[i+k]);
    }
    if(!bad.size)for(const v of intersectionVertices(base,current,faceArray))bad.add(v);
    if(!bad.size){
      const voxels=Float32Array.from({length:base.length},(_,j)=>points[3*ids[Math.floor(j/3)]+j%3]);
      const geometryNormals=indexedSurfaceNormals(voxels,faceArray,affine);
      for(const g of mutable){
        const j=local.get(g);if(j===undefined)continue;
        const n=[0,1,2].map(k=>.55*normals[g*3+k]+.45*geometryNormals[j*3+k]);
        const len=Math.hypot(...n);if(len)for(let k=0;k<3;k++)normals[g*3+k]=n[k]/len;
      }
      return true;
    }
    for(const j of bad){
      const g=ids[j];if(!mutable.has(g))continue;
      for(let k=0;k<3;k++)points[g*3+k]=source(g,k)+.40*(points[g*3+k]-source(g,k));
    }
  }
  return false;
}
