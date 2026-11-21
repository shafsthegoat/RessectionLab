import type {Affine, Shape3} from './coordinates';
import {fairCavityPatch} from './fairCavityPatch.ts';
import {guardCavityPatch} from './cavityPatchSafety.ts';
import type {CavityRail} from './fairCavityPatch';

export interface AcceptedCavity {region:number;cells:Uint32Array}
type Edge = {inside:number[];outside:number[];insideWeight:number;outsideWeight:number;insideCut:number;outsideCut:number;cut:boolean;exterior:boolean;pinned:boolean;free:boolean};

/** The accepted binary surface splices original material interfaces directly
 * onto cut faces. In a mixed tetrahedron that puts a diagonal across the
 * junction. Split the splice along the intersection of two linear fields:
 * original material membership, and retained-vs-removed membership within
 * that material. Other materials never vote on this layer's removed mass. */
export function joinCavityWalls(raw:Float32Array,shown:Float32Array,normals:Float32Array,indices:Uint32Array,
  edges:Map<number,Edge>,affine:Affine,limit:number) {
  if(edges.size>12000)return;
  const extra:number[]=[],extraOriginal:number[]=[],extraNormals:number[]=[],split=new Map<string,number>();
  const changed=new Map<number,number[]>(),rails=new Map<number,CavityRail>();
  for(const [id,e] of edges){
    if((e.cut||e.exterior)&&!e.pinned)rails.set(id,{inside:e.inside,outside:e.outside,free:e.free,anchor:Array.from(raw.subarray(3*id,3*id+3))});
  }
  let maximum=0,junctionTriangles=0,curvedEdges=0,additionalIndices=0;
  const vertex=(a:number,b:number)=>{
    const key=a<b?a+','+b:b+','+a;
    const known=split.get(key);if(known!==undefined)return known;
    const u=edges.get(a),v=edges.get(b);
    if(!u||!v||(!u.cut&&!v.cut&&!(u.exterior&&v.exterior)))return;
    const mixed=u.cut!==v.cut;
    let ideal:number[],fraction=.5,freeRail:Pick<CavityRail,'inside'|'outside'>|undefined;
    if(mixed){
      const cut=u.cut?u:v,other=u.cut?v:u;
      // In this tetrahedron the original interface and the cut are two
      // linear scalar fields. Their common zero has native retained mass R,
      // removed mass W and other material mass O. The cut requires rR=wW;
      // the native interface requires aR+bW=cO. Solve in barycentric space
      // so the reconstructed seam stays within the registered source simplex.
      const retained=[0,1,2].map(k=>(cut.inside[k]+2*other.inside[k])/3);
      const removedMass=(cut.insideCut+2*other.insideCut)/(3*cut.outsideCut);
      const otherMass=((cut.insideWeight+2*other.insideWeight)/3+cut.outsideWeight*removedMass)/other.outsideWeight;
      const total=1+removedMass+otherMass;
      ideal=retained.map((p,k)=>(p+removedMass*cut.outside[k]+otherMass*other.outside[k])/total);
      fraction=u.cut?2/3:1/3;
    }else{
      // Split a wall chord before fitting it. Both endpoint crossings give
      // a material barycentre and a vacant barycentre within the same native
      // simplex. Their segment is the allowed rail near any other material.
      const midpoint=[0,1,2].map(k=>(shown[3*a+k]+shown[3*b+k])/2);
      const fractionInside=(e:Edge,id:number)=>{
        const axis=e.inside.findIndex((x,k)=>x!==e.outside[k]);
        return (shown[id*3+axis]-e.outside[axis])/(e.inside[axis]-e.outside[axis]);
      };
      const fa=fractionInside(u,a),fb=fractionInside(v,b),f=(fa+fb)/2;
      const inside=u.inside.map((x,k)=>(fa*x+fb*v.inside[k])/(2*f));
      const outside=u.outside.map((x,k)=>((1-fa)*x+(1-fb)*v.outside[k])/(2*(1-f)));
      freeRail={inside,outside};
      ideal=midpoint;
      curvedEdges++;
    }
    const native=[0,1,2].map(k=>raw[3*a+k]*(1-fraction)+raw[3*b+k]*fraction);
    const original=[0,1,2].map(k=>shown[3*a+k]*(1-fraction)+shown[3*b+k]*fraction);
    const physical=(point:number[])=>affine.slice(0,3).map(row=>row.slice(0,3).reduce((s,w,k)=>s+w*(point[k]-native[k]),0));
    const delta=physical(ideal),length=Math.hypot(...delta);
    let point=ideal;
    if(length>limit*.995){
      // Both endpoints are inside the original material halfspace. Follow
      // their segment up to the permitted physical error sphere.
      let lo=0,hi=1;
      for(let i=0;i<15;i++){
        const m=(lo+hi)/2,p=original.map((v,k)=>v+(ideal[k]-v)*m);
        if(Math.hypot(...physical(p))<=limit*.995)lo=m;else hi=m;
      }
      point=original.map((v,k)=>v+(ideal[k]-v)*lo);
    }
    maximum=Math.max(maximum,Math.hypot(...physical(point)));
    const id=(shown.length+extra.length)/3;
    extra.push(...point);extraOriginal.push(...original);
    if(!u.pinned&&!v.pinned){
      const rail=freeRail??{inside:ideal,outside:original};
      rails.set(id,{...rail,free:!mixed&&u.free&&v.free,anchor:native});
    }
    const n=[0,1,2].map(k=>normals[3*a+k]*(1-fraction)+normals[3*b+k]*fraction);
    const scale=Math.hypot(...n)||1;extraNormals.push(...n.map(x=>x/scale));
    split.set(key,id);return id;
  };
  for(let i=0;i<indices.length;i+=3){
    const a=indices[i],b=indices[i+1],c=indices[i+2];
    if(!edges.has(a)&&!edges.has(b)&&!edges.has(c))continue;
    const face=[a,b,c],mid=[vertex(a,b),vertex(b,c),vertex(c,a)];
    const count=mid.filter(x=>x!==undefined).length;if(!count)continue;
    if(face.some(v=>edges.get(v)?.cut)&&face.some(v=>edges.get(v)?.cut===false))junctionTriangles++;
    let triangles:number[];
    if(count===3){
      const [ab,bc,ca]=mid as number[];
      triangles=[a,ab,ca,b,bc,ab,c,ca,bc,ab,bc,ca];
    }else if(count===1){
      const k=mid.findIndex(v=>v!==undefined),m=mid[k]!,x=face[k],y=face[(k+1)%3],z=face[(k+2)%3];
      triangles=[x,m,z,m,y,z];
    }else{
      const k=mid.findIndex(v=>v===undefined),x=face[(k+2)%3],y=face[k],z=face[(k+1)%3];
      const xy=mid[(k+2)%3]!,xz=mid[(k+1)%3]!;
      triangles=[x,xy,xz,xy,y,z,xy,z,xz];
    }
    changed.set(i,triangles);additionalIndices+=triangles.length-3;
  }
  if(!changed.size)return;
  const positions=new Float32Array(shown.length+extra.length),smoothNormals=new Float32Array(positions.length);
  positions.set(shown);positions.set(extra,shown.length);
  smoothNormals.set(normals);smoothNormals.set(extraNormals,normals.length);
  const smoothIndices=new Uint32Array(indices.length+additionalIndices);
  let offset=0,first=0;
  for(const [i,triangles] of changed){
    smoothIndices.set(indices.subarray(first,i),offset);offset+=i-first;
    smoothIndices.set(triangles,offset);offset+=triangles.length;first=i+3;
  }
  smoothIndices.set(indices.subarray(first),offset);
  maximum=Math.max(maximum,fairCavityPatch(positions,smoothIndices,rails,affine,limit));
  const affected=new Set(rails.keys());
  for(let v=shown.length/3;v<positions.length/3;v++)affected.add(v);
  if(!guardCavityPatch(positions,smoothNormals,smoothIndices,shown,extraOriginal,affected,affine))return;
  return {positions,smoothNormals,smoothIndices,maximum,junctionTriangles,curvedEdges};
}

export type CavityEdge=Edge;

export function cavityNeighborhood(cavity:AcceptedCavity,shape:Shape3){
  const [nx,ny,nz]=shape,yz=ny*nz,cells=new Set(cavity.cells),near=new Set<number>();
  for(const i of cells){
    const x=Math.floor(i/yz),y=Math.floor(i/nz)%ny,z=i%nz;
    for(let a=Math.max(0,x-2);a<=Math.min(nx-1,x+2);a++)
      for(let b=Math.max(0,y-2);b<=Math.min(ny-1,y+2);b++)
        for(let c=Math.max(0,z-2);c<=Math.min(nz-1,z+2);c++)near.add(a*yz+b*nz+c);
  }
  return {region:cavity.region,cells,near};
}
