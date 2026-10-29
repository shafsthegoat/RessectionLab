import {inverseAffine} from './coordinates.ts';
import type {Affine,Shape3} from './coordinates';
import {SURFACE_POLICY,surfaceKind,sincCoefficients} from './surfacePolicy.ts';
import type {SurfaceDiagnostics} from './surfacePolicy';
import {intersectionVertices} from './surfaceSafety.ts';
import type {SurfaceIntersectionAudit} from './surfaceSafety';
import type {SurfacePolynomialFilter} from './incrementalSurfaceFilter';

// Raster source soups revisit at most three adjacent half-cell X planes.
// Keep that bounded lookup window; arbitrary triangle order falls back to the
// retained exact Map implementation instead of welding or reordering vertices.
const MAX_LATTICE_LOOKUP_BYTES=32*1024**2;
let latticeLookup:Uint32Array|undefined;
function indexLatticeSurface(positions:Float32Array,span:number[]){
  const planeSize=span[1]*span[2],count=planeSize*3;
  if(!latticeLookup||latticeLookup.length<count)latticeLookup=new Uint32Array(count);
  const lookup=latticeLookup,planes=[-1,-1,-1],indices=new Uint32Array(positions.length/3),fallback=new Map<string,number>();
  let highest=-Infinity,points=new Float32Array(Math.min(positions.length,49152)),length=0;
  function append(offset:number){
    if(length+3>points.length){const next=new Float32Array(Math.min(positions.length,points.length*2));next.set(points);points=next;}
    const id=length/3;points[length++]=positions[offset];points[length++]=positions[offset+1];points[length++]=positions[offset+2];return id;
  }
  for(let i=0;i<positions.length;i+=3){
    const x=positions[i]*2+1,y=positions[i+1]*2+1,z=positions[i+2]*2+1;
    const lattice=Number.isInteger(x)&&Number.isInteger(y)&&Number.isInteger(z)&&x>=0&&y>=0&&z>=0&&x<span[0]&&y<span[1]&&z<span[2];
    let id:number|undefined;
    if(lattice){
      if(x<highest-2)return null;
      if(x>highest)highest=x;
      const plane=x%3,base=plane*planeSize;
      if(planes[plane]!==x){lookup.fill(0,base,base+planeSize);planes[plane]=x;}
      const key=base+y*span[2]+z;
      const recorded=lookup[key];id=recorded?recorded-1:undefined;
      if(id===undefined){id=append(i);lookup[key]=id+1;}
    }else {
      const key=`${positions[i]},${positions[i+1]},${positions[i+2]}`;
      id=fallback.get(key);if(id===undefined){id=append(i);fallback.set(key,id);}
    }
    indices[i/3]=id;
  }
  return {positions:points.slice(0,length),indices};
}
export function indexSurfaceLookupStatistics(){return {retainedBytes:latticeLookup?.byteLength??0,maximumBytes:MAX_LATTICE_LOOKUP_BYTES};}

/** Exact-coordinate indexing only; no tolerance welds nearby branches. */
export function indexSurface(positions:Float32Array,shape?:Shape3){
  if(positions.length%9||!positions.every(Number.isFinite))throw Error('Display smoothing requires finite complete triangles.');
  const span=shape?.map(n=>2*n+1),grid=span&&span.every(n=>Number.isSafeInteger(n)&&n>0)&&Number.isSafeInteger(span[0]*span[1]*span[2]);
  if(grid&&positions.length>65536&&span[1]*span[2]*3*4<=MAX_LATTICE_LOOKUP_BYTES){const indexed=indexLatticeSurface(positions,span);if(indexed)return indexed;}
  const points:number[]=[],lookup=new Map<string|number,number>(),indices=new Uint32Array(positions.length/3);
  for(let i=0;i<positions.length;i+=3){
    const x=positions[i]*2+1,y=positions[i+1]*2+1,z=positions[i+2]*2+1;
    // Marching-tetrahedra vertices occupy the exact half-cell lattice. This
    // injective integer key welds exactly the same coordinates as the generic
    // string path; arbitrary/non-grid geometry keeps that retained fallback.
    const key=grid&&Number.isSafeInteger(x)&&Number.isSafeInteger(y)&&Number.isSafeInteger(z)&&
      x>=0&&y>=0&&z>=0&&x<span![0]&&y<span![1]&&z<span![2]?(x*span![1]+y)*span![2]+z:`${positions[i]},${positions[i+1]},${positions[i+2]}`;
    let id=lookup.get(key);
    if(id===undefined){id=points.length/3;lookup.set(key,id);points.push(positions[i],positions[i+1],positions[i+2]);}indices[i/3]=id;}
  return {positions:new Float32Array(points),indices};
}

/** Truncated physical EDT: radii >=2h already reach the displacement cap.
 * A sheared grid uses conservative h/8 instead of an invalid axis EDT. */
export function vesselLimits(points:Float32Array,affine:Affine,mask?:Uint8Array,shape?:Shape3){
  const inv=inverseAffine(affine),h=1/Math.max(...inv.slice(0,3).map(r=>Math.hypot(...r.slice(0,3))));
  const limits=new Float64Array(points.length/3).fill(h/8);if(!mask||!shape)return limits;
  const columns=[0,1,2].map(a=>affine.slice(0,3).map(r=>r[a])),spacing=columns.map(c=>Math.hypot(...c));
  if(columns.some((c,a)=>columns.some((d,b)=>a!==b&&Math.abs(c.reduce((s,v,k)=>s+v*d[k],0))>1e-5*spacing[a]*spacing[b])))return limits;
  const offsets:Array<[number,number,number,number]>=[];
  for(let x=-2;x<=2;x++)for(let y=-2;y<=2;y++)for(let z=-2;z<=2;z++){
    const d=Math.hypot(x*spacing[0],y*spacing[1],z*spacing[2]);if(d<=2*h+1e-8)offsets.push([x,y,z,d]);}
  offsets.sort((a,b)=>a[3]-b[3]);const cache=new Map<string,number>();
  for(let v=0;v<limits.length;v++){
    const x=Math.floor(points[v*3]+.5),y=Math.floor(points[v*3+1]+.5),z=Math.floor(points[v*3+2]+.5),key=`${x},${y},${z}`;let r=cache.get(key);
    if(r===undefined){r=2*h;for(const [dx,dy,dz,d]of offsets){const a=x+dx,b=y+dy,c=z+dz;
      if(a<0||b<0||c<0||a>=shape[0]||b>=shape[1]||c>=shape[2]||!mask[(a*shape[1]+b)*shape[2]+c]){r=d;break;}}cache.set(key,r);}
    limits[v]=Math.min(.5*h,.25*Math.max(r,.5*h));
  }
  return limits;
}

export function fairIndexedSurface(points:Float32Array,indices:Uint32Array,affine:Affine,kind?:string,limits?:Float64Array,audit?:SurfaceIntersectionAudit,filter?:SurfacePolynomialFilter){
  if(points.length%3||indices.length%3||!points.every(Number.isFinite)||indices.some(v=>v>=points.length/3))throw Error('Invalid indexed display surface');
  const inv=inverseAffine(affine),h=1/Math.max(...inv.slice(0,3).map(r=>Math.hypot(...r.slice(0,3)))),profile=surfaceKind(kind);
  if(profile==='context'&&indices.length/3>SURFACE_POLICY.maxContextTriangles)return {positions:points.slice(),diagnostics:{
    policy:SURFACE_POLICY.version,kind:profile,maxDisplacementMm:0,restoredComponents:0,intersectionCheck:'source-restored',
    fallbackReason:'high-complexity context; original surface retained'} as SurfaceDiagnostics};
  const policy=SURFACE_POLICY.profiles[profile],cap=Math.min(h*policy.voxelAltitudes,'maximumMm'in policy?policy.maximumMm:Infinity);
  const n=points.length/3,original=new Float64Array(points.length),parents=Uint32Array.from({length:n},(_,i)=>i);
  for(let i=0;i<points.length;i+=3)for(let a=0;a<3;a++)original[i+a]=affine[a][0]*points[i]+affine[a][1]*points[i+1]+affine[a][2]*points[i+2];
  const find=(v:number):number=>{let r=v;while(parents[r]!==r)r=parents[r];while(parents[v]!==v){const p=parents[v];parents[v]=r;v=p;}return r;};
  const offsets=new Uint32Array(n+1),pinned=new Uint8Array(n),edges=new Map<number,number>();for(const id of indices)offsets[id+1]+=2;
  for(let i=1;i<=n;i++)offsets[i]+=offsets[i-1];const neighbors=new Uint32Array(offsets[n]),cursor=offsets.slice();
  for(let f=0;f<indices.length;f+=3)for(let c=0;c<3;c++){
    const a=indices[f+c],b=indices[f+(c+1)%3],d=indices[f+(c+2)%3];neighbors[cursor[a]++]=b;neighbors[cursor[a]++]=d;parents[find(a)]=find(b);
    const key=Math.min(a,b)*n+Math.max(a,b);edges.set(key,(edges.get(key)??0)+1);
  }
  for(const [edge,count]of edges)if(count!==2){pinned[Math.floor(edge/n)]=1;pinned[edge%n]=1;}
  const coefficients=sincCoefficients(profile);
  const cached=filter?.compute(original,offsets,neighbors,pinned,coefficients);
  const filtered=cached??Float64Array.from(original,v=>coefficients[0]*v);
  if(!cached){
    let previous=original.slice(),current=new Float64Array(points.length),next=new Float64Array(points.length);
    for(let j=1;j<coefficients.length;j++){
      for(let v=0;v<n;v++)for(let a=0;a<3;a++){
        const i=v*3+a,degree=offsets[v+1]-offsets[v];let sum=0;for(let e=offsets[v];e<offsets[v+1];e++)sum+=previous[neighbors[e]*3+a];
        next[i]=pinned[v]||!degree?original[i]:j===1?sum/degree:2*sum/degree-current[i];filtered[i]+=coefficients[j]*next[i];
      }[current,previous,next]=[previous,next,current];
    }
  }
  filter?.capture(original,filtered);
  for(let v=0;v<n;v++){
    const i=v*3,d=Math.hypot(filtered[i]-original[i],filtered[i+1]-original[i+1],filtered[i+2]-original[i+2]);
    const limit=profile==='vessel'?(limits?.[v]??h/8):cap,scale=Math.min(1,limit*.999/Math.max(d,1e-30));for(let a=0;a<3;a++)filtered[i+a]=original[i+a]+scale*(filtered[i+a]-original[i+a]);
  }
  const output=new Float32Array(points.length),physical=new Float64Array(points.length),restored=new Set<number>();
  let geometryRevision=0,auditedRevision=-1,auditedContacts=new Set<number>();
  // Only quantize mutates physical. Face/contact inspection leaves it intact,
  // so a second audit of the same revision has precisely the same result.
  const contacts=()=>{
    if(auditedRevision!==geometryRevision){auditedContacts=audit?audit.run(original,physical,indices):intersectionVertices(original,physical,indices);auditedRevision=geometryRevision;}
    return auditedContacts;
  };
  function quantize(){for(let i=0;i<points.length;i+=3){
    const dx=filtered[i]-original[i],dy=filtered[i+1]-original[i+1],dz=filtered[i+2]-original[i+2];
    // Match each old reducer's initial zero and left-to-right additions, and
    // retain the Float32 assignment before physical-coordinate conversion.
    output[i]=points[i]+(((0+inv[0][0]*dx)+inv[0][1]*dy)+inv[0][2]*dz);
    output[i+1]=points[i+1]+(((0+inv[1][0]*dx)+inv[1][1]*dy)+inv[1][2]*dz);
    output[i+2]=points[i+2]+(((0+inv[2][0]*dx)+inv[2][1]*dy)+inv[2][2]*dz);
    physical[i]=((0+affine[0][0]*output[i])+affine[0][1]*output[i+1])+affine[0][2]*output[i+2];
    physical[i+1]=((0+affine[1][0]*output[i])+affine[1][1]*output[i+1])+affine[1][2]*output[i+2];
    physical[i+2]=((0+affine[2][0]*output[i])+affine[2][1]*output[i+1])+affine[2][2]*output[i+2];
    const limit=profile==='vessel'?(limits?.[i/3]??h/8):cap;
    if(Math.hypot(physical[i]-original[i],physical[i+1]-original[i+1],physical[i+2]-original[i+2])>limit){
      output.set(points.subarray(i,i+3),i);physical.set(original.subarray(i,i+3),i);filtered.set(original.subarray(i,i+3),i);
    }
  }geometryRevision++;}
  const faceNormals=new Float64Array(indices.length),faceAreaSquared=new Float64Array(indices.length/3);
  for(let f=0;f<indices.length;f+=3){const a=indices[f]*3,b=indices[f+1]*3,c=indices[f+2]*3;
    const ux=original[b]-original[a],uy=original[b+1]-original[a+1],uz=original[b+2]-original[a+2];
    const vx=original[c]-original[a],vy=original[c+1]-original[a+1],vz=original[c+2]-original[a+2];
    const nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx;
    faceNormals[f]=nx;faceNormals[f+1]=ny;faceNormals[f+2]=nz;faceAreaSquared[f/3]=nx*nx+ny*ny+nz*nz;
  }
  quantize();
  const backedOff=new Set<number>();
  const invalidVertices=()=>{
    const bad=new Set<number>(),minimum=SURFACE_POLICY.minimumFaceAreaRatio**2;
    for(let f=0;f<indices.length;f+=3){const a=indices[f]*3,b=indices[f+1]*3,c=indices[f+2]*3;
      const ux=physical[b]-physical[a],uy=physical[b+1]-physical[a+1],uz=physical[b+2]-physical[a+2];
      const vx=physical[c]-physical[a],vy=physical[c+1]-physical[a+1],vz=physical[c+2]-physical[a+2];
      const nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx;
      if(nx*faceNormals[f]+ny*faceNormals[f+1]+nz*faceNormals[f+2]<=0||nx*nx+ny*ny+nz*nz<minimum*faceAreaSquared[f/3])
        for(let corner=0;corner<3;corner++)bad.add(indices[f+corner]);
    }return bad;
  };
  const repairFaces=()=>{for(let attempt=0;attempt<16;attempt++){
    const bad=invalidVertices();if(!bad.size)return;
    for(const v of bad){backedOff.add(v);for(let a=0;a<3;a++)filtered[v*3+a]=original[v*3+a]+(attempt<14?.5:0)*(filtered[v*3+a]-original[v*3+a]);}quantize();
  }};
  repairFaces();
  for(let attempt=0;attempt<4;attempt++){
    const bad=contacts();if(!bad.size)break;
    for(const v of bad){backedOff.add(v);filtered.set(original.subarray(v*3,v*3+3),v*3);}quantize();repairFaces();
  }
  for(const v of invalidVertices())restored.add(find(v));
  for(const v of contacts())restored.add(find(v));
  // Recheck after component rollback: it must not introduce a contact with a
  // retained component. The restored set grows monotonically to termination.
  while(restored.size){for(let v=0;v<n;v++)if(restored.has(find(v)))filtered.set(original.subarray(v*3,v*3+3),v*3);quantize();
    let added=false;for(const v of contacts()){const c=find(v);if(!restored.has(c)){restored.add(c);added=true;}}if(!added)break;}
  let maximum=0;for(let i=0;i<points.length;i+=3)maximum=Math.max(maximum,Math.hypot(physical[i]-original[i],physical[i+1]-original[i+1],physical[i+2]-original[i+2]));
  const diagnostics:SurfaceDiagnostics={policy:SURFACE_POLICY.version,kind:profile,maxDisplacementMm:maximum,restoredComponents:restored.size,backedOffVertices:backedOff.size,intersectionCheck:restored.size?'source-restored':'passed'};
  audit?.complete(original,physical,indices,contacts());
  filter?.complete();
  return {positions:output,diagnostics};
}

/** Compatibility with the original unindexed helper. */
export function smoothSurface(positions:Float32Array,affine:Affine,kind?:string):Float32Array {
  const indexed=indexSurface(positions),result=fairIndexedSurface(indexed.positions,indexed.indices,affine,kind),out=new Float32Array(positions.length);
  for(let i=0;i<indexed.indices.length;i++)out.set(result.positions.subarray(indexed.indices[i]*3,indexed.indices[i]*3+3),i*3);return out;
}
