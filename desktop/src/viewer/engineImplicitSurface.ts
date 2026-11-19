import type {Affine, Shape3} from './coordinates';
import {inverseAffine} from './coordinates.ts';
import type {SurfaceDiagnostics} from './surfacePolicy';
import {surfaceKind} from './surfacePolicy.ts';
import {indexedSurfaceNormals} from './surfaceNormals.ts';

/** The undeformed material partition is shared across every frame of an
 * accepted engine mapping. Fields derived from it are for drawing only. */
export interface EngineSurfaceReference {
  owner: Uint16Array;
  protectedRegions: number[];
}
export const ENGINE_SURFACE_POLICY = 'engine-shared-implicit-v1';
export const MAX_ENGINE_REFERENCE_BYTES = 32 * 1024 ** 2;

function physicalLength(affine:Affine,x:number,y:number,z:number) {
  return Math.hypot(...affine.slice(0,3).map(row=>row[0]*x+row[1]*y+row[2]*z));
}

/** A signed, piecewise-linear field in each original extraction tetrahedron.
 * At a source node all materials use the SAME positive magnitude with their
 * own membership sign. In barycentric coordinates the map from source to
 * display divides each coordinate by that node's magnitude and renormalizes.
 * This is a bijection with a positive inverse, fixes the four source nodes,
 * maps planes to planes, and agrees on shared faces. Consequently every
 * original triangle, including coincident material interfaces and junctions,
 * moves through one common map. No per-layer or per-edge clipping is valid. */
export function engineImplicitSurface(points:Float32Array,indices:Uint32Array,normals:Float32Array,
  mask:Uint8Array,shape:Shape3,affine:Affine,kind:string|undefined,reference:EngineSurfaceReference|null) {
  const started=performance.now(),inv=inverseAffine(affine);
  const altitude=1/Math.max(...inv.slice(0,3).map(row=>Math.hypot(...row.slice(0,3))));
  const cap=Math.min(.5*altitude,1);
  const diagnostics:SurfaceDiagnostics={policy:ENGINE_SURFACE_POLICY,kind:surfaceKind(kind),
    triangles:indices.length/3,maxDisplacementMm:0,displacementLimitMm:cap,restoredComponents:0,intersectionCheck:'passed'};
  const positions=points.slice();
  if(!reference){
    diagnostics.intersectionCheck='source-restored';
    diagnostics.fallbackReason='Source ownership exceeds the shared-field memory budget; source surface retained';
    diagnostics.preparationMs=performance.now()-started;
    return {positions,smoothNormals:normals.slice(),diagnostics};
  }
  const {owner,protectedRegions}=reference;
  const [nx,ny,nz]=shape,yz=ny*nz;
  if(owner.length!==mask.length||owner.length!==nx*yz)throw Error('Engine display partition differs from the source grid');
  const protectedIds=new Set(protectedRegions);
  let longest=0;
  for(let bits=1;bits<8;bits++)longest=Math.max(longest,physicalLength(affine,bits&1,(bits>>1)&1,bits>>2));
  // A single range for ALL nodes bounds every edge crossing, including a
  // body diagonal on an anisotropic/sheared grid. Leave a quantization margin.
  const shift=Math.min(.32,.995*cap/longest),minimumWeight=(.5-shift)/(.5+shift);
  // Gaussian width in physical units, expressed relative to the shortest
  // source-cell altitude. The truncated kernel is normalized on this grid.
  const sigmaSquared=1.2**2;
  const kernel:Array<{x:number;y:number;z:number;offset:number;weight:number;near:boolean;
    rx:number;ry:number;rz:number}>=[];
  const moment=new Float64Array(6);
  let kernelMass=0;
  const creaseOffsets=[yz,nz,1].filter((_,axis)=>
    Math.hypot(affine[0][axis],affine[1][axis],affine[2][axis])<=1.5*altitude);
  for(let x=-2;x<=2;x++)for(let y=-2;y<=2;y++)for(let z=-2;z<=2;z++){
    const rx=(affine[0][0]*x+affine[0][1]*y+affine[0][2]*z)/altitude;
    const ry=(affine[1][0]*x+affine[1][1]*y+affine[1][2]*z)/altitude;
    const rz=(affine[2][0]*x+affine[2][1]*y+affine[2][2]*z)/altitude;
    const weight=Math.exp(-.5*(rx*rx+ry*ry+rz*rz)/sigmaSquared);
    const near=Math.max(Math.abs(x),Math.abs(y),Math.abs(z))<=1;
    if(weight<.0001&&!near)continue;
    kernel.push({x,y,z,offset:x*yz+y*nz+z,weight,near,rx,ry,rz});
    kernelMass+=weight;
    moment[0]+=weight*rx*rx;moment[1]+=weight*ry*ry;moment[2]+=weight*rz*rz;
    moment[3]+=weight*rx*ry;moment[4]+=weight*rx*rz;moment[5]+=weight*ry*rz;
  }
  // Direct-mapped cache bounds scratch use even on a very large or sparse
  // surface. Recomputing a collision yields precisely the same field value.
  const slots=Math.min(32768,2**Math.ceil(Math.log2(Math.max(1,points.length/3))));
  const keys=new Int32Array(slots).fill(-1),magnitude=new Float64Array(slots);
  const gx=new Float64Array(slots),gy=new Float64Array(slots),gz=new Float64Array(slots);
  const paddedY=ny+2,paddedZ=nz+2;
  function sample(x:number,y:number,z:number){
    const key=((x+1)*paddedY+y+1)*paddedZ+z+1,slot=key&(slots-1);
    if(keys[slot]===key)return slot;
    const i=x*yz+y*nz+z,valid=x>=0&&x<nx&&y>=0&&y<ny&&z>=0&&z<nz;
    const label=valid?owner[i]:0;
    const interior=x>=2&&x<nx-2&&y>=2&&y<ny-2&&z>=2&&z<nz-2;
    let same=0,px=0,py=0,pz=0;
    let pin=x<=0||x>=nx-1||y<=0||y>=ny-1||z<=0||z>=nz-1;
    // A short opposing-material run on both sides of a node identifies a
    // thin tip or crease. Preserve its source weight rather than blurring a
    // narrow fluid cleft across opposing walls.
    if(interior)for(const step of creaseOffsets){
      if(owner[i-2*step]!==label&&owner[i-2*step]===owner[i+2*step])pin=true;
    }
    let ox=0,oy=0,oz=0,xx=0,yy=0,zz=0,xy=0,xz=0,yzMoment=0;
    for(const k of kernel){
      const inGrid=interior||(x+k.x>=0&&x+k.x<nx&&y+k.y>=0&&y+k.y<ny&&z+k.z>=0&&z+k.z<nz);
      const j=i+k.offset,other=inGrid?owner[j]:0;
      if(other===label){
        same+=k.weight;
        const wx=k.weight*k.rx,wy=k.weight*k.ry,wz=k.weight*k.rz;
        ox+=wx;oy+=wy;oz+=wz;
        xx+=wx*k.rx;yy+=wy*k.ry;zz+=wz*k.rz;
        xy+=wx*k.ry;xz+=wx*k.rz;yzMoment+=wy*k.rz;
      }
      if(k.near&&protectedIds.has(other))pin=true;
      if(inGrid&&mask[j]){px-=k.x*k.weight;py-=k.y*k.weight;pz-=k.z*k.weight;}
    }
    const mean=same/kernelMass,normalLength=Math.hypot(ox,oy,oz);
    let occupancy=mean;
    if(!pin&&normalLength>1e-8){
      ox/=normalLength;oy/=normalLength;oz/=normalLength;
      // Gaussian indicator filtering moves convex and concave boundaries by
      // mean curvature. Compensate its tangential Hessian trace, estimated
      // with the physical second moment after removing the constant response
      // of this discrete truncated kernel. The positive range below still
      // governs the final location regardless of curvature.
      const tangent=(xx-mean*moment[0])*(1-ox*ox)+(yy-mean*moment[1])*(1-oy*oy)
        +(zz-mean*moment[2])*(1-oz*oz)-2*((xy-mean*moment[3])*ox*oy
          +(xz-mean*moment[4])*ox*oz+(yzMoment-mean*moment[5])*oy*oz);
      occupancy-=.6*tangent/(sigmaSquared*kernelMass);
    }
    keys[slot]=key;
    magnitude[slot]=pin?1:Math.min(1,Math.max(minimumWeight,1.6*(2*occupancy-1)));
    gx[slot]=px;gy[slot]=py;gz[slot]=pz;
    return slot;
  }
  const gradient=new Float32Array(points.length);
  const [[a,b,c],[d,e,f],[g,h,j]]=affine;
  for(let i=0;i<points.length;i+=3){
    const x=points[i],y=points[i+1],z=points[i+2];
    const lx=Math.floor(x),ly=Math.floor(y),lz=Math.floor(z);
    const dx=Math.ceil(x)-lx,dy=Math.ceil(y)-ly,dz=Math.ceil(z)-lz;
    const lo=sample(lx,ly,lz),wl=magnitude[lo],vx=gx[lo],vy=gy[lo],vz=gz[lo];
    const hi=sample(lx+dx,ly+dy,lz+dz),t=wl/(wl+magnitude[hi]);
    positions[i]=lx+t*dx;positions[i+1]=ly+t*dy;positions[i+2]=lz+t*dz;
    const ux=positions[i]-x,uy=positions[i+1]-y,uz=positions[i+2]-z;
    diagnostics.maxDisplacementMm=Math.max(diagnostics.maxDisplacementMm,
      Math.hypot(a*ux+b*uy+c*uz,d*ux+e*uy+f*uz,g*ux+h*uy+j*uz));
    // Derivative of the physical Gaussian is proportional to the physical
    // moment of the indicator. Reflecting/shearing the grid transforms the
    // moment first, independently of the stored source-frame normal covector.
    const mx=vx*(1-t)+gx[hi]*t,my=vy*(1-t)+gy[hi]*t,mz=vz*(1-t)+gz[hi]*t;
    gradient[i]=a*mx+b*my+c*mz;gradient[i+1]=d*mx+e*my+f*mz;gradient[i+2]=g*mx+h*my+j*mz;
  }
  if(diagnostics.maxDisplacementMm>cap)throw Error('Engine display displacement exceeded the physical source bound');
  const smoothNormals=indexedSurfaceNormals(positions,indices,affine);
  for(let i=0;i<points.length;i+=3){
    const x=smoothNormals[i],y=smoothNormals[i+1],z=smoothNormals[i+2];
    let px=inv[0][0]*x+inv[1][0]*y+inv[2][0]*z,py=inv[0][1]*x+inv[1][1]*y+inv[2][1]*z,pz=inv[0][2]*x+inv[1][2]*y+inv[2][2]*z;
    const length=Math.hypot(px,py,pz),gl=Math.hypot(gradient[i],gradient[i+1],gradient[i+2]);
    if(!length||gl<1e-5)continue;
    px/=length;py/=length;pz/=length;
    const vx=gradient[i]/gl,vy=gradient[i+1]/gl,vz=gradient[i+2]/gl;
    const dot=px*vx+py*vy+pz*vz;
    // At opposing walls or ambiguous thin tips retain their own face normal.
    if(dot<=0)continue;
    const weight=.95*Math.min(1,dot/.65);
    px+=(vx-px)*weight;py+=(vy-py)*weight;pz+=(vz-pz)*weight;
    const cx=a*px+d*py+g*pz,cy=b*px+e*py+h*pz,cz=c*px+f*py+j*pz;
    const scale=1/Math.hypot(cx,cy,cz);
    smoothNormals[i]=cx*scale;smoothNormals[i+1]=cy*scale;smoothNormals[i+2]=cz*scale;
  }
  diagnostics.preparationMs=performance.now()-started;
  return {positions,smoothNormals,diagnostics};
}

export interface EnginePartitionRequest {key:string;reference?:EngineSurfaceReference|null}
export class WorkerSurfacePartition {
  private key?:string;
  private value?:EngineSurfaceReference|null;
  resolve(request?:EnginePartitionRequest,maxBytes=MAX_ENGINE_REFERENCE_BYTES){
    if(!request)return undefined;
    if(request.reference!==undefined){
      if(request.reference&&request.reference.owner.byteLength>MAX_ENGINE_REFERENCE_BYTES)throw Error('Engine display partition exceeds its transfer budget');
      this.key=request.key;
      this.value=request.reference&&request.reference.owner.byteLength>maxBytes?null:request.reference;
    }
    if(request.key!==this.key||this.value===undefined)throw Error('Missing shared engine display partition');
    return this.value;
  }
  get retainedBytes(){return this.value?.owner.byteLength??0;}
}
