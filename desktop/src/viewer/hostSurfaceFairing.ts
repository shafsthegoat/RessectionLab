import type {Affine, Shape3} from './coordinates';
import {indexedSurfaceNormals} from './surfaceNormals.ts';

/** Display fairing of a host-head operative surface (a real head on a 1 mm
 * grid). The shared implicit field keeps every crossing on its source edge,
 * so long gentle slopes of the voxel boundary keep 1 mm terraces, and
 * protected vessel and membrane labels keep raw cell faces. Here the free
 * surface of each layer relaxes in 3D within a physical bound from its source
 * vertex (less than half a cell). Faces shared with another retained material
 * keep their shared positions, so tissue interfaces stay coincident; crop
 * faces stay exactly as prepared.
 * Picking, contours and the engine's cells use the source geometry. */
export const HOST_FAIRING_POLICY = 'engine-host-display-fairing-v1';

export interface HostFairingResult {positions:Float32Array;normals:Float32Array;maximumMm:number;limitMm:number;movedVertices:number;backedOffVertices:number}

function neighbours(indices:Uint32Array,count:number){
  const start=new Uint32Array(count+1);
  for(let i=0;i<indices.length;i++)start[indices[i]+1]+=2;
  for(let v=0;v<count;v++)start[v+1]+=start[v];
  const cursor=start.slice(0,count),list=new Uint32Array(start[count]);
  for(let f=0;f<indices.length;f+=3)for(let c=0;c<3;c++){
    const v=indices[f+c];list[cursor[v]++]=indices[f+(c+1)%3];list[cursor[v]++]=indices[f+(c+2)%3];
  }
  // Each edge of a closed surface is listed once from each of its two faces; keep one.
  const unique=new Uint32Array(count+1),compact=new Uint32Array(list.length);let n=0;
  for(let v=0;v<count;v++){
    unique[v]=n;
    for(let e=start[v];e<start[v+1];e++){
      const u=list[e];let seen=false;
      for(let q=unique[v];q<n;q++)if(compact[q]===u){seen=true;break;}
      if(!seen)compact[n++]=u;
    }
  }
  unique[count]=n;
  return {start:unique,list:compact.subarray(0,n)};
}

/** Display kinds of the brain parenchyma group and of the head-wall group. */
export const HOST_BRAIN_KINDS=new Set(['grayMatter','whiteMatter','tract','tumor','brain','gray-matter','white-matter','cerebral-cortex','cerebellar-cortex']);
export const HOST_HEAD_KINDS=new Set(['scalp','galea','tissue','extracranial','skull','diploe']);

/** Which display vertices may move. A vertex of the source extraction lies on
 * one lattice edge between an inside node (this layer) and an outside node.
 * It stays where the outside node is another material of the same group
 * (brain parenchyma or head wall): those interfaces are shared between two
 * layers and keep one position. It is free against free space, a protected
 * overlay (vessel, membrane) or the other group (brain against bone).
 *
 * Where such a shared interface reaches the free surface (a white-matter
 * island in a thin cortex), its vertex lies half a cell below the surface and
 * shades as a groove tracing the island. It is lifted toward the free side by
 * a fixed physical amount computed only from the owner grid, so both layers
 * move it identically and stay joined. */
export function hostFairingMobility(raw:Float32Array,shown:Float32Array,owner:Uint16Array,mask:Uint8Array,
  shape:Shape3,overlay:ReadonlySet<number>,groups?:ReadonlyArray<ReadonlyArray<number>>,affine?:Affine,liftMm=.45):{mobile:Uint8Array;start:Float32Array} {
  const [nx,ny,nz]=shape,yz=ny*nz,count=raw.length/3,mobile=new Uint8Array(shown.length/3);
  const group=new Map<number,number>();
  groups?.forEach((ids,g)=>ids.forEach(id=>group.set(id,g+1)));
  let start=shown;
  for(let v=0;v<count;v++){
    const i=3*v;
    const x=raw[i],y=raw[i+1],z=raw[i+2];
    const lx=Math.floor(x),ly=Math.floor(y),lz=Math.floor(z),hx=Math.ceil(x),hy=Math.ceil(y),hz=Math.ceil(z);
    // Crop faces and their first cell stay flat: they are the section of the working grid.
    if(lx<1||ly<1||lz<1||hx>nx-2||hy>ny-2||hz>nz-2)continue;
    const lo=lx*yz+ly*nz+lz,hi=hx*yz+hy*nz+hz;
    const inside=mask[lo]?lo:hi,outside=inside===lo?hi:lo;
    if(mask[outside])continue;
    const self=owner[inside],other=owner[outside];
    // A removed or displaced cell of the same material leaves a free wall.
    if(other===0||other===self||overlay.has(other)||overlay.has(self)){mobile[v]=1;continue;}
    const a=group.get(self),b=group.get(other);
    if(a&&b&&a!==b){mobile[v]=1;continue;}
    if(!a||a!==b||!affine)continue;
    let dx=0,dy=0,dz=0;
    for(let p=lx-1;p<=hx+1;p++)for(let q=ly-1;q<=hy+1;q++)for(let r=lz-1;r<=hz+1;r++){
      const o=owner[p*yz+q*nz+r];
      if(o&&!overlay.has(o)&&group.get(o)===a)continue;
      dx+=p-x;dy+=q-y;dz+=r-z;
    }
    const length=Math.hypot(affine[0][0]*dx+affine[0][1]*dy+affine[0][2]*dz,affine[1][0]*dx+affine[1][1]*dy+affine[1][2]*dz,affine[2][0]*dx+affine[2][1]*dy+affine[2][2]*dz);
    if(length<1e-6)continue;
    if(start===shown)start=shown.slice();
    const scale=liftMm/length;
    start[i]=x+dx*scale;start[i+1]=y+dy*scale;start[i+2]=z+dz*scale;
  }
  // Vertices an accepted cavity join inserted lie on that free wall.
  mobile.fill(1,count);
  return {mobile,start};
}

/** Bounded Laplacian relaxation of the mobile vertices. Each display vertex
 * stays within `limitMm` (physical) of its source vertex; a weak restoring
 * term toward the prepared display keeps gyral shape. Vessel overlays use a
 * volume-preserving λ|μ pair so thin branches keep their calibre. */
export function fairHostSurface(raw:Float32Array,shown:Float32Array,indices:Uint32Array,mobile:Uint8Array,affine:Affine,
  options:{limitMm:number;vessel?:boolean;iterations?:number;restore?:number}):HostFairingResult|undefined {
  const count=shown.length/3,rawCount=raw.length/3;
  let active=0;for(let v=0;v<count;v++)if(mobile[v])active++;
  if(!active||!indices.length)return;
  // Source vertices are bounded from their source position; vertices a cavity
  // join inserted (no source vertex) from their prepared position, by half as much.
  const anchor=shown.slice();anchor.set(raw);
  const [[a,b,c],[d,e,f],[g,h,k]]=affine;
  const {start,list}=neighbours(indices,count);
  const limit=options.limitMm,limitSquared=limit*limit;
  let current=shown.slice(),next=shown.slice();
  const iterations=options.iterations??(options.vessel?40:22);
  const ids=new Uint32Array(active);{let n=0;for(let v=0;v<count;v++)if(mobile[v])ids[n++]=v;}
  const insertedSquared=limitSquared/4;
  for(let pass=0;pass<iterations;pass++){
    const step=options.vessel?(pass%2?-.53:.5):.55,restore=options.vessel?0:options.restore??.015;
    for(const v of ids){
      const i=3*v,s=start[v],t=start[v+1];
      if(t===s)continue;
      let x=0,y=0,z=0;
      for(let n=s;n<t;n++){const j=3*list[n];x+=current[j];y+=current[j+1];z+=current[j+2];}
      const inv=1/(t-s);
      let px=current[i]+step*(x*inv-current[i])+restore*(shown[i]-current[i]);
      let py=current[i+1]+step*(y*inv-current[i+1])+restore*(shown[i+1]-current[i+1]);
      let pz=current[i+2]+step*(z*inv-current[i+2])+restore*(shown[i+2]-current[i+2]);
      const ux=px-anchor[i],uy=py-anchor[i+1],uz=pz-anchor[i+2],bound=v<rawCount?limitSquared:insertedSquared;
      const rx=a*ux+b*uy+c*uz,ry=d*ux+e*uy+f*uz,rz=g*ux+h*uy+k*uz,squared=rx*rx+ry*ry+rz*rz;
      if(squared>bound){
        const scale=Math.sqrt(bound/squared);
        px=anchor[i]+ux*scale;py=anchor[i+1]+uy*scale;pz=anchor[i+2]+uz*scale;
      }
      next[i]=px;next[i+1]=py;next[i+2]=pz;
    }
    // Only mobile vertices change, and both buffers start from the same display.
    [current,next]=[next,current];
  }
  // A relaxed face must not fold over its neighbourhood or turn back against
  // its source orientation. A riser that lies down into a ramp may shrink to
  // almost nothing; that is the intended result. Back off folded corners toward
  // the prepared display, finally restoring them.
  const physicalNormal=(p:Float32Array,face:number,out:Float64Array)=>{
    const i=3*indices[face],j=3*indices[face+1],l=3*indices[face+2];
    const jx=p[j]-p[i],jy=p[j+1]-p[i+1],jz=p[j+2]-p[i+2],kx=p[l]-p[i],ky=p[l+1]-p[i+1],kz=p[l+2]-p[i+2];
    const ax=a*jx+b*jy+c*jz,ay=d*jx+e*jy+f*jz,az=g*jx+h*jy+k*jz,bx=a*kx+b*ky+c*kz,by=d*kx+e*ky+f*kz,bz=g*kx+h*ky+k*kz;
    out[0]=ay*bz-az*by;out[1]=az*bx-ax*bz;out[2]=ax*by-ay*bx;
  };
  // Only faces with a mobile corner can change; the rest add a fixed share to the vertex normals.
  let activeFaces=0;
  for(let face=0;face<indices.length;face+=3)if(mobile[indices[face]]||mobile[indices[face+1]]||mobile[indices[face+2]])activeFaces++;
  const faces=new Uint32Array(activeFaces);
  {let n=0;for(let face=0;face<indices.length;face+=3)if(mobile[indices[face]]||mobile[indices[face+1]]||mobile[indices[face+2]])faces[n++]=face;}
  const before=new Float64Array(3*activeFaces),faceNow=new Float64Array(3*activeFaces);
  const fixedSum=new Float64Array(shown.length),vertexSum=new Float64Array(shown.length);
  const scratch=new Float64Array(3);
  for(let face=0,n=0;face<indices.length;face+=3){
    physicalNormal(shown,face,scratch);
    if(n<activeFaces&&faces[n]===face){before.set(scratch,3*n);n++;continue;}
    for(let corner=0;corner<3;corner++){const v=3*indices[face+corner];fixedSum[v]+=scratch[0];fixedSum[v+1]+=scratch[1];fixedSum[v+2]+=scratch[2];}
  }
  const badMark=new Uint8Array(count);let backedOff=0;
  for(let attempt=0;attempt<4;attempt++){
    vertexSum.set(fixedSum);
    for(let n=0;n<activeFaces;n++){
      const face=faces[n];physicalNormal(current,face,scratch);faceNow.set(scratch,3*n);
      for(let corner=0;corner<3;corner++){const v=3*indices[face+corner];vertexSum[v]+=scratch[0];vertexSum[v+1]+=scratch[1];vertexSum[v+2]+=scratch[2];}
    }
    const bad:number[]=[];
    for(let n=0;n<activeFaces;n++){
      const face=faces[n],m=3*n,nx=faceNow[m],ny=faceNow[m+1],nz=faceNow[m+2],now=Math.hypot(nx,ny,nz);
      if(now<1e-12)continue;
      let sx=0,sy=0,sz=0;
      for(let corner=0;corner<3;corner++){const v=3*indices[face+corner];sx+=vertexSum[v];sy+=vertexSum[v+1];sz+=vertexSum[v+2];}
      const folded=nx*sx+ny*sy+nz*sz<0;
      const reversed=nx*before[m]+ny*before[m+1]+nz*before[m+2]<-.5*now*Math.hypot(before[m],before[m+1],before[m+2]);
      if(folded||reversed)for(let corner=0;corner<3;corner++){const v=indices[face+corner];if(badMark[v]!==attempt+1){if(!badMark[v])backedOff++;badMark[v]=attempt+1;bad.push(v);}}
    }
    if(!bad.length)break;
    const keep=attempt<3?.4:0;
    for(const v of bad)for(let axis=0;axis<3;axis++){const i=3*v+axis;current[i]=shown[i]+keep*(current[i]-shown[i]);}
  }
  let maximum=0,moved=0;
  for(let v=0;v<rawCount;v++){
    const i=3*v;
    if(current[i]!==shown[i]||current[i+1]!==shown[i+1]||current[i+2]!==shown[i+2])moved++;
    const ux=current[i]-raw[i],uy=current[i+1]-raw[i+1],uz=current[i+2]-raw[i+2];
    maximum=Math.max(maximum,Math.hypot(a*ux+b*uy+c*uz,d*ux+e*uy+f*uz,g*ux+h*uy+k*uz));
  }
  return {positions:current,normals:smoothHostNormals(indexedSurfaceNormals(current,indices,affine),start,list,mobile),
    maximumMm:maximum,limitMm:limit,movedVertices:moved,backedOffVertices:backedOff};
}

/** Lighting normals of the relaxed free surface: the residual sub-cell zigzag of
 * the tetrahedral extraction is shaded as the surface it approximates. Only
 * neighbours within 50 degrees average in, so folds and cut edges stay crisp.
 * Fixed vertices keep their own normal. */
function smoothHostNormals(normals:Float32Array,start:Uint32Array,list:Uint32Array,mobile:Uint8Array,iterations=8){
  const limit=Math.cos(50*Math.PI/180),count=normals.length/3;
  let current:Float32Array=normals,next:Float32Array=normals.slice();
  const ids:number[]=[];for(let v=0;v<count;v++)if(mobile[v])ids.push(v);
  for(let pass=0;pass<iterations;pass++){
    for(const v of ids){
      const i=3*v;
      let x=current[i],y=current[i+1],z=current[i+2];
      for(let n=start[v];n<start[v+1];n++){
        const j=3*list[n],c=current[i]*current[j]+current[i+1]*current[j+1]+current[i+2]*current[j+2];
        if(c<limit)continue;
        x+=current[j];y+=current[j+1];z+=current[j+2];
      }
      const l=Math.hypot(x,y,z)||1;next[i]=x/l;next[i+1]=y/l;next[i+2]=z/l;
    }
    [current,next]=[next,current];
  }
  return current;
}

const closedOwners=new WeakMap<Uint16Array,{key:string;owner:Uint16Array;filled:number}>();
/** Display ownership of a host head with its brain pinholes closed: a free
 * cell walled by brain on four or more faces, and an enclosed free pocket
 * of at most 12 cells walled only by brain, take a neighbouring brain layer.
 * At 1 mm such cells are segmentation gaps in the cortical ribbon; drawn open,
 * a relaxed thin sheet over them shows a dark crack. The engine's cells and
 * the picking surface keep the source ownership. */
export function hostDisplayOwner(owner:Uint16Array,shape:Shape3,brainIds:readonly number[]){
  const key=brainIds.join(','),cached=closedOwners.get(owner);
  if(cached?.key===key)return cached;
  const [nx,ny,nz]=shape,yz=ny*nz,brain=new Set(brainIds),closed=owner.slice(),strides=[yz,nz,1];
  let filled=0;
  const interior=(i:number)=>{const x=Math.floor(i/yz),y=Math.floor(i/nz)%ny,z=i%nz;return x>0&&y>0&&z>0&&x<nx-1&&y<ny-1&&z<nz-1;};
  if(brain.size)for(let i=0;i<owner.length;i++){
    if(owner[i]||!interior(i))continue;
    // Mostly walled by brain (four or more of six faces): a pinhole, not a sulcus,
    // which is open along its length and depth.
    let walls=0,region=0;
    for(const s of strides)for(const n of [owner[i-s],owner[i+s]])if(brain.has(n)){walls++;region=region?Math.min(region,n):n;}
    if(walls>=4){closed[i]=region;filled++;}
  }
  if(brain.size){
    const seen=new Uint8Array(owner.length),queue=new Int32Array(64);
    for(let i=0;i<owner.length;i++){
      if(closed[i]||seen[i])continue;
      let size=0,open=false,head=0;const votes=new Map<number,number>();
      queue[size++]=i;seen[i]=1;
      while(head<size){
        const c=queue[head++];
        if(!interior(c)){open=true;}
        for(const s of strides)for(const n of [c-s,c+s]){
          if(n<0||n>=owner.length)continue;
          const o=closed[n];
          if(o){if(brain.has(o))votes.set(o,(votes.get(o)??0)+1);else open=true;continue;}
          if(seen[n])continue;
          seen[n]=1;
          if(size<queue.length)queue[size++]=n;else open=true;
        }
        if(open)break;
      }
      if(open||size>12||!votes.size){
        // Mark the rest of a large or open component without filling it.
        if(open||size>12){const rest=[i];while(rest.length){const c=rest.pop()!;for(const s of strides)for(const n of [c-s,c+s])if(n>=0&&n<owner.length&&!closed[n]&&!seen[n]){seen[n]=1;rest.push(n);}}}
        continue;
      }
      const region=[...votes].sort((a,b)=>b[1]-a[1]||a[0]-b[0])[0][0];
      for(let k=0;k<size;k++){closed[queue[k]]=region;filled++;}
    }
  }
  const result={key,owner:closed,filled};closedOwners.set(owner,result);
  return result;
}

/** This layer's display cells: its current cells plus closed pinholes given to its region. */
export function hostDisplayMask(mask:Uint8Array,owner:Uint16Array,closed:Uint16Array,shape:Shape3):Uint8Array{
  let region=0;
  for(let i=0;i<mask.length&&!region;i++)if(mask[i]&&owner[i])region=owner[i];
  if(!region)return mask;
  const [,ny,nz]=shape,strides=[ny*nz,nz,1];
  let display=mask;
  for(let i=0;i<mask.length;i++){
    if(closed[i]!==region||owner[i]===region||mask[i])continue;
    // Only where the region is still present around the cell (never across a cut).
    let support=0;for(const s of strides){if(mask[i-s])support++;if(mask[i+s])support++;}
    if(support<2)continue;
    if(display===mask)display=mask.slice();
    display[i]=1;
  }
  return display;
}
