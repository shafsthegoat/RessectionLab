import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import test from 'node:test';
import {prepareSurface} from './prepareSurface.ts';
import {maskSurfaceRegion} from './surface.ts';
import {IncrementalSurfaceEngine} from './incrementalSurface.ts';
import {ENGINE_SURFACE_POLICY} from './engineImplicitSurface.ts';
import {NATIVE_BRAIN_SURFACE_POLICY} from './operativeBrainFairing.ts';
import {intersectionVertices} from './surfaceSafety.ts';

const identity=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
const hash=a=>createHash('sha256').update(new Uint8Array(a.buffer,a.byteOffset,a.byteLength)).digest('hex');
function grid(shape,fn,ArrayType=Uint8Array){
  const [nx,ny,nz]=shape;
  return ArrayType.from({length:nx*ny*nz},(_,i)=>fn(Math.floor(i/(ny*nz)),Math.floor(i/nz)%ny,i%nz));
}
function locations(surface){
  const points=new Map();
  for(let i=0;i<surface.positions.length;i+=3){
    const raw=surface.positions.subarray(i,i+3),shown=Array.from(surface.smoothPositions.subarray(i,i+3));
    points.set(raw.join(','),shown);
  }
  return points;
}
function angleAndRadius(points,normals,center,axes){
  let degrees=0,sumSquared=0;
  for(let i=0;i<points.length;i+=3){
    const offset=center.map((v,a)=>points[i+a]-v),gradient=offset.map((v,a)=>v/axes[a]**2);
    const length=Math.hypot(...offset),scale=Math.hypot(...offset.map((v,a)=>v/axes[a]));
    sumSquared+=(length-length/scale)**2;
    const cosine=gradient.reduce((sum,v,k)=>sum+v*normals[i+k],0)/Math.hypot(...gradient);
    degrees+=Math.acos(Math.max(-1,Math.min(1,cosine)))*180/Math.PI;
  }
  return {angle:degrees/(points.length/3),radialRms:Math.sqrt(sumSquared/(points.length/3))};
}

test('all 256 binary cube cases retain the original soup bytes and winding',()=>{
  const sha=createHash('sha256');
  for(let bits=0;bits<256;bits++){
    const mask=Uint8Array.from({length:8},(_,i)=>(bits>>i)&1);
    sha.update(Buffer.from(maskSurfaceRegion(mask,[2,2,2],[0,0,0],[0,0,0]).positions.buffer));
  }
  assert.equal(sha.digest('hex'),'7c07d484a37fcb63f087fadb055ad42cc642d8616e92bdcdbb76654b8c5456fe');
});

test('analytic curved control measures raw, sinc and implicit surfaces against the same ellipsoid',()=>{
  const shape=[39,39,39],center=[18.3,19.1,17.6],axes=[15,11,9];
  const mask=grid(shape,(...p)=>p.reduce((s,v,k)=>s+((v-center[k])/axes[k])**2,0)<=1);
  const reference={owner:Uint16Array.from(mask),protectedRegions:[]},before=[hash(mask),hash(reference.owner)];
  const old=prepareSurface(mask,shape,identity,'soft_tissue'),current=prepareSurface(mask,shape,identity,'soft_tissue',true,reference);
  const raw=angleAndRadius(old.positions,old.normals,center,axes);
  const sinc=angleAndRadius(old.smoothPositions,old.smoothNormals,center,axes);
  const implicit=angleAndRadius(current.smoothPositions,current.smoothNormals,center,axes);
  assert.deepEqual([hash(mask),hash(reference.owner)],before);
  for(const key of ['positions','normals','indices'])assert.deepEqual(current[key],old[key]);
  assert(raw.angle>20&&sinc.angle<6&&implicit.angle<3.5);
  assert(implicit.angle<sinc.angle*.7);
  assert(implicit.radialRms<raw.radialRms*.55&&sinc.radialRms<implicit.radialRms);
  assert.equal(current.diagnostics.policy,ENGINE_SURFACE_POLICY);
  assert(current.diagnostics.maxDisplacementMm>0&&current.diagnostics.maxDisplacementMm<=.5);
});

test('multi-material junctions share identical crossings, original branch and membrane interfaces are pinned',()=>{
  const shape=[24,19,17],owner=grid(shape,(x,y,z)=>{
    if(Math.abs(x-13)+Math.abs(y-9)<2&&z>4&&z<11||y===9&&z===9&&x>=9&&x<17)return 3;
    if(z===6&&y>8&&y<14&&x>4&&x<12)return 4;
    if(((x-12)/10)**2+((y-9)/7)**2+((z-8)/6)**2>=1)return 0;
    return z<7+Math.sin(x*.6)*2?1:2;
  },Uint16Array);
  const before=hash(owner),reference={owner,protectedRegions:[3,4]};
  const affine=[[-.7,.14,0,70],[0,1.2,.2,20],[0,0,1.8,10],[0,0,0,1]];
  const surfaces=[1,2,3,4].map(id=>prepareSurface(Uint8Array.from(owner,v=>v===id),shape,affine,'soft_tissue',true,reference));
  const pairs=surfaces.map(locations);
  let shared=0,moved=0;
  for(const [k,p] of pairs[0]){
    const other=pairs[1].get(k);if(!other)continue;
    assert.deepEqual(p,other);shared++;
    if(k!==p.join(','))moved++;
  }
  assert(shared>100&&moved>20);
  for(const s of surfaces){
    assert.equal(intersectionVertices(Float64Array.from(s.positions),Float64Array.from(s.smoothPositions),s.indices).size,0);
    assert(s.diagnostics.maxDisplacementMm<=s.diagnostics.displacementLimitMm);
    assert(s.smoothNormals.every(Number.isFinite));
    // Every crossing remains strictly within its original simplex edge.
    for(let i=0;i<s.positions.length;i++){
      if(Number.isInteger(s.positions[i]))assert.equal(s.smoothPositions[i],s.positions[i]);
      else assert(Math.floor(s.positions[i])<s.smoothPositions[i]&&s.smoothPositions[i]<Math.ceil(s.positions[i]));
    }
  }
  for(const s of surfaces.slice(2))assert.deepEqual(s.smoothPositions,s.positions);
  assert.equal(hash(owner),before);
});

test('one source partition spans workers, source chunks, cavity cuts and undo without stale gradients or seams',()=>{
  const shape=[37,15,13],volume=grid(shape,(x,y,z)=>((x-27)/9)**2+((y-7)/5)**2+((z-6)/4)**2<=1);
  const reference={owner:Uint16Array.from(volume),protectedRegions:[]};
  const engine=new IncrementalSurfaceEngine(),key='source-owner',base={layerId:'tissue',shape,affine:identity,kind:'soft_tissue'};
  const admit=engine.prepare({...base,jobId:1,identity:'start',previousIdentity:'',mask:volume.slice(),partition:{key,reference}});
  const i=(31*shape[1]+7)*shape[2]+6,keys=['positions','indices','normals','smoothPositions','smoothNormals'];
  volume[i]=0;
  const cut=engine.prepare({...base,jobId:2,identity:'cut',previousIdentity:'start',indices:Uint32Array.of(i),values:Uint8Array.of(0),partition:{key}});
  assert.equal(cut.work.incremental,true);assert.equal(cut.work.remeshedChunks,2);
  const full=prepareSurface(volume,shape,identity,'soft_tissue',true,reference);
  for(const k of keys)assert.deepEqual(cut[k],full[k]);
  const undo=engine.prepare({...base,jobId:3,identity:'undo',previousIdentity:'cut',indices:Uint32Array.of(i),values:Uint8Array.of(1),partition:{key}});
  for(const k of keys)assert.deepEqual(undo[k],admit[k]);
  assert.throws(()=>new IncrementalSurfaceEngine().prepare({...base,jobId:4,identity:'missing',previousIdentity:'',mask:volume.slice(),partition:{key}}),/partition/);
  const limited=engine.prepare({...base,jobId:5,identity:'fallback',previousIdentity:'',mask:volume.slice(),partition:{key:'large',reference:null}});
  assert.deepEqual(limited.smoothPositions,limited.positions);assert.deepEqual(limited.smoothNormals,limited.normals);
  assert.match(limited.diagnostics.fallbackReason,/budget/);
});

test('a cut spanning source material boundaries keeps raw contact geometry, closes wedge faces and leaves distant and protected surfaces unchanged',()=>{
  const shape=[26,21,19],owner=grid(shape,(x,y,z)=>x===18&&y===10&&z>4&&z<13?3:x<5||x>20?0:y<10?1:2,Uint16Array);
  const removed=[];
  for(let x=4;x<15;x++)for(let y=0;y<21;y++)for(let z=0;z<19;z++){
    const i=(x*21+y)*19+z;
    if(owner[i]&&(y-9.7)**2+(z-8.6)**2<6.6)removed.push(i);
  }
  const occupied=owner.slice();for(const i of removed)occupied[i]=0;
  const before=hash(owner);
  for(const region of [1,2,3]){
    const mask=Uint8Array.from(occupied,v=>v===region),cells=Uint32Array.from(removed.filter(i=>owner[i]===region));
    const reference={owner,protectedRegions:[3]},cavity=cells.length?{region,cells}:undefined;
    const previous=prepareSurface(mask,shape,identity,'soft_tissue',true,reference);
    const next=prepareSurface(mask,shape,identity,'soft_tissue',true,{...reference,cavity});
    for(const k of ['positions','normals','indices'])assert.deepEqual(next[k],previous[k]);
    if(region===3){
      assert.deepEqual(next.smoothPositions,previous.smoothPositions);
      assert.equal(next.smoothIndices,undefined);
      continue;
    }
    assert(next.smoothIndices.length>previous.indices.length);
    assert(next.smoothPositions.length>previous.positions.length);
    assert(next.diagnostics.maxDisplacementMm<=.5);
    const counts=new Map();
    for(let i=0;i<next.smoothIndices.length;i+=3){
      const face=Array.from(next.smoothIndices.slice(i,i+3));assert.equal(new Set(face).size,3);
      for(let a=0;a<3;a++){
        const u=face[a],v=face[(a+1)%3],key=u<v?u+','+v:v+','+u;
        counts.set(key,(counts.get(key)??0)+1);
      }
    }
    assert([...counts.values()].every(n=>n===2),'every display edge must have two incident triangles');
    let moved=0;
    for(let i=0;i<previous.positions.length;i+=3){
      const difference=[0,1,2].some(k=>previous.smoothPositions[i+k]!==next.smoothPositions[i+k]);
      if(difference){moved++;assert(previous.positions[i]<17&&previous.positions[i]>3);}
    }
    assert(moved>15);
    const engine=new IncrementalSurfaceEngine(),base={layerId:'tissue',shape,affine:identity,kind:'soft_tissue'};
    engine.prepare({...base,jobId:1,identity:'original',previousIdentity:'',mask:Uint8Array.from(owner,n=>n===region),partition:{key:'owner',reference}});
    const edited=engine.prepare({...base,jobId:2,identity:'cut',previousIdentity:'original',indices:cells,values:new Uint8Array(cells.length),partition:{key:'owner',cavity}});
    for(const k of ['smoothPositions','smoothNormals','smoothIndices'])assert.deepEqual(edited[k],next[k]);
  }
  assert.equal(hash(owner),before);
});

test('generated skin exterior settles onto its multilayer source envelope while inner interfaces and protected vessel contacts remain fixed',()=>{
  const shape=[22,22,25],affine=identity;
  const owner=grid(shape,(x,y,z)=>{
    const roof=Math.floor(18-.015*(x-10)**2-.034*(y-10)**2);
    if(x===8&&y===9&&z===roof)return 13;
    return z>roof?0:z>=roof-1?1:2;
  },Uint16Array);
  const mask=Uint8Array.from(owner,id=>Number(id===1)),base={owner,protectedRegions:[13]};
  const before=prepareSurface(mask,shape,affine,'soft_tissue',true,base);
  const after=prepareSurface(mask,shape,affine,'soft_tissue',true,{...base,exteriorSkinRegion:1});
  assert.deepEqual(before.indices,after.indices);assert.deepEqual(before.positions,after.positions);
  const cell=(x,y,z)=>owner[(x*shape[1]+y)*shape[2]+z];
  let changed=0,checkedInterfaces=0;
  for(let i=0;i<after.positions.length;i+=3){
    const p=[...after.positions.slice(i,i+3)],prev=before.smoothPositions.subarray(i,i+3),next=after.smoothPositions.subarray(i,i+3);
    assert(Math.hypot(...p.map((v,k)=>v-next[k]))<=.50001);
    const diff=Math.hypot(...p.map((_,k)=>prev[k]-next[k]));
    if(diff>1e-6)changed++;
    if(p.some((v,k)=>v<1||v>=shape[k]-1))continue;
    const a=cell(...p.map(Math.floor)),b=cell(...p.map(Math.ceil));
    if((a&&b)||Math.hypot(p[0]-8,p[1]-9)<.8){
      assert(diff<1e-6);checkedInterfaces++;
    }
  }
  assert(changed>120);assert(checkedInterfaces>100);
});

test('native brain fairing is layer-bound and leaves caps, adjacent vessels and accepted cuts fixed',()=>{
  const shape=[29,23,20],affine=[[-.55,0,0,30],[0,.55,0,2],[0,0,.75,1],[0,0,0,1]];
  const owner=grid(shape,(x,y,z)=>{
    const roof=Math.floor(13-.027*(x-14)**2-.043*(y-11)**2);
    if(x===18&&z===roof+1&&y>=7&&y<=15)return 2;
    return z<=roof?1:0;
  },Uint16Array);
  const mask=Uint8Array.from(owner,v=>Number(v===1));
  const index=(x,y,z)=>(x*shape[1]+y)*shape[2]+z;
  const removed=index(13,10,12),before=[hash(owner),hash(mask)];
  const reference={owner,protectedRegions:[2],nativeBrainRegion:1};
  const plain=prepareSurface(mask,shape,affine,'brain',true,reference);
  const wrong=prepareSurface(mask,shape,affine,'brain',true,{...reference,layerRegion:2});
  assert.deepEqual(wrong.smoothPositions,plain.smoothPositions);
  const intact=prepareSurface(mask,shape,affine,'brain',true,{...reference,layerRegion:1});
  assert.equal(intact.diagnostics.policy,NATIVE_BRAIN_SURFACE_POLICY);
  const cutMask=mask.slice();cutMask[removed]=0;
  // A single accepted deletion has no separate cavity packet.
  const cutBase=prepareSurface(cutMask,shape,affine,'soft_tissue',true,reference);
  const engine=new IncrementalSurfaceEngine(),job={layerId:'brain',shape,affine,kind:'soft_tissue'};
  engine.prepare({...job,jobId:1,identity:'rest',previousIdentity:'',mask:mask.slice(),partition:{key:'native',reference,layerRegion:1}});
  const cut=engine.prepare({...job,jobId:2,identity:'cut',previousIdentity:'rest',indices:Uint32Array.of(removed),values:Uint8Array.of(0),
    partition:{key:'native',layerRegion:1}});
  assert.equal(cut.work.incremental,true);
  assert.equal(cut.diagnostics.policy,NATIVE_BRAIN_SURFACE_POLICY);
  for(const k of ['positions','indices','normals'])assert.deepEqual(cut[k],cutBase[k]);
  assert(cut.diagnostics.maxDisplacementMm<=.4);
  let changed=0,caps=0,nearCut=0,nearVessel=0;
  for(let i=0;i<cut.positions.length;i+=3){
    const [x,y,z]=cut.positions.subarray(i,i+3);
    const isCap=[x,y,z].some((v,k)=>v===-.5||v===shape[k]-.5);
    const isCut=Math.hypot((x-13)*.55,(y-10)*.55,(z-12)*.75)<1;
    const isVessel=Math.abs(x-18)<1.25&&y>7&&y<15&&z>9;
    const candidate=cut.smoothPositions.subarray(i,i+3),base=cutBase.smoothPositions.subarray(i,i+3);
    if(isCap||isCut||isVessel)assert.deepEqual(candidate,base);
    caps+=Number(isCap);nearCut+=Number(isCut);nearVessel+=Number(isVessel);
    if(candidate.some((v,k)=>v!==base[k]))changed++;
  }
  assert(caps>100&&nearCut>10&&nearVessel>10&&changed>100);
  const physical=points=>Float64Array.from(points,(v,i)=>v*affine[i%3][i%3]);
  assert.equal(intersectionVertices(physical(cutBase.smoothPositions),physical(cut.smoothPositions),cut.indices).size,0);
  const vessels=Uint8Array.from(owner,v=>Number(v===2));
  const originalVessel=prepareSurface(vessels,shape,affine,'vessel',true,reference);
  const scopedVessel=prepareSurface(vessels,shape,affine,'vessel',true,{...reference,layerRegion:2});
  for(const k of ['positions','indices','normals','smoothPositions','smoothNormals'])assert.deepEqual(scopedVessel[k],originalVessel[k]);
  assert.deepEqual([hash(owner),hash(mask)],before);
});
