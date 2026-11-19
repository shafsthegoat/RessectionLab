import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import test from 'node:test';
import {prepareSurface} from './prepareSurface.ts';
import {maskSurfaceRegion} from './surface.ts';
import {IncrementalSurfaceEngine} from './incrementalSurface.ts';
import {ENGINE_SURFACE_POLICY} from './engineImplicitSurface.ts';
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
