import assert from 'node:assert/strict';
import {test} from 'node:test';
import {ChunkedMaskSurface} from './chunkedSurface.ts';
import {IncrementalSurfaceEngine} from './incrementalSurface.ts';
import {prepareSurface} from './prepareSurface.ts';
import {IncrementalSourceNormals} from './incrementalSurfaceNormals.ts';
import {indexedSurfaceNormals} from './surfaceNormals.ts';
const affines=[[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],[[-1,.23,0,10],[.11,2,.17,-20],[0,.03,1.37,30],[0,0,0,1]],[[1,.2,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,1]]];
const equal=(actual,expected,keys=['positions','indices','normals'])=>{for(const key of keys)assert.deepEqual(new Uint8Array(actual[key].buffer),new Uint8Array(expected[key].buffer),key);};
const equalCorners=(actual,expected)=>{
  const bytes=surface=>{const out=new Float32Array(surface.indices.length*6);let offset=0;for(const id of surface.indices)for(const key of ['positions','normals'])for(let axis=0;axis<3;axis++)out[offset++]=surface[key][id*3+axis];return new Uint8Array(out.buffer);};
  assert.equal(actual.indices.length,expected.indices.length);assert.deepEqual(bytes(actual),bytes(expected));
};
const shape=[35,12,10],at=(x,y,z)=>(x*shape[1]+y)*shape[2]+z;
const source=()=>Uint8Array.from({length:35*12*10},(_,i)=>{const x=Math.floor(i/120),y=Math.floor(i/10)%12,z=i%10;return Number(x>0&&x<34&&y>0&&y<11&&z>0&&z<9);});
test('dirty source normals remain byte-identical through removed faces, cut walls, seams, undo and affine changes',()=>{
  for(const affine of affines){
    const mask=source(),before=mask.slice(),surface=new ChunkedMaskSurface(mask,shape,4,8*1024**2);
    const check=()=>{const indexed=surface.indexed(affine);equal(indexed,prepareSurface(mask,shape,affine,'bone',false));assert(surface.normalWork.cachedNormals);return indexed;};
    check();
    for(const indices of [[at(3,4,4),at(4,4,4),at(31,8,4),at(32,8,4)],[at(0,0,0),at(34,11,9)],[at(16,4,4)]]){
      indices.forEach(i=>mask[i]=1-mask[i]);surface.update(indices);const result=check();assert(surface.normalWork.dirtyNormalVertices<result.positions.length/3);assert(surface.normalWork.recomputedNormalFaces<result.indices.length/3);
    }
    const restored=[];for(let i=0;i<mask.length;i++)if(mask[i]!==before[i]){mask[i]=before[i];restored.push(i);}surface.update(restored);check();
    const result=check();assert.equal(surface.normalWork.dirtyNormalVertices,0);assert.equal(surface.normalWork.recomputedNormalFaces,0);
    structuredClone(result,{transfer:[result.positions.buffer,result.indices.buffer,result.normals.buffer]});check();
    const removed=[];for(let i=0;i<mask.length;i++)if(mask[i]){mask[i]=0;removed.push(i);}surface.update(removed);assert.equal(check().positions.length,0);
    mask[at(4,4,4)]=1;surface.update([at(4,4,4)]);check();
    for(const changedAffine of affines)equal(surface.indexed(changedAffine),prepareSurface(mask,shape,changedAffine,'bone',false));
  }
});
test('random sparse changes preserve every normal byte under a nonbinary reflected shear',()=>{
  const mask=source(),affine=affines[1],surface=new ChunkedMaskSurface(mask,shape,4,8*1024**2);surface.indexed(affine);let seed=2301;
  for(let batch=0;batch<12;batch++){
    const ids=[];for(let j=0;j<7;j++){seed=(Math.imul(seed,1664525)+1013904223)>>>0;const id=seed%mask.length;mask[id]=1-mask[id];ids.push(id);}
    surface.update(ids);equal(surface.indexed(affine),prepareSurface(mask,shape,affine,'bone',false));
  }
});
test('normal-cache budget exhaustion preserves indexed chunks and falls back to complete exact normals',()=>{
  const mask=source(),affine=affines[1],plain=new ChunkedMaskSurface(mask.slice(),shape,32,8*1024**2),budget=plain.retainedBytes+128;
  const engine=new IncrementalSurfaceEngine(budget),result=engine.prepare({jobId:1,layerId:'bone',previousIdentity:'',identity:'raw',shape,affine,kind:'bone',mask:mask.slice(),smooth:false});
  equal(result,prepareSurface(mask,shape,affine,'bone',false));assert.equal(result.work.indexedChunks,true);assert.equal(result.work.cachedNormals,false);assert(result.work.retainedBytes<=budget);assert.deepEqual(result.resident,[{layerId:'bone',identity:'raw'}]);
});
test('raw bone worker normals cache survives preference changes and preserves complete conditioned buffers',()=>{
  const mask=source(),affine=affines[1],engine=new IncrementalSurfaceEngine(),base={layerId:'bone',shape,affine,kind:'bone'};
  let result=engine.prepare({...base,jobId:1,previousIdentity:'',identity:'a',mask:mask.slice(),smooth:false});equal(result,prepareSurface(mask,shape,affine,'bone',false));assert(result.work.cachedNormals);
  mask[at(31,4,4)]=0;
  result=engine.prepare({...base,jobId:2,previousIdentity:'a',identity:'b',indices:Uint32Array.of(at(31,4,4)),values:Uint8Array.of(0),smooth:true});
  equal(result,prepareSurface(mask,shape,affine,'bone',true),['positions','indices','normals','smoothPositions','smoothNormals']);assert.equal(result.work.cachedNormals,false);
  result=engine.prepare({...base,jobId:3,previousIdentity:'b',identity:'b',indices:new Uint32Array(),values:new Uint8Array(),smooth:false});equalCorners(result,prepareSurface(mask,shape,affine,'bone',false));assert.equal(result.representation,'stable-source-ids');assert(result.work.cachedNormals);
});
test('stable normal snapshots copy the complete registry prefix without detaching the retained cache',()=>{
  const points=Float32Array.of(0,0,0,1,0,0,0,1,0),indices=Uint32Array.of(0,1,2),cache=new IncrementalSourceNormals(()=>{}),affine=affines[1];
  cache.prepare(points,3,[{data:indices}],affine);const first=cache.emitStable(3),expected=indexedSurfaceNormals(points,indices,affine);assert.deepEqual(first,expected);
  structuredClone(first,{transfer:[first.buffer]});assert.equal(first.byteLength,0);assert.deepEqual(cache.emitStable(3),expected);
});
test('normal face scanning stays within changed chunks and halo while retaining exact global bytes',()=>{
  const shape=[21,19,17],at=(x,y,z)=>(x*shape[1]+y)*shape[2]+z,mask=Uint8Array.from({length:shape.reduce((a,b)=>a*b,1)},(_,i)=>{const x=Math.floor(i/(19*17)),y=Math.floor(i/17)%19,z=i%17;return Number(x>0&&x<20&&y>0&&y<18&&z>0&&z<16);}),surface=new ChunkedMaskSurface(mask,shape,4,8*1024**2),affine=affines[1];
  const check=()=>{const mesh=surface.indexed(affine);equal(mesh,prepareSurface(mask,shape,affine,'bone',false));assert(surface.retainedBytes<=8*1024**2);return mesh;};
  const initial=check();assert.equal(surface.normalWork.normalFullScan,true);assert.equal(surface.normalWork.scannedNormalFaces,initial.indices.length/3);
  assert.equal(surface.normalWork.normalSelectionBytes,Math.ceil(surface.totalChunks/8));
  mask[at(9,9,9)]=0;surface.update([at(9,9,9)]);const cavity=check();assert.equal(surface.normalWork.normalFullScan,false);assert(surface.normalWork.scannedNormalFaces<cavity.indices.length/6,'a new local cavity does not scan the complete outer source');assert(surface.normalWork.recomputedNormalFaces>0,'newly born cavity vertices must be normalized');
  // Union old and new chunk ownership over several edits, including an empty
  // border core, without assembling normals between updates.
  for(const ids of [[at(0,0,0),at(20,18,16)],[at(3,3,3),at(4,4,4),at(19,17,15)]]){for(const id of ids)mask[id]=1-mask[id];surface.update(ids);surface.indexed();}
  check();assert.equal(surface.normalWork.normalFullScan,false);
  mask[at(0,0,0)]=0;surface.update([at(0,0,0)]);check();assert.equal(surface.normalWork.normalFullScan,false,'deleting an isolated now-empty core keeps its old vertices dirty');
  const all=[];for(let i=0;i<mask.length;i++)if(mask[i]){mask[i]=0;all.push(i);}surface.update(all);assert.equal(check().positions.length,0);
  mask[at(20,18,16)]=1;surface.update([at(20,18,16)]);check();
  for(const changed of affines){const mesh=surface.indexed(changed);equal(mesh,prepareSurface(mask,shape,changed,'bone',false));assert.equal(surface.normalWork.normalFullScan,true);assert.equal(surface.normalWork.scannedNormalFaces,mesh.indices.length/3);}
});
test('missing chunk or row bookkeeping falls back to complete face scanning without changing normal reduction',()=>{
  const affine=affines[1],points=Float32Array.of(0,0,0,1,0,0,0,1,0,20,20,20,21,20,20,20,21,20),a=Uint32Array.of(0,1,2),b=Uint32Array.of(3,4,5),rows=[{x:0,y:0,z:0,data:a},{x:20,y:20,z:20,data:b}],indices=Uint32Array.of(...a,...b),remap=Uint32Array.of(1,2,3,4,5,6),admitted=[];
  const normals=new IncrementalSourceNormals(bytes=>admitted.push(bytes),{shape:[24,24,24],chunkSize:4});
  const check=currentRows=>{normals.prepare(points,6,currentRows,affine);equal({normals:normals.emit(remap,6)},{normals:indexedSurfaceNormals(points,indices,affine)},['normals']);};
  check(rows);assert.equal(normals.lastFullScan,true);assert.equal(normals.selectionBytes,Math.ceil(7**3/8));assert.equal(admitted.at(-1),1024*13+normals.selectionBytes);assert.equal(normals.retainedBytes,admitted.at(-1));
  points[11]=21;normals.mark(b);check(rows);assert.equal(normals.lastFullScan,true);assert.equal(normals.lastScannedFaces,2);
  points[11]=20;normals.mark(b,[5,5,5]);check(rows.map(({data})=>({data})));assert.equal(normals.lastFullScan,true);assert.equal(normals.lastScannedFaces,2);
  points[11]=21;normals.mark(b,[5,5,5]);check(rows);assert.equal(normals.lastFullScan,false);assert.equal(normals.lastScannedFaces,1);
});
test('halo bookkeeping remains bounded and cache reconstruction performs a full exact scan after eviction',()=>{
  const mask=source(),affine=affines[1],surface=new ChunkedMaskSurface(mask,shape,4,8*1024**2);surface.indexed(affine);const selectionBytes=surface.normalWork.normalSelectionBytes,ids=[at(3,4,4),at(4,4,4)],retainedByState=new Map();
  for(let i=0;i<8;i++){for(const id of ids)mask[id]=1-mask[id];surface.update(ids);equal(surface.indexed(affine),prepareSurface(mask,shape,affine,'bone',false));assert.equal(surface.normalWork.normalSelectionBytes,selectionBytes);if(i<2)retainedByState.set(i,surface.retainedBytes);else assert.equal(surface.retainedBytes,retainedByState.get(i%2));}
  const budget=surface.retainedBytes,restored=new ChunkedMaskSurface(mask.slice(),shape,4,budget);equal(restored.indexed(affine),prepareSurface(mask,shape,affine,'bone',false));assert.equal(restored.normalWork.normalFullScan,true);assert(restored.retainedBytes<=budget);
});
test('selection-bitset admission failure drops only cached normals and keeps the exact indexed surface',()=>{
  const mask=source(),affine=affines[1],plain=new ChunkedMaskSurface(mask.slice(),shape,32,8*1024**2),budget=plain.retainedBytes;
  const result=new IncrementalSurfaceEngine(budget).prepare({jobId:1,layerId:'bone',previousIdentity:'',identity:'bounded',shape,affine,kind:'bone',mask:mask.slice(),smooth:false});
  equal(result,prepareSurface(mask,shape,affine,'bone',false));assert.equal(result.work.indexedChunks,true);assert.equal(result.work.cachedNormals,false);assert.equal(result.work.normalSelectionBytes,0);assert(result.work.retainedBytes<=budget);
});
test('non-power-of-two 3D chunk seams and repeated sparse changes retain every canonical normal byte',()=>{
  const shape=[23,21,19],size=shape.reduce((a,b)=>a*b,1),mask=new Uint8Array(size),affine=affines[1];let seed=71009;
  const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed;};
  for(let id=0;id<size;id++)mask[id]=Number(random()%7===0);
  const surface=new ChunkedMaskSurface(mask,shape,3,32*1024**2);surface.indexed(affine);
  for(let batch=0;batch<8;batch++){
    const ids=[];for(let edit=0;edit<5;edit++){const id=random()%size;mask[id]=1-mask[id];ids.push(id);}surface.update(ids);
    equal(surface.indexed(affine),prepareSurface(mask,shape,affine,'bone',false));assert.equal(surface.normalWork.normalFullScan,false);assert(surface.retainedBytes<=32*1024**2);
  }
});
