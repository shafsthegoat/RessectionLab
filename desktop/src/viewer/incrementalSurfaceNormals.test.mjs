import assert from 'node:assert/strict';
import {test} from 'node:test';
import {ChunkedMaskSurface} from './chunkedSurface.ts';
import {IncrementalSurfaceEngine} from './incrementalSurface.ts';
import {prepareSurface} from './prepareSurface.ts';
const affines=[[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],[[-1,.23,0,10],[.11,2,.17,-20],[0,.03,1.37,30],[0,0,0,1]],[[1,.2,0,0],[0,2,0,0],[0,0,0,0],[0,0,0,1]]];
const equal=(actual,expected,keys=['positions','indices','normals'])=>{for(const key of keys)assert.deepEqual(new Uint8Array(actual[key].buffer),new Uint8Array(expected[key].buffer),key);};
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
  result=engine.prepare({...base,jobId:3,previousIdentity:'b',identity:'b',indices:new Uint32Array(),values:new Uint8Array(),smooth:false});equal(result,prepareSurface(mask,shape,affine,'bone',false));assert(result.work.cachedNormals);
});
