import assert from 'node:assert/strict';
import {test} from 'node:test';
import {GpuFrameTimer} from './gpuFrameTimer.ts';
import {geometryBufferBytes,renderTargetBytes} from './renderResourceEstimate.ts';
import * as THREE from 'three';
function fixture(supported=true){
  let counter=0,disjoint=false,lost=false;const callbacks=new Map(),deleted=[],samples=[],queries=[];
  const gl={QUERY_RESULT_AVAILABLE:1,QUERY_RESULT:2,getExtension:()=>supported?{TIME_ELAPSED_EXT:3,GPU_DISJOINT_EXT:4}:null,
    isContextLost:()=>lost,getParameter:()=>disjoint,createQuery:()=>{const q={id:++counter,ready:false,nanoseconds:1250000};queries.push(q);return q;},
    beginQuery(){},endQuery(){},deleteQuery:q=>deleted.push(q),getQueryParameter:(q,key)=>key===1?q.ready:q.nanoseconds};
  const timer=new GpuFrameTimer(gl,s=>samples.push(s),callback=>{const id=++counter;callbacks.set(id,callback);return id;},id=>callbacks.delete(id));
  return {timer,samples,queries,deleted,setDisjoint:v=>disjoint=v,setLost:v=>lost=v,poll(){const work=[...callbacks.values()];callbacks.clear();work.forEach(cb=>cb());},callbacks};
}
test('GPU query result reports actual asynchronous ns-to-ms separately, and missing extension creates no query',()=>{
  const f=fixture();f.timer.begin(7);f.timer.end();f.poll();assert.equal(f.samples.length,0);f.queries[0].ready=true;f.poll();
  assert.deepEqual(f.samples,[{serial:7,measurement:'asynchronous-GPU-elapsed',completed:true,milliseconds:1.25}]);assert.equal(f.deleted.length,1);
  const missing=fixture(false);missing.timer.begin(1);missing.timer.end();assert.equal(missing.timer.supported,false);assert.equal(missing.queries.length,0);assert.equal(missing.callbacks.size,0);
});
test('query count is capped and disjoint/context-loss results are rejected without elapsed values',()=>{
  const f=fixture();for(let i=0;i<9;i++){f.timer.begin(i);f.timer.end();}assert.equal(f.queries.length,3);f.setDisjoint(true);f.poll();
  assert.equal(f.samples.length,3);assert(f.samples.every(s=>!s.completed&&s.reason==='disjoint'&&!('milliseconds'in s)));assert.equal(f.deleted.length,3);
  const lost=fixture();lost.timer.begin(1);lost.timer.end();lost.setLost(true);lost.poll();assert.equal(lost.samples[0].reason,'context-lost');assert.equal(lost.deleted.length,0);
});
test('dispose cancels polls and releases outstanding queries once',()=>{
  const f=fixture();f.timer.begin(2);f.timer.end();f.timer.dispose();f.timer.dispose();assert.equal(f.deleted.length,1);assert.equal(f.callbacks.size,0);assert.equal(f.samples.length,1);
});
test('GPU allocation estimates count distinct attribute uploads, deduplicate shared attributes and count attachments',()=>{
  const data=new Float32Array(12),a=new THREE.BufferGeometry(),b=new THREE.BufferGeometry();a.setAttribute('position',new THREE.BufferAttribute(data,3));b.setAttribute('position',new THREE.BufferAttribute(data.subarray(0,9),3));
  assert.equal(geometryBufferBytes([a,b]),data.byteLength+data.subarray(0,9).byteLength);
  b.setAttribute('position',a.getAttribute('position'));assert.equal(geometryBufferBytes([a,b]),data.byteLength);
  const target=new THREE.WebGLRenderTarget(10,20,{type:THREE.HalfFloatType,depthBuffer:true});target.depthTexture=new THREE.DepthTexture(10,20,THREE.UnsignedIntType);
  assert.equal(renderTargetBytes(target),10*20*(8+4));target.dispose();a.dispose();b.dispose();
});
