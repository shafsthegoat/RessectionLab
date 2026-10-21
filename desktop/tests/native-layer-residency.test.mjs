import test from 'node:test';
import assert from 'node:assert/strict';
import {mergeResidentLayers,copyPackingArray} from '../src/native-layer-residency.ts';
import {drainLayerRequests} from '../src/native-layer-queue.ts';
import {hydrateAnatomy} from '../src/native-anatomy-data.ts';
const affine=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
const source=()=>({displayIdentity:'scan/revision/original/1',caseId:'patient',caseHash:'case-a',frame:'RAS+',shape:[2,2,2],affine,mri:new Float32Array(8),compartments:[{name:'vessels',mask:new Uint8Array(8),color:'#ff0000'}],labelSets:[{id:'regions',data:new Uint16Array(8)}]});
test('incremental layers retain original arrays and order, including hidden resident layers',()=>{
  const base=source(),tumor={name:'tumor',mask:new Uint8Array(8),color:'#ff00ff'};
  const next=mergeResidentLayers(base,{...base,compartments:[tumor]},['tumor','vessels']);
  assert.equal(next.mri,base.mri);assert.equal(next.compartments[1],base.compartments[0]);assert.equal(next.compartments[0],tumor);assert.equal(next.labelSets[0],base.labelSets[0]);assert.equal(base.compartments.length,1);
});
test('source, revision, affine and preview replacements cannot enter the resident layer path',()=>{
  const base=source();
  for(const change of [{displayIdentity:'scan/revision/preview/2'},{caseHash:'case-b'},{mri:new Float32Array(8)},{affine:affine.map(row=>row.map(v=>v*2))},{frame:'LPS+'}])assert.throws(()=>mergeResidentLayers(base,{...base,...change},['vessels']),/different displayed source/);
});
test('hydration reuses admitted image and shared label grid without asset reads',async()=>{
  const base=source(),payload={...base,mri:{},labelSets:[{id:'regions',data:{}}],compartments:[{name:'region',labelSet:'regions',labelValue:2,color:'#ffffff'}]};
  const hydrated=await hydrateAnatomy(payload,{readAsset(){throw Error('Resident assets must not transfer')}},base.mri,base);
  assert.equal(hydrated.mri,base.mri);assert.equal(hydrated.labelSets[0],base.labelSets[0]);
});

test('explicit empty surfaces hydrate without inventing triangles or transferring zero-length assets',async()=>{
  const base=source(),mask=new Uint8Array(8),descriptor={byteLength:8,byteOrder:'little',order:'C',assetId:'mask'};
  const payload={...base,mri:{},labelSets:[],compartments:[{name:'empty',kind:'tumor',mask:descriptor,surface:{empty:true},color:'#ff00ff'}]};
  const loaded=await hydrateAnatomy(payload,{readAsset:async()=>mask},base.mri);
  assert.equal(loaded.compartments[0].surface.indices.length,0);assert.equal(loaded.compartments[0].surface.source.length,0);
  await assert.rejects(hydrateAnatomy({...payload,compartments:[{...payload.compartments[0],surface:{empty:true,positions:{}}}]},{readAsset:async()=>mask},base.mri),/Invalid empty/);
});

test('worker preparation copies source cells without detaching them and honors cancellation',async()=>{
  const source=new Uint16Array([0,128,65535]),copy=await copyPackingArray(source);
  assert.deepEqual(copy,source);assert.notEqual(copy.buffer,source.buffer);
  copy[0]=9;assert.equal(source[0],0);
  const controller=new AbortController();controller.abort();await assert.rejects(copyPackingArray(source,controller.signal),{name:'AbortError'});
});
function queue(){
  const state={volume:source(),order:['vessels','a','b'],visible:{vessels:true,a:true},unavailable:[],failed:{}};
  let current=true;const requests=[],pending=[],installed=[];
  const options={current:()=>state,isCurrent:()=>current,pending:ids=>pending.push(ids),
    load:(base,ids)=>new Promise((resolve,reject)=>requests.push({base,ids,resolve,reject})),
    install:result=>{state.volume=result.volume;state.unavailable=result.unavailable;installed.push(result)},
    fail:(ids,message)=>{for(const id of ids)state.failed[id]=message}};
  const answer=(index)=>{const r=requests[index];r.resolve({volume:{...r.base,compartments:[...r.base.compartments,...r.ids.map(name=>({name,mask:new Uint8Array(8),color:'#ffffff'}))]},unavailable:[]});};
  return {state,requests,pending,installed,options,answer,invalidate:()=>{current=false}};
}
test('rapid layer changes retain completed layers while serially loading the latest requested selection',async()=>{
  const q=queue(),done=drainLayerRequests(q.options);assert.deepEqual(q.requests[0].ids,['a']);
  q.state.visible={vessels:false,a:false,b:true};q.answer(0);await new Promise(resolve=>setImmediate(resolve));
  assert.deepEqual(q.requests[1].ids,['b']);q.answer(1);await done;
  assert.deepEqual(q.state.volume.compartments.map(l=>l.name),['vessels','a','b']);assert.deepEqual(q.state.visible,{vessels:false,a:false,b:true});
  await drainLayerRequests(q.options);assert.equal(q.requests.length,2,'resident toggles perform no reads');
});
test('failed layer keeps existing anatomy and can be explicitly retried',async()=>{
  const q=queue(),base=q.state.volume,done=drainLayerRequests(q.options);q.requests[0].reject(Error('transfer failed'));await done;
  assert.equal(q.state.volume,base);assert.equal(q.state.failed.a,'transfer failed');assert.equal(q.installed.length,0);
  delete q.state.failed.a;const retry=drainLayerRequests(q.options);q.answer(1);await retry;assert.equal(q.installed.length,1);
});
test('case switch/cancellation ignores both stale data and stale error state',async()=>{
  for(const reject of [false,true]){
    const q=queue(),base=q.state.volume,done=drainLayerRequests(q.options);q.invalidate();
    if(reject)q.requests[0].reject(Error('old source failed'));else q.answer(0);
    await done;assert.equal(q.state.volume,base);assert.equal(q.installed.length,0);assert.deepEqual(q.state.failed,{});
  }
});

test('scope replacement discards in-flight filtered layers before installing full source',async()=>{
  const q=queue(),done=drainLayerRequests(q.options);
  q.invalidate();
  const full={...source(),displayIdentity:'scan/revision/all/2',compartments:[{name:'vessels',mask:new Uint8Array(8).fill(1),color:'#ff0000'}]};
  q.state.volume=full;q.answer(0);await done;
  assert.equal(q.state.volume,full);assert.equal(q.installed.length,0);
  assert.equal(q.state.volume.compartments[0].mask.reduce((a,b)=>a+b),8);
});
