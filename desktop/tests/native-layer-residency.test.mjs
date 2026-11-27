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
test('a layer review overlay stays resident beside its layer; stray overlays do not',()=>{
  const base=source(),overlay={name:'__vessel_review:vessels',mask:new Uint8Array(8),color:'#ffad54'};
  const tumor={name:'tumor',mask:new Uint8Array(8),color:'#ff00ff'},stray={name:'__vessel_review:retired',mask:new Uint8Array(8),color:'#ffad54'};
  const next=mergeResidentLayers(base,{...base,compartments:[overlay,tumor,stray]},['tumor','vessels']);
  assert.deepEqual(next.compartments.map(layer=>layer.name),['tumor','vessels','__vessel_review:vessels']);
});
test('derived compartments are kept once even when the order already names them',()=>{
  const base=source(),mask=new Uint8Array(8);
  const derived=['__vessel_review:vessels','__vessel_veins:vessels'].map(name=>({name,mask,color:'#ffad54'}));
  const current={...base,compartments:[base.compartments[0],...derived]};
  const tumor={name:'tumor',mask,color:'#ff00ff'};
  const next=mergeResidentLayers(current,{...base,compartments:[tumor]},[...current.compartments.map(l=>l.name),'tumor']);
  assert.deepEqual(next.compartments.map(l=>l.name),['vessels','__vessel_review:vessels','__vessel_veins:vessels','tumor']);
});
test('derived compartments give way before the viewer layer budget is exceeded',async()=>{
  const {MAX_ANATOMY_LAYERS}=await import('../src/viewer/labelPacking.ts');
  const base=source(),mask=new Uint8Array(8);
  const ids=Array.from({length:MAX_ANATOMY_LAYERS},(_,i)=>`layer-${i}`);
  const incoming=[...ids.map(name=>({name,mask,color:'#ff0000'})),{name:'__vessel_review:layer-0',mask,color:'#ffad54'}];
  const next=mergeResidentLayers({...base,compartments:[]},{...base,compartments:incoming},ids);
  assert.equal(next.compartments.length,MAX_ANATOMY_LAYERS);
  assert.ok(next.compartments.every(l=>!l.name.startsWith('__vessel_review:')));
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

test('detached CT support reuses only the exact source-bound native crop',async()=>{
  const base=source(),signalSha256='a'.repeat(64),sha256='b'.repeat(64);
  const mask={assetId:'sheet',dtype:'uint8',shape:[2,1,1],byteLength:2,byteOrder:'little',order:'C',sha256};
  const receipt={schema:'native-ct-exterior-sheet-v1',signalSha256,offset:[0,1,0],mask};
  const payload={...base,modality:'CTA',mri:{sha256:signalSha256},compartments:[],labelSets:[],ctExterior:receipt};
  const api={readAsset:async()=>new Uint8Array([1,0])};
  const view=await hydrateAnatomy(payload,api,base.mri);
  assert.equal(view.mri,base.mri);
  assert.deepEqual(view.ctExterior,{offset:[0,1,0],shape:[2,1,1],data:new Uint8Array([1,0]),sha256});
  const resident=await hydrateAnatomy(payload,{readAsset(){throw Error('Should reuse the checked crop');}},view.mri,view);
  assert.equal(resident.ctExterior,view.ctExterior);
  for(const bad of [{...receipt,signalSha256:'c'.repeat(64)},{...receipt,offset:[1,1,0]}]){
    await assert.rejects(hydrateAnatomy({...payload,ctExterior:bad},api,view.mri),/differs from its acquired source/);
  }
  assert.throws(()=>mergeResidentLayers(view,{...view,ctExterior:undefined},[]),/different displayed source/);
});

test('cortex from a different scan reuses its verified samples with the new physical transform',async()=>{
  const base=source(),mr=new Float32Array(12).fill(55);
  const mrFrame=affine.map(row=>[...row]);mrFrame[0][3]=17;
  const signal={sha256:'mr',shape:[3,2,2],dtype:'float32',byteLength:mr.byteLength};
  const payload={...base,mri:{sha256:'ct'},labelSets:[{id:'regions',data:{}}],
    appearanceSources:[{seriesId:'mr',modality:'T1',sourceSha256:'original',affine:mrFrame,signal}],
    compartments:['left','right','index-only'].map(name=>({name,sourceSeries:'mr',displayKind:'gray-matter',
      alignment:name==='index-only'?'source-grid preview':'reviewed',labelSet:'regions',labelValue:1,color:'#aaa'}))};
  const api={readAsset(){throw Error('Validated resident samples should not need another transfer')}};
  const loaded=await hydrateAnatomy(payload,api,base.mri,base,undefined,new Map([['mr',mr]]));
  const [left,right,indexOnly]=loaded.compartments;
  assert.equal(left.appearanceSource,right.appearanceSource);
  assert.equal(left.appearanceSource.data,mr);
  assert.deepEqual(left.appearanceSource.affine,mrFrame);
  assert.deepEqual(left.appearanceSource.shape,[3,2,2]);
  assert.equal(indexOnly.appearanceSource,undefined);
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
