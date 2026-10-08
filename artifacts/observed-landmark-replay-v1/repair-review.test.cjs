'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),path=require('node:path'),fs=require('node:fs');
const root=path.resolve(__dirname,'../..'),contract=require(path.join(root,'desktop/electron/observed-landmark-contract.cjs'));
const {Sidecar}=require(path.join(root,'desktop/electron/sidecar.cjs'));
const frames=JSON.parse(fs.readFileSync(path.join(__dirname,'repair-authentic-12.json')));
const request={action:'open',studyId:'resect-case4-sparse-update-v1'};
test('all12 fixed transport pins independently match complete authentic response bytes',()=>{
 assert.equal(frames.length,12);
 for(const f of frames){
   assert.equal(contract.PINS[f.key],f.independent_digest);
   assert.equal(contract.transportDigest(f.result),f.independent_digest);
   assert.equal(contract.validateObservedResult(f.result,request),f.result);
 }
});
function harness(){
 const sidecar=Object.create(Sidecar.prototype),events=[];let traversals=0,resolved;
 sidecar.pending=new Map([['r',{op:'inspectObservedLandmarkUpdate',observedArgs:request,resolve:r=>resolved=r,reject(){}}]]);
 sidecar.assets={expose:async x=>{traversals++;return x;}};
 sidecar.emit=(name,msg)=>{events.push([name,msg]);if(msg.event==='result'){
   assert.equal(Object.isFrozen(msg.result),true);
   assert.equal(Object.isFrozen(msg.result.state.observations[0].sourceRasMm),true);
   assert.throws(()=>{msg.result.state.observations[0].sourceRasMm[0]=0;},TypeError);
 }};
 return {sidecar,events,traversals:()=>traversals,resolved:()=>resolved};
}
test('sidecar same valid request rejects corrupt result without events/assets and accepts immutable authentic result',async()=>{
 const h=harness(),bad=structuredClone(frames[0].result);bad.validationLandmarks='FORBIDDEN_SENTINEL';
 await assert.rejects(h.sidecar.handle({id:'r',event:'result',result:bad}));
 assert.equal(h.traversals(),0);assert.equal(h.events.length,0);assert.equal(h.resolved(),undefined);
 const actual=structuredClone(frames[0].result);
 await h.sidecar.handle({id:'r',event:'result',result:actual});
 assert.equal(h.traversals(),0);assert.equal(h.events.length,1);assert.equal(h.resolved(),actual);
});
test('unexpected payload cannot hide in sidecar nonresult events',async()=>{
 for(const event of ['started','cancelled','progress','error']){
  const h=harness();const message={id:'r',event,validationLandmarks:'FORBIDDEN_SENTINEL'};
  if(event==='error')message.error={code:'OBSERVED_SOURCE_UNAVAILABLE',message:'Missing'};
  await assert.rejects(h.sidecar.handle(message));
  assert.equal(h.events.length,0);assert.equal(h.traversals(),0);
 }
});
