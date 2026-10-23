import test from 'node:test';
import assert from 'node:assert/strict';
import {checkedSurgicalPlayback} from '../src/surgical-playback.ts';
import {checkedFlowDisplayForView,MAX_FLOW_OPERATIONS} from '../src/surgical-flow-display.ts';
const hash=letter=>'sha256:'+letter.repeat(64);
function fixture(){
  const view={stateId:hash('a'),scene:{sceneId:hash('b'),shape:[3,3,3]},slots:{},metrics:{elapsedSeconds:2.2}};
  const flow={schema:'resectionlab.recorded-field-transfers.v1',sceneId:view.scene.sceneId,actionIndex:4,frameIndex:1,beforeStateId:hash('c'),stateId:view.stateId,startSeconds:2,endSeconds:2.2,truncated:false,operations:[{kind:'transfer',field:'fluid',provenance:'mixed',fromCell:13,toCell:14,volumeMm3:.5,startSeconds:2.1,endSeconds:2.2}]};
  const playback={readOnly:true,branchId:'main',position:4,actionIndex:4,frameIndex:1,frameCount:2,observedTimeSeconds:2.2,actionElapsedSeconds:.2,actionDurationSeconds:.2,isActionBoundary:true,frames:[{frameIndex:0,observedTimeSeconds:2,kind:'action_start'},{frameIndex:1,observedTimeSeconds:2.2,kind:'action_boundary'}],flowDisplay:flow};
  return {view,playback};
}
const request={branchId:'main',position:4,frameIndex:1};
test('a demanded historical interval is bound to its endpoint, independent of the UI previous frame',()=>{
  const response=fixture(),before=structuredClone(response);
  assert.equal(checkedSurgicalPlayback(response,request).flowDisplay,response.playback.flowDisplay);
  assert.deepEqual(response,before);
  delete response.playback.flowDisplay;assert.equal(checkedSurgicalPlayback(response,request).flowDisplay,undefined,'legacy absence remains available as static endpoint review');
});
test('transfer admission rejects substituted sources, frames, times, cells, routes and invented species',()=>{
  for(const change of [
    f=>f.schema='other',f=>f.sceneId=hash('d'),f=>f.stateId=hash('e'),f=>f.beforeStateId='broken',f=>f.actionIndex=5,f=>f.frameIndex=0,
    f=>f.startSeconds=1,f=>f.endSeconds=2.3,f=>f.truncated=1,f=>f.extra=true,f=>f.operations[0].volumeMm3=0,f=>f.operations[0].volumeMm3=Infinity,
    f=>f.operations[0].fromCell=true,f=>f.operations[0].toCell=27,f=>f.operations[0].toCell=17,f=>f.operations[0].toCell=15,
    f=>f.operations[0].fromCell=null,f=>f.operations[0].provenance='wound',f=>f.operations[0].field='blood',f=>f.operations[0].extra=1,
    f=>f.operations[0].startSeconds=1.9,f=>f.operations[0].endSeconds=2.3,f=>f.operations[0].endSeconds=2,
  ]){const response=fixture();change(response.playback.flowDisplay);assert.throws(()=>checkedSurgicalPlayback(response,request),/Recorded field transfers/);}
  const response=fixture();response.playback.flowDisplay=null;assert.throws(()=>checkedSurgicalPlayback(response,request));
});
test('local sources and sinks cannot fabricate a route; mixed fluid and debris stay distinct',()=>{
  for(const field of ['fluid','debris'])for(const kind of ['uptake','escape','deposit']){
    const {view,playback}=fixture(),flow=playback.flowDisplay,op=flow.operations[0];
    Object.assign(op,{kind,field,provenance:field==='fluid'?'mixed':'debris',fromCell:kind==='deposit'?null:13,toCell:kind==='deposit'?13:null});
    assert.equal(checkedFlowDisplayForView(flow,view),flow);
    op[kind==='deposit'?'fromCell':'toCell']=14;assert.throws(()=>checkedFlowDisplayForView(flow,view));
  }
  for(const provenance of ['wound','irrigation','declared-source']){
    const {view,playback}=fixture(),flow=playback.flowDisplay;Object.assign(flow.operations[0],{kind:'deposit',fromCell:null,toCell:13,provenance});
    assert.equal(checkedFlowDisplayForView(flow,view),flow);
  }
  const {view,playback}=fixture(),flow=playback.flowDisplay;Object.assign(flow.operations[0],{kind:'escape',fromCell:null,toCell:null});
  assert.equal(checkedFlowDisplayForView(flow,view),flow,'an off-grid outlet has no invented cell');
});
test('empty, truncated and unavailable are distinct; frame zero contains no historical transport',()=>{
  const {view,playback}=fixture(),flow=playback.flowDisplay;flow.operations=[];
  assert.equal(checkedFlowDisplayForView(flow,view).truncated,false);flow.truncated=true;assert.equal(checkedFlowDisplayForView(flow,view).truncated,true);
  assert.equal(checkedFlowDisplayForView(undefined,view),undefined);
  flow.frameIndex=0;flow.startSeconds=flow.endSeconds;flow.beforeStateId=flow.stateId;assert.equal(checkedFlowDisplayForView(flow,view),flow);
  flow.beforeStateId=hash('c');assert.throws(()=>checkedFlowDisplayForView(flow,view));
});
test('bounded display records never accept more than 256 writes or reorder operation time',()=>{
  const {view,playback}=fixture(),flow=playback.flowDisplay,op=flow.operations[0];
  flow.operations=Array.from({length:MAX_FLOW_OPERATIONS},()=>({...op}));assert.equal(checkedFlowDisplayForView(flow,view),flow);
  flow.operations.push({...op});assert.throws(()=>checkedFlowDisplayForView(flow,view));
  flow.operations=[{...op},{...op,startSeconds:2.05}];assert.throws(()=>checkedFlowDisplayForView(flow,view));
  flow.operations=[{...op},{...op,endSeconds:2.15}];assert.throws(()=>checkedFlowDisplayForView(flow,view));
  flow.startSeconds=flow.endSeconds;flow.operations=[{...op,startSeconds:flow.endSeconds,endSeconds:flow.endSeconds}];assert.throws(()=>checkedFlowDisplayForView(flow,view));
});
