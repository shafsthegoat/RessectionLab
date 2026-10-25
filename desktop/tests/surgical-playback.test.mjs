import test from 'node:test';
import assert from 'node:assert/strict';
import {surgicalPlaybackDelay,checkedSurgicalPlayback,nextSurgicalPlaybackFrame,surgicalPlaybackPosition,updatesEditableSurgicalCursor,surgicalCursorCommand,SurgicalPlaybackClock} from '../src/surgical-playback.ts';
test('playback deadlines include preparation time and never add an interval to a late frame',()=>{
  const clock=new SurgicalPlaybackClock();
  assert.equal(clock.delay(2,0,4,1000),0);
  assert.equal(clock.delay(2,50,4,1080),0,'80ms fetching/displaying the predecessor already exceeds a 50ms interval');
  assert.equal(clock.delay(2.2,50,4,1120),0);
  assert(Math.abs(clock.delay(2.4,50,4,1125)-25)<1e-7,'a fast next frame retains the remaining recorded interval');
  assert.equal(clock.delay(2.4,100,2,2000),100,'speed changes anchor at the displayed observation');
  clock.reset();assert.equal(clock.delay(2.4,50,4,3000),50,'pause time is excluded on resume');
  assert.equal(clock.delay(NaN,50,4,3000),50,'unavailable historical timing retains explicit review cadence');
});
test('recorded playback uses authoritative elapsed effects, including partial and zero-time commands',()=>{
  const history=[{elapsedSeconds:.025},{elapsedSeconds:0},{elapsedSeconds:4.5}];
  assert.deepEqual(history.map((_,i)=>surgicalPlaybackDelay(history,i)),[{milliseconds:25,recorded:true},{milliseconds:0,recorded:true},{milliseconds:4500,recorded:true}]);
});
test('missing historical timing explicitly selects review cadence',()=>{
  for(const elapsedSeconds of [undefined,-1,NaN,Infinity])assert.deepEqual(surgicalPlaybackDelay([{elapsedSeconds}],0),{milliseconds:600,recorded:false});
});
test('playback speed scales recorded and historical timing without altering the authoritative duration',()=>{
  const history=[{elapsedSeconds:4}];
  assert.deepEqual(surgicalPlaybackDelay(history,0,4),{milliseconds:1000,recorded:true});
  assert.deepEqual(surgicalPlaybackDelay(history,0,.25),{milliseconds:16000,recorded:true});
  assert.deepEqual(surgicalPlaybackDelay([],0,2),{milliseconds:300,recorded:false});
  for(const speed of [0,-1,NaN,Infinity])assert.equal(surgicalPlaybackDelay(history,0,speed).milliseconds,4000);
  assert.equal(history[0].elapsedSeconds,4);
});
const response=(frameIndex=0)=>({branchId:'main',position:8,view:{metrics:{elapsedSeconds:[2,2.2,2.4,2.4][frameIndex]},slots:{left:{active:false},right:{active:false}}},playback:{readOnly:true,branchId:'main',position:2,actionIndex:2,frameIndex,frameCount:4,observedTimeSeconds:[2,2.2,2.4,2.4][frameIndex],actionElapsedSeconds:[0,.2,.4,.4][frameIndex],actionDurationSeconds:.4,isActionBoundary:frameIndex===0||frameIndex===3,effectiveActivation:{left:true,right:false},frames:[{frameIndex:0,observedTimeSeconds:2,kind:'initial'},{frameIndex:1,observedTimeSeconds:2.2,kind:'microstep'},{frameIndex:2,observedTimeSeconds:2.4,kind:'microstep'},{frameIndex:3,observedTimeSeconds:2.4,kind:'action_end'}]}});
test('microstep player schedules actual adjacent observations and finalized endpoints, then the next action without an action-length pause',()=>{
  const r=response(0),before=structuredClone(r);
  const {milliseconds,...selection}=nextSurgicalPlaybackFrame(r,8,2);
  assert.deepEqual(selection,{branchId:'main',position:2,frameIndex:1});assert.ok(Math.abs(milliseconds-100)<1e-7);
  const lastMicro=nextSurgicalPlaybackFrame(response(2),8,1);assert.equal(lastMicro.frameIndex,3);assert.equal(lastMicro.milliseconds,0,'finalization at the same clock is still displayed');
  assert.deepEqual(nextSurgicalPlaybackFrame(response(3),8,1),{branchId:'main',position:3,frameIndex:0,milliseconds:0});
  assert.equal(nextSurgicalPlaybackFrame({...response(3),playback:{...response(3).playback,position:7}},8,1),null);
  assert.deepEqual(nextSurgicalPlaybackFrame({branchId:'main',position:1},8,1),{branchId:'main',position:1,frameIndex:0,milliseconds:0});
  assert.equal(surgicalPlaybackPosition(r.playback,8),2);assert.equal(surgicalPlaybackPosition(response(3).playback,8),3);
  assert.deepEqual(r,before);
});
test('read-only observations never replace the editable expected-state cursor and cannot authorize instrument mutations',()=>{
  for(const command of ['preview','seekFrame'])assert.equal(updatesEditableSurgicalCursor(command),false);
  for(const command of ['seek','step','fullCase','branch','inspect'])assert.equal(updatesEditableSurgicalCursor(command),true);
  for(const command of ['seek','seekFrame','selectBranch','inspect'])assert.equal(surgicalCursorCommand(command),true);
  for(const command of ['step','fullCase','branch','reset','plan','create'])assert.equal(surgicalCursorCommand(command),false);
});
test('source-bound frame admission rejects substituted observations, broken clocks and missing finalized boundaries',()=>{
  const request={branchId:'main',position:2,frameIndex:1};
  assert.equal(checkedSurgicalPlayback(response(1),request).frameIndex,1);
  for(const change of [r=>r.playback.branchId='other',r=>r.playback.position=4,r=>r.playback.frameIndex=0,r=>r.playback.frameCount=5,r=>r.playback.frames[1].observedTimeSeconds=1,r=>r.playback.actionElapsedSeconds=.3,r=>r.playback.actionDurationSeconds=10,r=>r.view.metrics.elapsedSeconds=9,r=>r.playback.effectiveActivation.left='true']){
    const r=response(1);change(r);assert.throws(()=>checkedSurgicalPlayback(r,request),/Recorded microstep/);
  }
  const final=response(3);final.playback.isActionBoundary=false;
  assert.throws(()=>checkedSurgicalPlayback(final,{...request,frameIndex:3}),/Recorded microstep/);
});
