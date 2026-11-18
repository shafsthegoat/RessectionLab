'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const {movieIntervals}=require('./study-engine-recording.cjs');
test('movie uses observed durations, simultaneous boundary updates and a terminal hold',()=>{
  const frames=[{time:6},{time:6},{time:6.125},{time:6.505},{time:6.505},{time:6.55000000001}];
  assert.deepEqual(movieIntervals(frames),[
    {index:1,milliseconds:0,duration:125},{index:2,milliseconds:125,duration:380},
    {index:4,milliseconds:505,duration:45},{index:5,milliseconds:550,duration:1000}]);
  assert.throws(()=>movieIntervals([{time:1},{time:.9}]),/out of order/);
  assert.throws(()=>movieIntervals([{time:NaN}]),/out of order/);
  assert.throws(()=>movieIntervals([]),/No observed/);
});

test('export refuses a substituted boundary, a skipped observation and a premature finish',async()=>{
  const {StudyReplayRecorder}=require('./study-engine-recording.cjs');
  const recorder=new StudyReplayRecorder();
  recorder.session={id:'capture',caseHash:'case',branchId:'branch',samples:[]};
  const response={caseHash:'case',branchId:'branch',timeline:[{action:{operation:'apply'},status:'partial',stateBefore:'start',stateAfter:'end'}],
    branches:[{id:'branch',steps:1}],playback:{readOnly:true,position:0,frameIndex:0,frameCount:3,
      observedTimeSeconds:0,frames:[{kind:'action_start'},{kind:'microstep'},{kind:'action_boundary'}]},
    view:{stateId:'substituted',metrics:{targetRemovedMm3:0,nonTargetRemovedMm3:0},slots:{},contact:[],removed:[]}};
  assert.throws(()=>recorder.observe(response),/differs from the archived action/);
  assert.equal(recorder.session.receipt,null);
  response.view.stateId='interior';response.playback.frameIndex=1;recorder.observe(response);
  await assert.rejects(recorder.command({id:'capture',command:'capture',stateId:'interior'},null),/expected next engine frame/);
  await assert.rejects(recorder.command({id:'capture',command:'finish'},null),/did not reach/);
});
