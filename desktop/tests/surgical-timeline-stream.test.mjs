import test from 'node:test';
import assert from 'node:assert/strict';
import {SurgicalTimelineStream} from '../src/surgical-timeline-stream.ts';

const full=()=>({caseHash:'case',branchId:'main',timelineToken:'archive/history',timeline:[{action:{operation:'inspect'}},{action:{operation:'apply'}}],fullCase:{status:'stopped'},view:{history:[],scene:{sceneId:'source-scene'},catalog:[{id:'cusa'}],protection:{contractId:'guards'}}});
const compact=()=>({caseHash:'case',branchId:'main',timelineToken:'archive/history',timelineOmitted:true,view:{history:[]},playback:{position:1,frameIndex:2,frameCount:3,readOnly:true}});
test('compact read-only frames recover the exact resident timeline and finalized prefix',()=>{
  const stream=new SurgicalTimelineStream(),first=full();assert.equal(stream.hydrate(first),first);
  assert.equal(stream.token('case','main'),first.timelineToken);
  const wire=compact(),display=stream.hydrate(wire);
  assert.equal(display.timeline,first.timeline);assert.equal(display.fullCase,first.fullCase);
  assert.equal(display.view.scene,first.view.scene);assert.equal(display.view.catalog,first.view.catalog);assert.equal(display.view.protection,first.view.protection);
  assert.deepEqual(display.view.history,first.timeline);assert.deepEqual(wire.view.history,[]);
  assert.deepEqual(stream.hydrate({...wire,playback:{...wire.playback,frameIndex:0}}).view.history,first.timeline.slice(0,1));
});
test('changed source, branch, archive generation, missing resident data and editable responses fail closed',()=>{
  const stream=new SurgicalTimelineStream();assert.throws(()=>stream.hydrate(compact()),/timeline changed/);
  stream.hydrate(full());
  for(const change of [{caseHash:'other'},{branchId:'other'},{timelineToken:'new'},{playback:undefined}])
    assert.throws(()=>stream.hydrate({...compact(),...change}),/timeline|read-only/);
  assert.equal(stream.token('other','main'),undefined);assert.equal(stream.token('case','other'),undefined);
  stream.clear();assert.throws(()=>stream.hydrate(compact()),/timeline changed/);
});
test('ordinary accepted responses invalidate metadata even when action history is unchanged',()=>{
  const stream=new SurgicalTimelineStream();stream.hydrate(full());
  stream.hydrate({caseHash:'case',branchId:'main',fullCase:{status:'stopped',stopReason:'budget'},view:{history:[]}});
  assert.equal(stream.token('case','main'),undefined);
  assert.throws(()=>stream.hydrate(compact()),/timeline changed/);
});
const hashA='a'.repeat(64),hashB='b'.repeat(64);
const fieldFull=()=>({...full(),playback:compact().playback,viewFieldTokens:{flapComponents:hashA},view:{...full().view,flapComponents:[{cells:[1,2,3],separable:null}]}});
const fieldCompact=(hash=hashA)=>({...compact(),viewFieldTokens:{flapComponents:hash},viewFieldsOmitted:['flapComponents']});
test('authenticated heavy fields hydrate by exact source/branch/timeline/hash and update at a finalized action boundary',()=>{
  const stream=new SurgicalTimelineStream(),first=fieldFull(),before=structuredClone(first);stream.hydrate(first);
  assert.deepEqual(stream.fieldTokens('case','main',first.timelineToken),{flapComponents:hashA});
  for(const [source,branch,token]of [['other','main',first.timelineToken],['case','other',first.timelineToken],['case','main','new']])
    assert.equal(stream.fieldTokens(source,branch,token),undefined);
  const wire=fieldCompact(),display=stream.hydrate(wire);
  assert.equal(display.view.flapComponents,first.view.flapComponents);assert.equal(Object.hasOwn(wire.view,'flapComponents'),false);assert.deepEqual(first,before);
  const changed=[{cells:[2,3],separable:true}],boundary={...compact(),viewFieldTokens:{flapComponents:hashB},view:{...compact().view,flapComponents:changed}};
  assert.equal(stream.hydrate(boundary).view.flapComponents,changed);
  assert.equal(stream.hydrate(fieldCompact(hashB)).view.flapComponents,changed);
  assert.deepEqual(stream.fieldTokens('case','main',first.timelineToken),{flapComponents:hashB});
  assert.throws(()=>stream.hydrate(fieldCompact(hashA)),/view fields changed/);
  assert.equal(stream.token('case','main'),undefined,'a stale field clears all acknowledgments for a full refresh');
});
test('missing hashes, implicit field omission, unknown fields and editable field references fail closed',()=>{
  for(const change of [
    {viewFieldTokens:undefined},
    {viewFieldTokens:undefined,viewFieldsOmitted:undefined},
    {viewFieldTokens:{flapComponents:hashB}},
    {viewFieldTokens:{flapComponents:'not-a-hash'}},
    {viewFieldsOmitted:['scene']},
    {viewFieldsOmitted:['flapComponents','flapComponents']},
    {view:{flapComponents:[]}},
    {timelineToken:'changed'},
    {playback:{...compact().playback,readOnly:false}},
  ]){
    const stream=new SurgicalTimelineStream();stream.hydrate(fieldFull());
    assert.throws(()=>stream.hydrate({...fieldCompact(),...change}),/view fields|timeline/);
    assert.equal(stream.token('case','main'),undefined);
  }
  const stream=new SurgicalTimelineStream();
  assert.throws(()=>stream.hydrate({...fieldFull(),view:{...full().view}}),/view fields/,'full hash without an explicit field value cannot establish residency');
  assert.throws(()=>stream.hydrate({...fieldCompact(),timelineToken:undefined}),/source-bound/);
  stream.hydrate(fieldFull());stream.hydrate(full());
  assert.equal(stream.fieldTokens('case','main',full().timelineToken),undefined,'absence of explicit authenticated field metadata invalidates its acknowledgment');
});
