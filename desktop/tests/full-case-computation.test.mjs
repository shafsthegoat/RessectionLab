import test from 'node:test';
import assert from 'node:assert/strict';
import {FullCaseComputation} from '../src/full-case-computation.ts';
const response=(status='running',scenario='primary')=>({fullCase:{status,scenario}});
test('autonomous computation bounds slices and waits for a new admitted display before resuming',()=>{
  const c=new FullCaseComputation(),initial={},next=response();c.start('primary');
  const first=c.claim(initial,false,true);assert.equal(first.maxActions,2);
  assert.equal(c.claim(initial,false,true),null,'one request in flight');
  c.settled(first.epoch,true);
  assert.equal(c.claim(initial,false,true),null,'old admitted display cannot launch a second slice');
  assert.equal(c.claim(next,true,true),null,'display preparation is still pending');
  assert.equal(c.claim(next,false,true).scenario,'primary');
});
test('pause during an in-flight slice prevents further slices; stale completion cannot restart computation',()=>{
  const c=new FullCaseComputation();c.start('primary');const first=c.claim({},false,true);
  c.pause();c.settled(first.epoch,true);
  assert.equal(c.scenario,null);assert.equal(c.claim(response(),false,true),null);
  c.start('vascular');const second=c.claim({},false,true);
  c.settled(first.epoch,false);assert.equal(c.scenario,'vascular','stale failure does not stop newly authorized computation');
  c.settled(second.epoch,true);assert.equal(c.claim(response('running','vascular'),false,true).scenario,'vascular');
});
test('completion, stops, rejection, request failure and leaving the workspace stop autonomous computation',()=>{
  for(const next of [response('complete'),response('stopped'),response('running','vascular'),{...response(),outcome:{status:'rejected'}}]){
    const c=new FullCaseComputation();c.start('primary');const first=c.claim({},false,true);c.settled(first.epoch,true);
    assert.equal(c.claim(next,false,true),null);assert.equal(c.scenario,null);
  }
  const c=new FullCaseComputation();c.start('primary');const first=c.claim({},false,true);c.settled(first.epoch,false);assert.equal(c.scenario,null);
  c.start('primary');assert.equal(c.claim({},false,false),null);assert.equal(c.scenario,null);
});
