import assert from 'node:assert/strict';
import {test} from 'node:test';
import {SurgicalRequestQueue} from '../src/surgical-request-queue.ts';

const deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};};
test('latest queued seek replaces obsolete display work and retains semantic FIFO barriers',async()=>{
  const gate=deferred(),calls=[],published=[];
  const queue=new SurgicalRequestQueue(async(item,current)=>{
    calls.push(item);
    if(item==='seek0')await gate.promise;
    if(current())published.push(item);
    return current();
  },item=>item.startsWith('seek'));
  const promises=[queue.request('seek0'),queue.request('seek1'),queue.request('seek2'),queue.request('STOP'),queue.request('deactivate'),queue.request('exchange'),queue.request('seek3'),queue.request('seek4')];
  gate.resolve();
  const results=await Promise.all(promises);
  assert.deepEqual(calls,['seek0','seek2','STOP','deactivate','exchange','seek4']);
  assert.deepEqual(published,['seek2','STOP','deactivate','exchange','seek4']);
  assert.deepEqual(results,[false,false,true,true,true,true,false,true]);
});
test('case invalidation withholds in-flight results and queued commands; new case waits for the old request',async()=>{
  const gate=deferred(),calls=[],published=[];
  const queue=new SurgicalRequestQueue(async(item,current)=>{calls.push(item);if(item==='old')await gate.promise;if(current())published.push(item);return true;},()=>false);
  const old=queue.request('old'),queued=queue.request('old-command');
  queue.invalidate();
  const next=queue.request('new');
  gate.resolve();
  assert.deepEqual(await Promise.all([old,queued,next]),[false,false,true]);
  assert.deepEqual(calls,['old','new']);assert.deepEqual(published,['new']);
});
test('request failures release the queue and overflow is explicit',async()=>{
  const gate=deferred(),calls=[];
  const queue=new SurgicalRequestQueue(async item=>{calls.push(item);if(item==='slow')await gate.promise;if(item==='fail')throw Error('backend failed');return true;},()=>false,1);
  const slow=queue.request('slow'),failure=queue.request('fail');
  const rejected=assert.rejects(queue.request('overflow'),/queue is full/);
  const failed=assert.rejects(failure,/backend failed/);
  gate.resolve();await Promise.all([slow,rejected,failed]);
  assert.equal(await queue.request('next'),true);assert.deepEqual(calls,['slow','fail','next']);
});
