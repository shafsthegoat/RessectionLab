import test from 'node:test';
import assert from 'node:assert/strict';
import {SurgicalReplayPrefetch,MAX_REPLAY_PREFETCH_BYTES} from '../src/surgical-replay-prefetch.ts';
const selection=frameIndex=>({branchId:'main',position:2,frameIndex});
const response=frameIndex=>({timelineOmitted:true,branchId:'main',view:{stateId:`frame${frameIndex}`},playback:{frameIndex}});
const prefixResponse=frameIndex=>({...response(frameIndex),timelineOmitted:undefined,timeline:[{status:'accepted'}],fullCase:null,view:{stateId:`frame${frameIndex}`,scene:{shape:[1,1,1]},catalog:[]},verification:{status:'reconstructed-prefix',readOnly:true}});
const flush=()=>new Promise(resolve=>setImmediate(resolve));
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};

test('one receipt overlaps preparation without fetching or replacing any later observation',async()=>{
  const calls=[],lane=new SurgicalReplayPrefetch(async s=>{calls.push(s.frameIndex);return response(s.frameIndex);});lane.start();
  assert.equal((await lane.take(selection(0))).view.stateId,'frame0');
  lane.prefetch(selection(1));lane.prefetch(selection(2));await flush();
  assert.deepEqual(calls,[0,1]);assert(lane.bufferedBytes>0);assert(lane.bufferedBytes<=MAX_REPLAY_PREFETCH_BYTES);
  assert.equal((await lane.take(selection(1))).view.stateId,'frame1');assert.equal(lane.bufferedBytes,0);
  lane.prefetch(selection(2));await flush();assert.equal((await lane.take(selection(2))).view.stateId,'frame2');assert.deepEqual(calls,[0,1,2]);
});
test('wire budget uses UTF-8 bytes; oversized and full contexts release storage then refetch serially without skipping',async()=>{
  for(const wire of [{...response(1),note:'é'.repeat(70)},{...response(1),timelineOmitted:false},{...response(1),view:{...response(1).view,scene:{}}}]){
    const calls=[],lane=new SurgicalReplayPrefetch(async s=>{calls.push(s.frameIndex);return wire;},220);lane.start();
    lane.prefetch(selection(1));await flush();assert.equal(lane.bufferedBytes,0);
    assert.equal(await lane.take(selection(1)),wire);assert.deepEqual(calls,[1,1]);
  }
  assert.throws(()=>new SurgicalReplayPrefetch(async()=>response(0),MAX_REPLAY_PREFETCH_BYTES+1),/budget/);
});
test('one full reconstructed-prefix receipt is retained without a second fetch or a second slot',async()=>{
  const calls=[],wire=prefixResponse(1),lane=new SurgicalReplayPrefetch(async s=>{calls.push(s.frameIndex);return wire;});lane.start();
  lane.prefetch(selection(1));lane.prefetch(selection(2));await flush();
  assert.equal(lane.bufferedBytes,new TextEncoder().encode(JSON.stringify(wire)).byteLength);
  assert.deepEqual(calls,[1]);assert.equal(await lane.take(selection(1)),wire);
  assert.equal(lane.bufferedBytes,0);assert.deepEqual(calls,[1]);
});
test('full receipt admission requires both explicit reconstructed-prefix status and true read-only authority',async()=>{
  for(const verification of [undefined,{status:'unverified-import',readOnly:true},{status:'reconstructed-prefix',readOnly:false},{status:'reconstructed-prefix',readOnly:'true'}]){
    const wire={...prefixResponse(1),verification},calls=[],lane=new SurgicalReplayPrefetch(async s=>{calls.push(s.frameIndex);return wire;});lane.start();
    lane.prefetch(selection(1));await flush();assert.equal(lane.bufferedBytes,0);assert.equal(await lane.take(selection(1)),wire);assert.deepEqual(calls,[1,1]);
  }
});
test('full prefix receipts obey the same serialized UTF-8 byte budget and oversize serial fallback',async()=>{
  const wire={...prefixResponse(1),note:'é'.repeat(70)},bytes=new TextEncoder().encode(JSON.stringify(wire)).byteLength;
  for(const maxBytes of [bytes,bytes-1]){
    const calls=[],lane=new SurgicalReplayPrefetch(async s=>{calls.push(s.frameIndex);return wire;},maxBytes);lane.start();lane.prefetch(selection(1));await flush();
    assert.equal(lane.bufferedBytes,maxBytes===bytes?bytes:0);assert.equal(await lane.take(selection(1)),wire);assert.equal(calls.length,maxBytes===bytes?1:2);
  }
});
test('exact eight MiB boundary is admitted and one byte beyond stays serial',async()=>{
  for(const receipt of [response(1),prefixResponse(1)]){
    const overhead=new TextEncoder().encode(JSON.stringify({...receipt,padding:''})).byteLength;
    for(const extra of [0,1]){
      const wire={...receipt,padding:'x'.repeat(MAX_REPLAY_PREFETCH_BYTES-overhead+extra)},calls=[];
      const lane=new SurgicalReplayPrefetch(async()=>{calls.push(1);return wire;});lane.start();lane.prefetch(selection(1));await flush();
      assert.equal(lane.bufferedBytes,extra?0:MAX_REPLAY_PREFETCH_BYTES);assert.equal(await lane.take(selection(1)),wire);assert.equal(calls.length,extra?2:1);
    }
  }
});
test('pause releases a full prefix receipt and a new generation must fetch it again',async()=>{
  const calls=[],lane=new SurgicalReplayPrefetch(async s=>{calls.push(s.frameIndex);return prefixResponse(s.frameIndex);});lane.start();lane.prefetch(selection(1));await flush();assert(lane.bufferedBytes>0);
  lane.stop();assert.equal(lane.bufferedBytes,0);assert.equal(await lane.take(selection(1)),null);
  lane.start();assert.equal((await lane.take(selection(1))).view.stateId,'frame1');assert.deepEqual(calls,[1,1]);
});
test('pause drops ready data and invalidates in-flight receipt without admitting it',async()=>{
  const pending=deferred(),calls=[],lane=new SurgicalReplayPrefetch(async s=>{calls.push(s.frameIndex);return calls.length===1?pending.promise:response(s.frameIndex);});
  lane.start();lane.prefetch(selection(1));await flush();lane.stop();assert.equal(lane.bufferedBytes,0);
  assert.equal(await lane.take(selection(1)),null);pending.resolve(response(1));await flush();assert.equal(lane.bufferedBytes,0);
  lane.start();assert.equal((await lane.take(selection(1))).view.stateId,'frame1');assert.deepEqual(calls,[1,1]);
  lane.prefetch(selection(2));await flush();assert(lane.bufferedBytes>0);lane.stop();assert.equal(lane.bufferedBytes,0);
});
test('manual barrier and new generations await old transport and never overlap calls',async()=>{
  const pending=deferred(),calls=[],lane=new SurgicalReplayPrefetch(async s=>{calls.push(s.frameIndex);return calls.length===1?pending.promise:response(s.frameIndex);});
  lane.start();const old=lane.take(selection(1));await flush();const barrier=lane.barrier();lane.start();const next=lane.take(selection(0));await flush();assert.deepEqual(calls,[1]);
  pending.resolve(response(1));await barrier;assert.equal(await old,null);assert.equal((await next).view.stateId,'frame0');assert.deepEqual(calls,[1,0]);
});
test('prefetch failures wait for demand; mismatched claims and concurrent claims fail without coalescing',async()=>{
  const failure=Error('frame unavailable'),lane=new SurgicalReplayPrefetch(async()=>{throw failure;});lane.start();lane.prefetch(selection(1));await flush();
  assert.equal(lane.bufferedBytes,0);await assert.rejects(lane.take(selection(1)),e=>e===failure);
  lane.prefetch(selection(2));await assert.rejects(lane.take(selection(3)),/exact next/);
  const pending=deferred(),other=new SurgicalReplayPrefetch(async()=>pending.promise);other.start();const first=other.take(selection(0));await assert.rejects(other.take(selection(1)),/already/);pending.resolve(response(0));assert.equal((await first).view.stateId,'frame0');
});
test('scheduler timing and incidental fields never cross the replay cursor transport boundary',async()=>{
  const calls=[],lane=new SurgicalReplayPrefetch(async s=>{calls.push(s);assert.deepEqual(Object.keys(s),['branchId','position','frameIndex']);return response(s.frameIndex);});lane.start();
  await lane.take({...selection(0),milliseconds:0});lane.prefetch({...selection(1),milliseconds:50,incidental:'scheduler-only'});await flush();
  assert.equal((await lane.take({...selection(1),milliseconds:50})).view.stateId,'frame1');
  assert.deepEqual(calls,[selection(0),selection(1)]);
});
