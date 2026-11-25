import test from 'node:test';
import assert from 'node:assert/strict';
import {InteractionSession,InteractionBudget,RefreshBudget,SliceScrollAccumulator,FenceWindow,nextTask,pollSoon} from './interactionBudget.ts';
import {SliceFrameCache} from './sliceFrameCache.ts';
import * as THREE from 'three';

test('wheel bursts remain interactive until 150ms after the last event',t=>{
  t.mock.timers.enable({apis:['setTimeout']});
  const changes=[],s=new InteractionSession(active=>changes.push(active));
  s.start();s.end();t.mock.timers.tick(100);assert.equal(s.active,true);
  s.start();s.end();t.mock.timers.tick(149);assert.equal(s.active,true);
  t.mock.timers.tick(1);assert.equal(s.active,false);assert.deepEqual(changes,[true,false]);
});
test('new movement cancels refinement; focus loss and disposal clean pending timers',t=>{
  t.mock.timers.enable({apis:['setTimeout']});
  const changes=[],s=new InteractionSession(active=>changes.push(active));
  s.start();s.end();s.start();t.mock.timers.tick(200);assert.equal(s.active,true);
  s.finish();assert.deepEqual(changes,[true,false]);
  s.start();s.end();s.dispose();t.mock.timers.tick(200);assert.deepEqual(changes,[true,false,true]);
});
test('adaptive quality is bounded, reacts to slow GPU completion, and has recovery hysteresis',()=>{
  const b=new InteractionBudget();assert.equal(b.longEdge,360);
  for(let i=0;i<4;i++)b.sample(40);assert.equal(b.longEdge,240);
  for(let i=0;i<100;i++)b.sample(40);assert.equal(b.longEdge,240);
  for(let i=0;i<31;i++)b.sample(10);assert.equal(b.longEdge,240);
  b.sample(10);assert.equal(b.longEdge,360);
  for(let i=0;i<200;i++)b.sample(10);assert.equal(b.longEdge,720);
});
test('path-traced motion targets one display refresh and leaves the standard budget alone',()=>{
  const b=new RefreshBudget(),standard=new InteractionBudget();
  assert.equal(b.longEdge,360);
  // The refresh is the least animation-frame step seen lately.
  for(let i=0;i<10;i++)b.frame(1000+i*8.33);
  assert.ok(Math.abs(b.refreshMs-8.33)<.01);
  b.frame(5000);assert.ok(Math.abs(b.refreshMs-8.33)<.01,'a pause is not a refresh');
  // Two refreshes per frame never lowers the standard budget, but does lower this one.
  for(let i=0;i<8;i++)standard.sample(16.7);
  b.sample(16.7);b.sample(16.7);
  assert.equal(standard.longEdge,360);assert.equal(b.longEdge,320);
  // Slightly late frames are followed up at once; they are not misses.
  for(let i=0;i<200;i++){b.sample(i%3?8.3:11.5);b.gpuTime(6);}
  assert.equal(b.longEdge,320,'no room under clock scaling: stays put');
  // A long run of hits with wide GPU room raises one step at a time.
  const fresh=new RefreshBudget();
  for(let i=0;i<10;i++)fresh.frame(i*8.33);
  fresh.sample(20);fresh.sample(20);assert.equal(fresh.longEdge,320);
  for(let i=0;i<239;i++){fresh.sample(8.3);fresh.gpuTime(1.5);}
  assert.equal(fresh.longEdge,320,'the failed tier needs a longer run');
  fresh.sample(8.3);assert.equal(fresh.longEdge,360);
  for(let i=0;i<63;i++){fresh.sample(8.3);fresh.gpuTime(1.5);}
  assert.equal(fresh.longEdge,360);
  fresh.sample(8.3);fresh.gpuTime(1.5);assert.equal(fresh.longEdge,400);
  for(let i=0;i<64;i++){b.gpuTime(1.5);b.sample(8.3);}
  assert.equal(b.longEdge,360);
  for(let i=0;i<64;i++){b.gpuTime(1.5);b.sample(8.3);}
  assert.equal(b.longEdge,400);
  // One miss in eight is tolerated; a second lowers the resolution at once.
  b.sample(20);for(let i=0;i<6;i++)b.sample(8.3);assert.equal(b.longEdge,400);
  b.sample(20);assert.equal(b.longEdge,360);
  for(let i=0;i<100;i++)b.sample(40);
  assert.equal(b.longEdge,200,'bounded below');
  b.sample(Number.NaN);b.gpuTime(Number.NaN);assert.equal(b.longEdge,200);
});
test('without GPU timer queries path-traced motion recovers resolution from fence evidence after a longer run',()=>{
  const b=new RefreshBudget();
  for(let i=0;i<10;i++)b.frame(i*16.7);
  // An untried tier: twice the usual run of hits.
  for(let i=0;i<127;i++)b.sample(16);
  assert.equal(b.longEdge,360);
  b.sample(16);assert.equal(b.longEdge,400);
  b.sample(40);b.sample(40);assert.equal(b.longEdge,360);
  // The tier that failed takes twice the longer run; a miss restarts the run.
  for(let i=0;i<300;i++)b.sample(16);
  b.sample(40);
  for(let i=0;i<479;i++)b.sample(16);
  assert.equal(b.longEdge,360);
  b.sample(16);assert.equal(b.longEdge,400);
  // With timer evidence the timer rule applies instead: no room, no raise.
  for(let i=0;i<600;i++){b.gpuTime(14);b.sample(16);}
  assert.equal(b.longEdge,400);
});
test('next tasks run in order after the current task, without timer clamping',async()=>{
  const order=[];
  nextTask(()=>order.push('a'));nextTask(()=>order.push('b'));
  order.push('sync');
  await new Promise(resolve=>nextTask(resolve));
  assert.deepEqual(order,['sync','a','b']);
});
test('a next task that throws is reported and does not drop the tasks queued with it',async()=>{
  const order=[],reported=[],error=console.error;
  console.error=value=>reported.push(value);
  try{
    nextTask(()=>order.push('a'));nextTask(()=>{throw Error('poll failed');});nextTask(()=>order.push('c'));
    await new Promise(resolve=>nextTask(resolve));
    // A later queue still runs: the channel was not left stalled.
    await new Promise(resolve=>nextTask(()=>{order.push('d');resolve();}));
  }finally{console.error=error;}
  assert.deepEqual(order,['a','c','d']);
  assert.deepEqual(reported.map(value=>value.message),['poll failed']);
});
test('fence polls wait about a millisecond each instead of spinning through tasks',async()=>{
  // A poll that reschedules itself until a deadline, as a pending GPU fence does.
  const count=async schedule=>{
    let polls=0;const end=performance.now()+40;
    await new Promise(resolve=>{const poll=()=>{polls++;if(performance.now()<end)schedule(poll);else resolve();};schedule(poll);});
    return polls;
  };
  const spinning=await count(nextTask),paced=await count(pollSoon);
  assert.ok(paced>=3&&paced<=80,`${paced} polls in 40 ms`);
  assert.ok(spinning>10*paced,`back-to-back tasks poll ${spinning} times`);
});
test('slice scroll batching accumulates all events, respects physical bounds, and clears the batch',()=>{
  const s=new SliceScrollAccumulator();s.add(2,.5);s.add(2,.5);s.add(2,2.5);s.add(1,-.7);
  assert.deepEqual(s.flush([0,2,1],[-10,-10,-10],[10,10,4]),[0,1.3,4]);
  assert.deepEqual(s.flush([0,1.3,4],[-10,-10,-10],[10,10,4]),[0,1.3,4]);
});
test('unchanged slices composite cached pixels; invalidation and size changes render once and disposal releases targets',()=>{
  const cache=new SliceFrameCache(),source=new THREE.Scene(),camera=new THREE.Camera();
  let sourceDraws=0,composites=0,restored=0,target,disposed=0;
  const renderer={getPixelRatio:()=>2,setRenderTarget(t){target=t;if(t&&!t.observed){t.observed=true;t.addEventListener('dispose',()=>disposed++);}},setScissorTest(){},setViewport(x,y,w,h){assert.equal(w*2,target.width);assert.equal(h*2,target.height);},clear(){},render(s){if(s===source){sourceDraws++;assert(target)}else{composites++;assert.equal(target,null)}}};
  const draw=(w=100,rev=1)=>cache.render(renderer,source,camera,w,80,rev,()=>restored++);
  assert.equal(draw(),true);assert.equal(draw(),false);assert.equal(draw(),false);
  assert.equal(draw(100,2),true);assert.equal(draw(120,2),true);
  assert.equal(sourceDraws,3);assert.equal(composites,5);assert.equal(restored,3);
  const before=disposed;cache.dispose();assert.equal(disposed,before+1);
});

test('a settling fence is polled without a pause only around its predicted completion',()=>{
  const window=new FenceWindow();
  assert.equal(window.eager(5),false,'nothing is predicted before a sample has finished');
  window.completed(20);
  assert.deepEqual([10,16.9,17,20,26,26.1,60].map(ms=>window.eager(ms)),[false,false,true,true,true,false,false]);
  // The prediction follows later samples and ignores unusable times.
  window.completed(30);window.completed(NaN);window.completed(0);
  assert.equal(window.eager(19.9),false);assert.equal(window.eager(20),true);assert.equal(window.eager(29.1),false);
  window.reset();assert.equal(window.eager(20),false);
});
