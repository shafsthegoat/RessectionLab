import test from 'node:test';
import assert from 'node:assert/strict';
import {InteractionSession,InteractionBudget,SliceScrollAccumulator} from './interactionBudget.ts';
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
