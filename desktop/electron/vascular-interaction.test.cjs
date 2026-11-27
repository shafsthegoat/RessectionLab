'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const vm=require('node:vm');
const {checkedInteraction,interactionScript,summarizeInteraction,percentile}=require('./vascular-interaction.cjs');

const drag=(count,step=16)=>Array.from({length:count},(_,index)=>({serial:index+1,at:1000+index*step,cpuMs:1,interactive:true,longEdge:540,
  width:540,height:480,samples:1,settled:null,opticsPending:false}));
const raw=(extra={})=>{
  const frames=drag(40);
  const rest=[1,2,3,4].map(sample=>({serial:40+sample,at:1700+sample*60,cpuMs:1,interactive:false,longEdge:1024,width:1024,height:900,
    samples:sample,settled:null,opticsPending:false}));
  const all=[...frames,...rest];
  return {started:990,released:1650,inputs:frames.map(row=>row.at-2),frames:all,
    gpu:Object.fromEntries(all.map(row=>[row.serial,row.interactive?6:50])),
    fences:Object.fromEntries(all.map(row=>[row.serial,{ms:row.interactive?9:55,at:row.at+(row.interactive?9:55),completed:true}])),
    refined:{at:2000,milliseconds:300},ended:2800,timedOut:false,...extra};
};

test('options are bounded and reject unknown fields',()=>{
  assert.deepEqual(checkedInteraction({frames:120}),{frames:120,quietMs:700,timeoutMs:20000});
  for(const value of [null,[],{frames:5},{frames:120,quietMs:10},{frames:120,timeoutMs:5},{frames:120,speed:2},{frames:1.5}])
    assert.throws(()=>checkedInteraction(value));
});

test('percentiles interpolate and ignore missing samples',()=>{
  assert.equal(percentile([1,2,3,4,5],.5),3);
  assert.equal(percentile([10,20],.95),19.5);
  assert.equal(percentile([NaN,undefined,7],.95),7);
  assert.equal(percentile([],.95),null);
});

test('drag frame time and settle time come from submitted frames',()=>{
  const summary=summarizeInteraction(raw());
  assert.equal(summary.dragFrames,40);
  assert.equal(summary.frameIntervalMs.p95,16);
  assert.equal(summary.frameCostMs.p95,7);
  assert.equal(summary.gpuMs.count,40);
  assert.equal(summary.gpuTimerCoverage,1);
  assert.equal(summary.framesPerInput,1);
  assert.equal(summary.inputIntervalMs.p50,16);
  assert.deepEqual(summary.dragLongEdge,{min:540,max:540});
  assert.equal(summary.settle.basis,'last-refinement-before-quiet');
  assert.equal(summary.settle.samples,4);
  assert.equal(summary.settle.milliseconds,1700+240+55-1650);
  assert.equal(summary.settle.refinedEventMs,350);
});

test('a renderer-declared settled frame defines settle time',()=>{
  const samples=raw();
  samples.frames[41].settled=true;
  const summary=summarizeInteraction(samples);
  assert.equal(summary.settle.basis,'renderer-declared');
  assert.equal(summary.settle.samples,2);
  assert.equal(summary.settle.milliseconds,1700+120+55-1650);
});

test('a missed deadline or missing GPU completion leaves settle unknown',()=>{
  assert.equal(summarizeInteraction(raw({timedOut:true})).settle,null);
  const incomplete=raw();
  for(const serial of [41,42,43,44])incomplete.fences[serial].completed=false;
  assert.equal(summarizeInteraction(incomplete).settle,null);
  assert.throws(()=>summarizeInteraction({...raw(),frames:drag(1)}),/fewer than two/);
});

test('the page script drags once per animation frame and restores the pane',async()=>{
  const listeners=new Map(),sent=[];
  const pane={getBoundingClientRect:()=>({x:100,y:50,width:800,height:600}),dispatchEvent:event=>sent.push(event)};
  let now=0,serial=0;
  const emit=(name,detail)=>listeners.get(name)?.({detail});
  const context={performance:{now:()=>now},document:{querySelector:()=>pane,
    addEventListener:(name,handler)=>listeners.set(name,handler),removeEventListener:name=>listeners.delete(name)},
    PointerEvent:class{constructor(type,init){Object.assign(this,{type},init);}},
    requestAnimationFrame:callback=>setImmediate(()=>{
      now+=16;
      if(sent.at(-1)?.type==='pointermove'){
        serial++;emit('anatomy-frame',{serial,at:now,cpuMs:1,interactive:true,longEdge:540,scanQuality:{width:540,height:400,samples:1}});
        emit('anatomy-gpu-time',{serial,completed:true,milliseconds:5});emit('anatomy-gpu',{serial,completed:true,completionPollMs:8});
      }
      callback();
    }),setTimeout,clearTimeout,Promise,Math};
  const pending=vm.runInNewContext(interactionScript({frames:30,quietMs:200,timeoutMs:5000}),context);
  await new Promise(resolve=>setTimeout(resolve,60));
  now+=100;serial++;
  emit('anatomy-frame',{serial,at:now,cpuMs:1,interactive:false,longEdge:1024,scanQuality:{width:1024,height:800,samples:4,settled:true}});
  emit('anatomy-gpu',{serial,completed:true,completionPollMs:40});
  const result=await pending;
  assert.deepEqual([sent[0].type,sent.at(-1).type],['pointerdown','pointerup']);
  assert.equal(sent.filter(event=>event.type==='pointermove').length,30);
  assert.equal(result.timedOut,false);
  assert.equal(listeners.size,0);
  assert.equal('setPointerCapture' in pane,false);
  assert.equal(summarizeInteraction(result).settle.basis,'renderer-declared');
});

test('dropped GPU timer samples lower the reported coverage',()=>{
  const samples=raw();
  for(let serial=1;serial<=10;serial++)delete samples.gpu[serial];
  const summary=summarizeInteraction(samples);
  assert.equal(summary.gpuTimerCoverage,.75);
  assert.equal(summary.frameCostMs.count,30);
  assert.equal(summary.frameIntervalMs.count,39);
});
