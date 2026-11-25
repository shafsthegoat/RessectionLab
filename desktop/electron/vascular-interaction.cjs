'use strict';

// Continuous-orbit timing for hidden anatomy captures. The page script drags
// the 3D pane once per animation frame, then waits for the view to settle.
const assert=require('node:assert/strict');

function checkedInteraction(value){
  assert(value&&typeof value==='object'&&!Array.isArray(value),'Interaction timing needs an options object');
  const {frames,quietMs=700,timeoutMs=20000,...rest}=value;
  assert.deepEqual(Object.keys(rest),[],'Unknown interaction timing option');
  assert(Number.isInteger(frames)&&frames>=30&&frames<=600,'Interaction timing needs 30 to 600 drag frames');
  assert(Number.isFinite(quietMs)&&quietMs>=200&&quietMs<=5000,'Invalid quiet interval');
  assert(Number.isFinite(timeoutMs)&&timeoutMs>=1000&&timeoutMs<=120000,'Invalid settle deadline');
  return {frames,quietMs,timeoutMs};
}

function interactionScript(options){
  const {frames,quietMs,timeoutMs}=checkedInteraction(options);
  return `new Promise(resolve=>{
    const pane=document.querySelector('.rl-viewer-anatomy-image'),box=pane.getBoundingClientRect();
    // OrbitControls requires pointer capture, which a synthetic pointer cannot obtain.
    pane.setPointerCapture=()=>{};pane.releasePointerCapture=()=>{};
    const x=Math.round(box.x+box.width*.42),y=Math.round(box.y+box.height*.48);
    const send=(type,dx,dy)=>pane.dispatchEvent(new PointerEvent(type,{bubbles:true,clientX:x+dx,clientY:y+dy,button:0,
      buttons:type==='pointerup'?0:1,pointerId:92,pointerType:'mouse',isPrimary:true}));
    const raw={started:performance.now(),released:null,inputs:[],frames:[],gpu:{},fences:{},refined:null,ended:null,timedOut:false};
    let quiet,deadline,moved=0;
    const frame=e=>{const d=e.detail,q=d.scanQuality;
      raw.frames.push({serial:d.serial,at:d.at,cpuMs:d.cpuMs,interactive:d.interactive,longEdge:d.longEdge,
        width:q?.width??null,height:q?.height??null,samples:q?.samples??null,settled:q?.settled??null,opticsPending:q?.opticsPending===true});};
    const gpu=e=>{if(e.detail.completed)raw.gpu[e.detail.serial]=e.detail.milliseconds;};
    const fence=e=>{raw.fences[e.detail.serial]={ms:e.detail.completionPollMs,at:performance.now(),completed:e.detail.completed};check();};
    const refined=e=>{if(raw.released!==null&&!raw.refined)raw.refined={at:performance.now(),milliseconds:e.detail.milliseconds};};
    const finish=timedOut=>{
      clearTimeout(quiet);clearTimeout(deadline);
      for(const [name,handler] of [['anatomy-frame',frame],['anatomy-gpu-time',gpu],['anatomy-gpu',fence],['anatomy-refined',refined]])
        document.removeEventListener(name,handler);
      delete pane.setPointerCapture;delete pane.releasePointerCapture;
      raw.ended=performance.now();raw.timedOut=timedOut;resolve(raw);
    };
    const rested=()=>raw.frames.some(row=>row.at>raw.released&&!row.interactive&&!row.opticsPending&&raw.fences[row.serial]?.completed);
    function check(){
      if(raw.released===null)return;
      if(raw.frames.some(row=>row.at>raw.released&&row.settled===true&&raw.fences[row.serial]?.completed)){
        // Leave time for the asynchronous GPU timer results of the last frames.
        clearTimeout(quiet);quiet=setTimeout(()=>finish(false),250);return;
      }
      clearTimeout(quiet);quiet=setTimeout(()=>{if(rested())finish(false);else check();},${quietMs});
    }
    document.addEventListener('anatomy-frame',frame);document.addEventListener('anatomy-gpu-time',gpu);
    document.addEventListener('anatomy-gpu',fence);document.addEventListener('anatomy-refined',refined);
    send('pointerdown',0,0);
    const step=()=>{
      if(moved<${frames}){
        moved++;raw.inputs.push(performance.now());send('pointermove',moved*2,-Math.round(moved*.6));requestAnimationFrame(step);
      }else{
        send('pointerup',moved*2,-Math.round(moved*.6));raw.released=performance.now();
        deadline=setTimeout(()=>finish(true),${timeoutMs});check();
      }
    };
    requestAnimationFrame(step);
  })`;
}

function percentile(values,fraction){
  const sorted=values.filter(Number.isFinite).sort((a,b)=>a-b);
  if(!sorted.length)return null;
  const position=(sorted.length-1)*fraction,low=Math.floor(position),high=Math.ceil(position);
  return sorted[low]+(sorted[high]-sorted[low])*(position-low);
}
const distribution=values=>({count:values.filter(Number.isFinite).length,p50:percentile(values,.5),p95:percentile(values,.95),
  max:values.some(Number.isFinite)?Math.max(...values.filter(Number.isFinite)):null});

function summarizeInteraction(raw){
  assert(raw&&Array.isArray(raw.frames)&&Number.isFinite(raw.released),'Missing interaction timing samples');
  const frames=[...raw.frames].sort((a,b)=>a.serial-b.serial);
  const drag=frames.filter(row=>row.interactive&&row.at>=raw.started&&row.at<=raw.released);
  assert(drag.length>=2,'The drag produced fewer than two interactive frames');
  const intervals=drag.slice(1).map((row,index)=>row.at-drag[index].at);
  const gpu=drag.map(row=>raw.gpu[row.serial]).filter(Number.isFinite);
  const cost=drag.filter(row=>Number.isFinite(raw.gpu[row.serial])).map(row=>row.cpuMs+raw.gpu[row.serial]);
  const fences=drag.map(row=>raw.fences[row.serial]?.ms).filter(Number.isFinite);
  const inputs=raw.inputs.slice(1).map((at,index)=>at-raw.inputs[index]);
  const after=frames.filter(row=>row.at>raw.released&&!row.interactive&&raw.fences[row.serial]?.completed);
  const declared=after.find(row=>row.settled===true);
  // A renderer that declares convergence defines the settled frame. Otherwise
  // the last refinement before the view went quiet is the settled frame.
  const settledFrame=declared??(raw.timedOut?undefined:after.filter(row=>!row.opticsPending).at(-1));
  return {
    dragFrames:drag.length,requestedInputs:raw.inputs.length,framesPerInput:drag.length/raw.inputs.length,
    // One input is dispatched per animation frame, so this is the display refresh interval.
    inputIntervalMs:distribution(inputs),
    frameIntervalMs:{...distribution(intervals),measurement:'start of one drawn interactive frame to the next'},
    // A diagnostic. The renderer waits for each GPU fence, so cost understates the delivered interval.
    frameCostMs:{...distribution(cost),measurement:'CPU submission plus asynchronous GPU elapsed time'},
    gpuMs:distribution(gpu),gpuTimerCoverage:gpu.length/drag.length,
    fencePollMs:{...distribution(fences),measurement:'submission to fence poll, an upper bound'},
    dragLongEdge:{min:Math.min(...drag.map(row=>row.longEdge)),max:Math.max(...drag.map(row=>row.longEdge))},
    settle:settledFrame?{milliseconds:raw.fences[settledFrame.serial].at-raw.released,samples:settledFrame.samples,
      width:settledFrame.width,height:settledFrame.height,basis:declared?'renderer-declared':'last-refinement-before-quiet',
      refinedEventMs:raw.refined?raw.refined.at-raw.released:null}:null,
    timedOut:raw.timedOut,
  };
}

module.exports={checkedInteraction,interactionScript,summarizeInteraction,percentile};
