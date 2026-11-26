'use strict';

const path = require('node:path');
const {setTimeout:delay}=require('node:timers/promises');
const assert=require('node:assert/strict');

const epoch = () => performance.timeOrigin + performance.now();

function observeSidecar(main) {
  const { Sidecar } = require(path.join(path.dirname(main), 'sidecar.cjs'));
  const instances = new Set();
  const stages = [];
  const lifecycle = [];
  const request = Sidecar.prototype.request;
  const emit = Sidecar.prototype.emit;
  let peakPending = 0;
  Sidecar.prototype.request = function (operation, args, ...rest) {
    instances.add(this);
    const reply = request.call(this, operation, args, ...rest);
    peakPending = Math.max(peakPending, this.pending.size);
    return reply;
  };
  Sidecar.prototype.emit = function (event, ...args) {
    const row = args[0];
    if (event === 'rehearsalTiming') stages.push(row);
    if (event === 'event' && row.op === 'surgicalRehearsal')
      lifecycle.push({ id: row.id, event: row.event, atEpochMs: epoch() });
    return emit.call(this, event, ...args);
  };
  return {
    stages, lifecycle,
    peakPending: () => peakPending,
    pids: () => [...instances].map(sidecar => sidecar.child.pid),
    stop: async () => Promise.all([...instances].map(sidecar => sidecar.stop())),
  };
}

async function alignRendererClock(js) {
  const samples = [];
  for (let i = 0; i < 7; i++) {
    const before = epoch();
    const renderer = await js('performance.now()');
    const after = epoch();
    // The sandboxed Chromium renderer can quantize performance.now().
    const clockResolutionAllowanceMs = .2;
    samples.push({ low: before - renderer - clockResolutionAllowanceMs,
      high: after - renderer + clockResolutionAllowanceMs });
  }
  const low = Math.max(...samples.map(sample => sample.low));
  const high = Math.min(...samples.map(sample => sample.high));
  if(low>high)return {valid:false,samples};
  return {valid:true,low,high,offset:(low + high) / 2,maxErrorMs:(high - low) / 2,samples};
}

// Installed in the ordinary renderer, with cheap event listeners. Artifact
// writing and independent geometry/state comparisons run after the fence.
function installPerformanceEvidence() {
  const data = window.practicePerformance = {
    active: false, input: null, frames: [], fences: [], publications: [],
    work: [], gpuTimers: [], motions: [],
  };
  const capture = (type, rows, convert) => document.addEventListener(type, event => {
    if (!data.active) return;
    rows.push(convert(event.detail, performance.now(), event.target));
  });
  capture('anatomy-frame', data.frames, (d, noticed, canvas) => {
    const pane=canvas.closest('.rl-viewer')?.querySelector('[data-accepted-tools]');
    return {
    at: d.at, submittedAt:d.submittedAt, noticed, serial: d.serial, cpuMs: d.cpuMs, source: d.sourceCaseHash,
    stateId: d.surgicalStateId, displayIdentity: d.displayIdentity, surgicalFrame: d.surgicalFrame,
    modelGeneration: d.modelGeneration, sourceSurfaceGeneration: d.sourceSurfaceGeneration,
    sceneId: d.sceneId, preview: d.preview, interactive: d.interactive, camera: d.cameraState,
    operativeView:pane?.dataset.operativeView,interactionMode:pane?.dataset.practiceInteraction,
    sourceShape: d.sourceShape, resources: d.resources, drawStatistics: d.drawStatistics,
    field: d.fieldOptics, displayWork: d.displayWork, labelUploads: d.labelUploads,
    renderStages: d.renderStages, surgicalAppearance:d.surgicalAppearance, fieldWork:d.fieldWork,
    estimatedAllocation: d.estimatedAllocation, surfaceCache: d.surfaceCache, workers: d.workers,
    surfaceQuality:d.surfaceQuality,
    };
  });
  capture('anatomy-gpu', data.fences, (d, at) => ({
    at, serial: d.serial, source: d.sourceCaseHash, stateId: d.surgicalStateId,
    displayIdentity: d.displayIdentity, surgicalFrame: d.surgicalFrame,
    modelGeneration: d.modelGeneration, sourceSurfaceGeneration: d.sourceSurfaceGeneration,
    sceneId: d.sceneId, preview: d.preview, completed: d.completed, pollMs: d.completionPollMs,
    fenceSubmittedAt:d.fenceSubmittedAt,completionAt:d.completionAt,fenceAfterSubmitMs:d.fenceAfterSubmitMs,
  }));
  capture('anatomy-surgical-display', data.publications, (d, at) => ({
    at, serial: d.frameSerial, source: d.sourceCaseHash, stateId: d.surgicalStateId,
    displayIdentity:d.displayIdentity, modelGeneration:d.modelGeneration, preview:d.preview,
  }));
  capture('anatomy-work', data.work, (d, at) => ({ at, ...d }));
  capture('anatomy-gpu-time', data.gpuTimers, (d, at) => ({ at, ...d }));
  const receive = event => {
    if (!event.isTrusted || !data.active) return;
    if (event.type === 'pointermove') {
      if (data.motion)data.motions.push({at:performance.now(),x:event.clientX,y:event.clientY,buttons:event.buttons});
      return;
    }
    if (data.input?.expected!==event.type||data.input.trusted)return;
    const target = event.target.closest?.('button,select,input');
    const label = target?.getAttribute('aria-label') || target?.textContent.trim();
    if (label===data.input.label)
      data.input.trusted = { type: event.type, label, key: event.key, at: performance.now(), eventAt: event.timeStamp };
  };
  for (const type of ['click', 'keydown', 'pointermove'])
    document.addEventListener(type, receive, true);
}

function findCompletion(trace, input, source, stateId) {
  const publication = trace.publications.find(item =>
    item.at >= input.at && item.source === source && item.stateId === stateId && !item.preview);
  if (!publication) return null;
  const fence = trace.fences.find(item => item.serial === publication.serial &&
    item.stateId === stateId && item.source === source && item.completed &&
    item.at >= input.at && item.at <= publication.at && !item.preview);
  const frame = trace.frames.find(item => item.serial === publication.serial &&
    item.stateId === stateId && item.source === source &&
    item.at >= input.at && item.at <= publication.at && !item.preview);
  if(frame&&fence&&(frame.displayIdentity!==fence.displayIdentity||
    frame.modelGeneration!==fence.modelGeneration||frame.sceneId!==fence.sceneId))return null;
  if(frame&&publication.displayIdentity!==undefined&&frame.displayIdentity!==publication.displayIdentity)return null;
  if(frame&&publication.modelGeneration!==undefined&&frame.modelGeneration!==publication.modelGeneration)return null;
  return frame && fence ? { frame, fence, publication } : null;
}

function viewerFramebuffer() {
  const pane=document.querySelector('[data-accepted-tools]');
  const canvas=pane?.closest('.rl-viewer')?.querySelector('canvas.rl-viewer-canvas');
  if(!canvas)throw Error('Missing operative viewer canvas');
  const bounds=canvas.getBoundingClientRect();
  const gl=canvas.getContext('webgl2');
  if(!gl||gl.isContextLost())throw Error('Missing live operative WebGL drawing buffer');
  return {dpr:devicePixelRatio,canvasCss:{width:bounds.width,height:bounds.height},
    canvasPixels:{width:canvas.width,height:canvas.height},
    drawingBuffer:{width:gl.drawingBufferWidth,height:gl.drawingBufferHeight},
    drawingBufferPerCssPixel:{x:gl.drawingBufferWidth/bounds.width,y:gl.drawingBufferHeight/bounds.height}};
}

function beginGesture() {
  const data=window.practicePerformance;
  const start={frameIndex:data.frames.length,fenceIndex:data.fences.length,motionIndex:data.motions.length};
  const panel=document.querySelector('[data-practice-proposed-position]');
  const pose=()=>({at:performance.now(),proposed:JSON.parse(panel.dataset.practiceProposedPosition),
    accepted:JSON.parse(panel.dataset.practiceAcceptedPosition),stateId:panel.dataset.practiceStateId});
  data.poseChanges=[pose()];
  data.poseObserver=new MutationObserver(()=>data.poseChanges.push(pose()));
  data.poseObserver.observe(panel,{attributes:true,attributeFilter:['data-practice-proposed-position',
    'data-practice-accepted-position','data-practice-state-id']});
  data.motion=true;
  return start;
}

function endGesture(start,identity) {
  const data=window.practicePerformance;
  data.motion=false;
  data.poseObserver.disconnect();
  delete data.poseObserver;
  const poses=data.poseChanges;
  delete data.poseChanges;
  const moves=data.motions.slice(start.motionIndex).filter(row=>row.buttons===1);
  const first=moves[0]?.at,last=moves.at(-1)?.at;
  const frames=data.frames.slice(start.frameIndex).filter(row=>row.at>=first&&row.at<=last&&row.source===identity.source);
  const usesAccepted=frames.every(row=>row.stateId===identity.stateId);
  const serials=new Set(frames.map(row=>row.serial));
  const relevant=data.fences.slice(start.fenceIndex).filter(row=>row.source===identity.source&&serials.has(row.serial));
  const completed=relevant.filter(row=>row.completed&&row.stateId===identity.stateId);
  const intervals=rows=>rows.slice(1).map((row,i)=>row.at-rows[i].at);
  return { first,last,poses,receivedMoves:moves.length,usesAccepted,
    renderSerials:frames.map(row=>row.serial),completedSerials:completed.map(row=>row.serial),
    failedFenceSerials:relevant.filter(row=>!row.completed).map(row=>row.serial),
    sourceMismatches:relevant.filter(row=>row.stateId!==identity.stateId).length,
    deliveredIntervalsMs:intervals(moves),submissionIntervalsMs:intervals(frames),completedIntervalsMs:intervals(completed),
    submittedCameras:frames.map(row=>row.camera),
    internalSurfaceTargets:[...new Set(frames.filter(row=>row.surfaceQuality).map(row=>JSON.stringify(row.surfaceQuality)))]
      .map(row=>JSON.parse(row)),
    proposedTransformSerials:frames.filter(row=>row.interactionMode==='translate').map(row=>row.serial) };
}

async function sampleGesture({ js, send, point, identity, kind }) {
  const {x,y}=point;
  const invoke=(type,extra)=>send('dispatchMouseEvent',{type,x,y,...extra});
  await invoke('mouseMoved');
  const start=await js(`(${beginGesture.toString()})()`);
  await invoke('mousePressed',{button:'left',buttons:1,clickCount:1});
  const scheduledStart=performance.now(),moves=240,intervalMs=1000/60,outstandingLimit=8;
  const sent=[],skipped=[],pending=[],errors=[];
  let outstanding=0,peakOutstanding=0;
  for(let i=1;i<=moves;i++){
    const due=i*intervalMs;
    const until=scheduledStart+due-performance.now();
    if(until>0)await delay(until);
    if(outstanding===outstandingLimit){skipped.push(i);continue;}
    const moved=performance.now()-scheduledStart;
    sent.push({i,dueMs:due,dispatchMs:moved});
    outstanding++;
    peakOutstanding=Math.max(peakOutstanding,outstanding);
    pending.push(invoke('mouseMoved',{x:x+Math.round(Math.sin(i*Math.PI/30)*27),
      y:y+Math.round(Math.sin(i*Math.PI/40)*9),buttons:1})
      .catch(error=>errors.push(String(error))).finally(()=>{outstanding--;}));
  }
  await Promise.all(pending);
  await invoke('mouseReleased',{button:'left',buttons:0,clickCount:1});
  await delay(150);
  assert.deepEqual(errors,[]);
  const observed=await js(`(${endGesture.toString()})(${JSON.stringify(start)},${JSON.stringify(identity)})`);
  assert(observed.receivedMoves>1&&observed.completedSerials.length>1,'Need trusted motion and matching GPU observations');
  assert(observed.usesAccepted&&observed.sourceMismatches===0,'A gesture rendered the wrong accepted state');
  assert(observed.poses.every(row=>row.stateId===identity.stateId&&
    JSON.stringify(row.accepted)===JSON.stringify(observed.poses[0].accepted)));
  if(kind==='proposal'){
    assert(observed.poses.some(row=>JSON.stringify(row.proposed)!==JSON.stringify(row.accepted)),
      'Gesture must move the proposed instrument');
    assert(observed.proposedTransformSerials.length,'No frame submitted while the proposed transform was active');
  }
  else assert(new Set(observed.submittedCameras.map(camera=>JSON.stringify(camera))).size>1,
    'Gesture must move the actual submitted camera');
  return {kind,scheduledMoves:moves,intervalMs,sent,skipped,peakOutstanding,lostBeforeDelivery:sent.length-observed.receivedMoves,...observed};
}

module.exports = { alignRendererClock, epoch, findCompletion, installPerformanceEvidence, observeSidecar, sampleGesture, viewerFramebuffer };
