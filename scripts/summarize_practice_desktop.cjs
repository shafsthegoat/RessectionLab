'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { createHash } = require('node:crypto');
const { digest } = require('../desktop/electron/manual-practice-evidence.cjs');

function distribution(values) {
  const sorted=values.filter(Number.isFinite).sort((a,b)=>a-b);
  if (!sorted.length) return {n:0,p50:null,p95:null,max:null};
  const percentile=q=>sorted[Math.ceil(sorted.length*q)-1];
  const middle=Math.floor(sorted.length/2);
  const p50=sorted.length%2?sorted[middle]:(sorted[middle-1]+sorted[middle])/2;
  return {n:sorted.length,p50,p95:percentile(.95),max:sorted.at(-1),
    ...(sorted.length>=300?{p99:percentile(.99)}:{})};
}

function witnessWithoutEpisodeId(witness) {
  const {branchId,...identity}=witness;
  return identity;
}

function exactActionRows(report) {
  return report.actions.map(({before,label,result})=>({
    before,label,request:result.requested,status:result.status,completedFraction:result.completedFraction,
    reason:result.reason,reasonCode:result.reasonCode,eventKinds:result.eventKinds,cusaAllocatedMm3:result.cusaAllocatedMm3,
    cusaRemovedCells:result.cusaRemovedCells,witness:witnessWithoutEpisodeId(result.witness),
  }));
}

function validatePair(reference,actual) {
  for (const key of ['originalSha256','actionStreamSha256','machine'])
    assert.deepEqual(actual[key],reference[key],key);
  assert.deepEqual(actual.performance.viewport,reference.performance.viewport,'Viewport changed');
  assert.deepEqual(actual.parent,reference.parent,'Verified parent differs between variants');
  assert.deepEqual(exactActionRows(actual),exactActionRows(reference),'Accepted physical states or ordered requests changed');
  assert.deepEqual(actual.actions.map(row=>row.display.camera),reference.actions.map(row=>row.display.camera),'Optical camera changed');
  assert.equal(actual.savedActionsVerified,reference.savedActionsVerified);
  return digest(exactActionRows(actual));
}

function verifyTimingRow(row) {
  const {performance: measured,result}=row;
  assert.equal(measured?.expectedStateId,row.before);
  if(result.status==='rejected'){
    assert.equal(measured.inputToGpuMs,null,'Rejection has no new accepted GPU frame');
    return;
  }
  assert(measured.inputToGpuMs>=0);
  assert.equal(measured.publication.stateId,result.witness.stateId);
  for (const observation of [measured.fence,measured.frame]) {
    assert.equal(observation.serial,measured.publication.serial);
    assert.equal(observation.stateId,result.witness.stateId);
    assert.equal(observation.source,result.witness.caseHash);
    assert(!observation.preview);
  }
}

const groupedDistributions = rows => Object.fromEntries(
  Object.entries(Object.groupBy(rows, row=>row.key)).map(([key,group])=>[key,distribution(group.map(row=>row.value))]));

function sumCounts(rows) {
  const counts={};
  for(const row of rows)for(const [key,value] of Object.entries(row??{})){
    if(Number.isFinite(value))counts[key]=(counts[key]??0)+value;
    else if(value&&typeof value==='object')counts[key]=sumCounts([counts[key],value]);
  }
  return counts;
}

function observations(reports) {
  const actions=reports.flatMap(report=>report.actions);
  const committed=actions.filter(row=>row.result.status!=='rejected');
  const stages=reports.flatMap(report=>{
    const byRequest=new Map(report.actions.map(row=>[row.performance.requestId,row]));
    return report.performance.stages.filter(stage=>byRequest.has(stage.requestId)).map(stage=>{
      const action=byRequest.get(stage.requestId);
      assert.equal(stage.caseHash,action.result.witness.caseHash);
      assert.equal(stage.expectedStateId,action.before);
      if (stage.responseStateId!==undefined) assert.equal(stage.responseStateId,action.result.witness.stateId);
      return stage;
    });
  });
  const duringActions=(report,row)=>row.at>=report.performance.manualTraceStart&&row.at<=report.performance.manualTraceFinish;
  const frames=reports.flatMap(report=>report.performance.trace.frames.filter(row=>duringActions(report,row)));
  const prep=reports.flatMap(report=>report.performance.trace.work.filter(row=>
    duringActions(report,row)&&row.kind==='modeled-surface-preparation'));
  const committedTimings=committed.map(row=>row.performance);
  const haveFinish=committedTimings.filter(t=>t.frame!==undefined);
  const firstWetActions=reports.map(report=>report.actions.find(row=>
    row.result.witness.freeField.fluidCells>(report.parent.freeField.fluidCells??0))).filter(Boolean);
  const timingRows=stages.flatMap(stage=>Object.entries(stage.durationsNs??{}).map(([key,value])=>
    ({key:stage.process+'.'+key,value:value/1e6})));
  const groupBuckets=Object.groupBy(committed,row=>row.result.requested.operation);
  return {
    runs:reports.length,
    committed:committed.length,
    rejected:actions.length-committed.length,
    inputToMatchingGpuMs:distribution(committedTimings.map(t=>t.inputToGpuMs)),
    rendererEventToListenerMs:distribution(committedTimings.map(t=>t.rendererEventToListenerMs)),
    inputToMainHandlerMs:distribution(committedTimings.map(t=>t.inputToMainMs)),
    mainHandlerToResultMs:distribution(committed.map(t=>t.result.ipcMs)),
    mainResultToRenderStartMs:distribution(committedTimings.map(t=>t.mainResponseToRenderStartMs)),
    renderStartToMatchingFenceMs:distribution(haveFinish.map(t=>t.fence.at-t.frame.at)),
    afterSubmissionFencePollMs:distribution(haveFinish.map(t=>t.fence.fenceAfterSubmitMs)),
    renderCpuSubmissionMs:distribution(haveFinish.map(t=>t.frame.cpuMs)),
    matchingGpuExecutionMs:distribution(reports.flatMap(report=>{
      const serials=new Set(report.actions.map(row=>row.performance.frame?.serial).filter(Number.isFinite));
      return report.performance.trace.gpuTimers.filter(row=>serials.has(row.serial)&&row.completed).map(row=>row.milliseconds);
    })),
    inputToAcceptancePublicationMs:distribution(committedTimings.map(t=>t.inputToAcceptancePublicationMs)),
    byOperation:Object.fromEntries(Object.entries(groupBuckets).map(([key,group])=>
      [key,distribution(group.map(row=>row.performance.inputToGpuMs))])),
    longestActions:[...committed].sort((a,b)=>b.performance.inputToGpuMs-a.performance.inputToGpuMs).slice(0,5).map(row=>({
      operation:row.result.requested.operation,position:row.result.witness.position,
      stateId:row.result.witness.stateId,requestId:row.performance.requestId,inputToMatchingGpuMs:row.performance.inputToGpuMs,
      mainHandlerToResultMs:row.result.ipcMs,cpuSubmissionMs:row.performance.frame.cpuMs,
    })),
    stageMs:groupedDistributions(timingRows),
    backendResponseBytes:distribution(stages.filter(row=>row.process==='python').map(row=>row.responseBytes)),
    stageCoverage:Object.fromEntries(Object.entries(Object.groupBy(stages,row=>row.process)).map(([key,rows])=>[key,rows.length])),
    nativeExecution:sumCounts(stages.map(row=>row.deformationKernels)),
    nativeBindings:sumCounts(stages.map(row=>row.nativeBindings)),
    meshWorkerCpuMs:distribution(prep.map(row=>row.preparationMs)),
    meshRequestToResultMs:distribution(prep.map(row=>row.requestToResultMs)),
    meshMainThreadBuildMs:distribution(prep.map(row=>row.mainThreadBuildMs)),
    displayNormalsMs:distribution(prep.map(row=>row.displayNormalsMs)),
    renderDetailMs:groupedDistributions(haveFinish.flatMap(row=>Object.entries(row.frame.renderStages??{})
      .filter(([key,value])=>key.endsWith('Ms')&&Number.isFinite(value)).map(([key,value])=>({key,value})))),
    meshVertexCounts:[...new Set(prep.map(row=>row.vertices).filter(Number.isFinite))],
    meshTriangleCounts:[...new Set(prep.map(row=>row.triangles).filter(Number.isFinite))],
    submittedTriangles:distribution(frames.map(row=>row.drawStatistics?.triangles)),
    firstMicroscopeCpuSubmissionMs:distribution(reports.map(report=>report.performance.firstMicroscope?.cpuMs)),
    launchToReadyMicroscopeMs:distribution(reports.map(report=>report.coldVerifiedBoundaryMs)),
    firstActionWithAddedFreeFluid:{
      cpuSubmissionMs:distribution(firstWetActions.map(row=>row.performance.frame?.cpuMs)),
      inputToMatchingGpuMs:distribution(firstWetActions.map(row=>row.performance.inputToGpuMs)),
      positions:[...new Set(firstWetActions.map(row=>row.result.witness.position))],
    },
    jogSubmissionIntervalMs:distribution(reports.flatMap(report=>report.performance.jog?.submissionIntervalsMs??[])),
    jogMatchingGpuIntervalMs:distribution(reports.flatMap(report=>report.performance.jog?.matchingGpuIntervalsMs??[])),
    peaks:reports.map(report=>({
      sidecarPending:report.performance.peakSidecarRequests,
      memoryWorkingSetMiB:Math.max(...report.memory.map(rows=>rows.reduce((sum,row)=>sum+row.workingSetKiB/1024,0))),
      knownGraphicsBytes:Math.max(...report.performance.trace.frames.map(frame=>frame.estimatedAllocation?.totalKnownBytes??0)),
    })),
  };
}

function compareFieldStills(directory,baseline,candidate) {
  const pairs=baseline.data.performance.stills.map((before,index)=>[before,candidate.data.performance.stills[index]]);
  assert.equal(pairs.length,candidate.data.performance.stills.length);
  const rgb=(entry,still)=>execFileSync('ffmpeg',['-hide_banner','-loglevel','error','-i',
    path.join(directory,path.dirname(entry.report),still.file),'-frames:v','1','-pix_fmt','rgb24','-f','rawvideo','pipe:1'],
    {maxBuffer:32*1024*1024});
  return {scope:'Color channels inside settled microscope pane, inset eight CSS pixels. Outer controls and recording overlays excluded.',
    frames:pairs.map(([before,after])=>{
      for(const key of ['name','cssCrop','pixels','capturePixelsPerCss','framebuffer','camera','caseHash','stateId','position','physicalDigest'])
        assert.deepEqual(after[key],before[key],'Matched image '+before.name+' '+key);
      const a=rgb(baseline,before),b=rgb(candidate,after);
      const pixels=before.pixels.width*before.pixels.height;
      assert.equal(a.length,pixels*3);
      assert.equal(a.length,b.length);
      let changedPixels=0,sum=0,max=0;
      for(let i=0;i<a.length;i+=3){
        const red=Math.abs(a[i]-b[i]),green=Math.abs(a[i+1]-b[i+1]),blue=Math.abs(a[i+2]-b[i+2]);
        if(red||green||blue)changedPixels++;
        sum+=red+green+blue;
        max=Math.max(max,red,green,blue);
      }
      const files=[baseline,candidate].map(entry=>path.join(path.dirname(entry.report),before.file));
      return {...before,files,exactPixels:changedPixels===0,changedPixels,pixelsCompared:pixels,
        changedFraction:changedPixels/pixels,meanAbsoluteChannelDifference:sum/a.length,maxAbsoluteChannelDifference:max,
        rgbSha256:[a,b].map(bytes=>createHash('sha256').update(bytes).digest('hex'))};
    })};
}

function cadence(report) {
  assert(!report.performance.timingEnabled&&!report.movie&&report.captures.length===0);
  return {framebuffer:report.performance.highCadenceFramebuffer,...Object.fromEntries(['proposal','camera'].map(kind=>{
    const takes=report.performance.highCadence.filter(row=>row.kind===kind&&!row.warmup);
    assert.equal(takes.length,3);
    return [kind,{
      repeats:takes.length,deadlineMs:takes[0].intervalMs,
      deliveredInputIntervalMs:distribution(takes.flatMap(row=>row.deliveredIntervalsMs)),
      submittedFrameIntervalMs:distribution(takes.flatMap(row=>row.submissionIntervalsMs)),
      matchingGpuIntervalMs:distribution(takes.flatMap(row=>row.completedIntervalsMs)),
      gestureRuns:takes.map(row=>({scheduled:row.scheduledMoves,delivered:row.receivedMoves,skippedByDriver:row.skipped.length,
        lostBeforeDelivery:row.lostBeforeDelivery,submissionCount:row.renderSerials.length,completedCount:row.completedSerials.length,
        failedFenceCount:row.failedFenceSerials.length,peakOutstanding:row.peakOutstanding,poseUpdates:row.poses.length,
        completionRateHz:1000*row.completedIntervalsMs.length/row.completedIntervalsMs.reduce((sum,value)=>sum+value,0)})),
    }];
  }))};
}

function summarize(directory) {
  const read=file=>JSON.parse(fs.readFileSync(path.join(directory,file),'utf8'));
  const plan=read('plan.json');
  const data=plan.reports.map(row=>({...row,data:read(row.report)}));
  assert(data.every(row=>row.data.passed&&!row.data.windowShown));
  const cases={};
  for(const {id} of plan.cases){
    const rows=data.filter(row=>row.case===id);
    const reference=rows.find(row=>row.phase!=='cold');
    const measured=rows.filter(row=>row.phase==='paired');
    const rounds=[...new Set(measured.map(row=>row.round))];
    const names=['baseline','candidate'];
    for(const row of rows.filter(row=>row.phase!=='cold')){
      for(const action of row.data.actions) verifyTimingRow(action);
      validatePair(reference.data,row.data);
    }
    const result=cases[id]={workloadParent:reference.data.parent,idDigest:digest(exactActionRows(reference.data))};
    if(measured.length){
      assert(rounds.length>=5,'Insufficient measured pairs');
      assert.equal(measured.length,rounds.length*2);
      for(const {data:report} of measured){
        assert(!report.performance.timingEnabled&&report.performance.stages.length===0,'Separate profiling from latency samples');
        assert(!report.movie&&report.captures.length===0,'Separate compositor capture from latency samples');
      }
      result.pairs=rounds.map(round=>{
        const baseline=measured.find(row=>row.round===round&&row.variant==='baseline');
        const candidate=measured.find(row=>row.round===round&&row.variant==='candidate');
        const rowsFor=entry=>entry.data.actions.map(row=>row.performance.inputToGpuMs);
        const first=measured.find(row=>row.round===round);
        assert.equal(first.variant,round%2===1?'baseline':'candidate','Paired run order changed');
        return {round,first:first.variant,baseline:distribution(rowsFor(baseline)),candidate:distribution(rowsFor(candidate)),
          acceptedIdentityDigest:validatePair(baseline.data,candidate.data)};
      });
      result.measurements=Object.fromEntries(names.map(variant=>[variant,observations(measured.filter(row=>row.variant===variant).map(row=>row.data))]));
    }
    if(rows.some(row=>row.phase==='diagnostic'))
      result.separatelyInstrumented=Object.fromEntries(names.map(variant=>{
        const trace=rows.filter(row=>row.variant===variant&&row.phase==='diagnostic');
        assert(trace.every(row=>row.data.performance.timingEnabled&&!row.data.movie));
        const summary=observations(trace.map(row=>row.data));
        const expected=summary.committed+summary.rejected;
        assert.equal(summary.stageCoverage.python,expected,'Python stage coverage differs from actual action count');
        assert.equal(summary.stageCoverage.electron,expected,'Electron stage coverage differs from actual action count');
        return [variant,summary];
      }));
    if(rows.some(row=>row.phase==='interaction-cadence')){
      result.highCadence=Object.fromEntries(names.map(variant=>[variant,cadence(
        rows.find(row=>row.variant===variant&&row.phase==='interaction-cadence').data)]));
      assert.deepEqual(result.highCadence.baseline.framebuffer,result.highCadence.candidate.framebuffer);
    }
    if(rows.some(row=>row.phase==='cold'))
      result.freshProcessReopen=Object.fromEntries(names.map(variant=>{
        const cold=rows.find(row=>row.variant===variant&&row.phase==='cold').data;
        return [variant,{verifiedBoundaryMs:cold.coldVerifiedBoundaryMs,canvasMs:cold.coldCanvasMs,
          stateIds:cold.replayed,historyRows:cold.coldHistory.length}];
      }));
    if(rows.some(row=>row.data.performance.stills))
      result.matchedMicroscopeStills=compareFieldStills(directory,
        rows.find(row=>row.variant==='baseline'&&row.data.performance.stills),
        rows.find(row=>row.variant==='candidate'&&row.data.performance.stills));
    if(rows.some(row=>row.phase==='recording'))
      result.separateRecordings=Object.fromEntries(names.map(variant=>{
        const recording=rows.find(row=>row.variant===variant&&row.phase==='recording');
        const offsets=recording.data.movie.captureOffsetsMs;
        return [variant,{file:path.join(path.dirname(recording.report),recording.data.movie.filename),
          durationSeconds:recording.data.movie.wallSeconds,frames:recording.data.movie.frames,
          spool:recording.data.recordingSpool??null,
          sampleSpacingMs:distribution(offsets.slice(1).map((at,i)=>at-offsets[i])),
          matchingCompositorSampleMs:distribution(recording.data.actions.map(row=>row.inputToFirstRecordedFrameMs)),
          immediateReadbackUpperBoundMs:distribution(recording.data.actions.map(row=>row.inputToCapturedViewportUpperBoundMs))}];
      }));
  }
  const definitions={
    inputToMatchingGpuMs:'Trusted DOM listener receipt to polling the successful WebGL fence for the serial that published the accepted state.',
    stageMs:'Inclusive monotonic durations from a separate instrumented run. Nested stages may overlap.',
    submittedFrameIntervalMs:'Spacing of frame submissions to the operative viewer during delivered native pointer motion.',
    matchingGpuIntervalMs:'Spacing of completed WebGL fence polls for the recorded submitted serials.',
    matchingCompositorSampleMs:'First separately recorded hidden compositor sample labeled with the matching accepted state. Capture cadence is about 10 Hz.',
  };
  return {schema:plan.schema,definitions,cases};
}

if(require.main===module) console.log(JSON.stringify(summarize(path.resolve(process.argv[2])),null,2));
module.exports={distribution,exactActionRows,summarize,validatePair,verifyTimingRow};
