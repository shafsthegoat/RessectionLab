'use strict';

const assert = require('node:assert/strict');
const {summary} = require('./manual-practice-evidence.cjs');

async function benchmarkHold(ui) {
  const {choose,number,press,js,report,click}=ui;
  await choose('cusa');
  for(const [axis,n] of ['R','A','S'].entries())await number('Tool position '+n,[32,32,45][axis]);
  await press('Place tool');
  await js(`(()=>{
    const panel=document.querySelector('[data-practice-proposed-position]');
    const data=window.manualEvidence;
    data.holdProposals=[];
    data.proposalObserver=new MutationObserver(()=>{
      const proposed=panel.dataset.practiceProposedPosition;
      if(proposed!==panel.dataset.practiceAcceptedPosition)
        data.holdProposals.push({at:performance.timeOrigin+performance.now(),before:panel.dataset.practiceStateId,position:JSON.parse(proposed)});
    });
    data.proposalObserver.observe(panel,{attributes:true,attributeFilter:['data-practice-proposed-position']});
  })()`);
  await require('./complete-cranial-practice.cjs').holdSteps({...ui,label:'Hold to advance',count:12});
  const held=report.actions.filter(a=>a.heldInput);
  const {fences,proposals,pressAt,frameIntervals}=await js(`(()=>{
    const data=window.manualEvidence;data.proposalObserver.disconnect();delete data.proposalObserver;
    const press=data.inputs.findLast(e=>e.type==='pointerdown'&&e.label==='Hold to advance');
    const start=data.holdProposals[0]?.at??press.at;
    const end=data.fences.findLast(f=>f.stateId===${JSON.stringify(held.at(-1).result.witness.stateId)}).at;
    const frames=data.frames.filter(f=>performance.timeOrigin+f.at>=start && performance.timeOrigin+f.at<=end);
    return {fences:data.fences,proposals:data.holdProposals,pressAt:press.at,frameIntervals:frames.slice(1).map((f,i)=>f.at-frames[i].at)};
  })()`);
  for(const row of held){
    const arrival=row.result.backendResponseEpochMs-row.result.ipcMs;
    const fence=fences.find(f=>f.stateId===row.result.witness.stateId&&f.at>=arrival);
    assert(fence,'Held command missing accepted GPU completion');
    const proposal=proposals.find(p=>p.before===row.before&&p.position.every((n,i)=>n===row.result.witness.tools.left.position[i]));
    assert(proposal,'Held request missing its proposed pose in the visible instrument controls');
    row.proposalEpochMs=proposal.at;
    row.ipcArrivalEpochMs=arrival;row.gpuCompleteEpochMs=fence.at;
  }
  report.heldPerformance={samples:held.length,
    initialPressToFirstAcceptedMs:held[0].result.backendResponseEpochMs-pressAt,
    initialPressToFirstVisibleMs:held[0].gpuCompleteEpochMs-pressAt,
    proposedPoseToAcceptedMs:summary(held.map(r=>r.result.backendResponseEpochMs-r.proposalEpochMs)),
    proposedPoseToVisibleMs:summary(held.map(r=>r.gpuCompleteEpochMs-r.proposalEpochMs)),
    activeFrameIntervalMs:summary(frameIntervals),
    ipcToAcceptedMs:summary(held.map(r=>r.result.ipcMs)),
    ipcToVisibleMs:summary(held.map(r=>r.gpuCompleteEpochMs-r.ipcArrivalEpochMs)),
    visibleIntervalMs:summary(held.slice(1).map((r,i)=>r.gpuCompleteEpochMs-held[i].gpuCompleteEpochMs))};
  await click('Jog and commit');
  await press('Withdraw tool');
}

function checkCameras(frames,enclosed){
  const sequences=enclosed?[[0,3,9,14,21]]:[[1,2,3,4,5],[9,11,16,22,28]];
  for(const sequence of sequences){
    const first=frames.find(f=>f.index===sequence[0]).drawn.camera;
    for(const index of sequence){
      const frame=frames.find(f=>f.index===index);
      assert.equal(frame.drawn.fieldMm,12);
      assert.deepEqual(frame.drawn.camera,first);
    }
  }
}

function checkOutlet(sampled){
  assert(sampled[0].reservoirMm3>=1.49);assert.equal(sampled[0].freeField.fluidMm3,0);
  assert.equal(sampled[3].reservoirMm3,0);assert(sampled[3].freeField.fluidMm3>1.49);
  assert(sampled[9].metrics.fluidRemovedMm3>1.49);
  assert(sampled[9].freeField.fluidMm3<sampled[3].freeField.fluidMm3);
  assert(sampled[13].bleedingSources>0&&sampled[14].bleedingSources===0);
  assert(Math.abs(sampled[21].metrics.fluidAddedMm3-sampled[20].metrics.fluidAddedMm3-5)<1e-6);
  return Object.fromEntries([0,3,9,14,21].map(i=>[i,{stateId:sampled[i].stateId,
    reservoirMm3:sampled[i].reservoirMm3,freeMm3:sampled[i].freeField.fluidMm3,emitting:sampled[i].bleedingSources,
    retainedMm3:sampled[i].metrics.fluidPresentMm3,removedMm3:sampled[i].metrics.fluidRemovedMm3}]));
}

module.exports={benchmarkHold,checkCameras,checkOutlet};
