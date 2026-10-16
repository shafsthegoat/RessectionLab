import assert from 'node:assert/strict';
import {test} from 'node:test';
import {checkedRehearsalEpisode,checkedRehearsalResponse,rehearsalViewerState} from '../rehearsal-data.ts';
import {checkedRehearsalState} from './rehearsal.ts';
const affine=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
const volume={caseHash:'patient-source-v1',caseId:'patient',frame:'RAS+',shape:[8,8,8],affine,mri:new Float32Array(512).fill(1),compartments:[{name:'target',color:'#aabbcc',mask:new Uint8Array(512).fill(1)}]};
const tools=[['aspirator','aspiration',2],['field_manager','field_management',5]].map(([toolId,mode,x])=>({toolId,mode,tipRasMm:[x,3,4],axis:[0,0,1],active:false,workingLengthMm:6,tipLengthMm:1,shaftRadiusMm:.2,tipRadiusMm:.4}));
const metrics=n=>({targetRemovedMm3:n,nonTargetRemovedMm3:0,residualTargetMm3:400-n,fieldBurden:0,vascularEvents:null,motorRisk:null,languageRisk:null});
function episode(){return {schema:'resectionlab.source-bound-rehearsal.v2',sourceBinding:{caseHash:volume.caseHash,shape:volume.shape,affine,frame:'RAS+'},initialStateId:'start',finalStateId:'end',tools,assumedExposureIndicesNative:[[0,0,0]],finalRemovedIndicesNative:[[1,1,1]],history:[{kind:'aspiration'},{kind:'field_management'}],replayFrames:[
  {frameIndex:0,actionIndex:-1,kind:'initial',elapsedSeconds:0,tools,stateBefore:'start',stateAfter:'start',removedIndicesNative:[],contactIndicesNative:[],fieldContactIndicesNative:[],metrics:metrics(0)},
  {frameIndex:1,actionIndex:0,kind:'aspiration',elapsedSeconds:1,tools:tools.map(t=>({...t,active:t.mode==='aspiration'})),stateBefore:'start',stateAfter:'middle',removedIndicesNative:[[1,1,1]],contactIndicesNative:[[1,1,2]],fieldContactIndicesNative:[],metrics:metrics(1)},
  {frameIndex:2,actionIndex:1,kind:'field_management',elapsedSeconds:2,tools:tools.map(t=>({...t,active:t.mode==='field_management'})),stateBefore:'middle',stateAfter:'end',removedIndicesNative:[],contactIndicesNative:[[1,1,2]],fieldContactIndicesNative:[[1,1,2]],metrics:metrics(1)},
]};}
test('native triples normalize z-fastest; source arrays stay immutable and field management preserves tissue',()=>{
  const raw=episode(),before=structuredClone(raw),source=volume.mri.slice(),mask=volume.compartments[0].mask.slice();
  const e=checkedRehearsalEpisode(volume,raw);assert.deepEqual(e.finalRemovedIndicesNative,[73]);assert.deepEqual(raw,before);
  const state=rehearsalViewerState(volume,e,2),display=checkedRehearsalState(volume,state);
  assert.equal(state.removedMask[73],1);assert.equal(state.assumedExposureMask[0],1);assert.equal(state.fieldContactMask[74],1);
  assert.equal(display.voidMask[74],0);assert.equal(display.effectMask[74],3);assert.equal(display.effectMask[0],2);assert.equal(display.effectMask[73],1);
  assert.equal(display.displays.length,2);assert.deepEqual(display.displays[0].shaftStart,[2,3,-2]);assert.deepEqual(display.displays[1].shaftStart,[5,3,-2]);
  assert.deepEqual(volume.mri,source);assert.deepEqual(volume.compartments[0].mask,mask);
});
test('scrubbing reconstructs cumulative irreversible effects without altering later frames',()=>{
  const e=checkedRehearsalEpisode(volume,episode());
  assert.equal(rehearsalViewerState(volume,e,2).removedMask[73],1);
  assert.equal(rehearsalViewerState(volume,e,0).removedMask[73],0);
  assert.equal(rehearsalViewerState(volume,e,1).removedMask[73],1);
  assert.equal(rehearsalViewerState(volume,e,1).fieldContactMask[74],0);
  assert.throws(()=>rehearsalViewerState(volume,e,3),/outside/);
});
test('source identity, native bounds, state ancestry, accounting and unknown-evidence guards reject tampering',()=>{
  for(const change of [e=>e.sourceBinding.caseHash='other',e=>e.sourceBinding.affine[0][3]=2,e=>e.sourceBinding.shape[0]=7,
    e=>e.replayFrames[1].stateBefore='stale',e=>e.replayFrames[1].frameIndex=5,e=>e.replayFrames[2].elapsedSeconds=.5,
    e=>e.replayFrames[1].removedIndicesNative=[[8,0,0]],e=>e.replayFrames[2].removedIndicesNative=[[1,1,1]],
    e=>e.replayFrames[1].removedIndicesNative=[[0,0,0]],e=>e.finalRemovedIndicesNative=[],e=>e.replayFrames[1].metrics.targetRemovedMm3=8,
    e=>e.replayFrames[2].metrics.motorRisk=.01,e=>e.replayFrames[1].tools[1].workingLengthMm=20,e=>delete e.replayFrames[1].tools[0].tipRasMm]){
    const e=structuredClone(episode());change(e);assert.throws(()=>checkedRehearsalEpisode(volume,e),/withheld/);
  }
});
test('branches must preserve the identical saved prefix and initial exposure',()=>{
  const live=episode();const branch=id=>({...structuredClone(live),episodeId:id});
  const raw={caseHash:volume.caseHash,episode:live,stateId:'end',suggestedActions:[],comparison:{snapshotId:'end',alternatives:[{id:'a',label:'A',episode:branch('a')},{id:'b',label:'B',episode:branch('b')}]}};
  assert.equal(checkedRehearsalResponse(volume,raw).comparison.alternatives.length,2);
  raw.comparison.alternatives[1].episode.replayFrames[1].tools[0].tipRasMm[0]=3;
  assert.throws(()=>checkedRehearsalResponse(volume,raw),/saved source state/);
});
test('display guard rejects source/frame mismatch, missing tool, exposed-removal overlap and understated capsule bounds',()=>{
  const state=rehearsalViewerState(volume,checkedRehearsalEpisode(volume,episode()),2);
  for(const change of [s=>s.caseHash='other',s=>s.frame='LPS+',s=>s.frameIndex=10,s=>s.tools.pop(),s=>s.removedMask[0]=1,s=>s.recordedEpisodeBounds.boundsRasMm[0][2]=0]){
    const s=structuredClone(state);change(s);assert.throws(()=>checkedRehearsalState(volume,s),/withheld/);
  }
});

test('a live continuation retains comparison branches from an earlier exact saved snapshot',()=>{
  const initial=episode(),live=structuredClone(initial);live.finalStateId='later';
  live.replayFrames.push({...structuredClone(live.replayFrames.at(-1)),frameIndex:3,actionIndex:2,kind:'inspect',elapsedSeconds:3,stateBefore:'end',stateAfter:'later',removedIndicesNative:[]});
  const raw={caseHash:volume.caseHash,episode:live,stateId:'later',suggestedActions:[],comparison:{snapshotId:'end',alternatives:[{id:'a',label:'A',episode:structuredClone(initial)},{id:'b',label:'B',episode:structuredClone(initial)}]}};
  assert.equal(checkedRehearsalResponse(volume,raw).stateId,'later');
  raw.comparison.snapshotId='unrelated';assert.throws(()=>checkedRehearsalResponse(volume,raw),/snapshot/);
});

test('old coagulation mechanism schemas cannot be relabeled as field management',()=>{
  const e=structuredClone(episode());e.schema='resectionlab.source-bound-rehearsal.v1';assert.throws(()=>checkedRehearsalEpisode(volume,e),/mechanism version/);
  e.schema='resectionlab.source-bound-rehearsal.v2';e.tools[1].mode='coagulation_field_management';assert.throws(()=>checkedRehearsalEpisode(volume,e),/unknown instrument behavior/);
});
