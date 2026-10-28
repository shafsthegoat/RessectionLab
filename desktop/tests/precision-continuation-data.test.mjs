import test from 'node:test';
import assert from 'node:assert/strict';
import {precisionGenerationEligibility,precisionContinuationDisplay} from '../src/precision-continuation-data.ts';
const seal=n=>'sha256:'+n.repeat(64);
const scene=seal('1'),start=seal('2'),end=seal('3'),comparison=seal('4');
function fixture(){
  const task={sourceSceneId:scene,sourceTargetMm3:1000,
    minimumWholeTargetMm3:500,minimumTargetPerZoneMm3:50,
    minimumContributingZones:3,minimumTargetPerConfigurationMm3:100,
    minimumContributingConfigurations:2,maximumWholeNonTargetMm3:200,
    maximumContinuationNonTargetMm3:20,maxNativeActions:240};
  const stages=[
    {stage:0,tileId:'tile-1',configuration:'radius-2.2',stateId:seal('5'),
      nativeActions:2,targetIncrementMm3:200,checkedExitStateId:seal('6'),checkedExitActions:1},
    {stage:1,tileId:'tile-2',configuration:'radius-2.8',stateId:seal('7'),
      nativeActions:4,targetIncrementMm3:200,checkedExitStateId:seal('8'),checkedExitActions:1}];
  const outcome={taskGateMet:true,wholeTargetRemovedMm3:500,wholeNonTargetRemovedMm3:80,
    residualTargetMm3:500,sourceTargetFractionRemoved:.5,
    zoneTargetRemovedMm3:{0:100,1:100,2:100,3:100,4:100,5:0,6:0,7:0},
    configurationTargetRemovedMm3:{'radius-2.2':200,'radius-2.8':200},
    cleanupComplete:true,failedGates:[],legacyTask:{taskGateMet:true,
      additionalTargetMm3:400,additionalNonTargetMm3:10,
      perSectorRemovedMm3:{near:20,far:30},requestedSectorOrder:['far','near'],
      observedFirstProgressOrder:['far','near'],orderPassed:true}};
  const alternative={branchId:'staged-0-superior-access',sourceMethod:'superior-access',hand:'left',
    sourceAccessPlanSeal:seal('c'),
    status:'passed',finalStateId:end,nativeActions:5,stopReason:'bounded_work_inventory_exhausted',
    stages,outcome,exactNativeReplay:true,resumeVerified:true,workAttempts:2,exitRefusals:0};
  const summary={schema:'resectionlab.staged-anatomy-planning-summary.v1',sourceSceneId:scene,
    sourceStateId:start,comparisonSeal:comparison,sourceRole:'DEVELOPMENT',optimizerUpdates:0,task,
    cost:{physicalCalls:100,maxPhysicalCalls:4000,previewCalls:40,directStepCalls:60},
    wallSeconds:12,alternatives:[alternative]};
  const planning={task:'anatomy-access',sourceSceneId:scene,startStateId:start,
    comparisonSeal:seal('9'),sourceRole:'Joint source-only shortlist · deterministic DEVELOPMENT simulation',
    alternatives:[{method:'superior-access',planSeal:seal('c')}]};
  const sectorPlanning={task:'anatomy-strategy',sourceSceneId:scene,startStateId:start,
    sourceAccessComparisonSeal:planning.comparisonSeal,alternatives:[{method:'left-near-first'}]};
  const artifact={branchId:'main',branches:{main:{stateId:start,actions:[]},
    [alternative.branchId]:{stateId:end,actions:Array.from({length:5},()=>({operation:'inspect'}))}},
    parents:{[alternative.branchId]:{branchId:'main',position:0}},planning,sectorPlanning,
    stagedSummary:summary};
  const response={caseHash:seal('a'),view:{scene:{sceneId:scene},stateId:start},
    planning,sectorPlanning,position:0,branchId:'main',
    branches:[{id:'main',steps:0},{id:alternative.branchId,steps:5}]};
  return {artifact,response,summary,alternative};
}


function precisionFixture(){
  const data=fixture(),{summary,response,alternative}=data;
  summary.cusaModel='legacy-instantaneous-with-v2-mechanics';
  response.branchId=alternative.branchId;response.position=alternative.nativeActions;response.view.stateId=alternative.finalStateId;
  response.view.slots={left:{instrument:'cusa',active:false,inserted:false,grip:[]}};
  response.view.holds={};response.view.fluid={};response.view.debris={};response.view.sources=[];
  const outcome=structuredClone(alternative.outcome);
  outcome.wholeTargetRemovedMm3+=20;outcome.wholeNonTargetRemovedMm3+=2;
  outcome.residualTargetMm3-=20;outcome.sourceTargetFractionRemoved=.52;
  outcome.zoneTargetRemovedMm3[0]+=20;outcome.configurationTargetRemovedMm3['radius-0.6']=15;
  outcome.legacyTask.additionalTargetMm3+=20;outcome.legacyTask.additionalNonTargetMm3+=2;
  const pose={position:[1,2,3],quaternion:[0,0,0,1]};
  const precision={schema:'resectionlab.precision-continuation-summary.v1',sourceSceneId:scene,
    sourceBranchId:alternative.branchId,sourceStateId:alternative.finalStateId,
    finalBranchId:'retained-sourcefit-branch',finalStateId:seal('e'),sourceNativeActions:5,finalNativeActions:10,
    sourceComparisonSeal:comparison,comparisonSeal:seal('f'),sourceRole:'DEVELOPMENT',optimizerUpdates:0,
    cusaModel:summary.cusaModel,sourceOutcome:structuredClone(alternative.outcome),outcome,
    additionalTargetMm3:20,additionalNonTargetMm3:2,newStageCount:1,
    stages:[{stage:0,configuration:'radius-0.6',candidate:{endpoint:pose,outsidePose:pose,
      externalDepthMm:12,radialFromOriginalRayMm:1,inwardRasUnit:[0,0,1]},nativeActions:8,
      finalNativeActions:10,entryTargetIncrementMm3:15,wholeTargetIncrementMm3:20,
      wholeBrainIncrementMm3:2,checkedExitStateId:seal('e')}],
    stopReason:'MAX_TWO_RETARGET_STAGES',cost:{physicalCalls:80,maxPhysicalCalls:600},wallSeconds:4,
    originalBaselineNonTargetMm3:70,limitations:['Partial anatomy; no device fidelity claim.'],exactNativeReplay:true};
  return {...data,precision};
}
function withLive(data){return {...data.response,branches:[...data.response.branches,
  {id:data.precision.finalBranchId,steps:data.precision.finalNativeActions}],precisionPlanning:data.precision};}

test('generation requires a full clean editable validated staged DEVELOPMENT endpoint',()=>{
  const data=precisionFixture();
  assert.equal(precisionGenerationEligibility(data.artifact,data.response).ready,true);
  for(const mutate of [d=>d.response.position--,d=>d.response.view.stateId=seal('f'),
    d=>d.response.readOnly=true,d=>d.response.playback={readOnly:true},
    d=>d.response.verification={readOnly:true},d=>d.response.view.slots.left.active=true,
    d=>d.response.view.slots.left.inserted=true,d=>d.response.view.slots.left.grip=[1],
    d=>d.response.view.holds={left:{}},d=>d.response.view.fluid={0:1},
    d=>d.response.view.debris={0:1},d=>d.response.view.sources=[{bleedingActive:true}],
    d=>d.response.view.slots.left.instrument='forceps',d=>d.summary.cusaModel='contact-capacity-v3',
    d=>d.response.planning.sourceRole='FROZEN_ASSESSMENT',d=>d.summary.sourceRole=undefined,
    d=>d.response.precisionPlanning=null,
    d=>d.artifact.precisionPlanning={}]){
    const changed=precisionFixture();mutate(changed);
    assert.equal(precisionGenerationEligibility(changed.artifact,changed.response).ready,false);
  }
});

test('live continuation binds actual whole effects and source gates without stale static fallback',()=>{
  const data=precisionFixture(),response=withLive(data);
  const shown=precisionContinuationDisplay(data.artifact,response);
  assert.ok(shown);assert.equal(shown.additionalTargetMm3,20);
  assert.equal(shown.outcome.wholeTargetRemovedMm3,520);
  assert.equal(shown.stages[0].entryTargetIncrementMm3,15);
  assert.equal(shown.sourceOutcome.wholeNonTargetRemovedMm3,80);
  assert.equal(shown.originalBaselineNonTargetMm3,70);
  data.artifact.precisionSummary=data.precision;
  assert.equal(precisionContinuationDisplay(data.artifact,{...response,precisionPlanning:null}),null);
  for(const mutate of [p=>p.sourceComparisonSeal=seal('e'),p=>p.sourceStateId=seal('a'),
    p=>p.sourceOutcome.wholeTargetRemovedMm3++,p=>p.additionalTargetMm3++,
    p=>p.originalBaselineNonTargetMm3=80,p=>p.finalNativeActions++,
    p=>p.stages[0].candidate.endpoint=[1,2,3],p=>p.stages[0].configuration='radius-2.2',
    p=>p.stages[0].wholeTargetIncrementMm3=19,p=>p.stages[0].checkedExitStateId=seal('d'),
    p=>p.optimizerUpdates=1,p=>p.exactNativeReplay=false,p=>p.outcome.failedGates=['wholeTarget'],
    p=>p.cost.physicalCalls=601,p=>p.stages[0].candidate.inwardRasUnit=[0,0,2]]){
    const next=precisionFixture();mutate(next.precision);
    assert.equal(precisionContinuationDisplay(next.artifact,withLive(next)),null);
  }
});

test('saved full reports project only display evidence and verify saved native ancestry',()=>{
  const data=precisionFixture(),response=withLive(data);delete response.precisionPlanning;
  data.artifact.precisionPlanning={...data.precision,schema:'resectionlab.precision-continuation.v1',attemptedStages:[{hidden:true}]};
  data.artifact.branches[data.precision.finalBranchId]={stateId:data.precision.finalStateId,actions:Array.from({length:10},()=>({}))};
  data.artifact.parents[data.precision.finalBranchId]={branchId:data.precision.sourceBranchId,position:5};
  assert.ok(precisionContinuationDisplay(data.artifact,response));
  data.artifact.parents[data.precision.finalBranchId].position=0;
  assert.equal(precisionContinuationDisplay(data.artifact,response),null);
});

test('bounded no-stage STOP aliases source and preserves cost without claiming task success',()=>{
  const data=precisionFixture(),p=data.precision;
  p.finalBranchId=p.sourceBranchId;p.finalStateId=p.sourceStateId;p.finalNativeActions=p.sourceNativeActions;
  p.newStageCount=0;p.stages=[];p.additionalTargetMm3=0;p.additionalNonTargetMm3=0;
  p.outcome=structuredClone(p.sourceOutcome);p.stopReason='NO_ADMITTED_NEW_REENTRY';
  data.response.precisionPlanning=p;
  assert.ok(precisionContinuationDisplay(data.artifact,data.response));
  assert.equal(precisionContinuationDisplay(data.artifact,data.response).cost.physicalCalls,80);
  p.finalBranchId='invented';
  assert.equal(precisionContinuationDisplay(data.artifact,data.response),null);
});
