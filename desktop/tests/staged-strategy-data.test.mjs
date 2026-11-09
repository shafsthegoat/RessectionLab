import test from 'node:test';
import assert from 'node:assert/strict';
import {stagedGenerationEligibility,stagedStrategyDisplay} from '../src/staged-strategy-data.ts';

import {fixture,seal,scene,start,end,comparison} from './staged-strategy-fixture.mjs';

test('staged evidence binds native endpoints and keeps larger and legacy gates separate',()=>{
  const {artifact,response}=fixture();
  const shown=stagedStrategyDisplay(artifact,response);
  assert.ok(shown);assert.equal(shown.sourceBranchId,'main');
  assert.equal(shown.alternatives[0].stages[1].nativeActions,4);
  assert.equal(shown.alternatives[0].outcome.taskGateMet,true);
  assert.equal(shown.alternatives[0].outcome.legacyTask.taskGateMet,true);
  assert.equal(shown.physicalCalls,100);
  for(const change of [
    data=>data.summary.sourceSceneId=seal('b'),
    data=>data.summary.optimizerUpdates=1,
    data=>data.summary.cost.physicalCalls=99,
    data=>data.summary.alternatives[0].nativeActions=4,
    data=>data.summary.alternatives[0].sourceAccessPlanSeal=seal('d'),
    data=>data.summary.alternatives[0].exactNativeReplay=false,
    data=>data.summary.alternatives[0].stages[0].checkedExitStateId='invented',
    data=>data.summary.alternatives[0].outcome.sourceTargetFractionRemoved=.6,
    data=>data.summary.alternatives[0].outcome.failedGates=['wholeTarget'],
    data=>data.summary.alternatives[0].outcome.legacyTask.perSectorRemovedMm3.near=0]){
    const data=fixture();change(data);
    assert.equal(stagedStrategyDisplay(data.artifact,data.response),null);
  }
});

test('configuration scope and failed export recovery remain distinct from planning cost',()=>{
  const {artifact,response,summary}=fixture();
  summary.comparisonScope='same-route-configurations';
  summary.limitations=['Initial geometry configurations share one access route.'];
  summary.exportRecovery={additionalAuthoritativeReplayCalls:7,
    fullSourceToRecoveredExport90SecondWallCapMet:false};
  const shown=stagedStrategyDisplay(artifact,response);
  assert.equal(shown.comparisonScope,'same-route-configurations');
  assert.equal(shown.physicalCalls,100);
  assert.equal(shown.exportRecovery.additionalReplayCalls,7);
  assert.equal(shown.exportRecovery.fullRunWithinWallCap,false);
  assert.deepEqual(shown.notes,summary.limitations);
  summary.exportRecovery={method:'earlier recovery metadata without measured accounting'};
  const legacy=stagedStrategyDisplay(artifact,response);
  assert.equal(legacy.exportRecovery.additionalReplayCalls,null);
  assert.equal(legacy.exportRecovery.fullRunWithinWallCap,null);
  summary.exportRecovery={additionalAuthoritativeReplayCalls:-1,
    fullSourceToRecoveredExport90SecondWallCapMet:false};
  assert.equal(stagedStrategyDisplay(artifact,response),null);
});

test('cumulative execution scope carries recorded time and travel without becoming a learned label',()=>{
  const {artifact,response,summary,alternative}=fixture();
  summary.controllerMode='cumulative-reentry';
  alternative.executionMetrics={simulatedSeconds:100,instrumentTravelMm:300,completedReentries:2,workingConfigurations:2};
  const shown=stagedStrategyDisplay(artifact,response);
  assert.equal(shown.controllerMode,'cumulative-reentry');
  assert.equal(shown.alternatives[0].executionMetrics.instrumentTravelMm,300);
  alternative.executionMetrics.completedReentries=3;
  assert.equal(stagedStrategyDisplay(artifact,response),null);
  alternative.executionMetrics.completedReentries=2;
  summary.controllerMode='trained-PPO';
  assert.equal(stagedStrategyDisplay(artifact,response),null);
});

test('native branch binding accepts configuration-study IDs without requiring a planner naming convention',()=>{
  const data=fixture(),row=data.alternative;
  const old=row.branchId,branch='narrow-ingress-initial-outside-radius-0.6';
  row.branchId=branch;row.sourceMethod='narrow-0.6-initial-config';
  data.summary.sourceAccessPlans=[{method:row.sourceMethod,planSeal:row.sourceAccessPlanSeal,
    sourceSceneId:scene,startStateId:start,routeSeal:seal('d')}];
  data.artifact.branches[branch]=data.artifact.branches[old];delete data.artifact.branches[old];
  data.artifact.parents[branch]=data.artifact.parents[old];delete data.artifact.parents[old];
  data.response.branches[1].id=branch;
  assert.ok(stagedStrategyDisplay(data.artifact,data.response));
  data.response.branches[1].id='unbound';
  assert.equal(stagedStrategyDisplay(data.artifact,data.response),null);
});

test('new source-only access plans bind staged branches while active classic routes remain unchanged',()=>{
  const data=fixture();
  const fresh={method:'source-bank-a',planSeal:seal('d'),startStateId:start,
    sourceSceneId:scene,routeSeal:seal('e')};
  data.summary.sourceAccessPlans=[fresh];
  const oldId=data.summary.alternatives[0].branchId;
  const newId='staged-0-'+fresh.method;
  data.summary.alternatives[0].sourceMethod=fresh.method;
  data.summary.alternatives[0].branchId=newId;
  data.summary.alternatives[0].sourceAccessPlanSeal=fresh.planSeal;
  data.artifact.branches[newId]=data.artifact.branches[oldId];
  data.artifact.parents[newId]=data.artifact.parents[oldId];
  delete data.artifact.branches[oldId];delete data.artifact.parents[oldId];
  data.response.branches[1].id=newId;
  data.summary.alternatives[0].stages[0].configuration='radius-0.6';
  data.summary.alternatives[0].outcome.configurationTargetRemovedMm3={
    'radius-0.6':200,'radius-2.8':200};
  assert.ok(stagedStrategyDisplay(data.artifact,data.response));
  data.summary.sourceAccessPlans[0].routeSeal='unsealed';
  assert.equal(stagedStrategyDisplay(data.artifact,data.response),null);
  data.summary.sourceAccessPlans=[];
  assert.equal(stagedStrategyDisplay(data.artifact,data.response),null);
});

test('legacy far-first order is not promoted when the order receipt disagrees',()=>{
  const data=fixture();
  data.summary.alternatives[0].outcome.legacyTask.orderPassed=false;
  assert.equal(stagedStrategyDisplay(data.artifact,data.response),null);
  data.summary.alternatives[0].outcome.legacyTask.observedFirstProgressOrder=['near','far'];
  data.summary.alternatives[0].outcome.legacyTask.taskGateMet=false;
  assert.ok(stagedStrategyDisplay(data.artifact,data.response));
  data.summary.alternatives[0].outcome.legacyTask.observedFirstProgressOrder=['far','near'];
  data.summary.alternatives[0].outcome.legacyTask.firstProgressSimultaneous=true;
  assert.ok(stagedStrategyDisplay(data.artifact,data.response));
  data.summary.alternatives[0].outcome.legacyTask.orderPassed=true;
  assert.equal(stagedStrategyDisplay(data.artifact,data.response),null);
});

test('native target outside admitted work stages remains visible in whole approach totals',()=>{
  const data=fixture(),outcome=data.summary.alternatives[0].outcome;
  outcome.legacyTask.additionalTargetMm3=405;
  outcome.wholeTargetRemovedMm3=505;outcome.residualTargetMm3=495;
  outcome.sourceTargetFractionRemoved=.505;outcome.zoneTargetRemovedMm3[0]=105;
  const displayed=stagedStrategyDisplay(data.artifact,data.response);
  assert.ok(displayed);
  assert.equal(displayed.alternatives[0].unassignedPostAccessTargetMm3,5);
  data.summary.alternatives[0].stages[0].targetIncrementMm3=206;
  assert.equal(stagedStrategyDisplay(data.artifact,data.response),null);
});

test('live compact summary wins over a stale static artifact and full saved report is a fallback',()=>{
  const data=fixture();
  const live=structuredClone(data.summary);live.alternatives[0].outcome.taskGateMet=false;
  live.alternatives[0].status='unpassed';live.alternatives[0].outcome.failedGates=['wholeTarget'];
  // A contradictory live task gate is not silently replaced by static evidence.
  assert.equal(stagedStrategyDisplay(data.artifact,{...data.response,stagedPlanning:live}),null);
  assert.ok(stagedStrategyDisplay(data.artifact,{...data.response,stagedPlanning:data.summary}));
  const full={schema:'resectionlab.staged-anatomy-planning.v1',sourceSceneId:scene,
    sourceStateId:start,comparisonSeal:comparison,sourceRole:'DEVELOPMENT',optimizerUpdates:0,
    task:data.summary.task,cost:data.summary.cost,wallSeconds:12,
    alternatives:[{...data.alternative,attemptedWork:[{status:'complete_work_and_exit_checked'},
      {status:'complete_work_and_exit_checked'}],interruptionRecovery:{exactResume:true}}]};
  delete data.artifact.stagedSummary;data.artifact.stagedPlanning=full;
  assert.equal(stagedStrategyDisplay(data.artifact,data.response)?.alternatives[0].workAttempts,2);
  assert.equal(stagedStrategyDisplay(data.artifact,{...data.response,stagedPlanning:null}),null);
});

test('newly generated native branches bind before the original source file is saved',()=>{
  const data=fixture();
  delete data.artifact.branches[data.alternative.branchId];
  delete data.artifact.parents[data.alternative.branchId];
  delete data.artifact.stagedSummary;
  assert.ok(stagedStrategyDisplay(data.artifact,{
    ...data.response,stagedPlanning:data.summary}));
  assert.equal(stagedStrategyDisplay(data.artifact,data.response),null);
  const changed=structuredClone(data.response);
  changed.branches[1].steps=1;
  assert.equal(stagedStrategyDisplay(data.artifact,{
    ...changed,stagedPlanning:data.summary}),null);
});

test('generation is explicit from either classic or joint intact source and rejects frozen anatomy',()=>{
  const data=fixture();delete data.artifact.stagedSummary;
  assert.equal(stagedGenerationEligibility(data.artifact,data.response).ready,true);
  data.artifact.jointPlanning={comparisonSeal:seal('d')};
  assert.equal(stagedGenerationEligibility(data.artifact,data.response).ready,true);
  data.response.position=1;
  assert.equal(stagedGenerationEligibility(data.artifact,data.response).ready,false);
  data.response.position=0;
  data.artifact.frozenAssessmentExecutions=[];
  assert.match(stagedGenerationEligibility(data.artifact,data.response).reason,/Frozen assessment/);
  delete data.artifact.frozenAssessmentExecutions;
  data.response.planning.sourceRole='FROZEN_ASSESSMENT source-only shortlist';
  assert.equal(stagedGenerationEligibility(data.artifact,data.response).ready,false);
  data.response.planning.sourceRole='DEVELOPMENT';
  data.response.stagedPlanning=null;
  assert.equal(stagedGenerationEligibility(data.artifact,data.response).ready,false);
});

test('bounded lookahead is displayed only with a valid declared search receipt',()=>{
 const f=fixture();f.summary.controllerMode='bounded-sequence-lookahead';assert.equal(stagedStrategyDisplay(f.artifact,f.response),null);
 f.summary.sequenceSearch={horizon:2,beamWidth:3,maxExpansionsPerSearch:24,baseline:'Immediate-removal baseline',preference:'Native constraints then target/non-target/time Pareto outcomes'};
 const shown=stagedStrategyDisplay(f.artifact,f.response);assert.equal(shown.controllerMode,'bounded-sequence-lookahead');assert.equal(shown.sequenceSearch.horizon,2);
 for(const patch of [{horizon:1},{horizon:4},{beamWidth:5},{maxExpansionsPerSearch:25},{maxExpansionsPerSearch:0},{preference:''}]){const search={...f.summary.sequenceSearch};Object.assign(f.summary.sequenceSearch,patch);assert.equal(stagedStrategyDisplay(f.artifact,f.response),null);f.summary.sequenceSearch=search;}
});
