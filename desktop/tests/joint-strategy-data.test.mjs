import test from 'node:test';
import assert from 'node:assert/strict';
import {jointStrategyDisplay} from '../src/joint-strategy-data.ts';

const scene='sha256:scene',start='sha256:intact';
const methods=[['superior-access',2],['axis0-access',0],['axis1-access',1]];
const actions=n=>Array.from({length:n},(_,index)=>({operation:'inspect',index}));
const snapshot=(id,n)=>({sceneId:scene,stateId:id,actions:actions(n)});

function fixture(){
  const branches={main:snapshot(start,0)};
  const route=(method,axis)=>({id:`source-${method}`,nativeAxis:axis,sourceSceneId:scene});
  const access=(method,axis,joint)=>{
    const branchId=`${joint?'joint-':''}access-${method}`;
    const finalStateId=`sha256:${branchId}`;
    branches[branchId]=snapshot(finalStateId,2);
    return {branchId,method,label:method,startStateId:start,finalStateId,
      prefixActions:0,decisions:actions(2),route:route(method,axis),
      cost:{optimizerUpdates:0}};
  };
  const sector=(source,joint,passed)=>{
    const branchId=`${joint?'joint-':''}sector-${source.method}-left-far-first`;
    const finalStateId=`sha256:${branchId}`;
    branches[branchId]=snapshot(finalStateId,3);
    return {branchId,method:`${source.method}-left-far-first`,label:source.method,
      startStateId:start,finalStateId,decisions:actions(3),
      sourceAccessPrefixActions:2,status:passed?'passed':'unpassed',
      strategy:{sourceAccessMethod:source.method,sourceAccessBranchId:source.branchId,
        hand:'left',sectorOrder:['far','near']},
      cost:{optimizerUpdates:0},
      continuationOutcome:{taskGateMet:passed,cleanupComplete:passed}};
  };
  const classicAccess=methods.map(([method,axis])=>access(method,axis,false));
  const classicSectors=classicAccess.slice(0,2).map((source,i)=>sector(source,false,i===0));
  const jointAccess=[methods[0],methods[2]].map(([method,axis])=>access(method,axis,true));
  const jointSectors=jointAccess.map((source,i)=>sector(source,true,i===0));
  const failureId='joint-failed-access-axis0-access';
  branches[failureId]=snapshot(`sha256:${failureId}`,1);
  const failure={branchId:failureId,schema:'resectionlab.joint-anatomy-failure.v1',
    status:'unpassed',retentionRole:'stopped_native_branch_not_admitted',
    startStateId:start,finalStateId:`sha256:${failureId}`,nativeActions:1,
    phase:'access-axis0-access',reason:'native refusal',cleanupComplete:false,
    cleanupPending:true,verification:{exactNativeReplay:true}};
  const classic={sourceAccessComparison:{schema:'resectionlab.surgical-planning.v1',
    task:'anatomy-access',sourceSceneId:scene,startStateId:start,
    comparisonSeal:'sha256:classic-access',alternatives:classicAccess},
    comparison:{schema:'resectionlab.anatomy-strategy.v1',task:'anatomy-strategy',
      sourceSceneId:scene,startStateId:start,comparisonSeal:'sha256:classic-sector',
      sourceAccessComparisonSeal:'sha256:classic-access',alternatives:classicSectors}};
  const planning={schema:'resectionlab.surgical-planning.v1',task:'anatomy-access',
    status:'insufficient_distinct_routes',sourceRole:
      'Joint source-only shortlist · deterministic DEVELOPMENT simulation',
    optimizerUpdates:0,
    sourceSceneId:scene,startStateId:start,comparisonSeal:'sha256:joint-access',
    alternatives:jointAccess};
  const sectorPlanning={schema:'resectionlab.anatomy-strategy.v1',task:'anatomy-strategy',
    status:'strategy_options_compared',sourceSceneId:scene,startStateId:start,
    sourceAccessComparisonSeal:'sha256:joint-access',comparisonSeal:'sha256:joint-sector',
    optionCount:3,retainedCount:2,passedCount:1,alternatives:jointSectors};
  const report={schema:'resectionlab.joint-anatomy-strategy.v1',
    status:'joint_strategies_compared',sourceSceneId:scene,startStateId:start,
    jointSeal:'sha256:joint',executionRule:
      'one far-first left-CUSA sector strategy per source-selected route',
    optimizerUpdates:0,sourcePreserved:true,threeApproachTaskGateMet:false,
    geometry:{selected:methods.map(([method,nativeAxis])=>({nativeAxis,routeId:`source-${method}`}))},
    trials:methods.map(([method,nativeAxis])=>({nativeAxis,routeId:`source-${method}`,
      status:nativeAxis===0?'unpassed_access':'unpassed_sector',
      failedBranchIds:nativeAxis===0?[failureId]:[]})),
    failedExecutions:[failure],cost:{physicalCalls:24,previewCalls:10,directStepCalls:14,
      trace:Array.from({length:24},()=>({})),byPhase:{}},
    accessComparison:{comparisonSeal:'sha256:joint-access',sourceSceneId:scene,
      startStateId:start,alternatives:structuredClone(jointAccess)},
    sectorComparison:{comparisonSeal:'sha256:joint-sector',sourceSceneId:scene,
      startStateId:start,alternatives:structuredClone(jointSectors)}};
  const artifact={branchId:'main',branches,planning,sectorPlanning,
    sectorSourceAccessComparison:structuredClone(planning),
    sectorPlanningHistory:[classic],jointPlanning:report};
  const response={caseHash:'sha256:case',view:{scene:{sceneId:scene}},
    branches:Object.entries(branches).map(([id,row])=>({id,steps:row.actions.length}))};
  return {artifact,response};
}

test('joint display binds both native comparisons and preserves inspectable classic steps',()=>{
  const {artifact,response}=fixture();
  const data=jointStrategyDisplay(artifact,response);
  assert.ok(data);
  assert.equal(data.selectedRoutes,3);
  assert.equal(data.accessExecuted,2);
  assert.equal(data.sectorPassed,1);
  assert.equal(data.threeApproachTaskGateMet,false);
  assert.equal(data.planningPhysicalCalls,24);
  assert.equal(data.failures[0].status,'unpassed');
  assert.equal(data.failures[0].cleanupComplete,false);
  assert.deepEqual([data.classicAccess.length,data.classicSectors.length],[3,2]);
  assert.deepEqual([data.classicSectors[0].start,data.classicSectors[0].end],[2,3]);
});

test('joint display refuses scene, branch, and recorded step substitutions',()=>{
  for(const edit of [
    ({artifact})=>{artifact.jointPlanning.sourceSceneId='sha256:other';},
    ({response})=>{response.branches.find(row=>row.id==='joint-access-superior-access').steps=1;},
    ({artifact})=>{artifact.planning.alternatives[0].branchId='access-superior-access';},
    ({artifact})=>{artifact.jointPlanning.accessComparison.alternatives[0].branchId='access-superior-access';},
    ({artifact})=>{artifact.sectorPlanning.alternatives[0].strategy.sourceAccessBranchId='access-superior-access';},
    ({artifact})=>{artifact.sectorPlanningHistory[0].comparison.alternatives[0].finalStateId='sha256:forged';},
    ({artifact})=>{artifact.sectorPlanningHistory[0].comparison.sourceAccessComparisonSeal='sha256:forged';},
  ]){
    const data=fixture();edit(data);
    assert.equal(jointStrategyDisplay(data.artifact,data.response),null);
  }
});

test('failed branch labels and gates cannot become positive display claims',()=>{
  for(const edit of [
    ({artifact})=>{artifact.jointPlanning.failedExecutions[0].status='passed';},
    ({artifact})=>{artifact.jointPlanning.failedExecutions[0].retentionRole='admitted';},
    ({artifact})=>{artifact.jointPlanning.failedExecutions[0].nativeActions=2;},
    ({artifact})=>{artifact.jointPlanning.failedExecutions[0].cleanupPending=false;},
    ({artifact})=>{artifact.jointPlanning.trials[1].failedBranchIds=[];},
    ({artifact})=>{artifact.jointPlanning.trials[1].status='passed';},
    ({artifact})=>{artifact.jointPlanning.threeApproachTaskGateMet=true;},
    ({artifact})=>{artifact.sectorPlanning.alternatives[1].status='passed';},
    ({artifact})=>{artifact.sectorPlanning.alternatives[0].strategy.sectorOrder=['near','far'];},
    ({artifact})=>{artifact.jointPlanning.optimizerUpdates=1;},
    ({artifact})=>{artifact.jointPlanning.executionRule='learned sector policy';},
  ]){
    const data=fixture();edit(data);
    assert.equal(jointStrategyDisplay(data.artifact,data.response),null);
  }
});
