// Reused native report/branch fixture for data and workflow behavior tests.
export const seal=n=>'sha256:'+n.repeat(64);
export const scene=seal('1'),start=seal('2'),end=seal('3'),comparison=seal('4');
export function fixture(){
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
    sourceStateId:start,comparisonSeal:comparison,optimizerUpdates:0,task,
    cost:{physicalCalls:100,maxPhysicalCalls:4000,previewCalls:40,directStepCalls:60},
    wallSeconds:12,alternatives:[alternative]};
  const planning={task:'anatomy-access',sourceSceneId:scene,startStateId:start,
    comparisonSeal:seal('9'),sourceRole:'DEVELOPMENT',
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
