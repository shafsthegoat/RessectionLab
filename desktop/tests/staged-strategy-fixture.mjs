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

// Compact actual native display regression from the source-preserving paired0007
// run retained under contact-continuation-v1/desktop/paired-0007-staged-warm.
// The protocol fixture context remains synthetic; this is not a patient simulator.
export function zeroPreparationFixture(){
 const f=fixture();
 const actual={
  "stages": [
    {
      "stage": 0,
      "tileId": "local-sha256:9d62cdc5b14332dcffeb45ecce2ca4a8576afae5c71c010258534d06904a093a-forward-4",
      "configuration": "radius-1",
      "stateId": "sha256:46bdc08a33cb5d38a210ca9224930291e1635bbc9d4ff70cce8359c8ff6ea549",
      "nativeActions": 10,
      "targetIncrementMm3": 15,
      "executionMode": "requested_motion",
      "requestedPoseReached": true,
      "checkedExitStateId": "sha256:95e2b473d6f2430446e61db3131ce71811787f198c7c97fe4594effffd5bef6a",
      "checkedExitActions": 2
    },
    {
      "stage": 1,
      "tileId": "surface-pivot-3329620",
      "configuration": "radius-1",
      "stateId": "sha256:f92acc74d84a83476834153cb52c28b4859d5a7e9391209d319fa4b492b79d9a",
      "nativeActions": 12,
      "targetIncrementMm3": 5,
      "executionMode": "checked_collision_prefix",
      "requestedPoseReached": false,
      "checkedExitStateId": "sha256:04b5a6fa4ab67ca0c69662db152811e7371f8a8c8f87017915abe121ec9360e9",
      "checkedExitActions": 9
    },
    {
      "stage": 2,
      "tileId": "axial-4",
      "configuration": "radius-1",
      "stateId": "sha256:89461b18daf6de23aef69d066d0d471b041d005d0c5668d7e8096f85199169d3",
      "nativeActions": 14,
      "targetIncrementMm3": 5,
      "executionMode": "requested_motion",
      "requestedPoseReached": true,
      "checkedExitStateId": "sha256:e970a3a84732d79125f5398b203d5ca5c7653c1a5be67978d991542bb42a310a",
      "checkedExitActions": 2
    },
    {
      "stage": 3,
      "tileId": "local-sha256:c63128c3d86c1943c4de8f344f360d593b119fac87e23f1a084b63a170a2aff7-aminus-1",
      "configuration": "radius-1",
      "stateId": "sha256:7df6da59157b9d91580b461c53bdcc3a423301ca998df2f5fc3f92b88d3afed4",
      "nativeActions": 16,
      "targetIncrementMm3": 1,
      "executionMode": "requested_motion",
      "requestedPoseReached": true,
      "checkedExitStateId": "sha256:75f350b3fb23560ec31b9b466849046a2bc5f798bf6726062e7e49c529d639d2",
      "checkedExitActions": 2
    },
    {
      "stage": 4,
      "tileId": "prepare-3330234",
      "configuration": "radius-1",
      "stateId": "sha256:55fa61fcbf25a8ac5c1b204ddb402aec3f8e4fb0584dc0b6dc59e1e67dd4e62e",
      "nativeActions": 18,
      "targetIncrementMm3": 0,
      "executionMode": "checked_collision_prefix",
      "requestedPoseReached": false,
      "checkedExitStateId": "sha256:85ece9fcfa9c2d23a5148f647b1c86c72a464f304174806b1e5362e656b09c89",
      "checkedExitActions": 13
    },
    {
      "stage": 5,
      "tileId": "local-sha256:b59340c7639da41cd1a11277c7ff2386606708a7ae57975ef9ef447a9284d8f1-forward-4",
      "configuration": "radius-1",
      "stateId": "sha256:b08cf48a2865956a86d90ed751827b601c0ba5116cce6b870725123febcc1b15",
      "nativeActions": 20,
      "targetIncrementMm3": 3,
      "executionMode": "requested_motion",
      "requestedPoseReached": true,
      "checkedExitStateId": "sha256:a3d396d374e5f416414a038a0aa07768683f5ea950e4e33a1b60f89730bad6c9",
      "checkedExitActions": 15
    },
    {
      "stage": 6,
      "tileId": "local-sha256:5b6ed722a35c2019005f63eb4661540c805cd2fe9ba535f1da6dd4a493377560-forward-2",
      "configuration": "radius-1",
      "stateId": "sha256:62e2d8f0440b6ac1b207191959d0b68816cbfea2496ab99bdf74120838200191",
      "nativeActions": 22,
      "targetIncrementMm3": 1,
      "executionMode": "requested_motion",
      "requestedPoseReached": true,
      "checkedExitStateId": "sha256:cd4ea3e768b706ce994167b67be126f131b0378e4d23956b912c60a24fc38f22",
      "checkedExitActions": 15
    }
  ],
  "outcome": {
    "taskGateMet": false,
    "checks": {
      "wholeTarget": false,
      "zones": false,
      "configurations": false,
      "wholeNonTarget": true,
      "continuationNonTarget": true,
      "ratio": false,
      "actions": true,
      "cleanup": true,
      "exactReplay": true
    },
    "failedGates": [
      "wholeTarget",
      "zones",
      "configurations",
      "ratio"
    ],
    "wholeTargetRemovedMm3": 41,
    "wholeNonTargetRemovedMm3": 135,
    "residualTargetMm3": 19327,
    "sourceTargetFractionRemoved": 0.002116893845518381,
    "wholeNonTargetToTargetRatio": 3.292682926829268,
    "zoneTargetRemovedMm3": {
      "0": 38,
      "1": 3,
      "2": 0,
      "3": 0,
      "4": 0,
      "5": 0,
      "6": 0,
      "7": 0
    },
    "configurationTargetRemovedMm3": {
      "radius-1": 30
    },
    "cleanupComplete": true,
    "legacyTask": {
      "additionalTargetMm3": 30,
      "additionalNonTargetMm3": 20,
      "perSectorRemovedMm3": {
        "near": 6,
        "far": 2
      },
      "requestedSectorOrder": [
        "far",
        "near"
      ],
      "observedFirstProgressOrder": [
        "far",
        "near"
      ],
      "orderPassed": false,
      "firstProgressSimultaneous": true,
      "taskGateMet": false
    }
  },
  "nativeActions": 37,
  "finalStateId": "sha256:cd4ea3e768b706ce994167b67be126f131b0378e4d23956b912c60a24fc38f22",
  "resumeVerified": true,
  "workAttempts": 95,
  "exitRefusals": 0,
  "status": "unpassed",
  "stopReason": "FINITE_GENERATOR_NO_CHECKED_PROGRESS"
};
 Object.assign(f.alternative,actual);
 Object.assign(f.summary.task,{sourceTargetMm3:19368,minimumWholeTargetMm3:1936.8,minimumTargetPerZoneMm3:193.68,minimumTargetPerConfigurationMm3:387.36});
 f.summary.controllerMode='bounded-sequence-lookahead';
 f.summary.maxContactReplansPerSkill=8;
 f.summary.sequenceSearch={horizon:2,beamWidth:3,maxExpansionsPerSearch:24,baseline:'Immediate-removal baseline',preference:'Hard native constraints then target/non-target/time Pareto outcomes'};
 f.artifact.branches[f.alternative.branchId]={stateId:actual.finalStateId,actions:Array.from({length:actual.nativeActions},()=>({operation:'inspect'}))};
 f.response.branches[1].steps=actual.nativeActions;
 return f;
}
