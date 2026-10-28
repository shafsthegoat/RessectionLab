import type {SurgicalResponse,SurgicalPose} from './surgical-types';
import {stagedStrategyDisplay} from './staged-strategy-data.ts';
import type {StagedBranch,StagedDisplay} from './staged-strategy-data';

const object=(v:unknown):v is Record<string,unknown>=>v!==null&&typeof v==='object'&&!Array.isArray(v);
const text=(v:unknown):v is string=>typeof v==='string'&&v.length>0&&v.length<=1024;
const sealed=(v:unknown):v is string=>typeof v==='string'&&/^sha256:[a-f0-9]{64}$/.test(v);
const count=(v:unknown):v is number=>Number.isSafeInteger(v)&&(v as number)>=0;
const amount=(v:unknown):v is number=>typeof v==='number'&&Number.isFinite(v)&&v>=0;
const close=(a:number,b:number)=>Math.abs(a-b)<=1e-5;
const vector=(v:unknown,n:number):v is number[]=>Array.isArray(v)&&v.length===n&&v.every(x=>typeof x==='number'&&Number.isFinite(x));
const pose=(v:unknown):v is SurgicalPose=>object(v)&&vector(v.position,3)&&vector(v.quaternion,4)&&close(Math.hypot(...v.quaternion),1);
const equal=(a:unknown,b:unknown):boolean=>{
  if(a===b)return true;
  if(Array.isArray(a)&&Array.isArray(b))return a.length===b.length&&a.every((v,i)=>equal(v,b[i]));
  if(object(a)&&object(b))return Object.keys(a).length===Object.keys(b).length&&Object.keys(a).every(k=>Object.hasOwn(b,k)&&equal(a[k],b[k]));
  return false;
};
export interface PrecisionStage {
  stage:number;configuration:'radius-0.6';candidate:{endpoint:SurgicalPose;outsidePose:SurgicalPose;
    externalDepthMm:number;radialFromOriginalRayMm:number;inwardRasUnit:number[]};
  nativeActions:number;finalNativeActions:number;entryTargetIncrementMm3:number;
  wholeTargetIncrementMm3:number;wholeBrainIncrementMm3:number;checkedExitStateId:string;
}
export interface PrecisionDisplay {
  schema:'resectionlab.precision-continuation-summary.v1';sourceSceneId:string;
  sourceBranchId:string;sourceStateId:string;finalBranchId:string;finalStateId:string;
  sourceNativeActions:number;finalNativeActions:number;sourceComparisonSeal:string;comparisonSeal:string;
  sourceRole:'DEVELOPMENT';optimizerUpdates:0;cusaModel:'legacy-instantaneous-with-v2-mechanics';
  sourceOutcome:StagedBranch['outcome'];outcome:StagedBranch['outcome'];additionalTargetMm3:number;
  additionalNonTargetMm3:number;newStageCount:number;stages:PrecisionStage[];stopReason:string;
  cost:{physicalCalls:number;maxPhysicalCalls:number};wallSeconds:number;
  originalBaselineNonTargetMm3:number;limitations:string[];exactNativeReplay:true;
}

/** Native commands remain authoritative; this only controls when generation is offered. */
export function precisionGenerationEligibility(artifact:unknown,response:SurgicalResponse|null):{ready:boolean;reason:string}{
  const record=object(artifact)?artifact:null;
  if(Object.hasOwn(response??{},'precisionPlanning')||record&&(
      Object.hasOwn(record,'precisionPlanning')||Object.hasOwn(record,'precisionSummary')))
    return {ready:false,reason:'A precision continuation is already recorded for this source.'};
  const data=stagedStrategyDisplay(artifact,response);
  if(!response||!data)return {ready:false,reason:'Open a validated staged source comparison.'};
  if(data.cusaModel!=='legacy-instantaneous-with-v2-mechanics'||data.sourceRole!=='DEVELOPMENT')
    return {ready:false,reason:'Precision continuation requires a declared DEVELOPMENT source with legacy CUSA and V2 mechanics.'};
  if(response.readOnly||response.verification?.readOnly||response.playback?.readOnly)
    return {ready:false,reason:'Return to an editable native action boundary.'};
  const row=data.alternatives.find(row=>row.branchId===response.branchId);
  if(!row||response.position!==row.nativeActions||response.view.stateId!==row.finalStateId)
    return {ready:false,reason:'Inspect the full native endpoint of a validated staged branch.'};
  if(!row.outcome.cleanupComplete||!object(response.view.slots)||
      response.view.slots.left?.instrument!=='cusa'||
      Object.values(response.view.slots).some(slot=>slot.active||slot.inserted||slot.grip?.length)||
      Object.keys(response.view.holds??{}).length||
      response.view.sources?.some(source=>source.bleedingActive!==false||Number(source.reservoirMm3??0)>1e-9)||
      Object.values(response.view.fluid??{}).some(v=>v>1e-9)||
      Object.values(response.view.debris??{}).some(v=>v>1e-9))
    return {ready:false,reason:'A checked clean endpoint with tools withdrawn and released is required.'};
  if(row.nativeActions>=240||row.outcome.wholeNonTargetRemovedMm3>200||row.outcome.legacyTask.additionalNonTargetMm3>20)
    return {ready:false,reason:'The source already reaches an original action or non-target bound.'};
  return {ready:true,reason:'Ready at a checked staged endpoint.'};
}

function outcomeValid(value:unknown,task:StagedDisplay['task'],actions:number):value is StagedBranch['outcome']{
  if(!object(value)||!amount(value.wholeTargetRemovedMm3)||!amount(value.wholeNonTargetRemovedMm3)||
      !amount(value.residualTargetMm3)||!amount(value.sourceTargetFractionRemoved)||
      typeof value.cleanupComplete!=='boolean'||typeof value.taskGateMet!=='boolean'||
      !Array.isArray(value.failedGates)||!object(value.zoneTargetRemovedMm3)||
      !object(value.configurationTargetRemovedMm3)||!object(value.legacyTask))return false;
  const target=value.wholeTargetRemovedMm3,other=value.wholeNonTargetRemovedMm3;
  if(target+value.residualTargetMm3<=0||!close(value.sourceTargetFractionRemoved,target/(target+value.residualTargetMm3)))return false;
  const zones=Object.values(value.zoneTargetRemovedMm3),configs=Object.values(value.configurationTargetRemovedMm3),legacy=value.legacyTask;
  if(!zones.length||zones.length>8||zones.some(v=>!amount(v))||configs.some(v=>!amount(v))||
      !close(zones.reduce<number>((sum,v)=>sum+(v as number),0),target)||
      !amount(legacy.additionalTargetMm3)||!amount(legacy.additionalNonTargetMm3)||
      legacy.additionalTargetMm3>target+1e-5||legacy.additionalNonTargetMm3>other+1e-5||
      typeof legacy.taskGateMet!=='boolean'||!object(legacy.perSectorRemovedMm3)||
      !amount(legacy.perSectorRemovedMm3.near)||!amount(legacy.perSectorRemovedMm3.far))return false;
  const failed=[target<task.minimumWholeTargetMm3?'wholeTarget':null,
    zones.filter(v=>(v as number)>=task.minimumTargetPerZoneMm3).length<3?'zones':null,
    configs.filter(v=>(v as number)>=task.minimumTargetPerConfigurationMm3).length<2?'configurations':null,
    other>200?'wholeNonTarget':null,legacy.additionalNonTargetMm3>20?'continuationNonTarget':null,
    target===0||other/target>.2?'ratio':null,actions>240?'actions':null,!value.cleanupComplete?'cleanup':null].filter(v=>v!==null);
  if(value.taskGateMet!==(failed.length===0)||failed.length!==value.failedGates.length||
      failed.some(v=>!(value.failedGates as unknown[]).includes(v)))return false;
  return !legacy.taskGateMet||(actions<=48&&value.cleanupComplete&&legacy.additionalTargetMm3>=30&&
    legacy.additionalNonTargetMm3<=20&&legacy.perSectorRemovedMm3.near>=10&&legacy.perSectorRemovedMm3.far>=10&&legacy.orderPassed!==false);
}

/** Reject inconsistent evidence rather than falling back from a malformed live receipt. */
export function precisionContinuationDisplay(artifact:unknown,response:SurgicalResponse|null):PrecisionDisplay|null{
  if(!response)return null;
  const record=object(artifact)?artifact:null,live=Object.hasOwn(response,'precisionPlanning');
  const raw=live?response.precisionPlanning:record?.precisionSummary??record?.precisionPlanning;
  const value=object(raw)&&raw.schema==='resectionlab.precision-continuation.v1'?
    {...raw,schema:'resectionlab.precision-continuation-summary.v1'}:raw;
  const staged=stagedStrategyDisplay(artifact,response);
  if(!staged||!object(value)||value.schema!=='resectionlab.precision-continuation-summary.v1'||
      value.sourceSceneId!==response.view.scene.sceneId||!sealed(value.sourceStateId)||!sealed(value.finalStateId)||
      !text(value.sourceBranchId)||!text(value.finalBranchId)||
      !sealed(value.comparisonSeal)||value.sourceComparisonSeal!==staged.comparisonSeal||
      value.sourceRole!=='DEVELOPMENT'||staged.sourceRole!=='DEVELOPMENT'||value.optimizerUpdates!==0||
      value.cusaModel!=='legacy-instantaneous-with-v2-mechanics'||staged.cusaModel!==value.cusaModel||
      value.exactNativeReplay!==true||!count(value.sourceNativeActions)||!count(value.finalNativeActions)||
      value.finalNativeActions>240||value.finalNativeActions<value.sourceNativeActions||!count(value.newStageCount)||value.newStageCount>2||
      !Array.isArray(value.stages)||value.newStageCount!==value.stages.length||value.stages.length>48||
      !amount(value.additionalTargetMm3)||!amount(value.additionalNonTargetMm3)||
      !amount(value.originalBaselineNonTargetMm3)||!amount(value.wallSeconds)||
      !object(value.cost)||!count(value.cost.physicalCalls)||!count(value.cost.maxPhysicalCalls)||
      value.cost.maxPhysicalCalls>600||value.cost.physicalCalls>value.cost.maxPhysicalCalls||!text(value.stopReason)||
      !Array.isArray(value.limitations)||value.limitations.length>32||value.limitations.some(v=>!text(v)))return null;
  const source=staged.alternatives.find(row=>row.branchId===value.sourceBranchId);
  const outcome=value.outcome;
  if((Object.hasOwn(value.cost,'previewCalls')||Object.hasOwn(value.cost,'directStepCalls'))&&(
      !count(value.cost.previewCalls)||!count(value.cost.directStepCalls)||
      value.cost.physicalCalls!==value.cost.previewCalls+value.cost.directStepCalls))return null;
  if(!source||source.finalStateId!==value.sourceStateId||source.nativeActions!==value.sourceNativeActions||
      !source.outcome.cleanupComplete||!equal(value.sourceOutcome,source.outcome)||
      !outcomeValid(outcome,staged.task,value.finalNativeActions)||!outcome.cleanupComplete||
      !close(value.originalBaselineNonTargetMm3,source.outcome.wholeNonTargetRemovedMm3-source.outcome.legacyTask.additionalNonTargetMm3)||
      !close(outcome.wholeTargetRemovedMm3-source.outcome.wholeTargetRemovedMm3,value.additionalTargetMm3)||
      !close(outcome.wholeNonTargetRemovedMm3-source.outcome.wholeNonTargetRemovedMm3,value.additionalNonTargetMm3)||
      !close(outcome.legacyTask.additionalTargetMm3,source.outcome.legacyTask.additionalTargetMm3+value.additionalTargetMm3)||
      !close(outcome.legacyTask.additionalNonTargetMm3,source.outcome.legacyTask.additionalNonTargetMm3+value.additionalNonTargetMm3)||
      Object.entries(source.outcome.zoneTargetRemovedMm3).some(([key,amount])=>
        !close(outcome.zoneTargetRemovedMm3[key]??-1,amount)&&!(outcome.zoneTargetRemovedMm3[key]>=amount))||
      Object.entries(source.outcome.configurationTargetRemovedMm3).some(([key,amount])=>
        !(outcome.configurationTargetRemovedMm3[key]>=amount))||
      !close(outcome.wholeTargetRemovedMm3+outcome.residualTargetMm3,
        source.outcome.wholeTargetRemovedMm3+source.outcome.residualTargetMm3)||
      !response.branches.some(branch=>branch.id===value.finalBranchId&&branch.steps===value.finalNativeActions))return null;
  let position=value.sourceNativeActions,prior=-1,target=0,brain=0,entryTarget=0;
  for(const stage of value.stages){
    if(!object(stage)||!count(stage.stage)||stage.stage<=prior||stage.configuration!=='radius-0.6'||
        !count(stage.nativeActions)||!count(stage.finalNativeActions)||stage.nativeActions<=position||
        stage.finalNativeActions<stage.nativeActions||stage.finalNativeActions>value.finalNativeActions||
        !sealed(stage.checkedExitStateId)||!amount(stage.entryTargetIncrementMm3)||
        !amount(stage.wholeTargetIncrementMm3)||stage.entryTargetIncrementMm3<=0||stage.entryTargetIncrementMm3>stage.wholeTargetIncrementMm3+1e-5||
        !amount(stage.wholeBrainIncrementMm3)||!object(stage.candidate)||
        !pose(stage.candidate.endpoint)||!pose(stage.candidate.outsidePose)||
        !amount(stage.candidate.externalDepthMm)||stage.candidate.externalDepthMm>38+1e-5||
        !amount(stage.candidate.radialFromOriginalRayMm)||stage.candidate.radialFromOriginalRayMm>6+1e-5||
        !vector(stage.candidate.inwardRasUnit,3)||!close(Math.hypot(...stage.candidate.inwardRasUnit),1))return null;
    position=stage.finalNativeActions;prior=stage.stage;target+=stage.wholeTargetIncrementMm3;brain+=stage.wholeBrainIncrementMm3;entryTarget+=stage.entryTargetIncrementMm3;
  }
  if(!close(target,value.additionalTargetMm3)||!close(brain,value.additionalNonTargetMm3)||
      position!==value.finalNativeActions||value.stages.length===0&&(
        value.finalBranchId!==value.sourceBranchId||value.finalStateId!==value.sourceStateId)||
      value.stages.length>0&&(value.finalBranchId===value.sourceBranchId||
        value.stages.at(-1)!.checkedExitStateId!==value.finalStateId))return null;
  if(!close((outcome.configurationTargetRemovedMm3['radius-0.6']??0)-
      (source.outcome.configurationTargetRemovedMm3['radius-0.6']??0),entryTarget))return null;
  const snapshots=object(record?.branches)?record.branches:null;
  if(snapshots&&(!live||Object.hasOwn(snapshots,value.finalBranchId))){
    const snapshot=snapshots[value.finalBranchId];
    if(!object(snapshot)||snapshot.stateId!==value.finalStateId||!Array.isArray(snapshot.actions)||snapshot.actions.length!==value.finalNativeActions)return null;
    const parent=object(record?.parents)?record.parents[value.finalBranchId]:null;
    if(value.newStageCount>0&&object(parent)&&(parent.branchId!==value.sourceBranchId||parent.position!==value.sourceNativeActions))return null;
  }
  const hidden=new Set(['attemptedStages','stoppedExecutions','sourceProgramSeal','sourceRowSeal','sourceRouteSeal','controller']);
  return {...Object.fromEntries(Object.entries(value).filter(([key])=>!hidden.has(key))),
    schema:'resectionlab.precision-continuation-summary.v1'} as unknown as PrecisionDisplay;
}
