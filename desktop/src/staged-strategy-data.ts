import type {SurgicalResponse} from './surgical-types';

const object=(value:unknown):value is Record<string,unknown>=>
  value!==null&&typeof value==='object'&&!Array.isArray(value);
const text=(value:unknown):value is string=>typeof value==='string'&&value.length>0&&value.length<=256;
const sealed=(value:unknown):value is string=>typeof value==='string'&&/^sha256:[a-f0-9]{64}$/.test(value);
const count=(value:unknown):value is number=>Number.isSafeInteger(value)&&(value as number)>=0;
const amount=(value:unknown):value is number=>typeof value==='number'&&Number.isFinite(value)&&value>=0;
const close=(a:number,b:number)=>Math.abs(a-b)<=1e-5;
const method=(value:unknown):value is string=>
  typeof value==='string'&&/^[a-z][a-z0-9-]{0,95}$/.test(value);
const configuration=(value:unknown):value is string=>{
  if(typeof value!=='string'||!/^radius-(?:\d+(?:\.\d+)?)$/.test(value))return false;
  const radius=Number(value.slice('radius-'.length));
  return Number.isFinite(radius)&&radius>=.1&&radius<=12;
};
const failures=new Set(['wholeTarget','zones','configurations','wholeNonTarget',
  'continuationNonTarget','ratio','actions','cleanup','exactReplay']);

export interface StagedStage {
  stage:number;tileId:string;configuration:string;stateId:string;nativeActions:number;
  targetIncrementMm3:number;checkedExitStateId:string;checkedExitActions:number;
}
export interface StagedBranch {
  branchId:string;sourceMethod:string;sourceAccessPlanSeal:string;hand:string;status:'passed'|'unpassed';
  finalStateId:string;nativeActions:number;stopReason:string;stages:StagedStage[];
  outcome:{taskGateMet:boolean;wholeTargetRemovedMm3:number;wholeNonTargetRemovedMm3:number;
    residualTargetMm3:number;sourceTargetFractionRemoved:number;zoneTargetRemovedMm3:Record<string,number>;
    configurationTargetRemovedMm3:Record<string,number>;cleanupComplete:boolean;failedGates:string[];
    legacyTask:{taskGateMet:boolean;additionalTargetMm3:number;additionalNonTargetMm3:number;
      perSectorRemovedMm3:{near:number;far:number};requestedSectorOrder?:string[];
      observedFirstProgressOrder?:string[];orderPassed?:boolean;
      firstProgressSimultaneous?:boolean}};
  resumeVerified:boolean;workAttempts:number;exitRefusals:number;
  unassignedPostAccessTargetMm3:number;
}
export interface StagedDisplay {
  cusaModel:string;
  caseHash:string;sceneId:string;sourceStateId:string;sourceBranchId:string|null;
  comparisonSeal:string;task:{minimumWholeTargetMm3:number;minimumTargetPerZoneMm3:number;
    minimumTargetPerConfigurationMm3:number;minimumContributingZones:3;
    maximumWholeNonTargetMm3:200;maximumContinuationNonTargetMm3:20;maxNativeActions:240};
  physicalCalls:number;maxPhysicalCalls:number;wallSeconds:number;alternatives:StagedBranch[];
}

/** The full saved report is display evidence; omit its large traces and failed probe bodies. */
function compactFullReport(report:Record<string,unknown>):Record<string,unknown>|null {
  if(report.schema!=='resectionlab.staged-anatomy-planning.v1'||
      !Array.isArray(report.alternatives)||!object(report.cost))return null;
  return {schema:'resectionlab.staged-anatomy-planning-summary.v1',
    sourceSceneId:report.sourceSceneId,sourceStateId:report.sourceStateId,
    comparisonSeal:report.comparisonSeal,sourceRole:report.sourceRole,
    optimizerUpdates:report.optimizerUpdates,task:report.task,
    cusaModel:report.cusaModel,
    sourceAccessPlans:report.sourceAccessPlans,
    cost:{physicalCalls:report.cost.physicalCalls,maxPhysicalCalls:report.cost.maxPhysicalCalls,
      previewCalls:report.cost.previewCalls,directStepCalls:report.cost.directStepCalls},
    wallSeconds:report.wallSeconds,
    alternatives:report.alternatives.map(value=>{
      if(!object(value))return value;
      const attempts=Array.isArray(value.attemptedWork)?value.attemptedWork:[];
      return {branchId:value.branchId,sourceMethod:value.sourceMethod,
        sourceAccessPlanSeal:value.sourceAccessPlanSeal,hand:value.hand,
        status:value.status,finalStateId:value.finalStateId,nativeActions:value.nativeActions,
        stopReason:value.stopReason,stages:value.stages,outcome:value.outcome,
        exactNativeReplay:value.exactNativeReplay,
        resumeVerified:object(value.interruptionRecovery)?value.interruptionRecovery.exactResume===true:false,
        workAttempts:attempts.length,
        exitRefusals:attempts.filter(row=>object(row)&&row.status==='checked_exit_refused').length};
    })};
}

export function stagedGenerationEligibility(artifact:unknown,response:SurgicalResponse|null):{ready:boolean;reason:string} {
  if(!response||!text(response.caseHash)||!sealed(response.view?.scene?.sceneId))
    return {ready:false,reason:'Open a native source case first.'};
  const record=object(artifact)?artifact:null;
  if((record&&Object.hasOwn(record,'stagedPlanning'))||Object.hasOwn(response,'stagedPlanning'))
    return {ready:false,reason:'A staged comparison is already saved. Open an original source for another run.'};
  const access=response.planning,sector=response.sectorPlanning;
  if((record&&('frozenAssessmentExecutions' in record||
      object(record.planning)&&String(record.planning.sourceRole).includes('FROZEN_ASSESSMENT')))||
      access?.sourceRole?.includes('FROZEN_ASSESSMENT')||
      response.planningAvailability?.sourceRole?.includes('FROZEN_ASSESSMENT')||
      response.view.scene.source?.adapter?.sourcePatient==='UCSF-PDGM-0008')
    return {ready:false,reason:'Frozen assessment anatomy is reserved from staged generation.'};
  if(!access||!sector||access.task!=='anatomy-access'||sector.task!=='anatomy-strategy'||
      access.sourceSceneId!==response.view.scene.sceneId||
      sector.sourceSceneId!==response.view.scene.sceneId||
      sector.sourceAccessComparisonSeal!==access.comparisonSeal||
      access.startStateId!==sector.startStateId||
      !access.alternatives.length||!sector.alternatives.length)
    return {ready:false,reason:'Open a paired native access and sector source comparison.'};
  if(response.verification?.readOnly||response.readOnly||response.playback?.readOnly)
    return {ready:false,reason:'Return to an editable native action boundary.'};
  if(response.position!==0||response.view.stateId!==access.startStateId||
      !response.branches.some(branch=>branch.id===response.branchId&&branch.steps===0))
    return {ready:false,reason:'Return to the common intact source branch at action zero.'};
  return {ready:true,reason:'Ready at the intact paired source.'};
}

/** Bind compact bridge evidence or saved report evidence to this scene and branch inventory. */
export function stagedStrategyDisplay(artifact:unknown,response:SurgicalResponse|null):StagedDisplay|null {
  if(!response||!text(response.caseHash)||!sealed(response.view?.scene?.sceneId))return null;
  const record=object(artifact)?artifact:null;
  const live=Object.hasOwn(response,'stagedPlanning');
  if(live&&response.stagedPlanning===null)return null;
  const saved=object(record?.stagedSummary)?record!.stagedSummary:record?.stagedPlanning;
  const chosen=live?response.stagedPlanning:saved;
  if(!object(chosen))return null;
  const summary=chosen.schema==='resectionlab.staged-anatomy-planning.v1'?
    compactFullReport(chosen):chosen;
  if(!object(summary)||summary.schema!=='resectionlab.staged-anatomy-planning-summary.v1'||
      summary.sourceSceneId!==response.view.scene.sceneId||!sealed(summary.sourceStateId)||
      !sealed(summary.comparisonSeal)||summary.optimizerUpdates!==0||
      summary.sourceRole!==undefined&&summary.sourceRole!=='DEVELOPMENT'||
      !object(summary.task)||!object(summary.cost)||!Array.isArray(summary.alternatives)||
      summary.alternatives.length>3||!amount(summary.wallSeconds)||summary.wallSeconds>240||
      !count(summary.cost.physicalCalls)||!count(summary.cost.maxPhysicalCalls)||
      !count(summary.cost.previewCalls)||!count(summary.cost.directStepCalls)||
      summary.cost.maxPhysicalCalls>4000||
      summary.cost.physicalCalls>summary.cost.maxPhysicalCalls||
      summary.cost.physicalCalls!==summary.cost.previewCalls+summary.cost.directStepCalls)return null;
  const task=summary.task;
  if(!amount(task.minimumWholeTargetMm3)||task.minimumWholeTargetMm3===0||
      !amount(task.minimumTargetPerZoneMm3)||task.minimumTargetPerZoneMm3===0||
      !amount(task.minimumTargetPerConfigurationMm3)||task.minimumTargetPerConfigurationMm3===0||
      task.minimumContributingZones!==3||
      task.minimumContributingConfigurations!==undefined&&task.minimumContributingConfigurations!==2||
      task.maximumWholeNonTargetMm3!==200||task.maximumContinuationNonTargetMm3!==20||
      task.maxNativeActions!==240||
      task.sourceSceneId!==undefined&&task.sourceSceneId!==summary.sourceSceneId)return null;
  const planning=response.planning;
  if(!planning||planning.task!=='anatomy-access'||
      planning.sourceSceneId!==summary.sourceSceneId||planning.startStateId!==summary.sourceStateId)return null;
  if(planning.sourceRole?.includes('FROZEN_ASSESSMENT')||
      response.planningAvailability?.sourceRole?.includes('FROZEN_ASSESSMENT')||
      response.view.scene.source?.adapter?.sourcePatient==='UCSF-PDGM-0008'||
      record&&'frozenAssessmentExecutions' in record)return null;
  const full=object(record?.stagedPlanning)?record!.stagedPlanning:null;
  if(full&&(full.comparisonSeal!==summary.comparisonSeal||
      full.sourceSceneId!==summary.sourceSceneId||
      full.sourceStateId!==summary.sourceStateId))return null;
  const snapshots=object(record?.branches)?record!.branches:null;
  const sourceAccessPlans:Array<Record<string,unknown>>=[];
  const declaredSourcePlans=summary.sourceAccessPlans!==undefined;
  if(declaredSourcePlans){
    if(!Array.isArray(summary.sourceAccessPlans)||summary.sourceAccessPlans.length>3)return null;
    for(const plan of summary.sourceAccessPlans){
      if(!object(plan)||!method(plan.method)||!sealed(plan.planSeal)||
          plan.sourceSceneId!==summary.sourceSceneId||
          plan.startStateId!==summary.sourceStateId||!sealed(plan.routeSeal))return null;
      sourceAccessPlans.push(plan);
    }
    if(new Set(sourceAccessPlans.map(plan=>plan.planSeal)).size!==sourceAccessPlans.length)return null;
  }
  const sourceSnapshots=snapshots?Object.entries(snapshots).filter(([,value])=>
    object(value)&&value.stateId===summary.sourceStateId&&Array.isArray(value.actions)&&value.actions.length===0):[];
  const sourceBranchId=sourceSnapshots.length===1?sourceSnapshots[0][0]:
    response.branches.find(branch=>branch.id==='main'&&branch.steps===0)?.id??
    response.branches.find(branch=>branch.id===response.branchId&&branch.steps===0&&
      response.view.stateId===summary.sourceStateId)?.id??null;
  const rows:StagedBranch[]=[];
  const seen=new Set<string>();
  let totalTarget:number|null=null;
  for(const [index,raw] of summary.alternatives.entries()){
    if(!object(raw)||!text(raw.branchId)||seen.has(raw.branchId)||
        !method(raw.sourceMethod)||raw.branchId!==`staged-${index}-${raw.sourceMethod}`||
        !sealed(raw.sourceAccessPlanSeal)||
        raw.hand!=='left'||!['passed','unpassed'].includes(String(raw.status))||
        !sealed(raw.finalStateId)||!count(raw.nativeActions)||raw.nativeActions>240||
        !text(raw.stopReason)||raw.exactNativeReplay!==true||
        typeof raw.resumeVerified!=='boolean'||
        !count(raw.workAttempts)&&!Array.isArray(raw.workAttempts)||
        !count(raw.exitRefusals)&&!Array.isArray(raw.exitRefusals)||
        !Array.isArray(raw.stages)||raw.stages.length>12||!object(raw.outcome))return null;
    const attempts=Array.isArray(raw.workAttempts)?raw.workAttempts.length:raw.workAttempts as number;
    const refusals=Array.isArray(raw.exitRefusals)?raw.exitRefusals.length:raw.exitRefusals as number;
    const matchingSource=declaredSourcePlans?
      sourceAccessPlans.some(plan=>plan.method===raw.sourceMethod&&plan.planSeal===raw.sourceAccessPlanSeal):
      planning.alternatives.some(plan=>plan.method===raw.sourceMethod&&plan.planSeal===raw.sourceAccessPlanSeal);
    if(refusals>attempts||!matchingSource||
        !response.branches.some(branch=>branch.id===raw.branchId&&branch.steps===raw.nativeActions))return null;
    if(full){
      const original=Array.isArray(full.alternatives)?full.alternatives.find(value=>
        object(value)&&value.branchId===raw.branchId):undefined;
      if(!object(original)||original.sourceMethod!==raw.sourceMethod||
          original.sourceAccessPlanSeal!==raw.sourceAccessPlanSeal||
          original.finalStateId!==raw.finalStateId||original.nativeActions!==raw.nativeActions)return null;
    }
    // The open file remains the original source after live generation. Its
    // snapshots cannot contain newly admitted branches until the user saves.
    // A live summary is already native-validated by the bridge and must bind
    // to the current response branch inventory. Check static snapshots only
    // when they contain this branch, or when static evidence is being shown.
    if(snapshots&&(!live||Object.hasOwn(snapshots,raw.branchId))){
      const snapshot=snapshots[raw.branchId];
      if(!object(snapshot)||snapshot.stateId!==raw.finalStateId||
          !Array.isArray(snapshot.actions)||snapshot.actions.length!==raw.nativeActions)return null;
      const parent=object(record?.parents)?record.parents[raw.branchId]:undefined;
      if(object(parent)&&sourceBranchId!==null&&
          (parent.branchId!==sourceBranchId||parent.position!==0))return null;
    }
    const outcome=raw.outcome;
    if(!amount(outcome.wholeTargetRemovedMm3)||!amount(outcome.wholeNonTargetRemovedMm3)||
        !amount(outcome.residualTargetMm3)||!amount(outcome.sourceTargetFractionRemoved)||
        outcome.sourceTargetFractionRemoved>1||typeof outcome.taskGateMet!=='boolean'||
        typeof outcome.cleanupComplete!=='boolean'||
        !Array.isArray(outcome.failedGates)||
        outcome.failedGates.some(gate=>!failures.has(String(gate)))||
        new Set(outcome.failedGates).size!==outcome.failedGates.length||
        outcome.taskGateMet!==(outcome.failedGates.length===0)||
        outcome.taskGateMet!==(raw.status==='passed')||
        !object(outcome.zoneTargetRemovedMm3)||!object(outcome.configurationTargetRemovedMm3)||
        !object(outcome.legacyTask))return null;
    const sourceTarget=outcome.wholeTargetRemovedMm3+outcome.residualTargetMm3;
    if(sourceTarget<=0||!close(outcome.sourceTargetFractionRemoved,
        outcome.wholeTargetRemovedMm3/sourceTarget)||
        totalTarget!==null&&!close(sourceTarget,totalTarget)||
        task.sourceTargetMm3!==undefined&&(!amount(task.sourceTargetMm3)||!close(sourceTarget,task.sourceTargetMm3)))return null;
    totalTarget=sourceTarget;
    const zones=outcome.zoneTargetRemovedMm3,configs=outcome.configurationTargetRemovedMm3;
    if(!Object.keys(zones).length||Object.keys(zones).length>8||
        Object.values(zones).some(value=>!amount(value))||
        Object.entries(configs).some(([name,value])=>!configuration(name)||!amount(value)))return null;
    if(!close(Object.values(zones).reduce<number>((sum,value)=>sum+(value as number),0),
        outcome.wholeTargetRemovedMm3))return null;
    const legacy=outcome.legacyTask;
    if(typeof legacy.taskGateMet!=='boolean'||!amount(legacy.additionalTargetMm3)||
        !amount(legacy.additionalNonTargetMm3)||!object(legacy.perSectorRemovedMm3)||
        !amount(legacy.perSectorRemovedMm3.near)||!amount(legacy.perSectorRemovedMm3.far))return null;
    if(legacy.additionalTargetMm3>outcome.wholeTargetRemovedMm3+1e-5||
        legacy.additionalNonTargetMm3>outcome.wholeNonTargetRemovedMm3+1e-5)return null;
    if(legacy.orderPassed!==undefined&&typeof legacy.orderPassed!=='boolean')return null;
    if(legacy.firstProgressSimultaneous!==undefined&&
        typeof legacy.firstProgressSimultaneous!=='boolean')return null;
    if(legacy.requestedSectorOrder!==undefined&&(
        !Array.isArray(legacy.requestedSectorOrder)||
        legacy.requestedSectorOrder.join(',')!=='far,near'))return null;
    if(legacy.observedFirstProgressOrder!==undefined&&(
        !Array.isArray(legacy.observedFirstProgressOrder)||
        legacy.observedFirstProgressOrder.some(value=>value!=='far'&&value!=='near')))return null;
    if(legacy.orderPassed!==undefined&&Array.isArray(legacy.observedFirstProgressOrder)&&
        legacy.orderPassed!==(legacy.observedFirstProgressOrder.join(',')==='far,near'&&
          legacy.firstProgressSimultaneous!==true))return null;
    if(legacy.taskGateMet&&(!outcome.cleanupComplete||raw.nativeActions>48||
        legacy.additionalTargetMm3<30||legacy.additionalNonTargetMm3>20||
        legacy.perSectorRemovedMm3.near<10||legacy.perSectorRemovedMm3.far<10||
        legacy.orderPassed===false))return null;
    const observedFailures=[
      outcome.wholeTargetRemovedMm3<task.minimumWholeTargetMm3?'wholeTarget':null,
      Object.values(zones).filter(value=>(value as number)>=(task.minimumTargetPerZoneMm3 as number)).length<3?'zones':null,
      Object.values(configs).filter(value=>(value as number)>=(task.minimumTargetPerConfigurationMm3 as number)).length<2?'configurations':null,
      outcome.wholeNonTargetRemovedMm3>200?'wholeNonTarget':null,
      legacy.additionalNonTargetMm3>20?'continuationNonTarget':null,
      outcome.wholeTargetRemovedMm3===0||outcome.wholeNonTargetRemovedMm3/outcome.wholeTargetRemovedMm3>.2?'ratio':null,
      raw.nativeActions>240?'actions':null,
      !outcome.cleanupComplete?'cleanup':null,
    ].filter((reason):reason is string=>reason!==null);
    if(observedFailures.length!==outcome.failedGates.length||
        observedFailures.some(reason=>!(outcome.failedGates as unknown[]).includes(reason)))return null;
    if(outcome.taskGateMet&&(
        !outcome.cleanupComplete||outcome.wholeTargetRemovedMm3<task.minimumWholeTargetMm3||
        outcome.wholeNonTargetRemovedMm3>200||
        outcome.wholeNonTargetRemovedMm3/outcome.wholeTargetRemovedMm3>.2||
        legacy.additionalNonTargetMm3>20||
        Object.values(zones).filter(value=>(value as number)>=(task.minimumTargetPerZoneMm3 as number)).length<3||
        Object.values(configs).filter(value=>(value as number)>=(task.minimumTargetPerConfigurationMm3 as number)).length<2))return null;
    const stages:StagedStage[]=[];
    let prior=-1,position=0;
    for(const value of raw.stages){
      if(!object(value)||!count(value.stage)||value.stage>=12||value.stage<=prior||
          !text(value.tileId)||!configuration(value.configuration)||
          !sealed(value.stateId)||!count(value.nativeActions)||value.nativeActions<=position||
          value.nativeActions>raw.nativeActions||!amount(value.targetIncrementMm3)||
          value.targetIncrementMm3===0||!sealed(value.checkedExitStateId)||
          !count(value.checkedExitActions))return null;
      stages.push(value as unknown as StagedStage);
      prior=value.stage;position=value.nativeActions;
    }
    if(attempts<stages.length||raw.resumeVerified&&stages.length===0)return null;
    const byConfiguration=new Map<string,number>();
    for(const stage of stages)byConfiguration.set(stage.configuration,
      (byConfiguration.get(stage.configuration)??0)+stage.targetIncrementMm3);
    const admittedTarget=stages.reduce((total,stage)=>total+stage.targetIncrementMm3,0);
    if(admittedTarget>legacy.additionalTargetMm3+1e-5||
        Object.keys(configs).length!==byConfiguration.size||
        Object.entries(configs).some(([name,value])=>!byConfiguration.has(name)||
          !close(value as number,byConfiguration.get(name)!)))return null;
    seen.add(raw.branchId);
    rows.push({...raw,workAttempts:attempts,exitRefusals:refusals,
      unassignedPostAccessTargetMm3:Math.max(0,legacy.additionalTargetMm3-admittedTarget),
      stages,outcome:{...outcome,legacyTask:legacy}} as StagedBranch);
  }
  return {caseHash:response.caseHash,sceneId:summary.sourceSceneId as string,
    cusaModel:summary.cusaModel==='legacy-instantaneous-with-v2-mechanics'||summary.cusaModel==='contact-capacity-v3'?summary.cusaModel:'not-recorded',
    sourceStateId:summary.sourceStateId as string,sourceBranchId,
    comparisonSeal:summary.comparisonSeal as string,
    task:task as unknown as StagedDisplay['task'],
    physicalCalls:summary.cost.physicalCalls as number,
    maxPhysicalCalls:summary.cost.maxPhysicalCalls as number,
    wallSeconds:summary.wallSeconds as number,alternatives:rows};
}
