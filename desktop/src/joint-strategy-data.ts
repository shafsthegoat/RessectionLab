import type {SurgicalResponse} from './surgical-types';

const object=(value:unknown):value is Record<string,unknown>=>
  value!==null&&typeof value==='object'&&!Array.isArray(value);
const text=(value:unknown):value is string=>typeof value==='string'&&value.length>0;
const count=(value:unknown):value is number=>Number.isSafeInteger(value)&&(value as number)>=0;
const quantity=(value:unknown):number|null=>
  typeof value==='number'&&Number.isFinite(value)&&value>=0?value:null;
const fraction=(value:unknown):number|null=>{
  const amount=quantity(value);return amount!==null&&amount<=1?amount:null;
};
const same=(a:number|null,b:number|null)=>
  a===null||b===null||Math.abs(a-b)<=1e-6;
const methodNames=new Set(['superior-access','axis0-access','axis1-access']);
const rule='one far-first left-CUSA sector strategy per source-selected route';

export function jointGenerationEligibility(artifact:unknown,response:SurgicalResponse|null):{ready:boolean;reason:string}{
  if(!response||!text(response.caseHash)||!text(response.view?.scene?.sceneId))
    return {ready:false,reason:'Inspect a native source case first.'};
  const record=object(artifact)?artifact:null;
  if((record&&Object.hasOwn(record,'jointPlanning'))||response.jointPlanning)
    return {ready:false,reason:'A joint comparison is already saved. Open an original classic source to generate another.'};
  if(response.planningAvailability?.accessAvailable!==true||
      response.planningAvailability.skullAvailable!==false||
      response.planningAvailability.modeledBrainLayerAvailable!==true)
    return {ready:false,reason:'Open an intact mapped-brain source with unknown skull coverage and available native access.'};
  const access=response.planning,sectors=response.sectorPlanning;
  if(!access||!sectors||access.task!=='anatomy-access'||sectors.task!=='anatomy-strategy'||
      typeof access.sourceRole!=='string'||
      access.sourceRole.startsWith('Joint source-only shortlist')||
      access.sourceSceneId!==response.view.scene.sceneId||
      sectors.sourceSceneId!==response.view.scene.sceneId||
      sectors.sourceAccessComparisonSeal!==access.comparisonSeal||
      access.startStateId!==sectors.startStateId||
      !access.alternatives.length||!sectors.alternatives.length)
    return {ready:false,reason:'Open a native classic access and sector comparison source.'};
  if(response.verification?.readOnly||response.readOnly||response.playback?.readOnly)
    return {ready:false,reason:'Return to an editable native action boundary.'};
  if(response.position!==0||response.view.stateId!==access.startStateId||
      !response.branches.some(branch=>branch.id===response.branchId&&branch.steps===0))
    return {ready:false,reason:'Return to the common intact source start.'};
  return {ready:true,reason:'Ready at the common intact classic source.'};
}

export interface JointBranchRow {
  branchId:string;
  label:string;
  method:string;
  start:number;
  end:number;
  status:'passed'|'unpassed'|'executed';
  cleanupComplete?:boolean;
  reason?:string;
  entryRas:[number,number,number]|null;
  hand:string|null;
  order:string|null;
  wholeTargetMm3:number|null;
  wholeNonTargetMm3:number|null;
  residualTargetMm3:number|null;
  sourceTargetMm3:number|null;
  sourceFractionRemoved:number|null;
  additionalTargetMm3:number|null;
  additionalNonTargetMm3:number|null;
  simulatedSeconds:number|null;
  optionPhysicalCalls:number|null;
  withdrawal:{fastRefusal:string|null;slowStatus:string|null;slowSeconds:number|null}|null;
}
export interface JointStrategyDisplay {
  caseHash:string;
  sceneId:string;
  startStateId:string;
  sourceBranchId:string;
  jointSeal:string;
  selectedRoutes:number;
  accessExecuted:number;
  sectorPassed:number;
  threeApproachTaskGateMet:boolean;
  planningPhysicalCalls:number;
  access:JointBranchRow[];
  sectors:JointBranchRow[];
  failures:JointBranchRow[];
  classicAccess:JointBranchRow[];
  classicSectors:JointBranchRow[];
}

/** A UI admission check for records already verified during native export. */
export function jointStrategyDisplay(artifact:unknown,response:SurgicalResponse|null):JointStrategyDisplay|null {
  if(!response||!text(response.caseHash)||
      !text(response.view?.scene?.sceneId))return null;
  const record=object(artifact)?artifact:null;
  const live=Object.hasOwn(response,'jointPlanning');
  if(live&&!object(response.jointPlanning))return null;
  if(!live&&(!record||!object(record.jointPlanning)||!object(record.branches)))return null;
  const report=(live?response.jointPlanning:record!.jointPlanning) as Record<string,unknown>;
  const branches=live?null:record!.branches as Record<string,unknown>;
  const planning=live?response.planning:record!.planning;
  const sectorPlanning=live?response.sectorPlanning:record!.sectorPlanning;
  const history=live?response.jointBaselineHistory:record!.sectorPlanningHistory;
  const sourceAccess=live?response.planning:record!.sectorSourceAccessComparison;
  const sceneId=response.view.scene.sceneId;
  if(report.schema!=='resectionlab.joint-anatomy-strategy.v1'||
      report.status!=='joint_strategies_compared'||report.sourceSceneId!==sceneId||
      !text(report.startStateId)||!text(report.jointSeal)||
      report.executionRule!==rule||report.optimizerUpdates!==0||
      report.sourcePreserved!==true||typeof report.threeApproachTaskGateMet!=='boolean'||
      !object(report.geometry)||!Array.isArray(report.geometry.selected)||
      report.geometry.selected.length>3||!Array.isArray(report.trials)||
      report.trials.length>3||!Array.isArray(report.failedExecutions)||
      report.failedExecutions.length>3||!object(report.cost)||
      !count(report.cost.physicalCalls)||report.cost.physicalCalls>1500||
      !count(report.cost.previewCalls)||!count(report.cost.directStepCalls)||
      report.cost.physicalCalls!==report.cost.previewCalls+report.cost.directStepCalls||
      !Array.isArray(report.cost.trace)||
      report.cost.trace.length!==report.cost.physicalCalls||
      !object(report.cost.byPhase)||
      Object.hasOwn(report.cost.byPhase,'export-reopen-replay')||
      !object(report.accessComparison)||!object(report.sectorComparison)||
      !object(planning)||!object(sectorPlanning)||
      !Array.isArray(history)||!history.length)return null;

  const selected=new Map<number,string>();
  for(const entry of report.geometry.selected){
    if(!object(entry)||!count(entry.nativeAxis)||entry.nativeAxis>2||
        !text(entry.routeId)||selected.has(entry.nativeAxis))return null;
    selected.set(entry.nativeAxis,entry.routeId);
  }
  const activeAccess=planning,activeSectors=sectorPlanning;
  if(report.accessComparison.comparisonSeal!==activeAccess.comparisonSeal||
      report.sectorComparison.comparisonSeal!==activeSectors.comparisonSeal||
      report.accessComparison.sourceSceneId!==sceneId||
      report.sectorComparison.sourceSceneId!==sceneId||
      report.accessComparison.startStateId!==report.startStateId||
      report.sectorComparison.startStateId!==report.startStateId||
      !Array.isArray(report.accessComparison.alternatives)||
      !Array.isArray(report.sectorComparison.alternatives)||
      !Array.isArray(activeAccess.alternatives)||
      !Array.isArray(activeSectors.alternatives)||
      report.accessComparison.alternatives.map((row:unknown)=>
        object(row)?row.branchId:null).join(',')!==
        activeAccess.alternatives.map((row:unknown)=>object(row)?row.branchId:null).join(',')||
      report.sectorComparison.alternatives.map((row:unknown)=>
        object(row)?row.branchId:null).join(',')!==
        activeSectors.alternatives.map((row:unknown)=>object(row)?row.branchId:null).join(',')||
      activeAccess.sourceRole!==
        'Joint source-only shortlist · deterministic DEVELOPMENT simulation'||
      activeAccess.optimizerUpdates!==0||
      activeAccess.status!==
        (Array.isArray(activeAccess.alternatives)&&activeAccess.alternatives.length===3?
          'three_source_tissue_routes_skull_unsupported':'insufficient_distinct_routes')||
      activeSectors.sourceAccessComparisonSeal!==activeAccess.comparisonSeal||
      activeSectors.status!==
        (Array.isArray(activeSectors.alternatives)&&activeSectors.alternatives.length?
          'strategy_options_compared':'no_executable_strategy')||
      !object(sourceAccess)||
      sourceAccess.comparisonSeal!==activeAccess.comparisonSeal)return null;

  const original=history.at(-1);
  if(!object(original)||!object(original.sourceAccessComparison)||
      !object(original.comparison))return null;
  const classicAccess=original.sourceAccessComparison,classicSectors=original.comparison;
  const comparisons=[activeAccess,activeSectors,classicAccess,classicSectors];
  if(comparisons.some(pair=>pair.sourceSceneId!==sceneId||
      pair.startStateId!==report.startStateId||!text(pair.comparisonSeal)||
      !Array.isArray(pair.alternatives)))return null;
  if(activeAccess.task!=='anatomy-access'||activeSectors.task!=='anatomy-strategy'||
      classicAccess.task!=='anatomy-access'||classicSectors.task!=='anatomy-strategy'||
      classicSectors.sourceAccessComparisonSeal!==classicAccess.comparisonSeal||
      (activeAccess.alternatives as unknown[]).length>3||
      (activeSectors.alternatives as unknown[]).length>3||
      (classicAccess.alternatives as unknown[]).length>3||
      (classicSectors.alternatives as unknown[]).length>12)return null;
  const sourceBranchId=text(record?.branchId)&&
    response.branches.some(branch=>branch.id===record.branchId&&branch.steps===0)?
      record.branchId:response.branches.find(branch=>branch.steps===0)?.id;
  if(!text(sourceBranchId))return null;
  if(!live){
    const source=branches![sourceBranchId];
    if(!object(source)||source.sceneId!==sceneId||source.stateId!==report.startStateId||
        !Array.isArray(source.actions)||source.actions.length!==0)return null;
  }

  const seen=new Set<string>();
  const branch=(plan:Record<string,unknown>,maximum:number):{branchId:string;end:number}|null=>{
    if(!text(plan.branchId)||seen.has(plan.branchId)||
        !Array.isArray(plan.decisions)||plan.decisions.length>maximum||
        plan.startStateId!==report.startStateId||!text(plan.finalStateId))return null;
    const saved=branches?.[plan.branchId];
    if(!response.branches.some(item=>item.id===plan.branchId&&
          item.steps===(plan.decisions as unknown[]).length)||
        (branches&&(!object(saved)||saved.sceneId!==sceneId||
          saved.stateId!==plan.finalStateId||!Array.isArray(saved.actions)||
          saved.actions.length!==plan.decisions.length)))return null;
    seen.add(plan.branchId);
    return {branchId:plan.branchId,end:plan.decisions.length};
  };
  const parseAccess=(comparison:Record<string,unknown>,joint:boolean):JointBranchRow[]|null=>{
    const rows:JointBranchRow[]=[];
    for(const plan of comparison.alternatives as unknown[]){
      if(!object(plan)||!methodNames.has(String(plan.method))||
          !object(plan.route)||plan.route.sourceSceneId!==sceneId||
          !count(plan.route.nativeAxis)||plan.route.nativeAxis>2||
          !text(plan.route.id)||
          (joint&&(plan.branchId!==`joint-access-${plan.method}`||
            selected.get(plan.route.nativeAxis)!==plan.route.id)))return null;
      const saved=branch(plan,48);
      if(!saved||!count(plan.prefixActions)||plan.prefixActions>saved.end||
          (joint&&(plan.prefixActions!==0||!object(plan.cost)||
            plan.cost.optimizerUpdates!==0)))return null;
      const summary=object(plan.summary)?plan.summary:null;
      const route=plan.route as Record<string,unknown>;
      const surface=object(route.surfacePose)?route.surfacePose:null;
      const entry=Array.isArray(surface?.position)&&surface.position.length===3&&
        surface.position.every(value=>typeof value==='number'&&Number.isFinite(value))?
        surface.position as [number,number,number]:null;
      const cost=object(plan.cost)?plan.cost:null;
      const preview=quantity(cost?.previewCalls),execution=quantity(cost?.executionCalls),
        replay=quantity(cost?.freshReplayCalls);
      const recovery=object(plan.withdrawalRecovery)?plan.withdrawalRecovery:null;
      rows.push({...saved,method:plan.method as string,
        label:text(plan.label)?plan.label:plan.method as string,
        start:plan.prefixActions,status:'executed',entryRas:entry,hand:null,order:null,
        wholeTargetMm3:quantity(summary?.targetRemovedMm3),
        wholeNonTargetMm3:quantity(summary?.nonTargetRemovedMm3),
        residualTargetMm3:quantity(summary?.residualTargetMm3),
        sourceTargetMm3:null,sourceFractionRemoved:fraction(plan.fullTargetFractionRemoved),
        additionalTargetMm3:null,additionalNonTargetMm3:null,
        simulatedSeconds:quantity(summary?.elapsedSeconds),
        optionPhysicalCalls:preview!==null&&execution!==null&&replay!==null?
          preview+execution+replay:null,
        withdrawal:recovery?{
          fastRefusal:text(recovery.fastPreviewRejection)?recovery.fastPreviewRejection:null,
          slowStatus:text(recovery.slowStatus)?recovery.slowStatus:null,
          slowSeconds:quantity(recovery.slowDurationSeconds)}:null});
    }
    if(new Set(rows.map(row=>row.method)).size!==rows.length)return null;
    if(joint&&comparison.status!==
        (rows.length===3?'three_source_tissue_routes_skull_unsupported':'insufficient_distinct_routes'))return null;
    return rows;
  };
  const access=parseAccess(activeAccess,true);
  const priorAccess=parseAccess(classicAccess,false);
  if(!access||!priorAccess)return null;
  const parseSectors=(comparison:Record<string,unknown>,sources:JointBranchRow[],joint:boolean):JointBranchRow[]|null=>{
    const rows:JointBranchRow[]=[];
    for(const plan of comparison.alternatives as unknown[]){
      if(!object(plan)||!object(plan.strategy)||
          !methodNames.has(String(plan.strategy.sourceAccessMethod))||
          !object(plan.continuationOutcome)||
          typeof plan.continuationOutcome.taskGateMet!=='boolean'||
          typeof plan.continuationOutcome.cleanupComplete!=='boolean'||
          !['passed','unpassed'].includes(String(plan.status))||
          (plan.status==='passed')!==plan.continuationOutcome.taskGateMet)return null;
      const strategy=plan.strategy as Record<string,unknown>;
      const sourceRow=sources.find(row=>row.method===strategy.sourceAccessMethod);
      if(!sourceRow||strategy.sourceAccessBranchId!==sourceRow.branchId||
          (joint&&(plan.branchId!==
            `joint-sector-${strategy.sourceAccessMethod}-left-far-first`||
            strategy.hand!=='left'||
            !Array.isArray(strategy.sectorOrder)||
            strategy.sectorOrder.join(',')!=='far,near')))return null;
      const saved=branch(plan,48);
      if(!saved||!count(plan.sourceAccessPrefixActions)||
          plan.sourceAccessPrefixActions>saved.end||
          (joint&&(!object(plan.cost)||plan.cost.optimizerUpdates!==0)))return null;
      const summary=object(plan.summary)?plan.summary:null;
      const modeled=object(plan.modeledOutcomes)?plan.modeledOutcomes:null;
      const target=object(modeled?.target)?modeled.target:null;
      const other=object(modeled?.nonTargetRemovedMm3)?modeled.nonTargetRemovedMm3:null;
      const outcome=plan.continuationOutcome as Record<string,unknown>;
      const wholeTarget=quantity(target?.removedMm3)??quantity(summary?.targetRemovedMm3);
      const wholeOther=quantity(other?.value)??quantity(summary?.nonTargetRemovedMm3);
      const residual=quantity(target?.residualMm3),sourceVolume=quantity(target?.sourceVolumeMm3);
      const sourceFraction=fraction(target?.fractionRemoved);
      if(!same(quantity(target?.removedMm3),quantity(summary?.targetRemovedMm3))||
          !same(quantity(other?.value),quantity(summary?.nonTargetRemovedMm3))||
          !same(wholeOther,quantity(outcome.totalNonTargetMm3))||
          (wholeTarget!==null&&residual!==null&&sourceVolume!==null&&
            !same(wholeTarget+residual,sourceVolume))||
          (wholeTarget!==null&&sourceVolume!==null&&sourceVolume>0&&
            sourceFraction!==null&&!same(wholeTarget/sourceVolume,sourceFraction)))return null;
      const cost=object(plan.cost)?plan.cost:null;
      rows.push({...saved,method:strategy.sourceAccessMethod as string,
        label:text(plan.label)?plan.label:String(plan.method),
        start:plan.sourceAccessPrefixActions,
        status:plan.status as 'passed'|'unpassed',
        cleanupComplete:outcome.cleanupComplete as boolean,
        entryRas:sourceRow.entryRas,
        hand:text(strategy.hand)?strategy.hand:null,
        order:Array.isArray(strategy.sectorOrder)&&
          strategy.sectorOrder.every(text)?strategy.sectorOrder.join(' → '):null,
        wholeTargetMm3:wholeTarget,wholeNonTargetMm3:wholeOther,
        residualTargetMm3:residual,sourceTargetMm3:sourceVolume,
        sourceFractionRemoved:sourceFraction,
        additionalTargetMm3:quantity(outcome.deltaTargetMm3),
        additionalNonTargetMm3:quantity(outcome.deltaNonTargetMm3),
        simulatedSeconds:quantity(summary?.elapsedSeconds),
        optionPhysicalCalls:quantity(cost?.totalPhysicalCalls),
        withdrawal:sourceRow.withdrawal});
    }
    return rows;
  };
  const sectors=parseSectors(activeSectors,access,true);
  const priorSectors=parseSectors(classicSectors,priorAccess,false);
  if(!sectors||!priorSectors)return null;
  const failures:JointBranchRow[]=[];
  for(const failure of report.failedExecutions){
    if(!object(failure)||!text(failure.branchId)||
        !failure.branchId.startsWith('joint-failed-')||seen.has(failure.branchId)||
        failure.status!=='unpassed'||
        failure.retentionRole!=='stopped_native_branch_not_admitted'||
        failure.startStateId!==report.startStateId||
        !text(failure.finalStateId)||!count(failure.nativeActions)||
        failure.nativeActions>48||typeof failure.cleanupComplete!=='boolean'||
        typeof failure.cleanupPending!=='boolean'||
        failure.cleanupComplete===failure.cleanupPending||
        !object(failure.verification)||
        typeof failure.verification.exactNativeReplay!=='boolean')return null;
    const saved=branches?.[failure.branchId];
    if(!response.branches.some(item=>item.id===failure.branchId&&item.steps===failure.nativeActions)||
        (branches&&(!object(saved)||saved.sceneId!==sceneId||
          saved.stateId!==failure.finalStateId||!Array.isArray(saved.actions)||
          saved.actions.length!==failure.nativeActions)))return null;
    seen.add(failure.branchId);
    failures.push({branchId:failure.branchId,
      label:failure.branchId.replace('joint-failed-','').replaceAll('-',' '),
      method:String(failure.phase??'failed'),start:0,end:failure.nativeActions,
      status:'unpassed',cleanupComplete:failure.cleanupComplete,
      reason:text(failure.reason)?failure.reason.slice(0,180):'Recorded native execution stopped',
      entryRas:null,hand:null,order:null,
      wholeTargetMm3:null,wholeNonTargetMm3:null,residualTargetMm3:null,
      sourceTargetMm3:null,sourceFractionRemoved:null,
      additionalTargetMm3:null,additionalNonTargetMm3:null,
      simulatedSeconds:null,optionPhysicalCalls:null,withdrawal:null});
  }
  const failedIds=failures.map(row=>row.branchId).sort();
  const trialFailures:string[]=[];
  const trialAxes=new Set<number>();
  for(const trial of report.trials){
    if(!object(trial)||!count(trial.nativeAxis)||trial.nativeAxis>2||
        trialAxes.has(trial.nativeAxis)||!text(trial.routeId)||
        selected.get(trial.nativeAxis)!==trial.routeId||
        !Array.isArray(trial.failedBranchIds)||
        trial.failedBranchIds.some(id=>!text(id))||
        (trial.failedBranchIds.length>0&&
          !['unpassed_access','unpassed_sector'].includes(String(trial.status))))return null;
    trialAxes.add(trial.nativeAxis);
    trialFailures.push(...trial.failedBranchIds as string[]);
  }
  if(trialAxes.size!==selected.size||failedIds.join(',')!==trialFailures.sort().join(','))return null;
  const passed=sectors.filter(row=>row.status==='passed').length;
  if(activeSectors.passedCount!==passed||activeSectors.retainedCount!==sectors.length||
      activeSectors.optionCount!==selected.size||
      report.threeApproachTaskGateMet!==
        (access.length===3&&sectors.length===3&&passed===3))return null;
  return {caseHash:response.caseHash,sceneId,startStateId:report.startStateId as string,
    sourceBranchId,
    jointSeal:report.jointSeal as string,selectedRoutes:selected.size,
    accessExecuted:access.length,sectorPassed:passed,
    threeApproachTaskGateMet:report.threeApproachTaskGateMet as boolean,
    planningPhysicalCalls:report.cost.physicalCalls as number,
    access,sectors,failures,classicAccess:priorAccess,classicSectors:priorSectors};
}
