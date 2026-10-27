import type {SurgicalResponse} from './surgical-types';
export interface SectorExecution {schema:string;branchId:string;sourceRole:'DEVELOPMENT';evaluationRole:string;executionSourceSceneId:string;executionAccessPlanSeal:string;executionAccessMethod:string;checkpointFileSha256:string;actorSha256:string;trainingReceiptSha256:string;observationSchema:string;optimizerUpdates:number;status:string;taskGateMet:boolean;cleanupComplete:boolean;deltaTargetMm3:number;deltaNonTargetMm3:number;totalNonTargetMm3:number;perSectorRemovedMm3:{near:number;far:number};highLevelDecisions:string[];nativeActions:number;prefixActions:0;finalStateId:string;exactNativeReplay:true;sectorStartPosition:number;policyDecisions?:unknown}
const object=(value:unknown):value is Record<string,unknown>=>!!value&&typeof value==='object'&&!Array.isArray(value);
const hash=(value:unknown)=>typeof value==='string'&&/^[a-f0-9]{64}$/.test(value);
const sealed=(value:unknown)=>typeof value==='string'&&/^sha256:[a-f0-9]{64}$/.test(value);
const count=(value:unknown)=>Number.isSafeInteger(value)&&(value as number)>=0;
const quantity=(value:unknown):value is number=>typeof value==='number'&&Number.isFinite(value)&&value>=0;
const options=new Set(['corridor:left','corridor:right','near:left','near:right','far:left','far:right','STOP']);
export interface SectorDecisionEvidence {index:number;optionId:string;forcedCleanup:boolean;observationHash:string|null;status:string;reason:string|null;exitStatus:'passed'|'refused'|null;discardedPreviewCalls:number;discardedCommittedCalls:number}
/** Saved decision receipts are display evidence, never policy observations. */
export function learnedSectorDecisionEvidence(row:SectorExecution):SectorDecisionEvidence[]{
  const decisions=row.policyDecisions;
  if(!Array.isArray(decisions)||!decisions.length||decisions.length>6)return[];
  const selected:string[]=[];const evidence:SectorDecisionEvidence[]=[];
  for(const [index,decision] of decisions.entries()){
    if(!object(decision)||decision.index!==index||!options.has(String(decision.optionId))||!object(decision.nativeOption))return[];
    const forced=decision.boundary==='forced_cleanup',option=decision.nativeOption,probe=option.escapePreflight;
    if(!forced&&!sealed(decision.observationHash)||forced&&decision.optionId!=='STOP'||
      option.id!==decision.optionId||!['accepted','refused'].includes(String(option.status))||
      option.reason!==null&&typeof option.reason!=='string')return[];
    if(probe!==undefined&&(!object(probe)||!['passed','refused'].includes(String(probe.status))||
      !count(probe.previewCalls)||!count(probe.committedCalls)))return[];
    if(!forced)selected.push(String(decision.optionId));
    evidence.push({index,optionId:String(decision.optionId),forcedCleanup:forced,
      observationHash:forced?null:String(decision.observationHash),status:String(option.status),reason:option.reason as string|null,
      exitStatus:object(probe)?probe.status as 'passed'|'refused':null,
      discardedPreviewCalls:object(probe)?probe.previewCalls as number:0,
      discardedCommittedCalls:object(probe)?probe.committedCalls as number:0});
  }
  return selected.length===row.highLevelDecisions.length&&selected.every((id,i)=>id===row.highLevelDecisions[i])?evidence:[];
}
export function learnedSectorExecutions(artifact:unknown,response:SurgicalResponse|null):SectorExecution[]{
  if(!object(artifact)||!response||!Array.isArray(artifact.learnedSectorExecutions)||artifact.learnedSectorExecutions.length>12)return[];
  const rows=artifact.learnedSectorExecutions;
  if(rows.some(row=>!object(row)||row.schema!=='resectionlab.patient-sector-learning.v1'||
    row.sourceRole!=='DEVELOPMENT'||row.executionSourceSceneId!==response.view.scene.sceneId||
    !sealed(row.executionAccessPlanSeal)||typeof row.executionAccessMethod!=='string'||!row.executionAccessMethod||row.executionAccessMethod.length>128||
    !hash(row.checkpointFileSha256)||!hash(row.actorSha256)||!sealed(row.trainingReceiptSha256)||
    row.observationSchema!=='published-surgical-rl-v2-plus-sector-ledger-v1'||
    !count(row.optimizerUpdates)||!count(row.nativeActions)||row.prefixActions!==0||row.exactNativeReplay!==true||
    !sealed(row.finalStateId)||!response.branches.some(branch=>branch.id===row.branchId&&branch.steps===row.nativeActions)||
    !['success','abstained','incomplete','refused'].includes(String(row.status))||
    typeof row.cleanupComplete!=='boolean'||row.taskGateMet!==(row.status==='success')||
    !quantity(row.deltaTargetMm3)||!quantity(row.deltaNonTargetMm3)||!quantity(row.totalNonTargetMm3)||
    !object(row.perSectorRemovedMm3)||!quantity(row.perSectorRemovedMm3.near)||!quantity(row.perSectorRemovedMm3.far)||
    !Array.isArray(row.highLevelDecisions)||row.highLevelDecisions.length>5||!row.highLevelDecisions.length||
    !row.highLevelDecisions.every(option=>typeof option==='string'&&options.has(option))||
    (row.taskGateMet&&(!row.cleanupComplete||row.deltaTargetMm3<30||row.deltaNonTargetMm3>20||row.totalNonTargetMm3>200||
      row.perSectorRemovedMm3.near<10||row.perSectorRemovedMm3.far<10||
      !row.highLevelDecisions.some(option=>String(option).startsWith('near:'))||
      !row.highLevelDecisions.some(option=>String(option).startsWith('far:'))))))return[];
  if(new Set(rows.map(row=>row.branchId)).size!==rows.length)return[];
  const comparison=object(artifact.planning)&&Array.isArray(artifact.planning.alternatives)?artifact.planning.alternatives:[];
  return rows.map(row=>{
    const plan=comparison.find(plan=>object(plan)&&plan.planSeal===row.executionAccessPlanSeal);
    const phase=object(plan)&&Array.isArray(plan.directorPhases)?plan.directorPhases.find(phase=>object(phase)&&phase.id==='corridor-withdrawal'):undefined;
    const position=object(phase)&&count(phase.startPosition)&&(phase.startPosition as number)<(row.nativeActions as number)?phase.startPosition:0;
    return {...row,sectorStartPosition:position};
  }) as unknown as SectorExecution[];
}
