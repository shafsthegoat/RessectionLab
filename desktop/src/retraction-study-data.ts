import type {SurgicalResponse} from './surgical-types';

const object=(x:unknown):x is Record<string,unknown>=>!!x&&typeof x==='object'&&!Array.isArray(x);
const seal=(x:unknown):x is string=>typeof x==='string'&&/^sha256:[a-f0-9]{64}$/.test(x);
const number=(x:unknown):x is number=>typeof x==='number'&&Number.isFinite(x)&&x>=0;
const count=(x:unknown):x is number=>number(x)&&Number.isSafeInteger(x);
type Effects={wholeTargetRemovedMm3:number;wholeNonTargetRemovedMm3:number;residualTargetMm3:number;
  cleanupComplete:boolean;taskGateMet:boolean;failedGates:string[]};
export interface RetractionVariant {
  branchId:string;assisted:boolean;status:string;failureReason:string|null;nativeActions:number;
  finalStateId:string;outcome:Effects;phases:Array<{label:string;position:number}>;
  finalMetrics:Record<string,number>;sampledMechanicalPeaks:Record<string,number>;
  publicCalls:number;costMode:string;originalPlanningAndReplayCalls:number|null;
  outsideRecoveryOffsetMm:number;
}
export interface RetractionDisplay {variants:RetractionVariant[];sourceStateId:string;
  physicalCalls:number;wallSeconds:number;comparisonSeal:string}

/** Only use bridge-validated evidence, bound to this scene's native inventory.
 * The bridge remeasures whole effects and task gates; normal replay verifies causality.
 */
export function retractionStudyDisplay(response:SurgicalResponse|null):RetractionDisplay|null {
  if(!response)return null;
  const report=response.retractionStudy;
  if(!object(report)||report.schema!=='resectionlab.retractor-native-study.v1'||
      report.sourceSceneId!==response.view.scene.sceneId||!seal(report.sourceStateId)||
      !seal(report.comparisonSeal)||report.sourceRole!=='DEVELOPMENT'||
      report.optimizerUpdates!==0||report.clinicalOutcomes!=='unavailable'||
      !count(report.physicalCalls)||report.physicalCalls>1200||
      !number(report.wallSeconds)||report.wallSeconds>180||
      !Array.isArray(report.variants)||report.variants.length<1||report.variants.length>3)return null;
  const variants:RetractionVariant[]=[];
  const ids=new Set();
  for(const row of report.variants){
    if(!object(row))return null;
    const nativeActions=row.nativeActions,metrics=row.finalMetrics;
    if(typeof row.branchId!=='string'||ids.has(row.branchId)||
        !['completed_native_program','failed_native_program'].includes(String(row.status))||
        row.failureReason!==null&&typeof row.failureReason!=='string'||
        typeof row.assisted!=='boolean'||!seal(row.finalStateId)||
        !count(nativeActions)||nativeActions>240||row.exactNativeReplay!==true||
        !response.branches.some(b=>b.id===row.branchId&&b.steps===nativeActions)||
        !object(row.outcome)||!object(metrics)||!object(row.sampledMechanicalPeaks)||
        !Array.isArray(row.phases)||row.phases.length>20||
        !row.phases.every(p=>object(p)&&typeof p.label==='string'&&p.label.length<100&&
          count(p.position)&&p.position<=nativeActions)||
        !number(row.outsideRecoveryOffsetMm)||!count(row.publicCalls)||
        !['recorded_program_replay','planning_and_replay'].includes(String(row.costMode)))return null;
    const out=row.outcome;
    if(!['wholeTargetRemovedMm3','wholeNonTargetRemovedMm3','residualTargetMm3'].every(k=>number(out[k]))||
        typeof out.cleanupComplete!=='boolean'||typeof out.taskGateMet!=='boolean'||
        !Array.isArray(out.failedGates)||!out.failedGates.every(s=>typeof s==='string')||
        out.taskGateMet!==(out.failedGates.length===0)||
        !['brainMaxDisplacementMm','contactCellSeconds','elapsedSeconds'].every(k=>number(metrics[k]))||
        !number(row.sampledMechanicalPeaks.brainMaxDisplacementMm)||
        row.costMode==='recorded_program_replay'&&!count(row.originalPlanningAndReplayCalls))return null;
    ids.add(row.branchId);variants.push(row as unknown as RetractionVariant);
  }
  if(variants.reduce((sum,row)=>sum+row.publicCalls,0)!==report.physicalCalls)return null;
  return {variants,sourceStateId:report.sourceStateId,physicalCalls:report.physicalCalls,
    wallSeconds:report.wallSeconds,comparisonSeal:report.comparisonSeal};
}
