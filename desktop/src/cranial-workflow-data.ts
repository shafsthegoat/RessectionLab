import type {SurgicalResponse,SurgicalSequenceOutcome} from './surgical-types';
import type {WorkflowAlternative} from './surgical-workflow-data';

export interface CranialBlocker {code:string;message:string}
export interface CranialRootAvailability {rootId:string;label:string;openingMethod:string;available:boolean;blockers:CranialBlocker[];sceneId?:string;sourceStateId?:string}
export interface CranialPlanningAvailability {present:true;available:boolean;rootId?:string;roots:CranialRootAvailability[];sourceSceneId:string;sourceStateId:string;blockers:CranialBlocker[]}
export interface CranialGoalAssessment {targetBasis:string;originalTargetCells:number;originalTargetVolumeMm3:number;residualOriginalTargetMm3:number;creditedOriginalTargetCells:number;removedOriginalTargetCells:number;residualOriginalTargetCells:number;originalTargetFractionRemoved:number;originalTargetRemovedMm3:number;originalTargetTreatmentFraction:number;originalTargetTreatmentMm3:number;completedFrozenTask:boolean;reasons:string[];[key:string]:unknown}
export interface CranialLedger {boneRemovedMm3:number;preparationNonboneRemovedMm3:number;workNonTargetRemovedMm3:number;wholeNonTargetRemovedMm3:number;modeledElapsedSeconds:number;withinLimits:boolean;refusalReasons:string[];[key:string]:unknown}
export interface CranialAlternative {rootId:string;sceneId:string;branchId:string;finalStateId:string;nativeActions:number;label:string;completedFrozenTask:boolean;stopReason:string;goalAssessment:CranialGoalAssessment;ledger:CranialLedger;sequenceOutcome:SurgicalSequenceOutcome;wholeExitComplete:boolean}
export interface CranialPlanningComparison {schema:'resectionlab.cranial-strategy-comparison.v1';comparisonSeal:string;sourceSceneId:string;sourceStateId:string;rootId?:string;decision:string;selectedBranchId:string|null;alternatives:CranialAlternative[];failures:CranialBlocker[];limitations:string[];cost:{physicalCalls:number;previewCalls:number;directStepCalls:number;wallSeconds:number}}
const object=(v:unknown):v is Record<string,unknown>=>v!==null&&typeof v==='object'&&!Array.isArray(v);
const amount=(v:unknown):v is number=>typeof v==='number'&&Number.isFinite(v)&&v>=0;
const count=(v:unknown):v is number=>amount(v)&&Number.isSafeInteger(v);
const text=(v:unknown):v is string=>typeof v==='string'&&v.length>0&&v.length<=2048;
const strings=(v:unknown):v is string[]=>Array.isArray(v)&&v.length<=128&&v.every(text);
const blockers=(v:unknown):v is CranialBlocker[]=>Array.isArray(v)&&v.length<=128&&v.every(x=>object(x)&&text(x.code)&&text(x.message));
const close=(a:number,b:number)=>Math.abs(a-b)<=1e-5*Math.max(1,a,b);
/** Presence is not qualification: skull-bearing cases never silently use exposed-only planning. */
export function cranialPlanningPresent(r:SurgicalResponse|null|undefined):boolean {
  return !!r&&(r.cranialPlanningAvailability!=null||r.cranialPlanning!=null||!!r.view?.craniotomy||!!r.view?.cranialStages||r.view?.scene?.regions?.some(x=>x.material==='bone')===true);
}
export function cranialAvailability(r:SurgicalResponse|null|undefined):CranialPlanningAvailability|null {
  const a=r?.cranialPlanningAvailability;
  if(!a||a.present!==true||typeof a.available!=='boolean'||!text(a.sourceSceneId)||!text(a.sourceStateId)||!blockers(a.blockers)||!Array.isArray(a.roots)||a.roots.length>3)return null;
  if(a.rootId!==undefined&&!text(a.rootId))return null;
  if(a.roots.some(x=>!object(x)||!text(x.rootId)||!text(x.label)||!text(x.openingMethod)||typeof x.available!=='boolean'||!blockers(x.blockers)))return null;
  if(new Set(a.roots.map(x=>x.rootId)).size!==a.roots.length||a.rootId!==undefined&&!a.roots.some(x=>x.rootId===a.rootId))return null;
  return a;
}
export function cranialGenerationIssue(r:SurgicalResponse|null|undefined):string|null {
  if(!r)return 'Open a native source case.';
  if(r.readOnly||r.verification?.readOnly||r.playback?.readOnly)return 'Return to a verified editable action boundary before generating strategies.';
  const a=cranialAvailability(r);
  if(!a)return 'A skull-bearing or cranial case requires a source-bound qualified opening registry; no opening contract is available.';
  if(!a.available||!a.roots.some(x=>x.available))return a.blockers.map(x=>`${x.code}: ${x.message}`).join('; ')||'No source-qualified opening root is available. Review its anatomy and opening blockers.';
  return null;
}
/** Reject mismatched/malformed bridge evidence rather than deriving credit from accessibility. */
export function cranialComparison(r:SurgicalResponse|null|undefined):CranialPlanningComparison|null {
  const c=r?.cranialPlanning,a=cranialAvailability(r);
  if(!r||!c||!a||c.schema!=='resectionlab.cranial-strategy-comparison.v1'||!text(c.comparisonSeal)||c.sourceSceneId!==a.sourceSceneId||c.sourceStateId!==a.sourceStateId||!text(c.decision)||!blockers(c.failures)||!strings(c.limitations)||!Array.isArray(c.alternatives)||c.alternatives.length>3||!object(c.cost))return null;
  if(!count(c.cost.physicalCalls)||!count(c.cost.previewCalls)||!count(c.cost.directStepCalls)||c.cost.physicalCalls!==c.cost.previewCalls+c.cost.directStepCalls||!amount(c.cost.wallSeconds))return null;
  const ids=new Set<string>();
  for(const row of c.alternatives){
    if(!object(row)||!text(row.branchId)||ids.has(row.branchId)||!text(row.rootId)||!text(row.sceneId)||!text(row.finalStateId)||!text(row.label)||!text(row.stopReason)||!count(row.nativeActions)||typeof row.completedFrozenTask!=='boolean'||typeof row.wholeExitComplete!=='boolean'||!a.roots.some(x=>x.rootId===row.rootId&&(!x.sceneId||x.sceneId===row.sceneId))||!r.branches.some(x=>x.id===row.branchId&&x.steps===row.nativeActions))return null;
    ids.add(row.branchId);const g=row.goalAssessment,l=row.ledger,s=row.sequenceOutcome;
    if(!object(g)||!text(g.targetBasis)||!count(g.originalTargetCells)||g.originalTargetCells===0||!count(g.creditedOriginalTargetCells)||!count(g.removedOriginalTargetCells)||!count(g.residualOriginalTargetCells)||g.creditedOriginalTargetCells>g.removedOriginalTargetCells||g.removedOriginalTargetCells+g.residualOriginalTargetCells!==g.originalTargetCells||![g.originalTargetFractionRemoved,g.originalTargetTreatmentFraction,g.originalTargetRemovedMm3,g.originalTargetTreatmentMm3,g.originalTargetVolumeMm3,g.residualOriginalTargetMm3].every(amount)||g.originalTargetFractionRemoved>1||g.originalTargetTreatmentFraction>1||g.originalTargetVolumeMm3<=0||!close(g.originalTargetVolumeMm3,g.originalTargetRemovedMm3+g.residualOriginalTargetMm3)||!close(g.originalTargetTreatmentMm3,g.originalTargetTreatmentFraction*g.originalTargetVolumeMm3)||!close(g.originalTargetRemovedMm3,g.originalTargetFractionRemoved*g.originalTargetVolumeMm3)||!close(g.originalTargetFractionRemoved,g.removedOriginalTargetCells/g.originalTargetCells)||!close(g.originalTargetTreatmentFraction,g.creditedOriginalTargetCells/g.originalTargetCells)||typeof g.completedFrozenTask!=='boolean'||!strings(g.reasons))return null;
    if(!object(l)||![l.boneRemovedMm3,l.preparationNonboneRemovedMm3,l.workNonTargetRemovedMm3,l.wholeNonTargetRemovedMm3,l.modeledElapsedSeconds].every(amount)||typeof l.withinLimits!=='boolean'||!strings(l.refusalReasons)||!close(l.wholeNonTargetRemovedMm3,l.boneRemovedMm3+l.preparationNonboneRemovedMm3+l.workNonTargetRemovedMm3))return null;
    if(!object(s)||s.schema!=='resectionlab.sequence-outcome.v1'||s.sceneId!==row.sceneId||s.stateId!==row.finalStateId||!object(s.dimensions)||!strings(s.missingEvidence)||!strings(s.limitations)||typeof s.withdrawalComplete!=='boolean')return null;
    if(row.completedFrozenTask!==g.completedFrozenTask||row.completedFrozenTask&&(!row.wholeExitComplete||!s.withdrawalComplete||!l.withinLimits||g.creditedOriginalTargetCells===0||g.reasons.length>0))return null;
  }
  if(c.selectedBranchId!==null&&(!text(c.selectedBranchId)||!c.alternatives.some(x=>x.branchId===c.selectedBranchId&&x.completedFrozenTask)))return null;
  return c;
}
export function cranialTargetBasisLabel(basis:string):string {
  return basis==='original_annotation'?'Full original annotation':basis==='owned_target_labels_fallback'?'Owned source target fallback · full annotation unavailable':basis;
}
export function cranialWorkflowAlternative(row:CranialAlternative,selected:string|null):WorkflowAlternative {
  const g=row.goalAssessment,l=row.ledger,s=row.sequenceOutcome;
  const residual=g.residualOriginalTargetMm3;
  return {id:row.branchId,label:row.label,status:row.completedFrozenTask?'Frozen engineering task complete':g.creditedOriginalTargetCells>0?'Checked target progress · task incomplete':'No credited target progress · task incomplete',removedMm3:g.originalTargetTreatmentMm3,residualMm3:residual,fractionRemoved:g.originalTargetTreatmentFraction,nonTargetMm3:l.wholeNonTargetRemovedMm3,seconds:l.modeledElapsedSeconds,coverageLabel:'Credited target fraction',coverageHelp:`Treatment credit against ${cranialTargetBasisLabel(g.targetBasis)}`,quantityScope:'whole-sequence',recommended:row.branchId===selected&&row.completedFrozenTask,
    reason:[row.stopReason,...g.reasons,...l.refusalReasons,`Whole-body exit ${row.wholeExitComplete&&s.withdrawalComplete?'complete':'incomplete'}. Bone clearance alone is not tumor-task completion.`].join('; '),
    residualLocation:`Original target: ${g.residualOriginalTargetCells} / ${g.originalTargetCells} cells remain. Basis: ${cranialTargetBasisLabel(g.targetBasis)}.`,
    unknowns:[...s.missingEvidence,...s.limitations,'Patient qualification and clinical consequences remain unassessed. Whole opening, preparation and work costs are included once.'].join('; ')};
}
