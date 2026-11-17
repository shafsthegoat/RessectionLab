import type {CranialRecordedGeometry} from './cranial-recorded-geometry';
import type {SurgicalResponse,SurgicalSequenceOutcome} from './surgical-types';
import type {WorkflowAlternative} from './surgical-workflow-data';

export interface CranialBlocker {code:string;message:string;constraintViolations?:Array<{metric:string;actual?:number|null;maximum?:number|null;unit?:string}>}
export interface ProcedurePolicy {schema:string;policyId:string;policyHash:string;available:boolean;startStage:string;requiredCoverage:string[];scenarios:string[];limitations:string[]}
export interface CranialRootAvailability {rootId:string;label:string;openingMethod:string;available:boolean;blockers:CranialBlocker[];sceneId?:string;sourceStateId?:string}
export interface CranialPlanningAvailability {present:true;available:boolean;rootId?:string;roots:CranialRootAvailability[];sourceSceneId:string;sourceStateId:string;blockers:CranialBlocker[]}
export interface CranialPlanningAvailability {procedurePolicy?:ProcedurePolicy}
export interface CranialGoalAssessment {targetBasis:string;originalTargetCells:number;originalTargetVolumeMm3:number;residualOriginalTargetMm3:number;creditedOriginalTargetCells:number;removedOriginalTargetCells:number;residualOriginalTargetCells:number;originalTargetFractionRemoved:number;originalTargetRemovedMm3:number;originalTargetTreatmentFraction:number;originalTargetTreatmentMm3:number;completedFrozenTask:boolean;reasons:string[];[key:string]:unknown}
export interface CranialLedger {boneRemovedMm3:number;preparationNonboneRemovedMm3:number;workNonTargetRemovedMm3:number;wholeNonTargetRemovedMm3:number;modeledElapsedSeconds:number;withinLimits:boolean;refusalReasons:string[];[key:string]:unknown}
export interface CranialEndpointAssessment {schema:string;configHash:string;classification:string;completedTask:boolean;completeOriginalTargetRemoval:boolean;validWithdrawal:boolean;cleanupComplete:boolean;pending:string[];fluidMm3:number;debrisMm3:number;terminated:boolean;truncated:boolean}
export interface CranialAlternative {candidateKey?:string;recommendationEligible?:boolean;validStop?:boolean;historicalInspectionOnly?:boolean;endpointAssessment?:CranialEndpointAssessment;recordedGeometry?:CranialRecordedGeometry;nativeHistory?:Array<{stateBefore:string;stateAfter:string}>;rootId:string;sceneId:string;branchId:string;finalStateId:string;nativeActions:number;label:string;completedFrozenTask:boolean;stopReason:string;goalAssessment:CranialGoalAssessment;ledger:CranialLedger;sequenceOutcome:SurgicalSequenceOutcome;wholeExitComplete:boolean}
export interface CranialPlanningComparison {schema:'resectionlab.cranial-strategy-comparison.v1'|'resectionlab.cranial-strategy-comparison.v2'|'resectionlab.cranial-strategy-comparison.v3';endpointContract?:Record<string,unknown>;selectionEvidence?:Record<string,unknown>;historicalInspectionOnly?:boolean;historicalSelectedBranchId?:string|null;procedurePolicy?:ProcedurePolicy;selectionRule?:string;stopReason?:string;comparisonSeal:string;sourceBranchId?:string;sourceSceneId:string;sourceStateId:string;rootId?:string;decision:string;selectedBranchId:string|null;alternatives:CranialAlternative[];failures:CranialBlocker[];limitations:string[];cost:{physicalCalls:number;previewCalls:number;directStepCalls:number;wallSeconds:number}}
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
  if(!r||!c||!a||!['resectionlab.cranial-strategy-comparison.v1','resectionlab.cranial-strategy-comparison.v2','resectionlab.cranial-strategy-comparison.v3'].includes(c.schema)||!text(c.comparisonSeal)||c.sourceSceneId!==a.sourceSceneId||c.sourceStateId!==a.sourceStateId||!text(c.decision)||!blockers(c.failures)||!strings(c.limitations)||!Array.isArray(c.alternatives)||c.alternatives.length>3||!object(c.cost))return null;
  const canonical=c.schema==='resectionlab.cranial-strategy-comparison.v3';
  const procedure=c.schema==='resectionlab.cranial-strategy-comparison.v2'||canonical&&c.procedurePolicy!==undefined;
  if(canonical&&(!object(c.endpointContract)||c.endpointContract.schema!=='resectionlab.canonical-cranial-endpoint.v1'||!text(c.endpointContract.configHash)||!object(c.selectionEvidence)))return null;
  if(procedure&&(!object(c.procedurePolicy)||c.procedurePolicy.schema!=='resectionlab.supratentorial-procedure-policy.v1'||!text(c.procedurePolicy.policyId)||!text(c.procedurePolicy.policyHash)||c.procedurePolicy.policyHash!==a.procedurePolicy?.policyHash||c.selectionRule!=='max_treatment_within_harm_bounds'||!text(c.stopReason)))return null;
  if(!count(c.cost.physicalCalls)||!count(c.cost.previewCalls)||!count(c.cost.directStepCalls)||c.cost.physicalCalls!==c.cost.previewCalls+c.cost.directStepCalls||!amount(c.cost.wallSeconds))return null;
  const ids=new Set<string>();
  for(const row of c.alternatives){
    if(!object(row)||!text(row.branchId)||ids.has(row.branchId)||!text(row.rootId)||!text(row.sceneId)||!text(row.finalStateId)||!text(row.label)||!text(row.stopReason)||!count(row.nativeActions)||typeof row.completedFrozenTask!=='boolean'||typeof row.wholeExitComplete!=='boolean'||!a.roots.some(x=>x.rootId===row.rootId&&(!x.sceneId||x.sceneId===row.sceneId))||!r.branches.some(x=>x.id===row.branchId&&x.steps===row.nativeActions))return null;
    ids.add(row.branchId);const g=row.goalAssessment,l=row.ledger,s=row.sequenceOutcome;
    if(!object(g)||!text(g.targetBasis)||!count(g.originalTargetCells)||g.originalTargetCells===0||!count(g.creditedOriginalTargetCells)||!count(g.removedOriginalTargetCells)||!count(g.residualOriginalTargetCells)||g.creditedOriginalTargetCells>g.removedOriginalTargetCells||g.removedOriginalTargetCells+g.residualOriginalTargetCells!==g.originalTargetCells||![g.originalTargetFractionRemoved,g.originalTargetTreatmentFraction,g.originalTargetRemovedMm3,g.originalTargetTreatmentMm3,g.originalTargetVolumeMm3,g.residualOriginalTargetMm3].every(amount)||g.originalTargetFractionRemoved>1||g.originalTargetTreatmentFraction>1||g.originalTargetVolumeMm3<=0||!close(g.originalTargetVolumeMm3,g.originalTargetRemovedMm3+g.residualOriginalTargetMm3)||!close(g.originalTargetTreatmentMm3,g.originalTargetTreatmentFraction*g.originalTargetVolumeMm3)||!close(g.originalTargetRemovedMm3,g.originalTargetFractionRemoved*g.originalTargetVolumeMm3)||!close(g.originalTargetFractionRemoved,g.removedOriginalTargetCells/g.originalTargetCells)||!close(g.originalTargetTreatmentFraction,g.creditedOriginalTargetCells/g.originalTargetCells)||typeof g.completedFrozenTask!=='boolean'||!strings(g.reasons))return null;
    if(!object(l)||![l.boneRemovedMm3,l.preparationNonboneRemovedMm3,l.workNonTargetRemovedMm3,l.wholeNonTargetRemovedMm3,l.modeledElapsedSeconds].every(amount)||typeof l.withinLimits!=='boolean'||!strings(l.refusalReasons)||!close(l.wholeNonTargetRemovedMm3,l.boneRemovedMm3+l.preparationNonboneRemovedMm3+l.workNonTargetRemovedMm3))return null;
    if(!object(s)||s.schema!==(procedure?'resectionlab.sequence-outcome.v2':'resectionlab.sequence-outcome.v1')||s.sceneId!==row.sceneId||s.stateId!==row.finalStateId||!object(s.dimensions)||!strings(s.missingEvidence)||!strings(s.limitations)||typeof s.withdrawalComplete!=='boolean')return null;
    const neuralValue=(s.dimensions.neuralRemovedMm3 as {cumulative?:unknown}|undefined)?.cumulative;
    if(procedure&&(!amount(l.neuralRemovedMm3)||neuralValue!==l.neuralRemovedMm3||!row.wholeExitComplete||!s.withdrawalComplete||!l.withinLimits))return null;
    if(row.endpointAssessment){
      const e=row.endpointAssessment;
      if(e.schema!=='resectionlab.sequential-task-config.v1'||!text(e.configHash)||!['successful-treatment','valid-stop-with-residual','truncated','constraint-failure','in-progress'].includes(e.classification)||![e.completedTask,e.completeOriginalTargetRemoval,e.validWithdrawal,e.cleanupComplete,e.terminated,e.truncated].every(v=>typeof v==='boolean')||!strings(e.pending)||![e.fluidMm3,e.debrisMm3].every(amount)||e.completedTask!==(e.classification==='successful-treatment')||['successful-treatment','valid-stop-with-residual'].includes(e.classification)&&(!e.validWithdrawal||!e.cleanupComplete||!e.terminated||e.truncated))return null;
    }
    if(canonical){
      const e=row.endpointAssessment;
      if(!e||!text(row.candidateKey)||typeof row.validStop!=='boolean'||typeof row.recommendationEligible!=='boolean'||row.completedFrozenTask!==e.completedTask||e.configHash!==c.endpointContract?.configHash||row.validStop!==(['successful-treatment','valid-stop-with-residual'].includes(e.classification)&&e.terminated&&!e.truncated&&e.validWithdrawal&&e.cleanupComplete))return null;
    }
    if(row.completedFrozenTask!==g.completedFrozenTask||row.completedFrozenTask&&(!row.wholeExitComplete||!s.withdrawalComplete||!l.withinLimits||g.creditedOriginalTargetCells===0||g.reasons.length>0))return null;
  }
  if(c.selectedBranchId!==null&&(!text(c.selectedBranchId)||!c.alternatives.some(x=>x.branchId===c.selectedBranchId&&(canonical?x.validStop:procedure||x.completedFrozenTask))))return null;
  if(canonical){
    const proof=c.selectionEvidence!;
    if(proof.schema!=='resectionlab.opening-forest-priority.v1'||!text(proof.reason)||!Array.isArray(proof.candidates)||proof.candidates.length>75||proof.candidateCount!==proof.candidates.length||!count(proof.eligibleCount))return null;
    const candidates=proof.candidates;
    if(candidates.some(x=>!object(x)||!text(x.candidateKey)||typeof x.eligible!=='boolean')||new Set(candidates.map(x=>x.candidateKey)).size!==candidates.length||proof.eligibleCount!==candidates.filter(x=>x.eligible).length)return null;
    if(c.alternatives.some(row=>!candidates.some(x=>x.candidateKey===row.candidateKey&&x.rootId===row.rootId&&x.sceneId===row.sceneId&&x.finalStateId===row.finalStateId&&x.eligible===row.recommendationEligible&&(!x.eligible||row.validStop))))return null;
    if(c.selectedBranchId!==null&&!c.alternatives.some(row=>row.branchId===c.selectedBranchId&&row.candidateKey===proof.selectedCandidateKey&&row.recommendationEligible))return null;
    if(c.selectedBranchId===null&&proof.selectedCandidateKey!==null&&proof.exportFailure!=='PREFERRED_ENDPOINT_EXPORT_UNAVAILABLE')return null;
    return c;
  }
  const historic=structuredClone(c);historic.historicalSelectedBranchId=c.historicalSelectedBranchId??c.selectedBranchId;historic.selectedBranchId=null;historic.decision='HISTORICAL_COMPARISON_INSPECTION_ONLY';historic.historicalInspectionOnly=true;
  historic.stopReason='Generate again for current endpoint and complete-forest rules.';
  for(const row of historic.alternatives){row.historicalInspectionOnly=true;row.completedFrozenTask=false;row.goalAssessment.completedFrozenTask=false;row.validStop=false;row.recommendationEligible=false;delete row.endpointAssessment;}
  return historic;
}
export function cranialTargetBasisLabel(basis:string):string {
  return basis==='original_annotation'?'Full original annotation':basis==='owned_target_labels_fallback'?'Owned source target fallback · full annotation unavailable':basis;
}
export function cranialDecisionExplanation(c:CranialPlanningComparison|null):string {
  if(!c)return '';
  if(c.historicalInspectionOnly)return 'Historical comparison. Generate again to apply the current endpoint and harm priorities.';
  const reason=c.selectionEvidence?.reason;
  if(typeof reason==='string'&&reason.startsWith('unresolved_neural'))return 'Neural or structure effects remain incomparable within the declared bounds. No treatment plan is recommended.';
  if(typeof reason==='string'&&reason.startsWith('unresolved_exposure'))return 'Supported exposure effects remain incomparable within the declared bounds. No treatment plan is recommended.';
  if(c.selectionEvidence?.exportFailure)return 'The preferred checked plan could not be exported within the remaining budget. No substitute is recommended.';
  if(c.selectedBranchId===null)return 'No checked plan has a qualifying endpoint. Inspect residual work, cleanup and withdrawal requirements.';
  return 'Selection uses every checked candidate with a qualifying endpoint before limiting the displayed alternatives.';
}
export function cranialAlternativeLabel(row:CranialAlternative):string {
  const stroke=row.recordedGeometry?.paths.filter(p=>p.instrument==='cusa'&&p.operation==='apply').at(-1);
  return stroke?`${row.label} · target path ${stroke.pointsRasMm.at(-1)!.map(n=>n.toFixed(1)).join(' / ')} mm`:row.label;
}
export function cranialWorkflowAlternative(row:CranialAlternative,selected:string|null):WorkflowAlternative {
  const g=row.goalAssessment,l=row.ledger,s=row.sequenceOutcome;
  const residual=g.residualOriginalTargetMm3;
  return {id:row.branchId,label:cranialAlternativeLabel(row),status:row.historicalInspectionOnly?'Historical trajectory · generate again for current endpoint rules':row.endpointAssessment?row.endpointAssessment.classification.replaceAll('-',' '):row.completedFrozenTask?'Frozen engineering task complete':g.creditedOriginalTargetCells>0?'Checked target progress · task incomplete':'No credited target progress · task incomplete',removedMm3:g.originalTargetTreatmentMm3,residualMm3:residual,fractionRemoved:g.originalTargetTreatmentFraction,nonTargetMm3:l.wholeNonTargetRemovedMm3,seconds:l.modeledElapsedSeconds,coverageLabel:'Credited target fraction',coverageHelp:`Treatment credit against ${cranialTargetBasisLabel(g.targetBasis)}`,quantityScope:'whole-sequence',recommended:!row.historicalInspectionOnly&&row.branchId===selected&&(row.recommendationEligible??row.completedFrozenTask),
    reason:[row.stopReason,...g.reasons,...l.refusalReasons,`Whole-body exit ${row.wholeExitComplete&&s.withdrawalComplete?'complete':'incomplete'}. Bone clearance alone is not tumor-task completion.`].join('; '),
    residualLocation:`Original target: ${g.residualOriginalTargetCells} / ${g.originalTargetCells} cells remain. Basis: ${cranialTargetBasisLabel(g.targetBasis)}.`,
    unknowns:[...s.missingEvidence,...s.limitations,'Patient qualification and clinical consequences remain unassessed. Whole opening, preparation and work costs are included once.'].join('; ')};
}
