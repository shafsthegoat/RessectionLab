import type {SurgicalResponse} from './surgical-types';
import type {PreparationComparison,PreparationRow} from './preparation-planning-data';
import {preparationCost,preparationSuffix,preparationIncomplete,preparationPhaseAvailable} from './preparation-planning-data.ts';
import type {TwoToolAlternative} from './two-tool-data';
import type {InteractivePlanningRoute} from './interactive-planning-data';
import {stagedGenerationEligibility,stagedStrategyDisplay} from './staged-strategy-data.ts';

export type SurgicalWorkflowStage='case'|'plan'|'compare'|'rehearse';
export type SurgicalWorkflowPlanner='staged'|'preparation'|'strategy'|'approach';
export interface WorkflowAlternative {
  id:string;label:string;status:string;removedMm3:number|null;residualMm3:number|null;
  fractionRemoved:number|null;nonTargetMm3:number|null;seconds:number|null;
  reason:string;unknowns:string;residualLocation:string;recommended:boolean;
}
const finite=(v:unknown):v is number=>typeof v==='number'&&Number.isFinite(v)&&v>=0;
export function workflowRefusalReason(reason:string|null|undefined):string {
  if(reason==='COMPUTATION_BUDGET'||reason==='GEOMETRY_WALL_BUDGET')return 'Search budget exhausted; complete-sequence feasibility remains unresolved.';
  if(reason==='REQUIRED_EVIDENCE_UNAVAILABLE')return 'Required evidence is missing; no treatment sequence is offered.';
  if(reason==='PREPARATION_GENERATOR_REFUSAL')return 'The candidate generator refused this preparation; this does not establish anatomical impossibility.';
  if(reason==='ANATOMICAL_CONTEXT_LIMIT')return 'The declared source-context constraint excludes the tested sequence.';
  if(reason==='NO_COMPLETE_TWO_TOOL_SEQUENCE'||reason==='NO_COMPLETE_TOOL_ACCESS')return 'The bounded candidates supplied no complete sequence; anatomical impossibility has not been established.';
  return reason??'No complete sequence was established for this preparation.';
}
export function workflowReportCurrent(response:SurgicalResponse|null|undefined,report:{current?:boolean;sourceSceneId:string;startStateId:string;prefixActions:number;sourcePosition?:number}|null|undefined):boolean {
  return !!response&&!!report&&report.current===true&&report.sourceSceneId===response.view.scene.sceneId&&
    report.startStateId===response.view.stateId&&report.prefixActions===response.position&&
    (report.sourcePosition===undefined||report.sourcePosition===response.position);
}
export function workflowPlannerOptions(r:SurgicalResponse|null|undefined,artifact?:unknown):SurgicalWorkflowPlanner[] {
  if(!r)return [];
  return [
    (stagedGenerationEligibility(artifact,r).ready||stagedStrategyDisplay(artifact,r))&&'staged',
    (r.preparationPlanningAvailability?.available||r.preparationPlanning)&&'preparation',
    (r.strategyPlanningAvailability?.available||r.strategyPlanning)&&'strategy',
    (r.interactivePlanningAvailability?.available||r.interactivePlanning)&&'approach',
  ].filter(Boolean) as SurgicalWorkflowPlanner[];
}
/** Prefer a recorded treatment comparison over incidental preparation eligibility. */
export function preferredWorkflowPlanner(r:SurgicalResponse|null|undefined,artifact?:unknown):SurgicalWorkflowPlanner|null {
  if(!r)return null;
  if(stagedStrategyDisplay(artifact,r)||stagedGenerationEligibility(artifact,r).ready)return 'staged';
  const meaningfulPreparation=r.preparationPlanning?.rows.some(row=>preparationPhaseAvailable(row,'withdraw',r.preparationPlanning!.controls.preparationFamily));
  const meaningfulStrategy=r.strategyPlanning?.publicPlan.alternatives.some(row=>row.id!=='stop'&&row.feasible&&row.taskProgressMet);
  if(meaningfulPreparation&&workflowReportCurrent(r,r.preparationPlanning))return 'preparation';
  if(meaningfulStrategy&&workflowReportCurrent(r,r.strategyPlanning))return 'strategy';
  if(meaningfulPreparation)return 'preparation';
  if(meaningfulStrategy)return 'strategy';
  if(workflowReportCurrent(r,r.preparationPlanning))return 'preparation';
  if(workflowReportCurrent(r,r.strategyPlanning))return 'strategy';
  if(workflowReportCurrent(r,r.interactivePlanning))return 'approach';
  // Replay may restore an earlier scene identity. Keep its recorded comparison
  // discoverable instead of replacing it with newly available preparation.
  if(r.preparationPlanning)return 'preparation';
  if(r.strategyPlanning)return 'strategy';
  if(r.interactivePlanning)return 'approach';
  return workflowPlannerOptions(r,artifact)[0]??null;
}
export function workflowOriginalTarget(r:SurgicalResponse|null|undefined):number|null {
  const original=r?.sequenceOutcome?.sceneId===r?.view.scene.sceneId?r?.sequenceOutcome?.originalTarget:null;
  // A declared export target can be narrower than the original source target.
  if(original)return finite(original.sourceVolumeMm3)&&original.sourceVolumeMm3>0?original.sourceVolumeMm3:null;
  return null;
}
export function workflowCoverage(residual:unknown,original:unknown):number|null {
  return finite(residual)&&finite(original)&&original>0&&residual<=original?1-residual/original:null;
}
export function strategyWorkflowAlternative(row:TwoToolAlternative,selectedId:string):WorkflowAlternative {
  return {id:row.id,label:row.label,status:row.id==='stop'?'STOP · retains current state':row.feasible?'Checked sequence':'Refused sequence',
    removedMm3:row.feasible&&finite(row.summary.additionalTargetMm3)?row.summary.additionalTargetMm3:null,
    residualMm3:finite(row.summary.residualTargetMm3)?row.summary.residualTargetMm3:null,
    // Legacy summaries count owned labels, not original annotation memberships.
    fractionRemoved:null,
    nonTargetMm3:finite(row.summary.additionalNonTargetMm3)?row.summary.additionalNonTargetMm3:null,
    seconds:finite(row.summary.elapsedSeconds)?row.summary.elapsedSeconds:null,
    reason:row.id==='stop'?'No new actions; STOP does not establish completed treatment.':!row.feasible?'This sequence cannot be installed under the declared constraints.':row.taskProgressMet?'Meets the declared research progress threshold.':'Does not meet the declared research progress threshold.',
    unknowns:'Original source-target coverage is not recorded for this alternative. Residual volume describes the mapped task target. Functional outcomes, clinical risk and visibility are unassessed.',residualLocation:'Residual location not recorded for this alternative.',
    recommended:row.id===selectedId};
}
export function approachWorkflowAlternative(route:InteractivePlanningRoute,response:SurgicalResponse|null):WorkflowAlternative {
  const ledger=response?.sequenceOutcome;
  const original=ledger?.sceneId===response?.view.scene.sceneId&&ledger?.stateId===response?.view.stateId?ledger?.originalTarget:null;
  const centroid=original?.residualCentroidRasMm;
  return {id:route.id,label:route.label,status:'Inactive access only · no treatment sequence',removedMm3:0,
    residualMm3:finite(original?.residualMm3)?original.residualMm3:null,
    fractionRemoved:workflowCoverage(original?.residualMm3,original?workflowOriginalTarget(response):null),nonTargetMm3:0,seconds:null,
    reason:`Inactive placement and withdrawal checked: ${finite(route.metrics.pathLengthMm)?route.metrics.pathLengthMm.toFixed(1):'unassessed'} mm; ${finite(route.metrics.actionCount)?route.metrics.actionCount:'unassessed'} actions.`,
    unknowns:'Target treatment, broader maneuverability and clinical outcomes are unassessed. Existing access burdens remain in the accepted-state ledger.',
    residualLocation:centroid?.length===3&&centroid.every(value=>typeof value==='number'&&Number.isFinite(value))?
      `Unchanged original residual centroid · RAS ${centroid.map(value=>value.toFixed(1)).join(' / ')} mm.`:
      'Residual location not recorded; target remains at the checkpoint.',recommended:false};
}
export function preparationWorkflowAlternative(row:PreparationRow,report:PreparationComparison):WorkflowAlternative {
  const suffix=preparationSuffix(row),cost=preparationCost(row),complete=preparationPhaseAvailable(row,'withdraw',report.controls.preparationFamily);
  const acceptedPreparation=preparationPhaseAvailable(row,'prepared',report.controls.preparationFamily);
  const plan=row.comparison?.publicPlan;
  const selected=plan?.alternatives.find(item=>item.id===plan.decision.selectedId);
  const residual=acceptedPreparation?selected?.summary.residualTargetMm3:null;
  const sum=(a:unknown,b:unknown)=>finite(a)&&finite(b)?a+b:null;
  return {id:row.id,label:`${row.widthMm} mm ${report.controls.preparationFamily==='paired-lanes-v1'?'paired lanes':'local sweep'}`,
    status:preparationIncomplete(row)?'Search incomplete':complete?'Preparation + checked sequence':'No complete treatment continuation',
    removedMm3:acceptedPreparation&&cost?sum(cost.additionalTargetMm3,suffix?.summary.additionalTargetMm3??0):null,
    residualMm3:finite(residual)?residual:null,fractionRemoved:null,
    nonTargetMm3:cost?sum(cost.additionalNonTargetMm3,suffix?.summary.additionalNonTargetMm3??0):null,
    seconds:cost?sum(cost.elapsedSeconds,suffix?.summary.elapsedSeconds??0):null,
    reason:workflowRefusalReason(row.reason??row.comparison?.publicPlan.decision.reason),
    unknowns:[...(row.missingEvidence??[]),...(acceptedPreparation?[]:['Displayed burdens are uncommitted attempt costs; rejected previews receive no treatment credit']),'Original source-target coverage is not recorded for this alternative; residual volume describes the mapped task target','Functional outcomes and clinical risk unassessed'].join('; '),
    residualLocation:'Residual location not recorded for this alternative.',
    recommended:complete&&report.policies.completePlanAware.selectedPreparationId===row.id};
}
