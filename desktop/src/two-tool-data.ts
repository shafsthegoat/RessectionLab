import type {InteractivePlanningConstraints,InteractivePlanningRoute,InteractivePlanningLimit} from './interactive-planning-data';
export interface EncounterControls {layerId:string;marginMm:number;maxNearMm3:number}
export interface AnatomicalEncounter {marginMm:number;maxNearMm3:number;nearMm3:number;shaftNearMm3:number;workingNearMm3:number;baselineRemovedContextMm3:number;newRemovedContextMm3:number;baselineRetainedContextMm3:number;withinLimit:boolean;scope:string;nearPositionsMm?:[number,number,number][]}
export interface TwoToolControls {geometry:InteractivePlanningConstraints;shortSeconds:number;longSeconds:number;minimumTargetMm3:number;encounter?:EncounterControls}
export type TwoToolPublicControls=Omit<TwoToolControls,'encounter'>&{encounter?:Omit<EncounterControls,'layerId'>};
export interface TwoToolSummary {additionalTargetMm3:number;additionalNonTargetMm3:number;elapsedSeconds:number;cusaPendingMm3:number;cusaActivationSeconds:number;fluidRemovedMm3:number;residualTargetMm3:number;removedSourceCells:number[]}
export interface TwoToolAlternative {id:string;label:string;contactSeconds?:number;feasible:boolean;taskProgressMet:boolean;summary:TwoToolSummary;actionProgram:unknown[];phases:Array<{id:string;startAction:number;endAction:number;contextHash?:string}>;reassessment?:{probeStatus:string;probeReason:string|null};visibility?:null;clinicalRisk?:null;anatomicalEncounter?:AnatomicalEncounter}
export interface ContextOption {layerId:string;labelValue:number;description:string;reviewStatus:string;coverage:string;scope:string}
export interface TwoToolAvailability {available:boolean;reason:string|null;scope:string;defaults:TwoToolControls|null;contextOptions?:ContextOption[]}
export interface TwoToolComparison {schema:string;reportSeal:string;sourceSceneId:string;startStateId:string;prefixActions:number;sourceBranchId?:string;sourcePosition?:number;current?:boolean;revision?:number;contextBinding?:{layerId:string;sourceLabel?:string;labelValue:number;reviewStatus:string;coverage:string;bindingSeal:string};publicPlan:{publicPlanSeal:string;publicObservation:{inputHash:string};constraints:TwoToolPublicControls;scope:string;alternatives:TwoToolAlternative[];decision:{operation:string;selectedId:string;reason:string;rule:string};rejections:Array<Record<string,unknown>>;chosenPair:{route:InteractivePlanningRoute}|null;approaches:{limitingStructures?:InteractivePlanningLimit[];missingRequiredEvidence?:string[]}};applications?:Array<{branchId:string;candidateId:string;receipt:{phases:Array<{id:string;position:number}>;anatomicalEncounter?:AnatomicalEncounter}}>}
export function strategyControls(report:TwoToolComparison):TwoToolControls {
  const result=structuredClone(report.publicPlan.constraints);
  const {encounter,...controls}=result;
  if(!encounter)return controls;
  if(!report.contextBinding)throw new Error('Saved source-context comparison has no native source binding');
  return {...controls,encounter:{...encounter,layerId:report.contextBinding.layerId}};
}
export const strategyControlKey=(c:TwoToolControls)=>JSON.stringify([c.shortSeconds,c.longSeconds,c.minimumTargetMm3,c.encounter?[c.encounter.layerId,c.encounter.marginMm,c.encounter.maxNearMm3]:null,
  Object.entries(c.geometry).sort(([a],[b])=>a.localeCompare(b)).map(([k,v])=>[k,k==='requiredEvidence'?[...(v as string[])].sort():v])]);
