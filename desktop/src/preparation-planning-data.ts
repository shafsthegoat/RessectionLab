import {planningBinding} from './interactive-planning-data.ts';
import {inverseAffine,rasAffine,transformPoint} from './viewer/coordinates.ts';
import type {Point3,Affine,Shape3} from './viewer/coordinates';
import type {SurgicalResponse} from './surgical-types';
import type {ContextOption,EncounterControls,TwoToolComparison,TwoToolAlternative} from './two-tool-data';

export const PREPARATION_SCHEMA='resectionlab.source-preparation-comparison.v1';
export const PREPARATION_SCHEMA_V2='resectionlab.source-preparation-comparison.v2';
export type PreparationFamily='cardinal-v1'|'paired-lanes-v1';
export type PreparationPhase='prepared'|'withdraw';
export interface PreparationRay {surfaceMm:Point3;sourceTargetMm:Point3}
export interface PreparationControls {
  preparationFamily?:PreparationFamily;
  sourceRay:PreparationRay|null;target:{centerMm:Point3;radiusMm:number}|null;widthsMm:number[];
  maxNonTargetMm3:number;maxActions:number;maxSeconds:number;entryRadiusMm:number;
  shortSeconds:number;longSeconds:number;minimumTargetMm3:number;requiredEvidence:string[];
  encounter:EncounterControls|null;
}
export interface PreparationSuggestion {sourceSceneId:string;startStateId:string;sourceRay:PreparationRay;target:{centerMm:Point3;radiusMm:number};entryRadiusMm:number}
export interface PreparationAvailability {available:boolean;reason:string|null;scope:string;defaults:PreparationControls|null;contextOptions?:ContextOption[]}
export interface PreparationSummary {additionalTargetMm3:number;additionalNonTargetMm3:number;elapsedSeconds:number;withdrawn:boolean}
export interface PreparationAttempt {complete:boolean;failure:string|null;summary:PreparationSummary;actionProgram:unknown[];physicalStateHashes:string[]}
export interface PreparationProof {schema:string;preparationSeal:string;publicPlan:{family?:PreparationFamily;attempts:PreparationAttempt[];decision:{operation:string;selectedIndex:number|null};scope:string}}
export type PreparationSuffixComparison=TwoToolComparison&{publicPlan:TwoToolComparison['publicPlan']&{approaches:TwoToolComparison['publicPlan']['approaches']&{routes?:unknown[];rejections?:Array<{reason?:string}>}}};
export interface PreparationRow {id:string;widthMm:number;preparation:PreparationProof|null;preparedAdmission:unknown|null;comparison:PreparationSuffixComparison|null;reason:string|null;missingEvidence?:string[]}
export interface PreparationApplication {branchId:string;preparationBranchId:string;revision:number;candidateId:string;widthMm:number;phase:PreparationPhase;preparationFamily?:PreparationFamily;preparationSummary:PreparationSummary;receipt:Record<string,unknown>}
export interface PreparationComparison {
  schema:string;reportSeal:string;sourceSceneId:string;startStateId:string;sourceBranchId:string;sourcePosition:number;
  prefixActions:number;sourceSnapshotHash?:string;current:boolean;revision:number;controls:PreparationControls;rows:PreparationRow[];
  baselineBurden?:{targetRemovedMm3:number;nonTargetRemovedMm3:number;elapsedSeconds:number};
  applications?:PreparationApplication[];
  policies:{corridorFirst:{selectedPreparationId:string|null;suffixId:string};completePlanAware:{selectedPreparationId:string|null;suffixId:string}};scope:string;
}
export function preparationBaselineText(report:PreparationComparison):string|null {
  const b=report.baselineBurden;
  if(!b||![b.targetRemovedMm3,b.nonTargetRemovedMm3,b.elapsedSeconds].every(x=>typeof x==='number'&&Number.isFinite(x)&&x>=0))return null;
  return `Prior recorded burdens: ${b.targetRemovedMm3.toFixed(2)} mm³ target and ${b.nonTargetRemovedMm3.toFixed(2)} mm³ non-target removed; ${b.elapsedSeconds.toFixed(2)} s elapsed. These persist before the additional preparation and sequence costs below.`;
}
export interface PreparationResponse extends SurgicalResponse {
  preparationPlanningAvailability?:PreparationAvailability;preparationSuggestion?:PreparationSuggestion|null;
  preparationPlanning?:PreparationComparison|null;
}
export interface PreparationPreview {kind?:string;phaseId?:string;candidateId:string;reportSeal:string;view:object;receipt?:Record<string,unknown>}
export interface PreparationSubmission {binding:string;revision:number}

const finitePoint=(p:unknown):p is Point3=>Array.isArray(p)&&p.length===3&&p.every(n=>typeof n==='number'&&Number.isFinite(n));
const bounded=(n:number,low:number,high:number)=>Number.isFinite(n)&&n>=low&&n<=high;
const familyValid=(family:unknown)=>family===undefined||family==='cardinal-v1'||family==='paired-lanes-v1';
export function preparationFamilyName(family:PreparationFamily|undefined):string {
  return family==='paired-lanes-v1'?'Paired lanes':family===undefined||family==='cardinal-v1'?'Cardinal local sweeps':'Unsupported preparation family';
}
export function preparationWidthPresets(family:PreparationFamily|undefined):number[] {
  return family==='paired-lanes-v1'?[0,1.71,3.42]:[0,.3,1];
}
export function preparationWidthOptions(controls:PreparationControls):number[] {
  return [...new Set([...preparationWidthPresets(controls.preparationFamily),...controls.widthsMm.filter(Number.isFinite)])].sort((a,b)=>a-b);
}
/** Only an explicit operator change adds a family or resets width choices. */
export function changePreparationFamily(controls:PreparationControls,family:PreparationFamily):PreparationControls {
  if(family!=='cardinal-v1'&&family!=='paired-lanes-v1')throw new Error('Unsupported preparation family');
  return {...structuredClone(controls),preparationFamily:family,widthsMm:preparationWidthPresets(family)};
}
export function preparationApplicationLabel(app:Pick<PreparationApplication,'widthMm'|'preparationFamily'>):string {
  return app.preparationFamily==='paired-lanes-v1'?`Inspect installed paired lanes at ${app.widthMm} mm separation`:`Inspect installed ${app.widthMm} mm preparation`;
}
function preparationReportSchemaMatches(report:PreparationComparison):boolean {
  return familyValid(report.controls.preparationFamily)&&report.schema===('preparationFamily' in report.controls?PREPARATION_SCHEMA_V2:PREPARATION_SCHEMA);
}
function preparationProofMatches(row:PreparationRow|undefined,family:PreparationFamily|undefined):boolean {
  return familyValid(family)&&row?.preparation?.schema===(family==='paired-lanes-v1'?'resectionlab.source-exposure-preparation.v3':'resectionlab.source-exposure-preparation.v2')&&row.preparation.publicPlan.family===(family==='paired-lanes-v1'?'paired-lanes-v1':undefined);
}
export function preparationLocked(response:PreparationResponse|null|undefined,pending=false):boolean {
  return pending||!response||!!response.readOnly||!!response.playback||!!response.verification?.readOnly;
}
/** All controls bind authority; width order also determines canonical row IDs. */
export function preparationControlKey(controls:PreparationControls):string {
  const canonical=(v:unknown):unknown=>Array.isArray(v)?v.map(canonical):v&&typeof v==='object'?
    Object.fromEntries(Object.entries(v).sort(([a],[b])=>a.localeCompare(b)).map(([k,x])=>[k,canonical(x)])):v;
  return JSON.stringify(canonical({...controls,requiredEvidence:[...controls.requiredEvidence].sort()}));
}
export function preparationDraftIssue(c:PreparationControls,contexts?:ContextOption[]):string|null {
  if(!familyValid(c.preparationFamily)||('preparationFamily' in c&&c.preparationFamily===undefined))return 'Choose a supported preparation family.';
  if(!c.sourceRay||!finitePoint(c.sourceRay.surfaceMm)||!finitePoint(c.sourceRay.sourceTargetMm))return 'Choose source surface and target-boundary cells in RAS millimeters.';
  if(Math.hypot(...c.sourceRay.surfaceMm.map((n,i)=>n-c.sourceRay!.sourceTargetMm[i]))<=3.5)return 'This preparation family requires a source ray longer than 3.5 mm.';
  if(!c.target||!finitePoint(c.target.centerMm)||!bounded(c.target.radiusMm,.5,15))return 'Choose a common target region with a radius from 0.5 to 15 mm.';
  const widthMax=c.preparationFamily==='paired-lanes-v1'?4:1;
  if(!c.widthsMm.length||c.widthsMm.length>3||new Set(c.widthsMm).size!==c.widthsMm.length||c.widthsMm.some(w=>!bounded(w,0,widthMax)))return `Select one to three distinct ${c.preparationFamily==='paired-lanes-v1'?'lane separations':'local sweeps'} from 0 to ${widthMax} mm.`;
  if(!bounded(c.maxNonTargetMm3,Number.MIN_VALUE,200)||!Number.isInteger(c.maxActions)||!bounded(c.maxActions,6,128)||!bounded(c.maxSeconds,2,180))return 'Preparation limits must be positive and at most 200 mm³, 128 actions and 180 seconds.';
  if(!bounded(c.entryRadiusMm,.5,15))return 'The shared entry-patch radius must be from 0.5 to 15 mm.';
  if(!bounded(c.shortSeconds,.1,2)||!bounded(c.longSeconds,.1,2)||c.shortSeconds>=c.longSeconds||!bounded(c.minimumTargetMm3,0,8))return 'Use increasing contact durations from 0.1 to 2 seconds and target progress from 0 to 8 mm³.';
  if(!Array.isArray(c.requiredEvidence)||c.requiredEvidence.some(x=>typeof x!=='string'||!x)||new Set(c.requiredEvidence).size!==c.requiredEvidence.length||!['brain','target'].every(x=>c.requiredEvidence.includes(x)))return 'Brain and target evidence must remain required.';
  if(c.encounter&&(!c.encounter.layerId||!bounded(c.encounter.marginMm,0,5)||!bounded(c.encounter.maxNearMm3,0,1000)||contexts&&!contexts.some(x=>x.layerId===c.encounter!.layerId)))return 'Choose available supplied context, a margin from 0 to 5 mm and a near-volume limit from 0 to 1000 mm³.';
  return null;
}
export function currentPreparationReport(response:PreparationResponse|null|undefined,draft:PreparationControls,submitted:PreparationSubmission|null,revision:number):PreparationComparison|null {
  const report=response?.preparationPlanning;
  return response&&report&&preparationReportSchemaMatches(report)&&report.current===true&&!preparationLocked(response)&&
    submitted?.binding===planningBinding(response)&&submitted.revision===revision&&
    report.sourceSceneId===response.view.scene.sceneId&&report.startStateId===response.view.stateId&&
    report.sourcePosition===response.position&&report.prefixActions===response.position&&
    !preparationDraftIssue(draft)&&preparationControlKey(report.controls)===preparationControlKey(draft)&&
    report.rows.every(row=>row.preparation===null||preparationProofMatches(row,report.controls.preparationFamily))?report:null;
}
export function currentPreparationSuggestion(response:PreparationResponse|null|undefined):PreparationSuggestion|null {
  const s=response?.preparationSuggestion;
  return response&&!preparationLocked(response)&&s?.sourceSceneId===response.view.scene.sceneId&&s.startStateId===response.view.stateId&&
    finitePoint(s.sourceRay?.surfaceMm)&&finitePoint(s.sourceRay?.sourceTargetMm)&&finitePoint(s.target?.centerMm)&&
    bounded(s.target.radiusMm,.5,15)&&bounded(s.entryRadiusMm,.5,15)?s:null;
}
export function preparationAttempt(row:PreparationRow|undefined):PreparationAttempt|null {
  const plan=row?.preparation?.publicPlan,index=plan?.decision.selectedIndex;
  return plan&&typeof index==='number'&&Number.isInteger(index)&&index>=0?plan.attempts[index]??null:null;
}
/** Failed attempts have costs too, but cannot authorize preview or install. */
export function preparationCost(row:PreparationRow|undefined):PreparationSummary|null {
  return (preparationAttempt(row)??row?.preparation?.publicPlan.attempts.at(-1))?.summary??null;
}
export function preparationSuffix(row:PreparationRow|undefined):TwoToolAlternative|null {
  const plan=row?.comparison?.publicPlan,id=plan?.decision.selectedId;
  return plan&&id&&id!=='stop'?plan.alternatives.find(x=>x.id===id)??null:null;
}
export function preparationPhaseAvailable(row:PreparationRow|undefined,phase:PreparationPhase,family?:PreparationFamily):boolean {
  const attempt=preparationAttempt(row);
  if(!preparationProofMatches(row,family)||row?.preparation?.publicPlan.decision.operation!=='PREPARE'||
    !attempt||attempt.complete!==true||attempt.failure!==null||attempt.summary.withdrawn!==true||!row.preparedAdmission||
    attempt.summary.additionalTargetMm3!==0||!Number.isFinite(attempt.summary.additionalNonTargetMm3)||attempt.summary.additionalNonTargetMm3<=0||
    !Number.isFinite(attempt.summary.elapsedSeconds)||attempt.summary.elapsedSeconds<=0)return false;
  if(phase==='prepared')return true;
  const suffix=preparationSuffix(row);
  return row.comparison?.publicPlan.decision.operation==='COMPARE'&&!!suffix&&suffix.feasible===true&&suffix.taskProgressMet===true&&(suffix as TwoToolAlternative&{withdrawal?:string}).withdrawal==='complete';
}
export function preparationPreviewMatches(report:PreparationComparison|null,row:PreparationRow|undefined,phase:PreparationPhase,preview:PreparationPreview|null|undefined,displayed:object|null|undefined):boolean {
  return !!report&&preparationReportSchemaMatches(report)&&!!row&&!!displayed&&!!preview&&preview.kind==='preparation'&&preview.phaseId===phase&&
    preview.candidateId===row.id&&preview.reportSeal===report.reportSeal&&preview.view===displayed&&preparationPhaseAvailable(row,phase,report.controls.preparationFamily);
}
export function preparationActions(input:{response:PreparationResponse|null|undefined;draft:PreparationControls;submitted:PreparationSubmission|null;revision:number;candidateId:string|null|undefined;phase:PreparationPhase;preview:PreparationPreview|null|undefined;displayed:object|null|undefined;pending:boolean;acknowledged:boolean;contexts?:ContextOption[]}) {
  const report=currentPreparationReport(input.response,input.draft,input.submitted,input.revision);
  const row=report?.rows.find(r=>r.id===input.candidateId);
  const matching=preparationPreviewMatches(report,row,input.phase,input.preview,input.displayed);
  const ray=input.draft.sourceRay;
  const cells=!!input.response&&!!ray&&[ray.surfaceMm,ray.sourceTargetMm].every(p=>preparationPointIsCell(p,input.response!.view.scene));
  const previewAllowed=!!report&&!!row&&!preparationLocked(input.response,input.pending)&&input.acknowledged&&cells&&!preparationDraftIssue(input.draft,input.contexts)&&
    preparationPhaseAvailable(row,input.phase,input.draft.preparationFamily)&&(!input.displayed||matching);
  return {report,row,matchingPreview:matching,previewAllowed,installAllowed:previewAllowed&&matching};
}
export function preparationIncomplete(row:PreparationRow):boolean {
  const p=row.comparison?.publicPlan;
  const reasons=[row.reason,p?.decision.reason,...(p?.rejections??[]).map(r=>r.reason),...(p?.approaches.rejections??[]).map(r=>r.reason)];
  return reasons.some(r=>r==='COMPUTATION_BUDGET'||r==='GEOMETRY_WALL_BUDGET');
}
export function snapPreparationPoint(point:Point3,scene:{shape:Shape3;affine:Affine}):Point3|null {
  if(!finitePoint(point)||scene.shape.length!==3||scene.shape.some(n=>!Number.isInteger(n)||n<=0))return null;
  try {
    const affine=rasAffine(scene.affine,'RAS+'),voxel=transformPoint(inverseAffine(affine),point).map(Math.round) as Point3;
    if(voxel.some((n,i)=>n<0||n>=scene.shape[i]))return null;
    return transformPoint(affine,voxel);
  } catch {return null;}
}
export function preparationPointIsCell(point:Point3,scene:{shape:Shape3;affine:Affine}):boolean {
  const snapped=snapPreparationPoint(point,scene);
  return !!snapped&&snapped.every((n,i)=>Math.abs(n-point[i])<=1e-8);
}
