import type {SurgicalResponse} from './surgical-types';

const object=(v:unknown):v is Record<string,unknown>=>!!v&&typeof v==='object'&&!Array.isArray(v);
const finite=(v:unknown):v is number=>typeof v==='number'&&Number.isFinite(v);
const strings=(v:unknown):v is string[]=>Array.isArray(v)&&v.every(s=>typeof s==='string');
const digest=(v:unknown):v is string=>typeof v==='string'&&/^sha256:[a-f0-9]{64}$/.test(v);
export interface ConsequenceDimension {unit:string;direction:'min'|'max';status:'source_count'|'modeled_unvalidated'|'unavailable';cumulative:number|null;incremental:number|null;reason:string|null}
export interface ConsequenceSample {branchId:string;scenarioId:string;vector:{sourceSceneId:string;stateId:string;nativeActions:number;cleanupComplete:boolean;clinicalValidation:false;residualTargetMm3:number|null;dimensions:Record<string,ConsequenceDimension>}}
export interface ConsequenceConstraint {metric:string;scope:'cumulative'|'incremental';operator:'at_most'|'at_least';value:number}
export interface ConsequencePolicy {id:string;basis:string;objectives:string[];requiredMetrics:string[];constraints?:ConsequenceConstraint[]}
export interface ConsequenceViolation {scenario:string;reason:string;metric?:string;scope?:'cumulative'|'incremental';operator?:'at_most'|'at_least';value?:number;actual?:number}
export interface ConsequenceDecision {policy:ConsequencePolicy;feasibleContinuations?:string[];dominatedBy?:Record<string,string[]>;scenarios?:string[];objectiveScopes?:Record<string,string>;dominanceMeaning?:string;selectedBranchId:string|null;stopFallback:boolean;stopExecutable:boolean;taskSatisfied:boolean;pairedScenarioParetoFront:string[];violations:Record<string,ConsequenceViolation[]>;reason:string}
export interface ConsequenceComparison {schema:string;sourceSceneId:string;comparisonSeal:string;baselineBranchId:string;stopBranchId:string;sourceRole:'DEVELOPMENT';declaration?:unknown;scenarios:Array<{id:string;sourceSceneId:string;relaxationTimeScale:number;interpretation:string}>;samples:ConsequenceSample[];decisions:ConsequenceDecision[];limitations:string[]}

/** Bridge checks source, checkpoint ledgers and decision algebra. A seal is
 * integrity metadata, not independent physical or counterfactual validation. */
export function consequenceComparisonData(response:SurgicalResponse|null):ConsequenceComparison|null{
  const r=response?.consequenceComparison;
  if(!response||!object(r)||r.schema!=='resectionlab.surgical-consequence-comparison.v1'||
      r.sourceSceneId!==response.view.scene.sceneId||r.sourceRole!=='DEVELOPMENT'||
      !digest(r.comparisonSeal)||r.optimizerUpdates!==0||r.physicalValidationPassed!==false||r.physiologicalRL!=='hold'||
      !Array.isArray(r.scenarios)||r.scenarios.length<1||r.scenarios.length>8||
      !Array.isArray(r.samples)||r.samples.length<2||r.samples.length>64||
      !Array.isArray(r.decisions)||r.decisions.length<1||r.decisions.length>8||!strings(r.limitations))return null;
  const worlds=new Set<string>();
  for(const s of r.scenarios){
    if(!object(s)||typeof s.id!=='string'||worlds.has(s.id)||!digest(s.sourceSceneId)||
        !finite(s.relaxationTimeScale)||s.relaxationTimeScale<=0||typeof s.interpretation!=='string')return null;
    worlds.add(s.id);
  }
  if(!worlds.has('nominal'))return null;
  const pairs=new Set<string>(),branches=new Set<string>();
  for(const s of r.samples){
    if(!object(s)||typeof s.branchId!=='string'||typeof s.scenarioId!=='string'||!worlds.has(s.scenarioId)||!object(s.vector))return null;
    const v=s.vector,key=JSON.stringify([s.branchId,s.scenarioId]);
    if(pairs.has(key)||!digest(v.sourceSceneId)||!digest(v.stateId)||v.clinicalValidation!==false||
        typeof v.cleanupComplete!=='boolean'||!Number.isSafeInteger(v.nativeActions)||Number(v.nativeActions)<0||
        !object(v.dimensions)||v.residualTargetMm3!==null&&(!finite(v.residualTargetMm3)||v.residualTargetMm3<0))return null;
    if(v.sourceSceneId!==(r.scenarios.find(w=>object(w)&&w.id===s.scenarioId) as Record<string,unknown>|undefined)?.sourceSceneId)return null;
    if(s.scenarioId==='nominal'&&(v.sourceSceneId!==response.view.scene.sceneId||
        !response.branches.some(b=>b.id===s.branchId&&b.steps===v.nativeActions)))return null;
    for(const d of Object.values(v.dimensions)){
      if(!object(d)||typeof d.unit!=='string'||!['min','max'].includes(String(d.direction))||
          !['source_count','modeled_unvalidated','unavailable'].includes(String(d.status)))return null;
      if(d.status==='unavailable'?(d.cumulative!==null||d.incremental!==null):
          (!finite(d.cumulative)||!finite(d.incremental)||d.cumulative<0||d.incremental<0))return null;
    }
    pairs.add(key);branches.add(s.branchId);
  }
  if(pairs.size!==branches.size*worlds.size||typeof r.stopBranchId!=='string'||!branches.has(r.stopBranchId))return null;
  for(const d of r.decisions){
    if(!object(d)||!object(d.policy)||typeof d.policy.id!=='string'||d.policy.basis!=='declared_research_task_not_clinical_threshold'||
        !strings(d.policy.objectives)||!strings(d.policy.requiredMetrics)||!strings(d.pairedScenarioParetoFront)||
        d.selectedBranchId!==null&&(typeof d.selectedBranchId!=='string'||!branches.has(d.selectedBranchId))||
        typeof d.stopFallback!=='boolean'||typeof d.stopExecutable!=='boolean'||typeof d.taskSatisfied!=='boolean'||
        typeof d.reason!=='string'||!object(d.violations))return null;
    if(d.pairedScenarioParetoFront.some(b=>!branches.has(b))||Object.keys(d.violations).length!==branches.size)return null;
    for(const [branch,rows] of Object.entries(d.violations)){
      if(!branches.has(branch)||!Array.isArray(rows)||rows.length>256||rows.some(row=>!object(row)||
          typeof row.scenario!=='string'||!worlds.has(row.scenario)||typeof row.reason!=='string'||
          row.metric!==undefined&&typeof row.metric!=='string'))return null;
    }
  }
  return r as unknown as ConsequenceComparison;
}

export const consequenceEvidenceMetrics=['supplyDeficitSeconds','motorEncounterMm3','languageEncounterMm3','patientShiftErrorMm'] as const;
export type ConsequenceEvidenceMetric=typeof consequenceEvidenceMetrics[number];
export interface ConsequencePolicyInput {
  minimumAdditionalTargetMm3:number;
  maximumCumulativeNonTargetMm3:number;
  requiredMetrics:ConsequenceEvidenceMetric[];
  maxBrainStrainSquaredMm3S:number|null;
  maxBrainStressSquaredPa2Mm3S:number|null;
}
export interface ConsequenceCheckpoint {branchId:string;position:number;stateId:string}
export interface ConsequenceAnalysisRecord {
  schema:'resectionlab.surgical-consequence-user-analysis.v1';revision:number;caseHash:string;sourceSceneId:string;comparisonSeal:string;
  binding:{baseline:ConsequenceCheckpoint&{snapshotHash:string;checkpointHash:string;actionHash:string};programs:Array<{branchId:string;snapshotHash:string;checkpointHash:string;actionHash:string}>;scenariosHash:string;sourceGeometryHash:string;sourceArchiveSha256:string;declarationHash:string;samplesHash:string};
  policyInput:ConsequencePolicyInput;policy:ConsequencePolicy;decision:ConsequenceDecision;
  optimizerUpdates:0;newPhysicalActions:0;clinicalValidation:false;analysisSeal:string;
}
export interface ConsequenceAnalysisView {
  schema:'resectionlab.surgical-consequence-analysis-view.v1';caseHash:string;sourceSceneId:string;comparisonSeal:string;
  context:ConsequenceCheckpoint;available:boolean;reason:string|null;baseline:ConsequenceCheckpoint;
  latest:ConsequenceAnalysisRecord|null;current:boolean;revisionCount:number;
}
const nonnegative=(v:unknown):v is number=>finite(v)&&v>=0;
const exactKeys=(v:Record<string,unknown>,keys:string[])=>Object.keys(v).sort().join('|')===[...keys].sort().join('|');
export function validConsequencePolicyInput(v:unknown):v is ConsequencePolicyInput {
  return object(v)&&exactKeys(v,['minimumAdditionalTargetMm3','maximumCumulativeNonTargetMm3','requiredMetrics','maxBrainStrainSquaredMm3S','maxBrainStressSquaredPa2Mm3S'])&&
    nonnegative(v.minimumAdditionalTargetMm3)&&nonnegative(v.maximumCumulativeNonTargetMm3)&&strings(v.requiredMetrics)&&
    new Set(v.requiredMetrics).size===v.requiredMetrics.length&&v.requiredMetrics.every(x=>(consequenceEvidenceMetrics as readonly string[]).includes(x))&&
    (v.maxBrainStrainSquaredMm3S===null||nonnegative(v.maxBrainStrainSquaredMm3S))&&
    (v.maxBrainStressSquaredPa2Mm3S===null||nonnegative(v.maxBrainStressSquaredPa2Mm3S));
}
export function sameConsequencePolicyInput(a:ConsequencePolicyInput,b:ConsequencePolicyInput){
  return a.minimumAdditionalTargetMm3===b.minimumAdditionalTargetMm3&&a.maximumCumulativeNonTargetMm3===b.maximumCumulativeNonTargetMm3&&
    a.maxBrainStrainSquaredMm3S===b.maxBrainStrainSquaredMm3S&&a.maxBrainStressSquaredPa2Mm3S===b.maxBrainStressSquaredPa2Mm3S&&
    [...a.requiredMetrics].sort().join('|')===[...b.requiredMetrics].sort().join('|');
}
export function defaultConsequencePolicyInput(data:ConsequenceComparison):ConsequencePolicyInput|null {
  const minimum=data.decisions[0]?.policy.constraints?.find(c=>c.metric==='targetRemovedMm3'&&c.scope==='incremental'&&c.operator==='at_least')?.value;
  const budget=data.samples.find(s=>s.branchId===data.stopBranchId&&s.scenarioId==='nominal')?.vector.dimensions.nonTargetRemovedMm3?.cumulative;
  return nonnegative(minimum)&&nonnegative(budget)?{minimumAdditionalTargetMm3:minimum,maximumCumulativeNonTargetMm3:budget,requiredMetrics:[],maxBrainStrainSquaredMm3S:null,maxBrainStressSquaredPa2Mm3S:null}:null;
}
const checkpoint=(v:unknown):v is ConsequenceCheckpoint=>object(v)&&typeof v.branchId==='string'&&Number.isSafeInteger(v.position)&&Number(v.position)>=0&&digest(v.stateId);
const sameCheckpoint=(a:ConsequenceCheckpoint,b:ConsequenceCheckpoint)=>a.branchId===b.branchId&&a.position===b.position&&a.stateId===b.stateId;
function validAnalysisDecision(v:unknown,data:ConsequenceComparison,input:ConsequencePolicyInput):v is ConsequenceDecision {
  if(!object(v)||!object(v.policy)||!strings(v.policy.objectives)||v.policy.objectives.join('|')!=='targetRemovedMm3|nonTargetRemovedMm3|elapsedSeconds'||
    v.policy.basis!=='declared_research_task_not_clinical_threshold'||typeof v.policy.id!=='string'||!strings(v.policy.requiredMetrics)||
    [...v.policy.requiredMetrics].sort().join('|')!==[...input.requiredMetrics].sort().join('|')||!Array.isArray(v.policy.constraints))return false;
  const expected:ConsequenceConstraint[]=[{metric:'targetRemovedMm3',scope:'incremental',operator:'at_least',value:input.minimumAdditionalTargetMm3},
    {metric:'nonTargetRemovedMm3',scope:'cumulative',operator:'at_most',value:input.maximumCumulativeNonTargetMm3}];
  if(input.maxBrainStrainSquaredMm3S!==null)expected.push({metric:'brainStrainSquaredMm3S',scope:'cumulative',operator:'at_most',value:input.maxBrainStrainSquaredMm3S});
  if(input.maxBrainStressSquaredPa2Mm3S!==null)expected.push({metric:'brainStressSquaredPa2Mm3S',scope:'cumulative',operator:'at_most',value:input.maxBrainStressSquaredPa2Mm3S});
  const constraints=v.policy.constraints;
  if(constraints.length!==expected.length||expected.some(c=>!constraints.some((x:unknown)=>object(x)&&exactKeys(x,['metric','scope','operator','value'])&&x.metric===c.metric&&x.scope===c.scope&&x.operator===c.operator&&x.value===c.value)))return false;
  const branches=new Set(data.samples.map(s=>s.branchId)),worlds=data.scenarios.map(s=>s.id);
  if(!strings(v.feasibleContinuations)||!strings(v.pairedScenarioParetoFront)||!strings(v.scenarios)||[...v.scenarios].sort().join('|')!==[...worlds].sort().join('|')||
    !object(v.violations)||!object(v.dominatedBy)||!object(v.objectiveScopes)||typeof v.dominanceMeaning!=='string'||typeof v.reason!=='string'||
    typeof v.stopFallback!=='boolean'||typeof v.stopExecutable!=='boolean'||typeof v.taskSatisfied!=='boolean'||
    v.selectedBranchId!==null&&(typeof v.selectedBranchId!=='string'||!branches.has(v.selectedBranchId))||Object.keys(v.violations).length!==branches.size)return false;
  const eligible=v.feasibleContinuations as string[],front=v.pairedScenarioParetoFront as string[];
  if(new Set(eligible).size!==eligible.length||new Set(front).size!==front.length||eligible.some(b=>!branches.has(b)||b===data.stopBranchId)||front.some(b=>!eligible.includes(b)))return false;
  const expectedSelected=front.length===1?front[0]:v.stopExecutable?data.stopBranchId:null;
  if(v.selectedBranchId!==expectedSelected||v.stopFallback!==(v.selectedBranchId===data.stopBranchId)||v.taskSatisfied!==(typeof v.selectedBranchId==='string'&&eligible.includes(v.selectedBranchId))||
    v.selectedBranchId!==null&&v.selectedBranchId!==data.stopBranchId&&(front.length!==1||front[0]!==v.selectedBranchId)||v.stopFallback&&!v.stopExecutable)return false;
  for(const [branch,rows] of Object.entries(v.violations)){
    if(!branches.has(branch)||!Array.isArray(rows)||rows.length>256||rows.some(row=>!object(row)||typeof row.scenario!=='string'||!worlds.includes(row.scenario)||typeof row.reason!=='string'||row.metric!==undefined&&typeof row.metric!=='string'||
      row.reason==='declared_research_constraint'&&(!nonnegative(row.actual)||!nonnegative(row.value)||!['cumulative','incremental'].includes(String(row.scope))||!['at_most','at_least'].includes(String(row.operator)))))return false;
    if(branch!==data.stopBranchId&&(eligible.includes(branch)!==(rows.length===0)))return false;
  }
  if(Object.keys(v.dominatedBy).sort().join('|')!==[...eligible].sort().join('|')||Object.values(v.dominatedBy).some(rows=>!strings(rows)||rows.some(b=>!eligible.includes(b))))return false;
  return v.objectiveScopes.targetRemovedMm3==='incremental'&&v.objectiveScopes.nonTargetRemovedMm3==='cumulative'&&v.objectiveScopes.elapsedSeconds==='cumulative';
}
/** Reject stale bridge responses and keep historical analyses explicitly separate
 * from a decision at the accepted common checkpoint. Hashes are bridge-verified
 * integrity bindings, not evidence of physiological validation. */
export function consequenceAnalysisData(response:SurgicalResponse|null,data=consequenceComparisonData(response)):ConsequenceAnalysisView|null {
  const v=response?.consequenceAnalysis;
  if(!response||!data||!object(v)||v.schema!=='resectionlab.surgical-consequence-analysis-view.v1'||v.caseHash!==response.caseHash||v.sourceSceneId!==response.view.scene.sceneId||
    v.comparisonSeal!==data.comparisonSeal||!checkpoint(v.context)||!sameCheckpoint(v.context,{branchId:response.branchId,position:response.position,stateId:response.view.stateId})||
    !checkpoint(v.baseline)||v.baseline.branchId!==data.baselineBranchId||!response.branches.some(b=>b.id===(v.baseline as ConsequenceCheckpoint).branchId&&b.steps===(v.baseline as ConsequenceCheckpoint).position)||
    typeof v.available!=='boolean'||typeof v.current!=='boolean'||!(v.reason===null||typeof v.reason==='string')||!Number.isSafeInteger(v.revisionCount)||Number(v.revisionCount)<0)return null;
  const atBaseline=sameCheckpoint(v.context,v.baseline)&&!response.playback&&!response.readOnly&&response.verification?.status!=='unverified-import';
  if(v.available&&!atBaseline)return null;
  if(v.latest===null)return v.current||v.revisionCount!==0?null:v as unknown as ConsequenceAnalysisView;
  const r=v.latest;
  if(!object(r)||r.schema!=='resectionlab.surgical-consequence-user-analysis.v1'||r.caseHash!==response.caseHash||r.sourceSceneId!==data.sourceSceneId||r.comparisonSeal!==data.comparisonSeal||
    !Number.isSafeInteger(r.revision)||Number(r.revision)<1||r.revision!==v.revisionCount||!validConsequencePolicyInput(r.policyInput)||
    r.optimizerUpdates!==0||r.newPhysicalActions!==0||r.clinicalValidation!==false||!digest(r.analysisSeal)||!object(r.binding)||!checkpoint(r.binding.baseline)||!sameCheckpoint(r.binding.baseline,v.baseline)||
    !['snapshotHash','checkpointHash','actionHash'].every(k=>digest((r.binding as Record<string,unknown>).baseline&&((r.binding as Record<string,unknown>).baseline as Record<string,unknown>)[k]))||
    !['scenariosHash','sourceGeometryHash','declarationHash','samplesHash'].every(k=>digest((r.binding as Record<string,unknown>)[k]))||typeof r.binding.sourceArchiveSha256!=='string'||!/^[a-f0-9]{64}$/.test(r.binding.sourceArchiveSha256)||!Array.isArray(r.binding.programs))return null;
  const programs=r.binding.programs,branches=[...new Set(data.samples.map(s=>s.branchId))];
  if(programs.length!==branches.length||new Set(programs.map(p=>object(p)?p.branchId:null)).size!==branches.length||programs.some(p=>!object(p)||typeof p.branchId!=='string'||!branches.includes(p.branchId)||!['snapshotHash','checkpointHash','actionHash'].every(k=>digest(p[k]))))return null;
  if(!validAnalysisDecision(r.decision,data,r.policyInput)||JSON.stringify(r.policy)!==JSON.stringify(r.decision.policy)||v.current&&(!v.available||!atBaseline))return null;
  return v as unknown as ConsequenceAnalysisView;
}
