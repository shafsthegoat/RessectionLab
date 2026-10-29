import type {SurgicalResponse} from './surgical-types';

const object=(v:unknown):v is Record<string,unknown>=>!!v&&typeof v==='object'&&!Array.isArray(v);
const finite=(v:unknown):v is number=>typeof v==='number'&&Number.isFinite(v);
const strings=(v:unknown):v is string[]=>Array.isArray(v)&&v.every(s=>typeof s==='string');
const digest=(v:unknown):v is string=>typeof v==='string'&&/^sha256:[a-f0-9]{64}$/.test(v);
export interface ConsequenceDimension {unit:string;direction:'min'|'max';status:'source_count'|'modeled_unvalidated'|'unavailable';cumulative:number|null;incremental:number|null;reason:string|null}
export interface ConsequenceSample {branchId:string;scenarioId:string;vector:{sourceSceneId:string;stateId:string;nativeActions:number;cleanupComplete:boolean;clinicalValidation:false;residualTargetMm3:number|null;dimensions:Record<string,ConsequenceDimension>}}
export interface ConsequenceDecision {policy:{id:string;basis:string;objectives:string[];requiredMetrics:string[]};selectedBranchId:string|null;stopFallback:boolean;stopExecutable:boolean;taskSatisfied:boolean;pairedScenarioParetoFront:string[];violations:Record<string,Array<{scenario:string;reason:string;metric?:string}>>;reason:string}
export interface ConsequenceComparison {schema:string;sourceSceneId:string;comparisonSeal:string;baselineBranchId:string;stopBranchId:string;sourceRole:'DEVELOPMENT';scenarios:Array<{id:string;sourceSceneId:string;relaxationTimeScale:number;interpretation:string}>;samples:ConsequenceSample[];decisions:ConsequenceDecision[];limitations:string[]}

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
