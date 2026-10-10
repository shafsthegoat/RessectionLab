import {sha256Bytes} from './source-integrity.ts';
import type {DevelopmentEpisode} from './episode-types';
import type {EpisodeVascularResult,EpisodeVascularEvaluation,VascularContactCount} from './episode-vascular-types';
const HASH=/^sha256:[a-f0-9]{64}$/;
const stable=(value:unknown):unknown=>Array.isArray(value)?value.map(stable):value&&typeof value==='object'?Object.fromEntries(Object.entries(value).sort(([a],[b])=>a<b?-1:a>b?1:0).map(([k,v])=>[k,stable(v)])):value;
const same=(a:unknown,b:unknown)=>JSON.stringify(stable(a))===JSON.stringify(stable(b));
function requireValue(value:unknown,message:string):asserts value{if(!value)throw new Error(`Vascular evaluation withheld: ${message}`)}
function contact(row:VascularContactCount){
 requireValue(row&&typeof row==='object','missing contact report');
 const p=row.positive_reference_cells,u=row.unknown_reference_cells,t=row.touched_reference_cells,o=row.outside_reference_fov;
 requireValue([p,u,t].every(n=>Number.isSafeInteger(n)&&n>=0)&&p+u<=t&&typeof o==='boolean','invalid cell counts');
 requireValue(row.annotated_positive_encounter===(p?true:u||o?null:false)&&row.annotation_coverage_complete_for_sweep===(!u&&!o),'inconsistent unknown coverage');
 requireValue(row.biological_vessel_free===null&&row.clinical_injury_probability===null,'unsupported biological outcome');
 requireValue([row.positive_cell_volume_upper_bound_mm3,row.unknown_in_grid_cell_volume_mm3].every(n=>Number.isFinite(n)&&n>=0),'invalid contact volume');
}
/** Scalar sidecar only; the existing episode and all replay masks stay unchanged. */
export async function checkedVascularEvaluation(input:EpisodeVascularResult,expected:DevelopmentEpisode):Promise<EpisodeVascularEvaluation>{
 const result=structuredClone(input),episode=structuredClone(expected),e=result.evaluation;
 requireValue(new TextEncoder().encode(JSON.stringify(result)).length<=512*1024,'report exceeds budget');
 requireValue(typeof result.evaluationCanonicalJson==='string'&&e?.evaluationId===`sha256:${await sha256Bytes(new TextEncoder().encode(result.evaluationCanonicalJson))}`,'result serialization changed');
 const {evaluationId,...body}=e;requireValue(same(JSON.parse(result.evaluationCanonicalJson),body),'result differs from bound serialization');
 requireValue(e.schema==='generated-shared-vascular-encounter-v1'&&e.status==='evaluated_generated_vascular_reference'&&e.scope==='generated_geometry_annotation_contact_only'&&e.patientAdmission===false&&e.clinicalInjuryProbability===null,'unsupported evaluation scope');
 requireValue([e.evaluationId,e.physicalHistoryHash,e.referenceBindingHash].every(h=>HASH.test(h))&&result.caseHash===episode.caseHash&&e.caseHash===episode.caseHash&&e.episodeId===episode.episodeId&&e.sourceHash===episode.sourceHash&&e.decisionModelHash===episode.decisionModelHash&&e.strategySeal===episode.planning.strategySeal&&episode.planning.sealedBeforeReferenceScoring===true,'result belongs to another executed strategy');
 requireValue(typeof e.physicalHistoryCanonicalJson==='string'&&e.physicalHistoryHash===`sha256:${await sha256Bytes(new TextEncoder().encode(e.physicalHistoryCanonicalJson))}`&&same(JSON.parse(e.physicalHistoryCanonicalJson),episode.history),'physical history differs from this replay');
 requireValue(same(e.actionIds,episode.history.map(row=>row.action_id))&&Array.isArray(e.perAction)&&e.perAction.length===episode.history.length,'action inventory differs from replay');
 for(const [i,row] of e.perAction.entries()){
  const action=episode.history[i];requireValue(row.actionIndex===i&&row.actionId===action.action_id&&row.interactionMode===action.interaction_mode,'action ordinal differs from replay');
  if(action.interaction_mode==='stop')requireValue(row.sweepCount===0&&row.shaft===null&&row.tip===null&&row.wholeTool===null,'STOP contains a tool sweep');
  else{requireValue(row.sweepCount===1&&row.shaft&&row.tip&&row.wholeTool,'missing complete tool sweep');contact(row.shaft);contact(row.tip);contact(row.wholeTool)}
 }
 contact(e.shaft);contact(e.tip);contact(e.wholeTool);
 requireValue(e.removedOverlap?.status==='not_evaluated_by_contact_kernel'&&e.removedOverlap.outcomes===null,'tool contact is not removed overlap');
 return e;
}
/** Do not install a late result after a different workspace or episode takes over. */
export async function loadVascularEvaluation(api:import('./episode-vascular-types').EpisodeVascularApi,episode:DevelopmentEpisode,isCurrent:()=>boolean):Promise<EpisodeVascularEvaluation>{
 const selected=structuredClone(episode);
 const current=()=>{if(!isCurrent())throw new DOMException('A newer episode replaced this evaluation.','AbortError')};
 current();requireValue(api.evaluateDevelopmentEpisodeVascular,'evaluator unavailable');
 const response=await api.evaluateDevelopmentEpisodeVascular({caseHash:selected.caseHash,episodeId:selected.episodeId});current();
 const result=await checkedVascularEvaluation(response,selected);current();return result;
}
