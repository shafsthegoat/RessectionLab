// Scalar test double only. Actual numerical evaluator output is tested separately.
import fs from 'node:fs';import {createHash} from 'node:crypto';
const hash=text=>'sha256:'+createHash('sha256').update(text).digest('hex');
export function reseal(result){const {evaluationId,...body}=result.evaluation;result.evaluationCanonicalJson=JSON.stringify(body);result.evaluation.evaluationId=hash(result.evaluationCanonicalJson);return result;}
export function fixture(){
 const episode=JSON.parse(fs.readFileSync(new URL('./fixtures/development-scripted.json',import.meta.url),'utf8')).result.episode;
 const count=(p,u=0,out=false)=>({touched_reference_cells:p+u+1,positive_reference_cells:p,unknown_reference_cells:u,outside_reference_fov:out,annotated_positive_encounter:p?true:u||out?null:false,annotation_coverage_complete_for_sweep:!u&&!out,positive_cell_volume_upper_bound_mm3:p,unknown_in_grid_cell_volume_mm3:u,biological_vessel_free:null,clinical_injury_probability:null});
 const history=JSON.stringify(episode.history),evaluation={schema:'generated-shared-vascular-encounter-v1',status:'evaluated_generated_vascular_reference',evaluationId:'',episodeId:episode.episodeId,caseHash:episode.caseHash,sourceHash:episode.sourceHash,decisionModelHash:episode.decisionModelHash,strategySeal:episode.planning.strategySeal,
  physicalHistoryHash:hash(history),physicalHistoryCanonicalJson:history,actionIds:episode.history.map(r=>r.action_id),referenceBindingHash:hash('generated-test-double'),
  perAction:episode.history.map((a,i)=>({actionIndex:i,actionId:a.action_id,interactionMode:a.interaction_mode,sweepCount:a.interaction_mode==='stop'?0:1,shaft:a.interaction_mode==='stop'?null:count(0,2,true),tip:a.interaction_mode==='stop'?null:count(i===0?1:0),wholeTool:a.interaction_mode==='stop'?null:count(i===0?1:0,2,true)})),
  shaft:count(0,2,true),tip:count(1),wholeTool:count(1,2,true),removedOverlap:{status:'not_evaluated_by_contact_kernel',outcomes:null},clinicalInjuryProbability:null,scope:'generated_geometry_annotation_contact_only',patientAdmission:false};
 return {episode,result:reseal({caseHash:episode.caseHash,evaluation,evaluationCanonicalJson:''})};
}
