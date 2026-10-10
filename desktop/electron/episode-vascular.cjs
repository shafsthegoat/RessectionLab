'use strict';
const {createHash}=require('node:crypto');
const {isDeepStrictEqual}=require('node:util');
const {plainArgs}=require('./security.cjs');
const HASH=/^sha256:[a-f0-9]{64}$/;
function requireValue(value,message){if(!value)throw new Error(`Vascular evaluation withheld: ${message}`)}
function keys(value,names){requireValue(value&&typeof value==='object'&&!Array.isArray(value)&&Object.keys(value).sort().join('|')===names.split(' ').sort().join('|'),'unexpected report fields');}
function vascularRequest(input){const args=plainArgs(input,['caseHash','episodeId']);keys(args,'caseHash episodeId');requireValue(HASH.test(args.caseHash)&&HASH.test(args.episodeId),'invalid episode selection');return {...args};}
function count(record){
 keys(record,'touched_reference_cells positive_reference_cells unknown_reference_cells outside_reference_fov annotated_positive_encounter annotation_coverage_complete_for_sweep positive_cell_volume_upper_bound_mm3 unknown_in_grid_cell_volume_mm3 biological_vessel_free clinical_injury_probability');
 const t=record.touched_reference_cells,p=record.positive_reference_cells,u=record.unknown_reference_cells,o=record.outside_reference_fov;
 requireValue([t,p,u].every(n=>Number.isSafeInteger(n)&&n>=0)&&p+u<=t&&typeof o==='boolean','invalid contact counts');
 requireValue(record.annotated_positive_encounter===(p>0?true:u>0||o?null:false)&&record.annotation_coverage_complete_for_sweep===(!u&&!o),'invalid encounter/coverage interpretation');
 requireValue(record.biological_vessel_free===null&&record.clinical_injury_probability===null,'unsupported biological claim');
 requireValue([record.positive_cell_volume_upper_bound_mm3,record.unknown_in_grid_cell_volume_mm3].every(n=>Number.isFinite(n)&&n>=0),'invalid cell volumes');
}
function validateVascularResult(result,request){
 keys(result,'caseHash evaluation evaluationCanonicalJson');const e=result.evaluation;
 keys(e,'schema status evaluationId episodeId caseHash sourceHash decisionModelHash strategySeal physicalHistoryHash physicalHistoryCanonicalJson actionIds referenceBindingHash perAction shaft tip wholeTool removedOverlap clinicalInjuryProbability scope patientAdmission');
 requireValue(Buffer.byteLength(JSON.stringify(result))<=512*1024,'report exceeds budget');
 requireValue(typeof result.evaluationCanonicalJson==='string'&&`sha256:${createHash('sha256').update(result.evaluationCanonicalJson).digest('hex')}`===e.evaluationId,'result serialization changed');
 const {evaluationId,...body}=e;requireValue(isDeepStrictEqual(JSON.parse(result.evaluationCanonicalJson),body),'result differs from bound serialization');
 requireValue(e.schema==='generated-shared-vascular-encounter-v1'&&e.status==='evaluated_generated_vascular_reference'&&e.scope==='generated_geometry_annotation_contact_only'&&e.patientAdmission===false&&e.clinicalInjuryProbability===null,'unsupported evaluation scope');
 requireValue([e.evaluationId,e.episodeId,e.caseHash,e.sourceHash,e.decisionModelHash,e.strategySeal,e.physicalHistoryHash,e.referenceBindingHash].every(h=>HASH.test(h))&&result.caseHash===request.caseHash&&e.caseHash===request.caseHash&&e.episodeId===request.episodeId,'stale episode result');
 requireValue(typeof e.physicalHistoryCanonicalJson==='string'&&`sha256:${createHash('sha256').update(e.physicalHistoryCanonicalJson).digest('hex')}`===e.physicalHistoryHash,'history serialization changed');
 const history=JSON.parse(e.physicalHistoryCanonicalJson);
 requireValue(Array.isArray(history)&&history.length>0&&history.length<=6&&Array.isArray(e.actionIds)&&Array.isArray(e.perAction)&&history.length===e.actionIds.length&&history.length===e.perAction.length,'invalid action inventory');
 for(const [i,row] of e.perAction.entries()){
  keys(row,'actionIndex actionId interactionMode sweepCount shaft tip wholeTool');
  requireValue(row.actionIndex===i&&row.actionId===e.actionIds[i]&&row.actionId===history[i].action_id&&row.interactionMode===history[i].interaction_mode,'changed action mapping');
  if(row.interactionMode==='stop')requireValue(row.actionId==='STOP'&&row.sweepCount===0&&row.shaft===null&&row.tip===null&&row.wholeTool===null,'STOP contains tool movement');
  else{requireValue(['aspirate','probe'].includes(row.interactionMode)&&row.sweepCount===1,'invalid action sweep');count(row.shaft);count(row.tip);count(row.wholeTool)}
 }
 for(const part of ['shaft','tip','wholeTool'])count(e[part]);
 keys(e.removedOverlap,'status outcomes');requireValue(e.removedOverlap.status==='not_evaluated_by_contact_kernel'&&e.removedOverlap.outcomes===null,'contact cannot certify removal overlap');
 return result;
}
module.exports={vascularRequest,validateVascularResult};
