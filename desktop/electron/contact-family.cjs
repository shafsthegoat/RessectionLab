'use strict';
const {createHash}=require('node:crypto');
const {isDeepStrictEqual}=require('node:util');
const {plainArgs}=require('./security.cjs');
const {checkedFamilyAvailability,requireInteractiveFamilyRequest,checkedPublicContactAssets,
  checkedFamilyAuthority,checkedFamilyExecution}=require('./contact-family-authority.cjs');
const hash=text=>'sha256:'+createHash('sha256').update(text).digest('hex');
function familyAvailabilityRequest(value){const args=plainArgs(value,[]);if(Object.keys(args).length)throw Error('Availability accepts no arguments');return {};}
function familyRequest(value,catalog){return requireInteractiveFamilyRequest(plainArgs(value,['fixture','layoutId','goalId','selector']),catalog).request;}
function validateFamilyResult(result,request,catalog){
  checkedPublicContactAssets(result);
  const e=result?.episode,s=result?.case;
  const expected=checkedFamilyAuthority(e,request,catalog);checkedFamilyExecution(result,catalog,request);
  if(typeof result.episodeCanonicalJson!=='string'||Buffer.byteLength(result.episodeCanonicalJson)>2*1024*1024||
    hash(result.episodeCanonicalJson)!==e.episodeId)throw Error('Family canonical result changed');
  const {episodeId,...body}=e;
  if(!isDeepStrictEqual(body,JSON.parse(result.episodeCanonicalJson))||hash(expected.objectiveJson)!==e.publicGoal.objectiveHash||
    hash(expected.declarationJson)!==e.taskContract.detachedObservationBinding.declaration_hash)throw Error('Family body/objective/declaration differs');
  if(s?.caseHash!==e.caseHash||s.metadata?.evidence_kind!=='generated_software_fixture'||s.metadata?.patient_admission!==false||
    s.metadata?.source_task_hash!==e.sourceHash||s.metadata?.family_hash!==e.familyHash||s.metadata?.layout_id!==e.layoutId||
    s.workspaceSession!==undefined||e.episodeAuthorship!==undefined||e.evidenceKind!=='generated_software_fixture'||
    e.backendStatus!=='generated_executed'||e.patientAdmission!==false||e.clinicalValidation!==false||
    e.frame!=='RAS+'||e.physicalUnits!=='mm'||e.sourceBinding?.private_reference_published!==false||
    e.sourceBinding?.display_case_hash!==e.caseHash||e.sourceBinding?.native_source_hash!==e.sourceHash||
    JSON.stringify(e.shape)!=='[15,15,14]'||e.geometryAudit?.feasible!==true||e.geometryAudit?.complete_tool_checked!==true||
    e.geometryAudit?.frontier_checked!==true||e.geometryAudit?.source_case_hash!==e.sourceHash||
    !Array.isArray(e.replayFrames)||e.replayFrames.length<2||e.replayFrames.length>512)throw Error('Family source/native geometry provenance differs');
  return result;
}
module.exports={familyAvailabilityRequest,familyRequest,checkedFamilyAvailability,validateFamilyResult};
