'use strict';
const {plainArgs}=require('./security.cjs');
const {createHash}=require('node:crypto');const {isDeepStrictEqual}=require('node:util');
const {checkedPublicContactAuthority,checkedPublicContactAssets}=require('./public-contact-authority.cjs');
function contactRequest(input){const args=plainArgs(input,['fixture','selector','goalId']);if(Object.keys(args).length!==3||args.fixture!=='generated-public-surface-contact-v1'||!['scripted','SEARCH'].includes(args.selector)||!['near','costly'].includes(args.goalId))throw Error('Choose only the fixed public contact fixture, goal and scripted/SEARCH selector');return {fixture:args.fixture,selector:args.selector,goalId:args.goalId};}
const digest=text=>'sha256:'+createHash('sha256').update(text).digest('hex');
function validateContactResult(result,args){
 if(!result||Object.keys(result).sort().join(',')!=='case,episode,episodeCanonicalJson')throw Error('Unexpected contact response fields');
 checkedPublicContactAssets(result);
 const e=result.episode,s=result.case,expected=checkedPublicContactAuthority(e);
 if(e.selector!==args.selector||e.publicGoal.goalId!==args.goalId||e.fixture!==args.fixture||typeof result.episodeCanonicalJson!=='string'||Buffer.byteLength(result.episodeCanonicalJson)>2*1024*1024||digest(result.episodeCanonicalJson)!==e.episodeId)throw Error('Contact request or serialization identity changed');
 const {episodeId,...body}=e;if(!isDeepStrictEqual(body,JSON.parse(result.episodeCanonicalJson)))throw Error('Contact body differs from canonical serialization');
 if(digest(expected.objective)!==e.publicGoal.objectiveHash||digest(expected.declaration)!==e.taskContract.detachedObservationBinding.declaration_hash)throw Error('Public objective/declaration digest differs');
 if(s?.caseHash!==e.caseHash||s.metadata?.evidence_kind!=='generated_software_fixture'||s.metadata?.patient_admission!==false||s.metadata?.source_task_hash!==e.sourceHash||s.workspaceSession?.episodeReplay||e.episodeAuthorship!==undefined||e.evidenceKind!=='generated_software_fixture'||e.backendStatus!=='generated_executed'||e.patientAdmission!==false||e.clinicalValidation!==false||e.frame!=='RAS+'||e.physicalUnits!=='mm'||e.sourceBinding?.private_reference_published!==false||e.sourceBinding?.display_case_hash!==e.caseHash||e.sourceBinding?.native_source_hash!==e.sourceHash||JSON.stringify(e.shape)!=='[13,13,12]'||e.geometryAudit?.feasible!==true||e.geometryAudit?.complete_tool_checked!==true||e.geometryAudit?.frontier_checked!==true||e.geometryAudit?.source_case_hash!==e.sourceHash||!Array.isArray(e.replayFrames)||e.replayFrames.length<2||e.replayFrames.length>512)throw Error('Contact source, geometry or transient provenance differs');
 const pending=[e];while(pending.length){const value=pending.pop();if(value&&typeof value==='object'){if(Object.hasOwn(value,'assetId')||(typeof value.path==='string'&&'dtype' in value&&'byteLength' in value))throw Error('Contact episode cannot introduce transfer assets');pending.push(...Object.values(value));}}
 return result;
}
module.exports={contactRequest,validateContactResult};
