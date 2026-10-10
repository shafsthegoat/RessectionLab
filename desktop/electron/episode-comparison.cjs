'use strict';
const {plainArgs}=require('./security.cjs');
const {createHash}=require('node:crypto');
const {isDeepStrictEqual}=require('node:util');
const {checkedEpisodeAuthority}=require('./episode-authority.cjs');
const HASH=/^sha256:[a-f0-9]{64}$/;
function comparisonRequest(input){const args=plainArgs(input,['caseHash','episodeId']);if(Object.keys(args).length!==2||![args.caseHash,args.episodeId].every(h=>typeof h==='string'&&HASH.test(h)))throw Error('Name the current generated actor case and episode only');return {caseHash:args.caseHash,episodeId:args.episodeId};}
function validateComparisonResult(result,request){
 const keys=['caseHash','actorEpisodeId','actorStrategySeal','pairSeal','projectionHash','initialProjectedObservationHash','companion'];
 if(!result||Object.keys(result).length!==keys.length||!keys.every(k=>Object.hasOwn(result,k))||result.caseHash!==request.caseHash||result.actorEpisodeId!==request.episodeId||!keys.filter(k=>k!=='companion').every(k=>HASH.test(result[k])))throw Error('Matched comparison actor/pair envelope differs');
 const c=result.companion,e=c?.episode,p=e?.planning;
 if(!c||Object.keys(c).length!==2||typeof c.episodeCanonicalJson!=='string'||Buffer.byteLength(c.episodeCanonicalJson)>2*1024*1024||'sha256:'+createHash('sha256').update(c.episodeCanonicalJson).digest('hex')!==e?.episodeId)throw Error('Matched comparison episode serialization changed');
 const {episodeId,...body}=e;if(!isDeepStrictEqual(body,JSON.parse(c.episodeCanonicalJson)))throw Error('Matched comparison body differs from serialization');
 checkedEpisodeAuthority(e,undefined,'comparison');
 const pending=[e];while(pending.length){const value=pending.pop();if(value&&typeof value==='object'){if(Object.hasOwn(value,'assetId')||(typeof value.path==='string'&&'dtype' in value&&'byteLength' in value))throw Error('Matched comparison cannot introduce transfer assets');pending.push(...Object.values(value));}}
 if(e.selector!=='RL256_ASPIRATION_MATCHED_SEARCH'||e.schema!=='resectionlab.shared-native-development-episode.v1'||e.caseHash!==result.caseHash||e.evidenceKind!=='generated_software_fixture'||e.backendStatus!=='generated_executed'||e.patientAdmission!==false||e.clinicalValidation!==false||e.sourceBinding?.private_reference_published!==false||e.sourceBinding?.display_case_hash!==e.caseHash||e.sourceBinding?.native_source_hash!==e.sourceHash||
  ![e.sourceHash,e.decisionModelHash].every(h=>HASH.test(h))||e.frame!=='RAS+'||e.physicalUnits!=='mm'||JSON.stringify(e.shape)!=='[13,13,12]'||
  !Array.isArray(e.replayFrames)||e.replayFrames.length<2||e.replayFrames.length>512||e.geometryAudit?.feasible!==true||e.geometryAudit?.complete_tool_checked!==true||e.geometryAudit?.frontier_checked!==true||e.geometryAudit?.source_case_hash!==e.sourceHash||
  p.matchedActorEpisodeId!==result.actorEpisodeId||p.matchedActorStrategySeal!==result.actorStrategySeal||p.pairSeal!==result.pairSeal||p.projectionHash!==result.projectionHash||p.projectionTrace[0].projected_observation_hash!==result.initialProjectedObservationHash)throw Error('Matched comparison source/history attribution differs');
 return result;
}
module.exports={comparisonRequest,validateComparisonResult};
