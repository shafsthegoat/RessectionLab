'use strict';
const {checkedEpisodeAuthority}=require('./episode-authority.cjs');
const { plainArgs } = require('./security.cjs');
const { createHash } = require('node:crypto');
const { isDeepStrictEqual } = require('node:util');
const HASH = /^sha256:[a-f0-9]{64}$/;
function episodeRequest(input) {
  const args = plainArgs(input, ['fixture','selector']);
  if (args.fixture !== 'generated-sequential-v1' || !['scripted','SEARCH','RL256_ASPIRATION_TRANSFER'].includes(args.selector) || Object.keys(args).length !== 2)
    throw new Error('Choose the fixed generated fixture and a supported episode selector');
  return {fixture:args.fixture,selector:args.selector};
}
/** Policy/identity envelope before asset traversal and renderer event forwarding.
 * Detailed cell/pose/prefix validation is shared by the renderer replay loader. */
function validateEpisodeResult(result, request, origin='live') {
  const e=result?.episode, c=result?.case;
  checkedEpisodeAuthority(e,result?.episodeAuthorship,origin);
  if (typeof result?.episodeCanonicalJson!=='string' || result.episodeCanonicalJson.length>2*1024*1024 ||
      `sha256:${createHash('sha256').update(result.episodeCanonicalJson).digest('hex')}` !== e?.episodeId)
    throw new Error('Generated episode serialization digest changed');
  const {episodeId,...body}=e;
  if (!isDeepStrictEqual(JSON.parse(result.episodeCanonicalJson),body)) throw new Error('Generated episode differs from its bound serialization');
  if (!e || !c || e.schema!=='resectionlab.shared-native-development-episode.v1' || e.selector!==request.selector ||
    e.evidenceKind!=='generated_software_fixture' || e.backendStatus!=='generated_executed' || e.patientAdmission!==false || e.clinicalValidation!==false ||
    c.metadata?.evidence_kind!=='generated_software_fixture' || c.metadata?.is_synthetic!==true || c.metadata?.patient_admission!==false ||
    e.caseHash!==c.caseHash || e.sourceBinding?.display_case_hash!==c.caseHash || e.sourceBinding?.native_source_hash!==e.sourceHash ||
    c.metadata?.source_task_hash!==e.sourceHash || ![e.episodeId,e.caseHash,e.sourceHash,e.decisionModelHash].every(h=>HASH.test(h)) ||
    JSON.stringify(e.shape)!==JSON.stringify([13,13,12]) || JSON.stringify(e.shape)!==JSON.stringify(c.shape) ||
    JSON.stringify(e.affine)!==JSON.stringify(c.affine) || e.frame!=='RAS+' || c.frame!=='RAS+' || e.physicalUnits!=='mm' ||
    !Array.isArray(e.history) || e.history.length<1 || e.history.length>6 || !Array.isArray(e.replayFrames) || e.replayFrames.length<2 || e.replayFrames.length>512 ||
    e.geometryAudit?.feasible!==true || e.geometryAudit?.complete_tool_checked!==true || e.geometryAudit?.frontier_checked!==true ||
    e.geometryAudit?.source_case_hash!==e.sourceHash || e.planning?.sealedBeforeReferenceScoring!==true)
    throw new Error('Generated episode response has an invalid source, execution or evidence binding');
  if (Buffer.byteLength(JSON.stringify(e))>2*1024*1024) throw new Error('Generated episode exceeds its result budget');
  return result;
}
module.exports={episodeRequest,validateEpisodeResult};
