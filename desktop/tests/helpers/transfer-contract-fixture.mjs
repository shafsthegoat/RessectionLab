import fs from 'node:fs';
import {createHash} from 'node:crypto';
export const hash=c=>'sha256:'+c.repeat(64);
export function reseal(result){const {episodeId,...body}=result.episode;result.episodeCanonicalJson=JSON.stringify(body);result.episode.episodeId='sha256:'+createHash('sha256').update(result.episodeCanonicalJson).digest('hex');}
/** Fictional metadata exercises the transport DTO only. The underlying motion is
 * an existing generated SEARCH fixture; this is never evidence of inference. */
export function transferFixture(status='verified_live_backend_run'){
 const f=JSON.parse(fs.readFileSync(new URL('../fixtures/development-search.json',import.meta.url),'utf8'));
 const e=f.result.episode;e.selector='RL256_ASPIRATION_TRANSFER';
 Object.assign(e.planning,{learnedPolicyExecuted:true,actorForwardCalls:e.history.length,actor_forward_calls:e.history.length,optimizerUpdates:0,optimizer_updates:0,
  attributionScope:'live_backend_checkpoint_run_only',actorObservationContract:'permitted-spatial-observation-v1',projectionHash:hash('a'),projectionReceiptHash:hash('b'),
  policyIdentity:{policy_id:'fictional-DTO-control-not-a-model-run',architecture_hash:hash('c'),parameter_hash:hash('d'),checkpoint_sha256:hash('e'),checkpoint_validation_receipt_sha256:hash('b'),observation_contract:'permitted-spatial-observation-v1',training_status:'trained_checkpoint'},
  projectionTrace:e.history.map(h=>({full_observation_hash:hash('f'),projected_observation_hash:hash('a'),retained_action_ids:h.action_id==='STOP'?['STOP']:['STOP',h.action_id],omitted_probe_action_ids:['fictional-omitted-probe'],chosen_action_id:h.action_id,projection_hash:hash('a')}))});
 f.result.episodeAuthorship={status,checkpointSha256:hash('e'),projectionHash:hash('a')};reseal(f.result);return f;
}
export const assetApi=f=>({readAsset:async id=>new Uint8Array(Buffer.from(f.assets[id],'base64'))});
