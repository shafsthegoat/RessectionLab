'use strict';
const HASH=/^sha256:[a-f0-9]{64}$/;
const object=(value)=>!!value&&typeof value==='object'&&!Array.isArray(value);
const exact=(value,keys)=>object(value)&&Object.keys(value).length===keys.length&&keys.every(key=>Object.hasOwn(value,key));
const digest=(value)=>typeof value==='string'&&HASH.test(value);
function need(value,message){if(!value)throw new Error(`Episode authority withheld: ${message}`);}
const ids=(value)=>Array.isArray(value)&&value.length<=4096&&value.every(id=>typeof id==='string'&&id.length>0&&id.length<=256)&&new Set(value).size===value.length;
/** Transport consistency, not proof that a saved file was produced by an actor.
 * Only the current backend response can qualify a live checkpoint run. */
function checkedEpisodeAuthority(episode,authorship,origin){
  const p=episode?.planning;
  need(object(p)&&p.sealedBeforeReferenceScoring===true,'missing planning seal.');
  need(origin==='live'||origin==='reopened','unknown response origin.');
  if(episode.selector!=='RL256_ASPIRATION_TRANSFER'){
    need(['scripted','SEARCH'].includes(episode.selector)&&p.learnedPolicyExecuted===false&&authorship===undefined,'unsupported selector or learned attribution.');
    return null;
  }
  need(p.learnedPolicyExecuted===true&&Number.isSafeInteger(p.actorForwardCalls)&&Number(p.actorForwardCalls)>0&&Number(p.actorForwardCalls)<=6&&p.optimizerUpdates===0&&p.actor_forward_calls===p.actorForwardCalls&&p.optimizer_updates===p.optimizerUpdates&&
    p.actorObservationContract==='permitted-spatial-observation-v1'&&p.attributionScope==='live_backend_checkpoint_run_only'&&
    digest(p.projectionHash)&&digest(p.projectionReceiptHash)&&digest(p.strategySeal),'invalid transfer identity or accounting.');
  const identity=p.policyIdentity;
  need(exact(identity,['policy_id','architecture_hash','parameter_hash','observation_contract','training_status','checkpoint_sha256','checkpoint_validation_receipt_sha256'])&&
    typeof identity.policy_id==='string'&&identity.policy_id.length>0&&identity.policy_id.length<=128&&
    identity.observation_contract==='permitted-spatial-observation-v1'&&identity.training_status==='trained_checkpoint'&&identity.checkpoint_validation_receipt_sha256===p.projectionReceiptHash&&
    ['architecture_hash','parameter_hash','checkpoint_sha256','checkpoint_validation_receipt_sha256'].every(key=>digest(identity[key])), 'invalid trained policy identity.');
  need(exact(authorship,['status','checkpointSha256','projectionHash'])&&
    authorship.status===(origin==='live'?'verified_live_backend_run':'unverified_imported')&&
    authorship.checkpointSha256===identity.checkpoint_sha256&&authorship.projectionHash===p.projectionHash,
    'live and imported authorship cannot be interchanged.');
  need(Array.isArray(episode.history)&&episode.history.length===p.actorForwardCalls&&Array.isArray(p.projectionTrace)&&p.projectionTrace.length===episode.history.length,
    'transfer decisions, forward count and projection trace differ.');
  const strategy=p.strategy;
  need(object(strategy)&&Array.isArray(strategy.actions)&&strategy.actions.length===episode.history.length&&strategy.actions.every((id,i)=>id===episode.history[i].action_id),'sealed action order differs from history.');
  for(const [i,row] of p.projectionTrace.entries()){
    need(exact(row,['full_observation_hash','projected_observation_hash','retained_action_ids','omitted_probe_action_ids','chosen_action_id','projection_hash'])&&
      digest(row.full_observation_hash)&&digest(row.projected_observation_hash)&&row.projection_hash===p.projectionHash&&
      ids(row.retained_action_ids)&&row.retained_action_ids[0]==='STOP'&&ids(row.omitted_probe_action_ids),'invalid current-inventory projection.');
    const retained=row.retained_action_ids,omitted=row.omitted_probe_action_ids;
    need(omitted.every(id=>!retained.includes(id))&&row.chosen_action_id===episode.history[i].action_id&&retained.includes(String(row.chosen_action_id)),
      'selected action is not in the retained current inventory.');
    const action=episode.history[i];
    need(['aspirate','stop'].includes(action.interaction_mode)&&Array.isArray(action.probe_contact_indices_native)&&action.probe_contact_indices_native.length===0,
      'the transferred actor cannot commit a probe.');
  }
  need(Array.isArray(episode.replayFrames)&&episode.replayFrames.every(frame=>frame.mode!=='probe'&&Array.isArray(frame.probeContactIndicesNative)&&frame.probeContactIndicesNative.length===0),
    'the transferred actor cannot replay probe contact.');
  return {...authorship};
}

module.exports={checkedEpisodeAuthority};
