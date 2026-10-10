import {transferFixture,reseal,hash,assetApi} from './transfer-contract-fixture.mjs';
export {reseal,hash,assetApi};
/** Fictional identities over saved generated geometry; never checkpoint evidence. */
export function comparisonFixture(){
 const f=transferFixture(),actor=f.result.episode;actor.planning.pairSeal=hash('1');reseal(f.result);
 const c={episode:structuredClone(actor)};c.episode.selector='RL256_ASPIRATION_MATCHED_SEARCH';
 const prior=actor.planning;c.episode.planning={sealedBeforeReferenceScoring:true,learnedPolicyExecuted:false,selector:'matched_projected_SEARCH_from_existing_pair',
 actor_forward_calls:0,actorForwardCalls:0,optimizer_updates:0,optimizerUpdates:0,companionExecutionTransitions:actor.history.length,originalSearchModelTransitionCalls:14,
 searchAccounting:{model_transition_calls:14,max_model_transition_calls:24},projectionTrace:structuredClone(prior.projectionTrace),pairSeal:prior.pairSeal,projectionHash:prior.projectionHash,
 matchedActorEpisodeId:actor.episodeId,matchedActorStrategySeal:prior.strategySeal,nativeSearchStrategySeal:prior.strategySeal,strategySeal:prior.strategySeal,strategy:structuredClone(prior.strategy),
 attributionScope:'matched_observed_search_record_from_live_pair_no_new_search',trainingDomainMatchesTarget:false,privateReferenceScored:false};reseal(c);
 f.comparison={caseHash:actor.caseHash,actorEpisodeId:actor.episodeId,actorStrategySeal:prior.strategySeal,pairSeal:prior.pairSeal,projectionHash:prior.projectionHash,initialProjectedObservationHash:prior.projectionTrace[0].projected_observation_hash,companion:c};return f;
}
