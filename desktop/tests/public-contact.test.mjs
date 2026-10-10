import assert from 'node:assert/strict';import {test}from'node:test';
import {hydratePublicContactEpisode,requirePersistableContact} from '../src/public-contact-data.ts';
// Actual generated bridge fixtures are supplied by the backend owner; these
// controls are completed with that frozen fixture before production admission.
test('contact cannot be silently omitted from saved workspace',()=>{assert.throws(()=>requirePersistableContact({}),/not saved/);assert.doesNotThrow(()=>requirePersistableContact(null));});
test('v1 or learned episode fails contact admission before source assets',async()=>{for(const episode of [{schema:'resectionlab.shared-native-development-episode.v1'},{schema:'resectionlab.shared-native-development-episode.v2',selector:'RL256_ASPIRATION_TRANSFER'}])await assert.rejects(hydratePublicContactEpisode({case:{},episode,episodeCanonicalJson:'{}'},{readAsset:()=>assert.fail('no source reads')}));});
import {contactFixture,assetApi,reseal} from './helpers/public-contact-fixture.mjs';
import {hydrateDevelopmentEpisode} from '../src/episode-data.ts';
test('generated near contact shares exact motion/masks with explicit goal and distinct metrics',async()=>{const f=contactFixture(),v=await hydratePublicContactEpisode(f.result,assetApi(f));assert.equal(v.episode.schema,'resectionlab.shared-native-development-episode.v2');assert.deepEqual(v.episode.history.map(h=>h.interaction_mode),['aspirate','probe']);assert.equal(v.outcome.contactedAndRetained,true);assert.equal(v.outcome.totalRemovedMm3,1);assert(Math.abs(v.outcome.modeledReturn-.708)<1e-12);assert.equal(v.outcome.completePathMm,2);assert.equal(v.outcome.termination,'horizon');assert.equal(v.prefixes[0].contactedAndRetained,false);assert.equal(v.prefixes.at(-1).contactedAndRetained,true);assert.equal(v.frames.at(-1).removedMask.reduce((a,b)=>a+b,0),1);assert(!('target_removed_mm3'in v.episode.metrics));});
test('costly SEARCH STOP is a checked completed decision, not contact, failure or no result',async()=>{const f=contactFixture('costly-SEARCH'),v=await hydratePublicContactEpisode(f.result,assetApi(f));assert.equal(v.outcome.termination,'STOP');assert.equal(v.frames.length,2);assert.equal(v.outcome.modeledReturn,0);assert.equal(v.outcome.totalRemovedMm3,0);assert.equal(v.outcome.retained,true);assert.equal(v.outcome.contactedAndRetained,false);});
for(const[name,change]of[
 ['goal relabel',e=>e.publicGoal.goalId='costly'],['native goal',e=>e.publicGoal.nativeIndex[2]++],['RAS goal',e=>e.publicGoal.rasMm[2]++],
 ['crop origin',e=>e.taskContract.detachedObservationBinding.crop_origin_native[2]++],['crop affine',e=>e.taskContract.detachedObservationBinding.crop_affine_hash='sha256:'+'a'.repeat(64)],
 ['goal grid',e=>e.publicGoal.goalGridHash='sha256:'+'a'.repeat(64)],['objective cost',e=>e.taskContract.objective.costs.normal_per_mm3=0],
 ['training claim',e=>e.planning.learnedPolicyExecuted=true],['reference score',e=>e.planning.referenceScoringPerformed=true],['old seal label',e=>e.planning.sealedBeforeReferenceScoring=true],
 ['boolean counter',e=>e.planning.actor_forward_calls=false],['horizon',e=>e.taskContract.maxSteps=6],['target metric',e=>e.metrics.target_removed_mm3=0]
])test(`resealed ${name} refuses before image assets`,async()=>{const f=contactFixture();change(f.result.episode);reseal(f.result);await assert.rejects(hydratePublicContactEpisode(f.result,{readAsset:()=>assert.fail('assets before task admission')}));});
test('contact envelope cannot silently enter old aspiration hydration',async()=>{const f=contactFixture();await assert.rejects(hydrateDevelopmentEpisode(f.result,{readAsset:()=>assert.fail('v1 asset read')}));});
for(const[name,change]of[
 ['motion',e=>e.replayFrames.find(f=>f.phase==='withdrawal').tipRasMm[0]++],
 ['goal outcome',e=>e.metrics.goal_contacted_and_retained=false],
 ['reward arithmetic',e=>{e.history[1].reward++;e.planning.strategy.history[1].reward++;e.metrics.history[1].reward++}],
 ['false goal credit',e=>{e.history[0].goal_potential_after=1;e.planning.strategy.history[0].goal_potential_after=1;e.metrics.history[0].goal_potential_after=1}]
])test(`resealed ${name} fails physical or objective replay`,async()=>{const f=contactFixture();change(f.result.episode);reseal(f.result);await assert.rejects(hydratePublicContactEpisode(f.result,assetApi(f)));});
test('resealed macro probe-contact removal cannot lower reward while leaving committed contact unchanged',async()=>{
 const f=contactFixture(),e=f.result.episode;
 for(const rows of [e.history,e.planning.strategy.history,e.metrics.history]){rows[1].probe_contact_indices_native=[];rows[1].goal_potential_after=0;rows[1].reward=-rows[1].effort_and_removal_cost;}
 e.metrics.total_reward=e.history.reduce((s,h)=>s+h.reward,0);reseal(f.result);
 await assert.rejects(hydratePublicContactEpisode(f.result,assetApi(f)),/macro probe contact/);
});
test('renderer rejects extra private source-metadata asset before any source read',async()=>{const f=contactFixture();f.result.case.metadata.privateReference={assetId:'private',dtype:'uint8',shape:[1],byteLength:1};await assert.rejects(hydratePublicContactEpisode(f.result,{readAsset:()=>assert.fail('no asset reads')}),/outside public source slots/);});
