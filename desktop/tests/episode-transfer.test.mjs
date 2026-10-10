import assert from 'node:assert/strict';
import {test} from 'node:test';
import fs from 'node:fs';
import {createRequire} from 'node:module';
import {hydrateDevelopmentEpisode} from '../src/episode-data.ts';
import {hydrateWorkspace,defaultImageView} from '../src/workspace-session.ts';
import {sourceFrameDigest} from '../src/source-integrity.ts';
import {transferFixture,reseal,assetApi,hash} from './helpers/transfer-contract-fixture.mjs';
const require=createRequire(import.meta.url),{episodeRequest,validateEpisodeResult}=require('../electron/development-episode.cjs'),{validateWorkspaceResult}=require('../electron/workspace-session.cjs');
const request={fixture:'generated-sequential-v1',selector:'RL256_ASPIRATION_TRANSFER'};
test('distinct transfer selector retains the fixed request boundary; no renderer checkpoint argument',()=>{
 assert.deepEqual(episodeRequest(request),request);
 for(const key of ['checkpoint','path','policyIdentity','episodeAuthorship'])assert.throws(()=>episodeRequest({...request,[key]:'ignored'}));
 for(const selector of ['RL','learned','RL256'])assert.throws(()=>episodeRequest({...request,selector}));
});
test('fictional transfer metadata passes the same actual generated motion and removal prefix checks',async()=>{
 const f=transferFixture();assert.equal(validateEpisodeResult(f.result,request),f.result);
 const v=await hydrateDevelopmentEpisode(f.result,assetApi(f));
 assert.equal(v.origin,'live');assert.equal(v.authorship.status,'verified_live_backend_run');
 assert.equal(v.frames.length,72);assert.equal(v.frames.at(-1).removedTargetVolumeMm3,4);assert.equal(v.frames.at(-1).removedNormalVolumeMm3,3);
 assert(v.probeCounts.every(n=>n===0));assert.equal(v.episode.history.at(-1).interaction_mode,'stop');
});
for(const [name,mutate] of [
 ['missing qualification',r=>delete r.episodeAuthorship],
 ['import claim on live response',r=>r.episodeAuthorship.status='unverified_imported'],
 ['extra qualification',r=>r.episodeAuthorship.verified=true],
 ['wrong checkpoint',r=>r.episodeAuthorship.checkpointSha256=hash('f')],
 ['wrong projection',r=>r.episodeAuthorship.projectionHash=hash('f')],
 ['training update',r=>r.episode.planning.optimizerUpdates=1],
 ['inconsistent forward alias',r=>r.episode.planning.actor_forward_calls=0],
 ['boolean update alias',r=>r.episode.planning.optimizer_updates=false],
 ['different receipt identity',r=>r.episode.planning.policyIdentity.checkpoint_validation_receipt_sha256=hash('c')],
 ['boolean update',r=>r.episode.planning.optimizerUpdates=false],
 ['fractional forward count',r=>r.episode.planning.actorForwardCalls=1.5],
 ['mismatched forward count',r=>r.episode.planning.actorForwardCalls++],
 ['omitted trace',r=>r.episode.planning.projectionTrace.pop()],
 ['different chosen action',r=>r.episode.planning.projectionTrace[0].chosen_action_id='STOP'],
 ['omitted action retained',r=>r.episode.planning.projectionTrace[0].omitted_probe_action_ids=['STOP']],
 ['missing STOP option',r=>r.episode.planning.projectionTrace[0].retained_action_ids.shift()],
 ['duplicate retained action',r=>r.episode.planning.projectionTrace[0].retained_action_ids.push('STOP')],
 ['wrong observation contract',r=>r.episode.planning.policyIdentity.observation_contract='sequential-spatial-observation-v1'],
 ['extra identity field',r=>r.episode.planning.policyIdentity.patientValidated=true],
 ['untrained identity',r=>r.episode.planning.policyIdentity.training_status='untrained_software_control'],
 ['strategy action mismatch',r=>r.episode.planning.strategy.actions.reverse()],
 ['probe history',r=>r.episode.history[0].interaction_mode='probe'],
 ['probe frame',r=>r.episode.replayFrames[1].mode='probe'],
 ['probe contact',r=>r.episode.replayFrames[1].probeContactIndicesNative=[[6,6,2]]],
 ['old selector claiming model',r=>r.episode.selector='SEARCH'],
])test(`both transport boundaries reject resealed ${name} before assets`,async()=>{
 const f=transferFixture();mutate(f.result);reseal(f.result);
 assert.throws(()=>validateEpisodeResult(f.result,{selector:f.result.episode.selector}));
 await assert.rejects(hydrateDevelopmentEpisode(f.result,{readAsset:()=>assert.fail('assets before authority rejection')}));
});
async function reopened(status='unverified_imported'){
 const f=transferFixture(status),source=f.result.case;
 source.workspaceSession={schema:'integrated-workspace-session-v1',sessionHash:hash('a'),referenceCaseHash:source.caseHash,displaySeries:[],
  imagingState:{selectedSeriesId:null,states:{primary:defaultImageView(source)}},
  episodeReplay:{episode:f.result.episode,episodeCanonicalJson:f.result.episodeCanonicalJson,episodeAuthorship:f.result.episodeAuthorship,frameIndex:1,visible:true,validation:'authoritative_generated_replay',accountingQualification:'imported_computational_provenance_unverified'},
  evidenceInventory:[{seriesId:'primary',sourceCaseHash:source.caseHash,sourceFrameHash:await sourceFrameDigest(source),modality:'primary',sourceKind:'primary_source',displayAvailable:true,planningInput:true,readiness:'primary_case_contract',reasons:[]}]};return f;
}
test('reopen admits checked geometry with unverified authorship, without executing any policy',async()=>{
 const f=await reopened();validateWorkspaceResult(f.result.case);
 const v=await hydrateWorkspace(f.result.case,assetApi(f));assert.equal(v.episodeView.origin,'reopened');assert.equal(v.episodeView.authorship.status,'unverified_imported');assert.equal(v.episodeStep,1);
});
test('saved live attribution cannot be promoted by either workspace boundary',async()=>{
 const f=await reopened('verified_live_backend_run');assert.throws(()=>validateWorkspaceResult(f.result.case));
 await assert.rejects(hydrateWorkspace(f.result.case,{readAsset:()=>assert.fail('assets before qualification rejection')}));
});
test('learned transport keeps full motion checks even after a fresh episode seal',async()=>{
 const f=transferFixture();f.result.episode.replayFrames.find(row=>row.phase==='withdrawal').tipRasMm[0]++;reseal(f.result);
 await assert.rejects(hydrateDevelopmentEpisode(f.result,assetApi(f)),/exact committed/);
});
test('an async caller cannot alter the already-checked authorship label',async()=>{
 const f=transferFixture(),api=assetApi(f);const view=await hydrateDevelopmentEpisode(f.result,{readAsset:async id=>{f.result.episodeAuthorship.status='unverified_imported';return api.readAsset(id)}});
 assert.equal(view.authorship.status,'verified_live_backend_run');
});

test('reopened transfer cannot claim only historical timings are unverified',async()=>{
 const f=await reopened();f.result.case.workspaceSession.episodeReplay.accountingQualification='historical_timings_not_remeasured';
 assert.throws(()=>validateWorkspaceResult(f.result.case));
 await assert.rejects(hydrateWorkspace(f.result.case,{readAsset:()=>assert.fail('assets before accounting qualification rejection')}));
});
for(const selector of ['scripted','search'])test(`legacy ${selector} cannot use the imported-transfer qualification`,async()=>{
 const f=await reopened(),saved=JSON.parse(fs.readFileSync(new URL(`./fixtures/development-${selector}.json`,import.meta.url),'utf8'));
 const replay=f.result.case.workspaceSession.episodeReplay;
 replay.episode=saved.result.episode;replay.episodeCanonicalJson=saved.result.episodeCanonicalJson;delete replay.episodeAuthorship;
 assert.throws(()=>validateWorkspaceResult(f.result.case));
 await assert.rejects(hydrateWorkspace(f.result.case,{readAsset:()=>assert.fail('assets before accounting qualification rejection')}));
});
