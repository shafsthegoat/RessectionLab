import assert from 'node:assert/strict';
import {test} from 'node:test';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {hydrateWorkspace,checkedImageView,checkedImagingState} from '../src/workspace-session.ts';
import {workspaceDisplay} from '../src/workspace-imaging-data.ts';
import {sourceFrameDigest} from '../src/source-integrity.ts';
const hash=c=>'sha256:'+c.repeat(64);
async function fixture() {
 const f=JSON.parse(fs.readFileSync(new URL('./fixtures/development-scripted.json',import.meta.url),'utf8'));
 const source=f.result.case, aux=structuredClone(source);delete aux.workspaceSession;
 aux.caseHash=hash('b');aux.affine[0][3]+=100;aux.metadata={...aux.metadata,workspace_display_only:true,selected_modality:'FLAIR'};
 const attached={schema:'workspace-display-series-v1',seriesId:hash('c'),referenceCaseHash:source.caseHash,
  referenceFrameHash:await sourceFrameDigest(source),sourceFrameHash:await sourceFrameDigest(aux),
  association:{kind:'explicit_user_selected',samePersonVerified:false,sameTimeVerified:false},
  registration:{status:'unreviewed',reason:'No transform',sourceToReferenceRasMm:null,registrationHash:null,overlayPermitted:false},
  scope:'display-only-native-grid',planningEligible:false,evaluationEligible:false,modality:'FLAIR',modalityOrigin:'user_declared',
  annotationKind:'source-provided',annotationCoverage:'unknown',acquisitionDatetime:null,expandedBytes:10000,volume:aux};
 const layers=Object.fromEntries(source.compartments.map(c=>[c.name,true]));
 source.workspaceSession={schema:'integrated-workspace-session-v1',sessionHash:hash('d'),referenceCaseHash:source.caseHash,
  displaySeries:[attached],imagingState:{selectedSeriesId:null,states:{primary:{cursor:[0,0,1.5],visibleLayers:layers},[attached.seriesId]:{cursor:[100,0,2],visibleLayers:layers}}},
  episodeReplay:{episode:f.result.episode,episodeCanonicalJson:f.result.episodeCanonicalJson,frameIndex:8,visible:true,validation:'authoritative_generated_replay',accountingQualification:'historical_timings_not_remeasured'},
  evidenceInventory:[{seriesId:'primary',sourceCaseHash:source.caseHash,sourceFrameHash:await sourceFrameDigest(source),modality:'primary',sourceKind:'primary_source',displayAvailable:true,planningInput:true,readiness:'primary_case_contract',reasons:['Existing primary contract only.']},
   {seriesId:attached.seriesId,sourceCaseHash:aux.caseHash,sourceFrameHash:attached.sourceFrameHash,modality:'FLAIR',sourceKind:'source_annotation',displayAvailable:true,planningInput:false,readiness:'display_only_registration_unreviewed',reasons:['Registration and coverage unknown.']}]};
 return {source,attached,api:{readAsset:async id=>new Uint8Array(Buffer.from(f.assets[id],'base64'))}};
}
test('reopened generated replay passes every prefix and restores source-frame image states',async()=>{
 const {source,api}=await fixture(),v=await hydrateWorkspace(source,api);
 assert.equal(v.episodeStep,8);assert.equal(v.episodeView.frames.length,96);assert.equal(v.episodeVisible,true);
 assert.equal(v.episodeView.frames[8].removedNormalVolumeMm3,1);assert.equal(v.series.length,1);
 assert.deepEqual(v.imagingState.states.primary.cursor,[0,0,1.5]);
 assert.equal(v.series[0].volume.affine[0][3],100); // separate native translation
 assert.equal(v.session.evidenceInventory[1].planningInput,false);
});
test('auxiliary selection retains replay without overlaying its unrelated native grid',async()=>{
 const {source,attached,api}=await fixture();source.workspaceSession.imagingState.selectedSeriesId=attached.seriesId;source.workspaceSession.episodeReplay.visible=false;
 const v=await hydrateWorkspace(source,api),display=workspaceDisplay(v.volume,v.series[0]);
 assert.equal(v.imagingState.selectedSeriesId,attached.seriesId);assert.equal(v.episodeStep,8);assert.equal(v.episodeVisible,false);
 assert.equal(display.primaryOverlaysPermitted,false);assert.equal(display.planningInteractionPermitted,false);
 assert.deepEqual(v.imagingState.states[attached.seriesId].cursor,[100,0,2]);
});
test('legacy bundle has no phantom attachments or replay',async()=>{
 const {source,api}=await fixture();delete source.workspaceSession;const v=await hydrateWorkspace(source,api);
 assert.equal(v.session,null);assert.equal(v.episodeView,null);assert.equal(v.episodeStep,0);assert.equal(v.episodeVisible,false);assert.deepEqual(v.series,[]);
});
for(const [name,mutate] of [
 ['stale selection',s=>s.imagingState.selectedSeriesId=hash('e')],
 ['missing source state',s=>delete s.imagingState.states.primary],
 ['extra source state',s=>s.imagingState.states.stale=s.imagingState.states.primary],
 ['wrong source owner',s=>s.referenceCaseHash=hash('e')],
 ['wrong frame',s=>s.evidenceInventory[1].sourceFrameHash=hash('e')],
 ['invented admission',s=>s.evidenceInventory[1].planningInput=true],
 ['changed provenance',s=>s.evidenceInventory[1].sourceKind='estimated_annotation'],
 ['unvalidated replay',s=>s.episodeReplay.validation='saved_hash_only'],
 ['unqualified timing',s=>delete s.episodeReplay.accountingQualification],
 ['invalid frame index',s=>s.episodeReplay.frameIndex=999],
 ['primary replay on auxiliary',s=>s.imagingState.selectedSeriesId=s.displaySeries[0].seriesId],
 ['stale layer',s=>s.imagingState.states.primary.visibleLayers.fake=true],
 ['outside image cursor',s=>s.imagingState.states.primary.cursor=[1000,0,0]],
])test(`${name} is rejected before assets`,async()=>{const {source}=await fixture();mutate(source.workspaceSession);await assert.rejects(hydrateWorkspace(source,{readAsset:()=>assert.fail('read before source checks')}));});
test('resealed replay motion is still rejected by authoritative prefix projection',async()=>{
 const {source,api}=await fixture(),r=source.workspaceSession.episodeReplay;
 r.episode.replayFrames.find(f=>f.phase==='withdrawal').tipRasMm[0]++;
 const {episodeId,...body}=r.episode;r.episodeCanonicalJson=JSON.stringify(body);r.episode.episodeId='sha256:'+createHash('sha256').update(r.episodeCanonicalJson).digest('hex');
 await assert.rejects(hydrateWorkspace(source,api),/exact committed/);
});
test('delayed reopen is invalidated even when its case identity has not changed',async()=>{
 const {source,api}=await fixture();let current=true,release,started;
 const gate=new Promise(resolve=>release=resolve),entered=new Promise(resolve=>started=resolve);
 const promise=hydrateWorkspace(source,{readAsset:async id=>{started();await gate;return api.readAsset(id)}},()=>current);
 await entered;current=false;release();await assert.rejects(promise,{name:'AbortError'});
});
test('incoming metadata mutation across asset awaits cannot replace checked selection',async()=>{
 const {source,api}=await fixture();let mutated=false;
 const v=await hydrateWorkspace(source,{readAsset:async id=>{if(!mutated){mutated=true;source.workspaceSession.imagingState.selectedSeriesId=hash('e');source.workspaceSession.episodeReplay.frameIndex=90;}return api.readAsset(id)}});
 assert.equal(v.episodeStep,8);assert.equal(v.imagingState.selectedSeriesId,null);
});
test('RAS display bounds allow empty oblique corners without silently moving the cursor',async()=>{
 const {source}=await fixture();source.affine=[[1,-1,0,0],[1,1,0,0],[0,0,1,0],[0,0,0,1]];
 const view={cursor:[13,-1,0],visibleLayers:Object.fromEntries(source.compartments.map(c=>[c.name,false]))};
 assert.deepEqual(checkedImageView(source,view),view);
 assert.throws(()=>checkedImageView(source,{...view,cursor:[13.00002,-1,0]}),/bounds/);
 const selected=checkedImagingState(source,[],{selectedSeriesId:null,states:{primary:{...view,cursor:null}}});
 assert.deepEqual(selected.states.primary.cursor,[0,12,5.5]);
});
