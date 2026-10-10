import assert from 'node:assert/strict';
import { test } from 'node:test';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { hydrateDevelopmentEpisode } from '../src/episode-data.ts';
import { recordedToolDisplay } from '../src/viewer/recordedTool.ts';
import { validateReplay } from '../src/viewer/replay.ts';
const fixture = (name='scripted') => JSON.parse(fs.readFileSync(new URL(`./fixtures/development-${name}.json`,import.meta.url),'utf8'));
const api = f => ({readAsset:async id=>new Uint8Array(Buffer.from(f.assets[id],'base64'))});
// Explicit resealing only exercises deeper guards after deliberate test mutations.
function reseal(f) {const {episodeId,...body}=f.result.episode;f.result.episodeCanonicalJson=JSON.stringify(body);f.result.episode.episodeId='sha256:'+createHash('sha256').update(f.result.episodeCanonicalJson).digest('hex');}
for (const selector of ['scripted','search']) test(`actual saved ${selector}: every frame, mask and full tool pose`,async()=>{
 const f=fixture(selector),v=await hydrateDevelopmentEpisode(f.result,api(f));
 assert.equal(v.frames.length,selector==='scripted'?96:72);
 for (const [i,replay] of v.frames.entries()) {validateReplay(v.volume,replay);const tool=recordedToolDisplay(v.volume,replay);
  assert.equal(tool!==null,!['initial','stop'].includes(v.episode.replayFrames[i].phase));
  if(tool) assert.deepEqual(tool.tip,v.episode.replayFrames[i].tipRasMm);
 }
 assert.equal(v.frames.at(-1).removedTargetVolumeMm3,selector==='scripted'?2:4);
 assert.equal(v.frames.at(-1).removedNormalVolumeMm3,3);
 assert.equal(v.episode.history.filter(h=>h.interaction_mode==='probe').length,selector==='scripted'?2:0);
 assert.equal(v.episode.attemptDiagnostics[0].status,'rejected');assert.equal(v.episode.history.at(-1).interaction_mode,'stop');
 const original=structuredClone(v.frames[1].recordedTool);f.result.episode.tools[0].shaft_radius_mm=4;
 assert.deepEqual(v.frames[1].recordedTool,original);
});
for (const [name,mutate] of [
 ['shaft dimensions',e=>e.tools[0].shaft_radius_mm*=2],['working length',e=>e.tools[0].working_length_mm*=2],
 ['recorded pose',e=>e.replayFrames[1].tipRasMm[2]+=1],['unsupported fidelity',e=>e.unsupported=[]],
 ['rejected attempt',e=>e.attemptDiagnostics[0].stateAfter='sha256:'+'f'.repeat(64)],
]) test(`digest rejects changed ${name}`,async()=>{const f=fixture();mutate(f.result.episode);await assert.rejects(hydrateDevelopmentEpisode(f.result,api(f)),/serialization/);});
for (const [name,mutate,reason] of [
 ['withdrawal removal',e=>e.replayFrames.find(f=>f.phase==='withdrawal').removedIndicesNative=[[6,6,2]],/exact committed/],
 ['probe removal',e=>e.history.find(h=>h.interaction_mode==='probe').removed_indices_native=[[6,6,3]],/macro and microstep|probe/],
 ['source identity',e=>e.sourceBinding.display_case_hash='sha256:'+'f'.repeat(64),/source identities/],
 ['prefix hash',e=>e.replayFrames[5].cavityHash='sha256:'+'f'.repeat(64),/digest/],
 ['final removal',e=>e.finalRemovedIndicesNative=[],/final cavity/],
 ['engine ancestry',e=>e.history[1].source_state_hash='sha256:'+'f'.repeat(64),/ancestry/],
 ['withdrawal pose',e=>e.replayFrames.find(f=>f.phase==='withdrawal').tipRasMm[0]+=1,/exact committed/],
 ['probe contact',e=>e.replayFrames.find(f=>f.probeContactIndicesNative.length).probeContactIndicesNative=[],/probe contact/],
 ['absent full-tool audit',e=>e.geometryAudit.complete_tool_checked=false,/audit/],
]) test(`deeper gate rejects ${name}`,async()=>{const f=fixture();mutate(f.result.episode);reseal(f);await assert.rejects(hydrateDevelopmentEpisode(f.result,api(f)),reason);});
test('tissue transfer source remains required',async()=>{const f=fixture();f.assets[f.result.case.brainMask.assetId]=Buffer.alloc(2028).toString('base64');await assert.rejects(hydrateDevelopmentEpisode(f.result,api(f)),/tissue transfer/);});
test('pose gate rejects stale frame and nonunit axis',async()=>{const f=fixture(),v=await hydrateDevelopmentEpisode(f.result,api(f)),replay=structuredClone(v.frames[1]);replay.recordedTool.frameIndex++;assert.throws(()=>recordedToolDisplay(v.volume,replay),/withheld/);replay.recordedTool.frameIndex--;replay.recordedTool.axis=[0,0,2];assert.throws(()=>recordedToolDisplay(v.volume,replay),/withheld/);});
