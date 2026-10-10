/** Inspect a saved generated bridge envelope only; no policy or engine call. */
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {hydrateDevelopmentEpisode} from '../src/episode-data.ts';
import {hydrateWorkspace} from '../src/workspace-session.ts';
const require=createRequire(import.meta.url),{validateEpisodeResult}=require('../electron/development-episode.cjs'),{validateWorkspaceResult}=require('../electron/workspace-session.cjs');
const [path,origin]=process.argv.slice(2);
assert(path&&['live','reopened'].includes(origin),'Usage: episode-transfer-crossstack.mjs generated-envelope.json live|reopened');
const f=JSON.parse(fs.readFileSync(path,'utf8'));
const api={readAsset:async id=>{assert(Object.hasOwn(f.assets,id),'Missing generated fixture asset');return new Uint8Array(Buffer.from(f.assets[id],'base64'));}};
let view;
if(origin==='live'){
 validateEpisodeResult(f.result,{selector:'RL256_ASPIRATION_TRANSFER'});
 view=await hydrateDevelopmentEpisode(f.result,api);
}else{
 validateWorkspaceResult(f.result);
 view=(await hydrateWorkspace(f.result,api)).episodeView;
}
assert(view&&view.episode.selector==='RL256_ASPIRATION_TRANSFER');assert.equal(view.origin,origin);
assert.equal(view.authorship.status,origin==='live'?'verified_live_backend_run':'unverified_imported');
assert(view.episode.history.every(h=>h.interaction_mode==='aspirate'||h.interaction_mode==='stop'));assert(view.probeCounts.every(n=>n===0));
console.log(JSON.stringify({status:'PASS',scope:'transport validation only; this checker does not establish model authorship or run a model/engine',origin,
 episodeId:view.episode.episodeId,authorshipField:view.authorship.status,actions:view.episode.history.length,frames:view.frames.length},null,2));
