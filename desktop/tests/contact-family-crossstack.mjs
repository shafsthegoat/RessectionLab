/** Saved generated BridgeSession transport only; no native/model execution.
 * Usage: node --experimental-strip-types contact-family-crossstack.mjs catalog.json transport.json [...]
 */
import fs from 'node:fs';import assert from 'node:assert/strict';import {createRequire} from 'node:module';
import {checkedFamilyAvailability} from '../src/contact-family-availability.ts';import {hydrateContactFamilyEpisode} from '../src/contact-family-data.ts';
const require=createRequire(import.meta.url),{validateFamilyResult}=require('../electron/contact-family.cjs');
assert(process.argv.length>=4,'A catalog and saved transport fixture are required');
const catalog=checkedFamilyAvailability(JSON.parse(fs.readFileSync(process.argv[2],'utf8')));
for(const name of process.argv.slice(3)){
 const f=JSON.parse(fs.readFileSync(name,'utf8')),e=f.result.episode,request={fixture:e.fixture,layoutId:e.layoutId,goalId:e.publicGoal.goalId,selector:e.selector};
 validateFamilyResult(f.result,request,catalog);
 const v=await hydrateContactFamilyEpisode(f.result,{readAsset:async id=>{assert(Object.hasOwn(f.assets,id));return new Uint8Array(Buffer.from(f.assets[id],'base64'))}},request,catalog);
 assert.equal(v.episode.schema,'resectionlab.shared-native-contact-learning-episode.v3');assert.equal(v.episode.metrics.target_removed_mm3,undefined);
 console.log(JSON.stringify({status:'PASS',scope:'saved generated family transport only',episodeId:e.episodeId,sourceHash:e.sourceHash,familyHash:e.familyHash,layout:e.layoutId,role:e.splitRole,goal:e.publicGoal.goalId,selector:e.selector,shape:e.shape,proposalMode:e.taskContract.proposalMode,sourceCandidateVersion:e.taskContract.sourceCandidateVersion,continuousEndpointPresent:e.history.some(a=>a.tip_mm?.some(n=>!Number.isInteger(n))),frames:v.frames.length,outcome:v.outcome,observation:v.observation,learnedAuthorship:e.learnedAuthorship!==null}));
}
