'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const read = name => fs.readFileSync(path.join(__dirname, name), 'utf8');

test('fixed generated passive operation is exposed and validated before assets/events', () => {
  const main=read('main.cjs'), preload=read('preload.cjs'), sidecar=read('sidecar.cjs');
  assert.match(main, /handle\('executePersistentPassiveEpisode'.*persistentPassiveRequest/);
  assert.match(preload, /research:executePersistentPassiveEpisode/);
  assert.match(sidecar, /executePersistentPassiveEpisode'\) \{\s*try \{ args = persistentPassiveRequest/);
  const validate = sidecar.indexOf('validatePersistentPassiveV1Result(message.result');
  assert(validate > 0 && validate < sidecar.indexOf('this.assets.expose(message.result)'));
  assert(validate < sidecar.indexOf("this.emit('event', { ...message"));
  assert.match(sidecar, /'executeDevelopmentEpisode', 'executePersistentPassiveEpisode'/);
});

function workspaceValidator() {
  const calls=[];
  const module={exports:{}};
  vm.runInNewContext(read('workspace-session.cjs'), {module, require(name) {
    if (name==='./development-episode.cjs') return {validateEpisodeResult() {
      throw Error('passive replay was sent to the legacy authority');
    }};
    if (name==='./persistent-passive.cjs') return {validatePersistentPassiveV1Result(...args) {
      calls.push(args);
    }};
    throw Error(`unexpected module ${name}`);
  }});
  return {validate:module.exports.validateWorkspaceResult,calls};
}
function reopened() {
  const hash=`sha256:${'a'.repeat(64)}`;
  return {caseHash:hash,workspaceSession:{schema:'integrated-workspace-session-v1',
    referenceCaseHash:hash,sessionHash:hash,displaySeries:[],
    imagingState:{selectedSeriesId:null},episodeReplay:{
      validation:'authoritative_generated_passive_replay',
      accountingQualification:'declared_generated_schedule_not_measured',frameIndex:2,visible:true,
      episode:{schema:'resectionlab.generated-persistent-passive-episode.v1',frames:Array(6)},
      episodeCanonicalJson:'fixed generated source bytes',sourceBinding:{caseHash:hash}}}};
}
test('passive saved branch is separate from the legacy episode authority', () => {
  const {validate,calls}=workspaceValidator();
  const result=reopened();
  assert.equal(validate(result),result);
  assert.equal(calls.length,1);
  assert.equal(calls[0][1].origin,'reopened');
  assert.equal(calls[0][1].revalidation,'authoritative_generated_passive_replay');
  assert.equal(calls[0][0].sourceBinding,result.workspaceSession.episodeReplay.sourceBinding);
});
for(const [name,change] of [
  ['missing native revalidation',r=>{r.workspaceSession.episodeReplay.validation='authoritative_generated_replay';}],
  ['wrong time qualification',r=>{r.workspaceSession.episodeReplay.accountingQualification='historical_timings_not_remeasured';}],
  ['seventh frame',r=>{r.workspaceSession.episodeReplay.episode.frames.push({});}],
  ['out-of-range selection',r=>{r.workspaceSession.episodeReplay.frameIndex=6;}],
  ['auxiliary image visibility',r=>{r.workspaceSession.imagingState.selectedSeriesId='aux';}],
  ['legacy replay coercion',r=>{r.workspaceSession.episodeReplay.episode.replayFrames=[];}],
]) test(`saved passive branch refuses ${name}`,()=>{
  const {validate,calls}=workspaceValidator();const result=reopened();change(result);
  assert.throws(()=>validate(result));assert.equal(calls.length,0);
});
