'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { persistentPassiveRequest, validatePersistentPassiveV1Result, bindingDigest } =
  require('./persistent-passive.cjs');

const hash = value => `sha256:${createHash('sha256').update(value).digest('hex')}`;
const actualFixture = JSON.parse(fs.readFileSync(path.join(__dirname, '../tests/fixtures/persistent-passive-paired.json'), 'utf8')).result;
const canonical = actualFixture.episodeCanonicalJson;
const original = actualFixture.episode;
const sort = value => Array.isArray(value) ? value.map(sort) : value && typeof value === 'object'
  ? Object.fromEntries(Object.keys(value).sort().map(key => [key, sort(value[key])])) : value;
function fixture() {
  const episode = structuredClone(original);
  const caseHash = `sha256:${'c'.repeat(64)}`;
  const tool = id => ({tool_id:id,tip_radius_mm:.15,shaft_radius_mm:.1,
    working_length_mm:6,max_access_angle_deg:30,tip_length_mm:.4,
    parameter_source:'generic research geometry; not a verified commercial device'});
  const sourceBinding = {caseHash, sourceHash:episode.sourceHash,
    decisionModelHash:episode.decisionModelHash,
    structuralIntensityHash:`sha256:${'1'.repeat(64)}`,
    supportHash:`sha256:${'2'.repeat(64)}`,
    nominalTargetMaskHash:`sha256:${'3'.repeat(64)}`,
    affineHash:`sha256:${'4'.repeat(64)}`,
    sourceFrameHash:`sha256:${'5'.repeat(64)}`,
    tools:[tool(episode.protocol.stationary_tool_id),tool(episode.protocol.moving_tool_id)]};
  sourceBinding.bindingHash = bindingDigest(sourceBinding);
  return {case: {caseHash, frame: 'RAS+', physicalUnits: 'mm', shape: [13, 13, 12],
    metadata: {is_synthetic: true, evidence_kind: 'generated_software_fixture',
      patient_admission: false, source_task_hash: episode.sourceHash}},
    episode, episodeCanonicalJson: canonical,
    sourceBinding};
}
function resign(result) {
  // Mutation controls rebind both transport digests so semantic checks, rather
  // than only the first digest guard, have an opportunity to reject the change.
  result.episode.physicalHistoryCanonicalJson = JSON.stringify(sort(result.episode.history));
  result.episode.physicalHistoryHash = hash(result.episode.physicalHistoryCanonicalJson);
  const {episodeId, ...body} = result.episode;
  result.episodeCanonicalJson = JSON.stringify(sort(body));
  result.episode.episodeId = hash(result.episodeCanonicalJson);
}
test('saved generated paired proof has its original backend canonical identity', () => {
  assert.equal(hash(canonical), original.episodeId);
  assert.equal(validatePersistentPassiveV1Result(fixture()).episode.episodeId, original.episodeId);
});
test('only the fixed paired v1 request is admitted', () => {
  assert.deepEqual(persistentPassiveRequest({fixture:'generated-persistent-passive-v1',paired:true}),
    {fixture:'generated-persistent-passive-v1',paired:true});
  for (const args of [{fixture:'generated-persistent-passive-v1',paired:false},
    {fixture:'generated-persistent-passive-v1',paired:true,inspectionStatus:'verified'},
    {fixture:'generated-sequential-v1',paired:true}])
    assert.throws(() => persistentPassiveRequest(args));
});
test('saved passive replay requires the distinct backend revalidation marker', () => {
  assert.throws(() => validatePersistentPassiveV1Result(fixture(), {origin:'reopened'}));
  assert.equal(validatePersistentPassiveV1Result(fixture(), {origin:'reopened',
    revalidation:'authoritative_generated_passive_replay'}).episode.episodeId, original.episodeId);
});
for (const [name, change, rebind] of [
  ['case source', r => {r.case.metadata.source_task_hash = `sha256:${'0'.repeat(64)}`;}, false],
  ['case identity', r => {r.sourceBinding.caseHash = `sha256:${'0'.repeat(64)}`;}, false],
  ['tool diameter', r => {r.sourceBinding.tools[0].tip_radius_mm = 1.5;}, false],
  ['tool binding digest', r => {r.sourceBinding.bindingHash = `sha256:${'0'.repeat(64)}`;}, false],
  ['raw canonical bytes', r => {r.episodeCanonicalJson += ' ';}, false],
  ['second tool withdrawal', r => {r.episode.frames[4].operative.other_tool.retained_at_depth = true;}, true],
  ['held moving pose', r => {r.episode.frames[2].operative.moving_tool.pose.tip_mm[2] += .5;}, true],
  ['declared elapsed time', r => {r.episode.history[2].elapsed_generated_seconds += 1;}, true],
  ['native source state', r => {r.episode.history[2].result_state_hash = `sha256:${'0'.repeat(64)}`;}, true],
  ['pre-acquisition hidden status', r => {r.episode.frames[1].operative.inspection_status='unresolved';}, true],
  ['unearned clinical probability', r => {r.episode.history[1].clinical_deficit_probability=.1;}, true],
  ['physical removal claim', r => {r.episode.history[0].removed_indices_native=[1];}, true],
  ['legacy stroke schema', r => {r.episode.schema='resectionlab.shared-native-development-episode.v1';}, true],
]) test(`refuses changed ${name}`, () => {
  const result = fixture(); change(result); if (rebind) resign(result);
  assert.throws(() => validatePersistentPassiveV1Result(result), /withheld/);
});
