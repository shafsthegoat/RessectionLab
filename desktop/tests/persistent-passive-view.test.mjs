import fs from 'node:fs';
import assert from 'node:assert/strict';
import {test} from 'node:test';
import {hydratePersistentPassiveEpisode} from '../src/persistent-passive-view.ts';
import {checkedPassiveRecordedTools} from '../src/viewer/recordedTool.ts';
import {arrayDigest,sha256Bytes} from '../src/source-integrity.ts';

// Exact generated BridgeRuntime payload and three generated source arrays.
// No observed scans, patient annotations, model checkpoint or clinical outcome.
const fixture=JSON.parse(fs.readFileSync(new URL('./fixtures/persistent-passive-paired.json',import.meta.url),'utf8'));
const make=()=>structuredClone(fixture);
const api=f=>({readAsset:async id=>{
  assert.equal(typeof f.assets[id],'string');
  return new Uint8Array(Buffer.from(f.assets[id],'base64'));
}});

test('the actual generated paired wrapper hydrates six discrete states with both returned tools',async()=>{
  const f=make(),view=await hydratePersistentPassiveEpisode(f.result,api(f));
  assert.equal(view.frames.length,6);
  assert.deepEqual(view.frames.map(x=>x.phase),['ready','held','inspected','withdrawn','both_withdrawn','stopped']);
  assert.deepEqual(view.frames.map(x=>x.elapsedGeneratedSeconds),[0,1,2,3,4,4]);
  assert.deepEqual(view.frames.map(x=>x.inspectionStatus),
    ['not_acquired','not_acquired','unresolved','unresolved','unresolved','unresolved']);
  for(const frame of view.frames){
    assert.equal(frame.recordedTools.length,2);
    assert(frame.recordedTools.every(tool=>tool.physicalAbsenceClaim===false));
    assert.equal(checkedPassiveRecordedTools(view.volume,frame).displays.length,2);
  }
  assert(view.frames[4].recordedTools.every(tool=>tool.poseStatus==='access_pose_recorded'));
  assert.equal(view.frames[2].event.kind,'generated_inspection');
  assert.equal(view.frames[5].event.kind,'stop');
});

for(const [name,change] of [
  ['native tool dimensions',f=>f.result.sourceBinding.tools[1].shaft_radius_mm=.9],
  ['binding digest',f=>f.result.sourceBinding.bindingHash='sha256:'+'0'.repeat(64)],
  ['operative clock',f=>f.result.episode.frames[3].operative.elapsed_generated_seconds=99],
  ['premature inspection',f=>f.result.episode.frames[1].operative.inspection_status='verified'],
  ['frame position',f=>f.result.episode.frames[4].operative.other_tool.pose.tip_mm[2]=100],
  ['source support bytes',f=>{const id=f.result.case.brainMask.assetId;f.assets[id]=Buffer.alloc(2028,1).toString('base64');}],
])test(`changed ${name} is withheld`,async()=>{
  const f=make();change(f);await assert.rejects(hydratePersistentPassiveEpisode(f.result,api(f)),/withheld|changed/);
});

test('saved display requires the separate backend revalidation marker',async()=>{
  const f=make();await assert.rejects(hydratePersistentPassiveEpisode(f.result,api(f),'reopened'),/revalidation/);
  f.result.validation='authoritative_generated_passive_replay';
  assert.equal((await hydratePersistentPassiveEpisode(f.result,api(f),'reopened')).origin,'reopened');
});

test('fully rebound support bytes and descriptor remain outside the fixed paired source',async()=>{
  const f=make(),source=f.result.case,binding=f.result.sourceBinding;
  const id=source.brainMask.assetId;
  const support=new Uint8Array(Buffer.from(f.assets[id],'base64'));
  support[0]=support[0]===0?1:0;
  f.assets[id]=Buffer.from(support).toString('base64');
  source.brainMask.sha256=await sha256Bytes(support);
  binding.supportHash=await arrayDigest(support,source.shape,'|b1');
  assert.notEqual(binding.supportHash,fixture.result.sourceBinding.supportHash);
  assert.equal(source.brainMask.sha256,await sha256Bytes(support));
  assert.equal(binding.supportHash,await arrayDigest(support,source.shape,'|b1'));
  assert.equal(binding.bindingHash,fixture.result.sourceBinding.bindingHash,
    'an attacker can retain the caller-supplied binding hash string');
  await assert.rejects(hydratePersistentPassiveEpisode(f.result,api(f)),/withheld/);
});
