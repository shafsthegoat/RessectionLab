import assert from 'node:assert/strict';
import { test } from 'node:test';
import fs from 'node:fs';
import vm from 'node:vm';
import { stripTypeScriptTypes } from 'node:module';

const source = fs.readFileSync(new URL('../src/App.tsx', import.meta.url), 'utf8');
const start = source.indexOf('  const save =');
const end = source.indexOf('  useEffect(() => {', start);
assert(start >= 0 && end > start);
const saveSource = stripTypeScriptTypes(source.slice(start, end));

async function savedReplay(episodeWorkflow, staleComparison = false, surgical) {
  let request;
  const context = {
    api: { saveCase: async value => { surgical?.assertBeforeSave?.();request = value; return { saved: true }; } },
    payload: { caseHash: 'case' },
    act: fn => fn(),
    requirePersistableContact: () => {},
    requirePersistableEpisode: value => {
      if (staleComparison && value) throw Error('temporary aspiration comparison');
    },
    publicContactView: null,
    familyView: null,
    displayedEpisodeView: { source: { caseHash: 'case' } },
    passiveView: { source: { caseHash: 'case' }, episode: { episodeId: 'passive' } },
    episodeView: { source: { caseHash: 'case' }, episode: { episodeId: 'aspiration' } },
    episodeWorkflow,
    passiveStep: 4,
    episodeStep: 2,
    episodeVisible: true,
    imaging: { selected: null, snapshot: () => ({}) },
    routeA: '', routeB: '', category: '', visibleLayers: [], overlayOpacity: 0.5,
    primaryViewer: {},
    rehearsal: {pause:()=>{},viewPreferences:null},
    surgical: surgical??{pauseFullCase:()=>{},pausePlayback:()=>{},view:null},
    cursor: [0, 0, 0], setMessage: () => {},
  };
  vm.runInNewContext(`${saveSource}\nglobalThis.saveAction=save;`, context);
  await context.saveAction();
  return request.workspace.episodeReplay;
}

test('saving after passive then aspiration selects only the active aspiration replay', async () => {
  const replay = await savedReplay('aspiration');
  assert.equal(replay.episodeId, 'aspiration');
  assert.equal(replay.frameIndex, 2);
  assert.equal(replay.visible, true);
});

test('saving after aspiration then passive selects only the active passive replay', async () => {
  const replay = await savedReplay('passive');
  assert.equal(replay.episodeId, 'passive');
  assert.equal(replay.frameIndex, 4);
  assert.equal(replay.visible, true);
});

test('a contact workflow does not save a stale passive or aspiration replay', async () => {
  assert.equal(await savedReplay('public-contact'), null);
});

test('an inactive temporary aspiration comparison cannot block saving the active passive replay', async () => {
  assert.equal((await savedReplay('passive', true)).episodeId, 'passive');
  await assert.rejects(savedReplay('aspiration', true), /temporary aspiration comparison/);
});
test('actual Save pauses computation and microstep playback and waits for a canonical inspect barrier',async()=>{
  const events=[];
  const surgical={view:{stateId:'read-only-frame'},pauseFullCase:()=>events.push('pause compute'),pausePlayback:()=>events.push('pause replay'),run:async command=>{assert.equal(command,'inspect');await Promise.resolve();events.push('canonical state');return true;},assertBeforeSave:()=>assert.deepEqual(events,['pause compute','pause replay','canonical state'])};
  await savedReplay('aspiration',false,surgical);
  await assert.rejects(savedReplay('aspiration',false,{...surgical,run:async()=>false,assertBeforeSave:()=>assert.fail('No save after a failed source-state barrier')}),/could not be synchronized/);
});
