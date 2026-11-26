import assert from 'node:assert/strict';
import {test} from 'node:test';
import {VIEWER_MODES, escapeTarget, modeExitLabel, resolveModes} from './viewerModes.ts';

const exit = () => {};
const open = {defaultPrevented: false, editableTarget: false};

test('exit names follow "Exit <mode>" and keep acronyms', () => {
  assert.equal(modeExitLabel('Section'), 'Exit section');
  assert.equal(modeExitLabel('Vessel slab'), 'Exit vessel slab');
  assert.equal(modeExitLabel('CTA cut'), 'Exit CTA cut');
  assert.equal(modeExitLabel('3D result'), 'Exit 3D result');
});

test('every registered mode has a sentence-case name and an Escape policy', () => {
  for (const [id, definition] of Object.entries(VIEWER_MODES)) {
    assert.match(definition.label, /^[A-Z0-9][^A-Z]*(?:[A-Z]{2,}[^A-Z]*)*$/, `${id} label is sentence case`);
    assert.ok(['always', 'guarded', 'none'].includes(definition.escape), id);
  }
  // The original operative-field contract: any Escape leaves the expanded field first.
  const top = Object.entries(VIEWER_MODES).sort((a, b) => b[1].priority - a[1].priority)[0];
  assert.deepEqual([top[0], top[1].escape], ['expanded-field', 'always']);
});

test('resolveModes drops inactive entries, fills defaults and orders by priority', () => {
  const modes = resolveModes([
    false,
    {id: 'prior-view', detail: 'Atlas · alignment review required', onExit: exit},
    null,
    {id: 'source-section', detail: 'Source MRI · Axial · Z 12.0 mm', onExit: exit},
    undefined,
    {id: 'expanded-pane', onExit: exit},
    {id: 'vessel-slab', exitLabel: 'Exit vessel slab in 3D', onExit: exit},
    {id: 'branch-crop', detail: '3D only'},
    0,
    '',
  ]);
  assert.deepEqual(modes.map(mode => mode.id), ['expanded-pane', 'source-section', 'vessel-slab', 'branch-crop', 'prior-view']);
  assert.equal(modes[1].label, 'Section');
  assert.equal(modes[1].exitLabel, 'Exit section');
  assert.equal(modes[2].exitLabel, 'Exit vessel slab in 3D', 'an explicit accessible name is kept');
  assert.equal(modes[3].escape, 'none', 'a mode without an exit never takes Escape');
  assert.equal(modes[4].detail, 'Atlas · alignment review required', 'provenance detail is carried through');
});

test('ties keep the order the host reported them in', () => {
  const modes = resolveModes([
    {id: 'generated-episode', label: 'Generated poses', onExit: exit},
    {id: 'modeled-replay', onExit: exit},
    {id: 'estimate-view', onExit: exit},
  ]);
  assert.deepEqual(modes.map(mode => mode.label), ['Generated poses', 'Modeled removal', 'Estimate view']);
});

test('one Escape leaves only the highest-priority mode', () => {
  const left = [];
  const modes = resolveModes([
    {id: 'prior-view', onExit: () => left.push('prior')},
    {id: 'source-section', onExit: () => left.push('section')},
    {id: 'expanded-pane', onExit: () => left.push('pane')},
  ]);
  escapeTarget(modes, open).onExit();
  assert.deepEqual(left, ['pane']);
  assert.equal(escapeTarget(modes.slice(1), open).id, 'source-section');
  assert.equal(escapeTarget(modes.slice(2), open).id, 'prior-view');
  assert.equal(escapeTarget([], open), null);
});

test('handled, typed or repeated Escape does not leave a guarded mode', () => {
  const modes = resolveModes([{id: 'source-section', onExit: exit}, {id: 'branch-crop'}]);
  assert.equal(escapeTarget(modes, {defaultPrevented: true, editableTarget: false}), null);
  assert.equal(escapeTarget(modes, {defaultPrevented: false, editableTarget: true}), null);
  assert.equal(escapeTarget(modes, {...open, repeat: true}), null);
  assert.equal(escapeTarget(modes, {...open, offscreen: true}), null, 'a hidden workspace keeps its modes');
  assert.equal(escapeTarget(resolveModes([{id: 'branch-crop'}]), open), null);
});

test('the expanded operative field keeps its unconditional Escape', () => {
  const modes = resolveModes([{id: 'source-section', onExit: exit}, {id: 'expanded-field', onExit: exit}]);
  for (const context of [open, {defaultPrevented: true, editableTarget: true, repeat: true, offscreen: true}])
    assert.equal(escapeTarget(modes, context).id, 'expanded-field');
});
