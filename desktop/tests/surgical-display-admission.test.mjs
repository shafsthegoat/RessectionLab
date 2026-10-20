import assert from 'node:assert/strict';
import {test} from 'node:test';
import {SurgicalDisplayAdmission} from '../src/surgical-display-admission.ts';

test('only exact desired view commits; same-ID previews and superseded callbacks cannot publish',()=>{
  const gate=new SurgicalDisplayAdmission(),accepted={stateId:'same'},preview={stateId:'same'};
  const initial=gate.expect('source',1,accepted);
  assert.equal(gate.commit(initial,structuredClone(accepted)),false);
  assert.equal(gate.commit(initial,accepted),true);
  assert.equal(gate.commit(initial,accepted),false,'repeated same-view callbacks cannot cause a render loop');
  const proposed=gate.expect('source',1,preview);
  assert.equal(gate.isCurrent(initial,accepted),false);
  assert.equal(gate.commit(initial,accepted),false);
  assert.equal(gate.commit(proposed,preview),true);
  const cancel=gate.expect('source',1,accepted);
  assert.equal(gate.commit(proposed,preview),false);
  assert.equal(gate.commit(cancel,accepted),true,'cancel restores the previously accepted display atomically');
});
test('case and request generations invalidate completed work even when view object is reused',()=>{
  const gate=new SurgicalDisplayAdmission(),view={stateId:'same'};
  const old=gate.expect('source',1,view),newGeneration=gate.expect('source',2,view);
  assert.equal(gate.commit(old,view),false);assert.equal(gate.commit(newGeneration,view),true);
  const other=gate.expect('other-source',2,view);
  assert.equal(gate.commit(newGeneration,view),false);assert.equal(gate.commit(other,view),true);
  gate.invalidate();assert.equal(gate.commit(other,view),false);
  assert.equal(gate.commit(gate.expect('other-source',3,null),null),false);
});
