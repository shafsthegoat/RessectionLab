import test from 'node:test';
import assert from 'node:assert/strict';
import {hasSavedFullCaseBranch} from '../src/full-case-workspace.ts';
test('a saved active full-case branch opens Rehearse while other cases retain ordinary navigation',()=>{
  const artifacts={surgical_toolbox:{branchId:'primary',branches:{primary:{}},fullCase:{primary:{schema:'resectionlab.full-case-rehearsal.v1'}}}};
  const before=structuredClone(artifacts);assert.equal(hasSavedFullCaseBranch(artifacts),true);assert.deepEqual(artifacts,before);
  for(const value of [undefined,{}, {surgical_toolbox:{}},{surgical_toolbox:{...artifacts.surgical_toolbox,branchId:'main'}},{surgical_toolbox:{...artifacts.surgical_toolbox,branches:{}}}])assert.equal(hasSavedFullCaseBranch(value),false);
});
