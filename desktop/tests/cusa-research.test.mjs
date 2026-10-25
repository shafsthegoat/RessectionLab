import test from 'node:test';
import assert from 'node:assert/strict';
import {cusaContactModel,cusaParameterSpecs} from '../src/cusa-research.ts';

test('CUSA fraction controls require the explicit new scene model',()=>{
  assert.equal(cusaContactModel({physics:{profile:{schema:'resectionlab.tissue-physics.v2'}}}),null);
  assert.equal(cusaContactModel({physics:{cusaV3:{enabled:true,profile:{schema:'other'}}}}),null);
  const model={enabled:true,profile:{schema:'resectionlab.cusa-contact.v3'}};
  assert.equal(cusaContactModel({physics:{cusaV3:model}}),model);
  assert.equal(cusaParameterSpecs.power.max,1);
  assert.equal(cusaParameterSpecs.radiusMm.min,cusaParameterSpecs.radiusMm.max);
});
