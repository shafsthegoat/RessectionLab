import test from 'node:test';
import assert from 'node:assert/strict';
import {cusaActionParameters,cusaContactModel,cusaParameterSpecs} from '../src/cusa-research.ts';

test('CUSA fraction controls require the explicit new scene model',()=>{
  assert.equal(cusaContactModel({physics:{profile:{schema:'resectionlab.tissue-physics.v2'}}}),null);
  assert.equal(cusaContactModel({physics:{cusaV3:{enabled:true,profile:{schema:'other'}}}}),null);
  const model={enabled:true,profile:{schema:'resectionlab.cusa-contact.v3'}};
  assert.equal(cusaContactModel({physics:{cusaV3:model}}),model);
  assert.equal(cusaParameterSpecs.power.max,1);
  assert.equal(cusaParameterSpecs.radiusMm.min,cusaParameterSpecs.radiusMm.max);
});

test('manual CUSA requests preserve historical admission and retain V2 duty',()=>{
  const parameters={radiusMm:1.5,power:.8,duty:.5};
  const legacy={physics:{cusaV3:{enabled:true,profile:{schema:'resectionlab.cusa-contact.v3'}}},scene:{},slots:{left:{instrument:'cusa'}}};
  assert.deepEqual(cusaActionParameters(legacy,'left',parameters),{radiusMm:1.5,power:.8});
  assert.deepEqual(parameters,{radiusMm:1.5,power:.8,duty:.5},'UI drafts stay caller-owned');
  const v2={...legacy,physics:{...legacy.physics,profile:{schema:'resectionlab.tissue-physics.v2'}}};
  assert.equal(cusaActionParameters(v2,'left',parameters),parameters);
  assert.equal(cusaActionParameters({...legacy,slots:{left:{instrument:'cranial_drill'}}},'left',parameters),parameters);
});
