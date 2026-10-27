import test from 'node:test';
import assert from 'node:assert/strict';
import {planningTimeLabel} from '../src/research-planning-data.ts';

test('planning receipts without optional timing remain inspectable',()=>{
  for(const value of [undefined,null,NaN,Infinity,-1,'12'])
    assert.equal(planningTimeLabel(value),'Computation time unavailable');
  assert.equal(planningTimeLabel(0),'0.00 s total');
  assert.equal(planningTimeLabel(12.001),'12.00 s total');
});
