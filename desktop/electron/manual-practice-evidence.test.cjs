'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {summary, stateWitness, checkAction} = require('./manual-practice-evidence.cjs');

const response = (id, position) => ({caseHash:'control',branchId:'episode',position,
  view:{stateId:id,geometryRevision:id,scene:{sceneId:'scene'},removed:[],fluid:{},
    slots:{left:{position:[1,2,3],quaternion:[0,0,0,1],instrument:'cusa',inserted:true}},
    originalTargetResidualMm3:10,metrics:{targetRemovedMm3:0,fluidPresentMm3:0,fluidAddedMm3:0,fluidRemovedMm3:0,fluidEscapedMm3:0}}});

test('refusal checks fluid and native metrics even with unchanged state identifier', () => {
  const source = response('a', 7);
  const before = stateWitness(source);
  const refusal = {status:'rejected',completedFraction:0};
  checkAction(before, stateWitness(source), refusal, {});
  source.view.fluid = {12:1};
  assert.throws(()=>checkAction(before,stateWitness(source),refusal,{}), /Refusal changed/);
});

test('partial stroke keeps a real endpoint distinct from its requested target', () => {
  const before = stateWitness(response('a',7));
  const after = stateWitness(response('b',8));
  const partial = {status:'partial',completedFraction:.25};
  const requested = {slot:'left',path:[{position:[1,2,5]}]};
  checkAction(before,after,partial,requested);
  after.tools.left.position=[1,2,5];
  assert.throws(()=>checkAction(before,after,partial,requested));
});

test('empty timing populations remain unavailable and nearest-rank percentiles include the slow tail',()=>{
  assert.deepEqual(summary([]),{samples:0,median:null,p95:null});
  assert.deepEqual(summary([1,4]),{samples:2,median:2.5,p95:4});
  assert.deepEqual(summary([3,1,4,2,100]),{samples:5,median:3,p95:100});
});
