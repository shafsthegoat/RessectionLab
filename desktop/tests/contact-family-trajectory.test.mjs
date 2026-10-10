import assert from 'node:assert/strict';import {test} from 'node:test';import {checkedFamilyTrajectory} from '../src/contact-family-authority.ts';
// Pure physical-codec control only; no generated episode or native acceptance claim.
function path(){const entry=[1.25,2.125,3.75],axis=[.6,.8,0],tip=entry.map((v,i)=>v+2*axis[i]),tool={tool_id:'probe',tip_radius_mm:.5,shaft_radius_mm:.2,tip_length_mm:.75,working_length_mm:2};let previous=entry;const microsteps=[0,1,2].map(distance=>{const end=entry.map((v,i)=>v+distance*axis[i]),m={tip_start_mm:previous,tip_end_mm:end,active_stroke_start_mm:previous.map((v,i)=>v-.75*axis[i]),active_stroke_end_mm:end,active_radius_mm:.5};previous=end;return m});return{tools:[tool],history:[{interaction_mode:'probe',tool_id:'probe',entry_mm:entry,tip_mm:tip,axis_unit:axis,microsteps}]};}
test('continuous rotated subvoxel trajectory is kept without lattice rounding',()=>{const e=path(),before=structuredClone(e);checkedFamilyTrajectory(e);assert.deepEqual(e,before);});
for(const[name,mutate]of[
 ['null physical coordinate',e=>e.history[0].tip_mm[0]=null],['macro endpoint',e=>e.history[0].tip_mm[0]+=.25],
 ['microstep discontinuity',e=>e.history[0].microsteps[1].tip_start_mm=[0,0,0]],
 ['active tip geometry',e=>e.history[0].microsteps[1].active_stroke_start_mm[1]+=.1],
 ['active radius',e=>e.history[0].microsteps[1].active_radius_mm=.2],
 ['missing final segment',e=>e.history[0].microsteps.pop()],['off-axis path',e=>{e.history[0].microsteps[1].tip_end_mm[2]+=.1}]
])test(`physical codec rejects ${name}`,()=>{const e=path();mutate(e);assert.throws(()=>checkedFamilyTrajectory(e));});
