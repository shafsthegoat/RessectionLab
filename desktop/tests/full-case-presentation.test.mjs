import test from 'node:test';
import assert from 'node:assert/strict';
import {fullCaseStateSummary,recordedFullCasePhase,operativeFieldFrame} from '../src/full-case-presentation.ts';
const view=()=>({scene:{environment:{target:{initialVolumeMm3:100}},regions:[{name:'declared skull',material:'bone'},{name:'target',material:'tumor'}]},metrics:{targetRemovedMm3:20,removedByLabelMm3:{'declared skull':7,target:20}},fluid:{10:2,12:3},sources:[{kind:'modeled-vessel-wound',bleedingActive:false,reservoirMm3:1},{kind:'modeled-vessel-wound',bleedingActive:true,reservoirMm3:2}],vesselsV2:{wounds:{one:{},two:{}},faces:{a:{patent:false},b:{patent:false},c:{patent:true}}}});
test('replay accounting uses selected source grid and label ownership, and separates spatial fluid from reservoirs and patency',()=>{
  const v=view(),before=structuredClone(v);
  assert.deepEqual(fullCaseStateSummary(v),{initialTargetMm3:100,targetRemovedMm3:20,targetRemainingMm3:80,targetRemovedPercent:20,boneRemovedMm3:7,woundCount:2,activeWounds:1,nonpatentFaces:2,spatialFluidMm3:5,reservoirFluidMm3:3});
  const cleared=view();cleared.fluid={};cleared.sources.forEach(source=>source.bleedingActive=false);
  const summary=fullCaseStateSummary(cleared);
  assert.equal(summary.spatialFluidMm3,0);assert.equal(summary.activeWounds,0);assert.equal(summary.nonpatentFaces,2);
  assert.deepEqual(v,before);
});
test('missing anatomy and vessel evidence remain unknown instead of becoming zero',()=>{
  const v={scene:{regions:[]},metrics:{},fluid:{}};
  const summary=fullCaseStateSummary(v);
  for(const key of ['initialTargetMm3','targetRemainingMm3','targetRemovedPercent','boneRemovedMm3','woundCount','activeWounds','nonpatentFaces','reservoirFluidMm3'])assert.equal(summary[key],null,key);
});
test('phase labels follow recorded snapshot ranges and retain the completed phase at shared boundaries',()=>{
  const record={phases:[{id:'access',startPosition:0,endPosition:5},{id:'resection',startPosition:5,endPosition:10}]};
  assert.equal(recordedFullCasePhase(record,0)?.id,'access');
  assert.equal(recordedFullCasePhase(record,5)?.id,'access');
  assert.equal(recordedFullCasePhase(record,6)?.id,'resection');
  assert.equal(recordedFullCasePhase(record,10)?.id,'resection');
  assert.equal(recordedFullCasePhase(record,11),undefined);
});
test('explicit operative framing chooses inserted active tip then target, changes only display settings, and retains cutaways and appearance',()=>{
  const v={slots:{left:{inserted:true,active:false,position:[1,2,3],quaternion:[0,0,0,1]},right:{inserted:true,active:true,position:[4,5,6],quaternion:[0,Math.SQRT1_2,0,Math.SQRT1_2]}}};
  const settings={layerOpacity:{vessels:.2},clipBox:{enabled:true,min:[0,0,0],max:[10,10,10]},camera:{target:[1,2,3],position:[1,-58,3],up:[0,0,1]}};
  const before=structuredClone({v,settings}),field=operativeFieldFrame(v,[20,20,20],settings);
  assert.deepEqual(field.center,[4,5,6]);assert.deepEqual(field.settings.camera.target,[4,5,6]);
  assert.ok(field.settings.camera.position.every((n,i)=>Math.abs(n-[-111,5,6][i])<1e-8));
  assert.deepEqual(field.settings.camera.up,[0,1,0]);
  assert.equal(field.settings.layerOpacity,settings.layerOpacity);assert.equal(field.settings.clipBox,settings.clipBox);
  assert.deepEqual({v,settings},before);
  v.slots.right.inserted=false;assert.deepEqual(operativeFieldFrame(v,[20,20,20],{}).center,[1,2,3]);
  v.slots.left.inserted=false;const fallback=operativeFieldFrame(v,[20,20,20],{});assert.deepEqual(fallback.center,[20,20,20]);
  assert.ok(Math.abs(Math.hypot(...fallback.settings.camera.position.map((n,i)=>n-fallback.center[i]))-115)<1e-8);
  assert.deepEqual(fallback.settings.camera.up,[0,0,1]);
  assert.equal(operativeFieldFrame(v,null,{}),null);
});
