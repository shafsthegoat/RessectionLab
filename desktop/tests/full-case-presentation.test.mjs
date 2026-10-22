import test from 'node:test';
import assert from 'node:assert/strict';
import {fullCaseStateSummary,recordedFullCasePhase,operativeFieldFrame,acceptedCloseTool,closeToolFrame,closeToolPresentation} from '../src/full-case-presentation.ts';
import {validCameraState} from '../src/viewer/workspaceSettings.ts';
const view=()=>({scene:{environment:{target:{initialVolumeMm3:100}},regions:[{name:'declared skull',material:'bone'},{name:'target',material:'tumor'}]},metrics:{targetRemovedMm3:20,removedByLabelMm3:{'declared skull':7,target:20}},fluid:{10:2,12:3},sources:[{kind:'modeled-vessel-wound',bleedingActive:false,reservoirMm3:1},{kind:'modeled-vessel-wound',bleedingActive:true,reservoirMm3:2}],vesselsV2:{wounds:{one:{},two:{}},faces:{a:{patent:false},b:{patent:false},c:{patent:true}}}});
test('replay accounting uses selected source grid and label ownership, and separates spatial fluid from reservoirs and patency',()=>{
  const v=view(),before=structuredClone(v);
  assert.deepEqual(fullCaseStateSummary(v),{initialTargetMm3:100,targetRemovedMm3:20,targetRemainingMm3:80,targetRemovedPercent:20,boneRemovedMm3:7,normalRemovedMm3:null,unknownRemovedMm3:null,normalExposureCellSeconds:null,unknownExposureCellSeconds:null,modeledBloodGeneratedMm3:null,fluidSuctionedMm3:null,elapsedSeconds:null,woundCount:2,activeWounds:1,nonpatentFaces:2,spatialFluidMm3:5,reservoirFluidMm3:3});
  const cleared=view();cleared.fluid={};cleared.sources.forEach(source=>source.bleedingActive=false);
  const summary=fullCaseStateSummary(cleared);
  assert.equal(summary.spatialFluidMm3,0);assert.equal(summary.activeWounds,0);assert.equal(summary.nonpatentFaces,2);
  assert.deepEqual(v,before);
});
test('missing anatomy and vessel evidence remain unknown instead of becoming zero',()=>{
  const v={scene:{regions:[]},metrics:{},fluid:{}};
  const summary=fullCaseStateSummary(v);
  for(const key of ['initialTargetMm3','targetRemainingMm3','targetRemovedPercent','boneRemovedMm3','normalRemovedMm3','unknownRemovedMm3','normalExposureCellSeconds','unknownExposureCellSeconds','modeledBloodGeneratedMm3','fluidSuctionedMm3','elapsedSeconds','woundCount','activeWounds','nonpatentFaces','reservoirFluidMm3'])assert.equal(summary[key],null,key);
});
test('live mapped-material removal excludes targets and bone while exposure and fluid use their distinct recorded ledgers',()=>{
  const v=view();
  v.scene.regions.push({name:'normal brain',material:'soft_tissue',target:false},{name:'dura',material:'membrane',target:false},{name:'functional label',material:'functional_region',target:false},{name:'unmapped source',material:'unknown',target:false},{name:'target soft label',material:'soft_tissue',target:true});
  Object.assign(v.metrics,{removedByLabelMm3:{'declared skull':7,target:20,'normal brain':3.5,dura:1.25,'functional label':2,'unmapped source':4.75,'target soft label':8},exposureByMaterialCellSeconds:{bone:99,tumor:500,soft_tissue:10,membrane:5,functional_region:7,unknown:2},modeledVesselFluidGeneratedMm3:12.5,fluidRemovedMm3:6.25,fluidAddedMm3:999,elapsedSeconds:8.4});
  const before=structuredClone(v),summary=fullCaseStateSummary(v);
  assert.equal(summary.normalRemovedMm3,6.75);assert.equal(summary.unknownRemovedMm3,4.75);assert.equal(summary.boneRemovedMm3,7);
  assert.equal(summary.normalExposureCellSeconds,22);assert.equal(summary.unknownExposureCellSeconds,2);
  assert.equal(summary.modeledBloodGeneratedMm3,12.5);assert.equal(summary.fluidSuctionedMm3,6.25);assert.equal(summary.elapsedSeconds,8.4);assert.deepEqual(v,before);
  delete v.metrics.modeledVesselFluidGeneratedMm3;v.metrics.vesselPhysicsV2={generatedMm3:13};assert.equal(fullCaseStateSummary(v).modeledBloodGeneratedMm3,13);
  delete v.metrics.exposureByMaterialCellSeconds;assert.equal(fullCaseStateSummary(v).normalExposureCellSeconds,null);
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
const closeView=()=>({stateId:'accepted-state',scene:{regions:[{name:'brain',material:'soft_tissue'},{name:'skull',material:'bone'},{name:'target',material:'tumor',target:true},{name:'artery',material:'vessel'},{name:'unknown',material:'unknown'}]},catalog:[{id:'drill',name:'Research drill',radiusMm:2}],slots:{left:{instrument:'drill',inserted:true,active:true,position:[1,2,3],quaternion:[0,0,0,1],parameters:{}},right:{instrument:'drill',inserted:true,active:false,position:[4,5,6],quaternion:[0,Math.SQRT1_2,0,Math.SQRT1_2],parameters:{}}}});
test('close framing uses exact accepted effective-active tip, bounded microscope field and valid mm camera without mutating source or settings',()=>{
  const view=closeView(),settings={layerOpacity:{brain:.6,custom:.7},layerColor:{brain:'#123456'},clipBox:{enabled:true,min:[0,0,0],max:[20,20,20]},smoothSurfaces:false},before=structuredClone({view,settings});
  const field=closeToolFrame(view,'left',{left:false,right:true},settings);
  assert.equal(field.name,'right');assert.deepEqual(field.center,[4,5,6]);assert.deepEqual(field.settings.camera.target,[4,5,6]);assert.equal(field.fieldMm,24);assert(validCameraState(field.settings.camera));
  const distance=Math.hypot(...field.settings.camera.position.map((value,i)=>value-field.center[i]));assert(Math.abs(distance-24/(2*Math.tan(17*Math.PI/180)))<1e-10);assert(distance<45);
  assert.equal(field.settings.layerOpacity.brain,.045);assert.equal(field.settings.layerOpacity.skull,.55);assert.equal(field.settings.layerOpacity.target,.9);assert.equal(field.settings.layerOpacity.artery,1);assert.equal(field.settings.layerOpacity.unknown,.08);assert.equal(field.settings.layerOpacity.custom,.7);
  assert.equal(field.settings.layerColor,settings.layerColor);assert.equal(field.settings.clipBox,settings.clipBox);assert.equal(field.settings.smoothSurfaces,false);assert.equal(field.settings.volumeRendering.enabled,false);assert.deepEqual({view,settings},before);
  const plain=closeToolFrame(view,'right',{left:false,right:false},settings,false);assert.equal(plain.name,'right');assert.equal(plain.settings.layerOpacity,settings.layerOpacity);
  for(const radius of [NaN,-1,0,.01,100]){view.slots.right.parameters.radiusMm=radius;const f=closeToolFrame(view,'right',{left:false,right:true},{});assert(f.fieldMm>=18&&f.fieldMm<=42);assert(validCameraState(f.settings.camera));}
});
test('withdrawn or invalid accepted tips cannot invent a close-up target; display alpha uses declared materials only',()=>{
  const v=closeView();v.slots.right.inserted=false;assert.equal(acceptedCloseTool(v,'right',{left:false,right:true}).name,'left');v.slots.left.position=[NaN,0,0];assert.equal(closeToolFrame(v,'left',undefined,{}),null);
  v.slots.left.position=[0,0,0];v.slots.left.inserted=false;assert.equal(acceptedCloseTool(v,'left'),null);assert.equal(acceptedCloseTool(null,'left'),null);
  v.scene.regions.push({name:'tumor-looking source name',material:'unknown'},{name:'declared vessel target',material:'vessel',target:true});
  const display=closeToolPresentation(v,{});assert.equal(display.layerOpacity['tumor-looking source name'],.08);assert.equal(display.layerOpacity['declared vessel target'],1);
});
