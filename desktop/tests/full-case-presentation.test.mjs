import test from 'node:test';
import assert from 'node:assert/strict';
import {fullCaseStateSummary,recordedFullCasePhase,operativeFieldFrame,acceptedCloseTool,closeToolFrame,closeToolPresentation,closeToolWindowPresentation} from '../src/full-case-presentation.ts';
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
test('operative framing excludes recorded approach tips outside the source scan and retains their exact poses',()=>{
  const source={shape:[220,220,181],affine:[[-1,0,0,109.53500366210938],[0,1,0,-123.558837890625],[0,0,1,-554.25],[0,0,0,1]],frame:'RAS+'};
  const v=closeView();v.slots.left.position=[-10.464996337890625,30.441162109375,-369.9433333333333];v.slots.right.inserted=false;
  const target=[-15.889134651090231,8.01669131252251,-428.68486645239943],before=structuredClone({source,v,target});
  const field=operativeFieldFrame(v,target,{},source);
  assert.equal(field.anchor,'target');assert.deepEqual(field.center,target);assert(validCameraState(field.settings.camera));
  assert.deepEqual({source,v,target},before);
  v.slots.right.inserted=true;v.slots.right.position=[0,0,-430];
  const inside=operativeFieldFrame(v,target,{},source);assert.equal(inside.anchor,'instrument');assert.deepEqual(inside.center,[0,0,-430]);
  v.slots.right.inserted=false;assert.equal(operativeFieldFrame(v,null,{},source),null);
  assert.equal(operativeFieldFrame(v,[0,0,-370],{},source),null);assert.equal(operativeFieldFrame(v,target,{},null),null);
});
test('operative source containment uses inverse RAS affine for oblique and LPS grids',()=>{
  const v=closeView();v.slots.right.inserted=false;
  const oblique={shape:[11,11,11],affine:[[1,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],frame:'RAS+'};
  v.slots.left.position=[1,9,5]; // Inside the world AABB, outside acquired voxel support.
  assert.deepEqual(operativeFieldFrame(v,[10,5,5],{},oblique).center,[10,5,5]);
  const lps={shape:[11,11,11],affine:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],frame:'LPS+'};
  v.slots.left.position=[-2,-3,4];assert.deepEqual(operativeFieldFrame(v,null,{},lps).center,[-2,-3,4]);
  assert.equal(operativeFieldFrame(v,null,{}, {...lps,affine:lps.affine.map(()=>[0,0,0,0])}),null);
});
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
test('local cutaway retains the exact arbitrary-orientation tip and behind-tip halfspace within a bounded display box',()=>{
  for(const quaternion of [[0,0,0,1],[.2,-.3,.4,.5],[1,2,3,4],[0,Math.SQRT1_2,0,Math.SQRT1_2]]){
    const v=closeView();v.slots.right.quaternion=quaternion;v.slots.right.position=[-10.465,30.441,-369.943];
    const settings={clipping:{axis:'x',value:80,flipped:true,enabled:true},clipBox:{enabled:true,min:[-100,-90,-80],max:[100,90,80]},obliqueClip:{enabled:true,normal:[0,1,0],offset:-50},layerOpacity:{brain:.4}};
    const before=structuredClone({v,settings}),field=closeToolFrame(v,'right',{left:false,right:true},settings,false,true),window=field.windowClipping;
    assert.deepEqual(field.center,v.slots.right.position);assert.deepEqual(field.settings.camera.target,field.center);assert(validCameraState(field.settings.camera));
    assert.equal(window.clipping.enabled,false);assert.deepEqual(field.settings.clipBox,window.clipBox);assert.deepEqual(field.settings.obliqueClip,window.obliqueClip);
    assert.equal(field.settings.layerOpacity,settings.layerOpacity);
    for(let i=0;i<3;i++){
      assert(Math.abs(window.clipBox.min[i]-(field.center[i]-.75*field.fieldMm))<1e-12);
      assert(Math.abs(window.clipBox.max[i]-(field.center[i]+.75*field.fieldMm))<1e-12);
    }
    const {normal,offset}=window.obliqueClip,signed=point=>point.reduce((sum,value,i)=>sum+value*normal[i],offset);
    assert(Math.abs(Math.hypot(...normal)-1)<1e-12);
    const toTip=field.center.map((value,i)=>value-field.settings.camera.position[i]),length=Math.hypot(...toTip);
    assert(normal.every((value,i)=>Math.abs(value-toTip[i]/length)<1e-12));
    assert(Math.abs(signed(field.center)-field.fieldMm/12)<1e-10);
    assert(signed(field.settings.camera.position)<0,'camera-facing context is clipped');
    assert(signed(field.center.map((value,i)=>value+normal[i]*field.fieldMm/4))>0,'behind-tip context is retained');
    assert(Math.abs(signed(field.center.map((value,i)=>value-normal[i]*field.fieldMm/12)))<1e-10,'front margin is fixed at focus');
    assert.deepEqual({v,settings},before);
  }
});
test('pinned window toggles preserve current orbit and restore exact prior clipping values and key absence',()=>{
  const v=closeView(),original={clipping:{axis:'y',value:12,enabled:true,flipped:true},clipBox:{enabled:false,min:[-1,-2,-3],max:[7,8,9]},obliqueClip:{enabled:true,normal:[0,0,1],offset:17},layerOpacity:{brain:.7}};
  const field=closeToolFrame(v,'right',{left:false,right:true},original,true,true),pinned=structuredClone(field.windowClipping);
  const camera={position:[90,80,70],target:[3,4,5],up:[0,0,1]},current={...field.settings,camera,contrast:2};
  v.slots.right.position=[100,200,300];v.slots.right.quaternion=[.4,.3,.2,.1];
  const off=closeToolWindowPresentation(current,field.windowClipping,original,false);
  assert.deepEqual(off.clipping,original.clipping);assert.deepEqual(off.clipBox,original.clipBox);assert.deepEqual(off.obliqueClip,original.obliqueClip);assert.equal(off.camera,camera);assert.equal(off.contrast,2);assert.equal(off.layerOpacity,current.layerOpacity);
  const on=closeToolWindowPresentation(off,field.windowClipping,original,true);
  assert.deepEqual(on.clipBox,pinned.clipBox);assert.deepEqual(on.obliqueClip,pinned.obliqueClip);assert.equal(on.clipping.enabled,false);assert.equal(on.camera,camera);assert.deepEqual(field.windowClipping,pinned);
  on.clipBox.min[0]++;assert.deepEqual(field.windowClipping,pinned,'applied settings cannot mutate the pinned descriptor');
  const absent=closeToolWindowPresentation(current,field.windowClipping,{},false);
  for(const key of ['clipping','clipBox','obliqueClip'])assert.equal(Object.hasOwn(absent,key),false);
  const plain=closeToolFrame(closeView(),'right',{left:false,right:true},original,false);
  assert.equal(plain.settings.clipping,original.clipping);assert.equal(plain.settings.clipBox,original.clipBox);assert.equal(plain.settings.obliqueClip,original.obliqueClip);
});
