import test from 'node:test';
import assert from 'node:assert/strict';
import {volumeSettings,volumeTransfer,volumeRayInterval,opticalAlpha} from './volumeTransfer.ts';
import {checkedViewerSettings} from '../workspace-view-settings.ts';
const identity=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
test('volume rays use millimetres on mirrored, anisotropic and oblique grids, including camera inside',()=>{
  assert.deepEqual(volumeRayInterval([-10,1,1],[1,0,0],[3,3,3],identity),[9.5,12.5]);
  assert.deepEqual(volumeRayInterval([1,1,1],[1,0,0],[3,3,3],identity),[0,1.5]);
  assert.equal(volumeRayInterval([-10,5,1],[1,0,0],[3,3,3],identity),null);
  const affine=[[0,-3,0,10],[-2,0,0,20],[0,0,4,30],[0,0,0,1]];
  const interval=volumeRayInterval([0,18,34],[1,0,0],[3,3,3],affine);
  assert.ok(Math.abs(interval[0]-2.5)<1e-12&&Math.abs(interval[1]-11.5)<1e-12);
  assert.deepEqual(affine,[[0,-3,0,10],[-2,0,0,20],[0,0,4,30],[0,0,0,1]]);
});
test('cutaways retain the declared RAS side and work when the ray parallels the plane',()=>{
  assert.deepEqual(volumeRayInterval([-10,1,1],[1,0,0],[3,3,3],identity,[1,0,0,-1]),[11,12.5]);
  assert.deepEqual(volumeRayInterval([-10,1,1],[1,0,0],[3,3,3],identity,[-1,0,0,1]),[9.5,11]);
  assert.equal(volumeRayInterval([-10,1,1],[1,0,0],[3,3,3],identity,[0,1,0,-2]),null);
});
test('opacity is invariant to ray sample spacing and presets retain transparent air/background',()=>{
  const once=opticalAlpha(.3,.7,12),small=opticalAlpha(.3,.7,.5);
  assert.ok(Math.abs(once-(1-(1-small)**24))<1e-12);
  assert.equal(opticalAlpha(.3,0,12),0);
  for(const preset of ['ct-bone','ct-tissue','mri']){
    const lut=volumeTransfer(preset,[0,900]);
    assert.equal(lut.data.length,4096);assert.equal(lut.data[3],0);
    assert.ok(lut.data[4095]>0);
    assert.ok(lut.range[1]>lut.range[0]);
  }
  assert.equal(volumeSettings({},'CT').preset,'ct-bone');
  assert.equal(volumeSettings({},'T1CE').preset,'mri');
  assert.equal(volumeSettings({density:99}).density,2);
});
test('saved volume choices round-trip detached from source arrays and reject malformed settings',()=>{
  const source={compartments:[],brainMask:null,metadata:{}},input={volumeRendering:{enabled:true,preset:'ct-bone',density:.45}};
  const saved=checkedViewerSettings(source,input);assert.deepEqual(saved,input);
  saved.volumeRendering.density=.1;assert.equal(input.volumeRendering.density,.45);
  for(const volumeRendering of [null,[],{enabled:1},{density:NaN},{density:-1},{density:3},{preset:'CTA-vessels'},{arbitrary:1}])
    assert.throws(()=>checkedViewerSettings(source,{volumeRendering}));
});

test('explicit skull modes survive settings validation',()=>{
 const source={compartments:[],brainMask:null,metadata:{}};
 for(const skullMode of ['exterior','transparent','cutaway'])assert.equal(checkedViewerSettings(source,{skullMode}).skullMode,skullMode);
 assert.throws(()=>checkedViewerSettings(source,{skullMode:'guess'}));
});

test('angiography is automatic only for declared CTA, with MRI sequence and segmentation-only behavior',()=>{
  assert.equal(volumeSettings({},'CT').preset,'ct-bone');assert.equal(volumeSettings({},'CTA').preset,'ct-angiography');
  assert.equal(volumeSettings({preset:'ct-angiography'},'CT').preset,'ct-angiography');
  assert.equal(volumeSettings({preset:'ct-angiography'},'T1CE').preset,'mri');
  assert.equal(volumeSettings({},'segmentation only').enabled,false);assert.equal(volumeSettings({},'analytic signal').enabled,false);
  assert.notDeepEqual(volumeTransfer('mri',[0,900],{},'T2').data,volumeTransfer('mri',[0,900],{},'T1').data);
  const custom={points:[[0,0,0,0,0],[1,.8,.5,.2,.4]]};
  assert.deepEqual(volumeTransfer('mri',[0,900],custom,'T1').data,volumeTransfer('mri',[0,900],custom,'T2').data);
  const source={compartments:[],brainMask:null,metadata:{}},settings={smoothSurfaces:false,volumeRendering:{preset:'ct-angiography',density:.3}};
  assert.deepEqual(checkedViewerSettings(source,settings),settings);assert.throws(()=>checkedViewerSettings(source,{smoothSurfaces:1}));
});
