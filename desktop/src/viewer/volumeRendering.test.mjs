import test from 'node:test';
import assert from 'node:assert/strict';
import {isCtModality,volumeSettings,volumeTransfer,volumeRayInterval,opticalAlpha,mriBoundaryExtinction} from './volumeTransfer.ts';
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

test('presentation and slice annotations reopen in legacy saves and reject malformed choices',()=>{
  const source={compartments:[],brainMask:null,metadata:{}};
  for(const presentation of ['segmentation','vascular'])for(const sliceOverlay of ['contour','fill']){
    const stored={presentation,sliceOverlay,volumeRendering:{enabled:true,preset:'auto',density:.11}};
    const reopened=checkedViewerSettings(source,JSON.parse(JSON.stringify(stored)));
    assert.deepEqual(reopened,stored);
    reopened.volumeRendering.density=.9;
    assert.equal(stored.volumeRendering.density,.11);
  }
  for(const key of ['presentation','sliceOverlay'])for(const invalid of ['unknown',null,1,[],{}])
    assert.throws(()=>checkedViewerSettings(source,{[key]:invalid}));
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

test('CT thresholds require an acquired CT modality even when a CT preset was saved',()=>{
  assert.equal(isCtModality('  CTA'),true);assert.equal(isCtModality('CTAX'),false);assert.equal(isCtModality('T1CE'),false);
  for(const preset of ['ct-bone','ct-tissue','ct-angiography']){
    for(const modality of ['MRI','T1CE','T2 FLAIR','3D TOF MRA'])
      assert.equal(volumeSettings({preset},modality).preset,'mri');
    for(const modality of ['CT','CTA','CT angiography'])
      assert.equal(volumeSettings({preset},modality).preset,preset);
  }
  assert.equal(volumeSettings({},'CT angiography').preset,'ct-angiography');
  assert.equal(volumeSettings({},'T1CE post-contrast').preset,'mri');
  const input={preset:'ct-bone',density:.3},saved=structuredClone(input);
  volumeSettings(input,'T1CE');assert.deepEqual(input,saved,'presentation must not overwrite persisted choices');
});

test('declared bright-blood MRI suppresses background and emphasizes acquired high signal without coloring vessels',()=>{
  const structural=volumeTransfer('mri',[0,900],{},'T1CE');
  const alpha=(lut,x)=>lut.data[Math.round(x*1023)*4+3];
  for(const modality of ['MRA','3D TOF','MRI angiography']){
    assert.equal(volumeSettings({},modality).preset,'mri');
    const lut=volumeTransfer('mri',[0,900],{},modality);
    assert.deepEqual(lut.range,structural.range);
    assert.equal(alpha(lut,0),0);
    assert.ok(alpha(lut,.4)<alpha(structural,.4),'background tissue must recede');
    assert.ok(alpha(lut,.9)>alpha(structural,.9),'bright acquired signal must remain detailed');
    const top=[...lut.data.slice(-4,-1)];
    assert.ok(Math.max(...top)-Math.min(...top)<.15,'signal remains neutral ivory, not an arterial classification');
  }
  assert.deepEqual(volumeTransfer('mri',[0,900],{},'T1CE post-contrast').data,structural.data);
  const points=[[0,0,0,0,0],[1,.8,.5,.2,.4]],custom=structuredClone(points);
  assert.deepEqual(volumeTransfer('mri',[0,900],{points},'MRA').data,volumeTransfer('mri',[0,900],{points},'T1CE').data);
  assert.deepEqual(points,custom);
});

test('transparent MRI context retains precise extinction and emphasizes physical boundaries only for automatic transfers',()=>{
  const automatic=volumeTransfer('mri',[0,900],undefined,'T1CE');
  assert.ok(automatic.data instanceof Float32Array);
  // This low extinction rounded to zero in the byte LUT. It must contribute
  // proportionally at low density instead of appearing as a transfer band.
  const faint=automatic.data[Math.round(.1*1023)*4+3];
  assert.ok(faint>0&&faint<1/510);
  assert.equal(automatic.gradientOpacity,true);
  assert.equal(volumeTransfer('ct-bone',[-1000,1000]).gradientOpacity,false);
  assert.equal(volumeTransfer('mri',[0,900],{points:[[0,0,0,0,0],[1,1,1,1,.4]]}).gradientOpacity,false);
  assert.equal(mriBoundaryExtinction(0),.06);
  assert.equal(mriBoundaryExtinction(.01),.06);
  assert.equal(mriBoundaryExtinction(.075),1);
  assert.ok(mriBoundaryExtinction(.025)<.2);
  // Equivalent physical boundaries on 1mm and 2mm grids have the same
  // derivative after central difference and the affine covector transform.
  assert.equal(mriBoundaryExtinction((.1-0)/2),mriBoundaryExtinction((.2-0)/4));
  const coarse=opticalAlpha(.07*.025*mriBoundaryExtinction(.025),1,200);
  const full=opticalAlpha(.07*.025,1,200);
  assert.ok(coarse<full*.25,'uniform interior context must recede visibly');
});

test('unclassified CTA signal is neutral ivory with the original HU extinction and custom transfers intact',()=>{
  const lut=volumeTransfer('ct-angiography',[-1000,2000],undefined,'CTA');
  const original=[[-1000,0],[80,0],[140,.01],[220,.12],[360,.38],[600,.5],[900,.52],[1800,.8],[3000,1]];
  assert.deepEqual(lut.range,[-1000,3000]);
  for(let i=0;i<1024;i++){
    const hu=lut.range[0]+i/1023*(lut.range[1]-lut.range[0]);
    let knot=0;while(knot<original.length-2&&hu>original[knot+1][0])knot++;
    const [a,alphaA]=original[knot],[b,alphaB]=original[knot+1],t=Math.max(0,Math.min(1,(hu-a)/(b-a)));
    assert.ok(Math.abs(lut.data[i*4+3]-(alphaA*(1-t)+alphaB*t))<1e-7,'classification-independent extinction must stay unchanged');
    const rgb=lut.data.slice(i*4,i*4+3);
    assert.ok(Math.max(...rgb)-Math.min(...rgb)<.1,'unclassified CTA must not acquire the rust arterial hue');
  }
  const points=[[0,0,0,0,0],[.5,.8,.2,.1,.2],[1,1,.8,.4,.9]],saved=structuredClone(points);
  assert.deepEqual(volumeTransfer('ct-angiography',[-1000,2000],{points},'CTA').data,
    volumeTransfer('ct-bone',[-1000,2000],{points,window:[-1000,3000]},'CTA').data,'explicit saved colors/extinction take precedence over the automatic preset');
  assert.deepEqual(points,saved);
});
