import assert from 'node:assert/strict';
import { test } from 'node:test';
import { ctaArterialCutaway, followSourceSection, restoreSourceSection, sourceDepthSlabAtCursor, sourceSectionAtCursor, sourceSectionSnapshot } from './sourceSection.ts';
import { inverseAffine, rasAffine, transformPoint, volumeBounds } from './coordinates.ts';

test('section uses linked RAS millimeters on an oblique LPS source and exposes its camera-facing cut', () => {
  const affine = rasAffine([[2,.25,0,10],[0,3,0,-20],[.5,0,4,30],[0,0,0,1]],'LPS+');
  const cursor = transformPoint(affine,[3,4,2]), bounds = volumeBounds([10,12,8],affine);
  const source = {activePlane:'axial',clipping:{axis:'x',value:-20,enabled:false,flipped:false},volumeRendering:{enabled:true,density:.18},clipBox:{enabled:true,min:[-30,-30,20],max:[20,30,70]},obliqueClip:{enabled:true,normal:[0,0,1],offset:-30}};
  const before = structuredClone(source), next = sourceSectionAtCursor(source,cursor,bounds,[100,100,100]);
  assert.equal(next.activePlane,'sagittal'); assert.equal(next.showPlane,true);
  assert.equal(next.clipping.value,cursor[0]); assert.equal(next.clipping.enabled,true); assert.equal(next.clipping.flipped,true);
  assert.ok(transformPoint(inverseAffine(affine),cursor).every((value,index)=>Math.abs(value-[3,4,2][index])<1e-10));
  assert.equal(sourceSectionAtCursor(source,cursor,bounds,[-100,0,0]).clipping.flipped,false);
  assert.equal(next.clipBox,source.clipBox); assert.equal(next.obliqueClip,source.obliqueClip); assert.equal(next.volumeRendering,source.volumeRendering);
  assert.deepEqual(source,before);
});

test('fresh CTA cutaway clears all source and display artery vertices and preserves acquired settings',()=>{
  const mesh=(z,displayed=z)=>({
    source:Float32Array.from([38,65,z,64,33,z,55,52,z-12]),
    positions:Float32Array.from([38,65,displayed,64,33,displayed,55,52,z-12]),
    indices:Uint32Array.from([0,1,2]),
  });
  const skull=mesh(20);
  skull.source.set([15,10,20,100,109,20,100,10,20]);
  skull.positions=skull.source;
  const artery={name:'artery',kind:'artery',sourceSeries:'cta',surface:mesh(50,51)};
  const volume={displayIdentity:'source',modality:'CTA',frame:'RAS+',shape:[140,140,130],
    affine:[[.5,0,0,-30],[0,.5,0,-35],[0,0,1,-110],[0,0,0,1]],
    compartments:[artery,{name:'bone',kind:'bone',sourceSeries:'cta',surface:skull}]};
  const settings={presentation:'segmentation',volumeRendering:{enabled:true,density:.12,shadows:true},sliceOverlay:'contour'};
  const visible={artery:true,bone:true};
  const cut=ctaArterialCutaway(volume,'cta',visible,settings);
  assert.equal(cut.clipping.value,-51);
  assert.equal(cut.clipping.flipped,true);
  assert.equal(cut.showPlane,false);
  assert.equal(cut.volumeRendering,settings.volumeRendering);
  assert.ok(cut.camera.position[2]>cut.clipping.value);
  assert.equal(ctaArterialCutaway({...volume,modality:'TOF'},'cta',visible,settings),undefined);
  assert.equal(ctaArterialCutaway(volume,'cta',{...visible,bone:false},settings),undefined);
  assert.equal(ctaArterialCutaway(volume,'different',visible,settings),undefined);
  const withTarget={...volume,compartments:[
    ...volume.compartments,{name:'target',kind:'tumor',sourceSeries:'cta',surface:mesh(100)},
  ]};
  assert.equal(ctaArterialCutaway(withTarget,'cta',{...visible,target:true},settings),undefined);
  assert.deepEqual(ctaArterialCutaway(withTarget,'cta',{...visible,target:true},settings,
    [{id:'target',role:'context',provenance:{kind:'simulated_observation'}}]),cut);
  const wide=mesh(53);
  wide.source[8]=-30;
  assert.equal(ctaArterialCutaway({...volume,compartments:[{...artery,surface:wide},volume.compartments[1]]},
    'cta',visible,settings),undefined);
});

test('section follows the linked cursor and chosen axis without changing source window or extra cuts', () => {
  const bounds = [[-50,-60,-70],[50,60,70]], settings = {activePlane:'axial',showPlane:true,contrast:1.25,clipping:{axis:'z',value:0,enabled:true,flipped:false}};
  const moved = followSourceSection(settings,[10,20,30],bounds);
  assert.equal(moved.clipping.value,30); assert.equal(moved.contrast,1.25); assert.equal(moved.clipping.flipped,false);
  const rotated = followSourceSection({...moved,clipping:{...moved.clipping,axis:'y'}},[10,20,30],bounds);
  assert.equal(rotated.activePlane,'coronal'); assert.equal(rotated.clipping.value,20);
  assert.equal(followSourceSection(rotated,[10,20,30],bounds),rotated);
});

test('exit restores exact previous source-plane and cut state while retaining later appearance changes', () => {
  const settings = {activePlane:'coronal',showPlane:false,clipping:{axis:'y',value:7,enabled:true,flipped:false},contrast:.8};
  const snapshot = sourceSectionSnapshot(settings);
  const section = sourceSectionAtCursor(settings,[1,2,3],[[-10,-10,-10],[10,10,10]],[-50,0,0]);
  const restored = restoreSourceSection({...section,contrast:1.4,contextStyle:'shell'},snapshot);
  assert.deepEqual(restored,{...settings,contrast:1.4,contextStyle:'shell'});
  const empty = {contrast:1.2}, emptySnapshot = sourceSectionSnapshot(empty);
  assert.deepEqual(restoreSourceSection(sourceSectionAtCursor(empty,[1,2,3],[[-10,-10,-10],[10,10,10]]),emptySnapshot),empty);
  settings.clipping.value = 9;
  assert.equal(snapshot.clipping.value,7);
});

test('a depth slab follows the cursor in physical RAS along one axis and retains the full transverse field', () => {
  const bounds = [[-85,-70,-8],[89,90,80]];
  assert.deepEqual(sourceDepthSlabAtCursor([0,12,8],bounds,'axial',44),
    {enabled:true,min:[-85,-70,-14],max:[89,90,30]});
  assert.deepEqual(sourceDepthSlabAtCursor([0,12,8],bounds,'coronal',28),
    {enabled:true,min:[-85,-2,-8],max:[89,26,80]});
  assert.deepEqual(sourceDepthSlabAtCursor([1,12,9],bounds,'sagittal',20),
    {enabled:true,min:[-9,-70,-8],max:[11,90,80]});
  assert.deepEqual(bounds,[[-85,-70,-8],[89,90,80]]);
});
