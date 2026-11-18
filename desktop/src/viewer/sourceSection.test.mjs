import assert from 'node:assert/strict';
import { test } from 'node:test';
import { followSourceSection, restoreSourceSection, sourceSectionAtCursor, sourceSectionSnapshot } from './sourceSection.ts';
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
