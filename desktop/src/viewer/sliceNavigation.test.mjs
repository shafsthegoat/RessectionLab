import assert from 'node:assert/strict';
import { test } from 'node:test';
import { inverseAffine, transformPoint } from './coordinates.ts';
import { patientSliceCursor, patientSliceNavigation } from './sliceNavigation.ts';

test('source-aligned counters use patient position and preserve anisotropic source sampling', () => {
  const affine = [[-2,0,0,18],[0,3,0,-12],[0,0,4,-20],[0,0,0,1]], shape = [10,9,11];
  const inverse = inverseAffine(affine), cursor = transformPoint(affine,[4,3,5]);
  const axial = patientSliceNavigation(affine,inverse,shape,cursor,'axial');
  assert.equal(axial.step,4); assert.equal(axial.index,6); assert.equal(axial.count,11); assert.equal(axial.resampled,false);
  assert.deepEqual(patientSliceCursor(affine,inverse,shape,cursor,'axial',100),[10,-3,20]);
  const sagittal = patientSliceNavigation(affine,inverse,shape,cursor,'sagittal');
  assert.equal(sagittal.count,10); assert.equal(sagittal.index,6); assert.equal(sagittal.step,2);
});

test('oblique scrubbing clips the patient-frame line to actual source support', () => {
  const q = Math.SQRT1_2, affine = [[q,-q,0,0],[q,q,0,0],[0,0,2,-4],[0,0,0,1]], shape = [11,11,5];
  const inverse = inverseAffine(affine), cursor = transformPoint(affine,[1,1,2]);
  const nav = patientSliceNavigation(affine,inverse,shape,cursor,'sagittal');
  assert.equal(nav.resampled,true); assert.ok(nav.max < 2); assert.ok(nav.min > -2);
  for (const value of [-100,-1,0,1,100]) {
    const next = patientSliceCursor(affine,inverse,shape,cursor,'sagittal',value);
    assert.equal(next[1],cursor[1]); assert.equal(next[2],cursor[2]);
    assert.ok(transformPoint(inverse,next).every((v,i)=>v>=-1e-6&&v<=shape[i]-1+1e-6));
  }
});

test('a linked cursor outside source support cannot fabricate a valid scrub range', () => {
  const affine = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], inverse = inverseAffine(affine);
  assert.equal(patientSliceNavigation(affine,inverse,[5,5,5],[8,2,2],'axial'),null);
  assert.deepEqual(patientSliceCursor(affine,inverse,[5,5,5],[8,2,2],'axial',3),[8,2,2]);
});

test('sheared source planes are identified from plane orientation rather than cursor travel direction', () => {
  const affine = [[1,.5,0,0],[0,1,0,0],[0,0,2,0],[0,0,0,1]], inverse = inverseAffine(affine), shape = [11,11,5];
  const cursor = transformPoint(affine,[5,5,2]);
  assert.equal(patientSliceNavigation(affine,inverse,shape,cursor,'sagittal').resampled,true);
  assert.equal(patientSliceNavigation(affine,inverse,shape,cursor,'axial').resampled,false);
  const low = patientSliceCursor(affine,inverse,shape,cursor,'sagittal',-100);
  const high = patientSliceCursor(affine,inverse,shape,cursor,'sagittal',100);
  assert.deepEqual(transformPoint(inverse,low),[0,5,2]);
  assert.deepEqual(transformPoint(inverse,high),[10,5,2]);
});
