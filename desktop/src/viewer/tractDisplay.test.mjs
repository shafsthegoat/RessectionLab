import assert from 'node:assert/strict';
import {test} from 'node:test';
import {tractPreview,TractSliceIndex,clipSegmentToSlab,TRACT_COLORS} from './tractDisplay.ts';
import {approachCamera} from './approachView.ts';
import {validCameraState} from './workspaceSettings.ts';
import {routeAppearance,COMPARISON_COLORS} from './routeAppearance.ts';
import * as THREE from 'three';

const affine=[[0,-2,0,10],[3,0,0,-5],[0,0,4,20],[0,0,0,1]];
const tract={id:'supplied',label:'Publisher bundle',points:new Float32Array([0,0,0,1,0,0,1,1,0,20,20,20,21,20,20]),offsets:new Uint32Array([0,3,5])};
test('tract display applies the displayed affine once and never joins separate source streamlines',()=>{
  const beforePoints=tract.points.slice(),beforeOffsets=tract.offsets.slice(),preview=tractPreview(tract,affine);
  assert.deepEqual([...preview.positions],[10,-5,20,10,-2,20,10,-2,20,8,-2,20,-30,55,100,-30,58,100]);
  assert.equal(preview.sourceSegments,3);assert.equal(preview.displaySegments,3);assert.equal(preview.streamlineStride,1);
  assert.deepEqual(tract.points,beforePoints);assert.deepEqual(tract.offsets,beforeOffsets);
  assert.equal(tractPreview(tract,affine),preview,'stable source geometry reuses the prepared display buffers');
});
test('preview stride selects complete streamlines, retains full-resolution sources and does not alter supplied identities',()=>{
  const preview=tractPreview(tract,affine,1);
  assert.equal(preview.streamlineStride,3);assert.equal(preview.displaySegments,2);assert.equal(preview.sourceSegments,3);
  assert.deepEqual([...preview.positions],[10,-5,20,10,-2,20,10,-2,20,8,-2,20]);
  assert.equal(tract.label,'Publisher bundle');assert.equal(new Set(TRACT_COLORS).size,TRACT_COLORS.length);
});
test('source-slice fibers use slab intersections rather than projecting distant anatomy',()=>{
  assert.equal(clipSegmentToSlab([1,2,9],[3,4,10],2,0,.65),null);
  assert.deepEqual(clipSegmentToSlab([0,0,-2],[0,4,2],2,0,.5),[[0,1.5,-.5],[0,2.5,.5]]);
  assert.deepEqual(clipSegmentToSlab([0,0,0],[0,4,0],2,0,.5),[[0,0,0],[0,4,0]]);
  const index=new TractSliceIndex(tractPreview(tract,affine));
  assert.equal(index.segments(2,20,.5).length,2);assert.equal(index.segments(2,50,.5).length,0);assert.equal(index.segments(2,100,.5).length,1);
});
test('along and oblique framing handle vertical/LPS axes and preserve supplied entry/target',()=>{
  const route={route_id:'a',entry_mm:[10,20,80],target_mm:[10,20,20],tool:{shaft_radius_mm:4,tip_radius_mm:4}};
  const before=structuredClone(route);
  for(const frame of ['RAS+','LPS+'])for(const mode of ['along','oblique'])for(const aspect of [.3,2]){
    const camera=approachCamera(route,frame,mode,aspect);assert.ok(validCameraState(camera));
    assert.deepEqual(camera.target,[frame==='LPS+'?-10:10,frame==='LPS+'?-20:20,50]);
    assert.ok(camera.position[2]>camera.target[2],'the viewing position remains on the entry side');
  }
  assert.deepEqual(route,before);
  assert.equal(approachCamera({...route,target_mm:route.entry_mm},'RAS+','along'),null);
});
test('third approach keeps its color when the visible comparison order changes',()=>{
  assert.deepEqual(routeAppearance({comparisonSlot:'C'},0),{slot:'C',color:COMPARISON_COLORS.C});
  assert.deepEqual(routeAppearance({},2),{slot:'C',color:COMPARISON_COLORS.C});
});
test('corridor camera keeps both endpoints and the complete external virtual axis in frame',()=>{
  for(const entry of [[-31,20,74],[40,-20,65],[10,20,80]])for(const aspect of [.35,1,2])for(const mode of ['along','oblique']){
    const route={route_id:'framing',entry_mm:entry,target_mm:[-31,20,45],tool:{shaft_radius_mm:6,tip_radius_mm:6},window:{radius_mm:15}};
    const pose=approachCamera(route,'RAS+',mode,aspect),camera=new THREE.PerspectiveCamera(34,aspect,.1,5000);
    camera.position.set(...pose.position);camera.up.set(...pose.up);camera.lookAt(new THREE.Vector3(...pose.target));camera.updateMatrixWorld();
    const target=new THREE.Vector3(...route.target_mm),start=new THREE.Vector3(...entry),outward=start.clone().sub(target).normalize(),external=start.clone().addScaledVector(outward,18);
    for(const centre of [target,start,external])for(const axis of [0,1,2])for(const sign of [-1,1]){
      const point=centre.clone();point.setComponent(axis,point.getComponent(axis)+sign*6);const screen=point.project(camera);
      assert.ok(Math.abs(screen.x)<.95&&Math.abs(screen.y)<.95&&Math.abs(screen.z)<1,JSON.stringify({entry,aspect,mode,screen}));
    }
  }
});
