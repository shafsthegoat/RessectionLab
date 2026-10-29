import test from 'node:test';
import assert from 'node:assert/strict';
import {microscopeCamera,microscopeFov} from './microscope.ts';
import {validCameraState,validMicroscopeSettings,layerAppearance} from './workspaceSettings.ts';
import {surgicalToolMesh,updateSurgicalToolAnimation} from './surgicalGeometry.ts';
import * as THREE from 'three';
const optical={target:[-10,15,70],outward:[0,0,1],angleDeg:12,azimuthDeg:30,workingDistanceMm:160,fieldMm:42,focusOffsetMm:0,light:2.5};
test('microscope optical controls preserve physical focus and do not depend on instrument state',()=>{
  const s=structuredClone(optical),before=JSON.stringify(s),camera=microscopeCamera(s);
  assert(validCameraState(camera));assert.equal(JSON.stringify(s),before);
  assert(Math.abs(new THREE.Vector3(...camera.position).distanceTo(new THREE.Vector3(...camera.target))-160)<1e-10);
  assert.deepEqual(camera.target,s.target);
  assert(microscopeFov({...s,fieldMm:21})<microscopeFov(s));
  const refocus=microscopeCamera({...s,focusOffsetMm:5});assert.deepEqual(refocus.target,[-10,15,75]);
  const tilt=microscopeCamera({...s,angleDeg:50});assert.notDeepEqual(tilt.position,camera.position);assert.deepEqual(tilt.target,camera.target);
});
test('surgeon mode enforces opaque tissue and inspection transparency remains explicit',()=>{
  const settings={operativeView:'surgeon',microscope:optical,layerOpacity:{brain:.04,tumor:.5}};
  assert.equal(layerAppearance('brain','#aaaaaa',settings,'brain').opacity,1);
  assert.equal(layerAppearance('tumor','#ff00ff',settings,'tumor').opacity,1);
  assert.equal(layerAppearance('brain','#aaaaaa',{...settings,microscope:{...optical,inspectionTransparency:true}},'brain').opacity,.04);
});
test('invalid persisted optics are refused including unknown physics-like fields',()=>{
  assert(validMicroscopeSettings(optical));
  for(const patch of [{outward:[0,0,0]},{workingDistanceMm:0},{fieldMm:NaN},{light:10},{target:[true,0,0]},{gravity:[0,0,1]},{inspectionTransparency:1}])assert(!validMicroscopeSettings({...optical,...patch}));
});
test('revised tool working meshes use accepted capsules; jaws never invent a close/release cycle',()=>{
  const slot={instrument:'bipolar_forceps',position:[1,2,3],quaternion:[0,0,0,1],active:true,parameters:{jawOpeningMm:4},contactGeometry:[{a:[-2,0,-2],b:[-2,0,0],radiusMm:.35,part:'working'},{a:[2,0,-2],b:[2,0,0],radiusMm:.35,part:'working'}]};
  const tool={id:slot.instrument,radiusMm:1,shaftRadiusMm:.5,workingLengthMm:40};
  const mesh=surgicalToolMesh(tool,slot);mesh.updateMatrixWorld(true);const before=new THREE.Box3().setFromObject(mesh);
  updateSurgicalToolAnimation(mesh,2,.5);mesh.updateMatrixWorld(true);const after=new THREE.Box3().setFromObject(mesh);
  assert.deepEqual(after.min.toArray(),before.min.toArray());assert.deepEqual(after.max.toArray(),before.max.toArray());
  assert(Math.abs(after.min.x-(1-2-.35))<1e-6);assert(Math.abs(after.max.z-(3+.35))<1e-6);
  assert.deepEqual(mesh.position.toArray(),slot.position);
});
