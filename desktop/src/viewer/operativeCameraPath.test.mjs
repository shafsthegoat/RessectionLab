import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {entryApproachEnd,operativeCameraPose,operativeSourceFit} from './operativeCameraPath.ts';

const field={stations:{bone:32.8,dura:25.6,brain:18.4,tumor:12.5},spacingMm:.3125,radiusMm:24,routeRadiusMm:2};
const source={center:new THREE.Vector3(25,-30,-40),size:210};
const pose=(phase,progress,view='operative',retract=true)=>operativeCameraPose(field,8.6,source,phase,progress,retract,view);

test('the source fit follows actual world-transformed geometry',()=>{
  const group=new THREE.Group();
  assert.equal(operativeSourceFit(group),null);
  const mesh=new THREE.Mesh(new THREE.BoxGeometry(30,40,50));group.add(mesh);
  group.position.set(100,20,-80);mesh.position.set(2,4,6);
  const fit=operativeSourceFit(group);
  assert(fit.center.distanceTo(new THREE.Vector3(102,24,-74))<1e-8);
  assert(fit.size>40&&fit.size<90);
});

test('the planned entry precedes a held view for both openings',()=>{
  const start=pose('overview',0),begin=pose('bone',0),half=pose('bone',entryApproachEnd/2);
  assert.deepEqual(start,begin); assert.equal(start.marker,1);
  assert(half.size<start.size&&half.size>pose('bone',1).size);
  assert.deepEqual(pose('bone',entryApproachEnd),pose('dura',1));
});

test('deep retraction and aspiration preserve the opaque opening until the user selects inspection',()=>{
  const opening=pose('dura',1);
  const apertureFraction=2*8.6/opening.size;
  assert(apertureFraction>.5&&apertureFraction<.7, 'frame the aperture and its surrounding intact rim');
  for(const phase of ['brain','tumor'])for(const progress of [0,.24,.5,1]){
    assert.deepEqual(pose(phase,progress),opening);
    assert.deepEqual(pose(phase,progress,'exterior').center,source.center);
    const section=pose(phase,progress,'section');
    assert.equal(section.center.z,field.stations[phase]);
    assert.notDeepEqual(section.bearing,opening.bearing);
  }
});
