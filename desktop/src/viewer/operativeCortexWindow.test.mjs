import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {corticalSourceWindow} from './operativeCortexWindow.ts';
import {operativeFieldContext} from './operativeFieldContext.ts';
import {pickOperativeStructure} from './operativeFieldPick.ts';
import {operativeDimensions,retractionOffset} from './operativeFieldModel.ts';

const basis=[[1,0,0],[0,1,0],[0,0,1]],origin=[-5,-5,-2],shape=[11,11,24],s=1;
const index=(x,y,z)=>((x-origin[0])*shape[1]+y-origin[1])*shape[2]+z-origin[2];
const cortex={id:'gm',label:'Source gray matter',sourceSeries:'mr',role:'brain',detail:'grayMatter',sampledVoxels:12,labelSpacingMm:[.6,.6,1]};
function study(){
  const voxels=new Uint16Array(11*11*24);
  voxels[index(0,0,0)]=8|36;
  const field={shape,origin,voxels,spacingMm:s,radiusMm:5,routeRadiusMm:2,stations:{brain:12},targetRasMm:[0,0,0],
    basisRas:basis,snapshotId:'r',layers:[cortex]};
  const positions=Float32Array.of(-.5,-.5,12, .5,-.5,12, 0,.5,12);
  const volume={displayIdentity:JSON.stringify(['p','mr','sha','r',false]),frame:'RAS+',affine:[[2,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
    compartments:[{name:'gm',kind:'gray-matter',alignment:'reference',sourceSeries:'mr',surfaceDiagnostics:{maxDisplacementMm:.15},
      surface:{positions,normals:Float32Array.of(0,0,1,0,0,1,0,0,1),indices:Uint32Array.of(0,1,2)}}]};
  return {field,volume};
}

test('all selected-target and conflicting gray-matter cells under the opening stay on the local side',()=>{
  const {field}=study();
  field.voxels[index(1,0,5)]=8; // also protect a target that has no GM bit
  field.voxels[index(0,1,9)]=36|16;
  field.voxels[index(4,4,15)]=36|2; // remote from this narrow opening
  const w=corticalSourceWindow(field,cortex,1,.15);
  assert.equal(w.protectedZ,9);
  assert(w.floor>9+Math.hypot(.6,.6,1)/2+.15);
  assert.equal(corticalSourceWindow(field,{...cortex,labelSpacingMm:undefined},1,.15),null);
  for(const unmeasured of [undefined,NaN,-1])assert.equal(corticalSourceWindow(field,cortex,1,unmeasured),null);
});

test('planning and exterior share triangles, native normals and physical retraction for registered gray matter',()=>{
  const {field,volume}=study();
  const group=operativeFieldContext(volume,field,'p',4);
  assert.equal(group.children.length,1);
  const mesh=group.children[0],source=volume.compartments[0].surface;
  assert.equal(mesh.geometry.index.array,source.indices);
  assert.equal(mesh.geometry.getAttribute('position').array,source.positions);
  assert.equal(mesh.geometry.getAttribute('normal').array,source.normals);
  assert.equal(mesh.userData.sourceLayer.exposedVertices,3);
  mesh.morphTargetInfluences[0]=1;
  const moved=new THREE.Vector3(), original=new THREE.Vector3(),{opening}=operativeDimensions(field);
  for(let i=0;i<3;i++){
    mesh.getVertexPosition(i,moved).applyMatrix4(mesh.matrixWorld);
    original.fromArray(source.positions,3*i).applyMatrix4(mesh.matrixWorld);
    const expected=new THREE.Vector3(...retractionOffset(...original.toArray(),opening,field.stations.brain));
    assert(moved.sub(original).distanceTo(expected)<1e-6);
  }
  for(const changed of [{...field,layers:[cortex,{...cortex,id:'second'}]}, {...field,snapshotId:'other'}])
    assert.equal(operativeFieldContext(volume,changed,'p').children.length,0);

  // Pick the centre of the *deformed* triangle. Confidence and provenance are
  // tied to its undeformed sample, and the raw local view clips this surface.
  const scene=new THREE.Scene();scene.add(group);scene.updateMatrixWorld(true);
  const centroid=new THREE.Vector3();
  for(let i=0;i<3;i++)centroid.add(mesh.getVertexPosition(i,new THREE.Vector3()).applyMatrix4(mesh.matrixWorld));
  centroid.divideScalar(3);
  const camera=new THREE.OrthographicCamera(-4,4,4,-4,.1,40);
  camera.position.copy(centroid).add(new THREE.Vector3(0,0,10));camera.lookAt(centroid);camera.updateMatrixWorld();
  const window=mesh.userData.sourceLayer.cortex;
  window.activeFloor.value=window.floor;
  const picked=pickOperativeStructure(scene,camera,new THREE.Vector2(),field);
  assert.equal(picked.layers[0].id,'gm');
  assert(new THREE.Vector3(...picked.ras).distanceTo(new THREE.Vector3(0,-1/6,12))<1e-6);
  window.activeFloor.value=window.top+1;
  assert.equal(pickOperativeStructure(scene,camera,new THREE.Vector2(),field),null);
});
