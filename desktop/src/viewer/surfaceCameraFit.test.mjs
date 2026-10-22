import assert from 'node:assert/strict';
import {test} from 'node:test';
import * as THREE from 'three';
import {surfaceCameraDistance} from './surfaceCameraFit.ts';
import {physicalBounds,placeInSourceFrame} from './sceneGeometry.ts';

test('surface fit retains 8% total margin for oblique source anatomy at narrow and wide pane sizes',()=>{
  const geometry=new THREE.SphereGeometry(50,48,32);
  geometry.scale(1.2,1.7,1.35);
  const before=geometry.attributes.position.array.slice();
  const mesh=new THREE.Mesh(geometry),anatomy=new THREE.Group();anatomy.add(mesh);
  placeInSourceFrame(mesh,[[-.8,-.6,0,-80],[-.6,.8,0,110],[0,0,1,45],[0,0,0,1]]);
  const centre=physicalBounds(anatomy).getCenter(new THREE.Vector3());
  for(const [width,height] of [[1070,580],[450,580],[280,800]])for(const view of [[.88,-1.65,.8],[0,1,0],[0,0,1]]){
    const direction=new THREE.Vector3(...view).normalize();
    const cameraUp=new THREE.Vector3(...(view[2]===1?[0,1,0]:[0,0,1]));
    const right=new THREE.Vector3().crossVectors(cameraUp,direction).normalize();
    const up=new THREE.Vector3().crossVectors(direction,right).normalize();
    const distance=surfaceCameraDistance(anatomy,centre,direction,right,up,THREE.MathUtils.degToRad(34),width/height);
    const camera=new THREE.PerspectiveCamera(34,width/height,.1,5000);camera.up.copy(cameraUp);
    camera.position.copy(centre).addScaledVector(direction,distance);camera.lookAt(centre);camera.updateMatrixWorld();
    let maximum=0;
    for(let i=0;i<geometry.attributes.position.count;i++){
      const p=new THREE.Vector3().fromBufferAttribute(geometry.attributes.position,i).applyMatrix4(mesh.matrixWorld).project(camera);
      maximum=Math.max(maximum,Math.abs(p.x),Math.abs(p.y));
      assert(Math.abs(p.x)<=.920001&&Math.abs(p.y)<=.920001&&Math.abs(p.z)<1,'Every original vertex remains within the requested screen margin.');
    }
    assert(maximum>.91999,'Fit reaches the requested useful size.');
  }
  assert.deepEqual(geometry.attributes.position.array,before);
});
test('empty source geometry preserves the caller’s existing image-bounds fallback',()=>{
  assert.equal(surfaceCameraDistance(new THREE.Group(),new THREE.Vector3(),new THREE.Vector3(0,-1,0),new THREE.Vector3(1,0,0),new THREE.Vector3(0,0,1),Math.PI/6,1),null);
});
test('stable source IDs fit active indexed corners without historical extreme vertices',()=>{
  const active=Float32Array.of(-20,-30,0,20,-30,0,20,30,10,-20,30,10),compact=new THREE.BufferGeometry();compact.setAttribute('position',new THREE.BufferAttribute(active,3));compact.setIndex([0,1,2,0,2,3]);
  const historical=new Float32Array(active.length+3);historical.set([1e6,-1e6,1e6]);historical.set(active,3);
  const stable=new THREE.BufferGeometry();stable.setAttribute('position',new THREE.BufferAttribute(historical,3));stable.setIndex([1,2,3,1,3,4]);stable.userData.sourceIndexing='stable-source-ids';
  const a=new THREE.Group(),b=new THREE.Group(),ma=new THREE.Mesh(compact),mb=new THREE.Mesh(stable);a.add(ma);b.add(mb);
  const transform=[[-1,.3,0,10],[.2,2,.1,-20],[0,.1,1,30],[0,0,0,1]];placeInSourceFrame(ma,transform);placeInSourceFrame(mb,transform);
  const centre=physicalBounds(a).getCenter(new THREE.Vector3()),direction=new THREE.Vector3(.8,-1.6,.7).normalize(),right=new THREE.Vector3().crossVectors(new THREE.Vector3(0,0,1),direction).normalize(),up=new THREE.Vector3().crossVectors(direction,right).normalize();
  const positions=stable.attributes.position,calls={getX:[],getY:[],getZ:[]};
  for(const key of Object.keys(calls)){const original=positions[key].bind(positions);positions[key]=i=>{calls[key].push(i);return original(i);};}
  for(const aspect of [.4,1,2]){
    for(const key of Object.keys(calls))calls[key].length=0;
    assert.equal(surfaceCameraDistance(b,centre,direction,right,up,.6,aspect),surfaceCameraDistance(a,centre,direction,right,up,.6,aspect));
    for(const key of Object.keys(calls))assert.deepEqual(calls[key],[1,2,3,4],'each unique active vertex transforms once; the historical slot is never read');
  }
  assert.deepEqual(stable.attributes.position.array,historical);assert.deepEqual([...stable.index.array],[1,2,3,1,3,4]);
});
