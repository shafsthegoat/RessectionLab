import assert from 'node:assert/strict';
import {before,after,test} from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import * as THREE from 'three';
let server,cache,VolumeRenderer,surgicalToolMesh;
before(async()=>{
  cache=await fs.mkdtemp(path.join(os.tmpdir(),'surgical-camera-'));
  server=await createServer({root:fileURLToPath(new URL('../../',import.meta.url)),cacheDir:cache,logLevel:'error',optimizeDeps:{noDiscovery:true,include:[]},server:{middlewareMode:true,hmr:false,ws:false},appType:'custom'});
  ({VolumeRenderer}=await server.ssrLoadModule('/src/viewer/VolumeRenderer.ts'));
  ({surgicalToolMesh}=await server.ssrLoadModule('/src/viewer/surgicalGeometry.ts'));
});
after(async()=>{await server?.close();if(cache)await fs.rm(cache,{recursive:true,force:true});});
function fixture(width=650,height=650){
  const r=Object.create(VolumeRenderer.prototype),camera=new THREE.PerspectiveCamera(34,width/height,.1,5000),anatomy=new THREE.Group();
  const brain=new THREE.Mesh(new THREE.BoxGeometry(120,100,80),new THREE.MeshBasicMaterial());anatomy.add(brain);
  const controls={target:new THREE.Vector3(),update(){camera.lookAt(this.target);camera.updateMatrixWorld();}};
  Object.assign(r,{camera,controls,anatomy,bounds:[[-60,-50,-40],[60,50,40]],cameraRestored:false,pendingFit:null,
    panes:{anatomy:{getBoundingClientRect:()=>({width,height}),dataset:{}}},scanVolume:{active:false},surfaces:new Map([['brain',brain]]),
    recordedTools:new THREE.Group(),inspectionTools:new THREE.Group(),tools:new THREE.Group(),instrumentDisplay:{inspected:false,setRoutes(){}},syncInstrumentDisplay(){},
    surgicalGroup:new THREE.Group(),surgicalSlots:new Map(),surfaceGeometry:new Map(),replayGroup:new THREE.Group(),pendingReplayGroup:null,
    volume:{compartments:[{name:'brain'}]},mode:'anatomy',cursor:[0,0,0],visible:{brain:true},visibilitySignature:'1',opacity:.5,routeSignature:'[]',
    materials:()=>[],requestRender(){}});
  return r;
}
const saved={position:[550,-750,480],target:[10,-12,3],up:[0,0,1]};
function corners(box){return [0,1].flatMap(x=>[0,1].flatMap(y=>[0,1].map(z=>new THREE.Vector3([box.min.x,box.max.x][x],[box.min.y,box.max.y][y],[box.min.z,box.max.z][z]))));}
test('actual restore-before-update ordering preserves wide camera through implicit Rehearse modes and route refreshes',()=>{
  const r=fixture();let fits=0;const original=r.fitCamera.bind(r);r.fitCamera=(...args)=>{fits++;return original(...args);};
  // This is the actual ViewerWorkspace effect order: restore first, update later.
  r.restoreCamera(saved);const before=r.getCameraState();
  r.update([0,0,0],{brain:true},.5,[],'instruments');
  r.updateRoutes([]);r.setSurfaceSmoothing(false);r.setSurfaceSmoothing(true);
  for(const mode of ['anatomy','instruments','instruments'])r.update([0,0,0],{brain:true},.5,[],mode);
  assert.deepEqual(r.getCameraState(),before);assert.equal(fits,0);assert.equal(r.mode,'instruments');
  r.restoreCamera(saved);assert.deepEqual(r.getCameraState(),before,'a rewind echo retains the saved camera');
});
test('explicit Fit still executes in an unchanged mode and persists its result; later route updates retain it',()=>{
  const r=fixture(),published=[];r.restoreCamera(saved);r.mode='instruments';r.onCameraChange=camera=>published.push(camera);
  r.frameCamera('instruments');assert.notDeepEqual(r.getCameraState(),saved);assert.equal(published.length,1);assert.deepEqual(published[0],r.getCameraState());
  r.camera.position.add(new THREE.Vector3(30,40,50));r.controls.update();const orbit=r.getCameraState();
  r.frameCamera('instruments');assert.notDeepEqual(r.getCameraState(),orbit);assert.equal(published.length,2,'same-mode explicit Fit is not lost to mode equality');
  const fitted=r.getCameraState();r.updateRoutes([]);assert.deepEqual(r.getCameraState(),fitted);
});
test('explicit Instruments Fit encloses complete actual inserted and withdrawn surgical bodies at narrow and wide aspects',()=>{
  for(const [width,height]of [[300,900],[1200,500]]){
    const r=fixture(width,height),slots=[
      {instrument:'cranial_drill',inserted:true,active:true,position:[140,-160,90],quaternion:new THREE.Quaternion().setFromEuler(new THREE.Euler(.4,.7,.2)).toArray(),parameters:{}},
      {instrument:'cusa',inserted:false,active:false,position:[-240,300,-80],quaternion:new THREE.Quaternion().setFromEuler(new THREE.Euler(1.1,-.5,.4)).toArray(),parameters:{}}];
    const groups=slots.map(slot=>surgicalToolMesh({id:slot.instrument,radiusMm:4,shaftRadiusMm:1,workingLengthMm:120},slot));
    for(let i=0;i<groups.length;i++){r.surgicalGroup.add(groups[i]);r.surgicalSlots.set(['left','right'][i],{group:groups[i]});}
    // Neither an effect marker nor an editable ghost is an actual tool body.
    const marker=new THREE.Mesh(new THREE.BoxGeometry(),new THREE.MeshBasicMaterial());marker.position.set(10000,10000,10000);r.surgicalGroup.add(marker);
    r.surgicalGhost=marker.clone();r.surgicalGhost.position.set(-10000,-10000,-10000);
    r.frameCamera('instruments');assert(r.camera.position.distanceTo(r.controls.target)<5000,'effect/ghost do not inflate tool framing');
    for(const group of groups)for(const corner of corners(new THREE.Box3().setFromObject(group))){const projected=corner.project(r.camera);assert(Math.abs(projected.x)<1&&Math.abs(projected.y)<1&&Math.abs(projected.z)<1,JSON.stringify({width,height,projected}));}
    const fullSize=JSON.parse(r.panes.anatomy.dataset.cameraFit).size;
    groups[1].visible=false;r.frameCamera('instruments');const visibleSize=JSON.parse(r.panes.anatomy.dataset.cameraFit).size;
    assert(visibleSize.some((size,axis)=>size<fullSize[axis]),'a hidden body is excluded while a withdrawn visible body was included');
    const retained=r.getCameraState();groups[0].position.x+=20;r.updateRoutes([]);assert.deepEqual(r.getCameraState(),retained,'tool motion/route refresh does not auto-refit each frame');
  }
});
