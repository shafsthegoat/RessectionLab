import assert from 'node:assert/strict';
import {before,after,test} from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import * as THREE from 'three';
let server,cache,VolumeRenderer,InstrumentDisplayState;
before(async()=>{
  cache=await fs.mkdtemp(path.join(os.tmpdir(),'native-planning-display-'));
  server=await createServer({root:fileURLToPath(new URL('../../',import.meta.url)),cacheDir:cache,logLevel:'error',optimizeDeps:{noDiscovery:true,include:[]},server:{middlewareMode:true,hmr:false,ws:false},appType:'custom'});
  ({VolumeRenderer}=await server.ssrLoadModule('/src/viewer/VolumeRenderer.ts'));({InstrumentDisplayState}=await server.ssrLoadModule('/src/viewer/inspectionTool.ts'));
});
after(async()=>{await server?.close();if(cache)await fs.rm(cache,{recursive:true,force:true});});
test('actual renderer displays all three source-frame corridor radii without admitting physical execution',()=>{
  const r=Object.create(VolumeRenderer.prototype),shaders=[{uniforms:{uRouteCount:{value:0},uSelectedRoute:{value:-1},...Object.fromEntries(['uShaftStart','uShaftEnd','uTipEnd'].map(k=>[k,{value:Array.from({length:3},()=>new THREE.Vector3())}])),uRadii:{value:Array.from({length:3},()=>new THREE.Vector2())},uRouteColors:{value:Array.from({length:3},()=>new THREE.Color())}}}];
  Object.assign(r,{volume:{frame:'LPS+'},tools:new THREE.Group(),inspectionTools:new THREE.Group(),recordedTools:new THREE.Group(),instrumentDisplay:new InstrumentDisplayState(),mode:'anatomy',materials:()=>shaders,requestRender(){}});
  const routes=['A','B','C'].map((slot,i)=>({route_id:slot,comparisonSlot:slot,scope:'geometric-approach',selected:i===1,entry_mm:[10+i,20,70],target_mm:[10,20,10],tool:{working_length_mm:120,tip_length_mm:5,shaft_radius_mm:3+i,tip_radius_mm:3+i}}));
  r.updateRoutes(routes);
  assert.equal(shaders[0].uniforms.uRouteCount.value,3);
  assert.equal(shaders[0].uniforms.uSelectedRoute.value,1,'selected corridor stays legible at overlapping source-slice capsules');
  for(let i=0;i<3;i++){
    assert.deepEqual(shaders[0].uniforms.uShaftStart.value[i].toArray(),[-10-i,-20,70]);
    assert.deepEqual(shaders[0].uniforms.uShaftEnd.value[i].toArray(),[-10,-20,10]);
    assert.deepEqual(shaders[0].uniforms.uRadii.value[i].toArray(),[3+i,3+i]);
  }
  assert.equal(r.tools.children.filter(p=>p.userData.geometryRole==='virtual-axis').length,3);
  assert.equal(r.tools.children.filter(p=>p.userData.geometryRole==='entry-footprint').length,3);
  assert.equal(r.tools.children.filter(p=>p.userData.comparisonSlot==='C').length,6);
  assert.equal(r.volume.frame,'LPS+');
  r.tools.traverse(object=>{object.geometry?.dispose();object.material?.dispose();});
});
test('concern focus preserves the assessed local point and can wait for a hidden label surface',()=>{
  const r=Object.create(VolumeRenderer.prototype),bundle=new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-50,-70,-20),new THREE.Vector3(50,70,100)]),new THREE.LineBasicMaterial());
  let camera;
  Object.assign(r,{surfaces:new Map(),tractObjects:new Map([['bundle',bundle]]),camera:new THREE.PerspectiveCamera(34,1,.1,1000),controls:{target:new THREE.Vector3()},panes:{anatomy:{getBoundingClientRect:()=>({width:600,height:600})}},restoreCamera:state=>{camera=state;}});
  r.camera.position.set(0,-200,150);
  assert.deepEqual(r.focusStructure('bundle',[-31,20,45]),[-31,20,45]);assert.deepEqual(camera.target,[-31,20,45]);
  assert.deepEqual(r.focusStructure('pending-label',[-30,19,43]),[-30,19,43]);assert.deepEqual(camera.target,[-30,19,43]);
  assert.deepEqual(r.focusStructure('bundle'),[0,0,40]);assert.equal(r.focusStructure('pending-label'),null);
  bundle.geometry.dispose();bundle.material.dispose();
});
