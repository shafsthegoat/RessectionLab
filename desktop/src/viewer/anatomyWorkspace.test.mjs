import assert from 'node:assert/strict';
import {before,after,test} from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import * as THREE from 'three';
import {checkedFusionTransform} from './registeredFusion.ts';
import {transformPoint} from './coordinates.ts';
import {validCameraState,clippingForBounds} from './workspaceSettings.ts';

const identity=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
const source={caseHash:'sha256:'+'a'.repeat(64),caseId:'patient-a',frame:'RAS+',affine:identity,shape:[3,4,5],mri:new Float32Array(60),compartments:[]};
const fusion=()=>({seriesId:'series-b',label:'T2',referenceCaseHash:source.caseHash,registration:{status:'source-provided',overlayPermitted:true,sourceToReferenceRasMm:identity},volume:{...source}});

test('fusion uses source LPS affine and supplied rotation/translation exactly once',()=>{
  const layer=fusion();
  layer.volume={...source,frame:'LPS+',affine:[[2,0,0,10],[0,3,0,-20],[0,0,4,30],[0,0,0,1]]};
  layer.registration.sourceToReferenceRasMm=[[0,-1,0,100],[1,0,0,50],[0,0,1,-10],[0,0,0,1]];
  // Source voxel [1,2,3] -> source RAS [-12,14,42] -> reference [86,38,32].
  assert.deepEqual(transformPoint(checkedFusionTransform(source,layer),[86,38,32]).map(n=>Math.round(n*1e9)/1e9),[1,2,3]);
  assert.deepEqual(layer.volume.affine,[[2,0,0,10],[0,3,0,-20],[0,0,4,30],[0,0,0,1]]);
});
test('matching dimensions never authorize unreviewed, wrong-patient or singular fusion',()=>{
  for(const mutate of [f=>f.registration.status='unreviewed',f=>f.registration.overlayPermitted=false,f=>f.referenceCaseHash='another-patient',f=>f.registration.sourceToReferenceRasMm=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]]){
    const layer=fusion();mutate(layer);assert.throws(()=>checkedFusionTransform(source,layer));
  }
});
test('camera rejects collapsed or collinear views and cutaway stays in physical bounds',()=>{
  assert.equal(validCameraState({position:[0,0,0],target:[0,0,0],up:[0,0,1]}),false);
  assert.equal(validCameraState({position:[0,0,50],target:[0,0,0],up:[0,0,1]}),false);
  assert.deepEqual(clippingForBounds({axis:'x',value:999,enabled:true},[[-20,-30,-40],[80,90,100]]),{axis:'x',value:80,enabled:true,flipped:false});
});

let server,cache,VolumeRenderer,ScanVolumeRenderer;
before(async()=>{
  cache=await fs.mkdtemp(path.join(os.tmpdir(),'anatomy-workspace-'));
  server=await createServer({root:fileURLToPath(new URL('../../',import.meta.url)),cacheDir:cache,logLevel:'error',optimizeDeps:{noDiscovery:true,include:[]},server:{middlewareMode:true,hmr:false,ws:false},appType:'custom'});
  VolumeRenderer=(await server.ssrLoadModule('/src/viewer/VolumeRenderer.ts')).VolumeRenderer;
  ScanVolumeRenderer=(await server.ssrLoadModule('/src/viewer/ScanVolumeRenderer.ts')).ScanVolumeRenderer;
});
after(async()=>{await server?.close();if(cache)await fs.rm(cache,{recursive:true,force:true,maxRetries:3,retryDelay:50});});
function renderer(){
  const r=Object.create(VolumeRenderer.prototype),mesh=new THREE.Mesh(new THREE.BoxGeometry(),new THREE.MeshStandardMaterial());
  const uniforms={uColors:{value:[new THREE.Color()]},uLayerOpacities:{value:new Float32Array(8)},uClipEnabled:{value:0},uClipPlane:{value:new THREE.Vector4()},uFusionOpacity:{value:.5},uFusionActive:{value:0},uFusion:{value:null},uFusionShape:{value:new THREE.Vector3()},uWorldToFusion:{value:new THREE.Matrix4()},uFusionWindow:{value:new THREE.Vector2()}};
  Object.assign(r,{volume:{...source,compartments:[{name:'brain',color:'#778899',mask:new Uint8Array(60)}]},bounds:[[-20,-30,-40],[80,90,100]],surfaces:new Map([['brain',mesh]]),materials:()=>[{uniforms}],requestRender(){},fusionTexture:{dispose(){}},renderer:{getContext:()=>({MAX_3D_TEXTURE_SIZE:1,getParameter:()=>2048})}});
  return {r,mesh,uniforms};
}
test('real renderer updates source material and all image uniforms without touching source data',()=>{
  const {r,mesh,uniforms}=renderer();const data=r.volume.mri.slice(),mask=r.volume.compartments[0].mask.slice();
  r.setWorkspaceSettings({layerOpacity:{brain:0},layerColor:{brain:'#ff3300'},clipping:{axis:'x',value:12,enabled:true},fusion:{seriesId:'series-b',opacity:.3}});
  assert.equal(mesh.material.opacity,0);assert.equal(mesh.material.color.getHexString(),'ff3300');assert.equal(uniforms.uLayerOpacities.value[0],0);
  assert.equal(mesh.material.clippingPlanes[0].distanceToPoint(new THREE.Vector3(13,0,0)),1);
  assert.equal(mesh.material.clippingPlanes[0].distanceToPoint(new THREE.Vector3(11,0,0)),-1);
  assert.deepEqual(uniforms.uClipPlane.value.toArray(),[1,0,0,-12]);assert.equal(uniforms.uFusionOpacity.value,.3);
  r.setWorkspaceSettings({clipping:{axis:'x',value:12,enabled:true,flipped:true}});
  assert.equal(mesh.material.clippingPlanes[0].distanceToPoint(new THREE.Vector3(11,0,0)),1);
  r.setWorkspaceSettings({});assert.equal(mesh.material.clippingPlanes.length,0);assert.equal(uniforms.uClipEnabled.value,0);
  assert.deepEqual(r.volume.mri,data);assert.deepEqual(r.volume.compartments[0].mask,mask);
});
test('real renderer drops old fusion before refusing a stale replacement',()=>{
  const {r,uniforms}=renderer();r.setRegisteredFusion(fusion());assert.equal(uniforms.uFusionActive.value,1);
  const wrong=fusion();wrong.referenceCaseHash='wrong';assert.throws(()=>r.setRegisteredFusion(wrong));assert.equal(uniforms.uFusionActive.value,0);
});
test('opaque anatomy occludes its own rear triangles and transparency restores without changing masks',()=>{
  const {r,mesh}=renderer(), mask=r.volume.compartments[0].mask.slice();
  r.setWorkspaceSettings({layerOpacity:{brain:1}});
  assert.equal(mesh.material.transparent,false);
  assert.equal(mesh.material.depthWrite,true);
  r.setWorkspaceSettings({layerOpacity:{brain:.2}});
  assert.equal(mesh.material.transparent,true);
  assert.equal(mesh.material.depthWrite,false);
  assert.deepEqual(r.volume.compartments[0].mask,mask);
});
test('smoothing toggles cached display geometry; cursor picking always uses the exact source surface',()=>{
  const {r,mesh}=renderer();
  const source=new THREE.BoxGeometry(2,2,2),smooth=new THREE.BoxGeometry(1.8,1.8,1.8);
  r.surfaceGeometry=new Map([['brain',{source,smooth}]]);
  r.panes={anatomy:{getBoundingClientRect:()=>({left:0,top:0,width:100,height:100})}};
  r.camera=new THREE.PerspectiveCamera();r.sourcePlane={visible:false};r.anatomy={visible:true};r.replayGroup={children:[]};
  let picked;
  r.pickRay={setFromCamera(){},intersectObjects(meshes){picked=meshes[0].geometry;return [{point:new THREE.Vector3(1,2,3)}];}};
  r.setSurfaceSmoothing(true);assert.equal(mesh.geometry,smooth);
  assert.deepEqual(r.pick(50,50),[1,2,3]);assert.equal(picked,source);assert.equal(mesh.geometry,smooth);
  r.setSurfaceSmoothing(false);assert.equal(mesh.geometry,source);
  r.setSurfaceSmoothing(true);assert.equal(mesh.geometry,smooth);
  r.pickRay.intersectObjects=()=>{throw new Error('raycast failed');};
  assert.throws(()=>r.pick(50,50),/raycast failed/);assert.equal(mesh.geometry,smooth);
});
test('saved camera restoration preserves orbit, target and up and cancels pending fit',()=>{
  const {r}=renderer();r.camera=new THREE.PerspectiveCamera(34,1,.1,1000);r.controls={target:new THREE.Vector3(),update(){}};r.pendingFit='anatomy';
  const state={position:[110,-220,130],target:[10,20,30],up:[0,0,1]};
  r.restoreCamera(state);assert.deepEqual(r.getCameraState(),state);assert.equal(r.pendingFit,null);assert.equal(r.cameraRestored,true);
  r.restoreCamera({position:[0,0,0],target:[0,0,0],up:[0,0,1]});assert.deepEqual(r.getCameraState(),state);
});
test('volume framebuffer restores viewport/clear state, bounds workload and retains the immutable source array',()=>{
  let target=null,restored=0;const renders=[],color=new THREE.Color('#05080d');let alpha=1;
  const renderer={extensions:{has:()=>true},getPixelRatio:()=>2,getClearColor:out=>out.copy(color),getClearAlpha:()=>alpha,
    setClearColor(c,a){color.set(c);alpha=a;},setRenderTarget(value){target=value;},clear(){},render(scene){renders.push({scene,target});}};
  const original=source.mri.slice(),r=new ScanVolumeRenderer(renderer,source,[0,100]);
  r.configure({enabled:true,preset:'mri',density:.4},'T1',null);
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera();
  r.render(scene,camera,2000,1000,false,()=>restored++);
  assert.equal(r.texture.image.data,source.mri);assert.deepEqual(source.mri,original);
  assert.equal(renders[0].target.width,1300);assert.equal(renders[0].target.height,650);
  assert.equal(renders[2].target,null);assert.equal(restored,1);assert.equal(alpha,1);
  assert.equal(color.getHexString(),'05080d');
  r.render(scene,camera,2000,1000,true,()=>restored++);assert.equal(r.sceneTarget.width,700);
  r.configure({enabled:false},'CT',null);const before=renders.length;
  r.render(scene,camera,2000,1000,false,()=>restored++);assert.equal(renders.length,before+1);assert.equal(renders.at(-1).scene,scene);
  let disposed=0;r.sceneTarget.addEventListener('dispose',()=>disposed++);r.volumeTarget.addEventListener('dispose',()=>disposed++);
  r.texture.addEventListener('dispose',()=>disposed++);r.transfer.addEventListener('dispose',()=>disposed++);r.dispose();assert.equal(disposed,4);
});
