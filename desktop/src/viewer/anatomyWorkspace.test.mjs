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
  const uniforms={uColors:{value:[new THREE.Color()]},uSliceContours:{value:0},uLayerOpacities:{value:new Float32Array(8)},uClipEnabled:{value:0},uClipPlane:{value:new THREE.Vector4()},uFusionOpacity:{value:.5},uFusionActive:{value:0},uFusion:{value:null},uFusionShape:{value:new THREE.Vector3()},uWorldToFusion:{value:new THREE.Matrix4()},uFusionWindow:{value:new THREE.Vector2()}};
  Object.assign(r,{beginInteraction(){},endInteraction(){},scanVolume:{configure(){},labels(){}},volume:{...source,compartments:[{name:'brain',color:'#778899',mask:new Uint8Array(60)}]},bounds:[[-20,-30,-40],[80,90,100]],surfaces:new Map([['brain',mesh]]),materials:()=>[{uniforms}],requestRender(){},fusionTexture:{dispose(){}},renderer:{getContext:()=>({MAX_3D_TEXTURE_SIZE:1,getParameter:()=>2048})}});
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
  assert.equal(renders.at(-1).target,null);assert.equal(restored,1);assert.equal(alpha,1);
  assert.equal(color.getHexString(),'05080d');
  const finalScene=r.sceneTarget,finalVolume=r.volumeTarget;
  let start=renders.length;r.render(scene,camera,2000,1000,true,()=>restored++);
  assert.equal(renders[start].target.width,360);assert.equal(renders[start].target.height,180);
  const dragScene=r.interactionTargets.scene,dragVolume=r.interactionTargets.volume;
  let resized=0;for(const target of [finalScene,finalVolume,dragScene,dragVolume])target.addEventListener('dispose',()=>resized++);
  start=renders.length;r.render(scene,camera,2000,1000,false,()=>restored++);assert.equal(renders[start].target,finalScene);
  start=renders.length;r.render(scene,camera,2000,1000,true,()=>restored++);assert.equal(renders[start].target,dragScene);
  assert.equal(r.material.uniforms.uScene.value,dragScene.texture);
  assert.equal(r.material.uniforms.uDepth.value,dragScene.depthTexture);
  assert.equal(r.blitMaterial.map,dragVolume.texture);
  assert.equal(resized,0,'quality changes must reuse the GPU attachments');
  r.configure({enabled:false},'CT',null);const before=renders.length;
  r.render(scene,camera,2000,1000,false,()=>restored++);assert.equal(renders.length,before+1);assert.equal(renders.at(-1).scene,scene);
  let disposed=0;r.sceneTarget.addEventListener('dispose',()=>disposed++);r.volumeTarget.addEventListener('dispose',()=>disposed++);
  r.texture.addEventListener('dispose',()=>disposed++);r.transfer.addEventListener('dispose',()=>disposed++);r.dispose();assert.equal(disposed,4);
  assert.equal(resized,4,'both quality levels release their attachments');
});

test('camera-only settings and cloned toolbar choices do not reapply layer materials',()=>{
  const {r,mesh,uniforms}=renderer();let configured=0,labels=0,rendered=0;
  r.scanVolume={configure(){configured++},labels(){labels++}};r.requestRender=()=>rendered++;
  const settings={layerOpacity:{brain:.2},clipping:{axis:'x',value:12,enabled:true}};
  r.setWorkspaceSettings(settings);const planes=mesh.material.clippingPlanes;
  r.setWorkspaceSettings({...structuredClone(settings),camera:{position:[110,-220,130],target:[10,20,30],up:[0,0,1]},layout:'mri-review',contrast:2});
  assert.equal(configured,1);assert.equal(labels,1);assert.equal(rendered,1);assert.equal(mesh.material.clippingPlanes,planes);
  settings.layerOpacity.brain=.8;settings.clipping.value=20;r.setWorkspaceSettings(settings);
  assert.equal(configured,2);assert.equal(mesh.material.opacity,.8);assert.equal(uniforms.uClipPlane.value.w,-20);
});

test('slice contour changes reach every image material and preserve source data and 3D appearance',()=>{
  const {r,mesh,uniforms}=renderer(),second={uniforms:THREE.UniformsUtils.clone(uniforms)};
  r.materials=()=>[{uniforms},second];let rendered=0;r.requestRender=()=>rendered++;
  const scalar=r.volume.mri.slice(),mask=r.volume.compartments[0].mask.slice();
  r.setWorkspaceSettings({});const opacity=mesh.material.opacity,layerOpacity=uniforms.uLayerOpacities.value[0];
  assert.equal(uniforms.uSliceContours.value,0,'existing saved views retain filled overlays');
  r.setWorkspaceSettings({sliceOverlay:'contour'});
  assert.equal(uniforms.uSliceContours.value,1);assert.equal(second.uniforms.uSliceContours.value,1);
  assert.equal(mesh.material.opacity,opacity);assert.equal(uniforms.uLayerOpacities.value[0],layerOpacity);
  assert.equal(rendered,2,'changing only the overlay must invalidate the image');
  r.setWorkspaceSettings({sliceOverlay:'fill'});
  assert.equal(uniforms.uSliceContours.value,0);assert.equal(second.uniforms.uSliceContours.value,0);
  assert.deepEqual(r.volume.mri,scalar);assert.deepEqual(r.volume.compartments[0].mask,mask);
});

test('unchanged cursor/layers/routes avoid redraw, and in-place visibility changes reach all views',()=>{
  const {r,mesh}=renderer();let labels=0,planeUpdates=0,rendered=0;
  const uniforms={uCursor:{value:new THREE.Vector3()},uVisibleBits:{value:0},uLayerEnabled:{value:new Float32Array(8)},uOverlay:{value:0}};
  Object.assign(r,{cursor:[1,2,3],visible:{},opacity:.4,routeSignature:'[]',mode:'anatomy',
    scanVolume:{labels(){labels++}},materials:()=>[{uniforms}],replayGroup:new THREE.Group(),cursorObject:new THREE.Object3D(),
    updateSourcePlane(){planeUpdates++},requestRender(){rendered++}});
  const visible={brain:true};r.update([1,2,3],visible,.4,[],'anatomy');
  assert.equal(labels,1);assert.equal(rendered,1);
  r.update([1,2,3],{...visible},.4,[],'anatomy');
  assert.equal(labels,1);assert.equal(rendered,1);assert.equal(planeUpdates,0);
  r.update([2,2,3],visible,.4,[],'anatomy');assert.equal(planeUpdates,1);assert.equal(labels,1);
  assert.deepEqual(r.cursorObject.position.toArray(),[2,2,3]);
  visible.brain=false;r.update([2,2,3],visible,.4,[],'anatomy');
  assert.equal(mesh.visible,false);assert.equal(uniforms.uLayerEnabled.value[0],0);assert.equal(labels,2);
  visible.brain=true;r.update([2,2,3],visible,.7,[],'anatomy');
  assert.equal(mesh.visible,true);assert.equal(uniforms.uLayerEnabled.value[0],1);assert.equal(uniforms.uOverlay.value,.7);
});

test('persisted camera echoes skip control updates after projection bounds settle',()=>{
  const {r}=renderer();let updated=0,rendered=0;
  r.camera=new THREE.PerspectiveCamera(34,1,.1,1000);
  r.controls={target:new THREE.Vector3(),update(){updated++}};r.requestRender=()=>rendered++;
  const state={position:[110,-220,130],target:[10,20,30],up:[0,0,1]};
  r.restoreCamera(state);r.restoreCamera(structuredClone(state));
  assert.equal(updated,1);assert.equal(rendered,1);assert.deepEqual(r.getCameraState(),state);
  state.position[0]+=1;r.restoreCamera(state);assert.equal(updated,2);
});

test('interaction never resizes the shared drawing buffer',()=>{
  const previousWindow=globalThis.window;globalThis.window={devicePixelRatio:2};
  try {
    const {r}=renderer();let pixelRatio=2;const changes=[];
    const empty={getBoundingClientRect:()=>({left:0,top:0,width:0,height:0})};
    Object.assign(r,{disposed:false,width:100,height:100,interacting:false,refinementStarted:0,budget:{longEdge:360},
      container:{getBoundingClientRect:()=>({left:0,top:0,width:100,height:100})},
      panes:{anatomy:empty,axial:empty,coronal:empty,sagittal:empty},canvas:{dispatchEvent(){}},scanVolume:{active:false},
      renderer:{info:{memory:{},render:{calls:0}},getContext:()=>({fenceSync:()=>null}),getPixelRatio:()=>pixelRatio,setPixelRatio(value){pixelRatio=value;changes.push(value)},setScissorTest(){},clear(){}}});
    r.render();r.render();assert.deepEqual(changes,[]);
    r.interacting=true;r.render();r.render();assert.deepEqual(changes,[]);
    r.interacting=false;r.render();r.render();assert.deepEqual(changes,[]);
  } finally {if(previousWindow===undefined)delete globalThis.window;else globalThis.window=previousWindow;}
});

test('volume shader receives only visible indices, including layers beyond bit 31 and zero/opaque isolation',()=>{
  const layers=Array.from({length:128},(_,i)=>({name:`layer-${i}`,color:'#778899',mask:new Uint8Array(60)}));
  const renderer={extensions:{has:()=>true}},r=new ScanVolumeRenderer(renderer,{...source,compartments:layers},[0,100],layers.map((_,i)=>[i%4,i%255+1]));
  const visible=Object.fromEntries(layers.map((l,i)=>[l.name,[0,64,127].includes(i)]));
  r.labels(null,visible,{layerOpacity:{'layer-64':0,'layer-127':1}},[]);
  assert.equal(r.material.uniforms.uActiveLayerCount.value,3);
  assert.deepEqual([...r.material.uniforms.uActiveLayers.value.slice(0,3)],[0,64,127]);
  r.labels(null,Object.fromEntries(layers.map(l=>[l.name,false])),{},[]);
  assert.equal(r.material.uniforms.uActiveLayerCount.value,0);
  r.dispose();
});

test('incremental layer install retains source texture, existing surfaces and camera, and rejects a stale source',()=>{
  const {r,mesh,uniforms}=renderer();
  r.volume.displayIdentity='native/revision/1';r.workspaceSettings={};
  const original=r.volume,oldLayer=original.compartments[0];
  r.affine=identity;r.extraClipPlanes=[];r.visible={brain:false,tumor:true};r.anatomy=new THREE.Group();r.anatomy.add(mesh);
  r.surfaceGeometry=new Map();r.camera=new THREE.PerspectiveCamera();const camera=r.camera;
  let disposed=0,layerUpdates=0;r.labelTexture={dispose(){disposed++}};
  r.mriTexture={unchanged:true};const image=r.mriTexture;
  r.scanVolume.setLayers=()=>layerUpdates++;
  Object.assign(uniforms,{uLabels:{value:null},uLayerCount:{value:1},uLayerCodes:{value:Array.from({length:128},()=>new THREE.Vector2())},uLayerEnabled:{value:new Float32Array(128)}});
  uniforms.uColors.value=Array.from({length:128},()=>new THREE.Color());
  const positions=new Float32Array([0,0,0,1,0,0,0,1,0]),normals=new Float32Array([0,0,1,0,0,1,0,0,1]);
  const layer={name:'tumor',mask:new Uint8Array(60),color:'#ff00ff',surface:{source:positions,positions:positions.slice(),sourceNormals:normals,normals:normals.slice(),indices:new Uint32Array([0,1,2])}};
  const next={...original,compartments:[layer,oldLayer],packedLabels:{data:new Uint8Array(240),codes:Array.from({length:128},()=>[0,0])}};
  r.setAnatomyLayers(next);assert.equal(r.volume,next);assert.equal(r.camera,camera);assert.equal(r.mriTexture,image);
  assert.equal(r.surfaces.get('brain'),mesh);assert(r.surfaces.has('tumor'));assert.equal(disposed,1);assert.equal(layerUpdates,1);
  assert.equal(uniforms.uLayerEnabled.value[0],1);assert.equal(uniforms.uLayerEnabled.value[1],0);
  r.setAnatomyLayers(next);assert.equal(layerUpdates,1);
  const installed=r.labelTexture;assert.throws(()=>r.setAnatomyLayers({...next,displayIdentity:'changed'}),/source changed/);assert.equal(r.labelTexture,installed);
});

test('shared scalar texture is not disposed by the volume pass',()=>{
  const texture=new THREE.Data3DTexture(source.mri,5,4,3);let disposed=0;texture.addEventListener('dispose',()=>disposed++);
  const r=new ScanVolumeRenderer({extensions:{has:()=>true}},source,[0,1],undefined,texture);
  assert.equal(r.texture,texture);r.dispose();assert.equal(disposed,0);texture.dispose();assert.equal(disposed,1);
});

test('GPU backpressure coalesces input and submits the latest pending frame only after completion',()=>{
  const previousWindow=globalThis.window,previousRaf=globalThis.requestAnimationFrame;
  const callbacks=new Map();let id=0,ready=false,deleted=0;
  globalThis.window={devicePixelRatio:1};globalThis.requestAnimationFrame=callback=>{callbacks.set(++id,callback);return id};
  try{
    const {r}=renderer(),empty={getBoundingClientRect:()=>({left:0,top:0,width:0,height:0})};
    const gl={SYNC_GPU_COMMANDS_COMPLETE:1,TIMEOUT_EXPIRED:2,WAIT_FAILED:3,fenceSync:()=>({}),flush(){},clientWaitSync:()=>ready?4:2,deleteSync(){deleted++}};
    Object.assign(r,{requestRender:VolumeRenderer.prototype.requestRender,disposed:false,frame:0,renderSerial:0,width:100,height:100,sliceRevision:0,interacting:false,refinementStarted:0,budget:{longEdge:360},gpuSync:null,
      container:{getBoundingClientRect:()=>({left:0,top:0,width:100,height:100})},panes:{anatomy:empty,axial:empty,coronal:empty,sagittal:empty},canvas:{dispatchEvent(){}},scanVolume:{active:false},controls:{update(){}},
      renderer:{info:{memory:{},render:{calls:0}},getContext:()=>gl,getPixelRatio:()=>1,setScissorTest(){},clear(){}}});
    r.render();for(let i=0;i<100;i++)r.requestRender(false);assert.equal(r.renderSerial,1);assert.equal(callbacks.size,1);
    const tick=()=>{const [key,callback]=callbacks.entries().next().value;callbacks.delete(key);callback()};
    tick();assert.equal(r.renderSerial,1);ready=true;tick();assert.equal(r.renderSerial,2);assert.equal(deleted,1);
    tick();assert.equal(r.gpuSync,null);assert.equal(callbacks.size,0);
  }finally{globalThis.window=previousWindow;globalThis.requestAnimationFrame=previousRaf;}
});
