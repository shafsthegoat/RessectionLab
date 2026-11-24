import assert from 'node:assert/strict';
import {before,after,test} from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import * as THREE from 'three';
import {checkedFusionTransform} from './registeredFusion.ts';
import {transformPoint,volumeBounds} from './coordinates.ts';
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
test('camera rejects collapsed or collinear views and out-of-bounds cutaways stay inactive',()=>{
  assert.equal(validCameraState({position:[0,0,0],target:[0,0,0],up:[0,0,1]}),false);
  assert.equal(validCameraState({position:[0,0,50],target:[0,0,0],up:[0,0,1]}),false);
  assert.deepEqual(clippingForBounds({axis:'x',value:999,enabled:true},[[-20,-30,-40],[80,90,100]]),{axis:'x',value:30,enabled:false,flipped:false});
});

let server,cache,VolumeRenderer,ScanVolumeRenderer,ScanVolumeExportTargets,exportHelpers;
before(async()=>{
  cache=await fs.mkdtemp(path.join(os.tmpdir(),'anatomy-workspace-'));
  server=await createServer({root:fileURLToPath(new URL('../../',import.meta.url)),cacheDir:cache,logLevel:'error',optimizeDeps:{noDiscovery:true,include:[]},server:{middlewareMode:true,hmr:false,ws:false},appType:'custom'});
  VolumeRenderer=(await server.ssrLoadModule('/src/viewer/VolumeRenderer.ts')).VolumeRenderer;
  ({ScanVolumeRenderer,ScanVolumeExportTargets}=await server.ssrLoadModule('/src/viewer/ScanVolumeRenderer.ts'));
  exportHelpers=await server.ssrLoadModule('/src/viewer/anatomyExport.ts');
});
after(async()=>{await server?.close();if(cache)await fs.rm(cache,{recursive:true,force:true,maxRetries:3,retryDelay:50});});
function renderer(){
  const r=Object.create(VolumeRenderer.prototype),mesh=new THREE.Mesh(new THREE.BoxGeometry(),new THREE.MeshStandardMaterial());
  r.vesselDepthSeparation={active:false};
  const uniforms={uColors:{value:[new THREE.Color()]},uSliceContours:{value:0},uLayerOpacities:{value:new Float32Array(8)},uClipEnabled:{value:0},uClipPlane:{value:new THREE.Vector4()},uFusionOpacity:{value:.5},uFusionActive:{value:0},uFusion:{value:null},uFusionShape:{value:new THREE.Vector3()},uWorldToFusion:{value:new THREE.Matrix4()},uFusionWindow:{value:new THREE.Vector2()}};
  Object.assign(r,{camera:new THREE.PerspectiveCamera(34,1,.1,1000),controls:{target:new THREE.Vector3(),update(){}},cursor:[0,0,0],beginInteraction(){},endInteraction(){},scanVolume:{configure(){},labels(){}},volume:{...source,compartments:[{name:'brain',color:'#778899',mask:new Uint8Array(60)}]},bounds:[[-20,-30,-40],[80,90,100]],surfaces:new Map([['brain',mesh]]),materials:()=>[{uniforms}],requestRender(){},fusionTexture:{dispose(){}},renderer:{getContext:()=>({MAX_3D_TEXTURE_SIZE:1,getParameter:()=>2048})}});
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
test('shared cutaway reaches the field edge without being shifted onto the selected scan',()=>{
  const {r,uniforms}=renderer();
  r.sourceClipBounds=volumeBounds([10,10,10],identity,true);
  r.patientClipBounds=volumeBounds([30,10,10],identity,true);
  for(const value of [20,29.5]){
    r.setWorkspaceSettings({presentation:'segmentation',clipping:{axis:'x',value,enabled:true}});
    assert.deepEqual(uniforms.uClipPlane.value.toArray(),[1,0,0,-value]);
    r.setWorkspaceSettings({presentation:'scan',clipping:{axis:'x',value,enabled:true}});
    assert.equal(uniforms.uClipEnabled.value,0);
  }
  r.setWorkspaceSettings({presentation:'scan',clipping:{axis:'x',value:9.5,enabled:true}});
  assert.deepEqual(uniforms.uClipPlane.value.toArray(),[1,0,0,-9.5]);
});
test('linked presentations share physical cuts and keep the selected CTA texture and transfer',()=>{
  const {r}=renderer();
  const affine=[[2,0,0,8],[0,3,0,1],[0,0,4,-2],[0,0,0,1]];
  const mri=new Float32Array(60).fill(340);
  const patientClipBounds=[[-20,-40,-50],[70,80,90]];
  const volume={...source,affine,mri,headSupport:{data:new Uint8Array(60).fill(1),sha256:'support'},
    patientContext:{sources:[],boundsRasMm:patientClipBounds}};
  r.sourceClipBounds=volumeBounds(source.shape,affine,true);
  r.patientClipBounds=patientClipBounds;
  r.volumeModality='CTA';
  const scan=new ScanVolumeRenderer({extensions:{has:()=>true}},volume,[0,1200]);
  r.scanVolume=scan;
  try{
    const u=scan.material.uniforms;
    const voxel=new THREE.Vector3(10,7,2).applyMatrix4(u.uWorldToVoxel.value);
    assert.ok(voxel.distanceTo(new THREE.Vector3(1,2,1))<1e-10);
    assert.equal(scan.spacing,2);
    for(const presentation of ['scan','segmentation','vascular']){
      r.setWorkspaceSettings({presentation,clipping:{axis:'x',value:40,enabled:true}});
      assert.equal(u.uVolume.value.image.data,mri);
      assert.deepEqual(u.uShape.value.toArray(),source.shape);
      assert.deepEqual(u.uRange.value.toArray(),[-1000,3000]);
      assert.equal(u.uCt.value,true);
      assert.equal(u.uUseHeadSupport.value,true);
      assert.equal(u.uClipEnabled.value,presentation!=='scan');
      if(presentation!=='scan')assert.deepEqual(u.uClipPlane.value.toArray(),[1,0,0,-40]);
    }
  }finally{scan.dispose();}
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
test('an explicit section image writes its physical depth and an uncut reference plane remains translucent',()=>{
  const {r,uniforms}=renderer();
  r.sourcePlane={material:new THREE.ShaderMaterial({depthWrite:false})};
  uniforms.uExtraClipCount={value:0};uniforms.uExtraClips={value:Array.from({length:7},()=>new THREE.Vector4())};
  const source=r.volume.mri.slice();
  r.setWorkspaceSettings({showPlane:true,clipping:{axis:'z',value:5,enabled:true},clipBox:{enabled:true,min:[0,0,0],max:[10,20,30]},obliqueClip:{enabled:true,normal:[1,1,0],offset:0}});
  assert.equal(r.sourcePlane.material.depthWrite,true);
  assert.equal(uniforms.uExtraClipCount.value,7);
  assert.ok(uniforms.uExtraClips.value[0].equals(new THREE.Vector4(1,0,0,0)));
  assert.ok(uniforms.uExtraClips.value[1].equals(new THREE.Vector4(-1,0,0,10)));
  r.setWorkspaceSettings({showPlane:true,clipping:{axis:'z',value:5,enabled:false}});
  assert.equal(r.sourcePlane.material.depthWrite,false);
  assert.equal(uniforms.uExtraClipCount.value,0);
  assert.deepEqual(r.volume.mri,source);
  r.sourcePlane.material.dispose();
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
  assert.equal(renders[0].target.width,1024);assert.equal(renders[0].target.height,512);
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
  r.render(scene,camera,2000,1000,false,()=>restored++,360,true);
  assert.equal(r.material.uniforms.uDrawScalar.value,false);
  assert.equal(renders.at(-1).scene,r.blitScene);
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

test('acquired Scan bounds its settled GPU draw while retaining the source grid and physical opacity step',()=>{
  const color=new THREE.Color('#000000');
  const renderer={extensions:{has:()=>true},getPixelRatio:()=>2,getClearColor:out=>out.copy(color),getClearAlpha:()=>1,
    setClearColor(){},setRenderTarget(){},clear(){},render(){}};
  const scan=new ScanVolumeRenderer(renderer,source,[0,100]);
  try{
    scan.labels(null,{}, {presentation:'scan'},[]);
    scan.render(new THREE.Scene(),new THREE.PerspectiveCamera(),2000,1000,false,()=>{});
    assert.equal(scan.volumeTarget.width,1024);assert.equal(scan.volumeTarget.height,512);
    assert.equal(scan.material.uniforms.uStep.value,1,'Step is one acquired physical spacing, not a changed source grid');
    assert.equal(scan.texture.image.data,source.mri);
    assert.deepEqual([scan.texture.image.depth,scan.texture.image.height,scan.texture.image.width],source.shape);
  }finally{scan.dispose();}
});

test('transparent CTA feathers only optical coverage while exact masks and scan support stay independent',()=>{
  const renderer={extensions:{has:()=>true},getPixelRatio:()=>1,getClearColor:out=>out.set(0),getClearAlpha:()=>1,
    setClearColor(){},setRenderTarget(){},clear(){},render(){}};
  const mask=new Uint8Array(60);mask[31]=1;
  const original=mask.slice(),layer={name:'bone',kind:'bone',color:'#ded0b6',mask};
  const volume={...source,affine:[[.4,0,0,0],[0,.7,0,0],[0,0,3,0],[0,0,0,1]],compartments:[layer]};
  const scan=new ScanVolumeRenderer(renderer,volume,[0,1000]);
  const supported=new ScanVolumeRenderer(renderer,{...volume,headSupport:{data:mask,sha256:'source-support'}},[0,1000]);
  const render=pass=>pass.render(new THREE.Scene(),new THREE.PerspectiveCamera(),640,480,false,()=>{});
  const settings={presentation:'vascular',skullMode:'transparent',layerOpacity:{bone:.04}};
  try{
    scan.configure({shadows:false},'CTA',null);
    scan.labels(null,{},settings,[]);render(scan);
    const uniforms=scan.material.uniforms,aux=uniforms.uAuxiliary.value;
    assert.equal(uniforms.uContextCoverageLayer.value,0);
    assert.equal(aux.magFilter,THREE.LinearFilter);
    assert.ok(aux.image.data[31]>0&&aux.image.data[31]<255);
    assert.equal(aux.image.data[30],aux.image.data[32]);
    assert.ok(aux.image.data[30]>0);
    assert.deepEqual(mask,original);
    assert.equal(scan.texture.image.data,source.mri);
    scan.configure({shadows:false},'CTA',new THREE.Plane(new THREE.Vector3(0,0,1),0));
    scan.labels(null,{}, {...settings,skullMode:'cutaway'},[]);render(scan);
    assert.equal(uniforms.uSkullMode.value,1);
    assert.equal(uniforms.uAcquiredBoneCut.value,true);
    assert.equal(uniforms.uAuxiliary.value,aux,'a same-grid solid cut reuses the continuous coverage texture');
    assert.equal(uniforms.uOpticalCache.value,false,'solid extinction still uses the direct clip-aware shadow probes');
    for(const [visible,view] of [ [{bone:false},settings], [{},{...settings,presentation:'scan'}], [{},{...settings,skullMode:'exterior'}],
      [{},{...settings,skullMode:'cutaway',presentation:'scan'}] ]){
      scan.labels(null,visible,view,[]);render(scan);
      assert.equal(uniforms.uContextCoverageLayer.value,-1);
      assert.equal(uniforms.uAuxiliary.value,null);
      assert.equal(uniforms.uAcquiredBoneCut.value,false);
    }
    scan.labels(null,{},settings,[]);render(scan);assert.equal(uniforms.uAuxiliary.value,aux);
    scan.configure({shadows:false,isolateLabels:true},'CTA',null);render(scan);
    assert.equal(uniforms.uContextCoverageLayer.value,-1);
    supported.configure({shadows:false},'CTA',null);
    supported.labels(null,{},settings,[]);render(supported);
    assert.equal(supported.material.uniforms.uAuxiliary.value,supported.supportTexture);
    assert.equal(supported.material.uniforms.uContextCoverageLayer.value,-1);
  }finally{scan.dispose();supported.dispose();}
});

test('CTA uses only the atlas matching its active optical mask and shares it across opacity changes',()=>{
  const renderer={extensions:{has:()=>true},getPixelRatio:()=>1,getClearColor:out=>out.set(0),getClearAlpha:()=>1,
    setClearColor(){},setRenderTarget(){},clear(){},render(){}};
  const makeBone=index=>({name:'bone',kind:'bone',color:'#ded0b6',mask:Uint8Array.from({length:60},(_,i)=>Number(i===index))});
  const scan=new ScanVolumeRenderer(renderer,{...source,compartments:[makeBone(30)]},[0,1000]);
  const render=()=>scan.render(new THREE.Scene(),new THREE.PerspectiveCamera(),640,480,false,()=>{});
  const settings={presentation:'vascular',skullMode:'transparent',layerOpacity:{bone:.12}};
  try{
    scan.configure({shadows:true},'CTA',null);
    scan.labels(null,{},settings,[]);render();
    const u=scan.material.uniforms,context=u.uTransfer.value;
    assert.ok(u.uOpticalCache.value&&context.image.height>1);
    assert.ok(u.uOpticalDepthRow.value>0&&u.uOpticalDepthRow.value<context.image.height);
    assert.equal(scan.material.defines.CONTEXT_SHADOWS,1);
    assert.equal(scan.material.defines.DIRECTIONAL_OPTICS,1);
    scan.labels(null,{}, {...settings,layerOpacity:{bone:.25}},[]);render();
    assert.equal(u.uTransfer.value,context);
    scan.labels(null,{},settings,[new THREE.Plane(new THREE.Vector3(0,0,1),0)]);render();
    assert.equal(scan.material.defines.CONTEXT_SHADOWS,2);
    assert.equal(scan.material.defines.DIRECTIONAL_OPTICS,0);
    assert.ok(u.uExtraOpticalWidths.value[0]>0);
    assert.equal(u.uTransfer.value,context);
    scan.labels(null,{bone:false},settings,[]);render();
    assert.equal(u.uOpticalCache.value,false);
    assert.equal(scan.material.defines.CONTEXT_SHADOWS,0);
    assert.equal(scan.material.defines.DIRECTIONAL_OPTICS,0);
    assert.equal(u.uTransfer.value,context);
    scan.labels(null,{}, {...settings,presentation:'scan'},[]);render();
    assert.equal(u.uOpticalCache.value,true);
    assert.notEqual(u.uTransfer.value,context);
    scan.labels(null,{},settings,[]);render();
    const current=u.uTransfer.value;
    scan.setLayers([makeBone(31)],[[0,1]]);scan.labels(null,{},settings,[]);render();
    assert.equal(u.uOpticalCache.value,true);
    assert.notEqual(u.uTransfer.value,current);
    scan.configure({shadows:true,isolateLabels:true},'CTA',null);render();
    assert.equal(u.uOpticalCache.value,false);
  }finally{scan.dispose();}
});

test('CTA exterior uses one lazy scalar derivative while raw views and bone coverage keep their textures',()=>{
  const renderer={extensions:{has:()=>true},getPixelRatio:()=>1,getClearColor:out=>out.set(0),getClearAlpha:()=>1,
    setClearColor(){},setRenderTarget(){},clear(){},render(){}};
  const bone={name:'bone',kind:'bone',color:'#ded0b6',mask:Uint8Array.from({length:60},(_,i)=>Number(i===30))};
  const ctExterior={offset:[1,2,1],shape:[2,1,2],data:Uint8Array.of(1,0,0,1),sha256:'native-crop'};
  const signal=new Float32Array(60).fill(700);
  const scan=new ScanVolumeRenderer(renderer,{...source,mri:signal,compartments:[bone],ctExterior},[0,1000]);
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(),plane=new THREE.Plane(new THREE.Vector3(0,0,1),0);
  const settings={presentation:'vascular',skullMode:'transparent',layerOpacity:{bone:.12}};
  const render=(options={},view=settings,clip=null,extra=[],modality='CTA')=>{
    scan.configure(options,modality,clip);
    scan.labels(null,{},view,extra);
    return scan.render(scene,camera,640,480,false,()=>{});
  };
  try{
    const u=scan.material.uniforms;
    assert.equal(render({}, {...settings,presentation:'scan'}).ctExteriorDisplay,false);
    assert.deepEqual(scan.estimatedCtExteriorBytes,{cpuBytes:0,gpuTextureBytes:0});
    assert.equal(render({window:[-1000,3000]}).ctExteriorDisplay,false);
    assert.equal(u.uVolume.value,scan.texture);
    assert.equal(scan.exteriorVolume,undefined);

    const qualified=render(),derivative=u.uVolume.value,filteredAtlas=u.uTransfer.value,coverage=u.uAuxiliary.value;
    assert.equal(qualified.ctExteriorDisplay,true);
    assert.equal(scan.material.defines.CONTEXT_SHADOWS,1);
    assert.equal(scan.material.defines.DIRECTIONAL_OPTICS,1);
    assert.notEqual(derivative,scan.texture);
    const expected=signal.slice();
    expected[31]=expected[52]=-1024;
    assert.deepEqual(derivative.image.data,expected);
    assert.equal(scan.opticalAtlasSelection.source.signal,derivative.image.data);
    assert.equal(scan.texture.image.data,signal);
    assert.equal(coverage.format,THREE.RedFormat);
    assert.equal(coverage.image.data.length,60);
    assert.equal(coverage.image.data,scan.opticalAtlasSelection.context.coverage,'bone has one R8 grid for the eye and shadow atlas');
    assert.deepEqual(qualified.ctExteriorBytes,{cpuBytes:240,gpuTextureBytes:240});

    const rawFrame=render({showCtExteriorSignal:true});
    const rawAtlas=u.uTransfer.value;
    assert.equal(rawFrame.ctExteriorDisplay,false);
    assert.equal(rawFrame.samples,1,'a mode change cannot accumulate with a filtered frame');
    assert.equal(u.uVolume.value,scan.texture);
    assert.equal(u.uAuxiliary.value,coverage);
    assert.equal(scan.opticalAtlasSelection.source.signal,signal);
    assert.notDeepEqual(rawAtlas.image.data,filteredAtlas.image.data);
    for(const [options,view,clip,extra,modality] of [
      [{},settings,plane,[]], [{},settings,null,[plane]], [{},{...settings,presentation:'scan'},null,[]],
      [{isolateLabels:true},settings,null,[]], [{preset:'mri'},settings,null,[]],
      [{points:[[0,1,1,1,.2],[1,1,1,1,1]]},settings,null,[]], [{},settings,null,[],'T1'],
    ]){
      assert.equal(render(options,view,clip,extra,modality).ctExteriorDisplay,false);
      assert.equal(u.uVolume.value,scan.texture);
    }
    assert.equal(render({}, {...settings,presentation:'segmentation'}).ctExteriorDisplay,true);
    assert.equal(u.uVolume.value,derivative);
    assert.deepEqual(u.uTransfer.value.image.data,filteredAtlas.image.data);
    assert.deepEqual(ctExterior.data,Uint8Array.of(1,0,0,1));
    assert.equal(signal.every(value=>value===700),true);
  }finally{scan.dispose();}
});

test('native CT Scan can omit the 3D bone overlay while slice membership and the acquired texture remain available',()=>{
  const renderer={extensions:{has:()=>true},getPixelRatio:()=>1,getClearColor:out=>out.set(0),getClearAlpha:()=>1,
    setClearColor(){},setRenderTarget(){},clear(){},render(){}};
  const bone={name:'bone',kind:'bone',color:'#ded0b6',mask:new Uint8Array(60).fill(1)};
  const other={...bone,name:'landmark',kind:'context'};
  const volume={...source,displayIdentity:'native-ct',compartments:[bone,other]};
  const scan=new ScanVolumeRenderer(renderer,volume,[0,1000]);
  const r=Object.assign(Object.create(VolumeRenderer.prototype),{volume,volumeModality:'CTA',visible:{bone:true,landmark:true}});
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera();
  const render=(options={},presentation='scan',modality='CTA')=>{
    const settings={presentation,skullMode:'exterior',volumeRendering:options};
    r.workspaceSettings=settings;r.volumeModality=modality;
    scan.configure(options,modality,null);
    scan.labels(null,r.visible,settings,[]);
    scan.render(scene,camera,640,480,false,()=>{});
  };
  const u=scan.material.uniforms;
  const acquired={showCtBoneOverlay:false};
  try{
    render(acquired);
    assert.equal(r.surfaceVisible('bone'),false);
    assert.equal(r.surfaceVisible('landmark'),true);
    assert.equal(u.uLayerEnabled.value[0],1);
    assert.equal(u.uAcquiredCtBone.value,true);
    assert.equal(u.uOpticalCache.value,true);
    assert.equal(u.uVolume.value,scan.texture);
    assert.equal(scan.opticalAtlasSelection.source.signal,volume.mri);
    const acquiredAtlas=u.uTransfer.value;
    render({showCtBoneOverlay:true});
    assert.equal(u.uTransfer.value,acquiredAtlas,'the mesh toggle shares the acquired CT atlas');
    assert.equal(u.uOpticalCache.value,true);
    render();
    assert.equal(u.uOpticalCache.value,false,'legacy opaque modeled bone bypasses the acquired CT atlas');
    for(const [options,presentation,modality] of [
      [{}], [{showCtBoneOverlay:true}], [{...acquired,isolateLabels:true}],
      [{...acquired,enabled:false}], [{...acquired,density:0}],
      [acquired,'segmentation'], [acquired,'scan','T1'],
    ]){
      render(options,presentation,modality);
      assert.equal(r.surfaceVisible('bone'),true);
    }
  }finally{scan.dispose();}
});

test('an early raw export cancels a CTA atlas prepared from the exterior display derivative',()=>{
  const renderer={extensions:{has:()=>true},getPixelRatio:()=>1,getClearColor:out=>out.set(0),getClearAlpha:()=>1,
    setClearColor(){},setRenderTarget(){},clear(){},render(){}};
  let publications=0;
  const ctExterior={offset:[2,0,1],shape:[1,2,3],data:Uint8Array.of(1,0,1,0,0,1),sha256:'other-crop'};
  const scan=new ScanVolumeRenderer(renderer,{...source,ctExterior},[0,1000],undefined,undefined,()=>{publications++;});
  const targets=new ScanVolumeExportTargets();
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera();
  try{
    scan.configure({},'CTA',null);
    scan.labels(null,{}, {presentation:'segmentation'},[]);
    const live=scan.render(scene,camera,640,480,false,()=>{});
    assert.equal(live.opticsPending,true);
    assert.equal(live.ctExteriorDisplay,true);
    const filtered=scan.opticalPreparation;
    const u=scan.material.uniforms;
    assert.equal(u.uAuxiliary.value,null,'the derivative uses the existing scalar slot');

    scan.configure({showCtExteriorSignal:true},'CTA',null);
    assert.equal(scan.opticsPending,false,'the superseded filtered generation is canceled before export');
    scan.renderExportSample(scene,camera,targets,640,480,0,0);
    assert.equal(scan.opticsPending,false);
    assert.equal(u.uVolume.value,scan.texture);
    assert.equal(u.uAuxiliary.value,null);
    assert.equal(u.uOpticalCache.value,true);
    assert.notEqual(scan.opticalAtlasSelection.source,filtered.selection.source);
    assert.equal(publications,0);
  }finally{targets.dispose();scan.dispose();}
});

test('acquired Scan export uses independent targets and rebinds the regular quality after capture',()=>{
  const renderer={extensions:{has:()=>true},getPixelRatio:()=>2,getClearColor:out=>out.set(0),getClearAlpha:()=>1,
    setClearColor(){},setRenderTarget(){},clear(){},render(){}};
  const scan=new ScanVolumeRenderer(renderer,source,[0,100]),targets=new ScanVolumeExportTargets();
  try{
    const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera();
    scan.configure({enabled:true},'T1',null);
    scan.labels(null,{}, {presentation:'scan'},[]);
    scan.render(scene,camera,2000,1000,false,()=>{});
    const live=scan.volumeTarget;
    let resized=0;
    live.addEventListener('dispose',()=>resized++);
    scan.renderExportSample(scene,camera,targets,2048,640,0,384);
    const last=scan.renderExportSample(scene,camera,targets,2048,640,31,384);
    assert.equal(last,targets.accumulation[1]);
    assert.equal(scan.accumulateMaterial.uniforms.uWeight.value,1/32);
    assert.equal(scan.material.uniforms.uStep.value,.5);
    assert.equal(scan.material.uniforms.uSampleIndex.value,31);
    assert.equal(scan.material.uniforms.uSampleOrigin.value.y,384);
    assert.equal(scan.material.uniforms.uExport.value,true);
    assert.equal(live.width,1024);
    assert.equal(scan.samples,1);
    assert.equal(resized,0);
    scan.invalidate();
    scan.render(scene,camera,2000,1000,false,()=>{});
    assert.equal(scan.material.uniforms.uStep.value,1);
    assert.equal(scan.material.uniforms.uSampleOrigin.value.y,0);
    assert.equal(scan.material.uniforms.uExport.value,false);
    assert.equal(scan.material.uniforms.uScene.value,scan.sceneTarget.texture);
    assert.equal(resized,0);
  }finally{targets.dispose();scan.dispose();}
});

test('Scan publishes only the current optical cache and completes it before an early export',async()=>{
  const renderer={extensions:{has:()=>true},getPixelRatio:()=>1,getClearColor:out=>out.set(0),getClearAlpha:()=>1,
    setClearColor(){},setRenderTarget(){},clear(){},render(){}};
  let publications=0;
  let resolve;
  const ready=new Promise(onReady=>{resolve=onReady;});
  const scan=new ScanVolumeRenderer(renderer,source,[0,100],undefined,undefined,()=>{publications++;resolve();});
  const targets=new ScanVolumeExportTargets();
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera();
  try{
    scan.labels(null,{}, {presentation:'scan'},[]);
    scan.configure({enabled:true},'T1',null);
    const first=scan.render(scene,camera,640,480,false,()=>{});
    assert.equal(first.opticsPending,true);
    assert.equal(scan.transfer.image.height,1);
    scan.configure({enabled:true},'CTA',null);
    assert.equal(scan.opticsPending,false);
    scan.render(scene,camera,640,480,false,()=>{});
    await ready;
    assert.equal(publications,1);
    assert.equal(scan.opticsPending,false);
    assert.ok(scan.transfer.image.height>1);
    assert.deepEqual(scan.material.uniforms.uRange.value.toArray(),[-1000,3000]);

    scan.configure({enabled:true},'T1',null);
    scan.render(scene,camera,640,480,false,()=>{});
    assert.equal(scan.opticsPending,true);
    scan.renderExportSample(scene,camera,targets,640,480,0,0);
    assert.equal(scan.opticsPending,false);
    assert.equal(scan.material.uniforms.uOpticalCache.value,true);
    assert.equal(publications,1);
  }finally{targets.dispose();scan.dispose();}
});

test('PNG bands preserve the full projection and pack flipped WebGL rows without guard pixels',()=>{
  const {width,height,bands}=exportHelpers.anatomyExportBands(1,2048);
  assert.equal(width,2048);assert.equal(height,2048);assert.ok(bands.length>1);
  assert.equal(bands.reduce((sum,band)=>sum+band.rows,0),height);
  const camera=new THREE.PerspectiveCamera(34,width/height,.1,1000);
  const tile=camera.clone(),x=width/3;
  bands.forEach((band,i)=>{
    assert.ok(width*band.height*128<=160*1024**2);
    if(i>0)assert.equal(band.row,bands[i-1].row+bands[i-1].rows);
    const y=band.row+.5;
    tile.setViewOffset(width,height,0,band.top,width,band.height);
    const expected=new THREE.Vector3(x/width*2-1,1-y/height*2,.5).unproject(camera);
    const actual=new THREE.Vector3(x/width*2-1,1-(y-band.top)/band.height*2,.5).unproject(tile);
    assert.ok(expected.distanceTo(actual)<1e-12);
  });
  const pixelRows=Uint8Array.from([51,41,31,21,11].flatMap(value=>[value,0,0,255]));
  const img={width:1,data:new Uint8ClampedArray(20)};
  exportHelpers.copyAnatomyExportBand(pixelRows,img,{row:1,rows:3,top:0,height:5});
  assert.deepEqual([...img.data],[0,0,0,0,21,0,0,255,31,0,0,255,41,0,0,255,0,0,0,0]);
});

test('a requested frame cancels an export waiting for the live GPU and releases its renderer lock',async()=>{
  const r=Object.create(VolumeRenderer.prototype);
  const bounds={width:800,height:640,left:0,top:0};
  let invalidations=0;
  Object.assign(r,{container:{getBoundingClientRect:()=>bounds},panes:{anatomy:{getBoundingClientRect:()=>bounds}},
    scanVolume:{invalidate(){invalidations++}},gpuSync:{},anatomyExport:null,needsFrame:false});
  const exportPromise=r.exportAnatomyPng(new AbortController().signal);
  assert.ok(r.anatomyExport);
  r.requestRender(false);
  await assert.rejects(exportPromise,/3D view changed during export/);
  assert.equal(r.anatomyExport,null);
  assert.equal(r.needsFrame,true,'the updated live frame is coalesced behind its existing GPU submission');
  assert.equal(invalidations,2);
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
    Object.assign(r,{disposed:false,width:100,height:100,renderSerial:0,interacting:false,refinementStarted:0,budget:{longEdge:360},estimateAllocation:()=>({}),retiredSurfaces:[],
      container:{getBoundingClientRect:()=>({left:0,top:0,width:100,height:100})},
      panes:{anatomy:empty,axial:empty,coronal:empty,sagittal:empty},canvas:{dispatchEvent(){}},scanVolume:{active:false},
      renderer:{info:{reset(){},memory:{},render:{calls:0}},getContext:()=>({fenceSync:()=>null}),getPixelRatio:()=>pixelRatio,setPixelRatio(value){pixelRatio=value;changes.push(value)},setScissorTest(){},clear(){}}});
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
  let disposed=0,layerUpdates=0;
  r.preparedSourceLayers=new Set(['brain']);
  r.canvas=new EventTarget();
  const prepared=[];
  r.canvas.addEventListener('anatomy-surface',event=>prepared.push(event.detail.name));
  r.mriTexture={unchanged:true};const image=r.mriTexture;
  r.scanVolume.setLayers=()=>layerUpdates++;
  Object.assign(uniforms,{uLabels:{value:null},uLayerCount:{value:1},uLayerCodes:{value:Array.from({length:128},()=>new THREE.Vector2())},uLayerEnabled:{value:new Float32Array(128)}});
  uniforms.uColors.value=Array.from({length:128},()=>new THREE.Color());
  const positions=new Float32Array([0,0,0,1,0,0,0,1,0]),normals=new Float32Array([0,0,1,0,0,1,0,0,1]);
  const layer={name:'tumor',mask:new Uint8Array(60),color:'#ff00ff',surface:{source:positions,positions:positions.slice(),sourceNormals:normals,normals:normals.slice(),indices:new Uint32Array([0,1,2])}};
  const labelTexture=new THREE.Data3DTexture(new Uint8Array(60),5,4,3);
  labelTexture.format=THREE.RedFormat;labelTexture.addEventListener('dispose',()=>disposed++);
  r.labelTexture=labelTexture;
  const codes=Array.from({length:128},(_,i)=>i===0?[1,-1]:[0,0]);
  const next={...original,compartments:[layer,oldLayer],packedLabels:{data:new Uint8Array(120),codes}};
  assert.throws(()=>r.setAnatomyLayers({...next,packedLabels:{data:new Uint8Array(60),codes}}),/Invalid anatomy label channel/);
  assert.equal(r.labelTexture,labelTexture);
  r.setAnatomyLayers(next);assert.equal(r.volume,next);assert.equal(r.camera,camera);assert.equal(r.mriTexture,image);
  assert.equal(r.surfaces.get('brain'),mesh);assert(r.surfaces.has('tumor'));assert.equal(disposed,1);assert.equal(layerUpdates,1);
  assert.deepEqual(prepared,['tumor']);
  assert.deepEqual([...r.preparedSourceLayers],['brain','tumor']);
  assert.equal(r.labelTexture.format,THREE.RGFormat);assert.equal(uniforms.uLabels.value,r.labelTexture);
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
  const callbacks=new Map();let id=0,ready=false,deleted=0,resets=0;
  globalThis.window={devicePixelRatio:1};globalThis.requestAnimationFrame=callback=>{callbacks.set(++id,callback);return id};
  try{
    const {r}=renderer(),empty={getBoundingClientRect:()=>({left:0,top:0,width:0,height:0})};
    const gl={SYNC_GPU_COMMANDS_COMPLETE:1,TIMEOUT_EXPIRED:2,WAIT_FAILED:3,ALREADY_SIGNALED:4,CONDITION_SATISFIED:5,fenceSync:()=>({}),flush(){},clientWaitSync:()=>ready?4:2,deleteSync(){deleted++}};
    Object.assign(r,{requestRender:VolumeRenderer.prototype.requestRender,disposed:false,frame:0,renderSerial:0,width:100,height:100,sliceRevision:0,interacting:false,refinementStarted:0,budget:{longEdge:360},gpuSync:null,estimateAllocation:()=>({}),retiredSurfaces:[],
      container:{getBoundingClientRect:()=>({left:0,top:0,width:100,height:100})},panes:{anatomy:empty,axial:empty,coronal:empty,sagittal:empty},canvas:{dispatchEvent(){}},scanVolume:{active:false},controls:{target:new THREE.Vector3(),update(){}},
      renderer:{info:{reset(){resets++},memory:{},render:{calls:0}},getContext:()=>gl,getPixelRatio:()=>1,setScissorTest(){},clear(){}}});
    r.render();for(let i=0;i<100;i++)r.requestRender(false);assert.equal(r.renderSerial,1);assert.equal(callbacks.size,1);
    const tick=()=>{const [key,callback]=callbacks.entries().next().value;callbacks.delete(key);callback()};
    tick();assert.equal(r.renderSerial,1);ready=true;tick();assert.equal(r.renderSerial,2);assert.equal(deleted,1);
    tick();assert.equal(r.gpuSync,null);assert.equal(callbacks.size,0);assert.equal(resets,2,'statistics reset once per complete frame');
  }finally{globalThis.window=previousWindow;globalThis.requestAnimationFrame=previousRaf;}
});

// Exercise the actual render/fence/poll pipeline with a deterministic GPU seam.
// These controls qualify publication ordering, not rendered medical pixels.
function fencedFrame(run){
  const previous={window:globalThis.window,raf:globalThis.requestAnimationFrame,cancel:globalThis.cancelAnimationFrame};
  const callbacks=new Map(),live=new Set(),status=new Map(),events=[],ready=[],errors=[];let id=0,syncId=0;
  globalThis.window={devicePixelRatio:1};globalThis.requestAnimationFrame=fn=>{callbacks.set(++id,fn);return id};globalThis.cancelAnimationFrame=key=>callbacks.delete(key);
  try{
    const {r}=renderer(),empty={getBoundingClientRect:()=>({left:0,top:0,width:0,height:0})};
    const gl={SYNC_GPU_COMMANDS_COMPLETE:1,TIMEOUT_EXPIRED:2,WAIT_FAILED:3,ALREADY_SIGNALED:4,CONDITION_SATISFIED:5,
      fenceSync(){if(gl.noFence)return null;if(gl.throwFence)throw Error('fence');assert.equal(live.size,0,'one live GPU fence');const sync={id:++syncId};live.add(sync);return sync},
      flush(){if(gl.throwFlush)throw Error('flush')},clientWaitSync(sync){if(gl.throwWait)throw Error('wait');return status.get(sync)??gl.TIMEOUT_EXPIRED},deleteSync(sync){assert(live.delete(sync))}};
    Object.assign(r,{requestRender:VolumeRenderer.prototype.requestRender,disposed:false,graphicsLost:false,frame:0,renderSerial:0,width:100,height:100,sliceRevision:0,interacting:false,refinementStarted:0,budget:{longEdge:360},gpuSync:null,estimateAllocation:()=>({}),retiredSurfaces:[],
      container:{getBoundingClientRect:()=>({left:0,top:0,width:100,height:100})},panes:{anatomy:empty,axial:empty,coronal:empty,sagittal:empty},canvas:{dispatchEvent(event){events.push(event);f.onEvent?.(event)}},scanVolume:{active:false},controls:{target:new THREE.Vector3(),update(){}},
      renderer:{info:{reset(){},memory:{},render:{calls:0}},getContext:()=>gl,getPixelRatio:()=>1,setScissorTest(){},clear(){}}});
    const f={r,gl,callbacks,events,ready,errors,live,onEvent:null,
      request(name,onReady){const interaction={view:{stateId:name},onDisplayReady:view=>{ready.push(view.stateId);events.push({type:'accepted',detail:{stateId:view.stateId}});onReady?.()},onDisplayError:(view,message)=>errors.push({stateId:view.stateId,message})};r.surgicalInteraction=interaction;r.queueSurgicalPublication(interaction);return interaction},
      tick(){const first=callbacks.entries().next().value;assert(first,'a frame or fence poll must be queued');callbacks.delete(first[0]);first[1]()},
      complete(value=gl.CONDITION_SATISFIED){assert(r.gpuSync);status.set(r.gpuSync,value);f.tick()}};
    run(f);
  }finally{globalThis.window=previous.window;globalThis.requestAnimationFrame=previous.raf;globalThis.cancelAnimationFrame=previous.cancel;}
}
test('native display publication waits for an explicitly successful GPU fence, after its event',()=>fencedFrame(f=>{
  f.request('A');f.tick();assert.deepEqual(f.ready,[]);assert.equal(f.r.renderSerial,1);
  for(let i=0;i<3;i++){f.tick();assert.deepEqual(f.ready,[]);assert.equal(f.r.renderSerial,1)}
  f.complete(f.gl.ALREADY_SIGNALED);assert.deepEqual(f.ready,['A']);assert.equal(f.live.size,0);
  assert.deepEqual(f.events.map(e=>e.type),['anatomy-frame','anatomy-gpu','anatomy-surgical-display','accepted']);
  assert.equal(f.events[2].detail.frameSerial,f.events[1].detail.serial);assert.equal(f.events[1].detail.surgicalStateId,'A');assert.equal(f.events[1].detail.sourceCaseHash,source.caseHash);
}));
test('a completed older frame cannot publish a newer tool request',()=>fencedFrame(f=>{
  f.request('B');f.tick();f.request('C');assert.deepEqual(f.ready,[]);f.complete();
  assert.deepEqual(f.ready,[]);assert.equal(f.r.renderSerial,2);f.complete();assert.deepEqual(f.ready,['C']);
}));
test('returning to published A during B submission must redraw and fence A',()=>fencedFrame(f=>{
  const a=f.request('A');f.tick();f.complete();f.request('B');f.tick();
  f.r.surgicalInteraction=a;f.r.queueSurgicalPublication(a);assert.deepEqual(f.ready,['A']);
  f.complete();assert.deepEqual(f.ready,['A']);assert.equal(f.r.renderSerial,3);f.complete();assert.deepEqual(f.ready,['A','A']);
}));
test('synchronous frame telemetry cannot substitute an undrawn publication',()=>fencedFrame(f=>{
  f.onEvent=event=>{if(event.type==='anatomy-frame'&&event.detail.surgicalStateId==='B')f.request('C')};
  f.request('B');f.tick();f.complete();assert.deepEqual(f.ready,[]);f.complete();assert.deepEqual(f.ready,['C']);
}));
test('synchronous GPU telemetry cannot certify a request queued by that event',()=>fencedFrame(f=>{
  f.onEvent=event=>{if(event.type==='anatomy-gpu'&&event.detail.surgicalStateId==='B')f.request('C')};
  f.request('B');f.tick();f.complete();assert.deepEqual(f.ready,[]);assert.equal(f.callbacks.size,1);
  f.tick();f.complete();assert.deepEqual(f.ready,['C']);
}));
test('publication callback and coalesced redraw retain a single scheduled frame and live fence',()=>fencedFrame(f=>{
  f.request('A',()=>f.r.requestRender(false));f.tick();f.r.requestRender(false);f.complete();
  assert.deepEqual(f.ready,['A']);assert.equal(f.r.renderSerial,1);assert.equal(f.callbacks.size,1);
  f.tick();assert.equal(f.r.renderSerial,2);assert.equal(f.live.size,1);f.complete();assert.equal(f.live.size,0);assert.equal(f.callbacks.size,0);
}));
for(const failure of ['noFence','throwFence','throwFlush','WAIT_FAILED','throwWait','unknown-status'])test(`GPU ${failure} rejects the exact unpublished state`,()=>fencedFrame(f=>{
  if(['noFence','throwFence','throwFlush','throwWait'].includes(failure))f.gl[failure]=true;
  f.request('A');f.tick();if(failure==='WAIT_FAILED')f.complete(f.gl.WAIT_FAILED);if(failure==='unknown-status')f.complete(999);if(failure==='throwWait')f.tick();
  assert.deepEqual(f.ready,[]);assert.equal(f.errors.length,1);assert.equal(f.errors[0].stateId,'A');assert.equal(f.r.pendingSurgicalPublication,null);assert.equal(f.live.size,0);assert.equal(f.callbacks.size,0);
}));
for(const flag of ['disposed','graphicsLost'])test(`${flag} before completion cannot admit or resubmit a view`,()=>fencedFrame(f=>{
  f.request('A');f.tick();f.r[flag]=true;f.tick();assert.deepEqual(f.ready,[]);assert.equal(f.r.renderSerial,1);
}));
test('publication telemetry supersession cannot admit a stale callback',()=>fencedFrame(f=>{
  f.onEvent=event=>{if(event.type==='anatomy-surgical-display'&&event.detail.surgicalStateId==='B')f.request('C')};
  f.request('B');f.tick();f.complete();assert.deepEqual(f.ready,[]);f.tick();f.complete();assert.deepEqual(f.ready,['C']);
}));
test('publication telemetry disposal cannot admit a view',()=>fencedFrame(f=>{
  f.onEvent=event=>{if(event.type==='anatomy-surgical-display')f.r.disposed=true};
  f.request('A');f.tick();f.complete();assert.deepEqual(f.ready,[]);assert.equal(f.r.renderSerial,1);
}));

test('completed frame telemetry retains the actual submitted source identity across a queued source change',()=>fencedFrame(f=>{
  f.r.volume={...f.r.volume,displayIdentity:'source-a',mri:new Float32Array(60).fill(1)};
  f.r.requestRender(false);f.tick();
  const submitted=f.events.find(e=>e.type==='anatomy-frame').detail;
  assert.equal(submitted.displayIdentity,'source-a');assert.deepEqual(submitted.sourceShape,[3,4,5]);
  assert.deepEqual(submitted.cursor,[0,0,0]);assert(submitted.cameraState.position.length===3);
  f.r.volume={...f.r.volume,displayIdentity:'source-b',mri:new Float32Array(60).fill(2)};
  f.complete();const completed=f.events.find(e=>e.type==='anatomy-gpu').detail;
  assert(completed.completed);assert.equal(completed.serial,submitted.serial);
  assert.equal(completed.displayIdentity,'source-a');assert.equal(completed.sourceSignalFingerprint,submitted.sourceSignalFingerprint);
}));
