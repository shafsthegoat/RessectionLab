import assert from 'node:assert/strict';
import {before,after,test} from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import * as THREE from 'three';
import {InstrumentDisplayState} from './inspectionTool.ts';
let server,cache,VolumeRenderer,originalWorker;
before(async()=>{
  cache=await fs.mkdtemp(path.join(os.tmpdir(),'rehearsal-renderer-'));
  server=await createServer({root:fileURLToPath(new URL('../../',import.meta.url)),cacheDir:cache,logLevel:'error',optimizeDeps:{noDiscovery:true,include:[]},server:{middlewareMode:true,hmr:false,ws:false},appType:'custom'});
  VolumeRenderer=(await server.ssrLoadModule('/src/viewer/VolumeRenderer.ts')).VolumeRenderer;
  originalWorker=globalThis.Worker;
  globalThis.Worker=class{messages=[];postMessage(value){this.messages.push(value);}terminate(){this.terminated=true;}};
});
after(async()=>{globalThis.Worker=originalWorker;await server?.close();if(cache)await fs.rm(cache,{recursive:true,force:true});});
const volume={caseHash:'patient',caseId:'patient',frame:'RAS+',shape:[8,8,8],affine:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],mri:new Float32Array(512).fill(7),compartments:[{name:'brain',color:'#88bbcc',mask:new Uint8Array(512).fill(1)}]};
function state(index=0){
  const mask=(indices=[])=>{const m=new Uint8Array(512);indices.forEach(i=>m[i]=1);return m;};
  return {scope:'source-bound-post-exposure-research',caseHash:'patient',episodeId:'episode',stateId:`state-${index}`,frameIndex:index,stepCount:3,shape:volume.shape,affine:volume.affine,frame:'RAS+',elapsedSeconds:index,kind:'aspirate',
    tools:[['aspirator','aspiration',2],['field_manager','field_management',5]].map(([toolId,mode,x])=>({toolId,mode,tipRasMm:[x,3,4+index],axis:[0,0,1],active:index>0,workingLengthMm:6,tipLengthMm:1,shaftRadiusMm:.2,tipRadiusMm:.4})),
    removedMask:mask(index?[2]:[]),assumedExposureMask:mask([1]),fieldContactMask:mask([3]),contactMask:mask([3]),metrics:{},
    recordedEpisodeBounds:{caseHash:'patient',episodeId:'episode',boundsRasMm:[[-2,-2,-3],[8,8,9]]}};
}
function renderer(){
  const shaders=Array.from({length:4},()=>({uniforms:{uColors:{value:[new THREE.Color()]},uSliceContours:{value:0},uLayerOpacities:{value:[.66]},uClipEnabled:{value:0},uClipPlane:{value:new THREE.Vector4()},uFusionOpacity:{value:.5},uVisibleBits:{value:1},uLayerEnabled:{value:[1]},uRemoved:{value:null},uReplayActive:{value:0},uRouteCount:{value:0},uShaftStart:{value:[new THREE.Vector3(),new THREE.Vector3()]},uShaftEnd:{value:[new THREE.Vector3(),new THREE.Vector3()]},uTipEnd:{value:[new THREE.Vector3(),new THREE.Vector3()]},uRadii:{value:[new THREE.Vector2(),new THREE.Vector2()]},uRouteColors:{value:[new THREE.Color(),new THREE.Color()]}}}));
  const r=Object.create(VolumeRenderer.prototype);
  Object.assign(r,{volume,bounds:[[0,0,0],[7,7,7]],workspaceSettings:{},appearanceSignature:null,
    extraClipPlanes:[],clipPlane:null,surfaces:new Map(),surfaceGeometry:new Map(),surfaceSmoothing:true,
    scanVolume:{configure(){},labels(){}},beginInteraction(){},endInteraction(){},cursor:[0,0,0],opacity:.4,routeSignature:'[]',visibilitySignature:'1',affine:volume.affine,materials:()=>shaders,mode:'anatomy',replayActive:false,passiveActive:false,passiveDisplays:[],recordedDisplay:null,replayBounds:null,rehearsalActive:false,rehearsalDisplays:[],removedTexture:new THREE.Texture(),visible:{},
    replayGeneration:0,replayWorker:null,pendingReplayGroup:null,recordedTools:new THREE.Group(),replayGroup:new THREE.Group(),anatomy:new THREE.Group(),tools:new THREE.Group(),inspectionTools:new THREE.Group(),instrumentDisplay:new InstrumentDisplayState(),requestRender(){},onSurfaceStatus(){},setInspectionTool(){},setDiagnosticLayer(){},setStructuralProposal(){},setPriorLayer(){}});
  return {r,shaders};
}
test('dedicated source rehearsal renders two full capsules on source plane and all MRI planes without legacy acceptance',()=>{
  const {r,shaders}=renderer(),source=volume.mri.slice(),brain=volume.compartments[0].mask.slice();r.setRehearsal(state(1));
  assert.equal(r.recordedTools.children.length,2);assert.equal(r.rehearsalActive,true);assert.equal(r.anatomy.visible,false);
  for(const group of r.recordedTools.children){assert.equal(group.children.length,2);assert(group.children.every(mesh=>mesh.userData.scope==='source-bound-post-exposure-research'));}
  for(const shader of shaders){assert.equal(shader.uniforms.uRouteCount.value,2);assert.equal(shader.uniforms.uReplayActive.value,1);assert.deepEqual(shader.uniforms.uShaftStart.value[0].toArray(),[2,3,-1]);assert.deepEqual(shader.uniforms.uTipEnd.value[1].toArray(),[5,3,5]);
    assert.equal(shader.uniforms.uRemoved.value.image.data[1],2);assert.equal(shader.uniforms.uRemoved.value.image.data[2],1);assert.equal(shader.uniforms.uRemoved.value.image.data[3],3);}
  const residual=r.replayWorker.messages.find(m=>m.name==='residual-0').mask;
  assert.equal(residual[1],0);assert.equal(residual[2],0);assert.equal(residual[3],1);
  assert.deepEqual(volume.mri,source);assert.deepEqual(volume.compartments[0].mask,brain);
  const priorWorker=r.replayWorker;r.setRehearsal(state(2));assert.equal(priorWorker.terminated,true);assert.equal(r.recordedTools.children.length,2);
  r.setRehearsal(null);assert.equal(r.recordedTools.children.length,0);assert.equal(r.anatomy.visible,true);assert(shaders.every(s=>s.uniforms.uReplayActive.value===0&&s.uniforms.uRouteCount.value===0));
});
test('bad source or frame replacement clears existing tools and effects',()=>{
  const {r,shaders}=renderer();r.setRehearsal(state(1));const invalid=state(2);invalid.caseHash='other';
  assert.throws(()=>r.setRehearsal(invalid),/withheld/);assert.equal(r.recordedTools.children.length,0);assert.equal(r.rehearsalActive,false);assert.equal(r.anatomy.visible,true);assert(shaders.every(s=>s.uniforms.uRouteCount.value===0));
});

function deliver(worker,name,{smooth=true}={}) {
  const positions=new Float32Array([1,1,1,2,1,1,1,2,1]);
  const normals=new Float32Array([0,0,1,0,0,1,0,0,1]);
  const smoothPositions=new Float32Array([1.05,1.05,1,1.95,1.05,1,1.05,1.95,1]);
  worker.onmessage({data:{name,positions,normals,indices:new Uint32Array([0,1,2]),
    ...(smooth?{smoothPositions,smoothNormals:normals.slice()}: {})}});
  return {positions,normals,smoothPositions};
}
function assertAppearance(mesh,color,opacity,planes,visible=true) {
  assert.equal(mesh.material.color.getHexString(),color);
  assert.equal(mesh.material.opacity,opacity);
  assert.equal(mesh.material.transparent,opacity<1);
  assert.equal(mesh.material.depthWrite,opacity>=1);
  assert.equal(mesh.material.clippingPlanes.length,planes);
  assert.equal(mesh.visible,visible);
}
test('residual and pending surfaces honor current layer appearance and all cutaways without clipping full instruments',()=>{
  const {r}=renderer(),source=volume.mri.slice(),brain=volume.compartments[0].mask.slice();
  const initial={layerColor:{brain:'#cc2233'},layerOpacity:{brain:.25},clipping:{enabled:true,axis:'z',value:3}};
  r.setWorkspaceSettings(initial);r.setRehearsal(state(1));
  const worker=r.replayWorker;deliver(worker,'residual-0');
  const residual=r.pendingReplayGroup.children.find(m=>m.userData.compartment==='brain');
  assertAppearance(residual,'cc2233',.25,1);
  assert.equal(residual.material.clippingPlanes[0].distanceToPoint(new THREE.Vector3(0,0,2)),-1);
  const changed={layerColor:{brain:'#2244cc','Assumed initial exposure':'#ff0000'},layerOpacity:{brain:1},
    clipping:{enabled:true,axis:'x',value:2,flipped:true},clipBox:{enabled:true,min:[1,1,1],max:[6,6,6]},
    obliqueClip:{enabled:true,normal:[0,1,1],offset:-3}};
  r.setWorkspaceSettings(changed);r.update([0,0,0],{brain:false},.4,[],'anatomy');
  assertAppearance(residual,'2244cc',1,8,false);
  for(const name of ['assumed-exposure','rehearsal-removal','rehearsal-field-contact'])deliver(worker,name,{smooth:false});
  assert.equal(r.pendingReplayGroup,null);assert.equal(r.replayGroup.children.length,4);
  const effects=r.replayGroup.children.filter(m=>!m.userData.compartment);
  assert.deepEqual(effects.map(m=>m.material.color.getHexString()),['b7a7e8','b7f0d8','efb566']);
  assert(effects.every(m=>m.material.clippingPlanes.length===8&&m.visible&&m.material.opacity===.78));
  // A new partial frame and the currently displayed frame must both respond.
  r.setRehearsal(state(2));deliver(r.replayWorker,'residual-0');
  const pending=r.pendingReplayGroup.children[0];
  r.setWorkspaceSettings({layerColor:{brain:'#22cc44'},layerOpacity:{brain:.4},clipping:{enabled:false,axis:'z',value:3}});
  r.update([0,0,0],{brain:true},.4,[],'anatomy');
  assertAppearance(residual,'22cc44',.4,0);assertAppearance(pending,'22cc44',.4,0);
  assert(effects.every(m=>m.material.clippingPlanes.length===0));
  for(const group of r.recordedTools.children)for(const mesh of group.children){assert.equal(mesh.material.clippingPlanes,null);assert.equal(mesh.visible,true);}
  assert.equal(r.recordedTools.visible,true);assert.deepEqual(volume.mri,source);assert.deepEqual(volume.compartments[0].mask,brain);
});
test('residual smoothing toggles cached bounded worker buffers before and after arrival and disposes both on replacement',()=>{
  const {r}=renderer(),source=volume.compartments[0].mask.slice();r.setRehearsal(state(1));const worker=r.replayWorker;
  const request=worker.messages.find(m=>m.name==='residual-0');assert.equal(request.displaySmoothing,true);assert.deepEqual(request.affine,volume.affine);
  assert(worker.messages.filter(m=>m.name!=='residual-0').every(m=>m.displaySmoothing===false));
  r.setSurfaceSmoothing(false);const buffers=deliver(worker,'residual-0');
  const residual=r.pendingReplayGroup.children[0],pair=residual.userData.replayGeometry;
  assert.equal(residual.geometry,pair.source);assert.deepEqual(residual.geometry.attributes.position.array,buffers.positions);
  r.setSurfaceSmoothing(true);assert.equal(residual.geometry,pair.smooth);assert.deepEqual(residual.geometry.attributes.position.array,buffers.smoothPositions);
  for(const name of ['assumed-exposure','rehearsal-removal','rehearsal-field-contact'])deliver(worker,name,{smooth:false});
  r.setSurfaceSmoothing(false);assert.equal(residual.geometry,pair.source);r.setSurfaceSmoothing(true);assert.equal(residual.geometry,pair.smooth);
  let sourceDisposed=0,smoothDisposed=0;pair.source.addEventListener('dispose',()=>sourceDisposed++);pair.smooth.addEventListener('dispose',()=>smoothDisposed++);
  r.setRehearsal(null);assert.equal(sourceDisposed,1);assert.equal(smoothDisposed,1);assert.deepEqual(volume.compartments[0].mask,source);
});
