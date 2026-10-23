import assert from 'node:assert/strict';
import {before,after,test} from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import * as THREE from 'three';
let server,cache,VolumeRenderer,registerModeledAnatomy,prepareSurface,packAnatomyLabels,IncrementalSurfaceEngine,LabelTextureUpdates,retainSurface,cachedSurface,surgicalFlowGroup;
before(async()=>{
  cache=await fs.mkdtemp(path.join(os.tmpdir(),'surgical-renderer-'));
  server=await createServer({root:fileURLToPath(new URL('../../',import.meta.url)),cacheDir:cache,logLevel:'error',optimizeDeps:{noDiscovery:true,include:[]},server:{middlewareMode:true,hmr:false,ws:false},appType:'custom',plugins:[{
    name:'test-conditioning-calls',enforce:'pre',transform(code,id){
      if(!id.endsWith('/surfaceSmoothing.ts'))return;
      const signature='export function fairIndexedSurface(points:Float32Array,indices:Uint32Array,affine:Affine,kind?:string,limits?:Float64Array){';
      assert(code.includes(signature),'conditioning call instrumentation must match the actual implementation');
      return code.replace(signature,signature+'\nglobalThis.__resectionLabConditioningCalls=(globalThis.__resectionLabConditioningCalls??0)+1;');
    }
  }]});
  VolumeRenderer=(await server.ssrLoadModule('/src/viewer/VolumeRenderer.ts')).VolumeRenderer;
  ({surgicalFlowGroup}=await server.ssrLoadModule('/src/viewer/surgicalFlowGeometry.ts'));
  ({registerModeledAnatomy}=await server.ssrLoadModule('/src/viewer/modeledAnatomy.ts'));
  ({prepareSurface}=await server.ssrLoadModule('/src/viewer/prepareSurface.ts'));
  ({packAnatomyLabels}=await server.ssrLoadModule('/src/viewer/labelPacking.ts'));
  ({IncrementalSurfaceEngine}=await server.ssrLoadModule('/src/viewer/incrementalSurface.ts'));
  ({LabelTextureUpdates}=await server.ssrLoadModule('/src/viewer/labelTextureUpdates.ts'));
  ({retainSurface,cachedSurface}=await server.ssrLoadModule('/src/viewer/surfaceCache.ts'));
});
after(async()=>{await server?.close();delete globalThis.__resectionLabConditioningCalls;if(cache)await fs.rm(cache,{recursive:true,force:true});});
const tool=id=>({id,radiusMm:2,shaftRadiusMm:.5,workingLengthMm:40});
const slot=id=>({instrument:id,active:false,parameters:{jawOpeningMm:3},position:[1,2,3],quaternion:[0,0,0,1]});
function interaction(){return {view:{stateId:'accepted',scene:{shape:[3,3,3],resolutionMm:[1,1,1],affine:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},
  slots:{left:slot('cusa'),right:slot('micro_scissors')},catalog:[tool('cusa'),tool('micro_scissors')],moved:[],removed:[],contact:[13],cuts:[],thermal:{},fluid:{},history:[]},
  pose:{position:[1,2,3],quaternion:[0,0,0,1]},path:[],parameters:{jawOpeningMm:3},selectedSlot:'left',mode:'navigate',onPose(){},onStroke(){}};}
function renderer(){const r=Object.create(VolumeRenderer.prototype);Object.assign(r,{scene:new THREE.Scene(),surgicalGroup:null,surgicalGhost:new THREE.Group(),surgicalGhostKey:null,
  surgicalGizmo:{attach(){},detach(){},setMode(){}},surgicalSlots:new Map(),surgicalEffects:null,surgicalEffectsView:null,surgicalPath:null,surgicalPathSource:null,
  displayWork:{surgicalUpdates:0,toolBuilds:0,effectBuilds:0,pathBufferBuilds:0},controls:{enabled:true},requestRender(){}});return r;}
test('pose-only rehearsal edits retain all accepted geometry and only transform the resident ghost',()=>{
  const r=renderer(),input=interaction(),before=structuredClone(input.view);r.setSurgical(input);
  const accepted=r.surgicalGroup.children.slice(),ghost=r.surgicalGhost.children[0],counts={...r.displayWork};
  const geometries=[];r.surgicalGroup.traverse(o=>{if(o.geometry)geometries.push(o.geometry);});ghost.traverse(o=>{if(o.geometry)geometries.push(o.geometry);});
  let disposed=0;geometries.forEach(g=>g.addEventListener('dispose',()=>disposed++));
  for(let i=0;i<100;i++)r.setSurgical({...input,pose:{position:[i/2,3,4],quaternion:[0,0,1,0]}});
  assert.deepEqual(r.surgicalGroup.children,accepted);assert.equal(r.surgicalGhost.children[0],ghost);assert.equal(disposed,0);
  assert.equal(r.displayWork.toolBuilds,counts.toolBuilds);assert.equal(r.displayWork.effectBuilds,counts.effectBuilds);assert.equal(r.displayWork.pathBufferBuilds,0);
  assert.deepEqual(r.surgicalGhost.position.toArray(),[49.5,3,4]);assert.deepEqual(r.surgicalGhost.quaternion.toArray(),[0,0,1,0]);assert.deepEqual(input.view,before);
});
test('activation updates material and recorded transform; parameter and tool exchange rebuild only affected instrument',()=>{
  const r=renderer(),input=interaction();r.setSurgical(input);const left=r.surgicalSlots.get('left').group,right=r.surgicalSlots.get('right').group,ghost=r.surgicalGhost.children[0];
  const view={...input.view,slots:{...input.view.slots,left:{...input.view.slots.left,active:true,position:[4,5,6]}}};
  r.setSurgical({...input,view});assert.equal(r.surgicalSlots.get('left').group,left);assert.equal(r.surgicalSlots.get('right').group,right);
  assert.equal(left.userData.surgicalAccent.color.getHexString(),'e0b35d');assert.deepEqual(left.position.toArray(),[4,5,6]);
  r.setSurgical({...input,view,parameters:{radiusMm:3}});assert.notEqual(r.surgicalGhost.children[0],ghost);assert.equal(r.surgicalSlots.get('left').group,left);
  r.setSurgical({...input,view,selectedSlot:'right'});assert.equal(r.surgicalGhost.children[0].name,'instrument:micro_scissors');assert.equal(r.surgicalSlots.get('right').group,right);
});
test('observed effective activation colors both tools without mutating latched state and hides the editable ghost',()=>{
  const r=renderer(),input=interaction(),before=structuredClone(input.view);
  r.setSurgical({...input,readOnly:true,effectiveActivation:{left:true,right:false}});
  assert.equal(r.surgicalSlots.get('left').group.userData.surgicalAccent.color.getHexString(),'e0b35d');
  assert.equal(r.surgicalSlots.get('right').group.userData.surgicalAccent.color.getHexString(),'41677a');
  assert.equal(r.surgicalGhost.visible,false);assert.deepEqual(input.view,before);
  r.setSurgical(input);assert.equal(r.surgicalGhost.visible,true);
  assert.equal(r.surgicalSlots.get('left').group.userData.surgicalAccent.color.getHexString(),'41677a');
});
test('draw path grows a reusable bounded-capacity buffer, hides on cancel and releases recorded state on exit',()=>{
  const r=renderer(),input=interaction();r.setSurgical(input);
  const path=Array.from({length:3},(_,i)=>({position:[i,2,3],quaternion:[0,0,0,1]}));r.setSurgical({...input,path});const line=r.surgicalPath,geometry=line.geometry;
  r.setSurgical({...input,path:[...path,{position:[9,8,7],quaternion:[0,0,0,1]}]});assert.equal(r.surgicalPath,line);assert.equal(line.geometry,geometry);
  assert.deepEqual([...geometry.attributes.position.array.slice(9,12)],[9,8,7]);assert.equal(geometry.drawRange.count,4);assert.equal(r.displayWork.pathBufferBuilds,1);
  r.setSurgical({...input,path:[]});assert.equal(line.visible,false);r.setSurgical(null);assert.equal(r.surgicalGroup.children.length,0);assert.equal(r.surgicalSlots.size,0);assert.equal(r.surgicalGhost.visible,false);
});
test('repeated accepted effect changes dispose point buffers and materials as well as meshes and lines',()=>{
  const r=renderer(),input=interaction();let created=0,disposed=0,materialsDisposed=0;
  for(let i=0;i<100;i++){
    const view={...input.view,stateId:'state-'+i,thermal:{13:i+1},fluid:{14:1},debris:{15:1}};
    r.setSurgical({...input,view});
    r.surgicalEffects.traverse(object=>{if(object instanceof THREE.Points){
      created++;let released=false;object.geometry.addEventListener('dispose',()=>{assert(!released);released=true;disposed++;});
      object.material.addEventListener('dispose',()=>materialsDisposed++);
    }});
    const current=r.surgicalEffects.children.filter(object=>object instanceof THREE.Points).length;
    assert.equal(created-disposed,current);
  }
  r.setSurgical(null);assert.equal(disposed,created);assert.equal(materialsDisposed,created);
});
function flowInteraction(){
  const input=interaction(),hash=c=>'sha256:'+c.repeat(64);
  input.view.stateId=hash('a');input.view.scene.sceneId=hash('b');input.view.metrics={elapsedSeconds:.2};
  input.view.scene.affine=[[-2,0,0,10],[0,3,0,20],[0,0,4,30],[0,0,0,1]];
  input.recordedFrame={branchId:'main',position:2,frameIndex:1};
  input.flowDisplay={schema:'resectionlab.recorded-field-transfers.v1',sceneId:hash('b'),actionIndex:2,frameIndex:1,beforeStateId:hash('c'),stateId:hash('a'),startSeconds:0,endSeconds:.2,truncated:false,operations:[
    {kind:'transfer',field:'fluid',provenance:'mixed',fromCell:13,toCell:14,volumeMm3:.5,startSeconds:0,endSeconds:.1},
    {kind:'uptake',field:'debris',provenance:'debris',fromCell:12,toCell:null,volumeMm3:.25,startSeconds:.1,endSeconds:.2},
  ]};return input;
}
test('recorded arrows follow the scene affine and stay local for uptake; visual size does not encode volume',()=>{
  const input=flowInteraction(),before=structuredClone(input.view),group=surgicalFlowGroup(input.view,input.flowDisplay);
  const [shaft,head,local]=group.children,m=new THREE.Matrix4();assert.equal(group.children.length,3);assert(group.children.every(o=>o.isInstancedMesh));
  shaft.getMatrixAt(0,m);const tail=new THREE.Vector3(0,-.5,0).applyMatrix4(m);assert(tail.distanceTo(new THREE.Vector3(8,23,34))<1e-6);
  head.getMatrixAt(0,m);const tip=new THREE.Vector3(0,.5,0).applyMatrix4(m);assert(tip.distanceTo(new THREE.Vector3(8,23,38))<1e-6);
  local.getMatrixAt(0,m);assert.deepEqual(new THREE.Vector3().setFromMatrixPosition(m).toArray(),[8,23,30]);
  assert.equal(local.count,1);assert.equal(shaft.count,1);assert.equal(head.count,1);
  const changed=structuredClone(input.flowDisplay);changed.operations[0].volumeMm3=900;const second=surgicalFlowGroup(input.view,changed);
  for(let i=0;i<3;i++)assert.deepEqual(second.children[i].instanceMatrix.array,group.children[i].instanceMatrix.array);
  assert.deepEqual(input.view,before);
  const unlocated=structuredClone(input.flowDisplay);unlocated.operations=[{...unlocated.operations[0],kind:'escape',fromCell:null,toCell:null}];
  const escaped=surgicalFlowGroup(input.view,unlocated);assert.equal(escaped.children.length,0,'off-grid escape has no invented location');
  const hits=[];group.children.forEach(o=>o.raycast(null,hits));assert.deepEqual(hits,[]);
  for(const g of [group,second])g.traverse(o=>{if(o.isMesh){o.geometry.dispose();o.material.dispose();o.dispose();}});
});
test('resident recorded transfers retain buffers on pose edits, clear on preview/toggle and release instance allocations',()=>{
  const r=renderer(),input=flowInteraction();r.setSurgical(input);const group=r.surgicalFlow;let disposed=0,instances=0;
  group.traverse(o=>{if(o.isMesh){instances++;o.addEventListener('dispose',()=>disposed++);}});
  for(let i=0;i<10;i++)r.setSurgical({...input,pose:{...input.pose,position:[i,0,0]}});assert.equal(r.surgicalFlow,group);
  assert.throws(()=>r.setSurgical({...input,recordedFrame:{...input.recordedFrame,position:3}}),/demanded frame/);
  assert.equal(r.surgicalInteraction.view,input.view);assert.equal(r.surgicalFlow,group);
  r.setSurgical({...input,preview:input.view});assert.equal(r.surgicalFlow,null);assert.equal(disposed,instances);
  r.setSurgical(input);r.setSurgical({...input,flowDisplay:undefined});assert.equal(r.surgicalFlow,null);
  r.setSurgical(input);r.setSurgical(null);assert.equal(r.surgicalFlow,null);
});
test('source-surface picking restores smoothed geometry and accepts the first hit inside every active clip plane',()=>{
  const r=Object.create(VolumeRenderer.prototype),source=new THREE.BufferGeometry(),smooth=new THREE.BufferGeometry(),mesh=new THREE.Mesh(smooth,new THREE.MeshBasicMaterial());
  const residual=new THREE.Mesh(smooth,new THREE.MeshBasicMaterial());residual.userData.replayGeometry={source};
  const hit=(x,y,z)=>({point:new THREE.Vector3(x,y,z)}),hits=[hit(1,-1,1),hit(-1,1,1),hit(2,3,4)];let observed=false;
  Object.assign(r,{panes:{anatomy:{getBoundingClientRect:()=>({left:0,top:0,width:100,height:100})}},camera:{},sourcePlane:{visible:false},anatomy:{visible:true},
    surfaces:new Map([['brain',mesh]]),surfaceGeometry:new Map([['brain',{source,smooth}]]),replayGroup:{children:[residual]},
    clipPlane:new THREE.Plane(new THREE.Vector3(1,0,0),0),extraClipPlanes:[new THREE.Plane(new THREE.Vector3(0,1,0),0)],
    pickRay:{setFromCamera(){},intersectObjects(objects){assert(objects.includes(mesh)&&objects.includes(residual));assert.equal(mesh.geometry,source);assert.equal(residual.geometry,source);observed=true;return hits;}}});
  assert.deepEqual(r.pick(50,50),[2,3,4]);assert(observed);assert.equal(mesh.geometry,smooth);assert.equal(residual.geometry,smooth);
  // Independent plane-distance comparator, retaining deeper hits after clipping.
  const expected=hits.find(h=>h.point.x>=0&&h.point.y>=0).point.toArray();assert.deepEqual(r.pick(50,50),expected);
});
test('graphics retry uses the cached recovery extension after the lost context stops exposing extensions',()=>{
  const r=Object.create(VolumeRenderer.prototype);let restored=0,ready=0,lookups=0;
  Object.assign(r,{disposed:false,graphicsRecovery:{restoreContext(){restored++}},renderer:{getContext:()=>({isContextLost:()=>true,getExtension(){lookups++;return null;}})},onGraphicsStatus:()=>ready++});
  r.retryGraphics();assert.equal(restored,1);assert.equal(lookups,0);assert.equal(ready,0);
  r.renderer={getContext:()=>({isContextLost:()=>false})};r.retryGraphics();assert.equal(ready,1);
});
test('source-plane picking follows its axis clip while extra box/oblique clips apply to mesh hits',()=>{
  const r=Object.create(VolumeRenderer.prototype),plane=new THREE.Mesh(),point=new THREE.Vector3(1,1,1);
  plane.material={uniforms:{uWorldToVoxel:{value:new THREE.Matrix4()}}};plane.visible=true;
  Object.assign(r,{panes:{anatomy:{getBoundingClientRect:()=>({left:0,top:0,width:100,height:100})}},camera:{},sourcePlane:plane,anatomy:{visible:false},surfaces:new Map(),surfaceGeometry:new Map(),replayGroup:{children:[]},
    volume:{shape:[2,2,2],mri:new Float32Array(8).fill(1)},clipPlane:null,extraClipPlanes:[new THREE.Plane(new THREE.Vector3(1,0,0),-2)],
    pickRay:{setFromCamera(){},intersectObjects(){return [{object:plane,point}];}}});
  assert.deepEqual(r.pick(50,50),[1,1,1]);r.clipPlane=new THREE.Plane(new THREE.Vector3(1,0,0),-2);assert.equal(r.pick(50,50),null);
});
test('picking skips source-plane pixels discarded by the MRI shader and hidden zero-opacity surfaces',()=>{
  const r=Object.create(VolumeRenderer.prototype),plane=new THREE.Mesh(),source=new THREE.BufferGeometry(),mesh=new THREE.Mesh(source,new THREE.MeshBasicMaterial({opacity:1})),hidden=new THREE.Mesh(source,new THREE.MeshBasicMaterial({opacity:0}));
  plane.material={uniforms:{uWorldToVoxel:{value:new THREE.Matrix4()}}};plane.visible=true;
  const planeHit={object:plane,point:new THREE.Vector3(1,1,1)},deeper={object:mesh,point:new THREE.Vector3(1,1,0)};
  Object.assign(r,{panes:{anatomy:{getBoundingClientRect:()=>({left:0,top:0,width:100,height:100})}},camera:{},sourcePlane:plane,anatomy:{visible:true},
    surfaces:new Map([['source',mesh],['hidden',hidden]]),surfaceGeometry:new Map(),replayGroup:{children:[]},clipPlane:null,extraClipPlanes:[],
    volume:{shape:[3,3,3],mri:new Float32Array(27)},pickRay:{setFromCamera(){},intersectObjects(objects){assert(!objects.includes(hidden));return [planeHit,deeper];}}});
  assert.deepEqual(r.pick(50,50),deeper.point.toArray());r.volume.mri[13]=2;
  assert.deepEqual(r.pick(50,50),planeHit.point.toArray());planeHit.point.set(-.5,1,1);
  assert.deepEqual(r.pick(50,50),deeper.point.toArray());
  // The shader tolerates a tiny edge overhang, clamps before interpolation,
  // and samples x-major source memory rather than texture storage order.
  planeHit.point.set(-.00005,1,1);r.volume.mri.fill(0);r.volume.mri[4]=3;
  assert.deepEqual(r.pick(50,50),planeHit.point.toArray());
  planeHit.point.x=-.0002;assert.deepEqual(r.pick(50,50),deeper.point.toArray());
});
function modeledFixture(){
  const shape=[6,5,4],affine=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],count=120;
  const source={caseHash:'source',caseId:'source',shape,affine,frame:'RAS+',mri:new Float32Array(count),compartments:[
    {name:'brain',kind:'brain',color:'#aabbcc',mask:new Uint8Array(count).fill(1)},
    {name:'vessel',kind:'vessel',color:'#bb3344',mask:Uint8Array.from({length:count},(_,i)=>Number(i%20===0))}]};
  const r=Object.create(VolumeRenderer.prototype),packed=packAnatomyLabels(source),work={};
  const uniforms={uLayerCount:{value:2},uVisibleBits:{value:3},uLayerEnabled:{value:new Float32Array(128)},uLayerCodes:{value:Array.from({length:128},()=>new THREE.Vector2())}};
  for(const key of ['surgicalUpdates','toolBuilds','effectBuilds','pathBufferBuilds','sourceWorkerBuilds','replayWorkerBuilds','sourceRemeshedLayers','replayRemeshedLayers','workerInputCloneBytes','workerInputTransferBytes','workerResultTransferBytes','modelWorkerBuilds','modelRemeshedLayers','modelRemeshedChunks','modelCommits'])work[key]=0;
  Object.assign(r,{volume:source,affine,bounds:[[0,0,0],[5,4,3]],workspaceSettings:{},extraClipPlanes:[],clipPlane:null,surfaceSmoothing:true,visible:{},
    anatomy:new THREE.Group(),surfaces:new Map(),surfaceGeometry:new Map(),preparedSourceLayers:new Set(),retiredSurfaces:[],
    worker:{terminate(){}},sourceRemaining:0,sourceFailed:false,sourceSurfaceGeneration:0,modelWorker:null,modelWorkerStates:new Map(),modelIdentities:new WeakMap(),modelIdentityCounter:0,modelGeneration:0,modelStage:null,
    labelTexture:new THREE.Data3DTexture(packed.data,4,5,6),labelTextureUpdates:new LabelTextureUpdates(),renderer:{initTexture(){},copyTextureToTexture(){}},labelCodes:packed.codes,displayWork:work,canvas:{dispatchEvent(){}},onSurfaceStatus(){},onError(message){r.lastError=message;},
    materials:()=>[{uniforms}],
    scanVolume:{setLayers(){}},setWorkspaceSettings(){},setSurgical(value){r.installedInteraction=value;r.surgicalInteraction=value;},requestRender(){}});
  for(const layer of source.compartments){r.installSurface(layer,prepareSurface(layer.mask,shape,affine,layer.kind));r.preparedSourceLayers.add(layer.name);}
  const derive=removed=>{const mask=source.compartments[0].mask.slice();removed.forEach(i=>mask[i]=0);const volume={...source,displayIdentity:'modeled-'+removed.join(','),compartments:[{...source.compartments[0],mask},source.compartments[1]]};registerModeledAnatomy(source,volume,'mapping',removed);return volume;};
  return {r,source,derive,uniforms};
}
function submitPreparedState(r){if(r.modelStage?.ready)r.commitModelStage(r.modelStage);r.labelTextureUpdates?.sync(r.renderer,r.labelTexture);r.finishSurgicalFrame();}
function modelWorkerMock(){
  const previous=globalThis.Worker,workers=[];
  globalThis.Worker=class{queue=[];messages=[];engine=new IncrementalSurfaceEngine();constructor(){workers.push(this);}postMessage(value,transfer=[]){const copy=structuredClone(value,{transfer});this.messages.push(copy);this.queue.push(copy);}terminate(){this.terminated=true;}
    flush(){const request=this.queue.shift();if(!request)return;
      let result;if(request.task==='pack')result={task:'pack',jobId:request.jobId,packed:packAnatomyLabels(request.volume)};
      else result=this.engine.prepare(request);
      this.onmessage({data:result});
    }};
  return {workers,restore(){globalThis.Worker=previous;}};
}
test('modeled tissue stages labels/surfaces/tools atomically, reuses unchanged MRI/layers, and backward seek reuses resident geometry',()=>{
  const mock=modelWorkerMock();try{
    const {r,source,derive}=modeledFixture(),a=derive([22]),b=derive([22,23]),oldBrain=r.surfaces.get('brain'),vessel=r.surfaces.get('vessel'),texture=r.labelTexture,sourceCopy=source.compartments[0].mask.slice();
    const ready=[],viewA={stateId:'a'},interactionA={view:viewA,onDisplayReady:view=>ready.push(view)};
    r.installSurgicalState(a,interactionA);const worker=r.modelWorker;assert.equal(worker.queue.length,1);assert.equal(r.volume,source);assert.equal(r.surfaces.get('brain'),oldBrain);assert.equal(ready.length,0);
    worker.flush();assert.equal(r.volume,source,'global fairing alone does not publish partial labels');assert.equal(worker.queue[0].task,'pack');worker.flush();
    assert.equal(r.volume,source,'a completed stage still waits for render submission');assert.equal(ready.length,0);submitPreparedState(r);
    assert.equal(r.volume,a);assert.deepEqual(ready,[viewA]);assert.equal(r.surfaces.get('vessel'),vessel);assert.equal(r.volume.mri,source.mri);assert.equal(r.labelTexture,texture);assert.deepEqual(texture.image.data,packAnatomyLabels(a).data);
    const brainA=r.surfaces.get('brain');r.installSurgicalState(b,{view:{stateId:'b'}});assert.equal(worker.messages.at(-1).mask,undefined);assert.deepEqual([...worker.messages.at(-1).indices],[23]);worker.flush();submitPreparedState(r);
    assert.equal(r.volume,b);assert.equal(r.surfaces.get('vessel'),vessel);assert.equal(r.labelTexture,texture);assert.deepEqual(texture.image.data,packAnatomyLabels(b).data);
    const requests=worker.messages.length;r.installSurgicalState(a,interactionA);submitPreparedState(r);assert.equal(r.volume,a);assert.equal(r.surfaces.get('brain'),brainA);assert.equal(worker.messages.length,requests);assert.deepEqual(source.compartments[0].mask,sourceCopy);
  }finally{mock.restore();}
});
test('new read-only tissue surfaces defer picking acceleration while editable boundaries keep eager admission',()=>{
  const mock=modelWorkerMock();try{
    const {r,derive}=modeledFixture(),calls=[];r.picking={install(...args){calls.push(args);}};
    r.installSurgicalState(derive([22]),{view:{stateId:'editable'}});
    while(r.modelWorker.queue.length)r.modelWorker.flush();submitPreparedState(r);
    assert(calls.length>0);assert(calls.every(([, ,eager])=>eager===true));
    calls.length=0;r.installSurgicalState(derive([22,23]),{view:{stateId:'observed'},readOnly:true});
    while(r.modelWorker.queue.length)r.modelWorker.flush();submitPreparedState(r);
    assert.equal(calls.length,1);assert.equal(calls[0][2],false);
    assert.equal(r.volume.compartments[0].mask[23],0,'picking policy cannot change accepted tissue');
  }finally{mock.restore();}
});
test('initial zero-effect mapping releases native GPU pairs, retains later undo pairs, and restores source from its CPU cache',()=>{
  const mock=modelWorkerMock();try{
    const {r,source,derive}=modeledFixture(),baseline=derive([]),cut=derive([22]),delta=derive([22,23]);
    for(const layer of source.compartments)retainSurface(layer.mask,r.affine,layer.kind,prepareSurface(layer.mask,source.shape,r.affine,layer.kind),source.shape);
    const native=r.surfaceGeometry.get('brain'),nativeMesh=r.surfaces.get('brain'),vessel=r.surfaces.get('vessel'),disposed={source:0,smooth:0,material:0};
    native.source.addEventListener('dispose',()=>disposed.source++);native.smooth.addEventListener('dispose',()=>disposed.smooth++);nativeMesh.material.addEventListener('dispose',()=>disposed.material++);
    r.installSurgicalState(baseline,{view:{stateId:'mapped'}});const worker=r.modelWorker;
    assert.equal(worker.messages[0].retainChunks,false);while(worker.queue.length)worker.flush();
    assert.deepEqual(disposed,{source:0,smooth:0,material:0},'source stays complete until the replacement commits');submitPreparedState(r);
    assert.deepEqual(disposed,{source:1,smooth:1,material:1});assert.equal(r.retiredSurfaces.length,0);assert.equal(r.modelWorkerStates.size,0);assert.equal(r.surfaces.get('vessel'),vessel);
    const mappedMesh=r.surfaces.get('brain');r.installSurgicalState(cut,{view:{stateId:'cut'}});
    assert.equal(worker.messages.at(-1).retainChunks,true);assert(worker.messages.at(-1).mask,'first real cut admits the previously deferred chunks');while(worker.queue.length)worker.flush();submitPreparedState(r);
    assert(r.retiredSurfaces.some(entry=>entry.pair.mesh===mappedMesh),'modeled baseline remains reusable after a tissue edit');
    r.installSurgicalState(delta,{view:{stateId:'delta'}});assert.equal(worker.messages.at(-1).mask,undefined);assert.deepEqual([...worker.messages.at(-1).indices],[23]);while(worker.queue.length)worker.flush();submitPreparedState(r);
    const messages=worker.messages.length;r.installSurgicalState(baseline,{view:{stateId:'undo'}});submitPreparedState(r);
    assert.equal(r.surfaces.get('brain'),mappedMesh);assert.equal(worker.messages.length,messages,'undo reuses the retained modeled baseline');
    r.installSurgicalState(source,null);assert.equal(worker.queue[0].task,'pack','source reconstruction uses cached CPU surfaces rather than another extraction');while(worker.queue.length)worker.flush();submitPreparedState(r);
    assert.equal(r.volume,source);assert.notEqual(r.surfaces.get('brain'),nativeMesh);assert.deepEqual(r.surfaceGeometry.get('brain').source.attributes.position.array,native.source.attributes.position.array);
    assert.equal(r.surfaces.get('vessel'),vessel);assert.deepEqual(r.labelTexture.image.data,packAnatomyLabels(source).data);assert.deepEqual(disposed,{source:1,smooth:1,material:1});
  }finally{mock.restore();}
});
test('obsolete modeled worker results and failures cannot publish a mixed state; retry keeps original source bytes attached',()=>{
  const mock=modelWorkerMock();try{
    const {r,source,derive}=modeledFixture(),a=derive([22]),b=derive([23]),errors=[],viewB={stateId:'b'};
    r.installSurgicalState(a,{view:{stateId:'a'}});const stale=r.modelWorker;r.installSurgicalState(b,{view:viewB,onDisplayError:(view,error)=>errors.push([view,error])});
    assert.equal(stale.terminated,true);stale.flush();assert.equal(r.volume,source);
    const activeStage=r.modelStage,activeWorker=r.modelWorker;stale.onerror();assert.equal(r.modelStage,activeStage);assert.equal(r.modelWorker,activeWorker);assert.equal(errors.length,0,'obsolete worker errors cannot reject newer state');
    r.lastError='Existing unrelated source error';
    r.modelWorker.onerror();assert.equal(r.volume,source);assert.equal(errors.length,1);assert.match(errors[0][1],/previous accepted state: surface worker failed/);assert.equal(r.lastError,'Existing unrelated source error','handled model failure does not poison or clear generic source/graphics errors');assert.equal(r.modelStage,null);assert.equal(source.mri.byteLength,480);assert.equal(source.compartments[0].mask.byteLength,120);
    const created=mock.workers.length;for(let i=0;i<10;i++)assert.equal(r.installSurgicalState(b,{view:viewB,onDisplayError:()=>{throw Error('Duplicate failure callback');}}),false);
    assert.equal(r.modelStage,null);assert.equal(mock.workers.length,created,'callback rerenders cannot silently retry a failed display');assert.equal(r.volume,source);
    r.installSurgicalState(b,{view:{stateId:'retry'}});const current=r.modelWorker;while(current.queue.length)current.flush();submitPreparedState(r);assert.equal(r.volume,b);assert.equal(r.labelTexture.image.data.length,480);
  }finally{mock.restore();}
});
test('idle model worker failure drops its lost cache and admits a fresh worker for the next tissue change',()=>{
  const mock=modelWorkerMock();try{
    const {r,derive}=modeledFixture(),a=derive([22]);r.installSurgicalState(a,{view:{stateId:'a'}});const worker=r.modelWorker;
    while(worker.queue.length)worker.flush();submitPreparedState(r);assert.equal(r.volume,a);assert(r.modelWorkerStates.size>0);
    worker.onerror();assert.equal(r.modelWorker,null);assert.equal(r.modelWorkerStates.size,0);assert.equal(worker.terminated,true);assert.equal(r.volume,a);assert.equal(r.lastError,undefined);
    r.installSurgicalState(derive([22,23]),{view:{stateId:'b'}});assert.notEqual(r.modelWorker,worker);assert(r.modelWorker.messages[0].mask,'replacement worker receives full source admission');
  }finally{mock.restore();}
});
test('modeled failure still reports a visible generic error for callers without a display callback',()=>{
  const mock=modelWorkerMock();try{
    const {r,source,derive}=modeledFixture();r.installSurgicalState(derive([22]),{view:{stateId:'without-handler'}});r.modelWorker.onerror();
    assert.equal(r.volume,source);assert.match(r.lastError,/previous accepted state: surface worker failed/);
  }finally{mock.restore();}
});
test('only the latest tool state publishes after submission and newer tissue preparation retains published effects',()=>{
  const mock=modelWorkerMock();try{
    const {r,source,derive}=modeledFixture(),ready=[],a={stateId:'a'},b={stateId:'b'},c={stateId:'c'},d={stateId:'d'};
    const request=view=>({view,onDisplayReady:reported=>ready.push(reported)});
    r.installSurgicalState(source,request(a));assert.equal(ready.length,0);r.finishSurgicalFrame();assert.deepEqual(ready,[a]);
    r.installSurgicalState(source,request(b));r.installSurgicalState(source,request(c));assert.deepEqual(ready,[a]);r.finishSurgicalFrame();assert.deepEqual(ready,[a,c]);
    r.installSurgicalState(source,request(d));const tissue={stateId:'tissue'},next=derive([22]);r.installSurgicalState(next,request(tissue));
    assert.equal(r.surgicalInteraction.view,c,'pending tool state was not published and must not leak into the retained display');r.finishSurgicalFrame();assert.deepEqual(ready,[a,c]);
    const worker=r.modelWorker;while(worker.queue.length)worker.flush();assert.equal(r.volume,source);assert.deepEqual(ready,[a,c]);
    r.commitModelStage(r.modelStage);assert.equal(r.volume,next);assert.deepEqual(ready,[a,c],'CPU resource installation is not submission');
    r.finishSurgicalFrame();assert.deepEqual(ready,[a,c,tissue]);
    const previous=r.modelWorker.messages.length;r.installSurgicalState(next,request(tissue));assert.deepEqual(ready,[a,c,tissue,tissue]);assert.equal(r.modelWorker.messages.length,previous,'already published pose update needs no anatomy work');
  }finally{mock.restore();}
});
test('mapping reorder rebinds hidden-layer visibility and exact label codes at the same commit',()=>{
  const mock=modelWorkerMock();try{
    const {r,source,uniforms}=modeledFixture();r.visible={brain:true,vessel:false};
    const next={...source,compartments:[source.compartments[1],source.compartments[0]],displayIdentity:'reordered'};registerModeledAnatomy(source,next,'reordered',[]);
    r.installSurgicalState(next,{view:{stateId:'reorder'}});const worker=r.modelWorker;while(worker.queue.length)worker.flush();submitPreparedState(r);
    assert.equal(uniforms.uVisibleBits.value,2);assert.deepEqual([...uniforms.uLayerEnabled.value.slice(0,2)],[0,1]);assert.deepEqual(r.labelTexture.image.data,packAnatomyLabels(next).data);
  }finally{mock.restore();}
});
test('OFF admits exact raw source geometry; ON lazily conditions that same tissue without changing cells or effects',()=>{
  const mock=modelWorkerMock();try{
    const {r,source,derive}=modeledFixture(),next=derive([22]),before=source.compartments[0].mask.slice(),ready=[];
    const view={stateId:'raw',slots:{left:{active:true},right:{active:false}},fluid:{23:1},debris:{22:2}},request={view,readOnly:true,effectiveActivation:{left:true,right:false},onDisplayReady:v=>ready.push(v)};
    const conditionedBefore=globalThis.__resectionLabConditioningCalls;
    r.setSurfaceSmoothing(false);r.installSurgicalState(next,request);const worker=r.modelWorker;
    assert.equal(worker.messages[0].smooth,false);while(worker.queue.length)worker.flush();submitPreparedState(r);
    assert.equal(globalThis.__resectionLabConditioningCalls,conditionedBefore,'OFF never invokes the actual fairing/diagnostic implementation');
    const raw=r.surfaceGeometry.get('brain'),reference=prepareSurface(next.compartments[0].mask,next.shape,next.affine,'brain',true),labels=r.labelTexture.image.data.slice();
    assert.equal(raw.smoothingPrepared,false);assert.equal(raw.source,raw.smooth);assert.equal(r.surfaces.get('brain').geometry,raw.source);
    assert.deepEqual(raw.source.attributes.position.array,reference.positions);assert.deepEqual(raw.source.attributes.normal.array,reference.normals);assert.deepEqual(raw.source.index.array,reference.indices);
    assert.equal(cachedSurface(next.compartments[0].mask,r.affine,'brain',next.shape),undefined,'raw buffers never satisfy conditioned cache requests');
    assert.equal(cachedSurface(next.compartments[0].mask,r.affine,'brain',next.shape,false).smoothingPrepared,false);
    assert.deepEqual(ready,[view]);assert.equal(r.installedInteraction,request);
    r.setSurfaceSmoothing(true);assert.equal(r.volume,next);assert.equal(r.modelStage.next,next);assert.equal(worker.messages.at(-1).smooth,true);
    assert.equal(worker.messages.at(-1).mask,undefined,'preference does not discard the resident source-cell cache');assert.equal(worker.messages.at(-1).indices.length,0);
    assert.equal(r.surfaces.get('brain').geometry,raw.source,'current raw tissue stays visible while its smoothing prepares');
    const conditionedOn=globalThis.__resectionLabConditioningCalls;
    while(worker.queue.length)worker.flush();submitPreparedState(r);
    assert.equal(globalThis.__resectionLabConditioningCalls,conditionedOn+1,'ON conditions only the raw displayed layer');
    const smooth=r.surfaceGeometry.get('brain');assert.equal(smooth.smoothingPrepared,true);assert.notEqual(smooth.smooth,smooth.source);
    assert.deepEqual(smooth.source.attributes.position.array,reference.positions);assert.deepEqual(smooth.smooth.attributes.position.array,reference.smoothPositions);
    assert.deepEqual(r.labelTexture.image.data,labels);assert.equal(r.installedInteraction,request);assert.equal(r.volume,next);assert.deepEqual(source.compartments[0].mask,before);
    r.setSurfaceSmoothing(false);const delta=derive([22,23]);r.installSurgicalState(delta,{view:{...view,stateId:'delta'}});
    assert.equal(worker.messages.at(-1).smooth,false);assert.equal(worker.messages.at(-1).mask,undefined);assert.deepEqual([...worker.messages.at(-1).indices],[23]);
    while(worker.queue.length)worker.flush();submitPreparedState(r);assert.equal(r.volume,delta);assert.equal(r.surfaceGeometry.get('brain').smoothingPrepared,false);
  }finally{mock.restore();}
});
test('raw stable source geometry installs active framing and sorting bounds before picking binds',()=>{
  const {r}=modeledFixture();r.surfaceSmoothing=false;let bound=false;
  const buffers={positions:Float32Array.of(1e6,-1e6,1e6,-1,-1,0,1,-1,0,0,1,0),normals:Float32Array.of(0,0,0,0,0,1,0,0,1,0,0,1),indices:Uint32Array.of(1,2,3),smoothingPrepared:false,representation:'stable-source-ids',activeBounds:[[-1,-1,0],[1,1,0]]};
  r.picking={install(mesh,geometry,eager){assert.equal(eager,false);assert.deepEqual(geometry.boundingBox.min.toArray(),[-1,-1,0]);assert.deepEqual(geometry.boundingBox.max.toArray(),[1,1,0]);assert.deepEqual(geometry.boundingSphere.center.toArray(),[0,0,0]);assert(geometry.boundingSphere.radius>=Math.sqrt(2));assert.equal(geometry.userData.sourceIndexing,'stable-source-ids');bound=true;}};
  const pair=r.makeSurface({name:'bone',kind:'bone',color:'#aabbcc',mask:new Uint8Array()},buffers,false);assert(pair);assert(bound);assert.equal(pair.source,pair.smooth);assert.equal(pair.mesh.geometry,pair.source);
  const reference=new THREE.BufferGeometry();reference.setAttribute('position',new THREE.BufferAttribute(buffers.positions.slice(3),3));reference.computeBoundingBox();reference.computeBoundingSphere();
  assert.deepEqual(pair.source.boundingBox,reference.boundingBox);assert.deepEqual(pair.source.boundingSphere.center,reference.boundingSphere.center,'transparent sort centre matches compact active geometry');
  r.anatomy.clear();r.anatomy.add(pair.mesh);pair.mesh.updateWorldMatrix(true,false);
  const expected=reference.boundingBox.clone().applyMatrix4(pair.mesh.matrixWorld);assert.deepEqual(new THREE.Box3().setFromObject(r.anatomy),expected,'complete-bounds camera framing excludes stale slots');
  pair.source.dispose();pair.mesh.material.dispose();reference.dispose();
});
test('mid-flight preference changes cancel the old quality and cannot publish stale target buffers',()=>{
  const mock=modelWorkerMock();try{
    const {r,source,derive}=modeledFixture(),a=derive([22]),b=derive([22,23]),ready=[];
    const viewA={stateId:'a'},viewB={stateId:'b'},request=view=>({view,onDisplayReady:v=>ready.push(v)});
    r.installSurgicalState(a,request(viewA));const staleSmooth=r.modelWorker;r.setSurfaceSmoothing(false);
    const rawWorker=r.modelWorker;assert.notEqual(rawWorker,staleSmooth);assert.equal(staleSmooth.terminated,true);assert.equal(rawWorker.messages[0].smooth,false);
    staleSmooth.flush();assert.equal(r.volume,source);assert.deepEqual(ready,[]);
    r.setSurfaceSmoothing(true);const staleRaw=rawWorker,active=r.modelWorker;assert.equal(staleRaw.terminated,true);assert.equal(active.messages[0].smooth,true);
    staleRaw.flush();assert.equal(r.volume,source);assert.deepEqual(ready,[]);
    r.installSurgicalState(b,request(viewB));const latest=r.modelWorker;assert.equal(active.terminated,true);active.flush();assert.deepEqual(ready,[]);
    while(latest.queue.length)latest.flush();submitPreparedState(r);assert.equal(r.volume,b);assert.deepEqual(ready,[viewB]);assert.equal(r.surfaceGeometry.get('brain').smoothingPrepared,true);
  }finally{mock.restore();}
});
