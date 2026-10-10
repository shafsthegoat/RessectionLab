import assert from 'node:assert/strict';
import {before,after,test} from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import * as THREE from 'three';
import {InstrumentDisplayState} from './inspectionTool.ts';
import {recordedEpisodeBounds,checkedRecordedEpisodeBounds} from './recordedTool.ts';
import {familyFixture,familyCatalog,familyRequest,familyAssets} from '../../tests/helpers/contact-family-fixture.mjs';

// Saved generated assets + real renderer methods on CPU THREE objects. No WebGL,
// Electron, Python, policy call, native transition or private reference access.
let server,cache,VolumeRenderer,view,stop;
before(async()=>{
  cache=await fs.mkdtemp(path.join(os.tmpdir(),'episode-camera-'));
  server=await createServer({root:fileURLToPath(new URL('../../',import.meta.url)),cacheDir:cache,logLevel:'error',server:{middlewareMode:true,hmr:false,ws:false},appType:'custom'});
  VolumeRenderer=(await server.ssrLoadModule('/src/viewer/VolumeRenderer.ts')).VolumeRenderer;
  const {hydrateContactFamilyEpisode}=await server.ssrLoadModule('/src/contact-family-data.ts');
  const hydrate=method=>{const f=familyFixture(method);return hydrateContactFamilyEpisode(f.result,familyAssets(f),familyRequest(f),familyCatalog());};
  view=await hydrate('SEARCH');stop=await hydrate('STOP');
});
after(async()=>{await server?.close();if(cache)await fs.rm(cache,{recursive:true,force:true,maxRetries:3,retryDelay:20});});
function renderer(volume,width=720,height=800){
  const r=Object.create(VolumeRenderer.prototype),camera=new THREE.PerspectiveCamera(34,width/height,.1,5000);camera.up.set(0,0,1);
  const controls={target:new THREE.Vector3(),update(){camera.lookAt(this.target);camera.updateMatrixWorld();}};
  Object.assign(r,{volume,camera,controls,bounds:[[-.5,-.5,-.5],[14.5,14.5,13.5]],
    panes:{anatomy:{getBoundingClientRect:()=>({width,height}),dataset:{}}},surfaces:new Map(),instrumentDisplay:new InstrumentDisplayState(),
    tools:new THREE.Group(),inspectionTools:new THREE.Group(),recordedTools:new THREE.Group(),recordedDisplay:null,anatomy:new THREE.Group(),
    replayBounds:null,fittedReplayId:null,replayGroup:new THREE.Group(),replayGeneration:0,replayWorker:null,pendingReplayGroup:null,replayActive:false,disposed:false,
    removedTexture:new THREE.Data3DTexture(new Uint8Array(1),1,1,1),materials:()=>[],requestRender(){},onSurfaceStatus(){},onReplayError(){},setStructuralProposal(){},setPriorLayer(){},mode:'instruments'});
  return r;
}
function replay(r,frame){const prior=globalThis.Worker;globalThis.Worker=class{postMessage(){}terminate(){}};try{r.setReplay(frame);}finally{globalThis.Worker=prior;}}
function corners(bounds){return [0,1].flatMap(x=>[0,1].flatMap(y=>[0,1].map(z=>new THREE.Vector3(bounds[x][0],bounds[y][1],bounds[z][2]))));}

test('saved full episode envelope includes every shaft and tip cap before the first pose',()=>{
  const b=view.frames[0].recordedEpisodeBounds;assert.ok(b);assert.equal(view.frames[0].recordedTool,null);
  assert(view.frames.every(f=>f.recordedEpisodeBounds===b));
  let fractional=false;
  for(const f of view.frames){const p=f.recordedTool;if(!p)continue;
    fractional ||=p.tipRasMm.some(n=>!Number.isInteger(n));
    const start=p.tipRasMm.map((n,i)=>n-p.axis[i]*p.workingLengthMm),end=p.tipRasMm.map((n,i)=>n-p.axis[i]*p.tipLengthMm);
    for(const [point,radius] of [[start,p.shaftRadiusMm],[end,p.shaftRadiusMm],[end,p.tipRadiusMm],[p.tipRasMm,p.tipRadiusMm]])
      for(let axis=0;axis<3;axis++){assert(point[axis]-radius>=b.boundsRasMm[0][axis]-1e-12);assert(point[axis]+radius<=b.boundsRasMm[1][axis]+1e-12);}
  }
  assert(fractional);assert.equal(stop.frames[0].recordedEpisodeBounds,null);
});

test('first replay installation fits full episode at narrow and wide pane aspect ratios',()=>{
  for(const [width,height]of [[720,800],[300,900],[1200,500]]){
    const r=renderer(view.volume,width,height);replay(r,view.frames[0]);assert.equal(r.fittedReplayId,view.episode.episodeId);
    for(const corner of corners(view.frames[0].recordedEpisodeBounds.boundsRasMm)){
      const projected=corner.project(r.camera);assert(Math.abs(projected.x)<1&&Math.abs(projected.y)<1&&Math.abs(projected.z)<1,JSON.stringify(projected));
    }
  }
});

test('tool switches and frame scrubbing preserve a user orbit; explicit Fit uses the whole episode',()=>{
  const r=renderer(view.volume);replay(r,view.frames[0]);
  r.camera.position.set(80,-120,70);r.controls.target.set(2,4,6);r.controls.update();
  const position=r.camera.position.toArray(),target=r.controls.target.toArray();
  for(const f of view.frames){replay(r,f);assert.deepEqual(r.camera.position.toArray(),position);assert.deepEqual(r.controls.target.toArray(),target);}
  r.fitCamera('instruments');
  for(const corner of corners(view.frames[0].recordedEpisodeBounds.boundsRasMm)){const p=corner.project(r.camera);assert(Math.abs(p.x)<1&&Math.abs(p.y)<1);}
});

test('new episode refits once while anatomy mode and replay hide/show do not reset orbit',()=>{
  const r=renderer(view.volume);replay(r,view.frames[0]);
  r.camera.position.set(80,90,100);const prior=r.camera.position.toArray();replay(r,null);replay(r,view.frames[0]);assert.deepEqual(r.camera.position.toArray(),prior);
  const next=structuredClone(view.frames[0]);next.recordedEpisodeBounds.episodeId='sha256:'+'b'.repeat(64);
  replay(r,next);assert.notDeepEqual(r.camera.position.toArray(),prior);
  r.mode='anatomy';r.camera.position.set(30,40,50);next.recordedEpisodeBounds.episodeId='sha256:'+'c'.repeat(64);
  replay(r,next);assert.deepEqual(r.camera.position.toArray(),[30,40,50]);
});

test('rotated fractional capsules use both cap radii and reject mixed episode identities',()=>{
  const f=structuredClone(view.frames.find(f=>f.recordedTool)),p=f.recordedTool;p.tipRasMm=[.25,-2.5,7.125];p.axis=[Math.SQRT1_2,0,Math.SQRT1_2];p.workingLengthMm=120;p.shaftRadiusMm=.4;p.tipRadiusMm=1.5;
  const b=recordedEpisodeBounds(view.volume,[f],p.episodeId);assert.equal(b.boundsRasMm[0][1],-4);assert.equal(b.boundsRasMm[1][1],-1);assert.equal(b.boundsRasMm[1][2],8.625);
  assert.throws(()=>recordedEpisodeBounds(view.volume,[f],'sha256:'+'e'.repeat(64)),/mix episodes/);
  f.recordedEpisodeBounds={...b,caseHash:'sha256:'+'e'.repeat(64)};assert.throws(()=>checkedRecordedEpisodeBounds(view.volume,f),/do not match/);
});
