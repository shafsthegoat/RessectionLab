import assert from 'node:assert/strict';
import {before,after,test} from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import * as THREE from 'three';
import {InstrumentDisplayState} from './inspectionTool.ts';

// CPU-only renderer seams, with generated geometry and three virtual MRI planes.
let server,cache,VolumeRenderer;
before(async()=>{
  cache=await fs.mkdtemp(path.join(os.tmpdir(),'passive-renderer-'));
  server=await createServer({root:fileURLToPath(new URL('../../',import.meta.url)),cacheDir:cache,
    logLevel:'error',server:{middlewareMode:true,hmr:false,ws:false},appType:'custom'});
  VolumeRenderer=(await server.ssrLoadModule('/src/viewer/VolumeRenderer.ts')).VolumeRenderer;
});
after(async()=>{await server?.close();if(cache)await fs.rm(cache,{recursive:true,force:true});});
const hash=c=>`sha256:${c.repeat(64)}`;
const volume={caseHash:hash('a'),frame:'RAS+',shape:[8,8,8],
  affine:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]};
function frame(step){
  const pose=index=>{
    const held=index===0?[1,2].includes(step):step<4;
    return {scope:'generated-persistent-passive-episode',caseHash:volume.caseHash,
      episodeId:hash('b'),frameIndex:step,toolId:index?'other':'moving',mode:index?'aspirate':'probe',
      tipRasMm:[index?4:2,3,held?4:2],axis:[0,0,1],workingLengthMm:4,tipLengthMm:1,
      shaftRadiusMm:.25,tipRadiusMm:.5,retainedAtDepth:held,physicalAbsenceClaim:false,
      poseStatus:held?'held_in_declared_void':'access_pose_recorded',
      activation:index===0&&held?'passive_only':'off'};
  };
  return {scope:'generated-persistent-passive-episode',caseHash:volume.caseHash,
    episodeId:hash('b'),frameIndex:step,stepCount:5,shape:volume.shape,affine:volume.affine,
    phase:['ready','held','inspected','withdrawn','both_withdrawn','stopped'][step],
    elapsedGeneratedSeconds:step,inspectionStatus:step<2?'not_acquired':'verified',
    event:step===2?{kind:'generated_inspection',status:'verified'}:
      step===5?{kind:'stop',reason:'declared_stop_without_injury_claim'}:null,
    recordedTools:[pose(0),pose(1)],recordedEpisodeBounds:{caseHash:volume.caseHash,
      episodeId:hash('b'),boundsRasMm:[[-3,-3,-5],[9,9,9]]},
    primaryPairClearanceChecked:true,independentTissueMotionChecked:true};
}
function fakeRenderer(){
  const shaders=Array.from({length:3},()=>({uniforms:{uReplayActive:{value:0},
    uRouteCount:{value:0},uShaftStart:{value:[new THREE.Vector3(),new THREE.Vector3()]},
    uShaftEnd:{value:[new THREE.Vector3(),new THREE.Vector3()]},
    uTipEnd:{value:[new THREE.Vector3(),new THREE.Vector3()]},
    uRadii:{value:[new THREE.Vector2(),new THREE.Vector2()]},
    uRouteColors:{value:[new THREE.Color(),new THREE.Color()]}}}));
  const r=Object.create(VolumeRenderer.prototype);
  Object.assign(r,{volume,materials:()=>shaders,mode:'anatomy',replayActive:false,
    passiveActive:false,passiveDisplays:[],recordedDisplay:null,replayBounds:null,
    replayGeneration:0,replayWorker:null,pendingReplayGroup:null,
    recordedTools:new THREE.Group(),replayGroup:new THREE.Group(),anatomy:new THREE.Group(),
    tools:new THREE.Group(),inspectionTools:new THREE.Group(),
    instrumentDisplay:new InstrumentDisplayState(),requestRender(){},onSurfaceStatus(){},
    setInspectionTool(){},setDiagnosticLayer(){},setStructuralProposal(){},setPriorLayer(){}});
  return {r,shaders};
}

test('two generated capsules reach all MRI shader planes and stay visible at access',()=>{
  const {r,shaders}=fakeRenderer();
  for(const step of [0,1,3,4,5]){
    r.setPassiveReplay(frame(step));
    assert.equal(r.recordedTools.children.length,2);
    assert.equal(r.recordedTools.children[0].children.length,2);
    assert.equal(r.recordedTools.children[1].children.length,2);
    assert(r.recordedTools.children.every(group=>group.children.every(mesh=>
      mesh.userData.scope==='generated-persistent-passive-episode')));
    for(const shader of shaders){
      assert.equal(shader.uniforms.uReplayActive.value,0);
      assert.equal(shader.uniforms.uRouteCount.value,2);
      assert.equal(shader.uniforms.uTipEnd.value[0].x,2);
      assert.equal(shader.uniforms.uTipEnd.value[1].x,4);
    }
  }
  r.setPassiveReplay(null);
  assert.equal(r.recordedTools.children.length,0);
  assert(shaders.every(shader=>shader.uniforms.uRouteCount.value===0));
});

test('invalid replacement clears old passive pose before refusal',()=>{
  const {r,shaders}=fakeRenderer();r.setPassiveReplay(frame(1));
  const invalid=frame(2);invalid.recordedTools[1].physicalAbsenceClaim=true;
  assert.throws(()=>r.setPassiveReplay(invalid),/withheld/);
  assert.equal(r.recordedTools.children.length,0);
  assert.equal(r.replayActive,false);
  assert(shaders.every(shader=>shader.uniforms.uRouteCount.value===0));
});
