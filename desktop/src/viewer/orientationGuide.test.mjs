import assert from 'node:assert/strict';
import {before,after,test} from 'node:test';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import * as THREE from 'three';

let server,OrientationGuide,orientationPose;
before(async()=>{
  server=await createServer({root:fileURLToPath(new URL('../../',import.meta.url)),logLevel:'error',optimizeDeps:{noDiscovery:true,include:[]},server:{middlewareMode:true,hmr:false,ws:false},appType:'custom'});
  ({OrientationGuide,orientationPose}=await server.ssrLoadModule('/src/viewer/orientationGuide.ts'));
});
after(async()=>{await server?.close();});
function element(){return {querySelectorAll:()=>[],querySelector:()=>null,parentElement:null,setAttribute(){}};}
function close(a,b,tolerance=1e-10){assert.equal(a.length,b.length);a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<tolerance,`${a} != ${b}`));}
function camera(position,up=[0,0,1]){
  const c=new THREE.PerspectiveCamera(34,1,.1,5000);c.position.fromArray(position);c.up.fromArray(up);c.lookAt(0,0,0);c.updateMatrixWorld(true);return c;
}

test('all anatomical camera directions and roll match; orbit target, pan, FOV and zoom do not affect the guide',()=>{
  const guide=new OrientationGuide(element());
  try{
    for(const [position,up] of [[[0,4,0],[0,0,1]],[[0,-4,0],[0,0,1]],[[4,0,0],[0,0,1]],[[-4,0,0],[0,0,1]],[[0,0,4],[0,1,0]],[[0,0,-4],[0,1,0]],[[3,4,2],[0,0,1]]]){
      const main=camera(position,up);main.rotateZ(.37);main.updateMatrixWorld(true);
      guide.syncCamera(main,144,142);assert.ok(guide.camera.quaternion.angleTo(main.quaternion)<1e-7);
      const state={position:guide.camera.position.toArray(),rotation:guide.camera.quaternion.toArray(),projection:guide.camera.projectionMatrix.toArray()};
      main.position.add(new THREE.Vector3(300,-400,50));main.fov=12;main.zoom=3;main.updateProjectionMatrix();main.updateMatrixWorld(true);
      guide.syncCamera(main,144,142);
      close(guide.camera.position.toArray(),state.position);close(guide.camera.quaternion.toArray(),state.rotation);close(guide.camera.projectionMatrix.toArray(),state.projection);
    }
    const front=camera([0,4,0]);guide.syncCamera(front,144,142);
    assert.ok(new THREE.Vector3(.2,0,0).project(guide.camera).x<0,'patient right appears on the left when facing the patient');
    guide.syncCamera(camera([0,-4,0]),144,142);
    assert.ok(new THREE.Vector3(.2,0,0).project(guide.camera).x>0,'patient right appears on the right from behind');
  }finally{guide.dispose();}
});

test('recorded pose and gravity agree with the actual Python backend for every preset and combined neck rotations',()=>{
  const root=fileURLToPath(new URL('../../../',import.meta.url));
  const effects=JSON.parse(execFileSync(fileURLToPath(new URL('../../../.venv/bin/python',import.meta.url)),['-c',
    "import json; from resectionlab.surgical_position import PRESETS, validate_position, describe_position; print(json.dumps([describe_position(validate_position({'preset':p,'headRotationRasDeg':r})) for p in PRESETS for r in ([0,0,0],[10,-25,35],[-30,15,-60])]))"],{cwd:root,encoding:'utf8'}));
  const guide=new OrientationGuide(element()),bodyGeometry=guide.mannequin.body.geometry,headGeometry=guide.mannequin.head.geometry;
  try{
    for(const e of effects){
      const original=structuredClone(e),pose=orientationPose(e);assert.ok(pose);
      const b=new THREE.Matrix3().fromArray(e.bodyRasToRoom.flat()).transpose(),h=new THREE.Matrix3().fromArray(e.headRasToRoom.flat()).transpose();
      const bodyToHead=new THREE.Matrix3().setFromMatrix4(pose.bodyToHead);
      close(h.clone().multiply(bodyToHead).toArray(),b.toArray());
      close(pose.down.toArray(),e.gravityRasMPerS2.map(n=>n/9.80665));
      guide.setPosture(e);assert.deepEqual(guide.mannequin.head.quaternion.toArray(),[0,0,0,1]);
      close(new THREE.Matrix3().setFromMatrix4(new THREE.Matrix4().makeRotationFromQuaternion(guide.mannequin.body.quaternion)).toArray(),bodyToHead.toArray());
      for(const [width,height] of [[144,142],[104,102]]){
        const main=camera([3,4,2]);guide.syncCamera(main,width,height);
        for(const mesh of [guide.mannequin.body,guide.mannequin.head]){
          const attribute=mesh.geometry.getAttribute('position');
          for(let i=0;i<attribute.count;i++){
            const p=new THREE.Vector3().fromBufferAttribute(attribute,i).applyMatrix4(mesh.matrixWorld).project(guide.camera);
            assert.ok(Math.abs(p.x)<1&&Math.abs(p.y)<1,'whole mannequin stays inside the inset');
          }
        }
      }
      assert.deepEqual(e,original);
    }
    assert.equal(guide.mannequin.body.geometry,bodyGeometry);assert.equal(guide.mannequin.head.geometry,headGeometry);
    assert.ok((bodyGeometry.index.count+headGeometry.index.count)/3<20000,'bounded mesh workload');
  }finally{guide.dispose();}
});

test('missing, malformed, reflected, non-orthogonal and stale posture transforms return to an anatomical reference',()=>{
  const identity=[[1,0,0],[0,1,0],[0,0,1]],good={preset:'supine',bodyRasToRoom:identity,headRasToRoom:identity};
  const guide=new OrientationGuide(element());
  try{
    for(const value of [null,undefined,{}, {...good,preset:'unknown'},...[
      null,[[1,0,0],[0,-1,0],[0,0,1]],[[2,0,0],[0,.5,0],[0,0,1]],[[1,0,0],[0,1,0],[0,0,NaN]],[[1,0],[0,1]],
    ].map(headRasToRoom=>({...good,headRasToRoom}))]){
      guide.setPosture(good);assert.equal(orientationPose(value),null);guide.setPosture(value);
      assert.deepEqual(guide.mannequin.body.quaternion.toArray(),[0,0,0,1]);assert.equal(guide.down,null);
    }
  }finally{guide.dispose();}
});

test('overlay draws only its scissor, preserves rendering state even on failure, and releases geometry',()=>{
  const guide=new OrientationGuide(element()),calls=[];
  let viewport=new THREE.Vector4(10,20,800,600),scissor=new THREE.Vector4(10,20,800,600),scissorTest=false;
  const renderer={autoClear:true,getViewport:v=>v.copy(viewport),getScissor:v=>v.copy(scissor),getScissorTest:()=>scissorTest,
    setViewport:(...args)=>{viewport=args[0]?.isVector4?args[0].clone():new THREE.Vector4(...args);},
    setScissor:(...args)=>{scissor=args[0]?.isVector4?args[0].clone():new THREE.Vector4(...args);},setScissorTest:v=>{scissorTest=v;},
    clearDepth:()=>calls.push({viewport:viewport.toArray(),scissor:scissor.toArray(),scissorTest}),render:()=>{throw Error('GPU unavailable');}};
  assert.equal(guide.render(renderer,camera([0,4,0]),{left:40,top:20,width:800,height:640},{left:52,top:100,width:0,height:0}),false);assert.equal(calls.length,0);
  assert.throws(()=>guide.render(renderer,camera([0,4,0]),{left:40,top:20,width:800,height:640},{left:52,top:480,width:144,height:142}),/GPU unavailable/);
  assert.deepEqual(calls,[{viewport:[12,38,144,142],scissor:[12,38,144,142],scissorTest:true}]);
  assert.deepEqual(viewport.toArray(),[10,20,800,600]);assert.deepEqual(scissor.toArray(),[10,20,800,600]);assert.equal(scissorTest,false);assert.equal(renderer.autoClear,true);
  let disposed=0;for(const mesh of [guide.mannequin.body,guide.mannequin.head]){mesh.geometry.addEventListener('dispose',()=>disposed++);mesh.material.addEventListener('dispose',()=>disposed++);}
  guide.dispose();assert.equal(disposed,4);
});
