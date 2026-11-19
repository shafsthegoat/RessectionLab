import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {ResultInspectionScene} from './ResultInspectionScene.ts';
import {prepareSurface} from './prepareSurface.ts';
import {modeledVolume} from '../modeled-volume.ts';

const shape=[20,20,20],affine=[[1,0,0,12],[0,1,0,-22],[0,0,1,34],[0,0,0,1]];
const target=new Uint8Array(8000).fill(1);
const source={caseId:'fixture',caseHash:'fixture',shape,affine,frame:'RAS+',mri:new Float32Array(8000),compartments:[{name:'target',mask:target}]};
const frame=(removed,id)=>({scene:{sceneId:'fixture',shape,affine,resolutionMm:[1,1,1],regions:[
  {id:1,material:'tumor',name:'target',target:true}]},removed,moved:[],slots:{},stateId:id});
function harness(t){
  const element=Object.assign(new EventTarget(),{dataset:{},style:{},ownerDocument:new EventTarget()});
  element.getRootNode=()=>element.ownerDocument;
  const scene=new ResultInspectionScene(element,()=>{},()=>{});
  const reference=new THREE.Mesh(new THREE.BoxGeometry(8,8,8));
  t.after(()=>{scene.dispose();reference.geometry.dispose();reference.material.dispose();});
  return {scene,element,draw(view){
    scene.render({render(){}},modeledVolume(source,view),view,new Map([['target',reference]]),450,390);
  }};
}

test('result uses conditioned GPU buffers, opaque residual and one shared cast geometry across unchanged observations',t=>{
  const {scene,draw,element}=harness(t);
  const cells=[3350,3351,3370,3371,3750,3751,3770,3771];
  draw(frame(cells,'one'));
  const geometry=scene.cavity.geometry;
  assert.equal(scene.buriedCast.geometry,geometry);
  assert(+element.dataset.resultCavityDisplacementMm>0&&+element.dataset.resultCavityDisplacementMm<=.5);
  assert.notEqual(element.dataset.resultCavityBounds,element.dataset.resultCavitySourceBounds);
  assert.equal(scene.tissue.transparent,false);
  assert.equal(scene.buriedCast.visible,true);
  scene.configure('cavity');
  assert.equal(scene.buriedCast.visible,false);
  assert.equal(scene.residual.visible,false);
  draw(frame([...cells],'two'));
  assert.equal(scene.cavity.geometry,geometry);
  assert.equal(element.dataset.resultCavityCells,'8');
  assert.equal(element.dataset.resultState,'two');
  assert.equal(target.reduce((a,b)=>a+b,0),8000);
});

test('a new seek retires settled-cavity work and a late response cannot replace the new geometry',t=>{
  t.mock.timers.enable({apis:['setTimeout']});
  const workers=[];
  const old=globalThis.Worker;
  globalThis.Worker=class {
    constructor(){workers.push(this);}
    postMessage(request){this.request=request;}
    terminate(){this.terminated=true;}
  };
  t.after(()=>{if(old)globalThis.Worker=old;else delete globalThis.Worker;});
  const {scene,draw,element}=harness(t);
  const large=Array.from(target.keys()).filter(i=>Math.hypot(Math.floor(i/400)-10,Math.floor(i/20)%20-10,i%20-10)<8);
  const v=frame(large,'large');
  draw(v);
  assert.equal(element.dataset.resultCavityDisplacementMm,'0');
  t.mock.timers.tick(200);
  const first=workers[0],r=first.request;
  const finished=prepareSurface(r.mask,r.shape,r.affine,r.kind);
  draw(frame([3350],'small'));
  assert(first.terminated);
  const small=scene.cavity.geometry;
  first.onmessage({data:finished});
  assert.equal(scene.cavity.geometry,small);
  assert.equal(element.dataset.resultCavityCells,'1');
  draw(v);
  t.mock.timers.tick(200);
  workers[1].onmessage({data:finished});
  assert(workers[1].terminated);
  assert.deepEqual(scene.cavity.geometry.getAttribute('position').array,finished.smoothPositions);
  assert.deepEqual(scene.cavity.geometry.getAttribute('normal').array,finished.smoothNormals);
});
