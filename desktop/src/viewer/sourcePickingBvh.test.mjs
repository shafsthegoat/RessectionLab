import assert from 'node:assert/strict';
import {before,after,test} from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import * as THREE from 'three';

let server,cache,SourcePickingBvh,buildSourcePickingBvh;
before(async()=>{
  cache=await fs.mkdtemp(path.join(os.tmpdir(),'source-bvh-'));
  server=await createServer({root:fileURLToPath(new URL('../../',import.meta.url)),cacheDir:cache,logLevel:'error',optimizeDeps:{noDiscovery:true,include:[]},server:{middlewareMode:true,hmr:false,ws:false},appType:'custom'});
  ({SourcePickingBvh}=await server.ssrLoadModule('/src/viewer/sourcePickingBvh.ts'));
  ({buildSourcePickingBvh}=await server.ssrLoadModule('/src/viewer/sourcePickingBvhBuild.ts'));
});
after(async()=>{await server?.close();if(cache)await fs.rm(cache,{recursive:true,force:true});});

function fixture(options={}){
  const workers=[],events=[];
  class Worker {
    messages=[];terminated=false;
    postMessage(input,transfer){this.messages.push(structuredClone(input,{transfer}));}
    terminate(){this.terminated=true;}
    complete(){const input=this.messages.shift();this.onmessage({data:buildSourcePickingBvh(input)});}
  }
  const owner=new SourcePickingBvh({...options,onEvent:e=>{events.push(e);options.onEvent?.(e);},createWorker:()=>{const worker=new Worker();workers.push(worker);return worker;}});
  return {owner,workers,events};
}
function mesh(geometry=new THREE.BoxGeometry(2,2,2)){return new THREE.Mesh(geometry,new THREE.MeshBasicMaterial({side:THREE.DoubleSide}));}
function hits(m,ray,reference=false){const out=[];(reference?THREE.Mesh.prototype.raycast:m.raycast).call(m,ray,out);return out.sort((a,b)=>a.distance-b.distance||a.faceIndex-b.faceIndex);}
function equalHits(a,b){assert.equal(a.length,b.length);for(let i=0;i<a.length;i++){assert.equal(a[i].faceIndex,b[i].faceIndex);assert.equal(a[i].face.materialIndex,b[i].face.materialIndex);assert(a[i].point.distanceTo(b[i].point)<1e-9);assert(Math.abs(a[i].distance-b[i].distance)<1e-9);}}

test('transient playback surfaces defer acceleration until demanded, preserve first-hit results and release unpicked bindings',()=>{
  const {owner,workers}=fixture(),a=mesh(),neverPicked=mesh();a.updateMatrixWorld(true);
  const ray=new THREE.Raycaster(new THREE.Vector3(.1,.2,8),new THREE.Vector3(0,0,-1));
  owner.install(a,a.geometry,false);owner.install(neverPicked,neverPicked.geometry,false);
  assert.equal(workers.length,0);assert.equal(owner.stats().copyBytes,0);assert.equal(owner.stats().buildsRequested,0);
  neverPicked.geometry.dispose();assert.equal(neverPicked.raycast,THREE.Mesh.prototype.raycast);
  equalHits(hits(a,ray),hits(a,ray,true));assert.equal(workers.length,1);assert.equal(owner.stats().buildsRequested,1);
  for(let i=0;i<4;i++)equalHits(hits(a,ray),hits(a,ray,true));
  assert.equal(owner.stats().buildsRequested,1);workers[0].complete();equalHits(hits(a,ray),hits(a,ray,true));
  assert.equal(owner.stats().acceleratedRaycasts,1);
  const boundary=mesh();owner.install(boundary,boundary.geometry,false);assert.equal(owner.stats().buildsRequested,1);
  owner.install(boundary,boundary.geometry);assert.equal(owner.stats().buildsRequested,2,'editable boundary may eagerly admit the same source geometry');
  owner.dispose();assert.equal(a.raycast,THREE.Mesh.prototype.raycast);assert.equal(boundary.raycast,THREE.Mesh.prototype.raycast);
});

test('one bounded worker copies only the active job, transfers disposable inputs, and preserves indices',()=>{
  const {owner,workers}=fixture(),a=mesh(),b=mesh(),positions=a.geometry.attributes.position.array,index=a.geometry.index.array;
  const initialPositions=positions.slice(),initialIndex=index.slice();owner.install(a,a.geometry);owner.install(b,b.geometry);
  assert.equal(workers.length,1);assert.equal(workers[0].messages.length,1);assert.equal(owner.stats().pendingGeometryRefs,1);
  assert.equal(owner.stats().copyBytes,positions.byteLength+index.byteLength);
  workers[0].complete();assert.equal(workers[0].messages.length,1);workers[0].complete();
  assert.equal(owner.stats().buildsCompleted,2);assert.equal(a.geometry.boundsTree.indirect,true);
  assert.equal(a.geometry.index.array,index);assert.deepEqual(index,initialIndex);assert.deepEqual(positions,initialPositions);
  owner.dispose();assert.equal(a.raycast,THREE.Mesh.prototype.raycast);assert.equal(a.geometry.boundsTree,undefined);
});
test('stable source IDs and unused historical positions preserve ordinary and accelerated triangle hits',()=>{
  const {owner,workers}=fixture(),reference=mesh(),source=reference.geometry,positions=source.attributes.position.array,normals=source.attributes.normal.array;
  const stable=new THREE.BufferGeometry(),p=new Float32Array(positions.length+3),n=new Float32Array(normals.length+3);p.set([1e6,-1e6,1e6]);p.set(positions,3);n.set(normals,3);
  stable.setAttribute('position',new THREE.BufferAttribute(p,3));stable.setAttribute('normal',new THREE.BufferAttribute(n,3));stable.setIndex(new THREE.BufferAttribute(Uint32Array.from(source.index.array,id=>id+1),1));
  source.computeBoundingBox();source.computeBoundingSphere();stable.boundingBox=source.boundingBox.clone();stable.boundingSphere=stable.boundingBox.getBoundingSphere(new THREE.Sphere());stable.userData.sourceIndexing='stable-source-ids';
  const actual=mesh(stable);actual.updateMatrixWorld(true);reference.updateMatrixWorld(true);const ray=new THREE.Raycaster(new THREE.Vector3(.1,.2,8),new THREE.Vector3(0,0,-1));
  equalHits(hits(actual,ray,true),hits(reference,ray,true));owner.install(actual,stable);workers[0].complete();equalHits(hits(actual,ray),hits(reference,ray,true));
  assert.equal(stable.attributes.position.array,p);assert.equal(stable.index.array[0],source.index.array[0]+1);owner.dispose();actual.geometry.dispose();reference.geometry.dispose();
});

test('all source triangle IDs and physical hits match ordinary picking under affine transforms and deeper clipping',()=>{
  const {owner,workers}=fixture(),a=mesh(),b=mesh();b.position.z=-4;
  const parent=new THREE.Group();parent.add(a,b);parent.position.set(11,-7,23);parent.rotation.set(.2,.3,-.4);parent.scale.set(1.7,.8,2.1);parent.updateMatrixWorld(true);
  const origin=new THREE.Vector3(.17,.23,10).applyMatrix4(parent.matrixWorld),direction=new THREE.Vector3(0,0,-1).transformDirection(parent.matrixWorld);
  const ray=new THREE.Raycaster(origin,direction,0,100);ray.firstHitOnly=true;
  owner.install(a,a.geometry);owner.install(b,b.geometry);workers[0].complete();workers[0].complete();
  const accelerated=[...hits(a,ray),...hits(b,ray)].sort((a,b)=>a.distance-b.distance),reference=[...hits(a,ray,true),...hits(b,ray,true)].sort((a,b)=>a.distance-b.distance);
  equalHits(accelerated,reference);assert.equal(ray.firstHitOnly,true);assert(accelerated.length>=4);
  const clip=new THREE.Plane().setFromNormalAndCoplanarPoint(direction.clone(),new THREE.Vector3(0,0,-2).applyMatrix4(parent.matrixWorld));
  const selected=accelerated.find(h=>clip.distanceToPoint(h.point)>=0),expected=reference.find(h=>clip.distanceToPoint(h.point)>=0);
  assert(selected&&expected);assert.equal(selected.object,b);assert(selected.point.distanceTo(expected.point)<1e-9);owner.dispose();
});

test('fallback handles pending trees, smoothed display geometry, version changes and nonindexed source data',()=>{
  const {owner,workers}=fixture(),a=mesh(),source=a.geometry;const ray=new THREE.Raycaster(new THREE.Vector3(.1,.2,8),new THREE.Vector3(0,0,-1));a.updateMatrixWorld(true);
  owner.install(a,source);equalHits(hits(a,ray),hits(a,ray,true));workers[0].complete();
  const smooth=source.clone();smooth.scale(.8,.8,.8);a.geometry=smooth;equalHits(hits(a,ray),hits(a,ray,true));
  a.geometry=source;source.attributes.position.setZ(0,1.1);source.attributes.position.needsUpdate=true;equalHits(hits(a,ray),hits(a,ray,true));
  const b=mesh(new THREE.BoxGeometry().toNonIndexed());b.updateMatrixWorld(true);owner.install(b,b.geometry);workers[0].complete();equalHits(hits(b,ray),hits(b,ray,true));
  assert.equal(b.geometry.index,null);assert(owner.stats().ordinaryRaycasts>=3);owner.dispose();
});

test('sheared source affine uses exact world near/far limits and restores the caller raycaster',()=>{
  const {owner,workers}=fixture(),a=mesh();a.matrixAutoUpdate=false;
  a.matrix.set(1,.7,.2,101,0,1,.3,-72,0,0,1,37,0,0,0,1);a.updateMatrixWorld(true);
  const origin=new THREE.Vector3(3,2,6).applyMatrix4(a.matrixWorld),target=new THREE.Vector3(0,.1,0).applyMatrix4(a.matrixWorld);
  const ray=new THREE.Raycaster(origin,target.sub(origin).normalize());const all=hits(a,ray,true);assert(all.length>0);
  ray.near=all[0].distance-.0001;ray.far=all[0].distance+.0001;const near=ray.near,far=ray.far;
  owner.install(a,a.geometry);workers[0].complete();equalHits(hits(a,ray),hits(a,ray,true));
  assert.equal(ray.near,near);assert.equal(ray.far,far);assert.equal(ray.firstHitOnly,undefined);owner.dispose();
});

test('overlapping grouped materials keep ordinary hit identity and downstream opacity selection',()=>{
  const {owner,workers}=fixture(),a=mesh(),b=mesh();b.position.z=-3;
  a.material=Array.from({length:6},(_,i)=>new THREE.MeshBasicMaterial({side:THREE.DoubleSide,transparent:true,opacity:i===4?0:.5}));
  a.position.set(.1,0,0);a.updateMatrixWorld(true);b.updateMatrixWorld(true);
  const ray=new THREE.Raycaster(new THREE.Vector3(.17,.23,8),new THREE.Vector3(0,0,-1));
  owner.install(a,a.geometry);owner.install(b,b.geometry);workers[0].complete();workers[0].complete();
  const accelerated=[...hits(a,ray),...hits(b,ray)].sort((a,b)=>a.distance-b.distance),reference=[...hits(a,ray,true),...hits(b,ray,true)].sort((a,b)=>a.distance-b.distance);
  equalHits(accelerated,reference);
  const visible=h=>{const material=Array.isArray(h.object.material)?h.object.material[h.face.materialIndex]:h.object.material;return material.visible&&material.opacity>0;};
  const picked=accelerated.find(visible),expected=reference.find(visible);assert(picked&&expected);assert.equal(picked.faceIndex,expected.faceIndex);assert(picked.point.distanceTo(expected.point)<1e-9);owner.dispose();
});

test('retired and stale responses never attach trees, and failures release the queue with ordinary fallback',()=>{
  const {owner,workers}=fixture(),a=mesh(),b=mesh();owner.install(a,a.geometry);owner.install(b,b.geometry);
  a.geometry.dispose();assert.equal(a.raycast,THREE.Mesh.prototype.raycast);workers[0].complete();assert.equal(a.geometry.boundsTree,undefined);workers[0].complete();assert(b.geometry.boundsTree);
  const c=mesh(),d=mesh();owner.install(c,c.geometry);owner.install(d,d.geometry);workers[0].onerror();assert.equal(workers[0].terminated,true);assert.equal(c.geometry.boundsTree,undefined);
  assert.equal(workers.length,2);workers[1].complete();assert(d.geometry.boundsTree);
  const changed=mesh();owner.install(changed,changed.geometry);changed.geometry.attributes.position.needsUpdate=true;workers[1].complete();assert.equal(changed.geometry.boundsTree,undefined);
  owner.dispose();assert.equal(owner.stats().retainedRootBytes,0);
});

test('root/input/count caps bound retained memory and keep ordinary fallback available',()=>{
  const small=fixture({maxInputBytes:1}),a=mesh();small.owner.install(a,a.geometry);assert.equal(small.workers.length,0);assert.equal(a.geometry.boundsTree,undefined);small.owner.dispose();
  const roots=fixture({maxRetainedBytes:1}),oversized=mesh();roots.owner.install(oversized,oversized.geometry);roots.workers[0].complete();assert.equal(oversized.geometry.boundsTree,undefined);assert.equal(roots.owner.stats().retainedRootBytes,0);roots.owner.dispose();
  const {owner,workers}=fixture({maxGeometries:1,maxPending:2}),b=mesh(),c=mesh(),d=mesh();owner.install(b,b.geometry);owner.install(c,c.geometry);owner.install(d,d.geometry);
  assert.equal(owner.stats().pendingGeometryRefs,1);workers[0].complete();workers[0].complete();assert.equal(owner.stats().retainedGeometries,1);assert.equal(b.geometry.boundsTree,undefined);assert.equal(owner.stats().evictions,1);
  assert(owner.stats().retainedRootBytes<=owner.stats().maxRetainedBytes);owner.dispose();
});

test('LRU-evicted source trees re-admit on picking or reinstall without duplicate in-flight jobs',()=>{
  const {owner,workers}=fixture({maxGeometries:1}),a=mesh(),b=mesh();a.updateMatrixWorld(true);b.updateMatrixWorld(true);
  const ray=new THREE.Raycaster(new THREE.Vector3(.1,.2,8),new THREE.Vector3(0,0,-1));
  owner.install(a,a.geometry);workers[0].complete();owner.install(b,b.geometry);workers[0].complete();
  assert.equal(a.geometry.boundsTree,undefined);assert(b.geometry.boundsTree);
  const ordinary=owner.stats().ordinaryRaycasts,accelerated=owner.stats().acceleratedRaycasts;
  equalHits(hits(a,ray),hits(a,ray,true));
  assert.equal(owner.stats().ordinaryRaycasts,ordinary+1);assert.equal(owner.stats().acceleratedRaycasts,accelerated);
  assert.equal(owner.stats().buildsRequested,3);assert.equal(workers[0].messages.length,1);
  for(let i=0;i<4;i++)equalHits(hits(a,ray),hits(a,ray,true));
  owner.install(a,a.geometry);assert.equal(owner.stats().buildsRequested,3);assert.equal(workers[0].messages.length,1);
  workers[0].complete();equalHits(hits(a,ray),hits(a,ray,true));assert.equal(owner.stats().acceleratedRaycasts,accelerated+1);
  assert.equal(b.geometry.boundsTree,undefined);owner.install(b,b.geometry);owner.install(b,b.geometry);
  assert.equal(owner.stats().buildsRequested,4);assert.equal(workers[0].messages.length,1);
  workers[0].complete();assert(b.geometry.boundsTree);assert.equal(owner.stats().retainedGeometries,1);
  assert(owner.stats().retainedRootBytes<=owner.stats().maxRetainedBytes);owner.dispose();
});

test('worker failures and over-budget results stay fallback while an evicted source respects the queue cap',()=>{
  const ray=new THREE.Raycaster(new THREE.Vector3(.1,.2,8),new THREE.Vector3(0,0,-1));
  for(const kind of ['error','oversized']){
    const {owner,workers}=fixture(kind==='oversized'?{maxRetainedBytes:1}:{}),a=mesh();a.updateMatrixWorld(true);owner.install(a,a.geometry);
    if(kind==='error')workers[0].onerror();else workers[0].complete();
    for(let i=0;i<4;i++){equalHits(hits(a,ray),hits(a,ray,true));owner.install(a,a.geometry);}
    assert.equal(owner.stats().buildsRequested,1);assert.equal(owner.stats().activeJobs,0);assert.equal(owner.stats().pendingGeometryRefs,0);owner.dispose();
  }
  const {owner,workers}=fixture({maxGeometries:1,maxPending:2}),a=mesh(),b=mesh(),c=mesh(),d=mesh();a.updateMatrixWorld(true);
  owner.install(a,a.geometry);workers[0].complete();owner.install(b,b.geometry);workers[0].complete();
  b.geometry.dispose();
  owner.install(c,c.geometry);owner.install(d,d.geometry);const requested=owner.stats().buildsRequested;
  equalHits(hits(a,ray),hits(a,ray,true));assert.equal(owner.stats().buildsRequested,requested);assert.equal(owner.stats().pendingGeometryRefs,1);
  workers[0].complete();equalHits(hits(a,ray),hits(a,ray,true));assert.equal(owner.stats().buildsRequested,requested+1);assert.equal(owner.stats().pendingGeometryRefs,1);
  workers[0].complete();workers[0].complete();assert(a.geometry.boundsTree);assert.equal(owner.stats().retainedGeometries,1);owner.dispose();
});

test('every received active-job completion publishes settled worker/queue telemetry, including stale and capped results',()=>{
  for(const kind of ['ready','stale','capped','error']){
    const snapshots=[];
    const {owner,workers}=fixture({...kind==='capped'?{maxRetainedBytes:1}:{},onEvent:event=>{
      if(event.kind==='source-bvh-complete')snapshots.push({status:event.status,...owner.stats()});
    }}),a=mesh(),b=mesh();owner.install(a,a.geometry);owner.install(b,b.geometry);
    if(kind==='stale')a.geometry.dispose();
    if(kind==='error'){const input=workers[0].messages.shift();workers[0].onmessage({data:{id:input.id,error:'build failed'}});}
    else workers[0].complete();
    assert.equal(snapshots.length,1);assert.equal(snapshots[0].activeJobs,1);assert.equal(snapshots[0].pendingGeometryRefs,0);
    assert.equal(snapshots[0].status,kind==='ready'?'ready':'fallback');
    workers[0].complete();assert.equal(snapshots.length,2);assert.equal(snapshots[1].activeJobs,0);assert.equal(snapshots[1].pendingGeometryRefs,0);
    assert.equal(owner.stats().activeJobs,0);owner.dispose();
  }
});
