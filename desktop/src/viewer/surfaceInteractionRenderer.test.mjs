import assert from 'node:assert/strict';
import {before,after,test} from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import * as THREE from 'three';
let server,cache,SurfaceInteractionRenderer,VesselSlabContext;
before(async()=>{
  cache=await fs.mkdtemp(path.join(os.tmpdir(),'surface-interaction-'));
  server=await createServer({root:fileURLToPath(new URL('../../',import.meta.url)),cacheDir:cache,logLevel:'error',optimizeDeps:{noDiscovery:true,include:[]},server:{middlewareMode:true,hmr:false,ws:false},appType:'custom'});
  SurfaceInteractionRenderer=(await server.ssrLoadModule('/src/viewer/surfaceInteractionRenderer.ts')).SurfaceInteractionRenderer;
  VesselSlabContext=(await server.ssrLoadModule('/src/viewer/vesselSlabContext.ts')).VesselSlabContext;
});

test('slab context reuses the visible vessel buffers and clears its depth before the foreground',()=>{
  const pass=new SurfaceInteractionRenderer(),scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera();
  const context=new VesselSlabContext(),original=new THREE.BoxGeometry(),updated=new THREE.SphereGeometry();
  const material=new THREE.MeshStandardMaterial({color:0xe9a174,opacity:.8}),mesh=new THREE.Mesh(original,material);
  const hidden=new THREE.Mesh(original,material),parent=new THREE.Group();
  hidden.visible=false;
  parent.position.set(1,2,3);mesh.position.set(3,4,5);parent.add(mesh,hidden);scene.add(parent);
  const planes=[new THREE.Plane(new THREE.Vector3(0,0,-1),-40)],calls=[];
  context.configure(2,-55,-43,planes);
  assert.equal(context.update([mesh,hidden]),true);
  const renderer={getPixelRatio:()=>1,setRenderTarget(){},setScissorTest(){},setViewport(){},clear(){},
    clearDepth:()=>calls.push('clear depth'),render:rendered=>calls.push(rendered)};
  for(const interactive of [false,true]){
    calls.length=0;
    pass.render(renderer,scene,camera,800,600,interactive,360,()=>{},context);
    assert.notEqual(calls[0],scene);
    assert.deepEqual(calls.slice(1,3),['clear depth',scene]);
    assert.equal(calls[0].children.length,1);
    const [background]=calls[0].children;
    assert.equal(background.geometry,mesh.geometry);
    assert.deepEqual(background.matrix.elements,mesh.matrixWorld.elements);
    assert.equal(background.material.clippingPlanes,planes);
    assert.deepEqual(background.material.uniforms.uRange.value.toArray(),[-55,-43]);
    assert.equal(background.material.uniforms.uAxis.value,2);
    assert.equal(background.material.uniforms.uOpacity.value,.8);
    mesh.geometry=updated;mesh.position.x++;
    assert.equal(context.update([mesh,hidden]),true);
  }
  let disposedGeometry=false;
  updated.addEventListener('dispose',()=>{disposedGeometry=true;});
  material.opacity=0;
  assert.equal(context.update([mesh,hidden]),false);
  assert.equal(calls[0].children.length,0);
  assert.equal(disposedGeometry,false);
  context.dispose();pass.dispose();original.dispose();updated.dispose();material.dispose();
});
after(async()=>{await server?.close();if(cache)await fs.rm(cache,{recursive:true,force:true});});
test('surface interaction changes pixel workload, retains every source mesh and returns to exact direct rendering on settle',()=>{
  const pass=new SurfaceInteractionRenderer(),scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(),plane=new THREE.Plane(new THREE.Vector3(1,0,0),-2);
  const geometry=new THREE.SphereGeometry(.1,8,8),material=new THREE.MeshBasicMaterial({opacity:.3,transparent:true,clippingPlanes:[plane]}),mesh=new THREE.Mesh(geometry,material);scene.add(mesh);
  const calls=[],targets=[],viewports=[];let restores=0;
  const renderer={getPixelRatio:()=>2,setRenderTarget:t=>targets.push(t),setScissorTest(){},setViewport:(...args)=>viewports.push(args),clear(){},render:(...args)=>calls.push(args)};
  const before=geometry.attributes.position.array.slice(),result=pass.render(renderer,scene,camera,800,600,true,360,()=>restores++);
  assert.deepEqual(result,{adaptive:true,width:360,height:270});assert.equal(calls[0][0],scene);assert.equal(calls[0][1],camera);
  assert.equal(mesh.visible,true);assert.equal(mesh.material,material);assert.equal(mesh.geometry,geometry);assert.equal(material.clippingPlanes[0],plane);assert.equal(material.opacity,.3);assert.deepEqual(geometry.attributes.position.array,before);
  assert.deepEqual(viewports[0],[0,0,180,135]);assert.equal(restores,1);const target=targets[0];
  pass.render(renderer,scene,camera,800,600,true,360,()=>{});assert.equal(targets[2],target,'target stays resident across interaction frames');
  calls.length=0;targets.length=0;
  assert.deepEqual(pass.render(renderer,scene,camera,800,600,false,360,()=>{}),{adaptive:false,width:1600,height:1200});
  assert.equal(calls.length,1);assert.deepEqual(calls[0],[scene,camera]);assert.equal(targets.length,0,'settled scene uses the original direct render path');
  pass.dispose();geometry.dispose();material.dispose();
});
