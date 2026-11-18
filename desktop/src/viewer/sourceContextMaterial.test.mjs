import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {applySourceContextMaterial,sourceContextOpacity} from './sourceContextMaterial.ts';
import {SurfaceVolumePass} from './SurfaceVolumePass.ts';
import {installNativeContextLighting,setNativeContextLighting} from './nativeContextLighting.ts';

test('shell context retains the source silhouette and configured opacity without filling the interior',()=>{
  assert.equal(sourceContextOpacity(.2,0),.2);
  assert.equal(sourceContextOpacity(.2,1),.012);
  assert.equal(sourceContextOpacity(.2,-1),.012,'front/back source boundaries have the same response');
  assert.ok(sourceContextOpacity(.2,.5)<.05);
  assert.equal(sourceContextOpacity(1,1),1,'explicit opaque context stays solid');
});

test('source material style toggles one uniform and survives transparent peel compilation',()=>{
  const material=new THREE.MeshStandardMaterial({opacity:.2,transparent:true});
  const color=material.color.clone(),opacity=material.opacity;
  applySourceContextMaterial(material,true);
  const shader={uniforms:{},fragmentShader:THREE.ShaderLib.standard.fragmentShader};
  material.onBeforeCompile(shader,{});
  assert.equal(shader.uniforms.uSourceContextShell.value,true);
  assert.match(shader.fragmentShader,/contextFacing=abs\(dot\(normal,normalize\(vViewPosition\)\)\)/);
  applySourceContextMaterial(material,false);
  assert.equal(shader.uniforms.uSourceContextShell.value,false);
  assert.equal(material.opacity,opacity);assert.deepEqual(material.color,color);
  const pass=new SurfaceVolumePass(),peel=pass.make(material,false),tail=pass.make(material,true);
  try{
    for(const clone of [peel,tail]){
      const peelShader={uniforms:{},fragmentShader:THREE.ShaderLib.standard.fragmentShader};
      clone.onBeforeCompile(peelShader,{});
      assert.equal(peelShader.uniforms.uSourceContextShell,shader.uniforms.uSourceContextShell);
      assert.match(peelShader.fragmentShader,/uPreviousSurface/);
      assert.match(clone.customProgramCacheKey(),/source-context-shell/);
    }
    applySourceContextMaterial(material,true);
    assert.equal(shader.uniforms.uSourceContextShell.value,true);
  }finally{peel.dispose();tail.dispose();pass.dispose();material.dispose();}
});


test('native context lighting uses physical view depth and restores the original programs in volume peels',()=>{
  const material=new THREE.MeshStandardMaterial({
    color:'#ece2ca', opacity:.35, transparent:true,
    clippingPlanes:[new THREE.Plane(new THREE.Vector3(0,0,1),2)],
  });
  const {opacity,transparent,depthWrite,clippingPlanes}=material;
  const color=material.color.clone();
  const compile=(m)=>{
    const shader={uniforms:{},fragmentShader:THREE.ShaderLib.standard.fragmentShader};
    m.onBeforeCompile(shader,{});
    return shader;
  };
  applySourceContextMaterial(material,true);
  const original=compile(material);
  const originalKey=material.customProgramCacheKey();
  installNativeContextLighting(material,[[40,-60,-22],[60,-20,18]],true);
  const direct=compile(material);
  assert.deepEqual(direct.uniforms.uNativeContextCenter.value.toArray(),[50,-40,-2]);
  assert.equal(direct.uniforms.uNativeContextRadius.value,30);
  assert.match(direct.fragmentShader,/dot\(normal, normalize\(vec3\(-0\.46, 0\.55, 0\.70\)\)\)/);
  assert.match(direct.fragmentShader,/vViewPosition\.z - centerDepth/);
  assert.match(direct.fragmentShader,/outgoingLight = diffuseColor\.rgb/);
  assert.match(direct.fragmentShader,/uSourceContextShell/);
  const scene=new THREE.Scene(), geometry=new THREE.PlaneGeometry(), mesh=new THREE.Mesh(geometry,material);
  mesh.userData.compartment='brain';scene.add(mesh);
  const volume=new SurfaceVolumePass(), camera=new THREE.Camera(), opaque=new THREE.Texture();
  const materials=new Set();
  const renderer={setRenderTarget(){},clear(){},render(){materials.add(mesh.material);}};
  try{
    volume.render(renderer,scene,camera,opaque,4,4,true);
    assert.equal(materials.size,2);
    const keys=new Map([...materials].map(clone=>[clone,{key:clone.customProgramCacheKey(),version:clone.version}]));
    for(const clone of materials)assert.match(compile(clone).fragmentShader,/uNativeContextCenter/);
    setNativeContextLighting(material,false);
    volume.render(renderer,scene,camera,opaque,4,4,true);
    assert.equal(materials.size,2);
    assert.deepEqual(compile(material),original);
    assert.equal(material.customProgramCacheKey(),originalKey);
    for(const clone of materials){
      assert.notEqual(clone.customProgramCacheKey(),keys.get(clone).key);
      assert.ok(clone.version > keys.get(clone).version);
      assert.doesNotMatch(compile(clone).fragmentShader,/uNativeContextCenter/);
    }
    assert.deepEqual(material.color,color);
    assert.deepEqual({opacity,transparent,depthWrite,clippingPlanes},{
      opacity:material.opacity, transparent:material.transparent, depthWrite:material.depthWrite, clippingPlanes:material.clippingPlanes,
    });
  }finally{geometry.dispose();opaque.dispose();volume.dispose();material.dispose();}
});
