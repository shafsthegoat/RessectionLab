import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {applySourceContextMaterial,sourceContextOpacity} from './sourceContextMaterial.ts';
import {SurfaceVolumePass} from './SurfaceVolumePass.ts';

test('head and cortex capture separate nearest depths with patient clipping during orbit and at rest',()=>{
  const pass=new SurfaceVolumePass(),scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera();
  const geometry=new THREE.PlaneGeometry(),opaque=new THREE.Texture();
  const clip=[new THREE.Plane(new THREE.Vector3(1,0,0),2)];
  const groups=['head','cortex','cortex',undefined];
  const meshes=groups.map((boundary,i)=>{
    const material=new THREE.MeshStandardMaterial({transparent:true,opacity:.5,clippingPlanes:clip});
    const mesh=new THREE.Mesh(geometry,material);
    mesh.userData={compartment:'layer'+i,cranialBoundary:boundary};
    scene.add(mesh);
    return mesh;
  });
  const source=meshes.map(m=>m.material),capture=[];
  const renderer={setRenderTarget(){},clear(){},render(){
    if(scene.overrideMaterial){
      assert.equal(scene.overrideMaterial.clippingPlanes,clip);
      capture.push(meshes.map(m=>m.visible));
    }else{
      for(const [i,mesh] of meshes.entries()){
        const shader={uniforms:{},fragmentShader:THREE.ShaderLib.standard.fragmentShader};
        mesh.material.onBeforeCompile(shader,renderer);
        assert.equal(shader.uniforms.uBoundaryDepth?.value,pass.cranial.targets.get(groups[i])?.depthTexture);
        if(groups[i]==='head')assert.match(shader.fragmentShader,/outgoingLight=diffuseColor\.rgb\*\([^;]+\)\+totalEmissiveRadiance;/);
      }
    }
  }};
  try{
    for(const interactive of [true,false]){
      pass.render(renderer,scene,camera,opaque,32,24,interactive);
      assert.deepEqual(capture.splice(0),[[true,false,false,false],[false,true,true,false]]);
      assert.deepEqual(meshes.map(m=>m.material),source);
      assert(meshes.every(m=>m.visible));
    }
    assert.notEqual(pass.cranial.targets.get('head'),pass.cranial.targets.get('cortex'));
  }finally{pass.dispose();source.forEach(m=>m.dispose());geometry.dispose();opaque.dispose();}
});
import {installNativeContextLighting,setNativeContextLighting} from './nativeContextLighting.ts';

test('shell context retains the source silhouette and configured opacity without filling the interior',()=>{
  assert.equal(sourceContextOpacity(.2,0),.2);
  assert.equal(sourceContextOpacity(.2,1),.012);
  assert.equal(sourceContextOpacity(.2,-1),.012,'front/back source boundaries have the same response');
  assert.ok(sourceContextOpacity(.2,.5)<.05);
  assert.equal(sourceContextOpacity(.52,1,true),.52*.48,'bone transmits enough front-face shading to show curvature');
  assert.equal(sourceContextOpacity(.52,0,true),.52);
  assert.equal(sourceContextOpacity(1,1,true),1);
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
  assert.match(direct.fragmentShader,/outgoingLight = diffuseColor\.rgb \* diffuseLighting \+ totalEmissiveRadiance;/);
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

test('cached transparent peels and sorting tail follow the current selection through A, B and clear',()=>{
  const pass=new SurfaceVolumePass(),scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera();
  const geometry=new THREE.PlaneGeometry(),opaque=new THREE.Texture();
  const meshes=['#cc6677','#77aadd'].map((color,index)=>{
    const mesh=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color,transparent:true,opacity:.5}));
    mesh.userData.compartment=String(index);scene.add(mesh);
    return mesh;
  });
  const sources=meshes.map(mesh=>mesh.material),pairs=meshes.map(()=>new Map());
  const renderer={setRenderTarget(){},clear(){},render(){
    meshes.forEach(({material},i)=>{
      const source=sources[i];
      assert.notEqual(material,source);
      assert.ok(material.emissive.equals(source.emissive),'emissive color must follow the current source');
      assert.equal(material.emissiveIntensity,source.emissiveIntensity);
      const first=pairs[i].get(material.transparent);
      if(first){
        assert.equal(material,first.material,'reuse the cached material');
        assert.equal(material.version,first.version,'selection changes uniforms only');
      }else pairs[i].set(material.transparent,{material,version:material.version});
    });
  }};
  try{
    for(const selected of [null,0,1,null]){
      sources.forEach((material,i)=>{
        material.emissive.set(i===selected?material.color:0);
        material.emissiveIntensity=i===selected?.35:0;
      });
      for(const interactive of [false,true])pass.render(renderer,scene,camera,opaque,8,8,interactive);
      assert.deepEqual(meshes.map(mesh=>mesh.material),sources);
    }
    for(const pair of pairs)assert.deepEqual([...pair.keys()],[false,true]);
  }finally{sources.forEach(m=>m.dispose());geometry.dispose();opaque.dispose();pass.dispose();}
});
