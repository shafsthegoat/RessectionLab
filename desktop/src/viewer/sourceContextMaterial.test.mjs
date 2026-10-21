import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {applySourceContextMaterial,sourceContextOpacity} from './sourceContextMaterial.ts';
import {SurfaceVolumePass} from './SurfaceVolumePass.ts';

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
