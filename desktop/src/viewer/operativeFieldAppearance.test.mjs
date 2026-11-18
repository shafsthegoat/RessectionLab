import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {operativeSignalTexture,operativeTissueMaterial} from './operativeFieldAppearance.ts';

test('C-order scan texture preserves acquired black, missing samples and flap coordinates',()=>{
  const signal={shape:[2,1,3],origin:[-1,2,7],spacingMm:[.5,1,2],
    unknownValue:0,voxels:Uint8Array.of(0,1,255,2,3,4)};
  const texture=operativeSignalTexture(signal);
  assert.deepEqual([texture.image.width,texture.image.height,texture.image.depth],[3,1,2]);
  assert.deepEqual([...texture.image.data],[0,0,1,255,255,255,2,255,3,255,4,255]);
  assert.equal(texture.minFilter,THREE.LinearFilter);
  const pivot=new THREE.Vector3(0,0,12);
  const section={value:4};
  const field={intensity:signal,origin:[-1,2,7],shape:[3,1,9],spacingMm:.5};
  const material=operativeTissueMaterial('boneFlap',field,texture,pivot,section);
  const shader={...THREE.ShaderLib.standard,uniforms:{}};
  material.onBeforeCompile(shader,{});
  assert.deepEqual(shader.uniforms.uFieldMin.value.toArray(),[-1.25,1.5,6]);
  assert.deepEqual(shader.uniforms.uFieldSize.value.toArray(),[1,1,6]);
  assert.deepEqual(shader.uniforms.uRestOffset.value.toArray(),pivot.toArray());
  assert.equal(shader.uniforms.uSectionY,section);
  assert.deepEqual(shader.uniforms.uCropMax.value.toArray(),[.25,2.25,11.25]);
  texture.dispose();material.dispose();
});
