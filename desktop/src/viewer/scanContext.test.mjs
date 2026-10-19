import test from 'node:test';
import assert from 'node:assert/strict';
import {registerHooks,stripTypeScriptTypes} from 'node:module';
import {readFileSync} from 'node:fs';
import * as THREE from 'three';
import {packAnatomyLabels} from './labelPacking.ts';

// The production bundler resolves extensionless viewer imports. Resolve the
// same local TS modules for this constructor/uniform integration test.
registerHooks({resolve(specifier,context,nextResolve){
  try{return nextResolve(specifier,context)}catch(error){
    if(specifier.startsWith('.')&&!/\.[a-z]+$/i.test(specifier))return nextResolve(specifier+'.ts',context);
    throw error;
  }
},load(url,context,nextLoad){
  if(url.endsWith('/ScanVolumeRenderer.ts'))return {format:'module',source:stripTypeScriptTypes(readFileSync(new URL(url),'utf8'),{mode:'transform',sourceUrl:url}),shortCircuit:true};
  return nextLoad(url,context);
}});
const {ScanVolumeRenderer}=await import('./ScanVolumeRenderer.ts');
const identity=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];

test('CT context uniforms retain a hidden source label at index127 and its exact packed membership',()=>{
  const atlas=Uint16Array.from([1,100]),bone=Uint8Array.from([1,0]);
  const compartments=Array.from({length:127},(_,i)=>({name:'region'+i,kind:i===99?'artery':'brain-region',labelSet:'atlas',labelValue:i+1,mask:new Uint8Array(),color:'#ad4730'}));
  compartments.push({name:'source-bone',kind:'bone',mask:bone,color:'#ece2ca'});
  const volume={shape:[2,1,1],affine:identity,frame:'RAS+',mri:Float32Array.from([1000,1000]),compartments,labelSets:[{id:'atlas',data:atlas}]};
  const source={bone:[...bone],atlas:[...atlas],mri:[...volume.mri]},packed=packAnatomyLabels(volume);
  const renderer=new ScanVolumeRenderer({extensions:{has:()=>false}},volume,[0,1200],packed.codes);
  const texture=new THREE.Data3DTexture(packed.data,1,1,2);
  try{
    renderer.configure({preset:'ct-angiography',density:.14},'CTA',null);
    renderer.labels(texture,{'source-bone':false},{skullMode:'exterior',layerColor:{'source-bone':'#ece2ca'}},[]);
    const uniforms=renderer.material.uniforms;
    assert.equal(uniforms.uContextLayerCount.value,1,'A hidden declared label must disable the intensity-only fallback');
    assert.equal(uniforms.uContextLayers.value[0],127);
    assert.equal(uniforms.uLayerEnabled.value[127],0);
    const code=uniforms.uLayerCodes.value[127];
    assert.equal(code.x,1);assert.equal(code.y,-1);
    assert.equal(packed.data[code.x]&(-code.y),1,'Bone source membership is present at the first voxel');
    assert.equal(packed.data[4+code.x]&(-code.y),0,'Equal-HU arterial voxel is outside source bone');
    const arteryCode=uniforms.uLayerCodes.value[99];
    assert.equal(packed.data[4+arteryCode.x],arteryCode.y,'Categorical arterial label remains independently addressable');
    renderer.labels(texture,{}, {skullMode:'transparent',layerOpacity:{'source-bone':.035},layerColor:{'source-bone':'#ece2ca'}},[]);
    const ivory=new THREE.Color('#ece2ca'),color=uniforms.uLayerColor.value[127];
    assert.deepEqual([color.x,color.y,color.z],[ivory.r,ivory.g,ivory.b]);
    assert.equal(color.w,.035);assert.equal(uniforms.uSkullMode.value,2);
    renderer.setLayers(compartments.slice(0,127),packed.codes);
    renderer.labels(texture,{}, {skullMode:'exterior'},[]);
    assert.equal(uniforms.uContextLayerCount.value,0,'Threshold fallback is available only without a declared context layer');
    assert.deepEqual({bone:[...bone],atlas:[...atlas],mri:[...volume.mri]},source);
  }finally{renderer.dispose();texture.dispose();}
});
