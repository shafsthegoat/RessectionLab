import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {applyStructureSurface,defaultStructureSurfaceStyle,glassRimFactor,GLASS_PROFILES,layerSurfaceStyle,
  structureSurfaceAlpha,structureSurfaceDisplay,structureSurfaceUniforms,structureTier,withLayerSurfaceStyle} from './structureGlass.ts';
import {applySourceContextMaterial} from './sourceContextMaterial.ts';
import {installNativeContextLighting} from './nativeContextLighting.ts';
import {SurfaceVolumePass} from './SurfaceVolumePass.ts';

const close=(a,b,message)=>assert.ok(Math.abs(a-b)<1e-9,`${message??''} ${a} != ${b}`);

test('rim factor is the Fresnel term pow(1-|n·v|, 2.5) for front and back faces',()=>{
  assert.equal(glassRimFactor(1),0);
  assert.equal(glassRimFactor(-1),0);
  assert.equal(glassRimFactor(0),1);
  close(glassRimFactor(.5),.5**2.5);
  close(glassRimFactor(-.5),glassRimFactor(.5));
  assert.equal(glassRimFactor(1.2),0,'normalisation error never produces a negative base');
  assert.ok(glassRimFactor(.2)>glassRimFactor(.6));
});

test('default styles: targets and context shells are glass; vessels, cortex and unknown kinds stay solid',()=>{
  for(const kind of ['tumor','enhancing-tumor','tumor-core','edema','lesion-context'])assert.equal(structureTier(kind),'target');
  for(const kind of ['tumor','ventricles','brain','head-envelope','outer-head-tissue','skull','bone','scalp','dura','csf','brain-region'])
    assert.equal(defaultStructureSurfaceStyle(kind),'glass',kind);
  for(const kind of ['artery','vein','sinus','vessel','vessels','gray-matter','cerebral-cortex','white-matter',undefined,'','label'])
    assert.equal(defaultStructureSurfaceStyle(kind),'solid',String(kind));
});

test('saved style wins; a missing or unrecognised field keeps the kind default (older workspaces)',()=>{
  assert.equal(layerSurfaceStyle('lesion','tumor',{}),'glass');
  assert.equal(layerSurfaceStyle('lesion','tumor',{layerOpacity:{lesion:.5}}),'glass');
  assert.equal(layerSurfaceStyle('lesion','tumor',{layerSurfaceStyle:{other:'solid'}}),'glass');
  assert.equal(layerSurfaceStyle('lesion','tumor',{layerSurfaceStyle:{lesion:'solid'}}),'solid');
  assert.equal(layerSurfaceStyle('mca','artery',{layerSurfaceStyle:{mca:'glass'}}),'glass');
  assert.equal(layerSurfaceStyle('lesion','tumor',{layerSurfaceStyle:{lesion:'matte'}}),'glass');
  const settings={layerSurfaceStyle:{a:'solid'},layerOpacity:{a:.4}};
  const next=withLayerSurfaceStyle(settings,'b','glass');
  assert.deepEqual(next.layerSurfaceStyle,{a:'solid',b:'glass'});
  assert.deepEqual(settings.layerSurfaceStyle,{a:'solid'},'settings are not mutated');
  assert.equal(next.layerOpacity,settings.layerOpacity);
});

test('an opaque target becomes a glass shell whose rim keeps the configured opacity',()=>{
  const d=structureSurfaceDisplay('lesion','tumor',1,{});
  assert.equal(d.glass,true);
  close(d.materialOpacity,GLASS_PROFILES.target.interior);
  assert.ok(d.materialOpacity<1,'translucent pass, no depth write');
  assert.equal(d.edgeAlpha,1);
  close(structureSurfaceAlpha(d,1),GLASS_PROFILES.target.interior);
  close(structureSurfaceAlpha(d,0),1);
  assert.ok(structureSurfaceAlpha(d,.3)>structureSurfaceAlpha(d,.8));
  assert.equal(d.glow,GLASS_PROFILES.target.glow);
  assert.equal(d.exponent,2.5);
  const faint=structureSurfaceDisplay('edema','edema',.13,{});
  close(faint.materialOpacity,.13*GLASS_PROFILES.target.interior);
  close(faint.edgeAlpha,.13*GLASS_PROFILES.target.edge);
});

test('context shells: subtle glass below full opacity, opaque choices stay solid',()=>{
  const opaqueSkull=structureSurfaceDisplay('skull','skull',1,{skullMode:'exterior'});
  assert.equal(opaqueSkull.glass,false);
  assert.equal(opaqueSkull.materialOpacity,1);
  assert.equal(structureSurfaceAlpha(opaqueSkull,0),1);
  const brain=structureSurfaceDisplay('brain','brain',.2,{});
  assert.equal(brain.glass,true);
  close(brain.materialOpacity,.2*GLASS_PROFILES.shell.interior);
  close(brain.edgeAlpha,.2*GLASS_PROFILES.shell.edge);
  assert.ok(brain.glow<GLASS_PROFILES.target.glow);
  // The source context shell already lowers facing alpha; glass then only adds the rim.
  const shell=structureSurfaceDisplay('brain','brain',.2,{},{contextShell:true});
  assert.equal(shell.materialOpacity,.2);
  const hidden=structureSurfaceDisplay('brain','brain',0,{});
  assert.equal(hidden.glass,false);assert.equal(hidden.materialOpacity,0);
});

test('solid choice, vessels, operative views and modeled tissue keep the configured opacity',()=>{
  const solid=structureSurfaceDisplay('lesion','tumor',1,{layerSurfaceStyle:{lesion:'solid'}});
  assert.deepEqual([solid.glass,solid.materialOpacity,solid.edgeAlpha,solid.glow],[false,1,1,0]);
  const vessel=structureSurfaceDisplay('mca','artery',1,{});
  assert.deepEqual([vessel.glass,vessel.materialOpacity,vessel.cavity,vessel.rimLight],[false,1,0,0]);
  for(const settings of [{operativeView:'surgeon'},{operativeView:'external'}]){
    const d=structureSurfaceDisplay('lesion','tumor',1,settings);
    assert.deepEqual([d.glass,d.materialOpacity,d.cavity,d.rimLight],[false,1,0,0]);
  }
  const modeled=structureSurfaceDisplay('lesion','tumor',.5,{},{modeled:true});
  assert.deepEqual([modeled.glass,modeled.materialOpacity,modeled.cavity],[false,.5,0]);
});

test('transparent draw order composites shells before inner structures and the target',()=>{
  const order=kind=>structureSurfaceDisplay('x',kind,.5,{}).renderOrder;
  assert.ok(order('outer-head-tissue')<order('brain'));
  assert.ok(order('brain')<order('ventricles'));
  assert.ok(order('ventricles')<order('tumor'));
  assert.ok(order('tumor')<=order('artery'));
});

test('glass shader shares uniforms with volume peels and leaves vessels compiled verbatim',()=>{
  const compile=material=>{
    const shader={uniforms:{},fragmentShader:THREE.ShaderLib.standard.fragmentShader};
    material.onBeforeCompile(shader,{});
    return shader;
  };
  const vessel=new THREE.MeshStandardMaterial(),vesselKey=vessel.customProgramCacheKey(),vesselShader=compile(vessel);
  applyStructureSurface(vessel,structureSurfaceDisplay('mca','artery',1,{}));
  assert.equal(structureSurfaceUniforms(vessel),undefined);
  assert.equal(vessel.customProgramCacheKey(),vesselKey);
  assert.deepEqual(compile(vessel),vesselShader);
  const modeled=new THREE.MeshStandardMaterial(),modeledKey=modeled.customProgramCacheKey();
  applyStructureSurface(modeled,structureSurfaceDisplay('brain','brain',.4,{},{modeled:true}));
  assert.equal(modeled.customProgramCacheKey(),modeledKey,'engine-modeled tissue keeps its program');
  // Created in an operative view: installed but inert, so native lighting can wrap it and Anatomy restores glass.
  const operative=new THREE.MeshStandardMaterial();
  applyStructureSurface(operative,structureSurfaceDisplay('lesion','tumor',1,{operativeView:'surgeon'}));
  assert.deepEqual(Object.values(structureSurfaceUniforms(operative)).map(u=>u.value),[false,1,2.5,0,0,0,0]);
  assert.equal(structureSurfaceDisplay('lesion','tumor',1,{},{nativeLit:true}).key,0,'native context lighting already supplies the view key');
  assert.ok(structureSurfaceDisplay('lesion','tumor',1,{}).key>0);

  const material=new THREE.MeshStandardMaterial({transparent:true,opacity:1});
  applySourceContextMaterial(material,false);
  const display=structureSurfaceDisplay('lesion','tumor',1,{});
  material.opacity=display.materialOpacity;
  applyStructureSurface(material,display);
  installNativeContextLighting(material,[[0,0,0],[10,10,10]],true);
  const shader=compile(material);
  assert.match(shader.fragmentShader,/#include <emissivemap_fragment>\s*\{/);
  assert.match(shader.fragmentShader,/diffuseColor\.a=mix\(diffuseColor\.a,uGlassEdgeAlpha,glassRim\)/);
  assert.match(shader.fragmentShader,/pow\(structureEdge,uGlassExponent\)/);
  assert.match(shader.fragmentShader,/dFdx\(normal\)/);
  assert.match(shader.fragmentShader,/uNativeContextCenter/);
  assert.ok(shader.fragmentShader.indexOf('uGlassEdgeAlpha,glassRim')<shader.fragmentShader.indexOf('#include <lights_fragment_begin>'),
    'glass terms precede lighting so later stages keep the emissive rim');
  assert.match(material.customProgramCacheKey(),/structure-glass-v1/);
  assert.equal(shader.uniforms.uGlassActive.value,true);
  assert.equal(shader.uniforms.uGlassEdgeAlpha.value,1);
  assert.ok(material.roughness<=.34);
  const pass=new SurfaceVolumePass(),peel=pass.make(material,false);
  try{
    const peelShader=compile(peel);
    assert.equal(peelShader.uniforms.uGlassEdgeAlpha,shader.uniforms.uGlassEdgeAlpha);
    applyStructureSurface(material,structureSurfaceDisplay('lesion','tumor',1,{layerSurfaceStyle:{lesion:'solid'}}));
    assert.equal(peelShader.uniforms.uGlassActive.value,false,'a toggle changes a uniform, not the program');
    assert.match(peel.customProgramCacheKey(),/structure-glass-v1/);
  }finally{peel.dispose();pass.dispose();material.dispose();vessel.dispose();}
});
