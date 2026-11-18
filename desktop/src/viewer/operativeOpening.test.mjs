import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {buildOperativeModel} from './operativeFieldModel.ts';
import {prepareOperativeOpening} from './operativeOpening.ts';
import {operativeSurface} from './operativeSurface.ts';
import {operativeTissueMaterial,operativeSignalTexture} from './operativeFieldAppearance.ts';

function triangles(geometry){
  const positions=geometry.getAttribute('position'),index=geometry.index,faces=[];
  for(let i=0;i<index.count;i+=3){
    const vertices=[0,1,2].map(j=>{
      const k=index.getX(i+j);
      return [positions.getX(k),positions.getY(k),positions.getZ(k)].map(n=>n.toFixed(4)).join(',');
    });
    const first=vertices.indexOf([...vertices].sort()[0]);
    faces.push([...vertices.slice(first),...vertices.slice(0,first)].join('/'));
  }
  return faces.sort();
}

for(const kind of ['bone','dura'])test(kind+' reuses opening surfaces across scrubbing, lift and inspection',()=>{
  const shape=[23,23,9],voxels=new Uint8Array(23*23*9);
  for(let i=0;i<23*23;i++){voxels[9*i+5]=2;voxels[9*i+7]=1;}
  const field={shape,voxels,origin:[-11,-11,0],spacingMm:1,radiusMm:11,routeRadiusMm:2,entryDepthMm:7,
    stations:{bone:7,dura:5,brain:null,tumor:null},
    intensity:{shape:[2,2,2],origin:[0,0,0],spacingMm:[1,1,1],unknownValue:0,voxels:new Uint8Array(8)}};
  const texture=operativeSignalTexture(field.intensity);
  const sectionY={value:field.origin[1]-.5*field.spacingMm};
  const material=()=>operativeTissueMaterial(kind,field,texture,new THREE.Vector3(),sectionY);
  const mesh=()=>new THREE.Mesh(new THREE.BufferGeometry(),material());
  const scene=new THREE.Scene(),intact=mesh(),flap=mesh(),model=buildOperativeModel(field);
  scene.add(intact,flap);
  const before=model.layers[kind].mask.slice(),cutEnd=kind==='bone'?.68:.50;
  const opening=prepareOperativeOpening(kind,model,intact,flap,material(),new THREE.Vector3());
  const cut=scene.getObjectByName(kind+'Kerf');
  opening.update(0,0);const initial=intact.geometry;
  assert.ok(intact.visible&&!flap.visible&&!cut.visible);
  const uniform=(mesh,depth)=>{
    const shader={...THREE.ShaderLib[depth?'depth':'standard'],uniforms:{}};
    (depth?mesh.customDepthMaterial:mesh.material).onBeforeCompile(shader,{});
    assert.match(shader.vertexShader,/vOpeningXY = position.xy/);
    assert.equal(shader.uniforms.uCutSurface.value,mesh===cut);
    assert.ok(shader.fragmentShader.includes('(theta < uOpeningAngle) != uCutSurface'));
    if(!depth){
      assert.equal(shader.uniforms.uFieldSignal.value,texture);
      assert.equal(shader.uniforms.uSectionY,sectionY);
    }
    return shader.uniforms.uOpeningAngle;
  };
  const angle=uniform(intact,false);
  assert.equal(intact.material.customProgramCacheKey(),cut.material.customProgramCacheKey());
  for(const mesh of [intact,cut])assert.equal(uniform(mesh,true),angle);
  assert.equal(uniform(cut,false),angle);
  const kerfed=cut.geometry;
  for(const amount of [.02,.12,cutEnd-.01]){
    const state=opening.update(0,amount);
    assert.ok(intact.visible&&cut.visible&&!flap.visible);
    const start=kind==='bone'?0:.70;
    assert.equal(angle.value,start+amount/cutEnd*(2*Math.PI-2*start));
    assert.equal(angle.value,state.angle);
    assert.equal(intact.geometry,initial);assert.equal(cut.geometry,kerfed);
  }
  opening.update(0,cutEnd);
  assert.ok(!intact.visible&&cut.visible&&!flap.visible);
  opening.update(0,1);
  assert.ok(intact.visible&&!cut.visible&&flap.visible);
  const detached=intact.geometry,patch=flap.geometry;
  assert.notEqual(detached,initial);
  // Terminal rendering uses the same masks and extractor as exact voxel edits.
  for(const [step,lifted,display] of [[0,false,initial],[128,false,kerfed],[128,true,detached]]){
    model.applyOpening(kind,step,lifted);
    const direct=operativeSurface(field,model.layers[kind].mask.slice(),model.layers[kind].bounds,0,kind);
    assert.deepEqual(triangles(direct),triangles(display));
    direct.dispose();
  }
  model.applyOpening(kind,0,false);
  opening.update(model.firstSectionRow,1);
  assert.notEqual(intact.geometry,detached);
  opening.update(0,1);
  assert.equal(intact.geometry,detached);assert.equal(flap.geometry,patch);
  opening.update(0,0);
  assert.equal(intact.geometry,initial);assert.equal(angle.value,-1);
  assert.ok(intact.visible&&!flap.visible&&!cut.visible);
  assert.deepEqual(model.layers[kind].mask,before);
  opening.dispose();texture.dispose();
  assert.equal(scene.children.length,0);
});
