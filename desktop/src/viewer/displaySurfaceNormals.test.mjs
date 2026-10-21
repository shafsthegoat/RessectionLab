import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {prepareSurface} from './prepareSurface.ts';
import {displaySurfaceNormals} from './displaySurfaceNormals.ts';
import {applyVesselDepthMaterial} from './anatomySurfaceMaterial.ts';
import {applySourceContextMaterial} from './sourceContextMaterial.ts';
import {SurfaceVolumePass} from './SurfaceVolumePass.ts';

const identity=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
test('vessel lighting becomes smoother without moving or connecting source anatomy',()=>{
  const shape=[25,25,25],mask=Uint8Array.from({length:25**3},(_,i)=>{
    const x=Math.floor(i/625),y=Math.floor(i/25)%25,z=i%25;
    return (x-12)**2+(y-12)**2<25&&z>2&&z<22?1:0;
  });
  const s=prepareSurface(mask,shape,identity,'vessels');
  const before=structuredClone(s),originalMask=mask.slice();
  const normals=displaySurfaceNormals(s.smoothNormals,s.indices,identity);
  const error=n=>{
    let sum=0,count=0;
    for(let i=0;i<n.length;i+=3){
      if(s.smoothPositions[i+2]<6||s.smoothPositions[i+2]>18)continue;
      const x=s.smoothPositions[i]-12,y=s.smoothPositions[i+1]-12;
      sum+=1-(x*n[i]+y*n[i+1])/Math.hypot(x,y);count++;
    }
    return sum/count;
  };
  assert.ok(error(normals)<error(s.smoothNormals)*.3,'cylindrical vessel normals must approach the physical radial direction');
  for(let i=0;i<normals.length;i+=3){
    const cosine=normals[i]*s.smoothNormals[i]+normals[i+1]*s.smoothNormals[i+1]+normals[i+2]*s.smoothNormals[i+2];
    assert.ok(cosine>=Math.cos(Math.PI/6)-1e-6,'normal change is bounded to thirty degrees');
  }
  assert.deepEqual(s,before);assert.deepEqual(mask,originalMask);
});

test('normal fairing never crosses disconnected surfaces and works in physical space',()=>{
  const ids=Uint32Array.from([0,1,2,3,4,5]);
  const physical=Float32Array.from([0,0,1, .1,0,.995, -.1,0,.995, 0,0,-1, .1,0,-.995, -.1,0,-.995]);
  const mirror=[[-.5,.2,0,70],[0,2,.3,-20],[0,0,3,90],[0,0,0,1]];
  const encode=(values,affine)=>Float32Array.from(values,(_,i)=>{
    const start=Math.floor(i/3)*3,a=i%3;
    return affine[0][a]*values[start]+affine[1][a]*values[start+1]+affine[2][a]*values[start+2];
  });
  const simple=displaySurfaceNormals(physical,ids,identity),mirrored=displaySurfaceNormals(encode(physical,mirror),ids,mirror);
  const expected=encode(simple,mirror);
  for(let i=0;i<expected.length;i+=3){const length=Math.hypot(expected[i],expected[i+1],expected[i+2]);for(let k=0;k<3;k++)assert.ok(Math.abs(expected[i+k]/length-mirrored[i+k])<1e-6);}
  for(let i=0;i<9;i+=3)assert.ok(simple[i+2]>.99);
  for(let i=9;i<18;i+=3)assert.ok(simple[i+2]<-.99);
});

test('vessel radiance depth cue survives volume peels without changing opacity or surface depth',()=>{
  const material=new THREE.MeshStandardMaterial({opacity:.42,transparent:true}),pass=new SurfaceVolumePass();
  applyVesselDepthMaterial(material,[[-10,-20,-30],[10,20,30]]);
  applySourceContextMaterial(material,false);
  const peel=pass.make(material,false),tail=pass.make(material,true);
  try{
    for(const m of [material,peel,tail]){
      const shader={uniforms:{},fragmentShader:THREE.ShaderLib.standard.fragmentShader};m.onBeforeCompile(shader,{});
      assert.deepEqual(shader.uniforms.uAnatomyCenter.value.toArray(),[0,0,0]);
      assert.ok(shader.uniforms.uAnatomyRadius.value>0);
      assert.match(shader.fragmentShader,/outgoingLight\*=mix\(1.08,0.56/);
      assert.match(shader.fragmentShader,/#include <opaque_fragment>/);
      assert.match(m.customProgramCacheKey(),/anatomy-vessel-depth/);
      assert.equal(m.opacity,.42);
    }
  }finally{material.dispose();peel.dispose();tail.dispose();pass.dispose();}
});
