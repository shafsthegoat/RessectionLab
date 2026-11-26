import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {prepareSurface} from './prepareSurface.ts';
import {displaySurfaceNormals} from './displaySurfaceNormals.ts';
import {applyCtaVesselFinish,applyVesselDepthMaterial,ctaDisplayKind,ctaVesselColor,vesselWallFinish} from './anatomySurfaceMaterial.ts';
import {applySourceContextMaterial} from './sourceContextMaterial.ts';
import {SurfaceVolumePass} from './SurfaceVolumePass.ts';
import {VesselDepthSeparation} from './vesselDepthSeparation.ts';
import {CtaVesselShadow} from './ctaVesselShadow.ts';

const identity=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];

test('CTA shadow depth uses only the visible opaque acquisition meshes and tracks the physical cut',()=>{
  const shadow=new CtaVesselShadow([[-10,-20,-15],[10,20,15]]);
  const scene=new THREE.Scene(),geometry=new THREE.BoxGeometry();
  const caster=new THREE.MeshStandardMaterial(),ordinary=new THREE.MeshStandardMaterial();
  caster.userData.ctaVesselReceiver=true;
  const visible=new THREE.Mesh(geometry,caster),hidden=new THREE.Mesh(geometry,caster),foreign=new THREE.Mesh(geometry,ordinary);
  visible.position.set(7,9,4);
  hidden.visible=false;
  scene.add(visible,hidden,foreign);
  let target={prior:true};
  const previous=target,frames=[];
  const renderer={
    getRenderTarget:()=>target,setRenderTarget:value=>{target=value;},
    getViewport:out=>out.set(1,2,30,40),setViewport:value=>assert.deepEqual(value.toArray(),[1,2,30,40]),
    getScissor:out=>out.set(3,4,20,30),setScissor:value=>assert.deepEqual(value.toArray(),[3,4,20,30]),
    getScissorTest:()=>true,setScissorTest:value=>assert(value),clear(){},
    render(casters,camera){
      assert(camera.isOrthographicCamera);
      assert.equal(target.depthTexture,shadow.texture);
      frames.push(casters.children.map(mesh=>{
        assert.equal(mesh.geometry,geometry);
        assert.deepEqual(new THREE.Vector3().setFromMatrixPosition(mesh.matrix).toArray(),[7,9,4]);
        return mesh.material.clippingPlanes[0].constant;
      }));
    },
  };
  const light=new THREE.Vector3(0,0,1),up=new THREE.Vector3(0,1,0),cut=new THREE.Vector4(0,0,-1,4);
  try{
    shadow.capture(renderer,scene,light,up,cut);
    shadow.capture(renderer,scene,light,up,cut);
    cut.w=6;
    shadow.capture(renderer,scene,light,up,cut);
    caster.opacity=.5;
    shadow.capture(renderer,scene,light,up,cut);
    assert.deepEqual(frames,[[4],[6],[]]);
    assert.equal(target,previous);
    assert.equal(scene.children.length,3);
  }finally{shadow.dispose();geometry.dispose();caster.dispose();ordinary.dispose();}
});

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
  const material=new THREE.MeshPhysicalMaterial({opacity:.42,transparent:true,...vesselWallFinish}),pass=new SurfaceVolumePass();
  const separation=new VesselDepthSeparation();
  applyVesselDepthMaterial(material,[[-10,-20,-30],[10,20,30]],undefined,[20,40,60]);
  separation.install(material);
  applySourceContextMaterial(material,false);
  const peel=pass.make(material,false),tail=pass.make(material,true);
  try{
    for(const m of [material,peel,tail]){
      const shader={uniforms:{},vertexShader:THREE.ShaderLib.physical.vertexShader,fragmentShader:THREE.ShaderLib.physical.fragmentShader};m.onBeforeCompile(shader,{});
      assert.deepEqual(shader.uniforms.uAnatomyCenter.value.toArray(),[0,0,0]);
      assert.ok(shader.uniforms.uAnatomyRadius.value>0);
      assert.equal(shader.uniforms.uLocalVesselEnabled.value,false);
      assert.match(shader.fragmentShader,/outgoingLight \*= 1.0 - 0.72 \* contact/);
      assert.match(shader.fragmentShader,/vViewPosition.z\+\(viewMatrix\*vec4\(uAnatomyCenter,1.0\)\).z/);
      assert.match(shader.fragmentShader,/material.clearcoat\*=1.0-sourceBoundary/);
      assert.equal(m.clearcoat,vesselWallFinish.clearcoat);
      assert.match(shader.fragmentShader,/#include <opaque_fragment>/);
      assert.match(m.customProgramCacheKey(),/anatomy-vessel-depth/);
      assert.equal(m.opacity,.42);
    }
  }finally{material.dispose();peel.dispose();tail.dispose();pass.dispose();separation.dispose();}
});

test('vessel depth capture shares physical clipping and reuses the prior view across rendering passes',()=>{
  const separation=new VesselDepthSeparation(),geometry=new THREE.BoxGeometry();
  const plane=new THREE.Plane(new THREE.Vector3(0,0,1),-2);
  const material=new THREE.MeshPhysicalMaterial({clippingPlanes:[plane]});
  const mesh=new THREE.Mesh(geometry,material),camera=new THREE.PerspectiveCamera(34,1.5,0.2,1000);
  const viewport=new THREE.Vector4(20,50,300,200),scissor=viewport.clone();
  let target=null,draws=0;
  const renderer={
    getRenderTarget:()=>target, setRenderTarget:value=>{target=value;},
    getViewport:value=>value.copy(viewport), getScissor:value=>value.copy(scissor),
    getScissorTest:()=>true, setViewport:value=>assert.deepEqual(value,viewport),
    setScissor:value=>assert.deepEqual(value,scissor), setScissorTest:value=>assert.equal(value,true),
    clear:(color,depth)=>assert.deepEqual([color,depth],[false,true]),
    render:scene=>{
      const [copy]=scene.children;
      assert.equal(copy.geometry,geometry);
      assert.deepEqual(copy.material.clippingPlanes,[plane]);
      assert.equal(copy.material.colorWrite,false);
      assert.equal(target.height,200);
      draws++;
    },
  };
  try{
    camera.position.z=50;mesh.updateWorldMatrix(true,false);
    separation.capture(renderer,camera,[mesh],600,400,300);
    separation.capture(renderer,camera,[mesh],600,400,300);
    assert.equal(draws,1);
    assert.equal(separation.active,true);
    assert.equal(target,null);
    plane.constant=-4;
    separation.capture(renderer,camera,[mesh],600,400,300);
    assert.equal(draws,2);
    separation.capture(renderer,camera,[],600,400,300);
    assert.equal(draws,2);
    assert.equal(separation.active,false);
  }finally{separation.dispose();material.dispose();geometry.dispose();}
});

test('acquired CTA contrast gives every vessel type one colour and retains the original physical finish on exit',()=>{
  const settings={presentation:'segmentation',vesselShading:'cta-contrast'};
  const colors=['artery','vein','sinus','vessels'].map(kind=>ctaVesselColor(kind,settings,'CTA'));
  assert.equal(new Set(colors).size,1);
  assert.ok(colors.every(color=>/^#[0-9a-f]{6}$/.test(color)));
  assert.equal(ctaVesselColor('brain',settings,'CTA'),undefined);
  assert.equal(ctaVesselColor('artery',settings,'MRA'),undefined);
  assert.equal(ctaVesselColor('artery',settings,'CT'),undefined);
  assert.equal(ctaVesselColor('artery',settings,'CTA segmentation only'),undefined);
  assert.equal(ctaVesselColor('artery',{...settings,presentation:'scan'},'CTA'),undefined);
  assert.equal(ctaVesselColor('artery',{...settings,operativeView:'surgeon'},'CTA'),undefined);

  const material=new THREE.MeshPhysicalMaterial({opacity:.42,depthWrite:false,...vesselWallFinish});
  try{
    applyCtaVesselFinish(material,'artery',settings,'CTA');
    assert.equal('#'+material.color.getHexString(),colors[0]);
    assert.ok(material.clearcoat<vesselWallFinish.clearcoat);
    assert.ok(material.specularIntensity<vesselWallFinish.specularIntensity);
    assert.equal(material.opacity,.42);
    assert.equal(material.depthWrite,false);
    applyCtaVesselFinish(material,'artery',{...settings,vesselShading:'source'},'CTA');
    for(const [name,value] of Object.entries(vesselWallFinish))assert.equal(material[name],value);
  }finally{material.dispose();}
});

test('a connected-tree display takes its CTA kind from its name or layer, and the review overlay keeps its own colour',()=>{
  const rooted={scope:'rooted'};
  assert.equal(ctaDisplayKind({name:'model',kind:'vessels',color:'#aa352b',displayFilter:rooted}),'artery');
  assert.equal(ctaDisplayKind({name:'model',kind:'vessels',color:'#aa352b',displayFilter:{scope:'rooted-review'}}),'artery');
  assert.equal(ctaDisplayKind({name:'__vessel_veins:model',kind:'vein',color:'#aa352b'}),'vein');
  assert.equal(ctaDisplayKind({name:'__vessel_arteries:model',kind:'artery',color:'#aa352b'}),'artery');
  assert.equal(ctaDisplayKind({name:'__vessel_review:model',kind:'vessels',color:'#ffad54'}),undefined);
  assert.equal(ctaDisplayKind({name:'full',kind:'vessels',color:'#aa352b',displayFilter:{scope:'all'}}),'vessels');
  assert.equal(ctaDisplayKind({name:'sinus',kind:'sinus',color:'#aa352b',displayFilter:rooted}),'sinus');
});
