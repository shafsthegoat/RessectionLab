import * as THREE from 'three';
import {CranialDepth, type CranialBoundary} from './cranialDepth.ts';

export const SURFACE_PEELS=4;
/** Front-to-back mesh boundary layers for the scan ray marcher. This uses the
 * actual selected surface geometry, normals, lighting, clipping and opacity.
 * A final sorted transparency tail retains every remaining fragment instead
 * of dropping anatomy beyond the fixed peel budget. Only the tail's internal
 * depth ordering is approximate; the foreground peels/opaque depth are exact.
 */
export class SurfaceVolumePass {
  readonly cranial=new CranialDepth();
  readonly targets=Array.from({length:SURFACE_PEELS+1},()=>{
    const t=new THREE.WebGLRenderTarget(1,1,{type:THREE.HalfFloatType,depthBuffer:true});
    t.depthTexture=new THREE.DepthTexture(1,1,THREE.UnsignedIntType);return t;
  });
  readonly tail=new THREE.WebGLRenderTarget(1,1,{type:THREE.HalfFloatType,depthBuffer:false});
  private readonly materials=new Map<THREE.Material,{peel:THREE.Material;tail:THREE.Material;sourceProgramKey:string;boundary?:CranialBoundary}>();
  private readonly uniforms={uPreviousSurface:{value:null as THREE.Texture|null},uOpaqueSurface:{value:null as THREE.Texture|null},uSurfaceSize:{value:new THREE.Vector2()},uFirstSurface:{value:true}};
  private readonly corticalContact={
    uHasCorticalBoundary:{value:false},
    uCorticalBoundaryDepth:{value:null as THREE.Texture|null},
  };
  private make(original:THREE.Material,tail:boolean,corticalContact=false){
    const material=original.clone();material.transparent=tail;material.depthWrite=!tail;
    material.depthTest=!tail;material.blending=tail?THREE.NormalBlending:THREE.NoBlending;
    material.premultipliedAlpha=true;material.toneMapped=false;
    material.forceSinglePass=true;
    material.onBeforeCompile=(shader,renderer)=>{
      // Material.clone() omits compile callbacks. Keep the same display-only
      // source-context appearance in the peels and Structures-only rendering.
      original.onBeforeCompile.call(material,shader,renderer);
      Object.assign(shader.uniforms,this.uniforms);
      shader.fragmentShader='uniform sampler2D uPreviousSurface;\nuniform sampler2D uOpaqueSurface;\nuniform vec2 uSurfaceSize;\nuniform bool uFirstSurface;\n'+shader.fragmentShader;
      if(corticalContact){
        Object.assign(shader.uniforms,this.corticalContact);
        shader.fragmentShader='uniform sampler2D uCorticalBoundaryDepth; uniform bool uHasCorticalBoundary;\n'+shader.fragmentShader;
      }
      shader.fragmentShader=shader.fragmentShader.replace('#include <clipping_planes_fragment>',`#include <clipping_planes_fragment>
        vec2 surfaceUv=gl_FragCoord.xy/uSurfaceSize;
        if(gl_FragCoord.z>=texture2D(uOpaqueSurface,surfaceUv).r-0.0000001)discard;
        float fragmentDepth=gl_FragCoord.z;
        ${corticalContact ? `
        // An adjacent context label can describe the same surface as cortex.
        // Preserve both colors. Reserve a separate peel for the superficial layer
        // only when the depth attachment cannot reliably split the surfaces.
        if(uHasCorticalBoundary){
          float cortex=texelFetch(uCorticalBoundaryDepth,ivec2(gl_FragCoord.xy),0).r;
          const float depthUnit=1./16777215.;
          if(cortex<1. && abs(fragmentDepth-cortex)<=4.*depthUnit)
            fragmentDepth=max(0.,cortex-6.*depthUnit);
        }
        gl_FragDepth=fragmentDepth;` : ''}
        if(!uFirstSurface&&fragmentDepth<=texture2D(uPreviousSurface,surfaceUv).r+0.0000001)discard;`);
    };
    material.customProgramCacheKey=()=>`physical-surface-peeling-v3-${tail}-${corticalContact}-${original.customProgramCacheKey()}`;
    return material;
  }
  render(renderer:THREE.WebGLRenderer,scene:THREE.Scene,camera:THREE.PerspectiveCamera,opaque:THREE.Texture,w:number,h:number,interactive:boolean){
    const selected:Array<{mesh:THREE.Mesh;material:THREE.Material}>=[],hidden:THREE.Object3D[]=[];
    scene.traverseVisible(o=>{
      if(o instanceof THREE.Mesh&&!Array.isArray(o.material)&&o.userData.compartment&&o.material.opacity>0&&o.material.opacity<1)selected.push({mesh:o,material:o.material});
    });
    const set=new Set(selected.map(s=>s.mesh));
    scene.traverseVisible(o=>{if((o instanceof THREE.Mesh||o instanceof THREE.Line||o instanceof THREE.Points||o instanceof THREE.Sprite)&&!set.has(o as THREE.Mesh)){hidden.push(o);o.visible=false;}});
    const count=selected.length?(interactive?0:SURFACE_PEELS):0;
    this.uniforms.uSurfaceSize.value.set(w,h);this.uniforms.uOpaqueSurface.value=opaque;
    const target=(t:THREE.WebGLRenderTarget)=>{if(t.width!==w||t.height!==h)t.setSize(w,h);renderer.setRenderTarget(t);renderer.clear();};
    try{
      const hasBoundary=selected.some(s=>s.mesh.userData.cranialBoundary);
      if(hasBoundary)this.cranial.capture(renderer,scene,camera,selected,w,h);
      this.corticalContact.uHasCorticalBoundary.value=selected.some(s=>s.mesh.userData.cranialBoundary==='cortex');
      this.corticalContact.uCorticalBoundaryDepth.value=this.cranial.targets.get('cortex')?.depthTexture??null;
      for(const s of selected){
        const boundary=s.mesh.userData.cranialBoundary;
        const sourceProgramKey=s.material.customProgramCacheKey();
        let pair=this.materials.get(s.material);
        if(!pair||pair.boundary!==boundary){
          pair?.peel.dispose();pair?.tail.dispose();
          const contact=s.mesh.userData.corticalContact===true;
          pair={peel:this.make(s.material,false,contact),tail:this.make(s.material,true,contact),sourceProgramKey,boundary};
          if(boundary){this.cranial.install(pair.peel,boundary);this.cranial.install(pair.tail,boundary);}
          this.materials.set(s.material,pair);
        }else if(pair.sourceProgramKey!==sourceProgramKey){
          // The existing clones compile the currently selected source variant.
          pair.sourceProgramKey=sourceProgramKey;
          pair.peel.needsUpdate=true;
          pair.tail.needsUpdate=true;
        }
        const source=s.material as THREE.MeshStandardMaterial;
        for(const m of [pair.peel,pair.tail]){
          const material=m as THREE.MeshStandardMaterial;
          material.color.copy(source.color);
          if(material.emissive){
            material.emissive.copy(source.emissive);
            material.emissiveIntensity=source.emissiveIntensity;
          }
          material.opacity=source.opacity;
          material.clippingPlanes=source.clippingPlanes;
          material.roughness=source.roughness;
          material.metalness=source.metalness;
          if(material instanceof THREE.MeshPhysicalMaterial&&source instanceof THREE.MeshPhysicalMaterial){
            material.specularIntensity=source.specularIntensity;
            material.clearcoat=source.clearcoat;
            material.clearcoatRoughness=source.clearcoatRoughness;
          }
        }
        s.mesh.material=pair.peel;
      }
      for(let i=0;i<=count;i++){
        this.uniforms.uFirstSurface.value=i===0;this.uniforms.uPreviousSurface.value=i?this.targets[i-1].depthTexture:opaque;
        target(this.targets[i]);if(selected.length)renderer.render(scene,camera);
      }
      // The extra layer's depth bounds the remaining transparency. Its color
      // is replaced by all remaining fragments composited in Three's order.
      for(const s of selected)s.mesh.material=this.materials.get(s.material)!.tail;
      this.uniforms.uFirstSurface.value=count===0;this.uniforms.uPreviousSurface.value=count?this.targets[count-1].depthTexture:opaque;
      target(this.tail);if(selected.length)renderer.render(scene,camera);
    }finally{for(const s of selected)s.mesh.material=s.material;for(const o of hidden)o.visible=true;}
    return {count:selected.length?count+1:0,tailIndex:count};
  }
  dispose(){this.cranial.dispose();this.targets.forEach(t=>t.dispose());this.tail.dispose();for(const p of this.materials.values()){p.peel.dispose();p.tail.dispose();}this.materials.clear();}
}
