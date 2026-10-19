import * as THREE from 'three';

export const SURFACE_PEELS=4;
/** Front-to-back mesh boundary layers for the scan ray marcher. This uses the
 * actual selected surface geometry, normals, lighting, clipping and opacity.
 * A final sorted transparency tail retains every remaining fragment instead
 * of dropping anatomy beyond the fixed peel budget. Only the tail's internal
 * depth ordering is approximate; the foreground peels/opaque depth are exact.
 */
export class SurfaceVolumePass {
  readonly targets=Array.from({length:SURFACE_PEELS+1},()=>{
    const t=new THREE.WebGLRenderTarget(1,1,{type:THREE.HalfFloatType,depthBuffer:true});
    t.depthTexture=new THREE.DepthTexture(1,1,THREE.UnsignedIntType);return t;
  });
  readonly tail=new THREE.WebGLRenderTarget(1,1,{type:THREE.HalfFloatType,depthBuffer:false});
  private readonly materials=new Map<THREE.Material,{peel:THREE.Material;tail:THREE.Material}>();
  private readonly uniforms={uPreviousSurface:{value:null as THREE.Texture|null},uOpaqueSurface:{value:null as THREE.Texture|null},uSurfaceSize:{value:new THREE.Vector2()},uFirstSurface:{value:true}};
  private make(original:THREE.Material,tail:boolean){
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
      shader.fragmentShader=shader.fragmentShader.replace('#include <clipping_planes_fragment>',`#include <clipping_planes_fragment>
        vec2 surfaceUv=gl_FragCoord.xy/uSurfaceSize;
        if(gl_FragCoord.z>=texture2D(uOpaqueSurface,surfaceUv).r-0.0000001)discard;
        if(!uFirstSurface&&gl_FragCoord.z<=texture2D(uPreviousSurface,surfaceUv).r+0.0000001)discard;`);
    };
    material.customProgramCacheKey=()=>`physical-surface-peeling-v2-${tail}-${original.customProgramCacheKey()}`;
    return material;
  }
  render(renderer:THREE.WebGLRenderer,scene:THREE.Scene,camera:THREE.Camera,opaque:THREE.Texture,w:number,h:number,interactive:boolean){
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
      for(const s of selected){let pair=this.materials.get(s.material);if(!pair){pair={peel:this.make(s.material,false),tail:this.make(s.material,true)};this.materials.set(s.material,pair);}
        // Opacity/color/clip changes update values without rebuilding programs.
        for(const m of [pair.peel,pair.tail]){const p=m as THREE.MeshStandardMaterial,o=s.material as THREE.MeshStandardMaterial;
          p.color.copy(o.color);p.opacity=o.opacity;p.clippingPlanes=o.clippingPlanes;p.roughness=o.roughness;p.metalness=o.metalness;}
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
  dispose(){this.targets.forEach(t=>t.dispose());this.tail.dispose();for(const p of this.materials.values()){p.peel.dispose();p.tail.dispose();}this.materials.clear();}
}
