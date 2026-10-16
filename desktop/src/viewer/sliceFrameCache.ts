import * as THREE from 'three';

/** Explicit offscreen storage; never relies on a preserved default framebuffer.
 * The copy shader deliberately does no second color/tone conversion. */
export class SliceFrameCache {
  private readonly target = new THREE.WebGLRenderTarget(1, 1, {depthBuffer:false});
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.Camera();
  private revision = -1;
  private readonly material = new THREE.RawShaderMaterial({
    glslVersion:THREE.GLSL3, depthTest:false, depthWrite:false,
    uniforms:{image:{value:this.target.texture}},
    vertexShader:'in vec3 position; in vec2 uv; out vec2 vUv; void main(){vUv=uv;gl_Position=vec4(position.xy,0.,1.);}',
    fragmentShader:'precision highp float; uniform sampler2D image; in vec2 vUv; out vec4 color; void main(){color=texture(image,vUv);}',
  });
  private readonly geometry = new THREE.PlaneGeometry(2,2);
  constructor() { this.scene.add(new THREE.Mesh(this.geometry,this.material)); }
  render(renderer:THREE.WebGLRenderer, scene:THREE.Scene, camera:THREE.Camera,
    width:number, height:number, revision:number, restore:()=>void): boolean {
    const ratio=renderer.getPixelRatio(),w=Math.max(1,Math.round(width*ratio)),h=Math.max(1,Math.round(height*ratio));
    const changed=this.revision!==revision||this.target.width!==w||this.target.height!==h;
    if(changed){
      if(this.target.width!==w||this.target.height!==h)this.target.setSize(w,h);
      try{
        renderer.setRenderTarget(this.target);renderer.setScissorTest(false);
        // setViewport scales by devicePixelRatio even for offscreen targets.
        // The texture already uses physical pixels, so supply logical units.
        renderer.setViewport(0,0,w/ratio,h/ratio);
        renderer.clear();renderer.render(scene,camera);
      }finally{renderer.setRenderTarget(null);renderer.setScissorTest(true);restore();}
      this.revision=revision;
    }
    renderer.render(this.scene,this.camera);
    return changed;
  }
  dispose(){this.target.dispose();this.material.dispose();this.geometry.dispose();}
}
