import * as THREE from 'three';
import type { ViewerVolume } from './contracts';
import { inverseAffine, rasAffine } from './coordinates.ts';
import { volumeSettings, volumeTransfer } from './volumeTransfer.ts';
import type { VolumeRenderingSettings } from './volumeTransfer';
import { volumeVertexShader, volumeFragmentShader } from './volumeRaycastShaders';

/** Continuous scalar-volume compositing, in the original RAS physical frame.
 * Offscreen targets bound ray workload; source data is never downsampled or
 * overwritten. Opaque anatomy depth terminates rays before the surface. */
export class ScanVolumeRenderer {
  private readonly sceneTarget: THREE.WebGLRenderTarget;
  private readonly volumeTarget: THREE.WebGLRenderTarget;
  private readonly rayScene = new THREE.Scene();
  private readonly blitScene = new THREE.Scene();
  private readonly camera = new THREE.Camera();
  private readonly material: THREE.ShaderMaterial;
  private readonly blitMaterial: THREE.MeshBasicMaterial;
  private readonly texture: THREE.Data3DTexture;
  private transfer: THREE.DataTexture | null = null;
  private preset = '';
  private enabled = true;
  private readonly spacing: number;
  private readonly background = new THREE.Color();

  constructor(private readonly renderer: THREE.WebGLRenderer, volume: ViewerVolume,
    private readonly window: [number,number]) {
    const affine=rasAffine(volume.affine,volume.frame),inverse=inverseAffine(affine);
    this.spacing=1/Math.max(...inverse.slice(0,3).map(row=>Math.hypot(...row.slice(0,3))));
    this.texture=new THREE.Data3DTexture(volume.mri,volume.shape[2],volume.shape[1],volume.shape[0]);
    this.texture.format=THREE.RedFormat;this.texture.type=THREE.FloatType;
    const linear=renderer.extensions.has('OES_texture_float_linear');
    this.texture.minFilter=this.texture.magFilter=linear?THREE.LinearFilter:THREE.NearestFilter;
    this.texture.unpackAlignment=1;this.texture.needsUpdate=true;
    this.sceneTarget=new THREE.WebGLRenderTarget(1,1,{type:THREE.HalfFloatType,depthBuffer:true});
    this.sceneTarget.depthTexture=new THREE.DepthTexture(1,1,THREE.UnsignedIntType);
    this.volumeTarget=new THREE.WebGLRenderTarget(1,1,{type:THREE.HalfFloatType,depthBuffer:false});
    this.material=new THREE.ShaderMaterial({glslVersion:THREE.GLSL3,vertexShader:volumeVertexShader,
      fragmentShader:volumeFragmentShader,depthTest:false,depthWrite:false,toneMapped:false,uniforms:{
        uVolume:{value:this.texture},uTransfer:{value:null},uScene:{value:this.sceneTarget.texture},
        uDepth:{value:this.sceneTarget.depthTexture},uWorldToVoxel:{value:new THREE.Matrix4().set(...inverse.flat() as Parameters<THREE.Matrix4['set']>)},
        uCameraWorld:{value:new THREE.Matrix4()},uInverseProjection:{value:new THREE.Matrix4()},
        uShape:{value:new THREE.Vector3(...volume.shape)},uRange:{value:new THREE.Vector2()},
        uDensity:{value:.45},uStep:{value:this.spacing*.7},uLinear:{value:linear},
        uClipEnabled:{value:false},uClipPlane:{value:new THREE.Vector4()},
        uBackground:{value:this.background},
      }});
    this.rayScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2),this.material));
    this.blitMaterial=new THREE.MeshBasicMaterial({map:this.volumeTarget.texture,depthTest:false,depthWrite:false});
    this.blitScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2),this.blitMaterial));
  }
  configure(input: VolumeRenderingSettings | undefined, modality: string, clip: THREE.Plane | null) {
    const settings=volumeSettings(input,modality);this.enabled=settings.enabled&&settings.density>0;
    if(settings.preset!==this.preset){
      const lut=volumeTransfer(settings.preset,this.window);
      this.transfer?.dispose();this.transfer=new THREE.DataTexture(lut.data,1024,1,THREE.RGBAFormat,THREE.UnsignedByteType);
      this.transfer.minFilter=this.transfer.magFilter=THREE.LinearFilter;this.transfer.needsUpdate=true;
      this.material.uniforms.uTransfer.value=this.transfer;
      this.material.uniforms.uRange.value.set(...lut.range);this.preset=settings.preset;
    }
    this.material.uniforms.uDensity.value=settings.density;
    this.material.uniforms.uClipEnabled.value=!!clip;
    if(clip)this.material.uniforms.uClipPlane.value.set(clip.normal.x,clip.normal.y,clip.normal.z,clip.constant);
  }
  render(scene: THREE.Scene, camera: THREE.PerspectiveCamera, width: number, height: number,
    interactive: boolean, restoreViewport: ()=>void) {
    if(!this.enabled){this.renderer.render(scene,camera);return;}
    const pixelRatio=this.renderer.getPixelRatio();
    const scale=Math.min(pixelRatio,(interactive?700:1300)/Math.max(width,height));
    const w=Math.max(1,Math.round(width*scale)),h=Math.max(1,Math.round(height*scale));
    if(this.sceneTarget.width!==w||this.sceneTarget.height!==h){this.sceneTarget.setSize(w,h);this.volumeTarget.setSize(w,h);}
    const uniforms=this.material.uniforms;
    camera.updateMatrixWorld();
    uniforms.uCameraWorld.value.copy(camera.matrixWorld);
    uniforms.uInverseProjection.value.copy(camera.projectionMatrixInverse);
    uniforms.uStep.value=this.spacing*(interactive?1.65:.4);
    this.renderer.getClearColor(this.background);
    const clearAlpha=this.renderer.getClearAlpha();
    try {
      this.renderer.setClearColor(0,0);
      this.renderer.setRenderTarget(this.sceneTarget);this.renderer.clear();this.renderer.render(scene,camera);
      this.renderer.setRenderTarget(this.volumeTarget);this.renderer.clear();this.renderer.render(this.rayScene,this.camera);
    } finally {
      this.renderer.setClearColor(this.background,clearAlpha);
      this.renderer.setRenderTarget(null);restoreViewport();
    }
    this.renderer.render(this.blitScene,this.camera);
  }
  dispose(){
    this.sceneTarget.dispose();this.volumeTarget.dispose();this.texture.dispose();this.transfer?.dispose();
    for(const scene of [this.rayScene,this.blitScene])scene.traverse(child=>{if(child instanceof THREE.Mesh)child.geometry.dispose();});
    this.material.dispose();this.blitMaterial.dispose();
  }
}
