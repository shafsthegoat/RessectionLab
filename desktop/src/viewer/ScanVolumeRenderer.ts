import {packAnatomyLabels,MAX_ANATOMY_LAYERS} from './labelPacking';
import * as THREE from 'three';
import type { ViewerVolume } from './contracts';
import { inverseAffine, rasAffine } from './coordinates.ts';
import { volumeSettings, volumeTransfer } from './volumeTransfer.ts';
import {layerAppearance} from './workspaceSettings';
import type {ViewerWorkspaceSettings} from './workspaceSettings';
import type { VolumeRenderingSettings } from './volumeTransfer';
import { volumeVertexShader, volumeFragmentShader } from './volumeRaycastShaders';

/** Continuous scalar-volume compositing, in the original RAS physical frame.
 * Offscreen targets bound ray workload; source data is never downsampled or
 * overwritten. Opaque anatomy depth terminates rays before the surface. */
export class ScanVolumeRenderer {
  private readonly sceneTarget: THREE.WebGLRenderTarget;
  private readonly volumeTarget: THREE.WebGLRenderTarget;
  private interactionTargets: {scene: THREE.WebGLRenderTarget; volume: THREE.WebGLRenderTarget} | null = null;
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
  private layers: ViewerVolume["compartments"];
  private readonly background = new THREE.Color();

  constructor(private readonly renderer: THREE.WebGLRenderer, volume: ViewerVolume,
    private readonly window: [number,number], codes?:Array<[number,number]>) {
    this.layers=volume.compartments;
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
        uLayerCount:{value:volume.compartments.length},uLayerCodes:{value:(codes??packAnatomyLabels(volume).codes).map(c=>new THREE.Vector2(...c))},uLayerEnabled:{value:new Float32Array(MAX_ANATOMY_LAYERS).fill(1)},
        uLabels:{value:null},uLabelBits:{value:0},uVisibleBits:{value:0},uLayerColor:{value:Array.from({length:MAX_ANATOMY_LAYERS},()=>new THREE.Vector4())},
        uActiveLayerCount:{value:volume.compartments.length},uActiveLayers:{value:Float32Array.from({length:MAX_ANATOMY_LAYERS},(_,i)=>i)},
        uExtraClipCount:{value:0},uExtraClips:{value:Array.from({length:7},()=>new THREE.Vector4())},
        uShadows:{value:true},uIsolate:{value:false},uInteractive:{value:false},
      }});
    this.rayScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2),this.material));
    this.blitMaterial=new THREE.MeshBasicMaterial({map:this.volumeTarget.texture,depthTest:false,depthWrite:false});
    this.blitScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2),this.blitMaterial));
  }
  configure(input: VolumeRenderingSettings | undefined, modality: string, clip: THREE.Plane | null) {
    const settings=volumeSettings(input,modality);this.enabled=settings.enabled&&settings.density>0;
    const signature=JSON.stringify([settings.preset,input?.window,input?.points]);
    if(signature!==this.preset){
      const lut=volumeTransfer(settings.preset,this.window,input);
      this.transfer?.dispose();this.transfer=new THREE.DataTexture(lut.data,1024,1,THREE.RGBAFormat,THREE.UnsignedByteType);
      this.transfer.minFilter=this.transfer.magFilter=THREE.LinearFilter;this.transfer.needsUpdate=true;
      this.material.uniforms.uTransfer.value=this.transfer;
      this.material.uniforms.uRange.value.set(...lut.range);this.preset=signature;
    }
    this.material.uniforms.uShadows.value=input?.shadows!==false;
    this.material.uniforms.uIsolate.value=input?.isolateLabels===true;
    this.material.uniforms.uDensity.value=settings.density;
    this.material.uniforms.uClipEnabled.value=!!clip;
    if(clip)this.material.uniforms.uClipPlane.value.set(clip.normal.x,clip.normal.y,clip.normal.z,clip.constant);
  }
  labels(texture:THREE.Data3DTexture,visible:Record<string,boolean>,settings:ViewerWorkspaceSettings,planes:THREE.Plane[]){
    const u=this.material.uniforms;u.uLabels.value=texture;let bits=0,all=0,active=0;
    this.layers.forEach((layer,i)=>{const a=layerAppearance(layer.name,layer.color,settings);
      u.uLayerEnabled.value[i]=visible[layer.name]!==false?1:0;
      if(visible[layer.name]!==false){all|=1<<i;u.uActiveLayers.value[active++]=i;}
      if(visible[layer.name]!==false && a.opacity<1 && a.opacity>0)bits|=1<<i;
      const color=new THREE.Color(a.color);u.uLayerColor.value[i].set(color.r,color.g,color.b,a.opacity);
    });
    u.uActiveLayerCount.value=active;
    u.uLabelBits.value=bits;u.uVisibleBits.value=all;u.uExtraClipCount.value=Math.min(7,planes.length);
    planes.slice(0,7).forEach((p,i)=>u.uExtraClips.value[i].set(p.normal.x,p.normal.y,p.normal.z,p.constant));
  }
  get active(){return this.enabled;}
  render(scene: THREE.Scene, camera: THREE.PerspectiveCamera, width: number, height: number,
    interactive: boolean, restoreViewport: ()=>void) {
    if(!this.enabled){this.renderer.render(scene,camera);return;}
    const pixelRatio=this.renderer.getPixelRatio();
    const scale=Math.min(pixelRatio,(interactive?360:1300)/Math.max(width,height));
    const w=Math.max(1,Math.round(width*scale)),h=Math.max(1,Math.round(height*scale));
    // Keep the small drag buffers separate so pointer release does not discard
    // the full-quality attachments on every gesture. Sizes/steps are unchanged.
    if(interactive && !this.interactionTargets){
      const scene=new THREE.WebGLRenderTarget(w,h,{type:THREE.HalfFloatType,depthBuffer:true});
      scene.depthTexture=new THREE.DepthTexture(w,h,THREE.UnsignedIntType);
      this.interactionTargets={scene,volume:new THREE.WebGLRenderTarget(w,h,{type:THREE.HalfFloatType,depthBuffer:false})};
    }
    const sceneTarget=interactive?this.interactionTargets!.scene:this.sceneTarget;
    const volumeTarget=interactive?this.interactionTargets!.volume:this.volumeTarget;
    if(sceneTarget.width!==w||sceneTarget.height!==h){sceneTarget.setSize(w,h);volumeTarget.setSize(w,h);}
    const uniforms=this.material.uniforms;
    uniforms.uScene.value=sceneTarget.texture;uniforms.uDepth.value=sceneTarget.depthTexture;
    this.blitMaterial.map=volumeTarget.texture;
    camera.updateMatrixWorld();
    uniforms.uCameraWorld.value.copy(camera.matrixWorld);
    uniforms.uInverseProjection.value.copy(camera.projectionMatrixInverse);
    uniforms.uStep.value=this.spacing*(interactive?3:.6);
    uniforms.uInteractive.value=interactive;
    this.renderer.getClearColor(this.background);
    const clearAlpha=this.renderer.getClearAlpha();
    const hidden:THREE.Object3D[]=[];
    scene.traverse(o=>{if(o instanceof THREE.Mesh && o.visible && o.userData.compartment && !Array.isArray(o.material) && o.material.opacity<1){hidden.push(o);o.visible=false;}});
    try {
      this.renderer.setClearColor(0,0);
      this.renderer.setRenderTarget(sceneTarget);this.renderer.clear();this.renderer.render(scene,camera);
      this.renderer.setRenderTarget(volumeTarget);this.renderer.clear();this.renderer.render(this.rayScene,this.camera);
    } finally {
      hidden.forEach(o=>o.visible=true);
      this.renderer.setClearColor(this.background,clearAlpha);
      this.renderer.setRenderTarget(null);restoreViewport();
    }
    this.renderer.render(this.blitScene,this.camera);
  }
  dispose(){
    this.sceneTarget.dispose();this.volumeTarget.dispose();this.texture.dispose();this.transfer?.dispose();
    this.interactionTargets?.scene.dispose();this.interactionTargets?.volume.dispose();
    for(const scene of [this.rayScene,this.blitScene])scene.traverse(child=>{if(child instanceof THREE.Mesh)child.geometry.dispose();});
    this.material.dispose();this.blitMaterial.dispose();
  }
}
