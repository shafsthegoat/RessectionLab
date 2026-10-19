import {packAnatomyLabels,MAX_ANATOMY_LAYERS} from './labelPacking';
import * as THREE from 'three';
import type { ViewerVolume } from './contracts';
import { inverseAffine, rasAffine } from './coordinates.ts';
import { isCtModality, volumeSettings, volumeTransfer } from './volumeTransfer.ts';
import {layerAppearance} from './workspaceSettings';
import type {ViewerWorkspaceSettings} from './workspaceSettings';
import type { VolumeRenderingSettings } from './volumeTransfer';
import { volumeVertexShader, volumeFragmentShader } from './volumeRaycastShaders';
import {SurfaceVolumePass} from './SurfaceVolumePass';

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
  private automaticMriTransfer=false;
  private shellContext=false;
  private enabled = true;
  private readonly spacing: number;
  private layers: ViewerVolume["compartments"];
  private readonly ownsTexture: boolean;
  private readonly background = new THREE.Color();
  private readonly surfacePass=new SurfaceVolumePass();
  private readonly accumulation=Array.from({length:2},()=>new THREE.WebGLRenderTarget(1,1,{type:THREE.HalfFloatType,depthBuffer:false}));
  private readonly accumulateScene=new THREE.Scene();
  private readonly accumulateMaterial=new THREE.ShaderMaterial({glslVersion:THREE.GLSL3,vertexShader:volumeVertexShader,
    fragmentShader:'precision highp float; in vec2 vUv; out vec4 outColor; uniform sampler2D uCurrent; uniform sampler2D uPrevious; uniform float uWeight; void main(){outColor=mix(texture(uPrevious,vUv),texture(uCurrent,vUv),uWeight);}',
    depthTest:false,depthWrite:false,toneMapped:false,uniforms:{uCurrent:{value:null},uPrevious:{value:null},uWeight:{value:1}}});
  private samples=0;
  invalidate(){this.samples=0;}
  get needsRefinement(){return this.enabled&&this.samples>0&&this.samples<4;}

  constructor(private readonly renderer: THREE.WebGLRenderer, volume: ViewerVolume,
    private readonly window: [number,number], codes?:Array<[number,number]>, sourceTexture?:THREE.Data3DTexture) {
    this.layers=volume.compartments;
    const affine=rasAffine(volume.affine,volume.frame),inverse=inverseAffine(affine);
    this.spacing=1/Math.max(...inverse.slice(0,3).map(row=>Math.hypot(...row.slice(0,3))));
    this.ownsTexture=!sourceTexture;
    this.texture=sourceTexture??new THREE.Data3DTexture(volume.mri,volume.shape[2],volume.shape[1],volume.shape[0]);
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
        uSkullMode:{value:0},uCt:{value:false},uGradientOpacity:{value:false},uDensity:{value:.45},uStep:{value:this.spacing*.7},uLinear:{value:linear},
        uClipEnabled:{value:false},uClipPlane:{value:new THREE.Vector4()},
        uBackground:{value:this.background},
        uLayerCount:{value:volume.compartments.length},uLayerCodes:{value:(codes??packAnatomyLabels(volume).codes).map(c=>new THREE.Vector2(...c))},uLayerEnabled:{value:new Float32Array(MAX_ANATOMY_LAYERS).fill(1)},
        uLabels:{value:null},uLabelBits:{value:0},uVisibleBits:{value:0},uLayerColor:{value:Array.from({length:MAX_ANATOMY_LAYERS},()=>new THREE.Vector4())},
        uActiveLayerCount:{value:volume.compartments.length},uActiveLayers:{value:Float32Array.from({length:MAX_ANATOMY_LAYERS},(_,i)=>i)},
        uContextLayerCount:{value:0},uContextLayers:{value:new Float32Array(MAX_ANATOMY_LAYERS)},
        uExtraClipCount:{value:0},uExtraClips:{value:Array.from({length:7},()=>new THREE.Vector4())},
        uShadows:{value:true},uIsolate:{value:false},uInteractive:{value:false},
        uSampleIndex:{value:0},uSurfaceCount:{value:0},
        ...Object.fromEntries(Array.from({length:5},(_,i)=>[[`uSurfaceColor${i}`,{value:null}],[`uSurfaceDepth${i}`,{value:null}]]).flat()),
      }});
    this.rayScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2),this.material));
    this.blitMaterial=new THREE.MeshBasicMaterial({map:this.volumeTarget.texture,depthTest:false,depthWrite:false});
    this.blitScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2),this.blitMaterial));
    this.accumulateScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2),this.accumulateMaterial));
  }
  configure(input: VolumeRenderingSettings | undefined, modality: string, clip: THREE.Plane | null) {
    const settings=volumeSettings(input,modality);this.enabled=settings.enabled&&settings.density>0;
    const signature=JSON.stringify([settings.preset,input?.window,input?.points,modality]);
    if(signature!==this.preset){
      const lut=volumeTransfer(settings.preset,this.window,input,modality);
      this.transfer?.dispose();this.transfer=new THREE.DataTexture(lut.data,1024,1,THREE.RGBAFormat,THREE.FloatType);
      this.transfer.minFilter=this.transfer.magFilter=this.material.uniforms.uLinear.value?THREE.LinearFilter:THREE.NearestFilter;this.transfer.needsUpdate=true;
      this.material.uniforms.uTransfer.value=this.transfer;
      this.automaticMriTransfer=lut.gradientOpacity;
      this.material.uniforms.uRange.value.set(...lut.range);this.preset=signature;
    }
    this.material.uniforms.uCt.value=isCtModality(modality);
    this.material.uniforms.uGradientOpacity.value=this.automaticMriTransfer&&this.shellContext;
    this.material.uniforms.uShadows.value=input?.shadows!==false;
    this.material.uniforms.uIsolate.value=input?.isolateLabels===true;
    this.material.uniforms.uDensity.value=settings.density;
    this.material.uniforms.uClipEnabled.value=!!clip;
    if(clip)this.material.uniforms.uClipPlane.value.set(clip.normal.x,clip.normal.y,clip.normal.z,clip.constant);
  }
  labels(texture:THREE.Data3DTexture,visible:Record<string,boolean>,settings:ViewerWorkspaceSettings,planes:THREE.Plane[]){
    this.shellContext=settings.contextStyle==='shell';
    this.material.uniforms.uGradientOpacity.value=this.automaticMriTransfer&&this.shellContext;
    const u=this.material.uniforms;u.uLabels.value=texture;u.uSkullMode.value=settings.skullMode==='transparent'?2:settings.skullMode?1:0;let bits=0,all=0,active=0,context=0;
    this.layers.forEach((layer,i)=>{const a=layerAppearance(layer.name,layer.color,settings,layer.kind);
      // Keep declared context even when hidden: the scalar threshold fallback
      // must not resurrect a hidden source label or classify contrast as bone.
      if(layer.kind==='bone'||layer.kind==='skull')u.uContextLayers.value[context++]=i;
      u.uLayerEnabled.value[i]=visible[layer.name]!==false?1:0;
      if(visible[layer.name]!==false){all|=1<<i;u.uActiveLayers.value[active++]=i;}
      if(visible[layer.name]!==false && a.opacity<1 && a.opacity>0)bits|=1<<i;
      const color=new THREE.Color(a.color);u.uLayerColor.value[i].set(color.r,color.g,color.b,a.opacity);
    });
    u.uActiveLayerCount.value=active;
    u.uContextLayerCount.value=context;
    u.uLabelBits.value=bits;u.uVisibleBits.value=all;u.uExtraClipCount.value=Math.min(7,planes.length);
    planes.slice(0,7).forEach((p,i)=>u.uExtraClips.value[i].set(p.normal.x,p.normal.y,p.normal.z,p.constant));
  }
  setLayers(layers:ViewerVolume["compartments"],codes:Array<[number,number]>) {
    this.layers=layers;this.material.uniforms.uLayerCount.value=layers.length;
    codes.forEach((code,i)=>this.material.uniforms.uLayerCodes.value[i].set(...code));
  }
  get active(){return this.enabled;}
  render(scene: THREE.Scene, camera: THREE.PerspectiveCamera, width: number, height: number,
    interactive: boolean, restoreViewport: ()=>void, longEdge=360) {
    if(!this.enabled){this.renderer.render(scene,camera);return;}
    const pixelRatio=this.renderer.getPixelRatio();
    const scale=Math.min(pixelRatio,(interactive?longEdge:1300)/Math.max(width,height));
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
    uniforms.uSampleIndex.value=interactive?0:this.samples;
    this.renderer.getClearColor(this.background);
    const clearAlpha=this.renderer.getClearAlpha();
    const hidden:THREE.Object3D[]=[];
    scene.traverse(o=>{if(o instanceof THREE.Mesh && o.visible && o.userData.compartment && !Array.isArray(o.material) && o.material.opacity<1){hidden.push(o);o.visible=false;}});
    try {
      this.renderer.setClearColor(0,0);
      if(interactive||this.samples===0){
        this.renderer.setRenderTarget(sceneTarget);this.renderer.clear();this.renderer.render(scene,camera);
        hidden.forEach(o=>o.visible=true);
        const surfaces=this.surfacePass.render(this.renderer,scene,camera,sceneTarget.depthTexture!,w,h,interactive);
        uniforms.uSurfaceCount.value=surfaces.count;
        for(let i=0;i<5;i++){
          uniforms[`uSurfaceColor${i}`].value=i===surfaces.tailIndex?this.surfacePass.tail.texture:this.surfacePass.targets[i].texture;
          uniforms[`uSurfaceDepth${i}`].value=this.surfacePass.targets[i].depthTexture;
        }
      }
      this.renderer.setRenderTarget(volumeTarget);this.renderer.clear();this.renderer.render(this.rayScene,this.camera);
      if(!interactive){
        const current=this.accumulation[this.samples%2],previous=this.accumulation[(this.samples+1)%2];
        if(current.width!==w||current.height!==h){current.setSize(w,h);previous.setSize(w,h);}
        const u=this.accumulateMaterial.uniforms;u.uCurrent.value=volumeTarget.texture;u.uPrevious.value=previous.texture;u.uWeight.value=1/(this.samples+1);
        this.renderer.setRenderTarget(current);this.renderer.clear();this.renderer.render(this.accumulateScene,this.camera);
        this.blitMaterial.map=current.texture;this.samples++;
      }else this.samples=0;
    } finally {
      hidden.forEach(o=>o.visible=true);
      this.renderer.setClearColor(this.background,clearAlpha);
      this.renderer.setRenderTarget(null);restoreViewport();
    }
    this.renderer.render(this.blitScene,this.camera);
  }
  dispose(){
    this.sceneTarget.dispose();this.volumeTarget.dispose();if(this.ownsTexture)this.texture.dispose();this.transfer?.dispose();
    this.interactionTargets?.scene.dispose();this.interactionTargets?.volume.dispose();
    this.surfacePass.dispose();this.accumulation.forEach(t=>t.dispose());this.accumulateMaterial.dispose();
    for(const scene of [this.rayScene,this.blitScene,this.accumulateScene])scene.traverse(child=>{if(child instanceof THREE.Mesh)child.geometry.dispose();});
    this.material.dispose();this.blitMaterial.dispose();
  }
}
