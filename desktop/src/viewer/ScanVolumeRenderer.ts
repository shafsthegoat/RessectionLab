import {packAnatomyLabels,MAX_ANATOMY_LAYERS} from './labelPacking';
import * as THREE from 'three';
import type { ViewerVolume } from './contracts';
import { inverseAffine, rasAffine } from './coordinates.ts';
import { isCtModality, volumeSettings, volumeTransfer } from './volumeTransfer.ts';
import {finishOpticalPreparation, opticalAtlasSteps, volumeOpticalAtlas, type OpticalSource} from './volumeOpticalDensity.ts';
import {layerAppearance} from './workspaceSettings';
import type {ViewerWorkspaceSettings} from './workspaceSettings';
import type { VolumeRenderingSettings } from './volumeTransfer';
import { volumeVertexShader, volumeFragmentShader } from './volumeRaycastShaders';
import {SurfaceVolumePass} from './SurfaceVolumePass';
import {renderTargetBytes} from './renderResourceEstimate';

type VolumeTargets = {
  scene: THREE.WebGLRenderTarget;
  volume: THREE.WebGLRenderTarget;
  accumulation: THREE.WebGLRenderTarget[];
  surfaces: SurfaceVolumePass;
};

function targetBytes(targets:VolumeTargets) {
  return [targets.scene,targets.volume,...targets.accumulation,...targets.surfaces.targets,
    ...targets.surfaces.cranial.targets.values(),targets.surfaces.tail].reduce((sum,target)=>sum+renderTargetBytes(target),0);
}

/** Export owns its targets so a tile cannot resize or invalidate live attachments. */
export class ScanVolumeExportTargets implements VolumeTargets {
  readonly scene = new THREE.WebGLRenderTarget(1,1,{type:THREE.HalfFloatType});
  readonly volume = new THREE.WebGLRenderTarget(1,1,{type:THREE.HalfFloatType,depthBuffer:false});
  readonly accumulation = Array.from({length:2},()=>new THREE.WebGLRenderTarget(1,1,{type:THREE.HalfFloatType,depthBuffer:false}));
  readonly surfaces = new SurfaceVolumePass();

  constructor() { this.scene.depthTexture=new THREE.DepthTexture(1,1,THREE.UnsignedIntType); }
  get estimatedBytes() { return targetBytes(this); }
  dispose() {
    this.scene.dispose();
    this.volume.dispose();
    this.accumulation.forEach(target=>target.dispose());
    this.surfaces.dispose();
  }
}

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
  private transferLut?: ReturnType<typeof volumeTransfer>;
  private opticalAtlasReady = false;
  private opticalPreparation: {steps:ReturnType<typeof opticalAtlasSteps>;timer:ReturnType<typeof setTimeout> | null} | null = null;
  private opticalPreparationFailed = false;
  private readonly sourceOptics: OpticalSource;
  private preset = '';
  private automaticMriTransfer=false;
  private shellContext=false;
  private enabled = true;
  private readonly spacing:number;
  readonly supportTexture?:THREE.Data3DTexture;
  private layers: ViewerVolume["compartments"];
  private readonly ownsTexture: boolean;
  private readonly background = new THREE.Color();
  private readonly surfacePass=new SurfaceVolumePass();
  private readonly accumulation=Array.from({length:2},()=>new THREE.WebGLRenderTarget(1,1,{type:THREE.HalfFloatType,depthBuffer:false}));
  private readonly accumulateScene=new THREE.Scene();
  private readonly accumulateMaterial=new THREE.ShaderMaterial({glslVersion:THREE.GLSL3,vertexShader:volumeVertexShader,
    fragmentShader:'precision highp float; in vec2 vUv; out vec4 outColor; uniform sampler2D uCurrent; uniform sampler2D uPrevious; uniform float uWeight; void main(){vec4 current=texture(uCurrent,vUv);outColor=uWeight==1.0?current:mix(texture(uPrevious,vUv),current,uWeight);}',
    depthTest:false,depthWrite:false,toneMapped:false,uniforms:{uCurrent:{value:null},uPrevious:{value:null},uWeight:{value:1}}});
  private samples=0;
  invalidate(){this.samples=0;}
  get needsRefinement(){return this.enabled&&this.samples>0&&this.samples<4;}
  get opticsPending(){return this.opticalPreparation!==null;}

  constructor(private readonly renderer: THREE.WebGLRenderer, volume: ViewerVolume,
    private readonly window: [number,number], codes?:Array<[number,number]>, sourceTexture?:THREE.Data3DTexture,
    private readonly onOpticsReady?:()=>void) {
    this.layers=volume.compartments;
    const affine=rasAffine(volume.affine,volume.frame),inverse=inverseAffine(affine);
    this.spacing=1/Math.max(...inverse.slice(0,3).map(row=>Math.hypot(...row.slice(0,3))));
    const worldToVoxel=new THREE.Matrix4().set(...inverse.flat() as Parameters<THREE.Matrix4['set']>);
    const shape=new THREE.Vector3(...volume.shape);
    this.sourceOptics={signal:volume.mri,shape:volume.shape,affine,inverse,support:volume.headSupport?.data};
    this.ownsTexture=!sourceTexture;
    this.texture=sourceTexture??new THREE.Data3DTexture(volume.mri,volume.shape[2],volume.shape[1],volume.shape[0]);
    this.texture.format=THREE.RedFormat;this.texture.type=THREE.FloatType;
    const linear=renderer.extensions.has('OES_texture_float_linear');
    this.texture.minFilter=this.texture.magFilter=linear?THREE.LinearFilter:THREE.NearestFilter;
    this.texture.unpackAlignment=1;this.texture.needsUpdate=true;
    if(volume.headSupport){
      this.supportTexture=new THREE.Data3DTexture(volume.headSupport.data,volume.shape[2],volume.shape[1],volume.shape[0]);
      this.supportTexture.format=THREE.RedFormat;this.supportTexture.type=THREE.UnsignedByteType;
      this.supportTexture.minFilter=this.supportTexture.magFilter=THREE.NearestFilter;
      this.supportTexture.unpackAlignment=1;this.supportTexture.needsUpdate=true;
    }
    this.sceneTarget=new THREE.WebGLRenderTarget(1,1,{type:THREE.HalfFloatType,depthBuffer:true});
    this.sceneTarget.depthTexture=new THREE.DepthTexture(1,1,THREE.UnsignedIntType);
    this.volumeTarget=new THREE.WebGLRenderTarget(1,1,{type:THREE.HalfFloatType,depthBuffer:false});
    this.material=new THREE.ShaderMaterial({glslVersion:THREE.GLSL3,vertexShader:volumeVertexShader,
      fragmentShader:volumeFragmentShader,depthTest:false,depthWrite:false,toneMapped:false,uniforms:{
        uHeadSupport:{value:this.supportTexture??null},uUseHeadSupport:{value:!!this.supportTexture},
        uVolume:{value:this.texture},uTransfer:{value:null},uScene:{value:this.sceneTarget.texture},uDrawScalar:{value:true},
        uOpticalSize:{value:new THREE.Vector2()},uOpticalGrid:{value:new THREE.Vector3()},uOpticalColumns:{value:1},
        uOpticalCache:{value:false},
        uLabelWorldToVoxel:{value:worldToVoxel},uLabelShape:{value:shape},
        uDepth:{value:this.sceneTarget.depthTexture},uWorldToVoxel:{value:worldToVoxel},
        uCameraWorld:{value:new THREE.Matrix4()},uInverseProjection:{value:new THREE.Matrix4()},
        uShape:{value:shape},uRange:{value:new THREE.Vector2()},
        uSkullMode:{value:0},uCt:{value:false},uScanPresentation:{value:false},uGradientOpacity:{value:false},uDensity:{value:.45},uStep:{value:this.spacing*.7},uLinear:{value:linear},
        uClipEnabled:{value:false},uClipPlane:{value:new THREE.Vector4()},
        uBackground:{value:this.background},
        uLayerCount:{value:volume.compartments.length},uLayerCodes:{value:(codes??packAnatomyLabels(volume).codes).map(c=>new THREE.Vector2(...c))},uLayerEnabled:{value:new Float32Array(MAX_ANATOMY_LAYERS).fill(1)},
        uLabels:{value:null},uLabelBits:{value:0},uVisibleBits:{value:0},uLayerColor:{value:Array.from({length:MAX_ANATOMY_LAYERS},()=>new THREE.Vector4())},
        uActiveLayerCount:{value:volume.compartments.length},uActiveLayers:{value:Float32Array.from({length:MAX_ANATOMY_LAYERS},(_,i)=>i)},
        uContextLayerCount:{value:0},uContextLayers:{value:new Float32Array(MAX_ANATOMY_LAYERS)},
        uExtraClipCount:{value:0},uExtraClips:{value:Array.from({length:7},()=>new THREE.Vector4())},
        uShadows:{value:true},uIsolate:{value:false},uInteractive:{value:false},
        uSampleIndex:{value:0},uSurfaceCount:{value:0},
        uExport:{value:false},uSampleOrigin:{value:new THREE.Vector2()},
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
      this.cancelOpticalPreparation();
      const lut=volumeTransfer(settings.preset,this.window,input,modality);
      this.transferLut=lut;
      this.installOpticalAtlas(volumeOpticalAtlas(lut));
      this.opticalPreparationFailed=false;
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
    if(!this.enabled||input?.shadows===false||input?.isolateLabels)this.cancelOpticalPreparation();
  }

  private installOpticalAtlas(atlas:ReturnType<typeof volumeOpticalAtlas>){
    const u=this.material.uniforms;
    this.transfer?.dispose();
    this.transfer=new THREE.DataTexture(atlas.data,atlas.width,atlas.height,THREE.RGBAFormat,THREE.HalfFloatType);
    this.transfer.minFilter=this.transfer.magFilter=THREE.LinearFilter;
    this.transfer.needsUpdate=true;
    u.uTransfer.value=this.transfer;
    u.uOpticalSize.value.set(atlas.width,atlas.height);
    u.uOpticalGrid.value.set(...atlas.grid);
    u.uOpticalColumns.value=atlas.columns;
    this.opticalAtlasReady=atlas.height>1;
  }

  private cancelOpticalPreparation(){
    const task=this.opticalPreparation;
    if(task?.timer!=null)clearTimeout(task.timer);
    this.opticalPreparation=null;
    return task?.steps;
  }

  private prepareOpticalAtlas(exporting:boolean){
    const u=this.material.uniforms;
    const overrides=u.uIsolate.value || (u.uCt.value &&
      (u.uSkullMode.value===1 || (!u.uScanPresentation.value && (u.uContextLayerCount.value>0 || u.uSkullMode.value===2))));
    const useCache=u.uShadows.value&&!overrides;
    if(!useCache)this.cancelOpticalPreparation();
    if(useCache&&!this.opticalAtlasReady&&!this.opticalPreparationFailed&&this.transferLut){
      if(exporting||!this.onOpticsReady){
        const steps=this.cancelOpticalPreparation()??opticalAtlasSteps(this.transferLut,this.sourceOptics);
        this.installOpticalAtlas(finishOpticalPreparation(steps));
      }else if(!this.opticalPreparation){
        const task={steps:opticalAtlasSteps(this.transferLut,this.sourceOptics),timer:null as ReturnType<typeof setTimeout>|null};
        this.opticalPreparation=task;
        const advance=()=>{
          if(this.opticalPreparation!==task)return;
          const deadline=performance.now()+5;
          try {
            do {
              const next=task.steps.next();
              if(next.done){
                this.installOpticalAtlas(next.value);
                this.opticalPreparation=null;
                break;
              }
            }while(performance.now()<deadline);
          }catch(error){
            this.opticalPreparationFailed=true;
            this.opticalPreparation=null;
            console.warn('Volume optical cache unavailable; using exact probes.',error);
          }
          if(this.opticalPreparation===task)task.timer=setTimeout(advance,0);
          else this.onOpticsReady?.();
        };
        task.timer=setTimeout(advance,0);
      }
    }
    u.uOpticalCache.value=useCache&&this.opticalAtlasReady;
  }
  labels(texture:THREE.Data3DTexture,visible:Record<string,boolean>,settings:ViewerWorkspaceSettings,planes:THREE.Plane[]){
    this.shellContext=settings.contextStyle==='shell';
    this.material.uniforms.uScanPresentation.value=settings.presentation==='scan'&&settings.operativeView!=='surgeon';
    this.material.uniforms.uGradientOpacity.value=this.automaticMriTransfer&&this.shellContext;
    const u=this.material.uniforms;u.uLabels.value=texture;u.uSkullMode.value=settings.skullMode==='transparent'?2:settings.skullMode?1:0;let bits=0,all=0,active=0,context=0;
    this.layers.forEach((layer,i)=>{const a=layerAppearance(layer.name,layer.color,settings,layer.displayKind??layer.kind);
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
  get estimatedTargetBytes(){
    const live=targetBytes({scene:this.sceneTarget,volume:this.volumeTarget,accumulation:this.accumulation,surfaces:this.surfacePass});
    const drag=this.interactionTargets?renderTargetBytes(this.interactionTargets.scene)+renderTargetBytes(this.interactionTargets.volume):0;
    return live+drag+(this.transfer?.image.data?.byteLength??0);
  }
  render(scene: THREE.Scene, camera: THREE.PerspectiveCamera, width: number, height: number,
    interactive: boolean, restoreViewport: ()=>void, longEdge=360, cranialSurfaces=false) {
    if(!this.enabled&&!cranialSurfaces){this.renderer.render(scene,camera);return;}
    const pixelRatio=this.renderer.getPixelRatio();
    // Both scalar and translucent-surface compositing reuse the same live
    // targets. Linked slices and the high-resolution export have separate budgets.
    const scale=Math.min(pixelRatio,(interactive?longEdge:1024)/Math.max(width,height));
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
    try {
      this.blitMaterial.map=this.renderSample(scene,camera,{scene:sceneTarget,volume:volumeTarget,accumulation:this.accumulation,surfaces:this.surfacePass},
        w,h,interactive,this.samples).texture;
      this.samples=interactive?0:this.samples+1;
    } finally {
      this.renderer.setRenderTarget(null);
      restoreViewport();
    }
    this.renderer.render(this.blitScene,this.camera);
    return {width:w,height:h,samples:interactive?1:this.samples,opticsPending:this.opticsPending};
  }

  renderExportSample(scene:THREE.Scene,camera:THREE.PerspectiveCamera,targets:ScanVolumeExportTargets,
    width:number,height:number,sample:number,originY:number) {
    return this.renderSample(scene,camera,targets,width,height,false,sample,originY);
  }

  private renderSample(scene:THREE.Scene,camera:THREE.PerspectiveCamera,targets:VolumeTargets,
    w:number,h:number,interactive:boolean,sample:number,exportOriginY?:number) {
    const {scene:sceneTarget,volume:volumeTarget,accumulation,surfaces:surfacePass}=targets;
    if(sceneTarget.width!==w||sceneTarget.height!==h){sceneTarget.setSize(w,h);volumeTarget.setSize(w,h);}
    const isExport=exportOriginY!==undefined;
    if(this.enabled)this.prepareOpticalAtlas(isExport);
    const uniforms=this.material.uniforms;
    uniforms.uDrawScalar.value=this.enabled;
    uniforms.uScene.value=sceneTarget.texture;uniforms.uDepth.value=sceneTarget.depthTexture;
    camera.updateMatrixWorld();
    uniforms.uCameraWorld.value.copy(camera.matrixWorld);
    uniforms.uInverseProjection.value.copy(camera.projectionMatrixInverse);
    uniforms.uStep.value=this.spacing*(interactive ? 3 : isExport ? .5 : uniforms.uScanPresentation.value ? 1 : .6);
    uniforms.uInteractive.value=interactive;
    uniforms.uSampleIndex.value=interactive?0:sample;
    uniforms.uExport.value=isExport;
    uniforms.uSampleOrigin.value.set(0,exportOriginY??0);
    this.renderer.getClearColor(this.background);
    const clearAlpha=this.renderer.getClearAlpha();
    const hidden:THREE.Object3D[]=[];
    try {
      this.renderer.setClearColor(0,0);
      if(interactive||sample===0){
        scene.traverse(o=>{if(o instanceof THREE.Mesh && o.visible && o.userData.compartment && !Array.isArray(o.material) && o.material.opacity<1){hidden.push(o);o.visible=false;}});
        this.renderer.setRenderTarget(sceneTarget);this.renderer.clear();this.renderer.render(scene,camera);
        hidden.forEach(o=>o.visible=true);
        const surfaces=surfacePass.render(this.renderer,scene,camera,sceneTarget.depthTexture!,w,h,interactive);
        uniforms.uSurfaceCount.value=surfaces.count;
        for(let i=0;i<5;i++){
          uniforms[`uSurfaceColor${i}`].value=i===surfaces.tailIndex?surfacePass.tail.texture:surfacePass.targets[i].texture;
          uniforms[`uSurfaceDepth${i}`].value=surfacePass.targets[i].depthTexture;
        }
      }
      this.renderer.setRenderTarget(volumeTarget);this.renderer.clear();this.renderer.render(this.rayScene,this.camera);
      if(interactive||isExport&&!this.enabled)return volumeTarget;
      const current=accumulation[sample%2],previous=accumulation[(sample+1)%2];
      if(current.width!==w||current.height!==h){current.setSize(w,h);previous.setSize(w,h);}
      const u=this.accumulateMaterial.uniforms;u.uCurrent.value=volumeTarget.texture;u.uPrevious.value=previous.texture;u.uWeight.value=1/(sample+1);
      this.renderer.setRenderTarget(current);this.renderer.clear();this.renderer.render(this.accumulateScene,this.camera);
      return current;
    } finally {
      hidden.forEach(o=>o.visible=true);
      this.renderer.setClearColor(this.background,clearAlpha);
    }
  }
  dispose(){
    this.cancelOpticalPreparation();
    this.sceneTarget.dispose();this.volumeTarget.dispose();if(this.ownsTexture)this.texture.dispose();this.supportTexture?.dispose();this.transfer?.dispose();
    this.interactionTargets?.scene.dispose();this.interactionTargets?.volume.dispose();
    this.surfacePass.dispose();this.accumulation.forEach(t=>t.dispose());this.accumulateMaterial.dispose();
    for(const scene of [this.rayScene,this.blitScene,this.accumulateScene])scene.traverse(child=>{if(child instanceof THREE.Mesh)child.geometry.dispose();});
    this.material.dispose();this.blitMaterial.dispose();
  }
}
