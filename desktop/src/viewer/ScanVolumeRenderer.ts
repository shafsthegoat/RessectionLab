import {packAnatomyLabels,MAX_ANATOMY_LAYERS} from './labelPacking';
import * as THREE from 'three';
import type { ViewerVolume } from './contracts';
import { inverseAffine, rasAffine, volumeBounds } from './coordinates.ts';
import { canBalanceTofContext, isCtAngiography, isCtModality, volumeSettings, volumeTransfer, zeroExtinctionLevel } from './volumeTransfer.ts';
import {balancedTofDensity, CT_EXTERIOR_AIR_HU, CT_EXTERIOR_TEXTURE_LIMIT_BYTES, ctExteriorDisplaySignal, ctOpticalCoverage, finishOpticalPreparation, opticalAtlasSteps, OPTICAL_KEY_LIGHTS, volumeOpticalAtlas, type OpticalContext, type OpticalSource} from './volumeOpticalDensity.ts';
import {layerAppearance} from './workspaceSettings';
import type {ViewerWorkspaceSettings} from './workspaceSettings';
import type { CinematicBone, VolumeRenderingSettings } from './volumeTransfer';
import { volumeVertexShader, volumeFragmentShader } from './volumeRaycastShaders';
import {SurfaceVolumePass} from './SurfaceVolumePass';
import {renderTargetBytes} from './renderResourceEstimate';
import {ctaCutKey,installCtaVesselLight} from './ctaVesselLight';
import {CtaVesselShadow} from './ctaVesselShadow';
import {renderAnatomyScene,type VesselSlabContext} from './vesselSlabContext';
import {CINEMATIC_VESSEL_ALBEDO,CinematicVolumePass} from './CinematicVolumePass';
import {cinematicTagClass,type CinematicTagLayer} from './cinematicTags.ts';

type VolumeTargets = {
  scene: THREE.WebGLRenderTarget;
  volume: THREE.WebGLRenderTarget;
  accumulation: THREE.WebGLRenderTarget[];
  surfaces: SurfaceVolumePass;
};
type OpticalSelection = {source:OpticalSource; context:OpticalContext|null};
const sameOptics=(a:OpticalSelection|undefined,b:OpticalSelection)=>a?.source===b.source&&a.context===b.context;

function targetBytes(targets:VolumeTargets) {
  return [targets.scene,targets.volume,...targets.accumulation,...targets.surfaces.targets,
    ...targets.surfaces.cranial.targets.values(),targets.surfaces.tail].reduce((sum,target)=>sum+renderTargetBytes(target),0);
}

const sourceIds=new WeakMap<object,number>();
let nextSourceId=0;
function sourceIdentity(source:CinematicTagLayer['source']){
  const array=source.kind==='mask'?source.mask:source.kind==='labels'?source.labels:source.probability;
  let id=sourceIds.get(array);
  if(id===undefined){id=++nextSourceId;sourceIds.set(array,id);}
  return source.kind==='labels'?`${id}:${source.value}`:id;
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
  private opticalAtlasSelection?: OpticalSelection;
  private opticalPreparation: {selection:OpticalSelection;steps:ReturnType<typeof opticalAtlasSteps>;timer:ReturnType<typeof setTimeout> | null} | null = null;
  private failedOpticalSelection?: OpticalSelection;
  private opticalCellAxes: THREE.Vector3[]=[];
  private readonly sourceOptics: OpticalSource;
  private readonly nativeAcquisition:boolean;
  private readonly sourceSeriesId?:string;
  private readonly ctaVesselLight={value:false};
  private readonly ctaKeyDirection=new THREE.Vector3();
  private readonly ctaKeyUp=new THREE.Vector3();
  private ctaVesselShadow?:CtaVesselShadow;
  private readonly savedKeyPosition=new THREE.Vector3();
  private anatomyKey?:THREE.DirectionalLight;
  private hasCtaVesselReceiver=false;
  private readonly ctExterior?: ViewerVolume['ctExterior'];
  private exteriorVolume?: {source:OpticalSource; texture:THREE.Data3DTexture} | null;
  private standardCtAir=false;
  private exteriorPresentation=false;
  private showCtExteriorSignal=false;
  private preset = '';
  private standardCtaTransfer=false;
  private calibratedTofDensity?:number|null;
  private activeTofContextDensity?:number;
  private automaticMriTransfer=false;
  private shellContext=false;
  private enabled = true;
  private readonly spacing:number;
  readonly supportTexture?:THREE.Data3DTexture;
  private contextCoverage?: OpticalContext & {mask:Uint8Array;texture:THREE.Data3DTexture};
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
  private cinematicMode=false;
  private cinematicBone?:CinematicBone;
  private cinematic?:CinematicVolumePass;
  private labelState:{visible:Record<string,boolean>;settings:ViewerWorkspaceSettings}={visible:{},settings:{}};
  private labelSets?:ViewerVolume['labelSets'];
  private cinematicHidden=new Set<string>();
  private cinematicSurfaces={count:0,tailIndex:0};
  invalidate(){this.samples=0;this.cinematic?.invalidate();}
  /** Set when cinematic mode shows the standard image instead (MR without a visible vessel layer). */
  private cinematicFallback:string|null=null;
  /** The path-traced volume draws vessels itself; mesh-only vessel passes can be skipped. */
  get cinematicActive(){return this.enabled&&this.cinematicMode&&!!this.cinematic?.ready&&!this.cinematicFallback;}
  /** Path-traced refinement runs back to back until the view has settled, then paced. */
  get cinematicPace(){return this.cinematicActive?this.cinematic!.settled?'paced' as const:'fast' as const:null;}
  get needsRefinement(){return this.cinematicActive?this.cinematic!.needsRefinement:this.enabled&&this.samples>0&&this.samples<4;}
  get opticsPending(){return this.opticalPreparation!==null||(this.enabled&&this.cinematicMode&&!!this.cinematic?.pending);}

  constructor(private readonly renderer: THREE.WebGLRenderer, volume: ViewerVolume,
    private readonly window: [number,number], codes?:Array<[number,number]>, sourceTexture?:THREE.Data3DTexture,
    private readonly onOpticsReady?:()=>void) {
    this.layers=volume.compartments;
    this.labelSets=volume.labelSets;
    this.nativeAcquisition=!!volume.displayIdentity;
    this.sourceSeriesId=volume.displayPreview?undefined:volume.sourceSeriesId;
    const affine=rasAffine(volume.affine,volume.frame),inverse=inverseAffine(affine);
    this.spacing=1/Math.max(...inverse.slice(0,3).map(row=>Math.hypot(...row.slice(0,3))));
    const worldToVoxel=new THREE.Matrix4().set(...inverse.flat() as Parameters<THREE.Matrix4['set']>);
    const shape=new THREE.Vector3(...volume.shape);
    this.sourceOptics={signal:volume.mri,shape:volume.shape,affine,inverse,support:volume.headSupport?.data};
    if(volume.mri.byteLength<=CT_EXTERIOR_TEXTURE_LIMIT_BYTES)this.ctExterior=volume.ctExterior;
    const keyLight=new THREE.Vector3();
    for(const direction of OPTICAL_KEY_LIGHTS)keyLight.add(new THREE.Vector3(...direction).normalize());
    keyLight.normalize();
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
        uAuxiliary:{value:this.supportTexture??null},uUseHeadSupport:{value:!!this.supportTexture},uContextCoverageLayer:{value:-1},
        uVolume:{value:this.texture},uTransfer:{value:null},uScene:{value:this.sceneTarget.texture},uDrawScalar:{value:true},
        uOpticalSize:{value:new THREE.Vector2()},uOpticalGrid:{value:new THREE.Vector3()},uOpticalColumns:{value:1},
        uOpticalDepthRow:{value:0},uOpticalLight:{value:keyLight},
        uOpticalCache:{value:false},uClipOpticalWidth:{value:0},uExtraOpticalWidths:{value:new Float32Array(7)},
        uLabelWorldToVoxel:{value:worldToVoxel},uLabelShape:{value:shape},
        uDepth:{value:this.sceneTarget.depthTexture},uWorldToVoxel:{value:worldToVoxel},
        uCameraWorld:{value:new THREE.Matrix4()},uInverseProjection:{value:new THREE.Matrix4()},
        uRakingBoneCut:{value:false},uCtaCutMaterial:{value:false},
        uCtaCutPlane:{value:new THREE.Vector4()},uCtaCutSampleOffset:{value:new THREE.Vector3()},
        uCtaShadow:{value:null},uCtaShadowProjection:{value:new THREE.Matrix4()},
        uCtaShadowDepthMm:{value:1},uCtaShadowPenumbra:{value:0},
        uShape:{value:shape},uRange:{value:new THREE.Vector2()},
        uZeroExtinctionLevel:{value:0},
        uSkullMode:{value:0},uAcquiredBoneCut:{value:false},uAcquiredCtBone:{value:false},uCt:{value:false},uScanPresentation:{value:false},uGradientOpacity:{value:false},uDensity:{value:.45},uStep:{value:this.spacing*.7},uLinear:{value:linear},
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
  configure(input: VolumeRenderingSettings | undefined, modality: string, clip: THREE.Plane | null, balanceTof=false) {
    const settings=volumeSettings(input,modality);this.enabled=settings.enabled&&settings.density>0;
    const cinematic=input?.mode==='cinematic';
    if(cinematic!==this.cinematicMode)this.invalidate();
    this.cinematicMode=cinematic;
    this.cinematicBone=input?.cinematicBone;
    if(!this.enabled||!cinematic)this.releaseCinematic();
    this.ctaVesselLight.value=false;
    this.showCtExteriorSignal=input?.showCtExteriorSignal===true;
    const signature=JSON.stringify([settings.preset,input?.window,input?.points,modality]);
    if(signature!==this.preset){
      this.cancelOpticalPreparation();
      const lut=volumeTransfer(settings.preset,this.window,input,modality);
      this.transferLut=lut;
      this.installOpticalAtlas(volumeOpticalAtlas(lut));
      this.failedOpticalSelection=undefined;
      this.calibratedTofDensity=undefined;
      this.automaticMriTransfer=lut.gradientOpacity;
      this.material.uniforms.uRange.value.set(...lut.range);
      this.material.uniforms.uZeroExtinctionLevel.value=zeroExtinctionLevel(lut.data);
      this.preset=signature;
    }
    this.material.uniforms.uCt.value=isCtModality(modality);
    this.standardCtaTransfer=this.nativeAcquisition&&isCtAngiography(modality)&&
      settings.preset==='ct-angiography'&&input?.window===undefined&&input?.points===undefined;
    // An explicit overlay preference keeps Scan's scalar field independent of
    // hull visibility in both positions. Legacy and isolate modes retain their transfer.
    this.material.uniforms.uAcquiredCtBone.value=this.nativeAcquisition&&input?.showCtBoneOverlay!==undefined&&!input.isolateLabels;
    this.standardCtAir=this.material.uniforms.uCt.value&&settings.preset!=='mri'&&input?.window===undefined&&input?.points===undefined&&
      !!this.transferLut&&this.transferLut.range[0]>=CT_EXTERIOR_AIR_HU&&this.transferLut.data[3]===0;
    this.material.uniforms.uGradientOpacity.value=this.automaticMriTransfer&&this.shellContext;
    this.material.uniforms.uShadows.value=input?.shadows!==false;
    this.material.uniforms.uIsolate.value=input?.isolateLabels===true;
    const scalarOnly=!this.material.uniforms.uCt.value&&!this.material.uniforms.uIsolate.value?1:0;
    if((this.material.defines.SCALAR_ONLY??0)!==scalarOnly){
      this.material.defines.SCALAR_ONLY=scalarOnly;
      this.material.needsUpdate=true;
    }
    this.activeTofContextDensity=undefined;
    if(this.enabled&&balanceTof&&!clip&&input?.balanceTofContext&&canBalanceTofContext(input,modality)&&this.transferLut){
      if(this.calibratedTofDensity===undefined)this.calibratedTofDensity=balancedTofDensity(this.sourceOptics,this.transferLut)??null;
      this.activeTofContextDensity=this.calibratedTofDensity??undefined;
    }
    this.material.uniforms.uDensity.value=this.activeTofContextDensity??settings.density;
    this.material.uniforms.uClipEnabled.value=!!clip;
    if(clip)this.material.uniforms.uClipPlane.value.set(clip.normal.x,clip.normal.y,clip.normal.z,clip.constant);
    if(!this.enabled||input?.shadows===false||input?.isolateLabels)this.cancelOpticalPreparation();
    else this.cancelStaleOpticalSource();
  }

  private installOpticalAtlas(atlas:ReturnType<typeof volumeOpticalAtlas>,selection?:OpticalSelection){
    const u=this.material.uniforms;
    this.transfer?.dispose();
    this.transfer=new THREE.DataTexture(atlas.data,atlas.width,atlas.height,THREE.RGBAFormat,THREE.HalfFloatType);
    this.transfer.minFilter=this.transfer.magFilter=THREE.LinearFilter;
    this.transfer.needsUpdate=true;
    u.uTransfer.value=this.transfer;
    u.uOpticalSize.value.set(atlas.width,atlas.height);
    u.uOpticalGrid.value.set(...atlas.grid);
    u.uOpticalColumns.value=atlas.columns;
    u.uOpticalDepthRow.value=atlas.depthRow;
    this.opticalAtlasSelection=atlas.height>1?selection:undefined;
    if(selection?.context){
      // Rounded-up source bin widths bound nonuniform final bins as well.
      const {affine,shape}=this.sourceOptics;
      this.opticalCellAxes=shape.map((size,axis)=>new THREE.Vector3(affine[0][axis],affine[1][axis],affine[2][axis])
        .multiplyScalar(Math.ceil(size/atlas.grid[axis])));
    }
  }

  /** Leaving cinematic mode stops its preparation and frees its tags, field and attachments. */
  private releaseCinematic(){
    this.cinematic?.dispose();
    this.cinematic=undefined;
    this.cinematicFallback=null;
    this.cinematicHidden.clear();
    this.cinematicSurfaces={count:0,tailIndex:0};
  }

  private cancelOpticalPreparation(){
    const task=this.opticalPreparation;
    if(task?.timer!=null)clearTimeout(task.timer);
    this.opticalPreparation=null;
    return task?.steps;
  }

  private get showCleanExterior(){
    const u=this.material.uniforms;
    return !!this.ctExterior&&this.enabled&&this.exteriorPresentation&&this.standardCtAir&&!this.showCtExteriorSignal&&
      !u.uClipEnabled.value&&u.uExtraClipCount.value===0&&!u.uIsolate.value;
  }

  private get displayOpticalSource(){
    return this.showCleanExterior&&this.exteriorVolume?this.exteriorVolume.source:this.sourceOptics;
  }

  private selectScalarTexture(){
    const clean=this.showCleanExterior;
    if(clean&&this.ctExterior&&this.exteriorVolume===undefined){
      const signal=ctExteriorDisplaySignal(this.sourceOptics.signal,this.sourceOptics.shape,this.ctExterior);
      this.exteriorVolume=null;
      if(signal){
        const [x,y,z]=this.sourceOptics.shape;
        const texture=new THREE.Data3DTexture(signal,z,y,x);
        texture.format=this.texture.format;
        texture.type=this.texture.type;
        texture.minFilter=this.texture.minFilter;
        texture.magFilter=this.texture.magFilter;
        texture.unpackAlignment=this.texture.unpackAlignment;
        texture.needsUpdate=true;
        this.exteriorVolume={source:{...this.sourceOptics,signal},texture};
      }
    }
    const texture=clean&&this.exteriorVolume?this.exteriorVolume.texture:this.texture;
    const uniform=this.material.uniforms.uVolume;
    const changed=texture!==uniform.value;
    uniform.value=texture;
    return changed;
  }

  private cancelStaleOpticalSource(){
    if(this.opticalPreparation&&this.opticalPreparation.selection.source!==this.displayOpticalSource)
      this.cancelOpticalPreparation();
  }

  private prepareOpticalAtlas(exporting:boolean){
    const u=this.material.uniforms;
    const context=u.uContextCoverageLayer.value>=0?this.contextCoverage??null:null;
    const source=this.displayOpticalSource;
    const selection={source,context};
    const scan=u.uScanPresentation.value;
    const opaqueLabels=u.uSkullMode.value===1&&(!scan||!u.uAcquiredCtBone.value);
    const overrides=u.uIsolate.value || (u.uCt.value && (u.uAcquiredBoneCut.value || !context &&
      (opaqueLabels || (!scan && (u.uContextLayerCount.value>0 || u.uSkullMode.value===2)))));
    const useCache=u.uShadows.value&&!overrides;
    if(!useCache||(this.opticalPreparation&&!sameOptics(this.opticalPreparation.selection,selection)))this.cancelOpticalPreparation();
    if(useCache&&!sameOptics(this.opticalAtlasSelection,selection)&&!sameOptics(this.failedOpticalSelection,selection)&&this.transferLut){
      if(exporting||!this.onOpticsReady){
        const steps=this.cancelOpticalPreparation()??opticalAtlasSteps(this.transferLut,source,context??undefined);
        this.installOpticalAtlas(finishOpticalPreparation(steps),selection);
      }else if(!this.opticalPreparation){
        const task={selection,steps:opticalAtlasSteps(this.transferLut,source,context??undefined),timer:null as ReturnType<typeof setTimeout>|null};
        this.opticalPreparation=task;
        const advance=()=>{
          if(this.opticalPreparation!==task)return;
          const deadline=performance.now()+5;
          try {
            do {
              const next=task.steps.next();
              if(next.done){
                this.installOpticalAtlas(next.value,task.selection);
                this.opticalPreparation=null;
                break;
              }
            }while(performance.now()<deadline);
          }catch(error){
            this.failedOpticalSelection=task.selection;
            this.opticalPreparation=null;
            console.warn('Volume optical cache unavailable; using exact probes.',error);
          }
          if(this.opticalPreparation===task)task.timer=setTimeout(advance,0);
          else this.onOpticsReady?.();
        };
        task.timer=setTimeout(advance,0);
      }
    }
    const ready=useCache&&sameOptics(this.opticalAtlasSelection,selection);
    u.uOpticalCache.value=ready;
    const clipped=u.uClipEnabled.value||u.uExtraClipCount.value>0;
    const mode=ready&&context?(clipped?2:1):0;
    // A prefix sweep is admission-specific. A cut or an unready/isolated
    // atlas goes through the existing shader paths for those conditions.
    const directional=ready&&!clipped&&u.uOpticalDepthRow.value>0?1:0;
    if((this.material.defines.CONTEXT_SHADOWS??0)!==mode||(this.material.defines.DIRECTIONAL_OPTICS??0)!==directional){
      this.material.defines.CONTEXT_SHADOWS=mode;
      this.material.defines.DIRECTIONAL_OPTICS=directional;
      this.material.needsUpdate=true;
    }
    if(mode===2){
      const projectedWidth=(plane:THREE.Vector4)=>this.opticalCellAxes.reduce((sum,axis)=>
        sum+Math.abs(plane.x*axis.x+plane.y*axis.y+plane.z*axis.z),0);
      if(u.uClipEnabled.value)u.uClipOpticalWidth.value=projectedWidth(u.uClipPlane.value);
      for(let i=0;i<u.uExtraClipCount.value;i++)u.uExtraOpticalWidths.value[i]=projectedWidth(u.uExtraClips.value[i]);
    }
  }
  labels(texture:THREE.Data3DTexture,visible:Record<string,boolean>,settings:ViewerWorkspaceSettings,planes:THREE.Plane[]){
    this.labelState={visible,settings};
    this.shellContext=settings.contextStyle==='shell';
    this.exteriorPresentation=settings.presentation==='segmentation'||settings.presentation==='vascular';
    this.material.uniforms.uScanPresentation.value=settings.presentation==='scan'&&settings.operativeView!=='surgeon';
    this.material.uniforms.uGradientOpacity.value=this.automaticMriTransfer&&this.shellContext;
    const u=this.material.uniforms;u.uLabels.value=texture;let bits=0,all=0,active=0,context=0;
    u.uSkullMode.value=settings.skullMode==='transparent'?2:settings.skullMode?1:0;
    u.uAcquiredBoneCut.value=this.exteriorPresentation&&!settings.operativeView&&settings.skullMode==='cutaway'&&
      (u.uClipEnabled.value||planes.length>0);
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
    this.cancelStaleOpticalSource();
  }
  setLayers(layers:ViewerVolume["compartments"],codes:Array<[number,number]>,labelSets=this.labelSets) {
    this.layers=layers;this.labelSets=labelSets;this.material.uniforms.uLayerCount.value=layers.length;
    codes.forEach((code,i)=>this.material.uniforms.uLayerCodes.value[i].set(...code));
  }
  get active(){return this.enabled;}
  setAnatomyKey(light:THREE.DirectionalLight){
    this.anatomyKey=light;
  }
  lightCtaVessel(material:THREE.MeshStandardMaterial,sourceSeries:string|undefined) {
    if(!this.sourceSeriesId||sourceSeries!==this.sourceSeriesId)return;
    material.userData.ctaVesselReceiver=true;
    const u=this.material.uniforms;
    installCtaVesselLight(material,{
      uCtaVesselLight:this.ctaVesselLight,uCtaVesselSignal:u.uVolume,uCtaVesselBone:u.uAuxiliary,
      uCtaVesselWorldToVoxel:u.uWorldToVoxel,uCtaVesselShape:u.uShape,uCtaVesselCut:u.uCtaCutPlane,
    });
  }
  setVesselDepthSeparation(enabled: boolean,hasCtaVesselReceiver=false) {
    this.material.uniforms.uRakingBoneCut.value = enabled;
    this.hasCtaVesselReceiver=hasCtaVesselReceiver;
    if(!enabled)this.ctaVesselLight.value=false;
  }
  get estimatedCtExteriorBytes(){
    const bytes=this.exteriorVolume?.source.signal.byteLength??0;
    return {cpuBytes:bytes,gpuTextureBytes:bytes};
  }
  get estimatedTargetBytes(){
    const live=targetBytes({scene:this.sceneTarget,volume:this.volumeTarget,accumulation:this.accumulation,surfaces:this.surfacePass});
    const drag=this.interactionTargets?renderTargetBytes(this.interactionTargets.scene)+renderTargetBytes(this.interactionTargets.volume):0;
    return live+drag+(this.transfer?.image.data?.byteLength??0)+(this.contextCoverage?.texture.image.data?.byteLength??0)
      +this.estimatedCtExteriorBytes.gpuTextureBytes+(this.ctaVesselShadow?.bytes??0)+(this.cinematic?.bytes??0);
  }

  /** In CT context, interpolate the optical coverage of one
   * independent skull mask. Packed labels stay nearest-filtered for all
   * membership queries. Reuse the auxiliary sampler on scans without a
   * source support field to fit the WebGL2 fragment sampler budget. */
  private configureContextCoverage() {
    const u=this.material.uniforms;
    const index=u.uContextLayers.value[0];
    const layer=this.layers[index];
    const opacity=u.uLayerColor.value[index]?.w;
    const transparent=u.uSkullMode.value===2&&opacity<1;
    const mask=this.enabled&&u.uCt.value&&(transparent||u.uAcquiredBoneCut.value)&&!u.uScanPresentation.value&&
      !u.uIsolate.value&&!this.supportTexture&&u.uContextLayerCount.value===1&&layer&&!layer.labelSet&&
      u.uLayerEnabled.value[index]===1&&opacity>0
      ? layer.mask : undefined;
    if(!mask||mask.length!==this.sourceOptics.signal.length){
      u.uAuxiliary.value=this.supportTexture??null;
      u.uContextCoverageLayer.value=-1;
      return;
    }
    if(this.contextCoverage?.mask!==mask){
      this.contextCoverage?.texture.dispose();
      const [x,y,z]=this.sourceOptics.shape;
      const coverage=ctOpticalCoverage(mask,this.sourceOptics.shape,this.sourceOptics.affine);
      const texture=new THREE.Data3DTexture(coverage,z,y,x);
      texture.format=THREE.RedFormat;
      texture.type=THREE.UnsignedByteType;
      texture.minFilter=texture.magFilter=THREE.LinearFilter;
      texture.unpackAlignment=1;
      texture.needsUpdate=true;
      this.contextCoverage={mask,texture,coverage};
    }
    u.uAuxiliary.value=this.contextCoverage.texture;
    u.uContextCoverageLayer.value=index;
  }
  render(scene: THREE.Scene, camera: THREE.PerspectiveCamera, width: number, height: number,
    interactive: boolean, restoreViewport: ()=>void, longEdge=360, cranialSurfaces=false, context?:VesselSlabContext) {
    if(!this.enabled&&!cranialSurfaces){renderAnatomyScene(this.renderer,scene,camera,context);return;}
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
    const cinematic=this.enabled&&this.cinematicMode?this.prepareCinematic():null;
    try {
      if(cinematic)this.blitMaterial.map=this.renderCinematic(cinematic,scene,camera,sceneTarget,w,h,interactive);
      else {
        this.blitMaterial.map=this.renderSample(scene,camera,{scene:sceneTarget,volume:volumeTarget,accumulation:this.accumulation,surfaces:this.surfacePass},
          w,h,interactive,this.samples,undefined,context).texture;
        this.samples=interactive?0:this.samples+1;
      }
    } finally {
      this.renderer.setRenderTarget(null);
      restoreViewport();
    }
    this.renderer.render(this.blitScene,this.camera);
    if(cinematic){
      const classes=cinematic.tagStats?.classes;
      return {width:w,height:h,samples:interactive?1:cinematic.samples,opticsPending:this.opticsPending,shadowSampling:'path-traced',
        mode:'cinematic' as const,settled:!interactive&&cinematic.settled,filterWeight:interactive?1:cinematic.filterWeight,
        sampleLimit:cinematic.sampleLimit,exportMode:this.exportMode,translucentLayers:this.cinematicSurfaces.count,
        vesselRamps:classes?.map(c=>c.present?{core:c.coreMedian,shell:c.shellMedian,ramp:c.ramp,tagBounded:c.tagBounded}:null)??null};
    }
    const u=this.material.uniforms;
    const shadowSampling=!this.enabled||!u.uShadows.value?'off':!u.uOpticalCache.value?'source-probes':
      this.material.defines.DIRECTIONAL_OPTICS===1?'directional-atlas':u.uContextCoverageLayer.value>=0?'skull-atlas':'source-atlas';
    const ctExteriorDisplay=!!this.exteriorVolume&&u.uVolume.value===this.exteriorVolume.texture;
    return {width:w,height:h,samples:interactive?1:this.samples,opticsPending:this.opticsPending,shadowSampling,
      ...(this.ctaVesselLight.value?{ctaVesselKeyLight:true}:{}),
      ...(this.material.defines.CTA_VESSEL_SHADOW===1?{ctaVesselCastShadow:true}:{}),
      ...(u.uCtaCutMaterial.value?{ctaCutMaterial:true}:{}),
      ctExteriorDisplay,ctExteriorBytes:this.estimatedCtExteriorBytes,tofContextDensity:this.activeTofContextDensity,
      // While cinematic tags are prepared the standard image is shown. If they
      // cannot be prepared, the standard image settles as usual.
      ...(this.enabled&&this.cinematicMode?{mode:'cinematic' as const,
        settled:(!!this.cinematic?.error||!!this.cinematicFallback)&&!interactive&&this.samples>=4,
        ...(this.cinematic?.error?{cinematicUnavailable:this.cinematic.error}:{}),
        ...(this.cinematicFallback?{cinematicFallback:this.cinematicFallback}:{})}:{})};
  }

  /** Visible vessel layers become soft material tags; their meshes are drawn by the volume. */
  private cinematicLayers(){
    const {visible,settings}=this.labelState;
    const layers:CinematicTagLayer[]=[],names:string[]=[];
    for(const layer of this.layers){
      const cls=cinematicTagClass(layer.kind);
      if(cls===null||visible[layer.name]===false||layerAppearance(layer.name,layer.color,settings,layer.displayKind??layer.kind).opacity<=0)continue;
      const labels=layer.labelSet?this.labelSets?.find(set=>set.id===layer.labelSet)?.data:undefined;
      if(layer.labelSet&&(!labels||layer.labelValue===undefined))continue;
      const tags=layer.vesselTags?.length===layer.mask.length?layer.vesselTags:undefined;
      layers.push({cls,source:tags?{kind:'probability',probability:tags}:labels?{kind:'labels',labels,value:layer.labelValue!}:{kind:'mask',mask:layer.mask}});
      names.push(layer.name);
    }
    const key=JSON.stringify(names.map((name,i)=>[name,layers[i].cls,sourceIdentity(layers[i].source)]));
    return {key,layers};
  }

  private prepareCinematic(){
    const u=this.material.uniforms,ct=!!u.uCt.value;
    const pass=this.cinematic??=new CinematicVolumePass({signal:this.sourceOptics.signal,shape:this.sourceOptics.shape,texture:this.texture,
      worldToVoxel:u.uWorldToVoxel.value,voxelMm:this.spacing,ct,floatLinear:!!u.uLinear.value},
      {uClipEnabled:u.uClipEnabled,uClipPlane:u.uClipPlane,uExtraClipCount:u.uExtraClipCount,uExtraClips:u.uExtraClips,uBackground:u.uBackground},
      this.onOpticsReady?()=>this.onOpticsReady!():undefined);
    const {visible,settings}=this.labelState;
    const appearance=(layer:ViewerVolume['compartments'][number])=>layerAppearance(layer.name,layer.color,settings,layer.displayKind??layer.kind);
    const skull=this.layers.filter(layer=>layer.kind==='bone'||layer.kind==='skull');
    const shownSkull=skull.filter(layer=>visible[layer.name]!==false);
    // Solid bone unless the shell is chosen; unset, any cut or an opaque skull
    // display chooses solid so the cut face reads. The see-through shell scales
    // bone extinction by half its layer opacity.
    const cut=u.uClipEnabled.value||u.uExtraClipCount.value>0;
    const bone=this.cinematicBone??(settings.skullMode!=='transparent'||cut?'solid':'shell');
    const boneWeight=!ct?0:skull.length&&!shownSkull.length?0:bone==='solid'?1:
      .5*(shownSkull.length?Math.max(...shownSkull.map(layer=>appearance(layer).opacity)):.08);
    const {key,layers}=this.cinematicLayers();
    pass.prepare(key,layers,{ct,boneWeight,window:this.window},!this.onOpticsReady);
    this.cinematicFallback=null;
    if(pass.error||!pass.ready)return null;
    // MR context alone is a dim fog; without a vessel layer the standard image says more.
    if(!ct&&!pass.tagStats?.channels){this.cinematicFallback='no-vessel-layer';return null;}
    // Meshes and colours follow the installed tags, which stay an earlier
    // layer set until a newly requested one has been prepared.
    const names=(JSON.parse(pass.key!) as Array<[string]>).map(([name])=>name);
    const tagged=this.layers.filter(layer=>names.includes(layer.name));
    // Arteries and unclassified vessels share the artery red; veins and sinuses are blue.
    pass.setLighting({
      vesselColors:[new THREE.Color(CINEMATIC_VESSEL_ALBEDO.artery),new THREE.Color(CINEMATIC_VESSEL_ALBEDO.vein)],
      boneColor:new THREE.Color(shownSkull.length?appearance(shownSkull[0]).color:'#e3d6bf')});
    this.cinematicHidden=new Set([...names,...(ct?skull.map(layer=>layer.name):[])]);
    return pass;
  }

  /** Opaque meshes not represented by the volume stop primary rays at their
   * depth. Tagged vessels and CT bone are drawn by the volume; translucent
   * meshes are peeled as in the standard path and composited over the traced
   * image wherever they lie in front of it. */
  private renderCinematic(pass:CinematicVolumePass,scene:THREE.Scene,camera:THREE.PerspectiveCamera,
    sceneTarget:THREE.WebGLRenderTarget,w:number,h:number,interactive:boolean) {
    if(sceneTarget.width!==w||sceneTarget.height!==h)sceneTarget.setSize(w,h);
    this.renderer.getClearColor(this.background);
    const clearAlpha=this.renderer.getClearAlpha();
    const hidden:THREE.Object3D[]=[],translucent:THREE.Object3D[]=[];
    try {
      this.renderer.setClearColor(0,0);
      if(interactive||pass.samples===0){
        scene.traverse(o=>{if(!(o instanceof THREE.Mesh)||!o.visible||!o.userData.compartment||Array.isArray(o.material))return;
          if(this.cinematicHidden.has(o.userData.compartment)){hidden.push(o);o.visible=false;}
          else if(o.material.opacity<1){translucent.push(o);o.visible=false;}});
        this.renderer.setRenderTarget(sceneTarget);this.renderer.clear();
        this.renderer.render(scene,camera);
        translucent.forEach(o=>o.visible=true);
        this.cinematicSurfaces=this.surfacePass.render(this.renderer,scene,camera,sceneTarget.depthTexture!,w,h,interactive);
      }
      const {count,tailIndex}=this.cinematicSurfaces,surfaces=this.surfacePass;
      return pass.render(this.renderer,camera,sceneTarget,w,h,interactive,count?{count,
        colors:Array.from({length:count},(_,i)=>i===tailIndex?surfaces.tail.texture:surfaces.targets[i].texture),
        depths:Array.from({length:count},(_,i)=>surfaces.targets[i].depthTexture)}:undefined);
    } finally {
      for(const o of [...hidden,...translucent])o.visible=true;
      this.renderer.setClearColor(this.background,clearAlpha);
    }
  }

  /** The 3D PNG is always the standard ray-cast image: in cinematic mode the
   * path-traced still is not exported. */
  get exportMode(){return this.enabled&&this.cinematicMode?'standard' as const:undefined;}

  renderExportSample(scene:THREE.Scene,camera:THREE.PerspectiveCamera,targets:ScanVolumeExportTargets,
    width:number,height:number,sample:number,originY:number,context?:VesselSlabContext) {
    // Deliberately the standard image in either mode; see exportMode.
    return this.renderSample(scene,camera,targets,width,height,false,sample,originY,context);
  }

  private renderSample(scene:THREE.Scene,camera:THREE.PerspectiveCamera,targets:VolumeTargets,
    w:number,h:number,interactive:boolean,sample:number,exportOriginY?:number,context?:VesselSlabContext) {
    const {scene:sceneTarget,volume:volumeTarget,accumulation,surfaces:surfacePass}=targets;
    // Cinematic frames resize the shared scene target alone.
    if(sceneTarget.width!==w||sceneTarget.height!==h)sceneTarget.setSize(w,h);
    if(volumeTarget.width!==w||volumeTarget.height!==h)volumeTarget.setSize(w,h);
    const isExport=exportOriginY!==undefined;
    if(this.selectScalarTexture()&&!isExport){this.invalidate();sample=0;}
    this.configureContextCoverage();
    if(this.enabled)this.prepareOpticalAtlas(isExport);
    const uniforms=this.material.uniforms;
    const hasPrimaryCut=uniforms.uClipEnabled.value;
    uniforms.uCtaCutMaterial.value=this.standardCtaTransfer&&uniforms.uAcquiredBoneCut.value&&
      uniforms.uContextCoverageLayer.value>=0&&Number(hasPrimaryCut)+uniforms.uExtraClipCount.value===1;
    if(uniforms.uCtaCutMaterial.value){
      const plane=hasPrimaryCut?uniforms.uClipPlane.value:uniforms.uExtraClips.value[0];
      uniforms.uCtaCutPlane.value.copy(plane);
      uniforms.uCtaCutSampleOffset.value.set(plane.x,plane.y,plane.z)
        .applyMatrix3(new THREE.Matrix3().setFromMatrix4(uniforms.uWorldToVoxel.value)).multiplyScalar(.75);
    }
    const boneContext=this.layers[uniforms.uContextCoverageLayer.value];
    this.ctaVesselLight.value=this.hasCtaVesselReceiver&&uniforms.uCtaCutMaterial.value&&uniforms.uRakingBoneCut.value&&
      uniforms.uShadows.value&&uniforms.uLinear.value&&!!this.sourceSeriesId&&boneContext?.sourceSeries===this.sourceSeriesId;
    // This path uses the independent skull mask. It does not sample packed
    // labels, so the light-space depth can occupy that fragment sampler.
    const castVesselShadow=this.ctaVesselLight.value&&!uniforms.uIsolate.value;
    const castMode=Number(castVesselShadow);
    if((this.material.defines.CTA_VESSEL_SHADOW??0)!==castMode){
      this.material.defines.CTA_VESSEL_SHADOW=castMode;
      this.material.needsUpdate=true;
    }
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
    const anatomyKey=this.ctaVesselLight.value?this.anatomyKey:undefined;
    try {
      if(this.ctaVesselLight.value)this.ctaKeyDirection.set(...ctaCutKey).transformDirection(camera.matrixWorld);
      if(anatomyKey){
        this.savedKeyPosition.copy(anatomyKey.position);
        anatomyKey.position.copy(anatomyKey.target.position).addScaledVector(this.ctaKeyDirection,300);
      }
      if(castVesselShadow){
        const shadow=this.ctaVesselShadow??=new CtaVesselShadow(volumeBounds(this.sourceOptics.shape,this.sourceOptics.affine,true));
        this.ctaKeyUp.setFromMatrixColumn(camera.matrixWorld,1);
        if(interactive||sample===0)shadow.capture(this.renderer,scene,this.ctaKeyDirection,this.ctaKeyUp,uniforms.uCtaCutPlane.value);
        uniforms.uCtaShadow.value=shadow.texture;
        uniforms.uCtaShadowProjection.value.copy(shadow.projection);
        uniforms.uCtaShadowDepthMm.value=shadow.depthMm;
        uniforms.uCtaShadowPenumbra.value=shadow.penumbraUv;
      }
      this.renderer.setClearColor(0,0);
      if(interactive||sample===0){
        scene.traverse(o=>{if(o instanceof THREE.Mesh && o.visible && o.userData.compartment && !Array.isArray(o.material) && o.material.opacity<1){hidden.push(o);o.visible=false;}});
        this.renderer.setRenderTarget(sceneTarget);this.renderer.clear();
        renderAnatomyScene(this.renderer,scene,camera,context);
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
      if(anatomyKey)anatomyKey.position.copy(this.savedKeyPosition);
      hidden.forEach(o=>o.visible=true);
      this.renderer.setClearColor(this.background,clearAlpha);
    }
  }
  dispose(){
    this.cancelOpticalPreparation();
    this.cinematic?.dispose();
    this.ctaVesselShadow?.dispose();
    this.sceneTarget.dispose();this.volumeTarget.dispose();if(this.ownsTexture)this.texture.dispose();this.supportTexture?.dispose();this.contextCoverage?.texture.dispose();this.exteriorVolume?.texture.dispose();this.transfer?.dispose();
    this.interactionTargets?.scene.dispose();this.interactionTargets?.volume.dispose();
    this.surfacePass.dispose();this.accumulation.forEach(t=>t.dispose());this.accumulateMaterial.dispose();
    for(const scene of [this.rayScene,this.blitScene,this.accumulateScene])scene.traverse(child=>{if(child instanceof THREE.Mesh)child.geometry.dispose();});
    this.material.dispose();this.blitMaterial.dispose();
  }
}
