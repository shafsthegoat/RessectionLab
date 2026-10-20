import {TransformControls} from "three/addons/controls/TransformControls.js";
import {RoomEnvironment} from 'three/addons/environments/RoomEnvironment.js';
import {cachedSurface,retainSurface} from './surfaceCache';
import type {SurfaceDiagnostics} from './surfacePolicy';
import {surgicalToolMesh,surgicalEffectGroup,surgicalToolGeometryKey,updateSurgicalToolMesh} from "./surgicalGeometry";
import type {SurgicalInteraction} from "../surgical-types";
import {InteractionSession, InteractionBudget} from './interactionBudget';
import {SliceFrameCache} from './sliceFrameCache';
import {GpuFrameTimer} from './gpuFrameTimer';
import {geometryBufferBytes,renderTargetBytes} from './renderResourceEstimate';
import {SurfaceInteractionRenderer} from './surfaceInteractionRenderer';
import {LabelTextureUpdates} from './labelTextureUpdates';
import {modeledAnatomy} from './modeledAnatomy';
import {SourcePickingBvh} from './sourcePickingBvh';
import type {PreparedSurface} from './prepareSurface';
import type {IncrementalSurfaceRequest} from './incrementalSurface';
import {scanFitPoints} from './scanCameraFit';
import {mprView,sourceSignalBounds} from './mprFraming';
import {applySourceContextMaterial} from './sourceContextMaterial';
import {displaySurfaceNormals} from './displaySurfaceNormals';
import {applyVesselDepthMaterial} from './anatomySurfaceMaterial';
import {surfaceKind} from './surfacePolicy';
import {packAnatomyLabels,materializeLayerMask,MAX_ANATOMY_LAYERS} from './labelPacking';
import {checkedPublicGoal,PUBLIC_GOAL_COLOR} from './publicGoal';
import type {ViewerPublicGoal} from './publicGoal';
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { clippingForBounds, layerAppearance, validCameraState } from './workspaceSettings';
import type { ViewerCameraState, ViewerWorkspaceSettings } from './workspaceSettings';
import { checkedFusionTransform } from './registeredFusion';
import type { ViewerRegisteredFusion } from './registeredFusion';
import {
  PLANE_AXES,
  inverseAffine,
  rasAffine,
  transformPoint,
  sampleTrilinear,
  volumeBounds,
} from "./coordinates";
import type { Affine, Bounds3, Point3, SlicePlane } from "./coordinates";
import type {
  ViewerReplay,
  ViewerPassiveReplay,
  ViewerRehearsalState,
  ViewerPriorLayer,
  ViewerRoute,
  ViewerStructuralProposal,
  ViewerVolume,
  ViewerInspectionTool,
} from "./contracts";
import { InstrumentDisplayState } from "./inspectionTool";
import type { InspectionToolDisplay, InstrumentCapsuleDisplay } from "./inspectionTool";
import { checkedPassiveRecordedTools, checkedRecordedEpisodeBounds, recordedToolDisplay } from "./recordedTool";
import type { RecordedEpisodeBounds, RecordedToolDisplay } from "./recordedTool";
import { instrumentCapsuleMeshes, inspectionToolMeshes } from "./inspectionToolGeometry";
import { fragmentShader, vertexShader } from "./shaders";
import { validateDiagnosticLayer } from "./diagnosticLayer";
import type { CheckedDiagnosticLayer, LoadedDiagnosticLayer } from "./diagnosticLayer";
import { physicalBounds, placeInSourceFrame } from "./sceneGeometry";
import { surfaceCameraDistance } from './surfaceCameraFit';
import { ScanVolumeRenderer } from './ScanVolumeRenderer';
import type { VolumeRenderingSettings } from './volumeTransfer';
import { paneViewport } from "./layout";
import {
  PRIOR_COLORS,
  priorSamplingTolerance,
  validatePriorLayer,
} from "./priorLayer";
import { residualMask, validateReplay } from "./replay";
import {checkedRehearsalState, REHEARSAL_COLORS} from "./rehearsal";
import {
  STRUCTURAL_PROPOSAL_COLOR,
  validateStructuralProposal,
} from "./structuralProposal";
import {
  COMPARISON_COLORS,
  FAILURE_COLOR,
  routeAppearance,
} from "./routeAppearance";

type Panes = { anatomy: HTMLElement } & Record<SlicePlane, HTMLElement>;
type SurfacePair={mesh:THREE.Mesh;source:THREE.BufferGeometry;smooth:THREE.BufferGeometry};
type ModelStage={generation:number;next:ViewerVolume;interaction:SurgicalInteraction|null;
  prepared:Map<string,SurfacePair|null>;jobs:Array<{layer:ViewerVolume['compartments'][number];indices?:Uint32Array;values?:Uint8Array}>;
  pack:boolean;packed?:{data:Uint8Array;codes:Array<[number,number]>};patch?:{indices:Uint32Array;values:Uint8Array};
  waiting?:ViewerVolume['compartments'][number]|'pack';ready?:boolean;changedLayers:number;changedIndices:number;initialMapping:boolean};
const SLICE_PLANES: SlicePlane[] = ["axial", "coronal", "sagittal"];
const RETIRED_SURFACE_BUDGET_BYTES=256*1024*1024;

function matrix(affine: Affine): THREE.Matrix4 {
  return new THREE.Matrix4().set(
    ...(affine.flat() as [
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
    ]),
  );
}
function disposeObject(object: THREE.Object3D): void {
  object.traverse((child) => {
    if (child instanceof THREE.Mesh || child instanceof THREE.Line || child instanceof THREE.Points) {
      // Derived replay meshes retain both source and smoothed display buffers.
      // Dispose the inactive geometry too when a frame is replaced.
      const replayGeometry = child.userData.replayGeometry as {source:THREE.BufferGeometry;smooth:THREE.BufferGeometry}|undefined;
      for (const geometry of new Set([child.geometry, replayGeometry?.source, replayGeometry?.smooth]))
        geometry?.dispose();
      const materials = Array.isArray(child.material)
        ? child.material
        : [child.material];
      materials.forEach((material) => material.dispose());
    }
  });
}
function dataTexture(
  data: Float32Array | Uint8Array,
  shape: [number, number, number],
): THREE.Data3DTexture {
  const texture = new THREE.Data3DTexture(data, shape[2], shape[1], shape[0]);
  texture.format = THREE.RedFormat;
  texture.type =
    data instanceof Float32Array ? THREE.FloatType : THREE.UnsignedByteType;
  texture.minFilter = THREE.NearestFilter;
  texture.magFilter = THREE.NearestFilter;
  texture.unpackAlignment = 1;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return texture;
}
function intensityWindow(values: Float32Array): [number, number] {
  const sampled: number[] = [],
    stride = Math.max(1, Math.floor(values.length / 16000));
  for (let i = 0; i < values.length; i += stride)
    if (Number.isFinite(values[i]) && values[i] !== 0) sampled.push(values[i]);
  sampled.sort((a, b) => a - b);
  if (!sampled.length) return [0, 1];
  const low = sampled[Math.floor(sampled.length * 0.005)],
    high =
      sampled[Math.min(sampled.length - 1, Math.floor(sampled.length * 0.995))];
  return [low, Math.max(low + 1e-6, high)];
}

/** One GPU context shares the source textures across the 3-D scene and all MPRs. */
export class VolumeRenderer {
  readonly bounds: Bounds3;
  private readonly sliceSupportBounds: Bounds3;
  private sliceFramingSignature = '';
  readonly defaultWindow: [number, number];
  readonly affine: Affine;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly environmentTarget:THREE.WebGLRenderTarget;
  private readonly scanVolume: ScanVolumeRenderer;
  private readonly surfaceInteraction=new SurfaceInteractionRenderer();
  private readonly labelTextureUpdates=new LabelTextureUpdates();
  private readonly picking=new SourcePickingBvh({maxInputBytes:128*1024*1024,onEvent:detail=>{
    this.canvas?.dispatchEvent(new CustomEvent('anatomy-work',{bubbles:true,detail}));
    if(detail.kind==='source-bvh-ready'||detail.kind==='source-bvh-error')this.requestRender(false);
  }});
  private volumeModality = 'MRI';
  private interacting = false;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(34, 1, 0.1, 5000);
  private readonly controls: OrbitControls;
  private readonly slices = new Map<
    SlicePlane,
    {
      scene: THREE.Scene;
      material: THREE.ShaderMaterial;
      camera: THREE.OrthographicCamera;
    }
  >();
  private readonly anatomy = new THREE.Group();
  private readonly tools = new THREE.Group();
  private readonly inspectionTools = new THREE.Group();
  private readonly recordedTools = new THREE.Group();
  private recordedDisplay: RecordedToolDisplay | null = null;
  private passiveDisplays: RecordedToolDisplay[] = [];
  private passiveActive = false;
  private rehearsalActive = false;
  private rehearsalDisplays: RecordedToolDisplay[] = [];
  private replayBounds: RecordedEpisodeBounds | null = null;
  private fittedReplayId: string | null = null;
  private readonly instrumentDisplay = new InstrumentDisplayState();
  private readonly replayGroup = new THREE.Group();
  private replayWorker: Worker | null = null;
  private replayGeneration = 0;
  private pendingReplayGroup: THREE.Group | null = null;
  private replayActive = false;
  private removedTexture: THREE.Data3DTexture;
  private proposalTexture: THREE.Data3DTexture;
  private diagnosticStateTexture: THREE.Data3DTexture;
  private diagnosticCoverageTexture: THREE.Data3DTexture;
  private priorTexture: THREE.Data3DTexture;
  private priorCoverageTexture: THREE.Data3DTexture;
  private readonly surfaces = new Map<string, THREE.Mesh>();
  private readonly surfaceGeometry = new Map<string, { source: THREE.BufferGeometry; smooth: THREE.BufferGeometry }>();
  private surfaceSmoothing = true;
  private scanFramingPoints:Float32Array=new Float32Array();
  private readonly mriTexture: THREE.Data3DTexture;
  private fusionTexture: THREE.Data3DTexture;
  private labelTexture: THREE.Data3DTexture;
  private readonly sourcePlane: THREE.Mesh<
    THREE.BufferGeometry,
    THREE.ShaderMaterial
  >;
  private readonly publicGoalObject = new THREE.Group();
  private readonly cursorObject: THREE.LineSegments;
  private readonly observer: ResizeObserver;
  private readonly worker: Worker;
  private sourceSurfaceGeneration=0;
  private sourceRemaining=0;
  private sourceFailed=false;
  private sourceWorkerActive=true;
  private readonly preparedSourceLayers=new Set<string>();
  private labelCodes:Array<[number,number]>=[];
  private modelWorker:Worker|null=null;
  private modelGeneration=0;
  private modelIdentityCounter=0;
  private readonly modelIdentities=new WeakMap<object,string>();
  private modelWorkerStates=new Map<string,string>();
  private modelStage:ModelStage|null=null;
  private failedModelState:{next:ViewerVolume;target:SurgicalInteraction['view']}|null=null;
  private rendering=false;
  private publishedSurgicalView:SurgicalInteraction['view']=null;
  private publishedSurgicalInteraction:SurgicalInteraction|null=null;
  private pendingSurgicalPublication:SurgicalInteraction|null=null;
  private initialSurgicalInteraction:SurgicalInteraction|null=null;
  private readonly retiredSurfaces:Array<{layer:Pick<ViewerVolume['compartments'][number],'name'|'kind'|'mask'>;pair:SurfacePair;bytes:number}>=[];
  private disposed = false;
  private frame = 0;
  private width = 0;
  private height = 0;
  private cursor: Point3;
  private visible: Record<string, boolean> = {};
  private opacity = 0.4;
  private workspaceSettings: ViewerWorkspaceSettings = {};
  private appearanceSignature: string | null = null;
  private visibilitySignature: string | null = null;
  private extraClipPlanes: THREE.Plane[] = [];
  private clipPlane: THREE.Plane | null = null;
  private cameraRestored = false;
  private activePlane: SlicePlane = "axial";
  private routeSignature = "";
  private mode: "anatomy" | "instruments" = "anatomy";
  private pendingFit: "anatomy" | "instruments" | null = null;
  private readonly pickRay = new THREE.Raycaster();
  private sliceRevision=0;
  private readonly sliceCaches=new Map<SlicePlane,SliceFrameCache>();
  private paneBounds: {root:DOMRect;panes:Map<HTMLElement,DOMRect>} | null=null;
  private gpuSync:WebGLSync|null=null;
  private gpuPoll=0;
  private needsFrame=false;
  private renderSerial=0;
  private gpuTimer:GpuFrameTimer|null=null;
  private graphicsLost=false;
  private graphicsRecovery:WEBGL_lose_context|null=null;
  private readonly budget=new InteractionBudget();
  private readonly interaction=new InteractionSession(active=>{
    this.interacting=active;
    if(active)this.refinementStarted=0;
    this.requestRender(false);
  });
  private readonly onLayout=()=>{this.paneBounds=null;this.requestRender();};
  private readonly onScroll=(event:Event)=>{
    if(event.target===document||event.target===window||(event.target instanceof HTMLElement&&event.target.contains(this.container)))this.onLayout();
  };
  private readonly onBlur=()=>this.interaction.finish();
  private readonly onControlsChange = () => this.requestRender(false);
  private refinementStarted=0;
  private readonly onControlsStart = () => { this.interaction.start(); };
  private readonly onControlsEnd = () => {
    this.interaction.end();
    this.refinementStarted=performance.now();
    this.requestRender(false);
    this.cameraRestored = true;
    this.onCameraChange?.(this.getCameraState());
  };
  private readonly onContextLost = (event: Event) => {
    event.preventDefault();
    this.graphicsLost=true;cancelAnimationFrame(this.frame);this.frame=0;cancelAnimationFrame(this.gpuPoll);this.gpuPoll=0;
    this.gpuSync=null;this.needsFrame=false;
    this.gpuTimer?.dispose('context-lost');
    this.onGraphicsStatus?.('lost');
    this.onError(
      "The graphics context was lost. Use Retry graphics to restore the current source-image views.",
    );
  };
  private readonly onContextRestored=()=>{if(!this.disposed)this.onGraphicsStatus?.('ready');};

  constructor(
    private readonly container: HTMLElement,
    private readonly canvas: HTMLCanvasElement,
    private readonly panes: Panes,
    private volume: ViewerVolume,
    private readonly onError: (message: string) => void,
    private readonly onSurfaceStatus: (remaining: number) => void,
    private readonly onReplayError: (message: string) => void,
    private readonly onCameraChange?: (camera: ViewerCameraState) => void,
    private readonly onGraphicsStatus?: (status:'lost'|'ready') => void,
  ) {
    const count = volume.shape.reduce((a, b) => a * b, 1);
    if (
      volume.shape.some((x) => !Number.isInteger(x) || x < 2) ||
      volume.mri.length !== count
    )
      throw new Error("MRI array dimensions do not match the source grid.");
    const packedLabels=volume.packedLabels??packAnatomyLabels(volume);
    this.labelCodes=packedLabels.codes;
    this.affine = rasAffine(volume.affine, volume.frame);
    const inverse = inverseAffine(this.affine);
    this.bounds = volumeBounds(volume.shape, this.affine);
    this.sliceSupportBounds = sourceSignalBounds(volume.mri,volume.shape,this.affine);
    this.cursor = this.bounds[0].map(
      (x, i) => (x + this.bounds[1][i]) / 2,
    ) as Point3;
    this.defaultWindow = intensityWindow(volume.mri);
    this.scanFramingPoints=scanFitPoints(volume,this.defaultWindow);
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });
    const gl = this.renderer.getContext() as WebGL2RenderingContext;
    this.graphicsRecovery=gl.getExtension('WEBGL_lose_context');
    this.renderer.info.autoReset=false;
    this.gpuTimer=new GpuFrameTimer(gl,sample=>this.canvas.dispatchEvent(new CustomEvent('anatomy-gpu-time',{bubbles:true,detail:sample})));
    const maximum = gl.getParameter(gl.MAX_3D_TEXTURE_SIZE) as number;
    if (volume.shape.some((x) => x > maximum)) {
      this.renderer.dispose();
      throw new Error(
        `The source grid exceeds this GPU's ${maximum}-voxel 3-D texture limit.`,
      );
    }
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(0x000000, 1);
    this.renderer.autoClear = false;
    this.renderer.localClippingEnabled = true;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = .95;
    const lighting=new RoomEnvironment(),pmrem=new THREE.PMREMGenerator(this.renderer);
    this.environmentTarget=pmrem.fromScene(lighting,.04);
    this.scene.environment=this.environmentTarget.texture;this.scene.environmentIntensity=.45;
    lighting.dispose();pmrem.dispose();
    this.renderer.debug.onShaderError = () =>
      onError(
        "The GPU could not compile the source-image renderer. No image or route geometry has been substituted.",
      );
    this.canvas.addEventListener("webglcontextlost", this.onContextLost);
    this.canvas.addEventListener('webglcontextrestored',this.onContextRestored);
    this.mriTexture = dataTexture(volume.mri, volume.shape);
    this.scanVolume = new ScanVolumeRenderer(this.renderer, volume, this.defaultWindow,packedLabels.codes,this.mriTexture);
    this.scanVolume.configure(undefined, this.volumeModality, null);
    this.fusionTexture = dataTexture(new Float32Array(1),[1,1,1]);
    volume.compartments.forEach(layer=>{this.visible[layer.name]=true;});
    this.labelTexture = dataTexture(packedLabels.data,volume.shape);
    this.labelTexture.format=THREE.RGBAFormat;
    this.scanVolume.labels(this.labelTexture,this.visible,this.workspaceSettings,[]);
    this.removedTexture = dataTexture(new Uint8Array(1), [1, 1, 1]);
    this.proposalTexture = dataTexture(new Uint8Array(1), [1, 1, 1]);
    this.diagnosticStateTexture = dataTexture(new Uint8Array(1), [1, 1, 1]);
    this.diagnosticCoverageTexture = dataTexture(new Uint8Array(1), [1, 1, 1]);
    this.priorTexture = dataTexture(new Float32Array(1), [1, 1, 1]);
    this.priorCoverageTexture = dataTexture(new Uint8Array(1), [1, 1, 1]);
    const material = (threeD: boolean, plane: SlicePlane) =>
      new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader,
        glslVersion: THREE.GLSL3,
        side: THREE.DoubleSide,
        transparent: threeD,
        depthWrite: !threeD,
        toneMapped: false,
        uniforms: {
          uMri: { value: this.mriTexture },
          uFusion: { value: this.fusionTexture },
          uFusionActive: { value: 0 },
          uFusionOpacity: { value: .5 },
          uFusionShape: { value: new THREE.Vector3(1,1,1) },
          uWorldToFusion: { value: new THREE.Matrix4() },
          uFusionWindow: { value: new THREE.Vector2(0,1) },
          uLabels: { value: this.labelTexture },
          uRemoved: { value: this.removedTexture },
          uReplayActive: { value: 0 },
          uProposal: { value: this.proposalTexture },
          uProposalActive: { value: 0 },
          uDiagnosticState: { value: this.diagnosticStateTexture },
          uDiagnosticCoverage: { value: this.diagnosticCoverageTexture },
          uDiagnosticActive: { value: 0 },
          uProposalColor: {
            value: new THREE.Color(
              STRUCTURAL_PROPOSAL_COLOR,
            ).convertLinearToSRGB(),
          },
          uPrior: { value: this.priorTexture },
          uPriorCoverage: { value: this.priorCoverageTexture },
          uPriorActive: { value: 0 },
          uPriorKind: { value: 0 },
          uWorldToPrior: { value: new THREE.Matrix4() },
          uPriorShape: { value: new THREE.Vector3(1, 1, 1) },
          uPriorVoxelTolerance: { value: new THREE.Vector3() },
          uPriorColors: {
            value: PRIOR_COLORS.map((color) =>
              new THREE.Color(color).convertLinearToSRGB(),
            ),
          },
          uWorldToVoxel: { value: matrix(inverse) },
          uShape: { value: new THREE.Vector3(...volume.shape) },
          uLow: { value: new THREE.Vector3(...this.bounds[0]) },
          uHigh: { value: new THREE.Vector3(...this.bounds[1]) },
          uCursor: { value: new THREE.Vector3(...this.cursor) },
          uAxes: { value: new THREE.Vector3(...PLANE_AXES[plane]) },
          uRect: { value: new THREE.Vector4(0, 0, 1, 1) },
          uResolution: { value: new THREE.Vector2(1, 1) },
          uWindow: { value: new THREE.Vector2(...this.defaultWindow) },
          uOverlay: { value: this.opacity },
          uSliceContours: { value: 0 },
          uLayerOpacities: { value: new Float32Array(MAX_ANATOMY_LAYERS).fill(.66) },
          uClipEnabled: { value: 0 },
          uClipPlane: { value: new THREE.Vector4(0,0,1,0) },
          uThreeD: { value: threeD ? 1 : 0 },
          uLayerCount: {value:volume.compartments.length},
          uLayerCodes: {value:packedLabels.codes.map(c=>new THREE.Vector2(...c))},
          uLayerEnabled: {value:new Float32Array(MAX_ANATOMY_LAYERS).fill(1)},
          uVisibleBits: { value: (1 << volume.compartments.length) - 1 },
          uColors: {
            value: Array.from({ length: MAX_ANATOMY_LAYERS }, (_, i) =>
              new THREE.Color(
                volume.compartments[i]?.color || "#738c9e",
              ).convertLinearToSRGB(),
            ),
          },
          uRouteCount: { value: 0 },
          uShaftStart: { value: [new THREE.Vector3(), new THREE.Vector3()] },
          uShaftEnd: { value: [new THREE.Vector3(), new THREE.Vector3()] },
          uTipEnd: { value: [new THREE.Vector3(), new THREE.Vector3()] },
          uRadii: { value: [new THREE.Vector2(), new THREE.Vector2()] },
          uRouteColors: {
            value: Object.values(COMPARISON_COLORS).map((color) =>
              new THREE.Color(color).convertLinearToSRGB(),
            ),
          },
        },
      });
    for (const plane of SLICE_PLANES) {
      const scene = new THREE.Scene(),
        shader = material(false, plane),
        camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 3);
      camera.position.z = 1;
      scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), shader));
      this.slices.set(plane, { scene, material: shader, camera });
    }
    const planeGeometry = new THREE.BufferGeometry();
    planeGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(new Float32Array(12), 3),
    );
    planeGeometry.setAttribute(
      "uv",
      new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2),
    );
    planeGeometry.setIndex([0, 1, 2, 0, 2, 3]);
    this.sourcePlane = new THREE.Mesh(planeGeometry, material(true, "axial"));
    this.sourcePlane.renderOrder = 1;
    this.scene.add(
      this.sourcePlane,
      this.anatomy,
      this.tools,
      this.inspectionTools,
      this.recordedTools,
      this.replayGroup,
    );
    this.scene.add(new THREE.HemisphereLight(0xb7d6df, 0x142738, .65));
    const centre = new THREE.Vector3(...this.cursor);
    const key = new THREE.DirectionalLight(0xe8f3f3, 2.0);
    key.position.copy(centre).add(new THREE.Vector3(170, -220, 240));
    key.target.position.copy(centre);
    const rim = new THREE.DirectionalLight(0x77bdb4, .75);
    rim.position.copy(centre).add(new THREE.Vector3(-190, 160, 70));
    rim.target.position.copy(centre);
    this.scene.add(key, key.target, rim, rim.target);
    const cursorGeometry = new THREE.BufferGeometry().setAttribute(
      "position",
      new THREE.Float32BufferAttribute(
        [-3, 0, 0, 3, 0, 0, 0, -3, 0, 0, 3, 0, 0, 0, -3, 0, 0, 3],
        3,
      ),
    );
    this.cursorObject = new THREE.LineSegments(
      cursorGeometry,
      new THREE.LineBasicMaterial({
        color: 0xb4eadb,
        transparent: true,
        opacity: 0.95,
        depthTest: false,
      }),
    );
    this.cursorObject.renderOrder = 5;
    this.cursorObject.position.copy(centre);
    this.scene.add(this.cursorObject);
    this.scene.add(this.publicGoalObject);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, panes.anatomy);
    // Direct controls settle at pointer/wheel end, so saved camera equals the
    // displayed camera even if Save is pressed immediately after a gesture.
    this.controls.enableDamping = false;
    this.controls.rotateSpeed = 0.65;
    this.controls.zoomSpeed = 0.8;
    this.controls.minDistance = 5;
    this.controls.maxDistance = 2200;
    this.controls.addEventListener("change", this.onControlsChange);
    this.controls.addEventListener("start", this.onControlsStart);
    this.controls.addEventListener("end", this.onControlsEnd);
    this.observer = new ResizeObserver(this.onLayout);
    window.addEventListener('scroll',this.onScroll,true);
    window.addEventListener('blur',this.onBlur);
    panes.anatomy.addEventListener('pointercancel',this.onControlsEnd);
    this.observer.observe(container);
    Object.values(panes).forEach((pane) => this.observer.observe(pane));
    this.worker = new Worker(new URL("./surface.worker.ts", import.meta.url), {
      type: "module",
    });
    const sourceRequests=new Map<string,number>();
    this.displayWork.sourceWorkerBuilds++;
    const sourceGeneration=this.sourceSurfaceGeneration;
    let remaining = volume.compartments.length;
    this.sourceRemaining=remaining;
    onSurfaceStatus(remaining);
    this.worker.onmessage = (
      event: MessageEvent<{
        name: string;
        positions?: Float32Array;
        normals?: Float32Array;
        smoothPositions?: Float32Array;
        smoothNormals?: Float32Array;
        indices?: Uint32Array;
        diagnostics?: SurfaceDiagnostics;
        error?: string;
      }>,
    ) => {
      if (this.disposed||sourceGeneration!==this.sourceSurfaceGeneration) return;
      const requestStarted=sourceRequests.get(event.data.name);
      if(requestStarted!==undefined){
        sourceRequests.delete(event.data.name);
        const buffers=[event.data.positions,event.data.normals,event.data.smoothPositions,event.data.smoothNormals,event.data.indices];
        const bytes=buffers.reduce((sum,array)=>sum+(array?.byteLength??0),0);
        this.displayWork.workerResultTransferBytes+=bytes;
        this.canvas.dispatchEvent(new CustomEvent('anatomy-work',{bubbles:true,detail:{kind:'surface-preparation',scope:'source',name:event.data.name,
          requestToResultMs:performance.now()-requestStarted,preparationMs:event.data.diagnostics?.preparationMs??null,resultTransferBytes:bytes,error:event.data.error??null}}));
      }
      remaining--;
      this.sourceRemaining=remaining;
      if (!this.replayActive) onSurfaceStatus(remaining);
      if (event.data.error) {
        this.sourceFailed=true;
        this.initialSurgicalInteraction?.onDisplayError?.(this.initialSurgicalInteraction.preview??this.initialSurgicalInteraction.view,event.data.error);
        onError(`Source surface unavailable: ${event.data.error}`);
        return;
      }
      const layer = volume.compartments.find(
          (item) => item.name === event.data.name,
        ),
        positions = event.data.positions,
        normals = event.data.normals;
      if(!layer||!positions)return;
      this.preparedSourceLayers.add(layer.name);
      if (!normals || normals.length !== positions.length) {
        onError("Source surface normals do not match its unchanged geometry.");
        return;
      }

      if(event.data.indices)retainSurface(layer.mask,this.affine,layer.kind,{...event.data,positions,normals:normals!,indices:event.data.indices},this.volume.shape);
      this.canvas.dispatchEvent(new CustomEvent('anatomy-surface',{bubbles:true,detail:{name:layer.name,diagnostics:event.data.diagnostics??layer.surfaceDiagnostics,qa:layer.qa,empty:!positions.length}}));
      if(remaining===0){this.worker.terminate();this.sourceWorkerActive=false;}
      if(!positions.length){
        if(remaining===0&&this.mode==='anatomy'&&!this.cameraRestored)this.fitCamera('anatomy');
        this.finishInitialSurgical();this.requestRender();return;
      }
      this.installSurface(layer,{positions,normals,smoothPositions:event.data.smoothPositions,smoothNormals:event.data.smoothNormals,indices:event.data.indices});
      this.finishInitialSurgical();
      if (remaining === 0 && this.mode === "anatomy" && !this.cameraRestored) this.fitCamera("anatomy");
      this.requestRender();
    };
    this.worker.onerror = () => {
      this.sourceFailed=true;
      this.initialSurgicalInteraction?.onDisplayError?.(this.initialSurgicalInteraction.preview??this.initialSurgicalInteraction.view,'source surface worker failed');
      onError(
        "A source surface could not be prepared. The original MRI and labels remain available in the linked slices.",
      );
    };
    volume.compartments.forEach((layer) => {
      const cached=!layer.labelSet?cachedSurface(layer.mask,this.affine,layer.kind,this.volume.shape):undefined;
      if(cached){this.worker.onmessage?.(new MessageEvent('message',{data:{name:layer.name,...cached}}));}
      else if (layer.surface) {
        const s = layer.surface;
        this.worker.onmessage?.(new MessageEvent('message', {data: {name: layer.name,
          positions: s.source, normals: s.sourceNormals, smoothPositions: s.positions,
          smoothNormals: s.normals, indices: s.indices,diagnostics:layer.surfaceDiagnostics}}));
      } else {
        const mask=materializeLayerMask(this.volume,layer);sourceRequests.set(layer.name,performance.now());
        this.displayWork.sourceRemeshedLayers++;this.displayWork.workerInputCloneBytes+=mask.byteLength;
        this.worker.postMessage({name:layer.name,kind:layer.kind,mask,shape:volume.shape,affine:this.affine,displaySmoothing:true});
      }
    });
    if(!remaining){this.worker.terminate();this.sourceWorkerActive=false;}
    for(const tract of volume.tractography??[]){
      const indices:number[]=[];
      for(let line=0;line<tract.offsets.length-1;line++)for(let i=tract.offsets[line];i<tract.offsets[line+1]-1;i++)indices.push(i,i+1);
      const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(tract.points,3));geometry.setIndex(indices);
      const lines=new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({color:'#91cffa',transparent:true,opacity:.45}));placeInSourceFrame(lines,this.affine);this.anatomy.add(lines);
    }
    this.updateSourcePlane();
    this.fitCamera("anatomy");
    this.requestRender();
  }

  private makeSurface(layer:ViewerVolume['compartments'][number],buffers:{positions:Float32Array;normals:Float32Array;smoothPositions?:Float32Array;smoothNormals?:Float32Array;indices?:Uint32Array}):SurfacePair|null {
    const {positions,normals}=buffers;
    const onError=this.onError;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(positions, 3),
    );
    geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
    const { smoothPositions, smoothNormals } = buffers;
    if (!smoothPositions || !smoothNormals || smoothPositions.length !== positions.length || smoothNormals.length !== positions.length) {
      geometry.dispose();
      onError("Display surface buffers do not match the source triangles.");
      return null;
    }
    const smooth = new THREE.BufferGeometry();
    if (buffers.indices) {
      geometry.setIndex(new THREE.BufferAttribute(buffers.indices, 1));
      smooth.setIndex(new THREE.BufferAttribute(buffers.indices, 1));
    }
    smooth.setAttribute("position", new THREE.BufferAttribute(smoothPositions, 3));
    const vessel=surfaceKind(layer.kind)==='vessel';
    const fairNormals=vessel||['brain','bone','skull','head-envelope'].includes(layer.kind??'');
    smooth.setAttribute("normal", new THREE.BufferAttribute(fairNormals&&buffers.indices ? displaySurfaceNormals(smoothNormals,buffers.indices,this.affine) : smoothNormals, 3));
    const appearance = layerAppearance(layer.name,layer.color,this.workspaceSettings,layer.kind);
    const mesh = new THREE.Mesh(
      this.surfaceSmoothing ? smooth : geometry,
      new THREE.MeshStandardMaterial({
        color: appearance.color,
        roughness: vessel ? .32 : layer.kind==='tumor' ? .48 : .52,
        metalness: 0,
        transparent: appearance.opacity < 1,
        opacity: appearance.opacity,
        clippingPlanes: [...this.extraClipPlanes, ...(this.clipPlane ? [this.clipPlane] : [])],
        side: THREE.DoubleSide,
        depthWrite: appearance.opacity >= 1,
      }),
    );
    placeInSourceFrame(mesh, this.affine);
    if(vessel)applyVesselDepthMaterial(mesh.material,this.bounds);
    applySourceContextMaterial(mesh.material,this.workspaceSettings.contextStyle==='shell'&&['brain','head-envelope','bone','skull'].includes(layer.kind??''));
    mesh.visible = this.visible[layer.name] !== false;
    mesh.userData.compartment = layer.name;
    this.picking?.install(mesh,geometry);
    return {mesh,source:geometry,smooth};
  }
  private installSurface(layer:ViewerVolume['compartments'][number],buffers:PreparedSurface|{positions:Float32Array;normals:Float32Array;smoothPositions?:Float32Array;smoothNormals?:Float32Array;indices?:Uint32Array}) {
    const pair=this.makeSurface(layer,buffers);if(!pair)return;
    this.surfaceGeometry.set(layer.name,{source:pair.source,smooth:pair.smooth});this.surfaces.set(layer.name,pair.mesh);this.anatomy.add(pair.mesh);
  }

  /** Append only, within an unchanged native source generation. */
  setAnatomyLayers(next:ViewerVolume) {
    if(next===this.volume)return;
    if(modeledAnatomy(this.volume))return; // source restoration is staged atomically below
    if(!next.displayIdentity||next.displayIdentity!==this.volume.displayIdentity||next.mri!==this.volume.mri||next.caseHash!==this.volume.caseHash||
      JSON.stringify([next.shape,next.affine,next.frame])!==JSON.stringify([this.volume.shape,this.volume.affine,this.volume.frame]))throw Error('Displayed source changed during layer installation');
    const packed=next.packedLabels;
    if(!packed||packed.data.length!==next.mri.length*4||packed.codes.length!==MAX_ANATOMY_LAYERS)throw Error('Missing prepared anatomy labels');
    for(const old of this.volume.compartments)if(next.compartments.find(layer=>layer.name===old.name)!==old)throw Error('Resident anatomy replacement requires a new source generation');
    const added=next.compartments.filter(layer=>!this.volume.compartments.some(old=>old.name===layer.name));
    for(const layer of added){const s=layer.surface;if(!s||s.source.length!==s.positions.length||s.normals.length!==s.positions.length||s.sourceNormals.length!==s.positions.length)throw Error('Incremental anatomy requires valid prepared surfaces');}
    this.beginInteraction();this.endInteraction();
    const texture=dataTexture(packed.data,next.shape);texture.format=THREE.RGBAFormat;
    const previous=this.labelTexture;
    this.labelTexture=texture;this.volume=next;this.labelCodes=packed.codes;
    for(const layer of added){const s=layer.surface!;if(s.source.length)this.installSurface(layer,{positions:s.source,normals:s.sourceNormals,smoothPositions:s.positions,smoothNormals:s.normals,indices:s.indices});}
    this.materials().forEach(shader=>{
      shader.uniforms.uLabels.value=texture;shader.uniforms.uLayerCount.value=next.compartments.length;
      packed.codes.forEach((code,i)=>shader.uniforms.uLayerCodes.value[i].set(...code));
      next.compartments.forEach((layer,i)=>shader.uniforms.uLayerEnabled.value[i]=this.visible[layer.name]!==false?1:0);
    });
    this.scanVolume.setLayers(next.compartments,packed.codes);
    this.appearanceSignature=null;this.visibilitySignature=null;
    this.setWorkspaceSettings(this.workspaceSettings);
    previous.dispose();this.requestRender();
  }

  private finishInitialSurgical(){
    if(this.sourceRemaining||this.sourceFailed||!this.initialSurgicalInteraction)return;
    const interaction=this.initialSurgicalInteraction;this.initialSurgicalInteraction=null;
    this.setSurgical(interaction);this.queueSurgicalPublication(interaction);
  }
  private queueSurgicalPublication(interaction:SurgicalInteraction|null){
    if(!interaction){this.pendingSurgicalPublication=null;this.publishedSurgicalView=null;this.publishedSurgicalInteraction=null;return;}
    const target=interaction.preview??interaction.view;
    if(target===this.publishedSurgicalView){this.publishedSurgicalInteraction=interaction;interaction.onDisplayReady?.(target);return;}
    this.pendingSurgicalPublication=interaction;this.requestRender();
  }
  private finishSurgicalFrame(){
    const interaction=this.pendingSurgicalPublication;if(!interaction)return;
    const target=interaction.preview??interaction.view,installed=this.surgicalInteraction?.preview??this.surgicalInteraction?.view??null;
    if(target!==installed)return;this.pendingSurgicalPublication=null;this.publishedSurgicalView=target;this.publishedSurgicalInteraction=interaction;interaction.onDisplayReady?.(target);
  }
  private modelIdentity(value:object){let key=this.modelIdentities.get(value);if(!key){key=`tissue-${++this.modelIdentityCounter}`;this.modelIdentities.set(value,key);}return key;}
  private disposeSurfacePair(pair:SurfacePair){
    disposeObject(pair.mesh);if(pair.source!==pair.mesh.geometry)pair.source.dispose();if(pair.smooth!==pair.mesh.geometry&&pair.smooth!==pair.source)pair.smooth.dispose();
  }
  private retireSurface(layer:ViewerVolume['compartments'][number],retain=true){
    const mesh=this.surfaces.get(layer.name),geometry=this.surfaceGeometry.get(layer.name);
    this.surfaces.delete(layer.name);this.surfaceGeometry.delete(layer.name);if(!mesh||!geometry)return;
    this.anatomy.remove(mesh);const pair={mesh,...geometry},bytes=geometryBufferBytes([geometry.source,geometry.smooth])+layer.mask.byteLength;
    if(!retain){this.disposeSurfacePair(pair);return;}
    this.retiredSurfaces.push({layer:{name:layer.name,kind:layer.kind,mask:layer.mask},pair,bytes});let total=this.retiredSurfaces.reduce((sum,e)=>sum+e.bytes,0);
    while(total>RETIRED_SURFACE_BUDGET_BYTES||this.retiredSurfaces.length>16){const old=this.retiredSurfaces.shift()!;total-=old.bytes;this.disposeSurfacePair(old.pair);}
  }
  private clearModelStage(){
    if(this.modelStage)for(const pair of this.modelStage.prepared.values())if(pair)this.disposeSurfacePair(pair);
    this.modelStage=null;
  }
  /** Admission is private/source-owned, never a user-provided timeline index.
   * Keep the prior complete state until labels, changed surfaces and tools can
   * commit in one task. The source MRI/context and unaffected meshes survive.
   * True admits a request (which may be pending); false leaves it failed. */
  installSurgicalState(next:ViewerVolume,interaction:SurgicalInteraction|null):boolean {
    const target=interaction?.preview??interaction?.view??null;
    // React can recreate callback wrappers after reporting failure. Keep that
    // exact failed request stopped; an explicit new view/seek may retry it.
    if(this.failedModelState?.next===next&&this.failedModelState.target===target)return false;
    this.failedModelState=null;
    try{
      const a=modeledAnatomy(this.volume),b=modeledAnatomy(next),owner=a?.source??this.volume,nextOwner=b?.source??next;
      if(owner!==nextOwner||next.mri!==this.volume.mri||next.caseHash!==this.volume.caseHash||
        JSON.stringify([next.shape,next.affine,next.frame])!==JSON.stringify([this.volume.shape,this.volume.affine,this.volume.frame]))throw Error('Modeled display source changed without a new renderer');
      if(this.modelStage?.next===next){this.modelStage.interaction=interaction;return true;}
      if(next===this.volume&&!this.modelStage){
        if(this.sourceFailed&&interaction?.view)throw Error('source surfaces are unavailable');
        if(this.sourceRemaining>0&&interaction?.view){this.initialSurgicalInteraction=interaction;return true;}
        this.setSurgical(interaction);this.queueSurgicalPublication(interaction);return true;
      }
      // A blocked extraction cannot consume cancellation messages. Replacing
      // its sole worker discards the obsolete job and all of its resident cells.
      if(this.modelStage){this.modelWorker?.terminate();this.modelWorker=null;this.modelWorkerStates.clear();this.clearModelStage();}
      // An accepted tool-only state may still be waiting behind a GPU fence.
      // Restore the published tools/effects while a newer tissue state prepares.
      if(this.pendingSurgicalPublication){this.pendingSurgicalPublication=null;this.setSurgical(this.publishedSurgicalInteraction);}
      ++this.sourceSurfaceGeneration;this.worker.terminate();this.sourceWorkerActive=false;this.sourceRemaining=0;this.initialSurgicalInteraction=null;
      const sameLayout=!!a&&!!b&&a.mappingKey===b.mappingKey&&!next.labelSets&&!this.volume.labelSets&&
        next.compartments.length===this.volume.compartments.length&&next.compartments.every((l,i)=>l.name===this.volume.compartments[i].name&&l.kind===this.volume.compartments[i].kind);
      const candidates=sameLayout?[...new Set([...a!.modifiedIndices,...b!.modifiedIndices])].sort((x,y)=>x-y):null;
      const stage:ModelStage={generation:++this.modelGeneration,next,interaction,prepared:new Map(),jobs:[],pack:!sameLayout,changedLayers:0,changedIndices:0,
        initialMapping:!a&&!!b&&!b.modifiedIndices.length};
      this.modelStage=stage;
      for(const layer of next.compartments){
        const old=this.volume.compartments.find(l=>l.name===layer.name);
        let indices:Uint32Array|undefined,values:Uint8Array|undefined;
        if(old&&old.kind===layer.kind&&this.preparedSourceLayers.has(layer.name)){
          if(old.mask===layer.mask&&!old.labelSet&&!layer.labelSet)continue;
          if(candidates){const changed=candidates.filter(i=>old.mask[i]!==layer.mask[i]);if(!changed.length)continue;indices=Uint32Array.from(changed);values=Uint8Array.from(changed,i=>layer.mask[i]);stage.changedIndices+=changed.length;}
        }
        stage.changedLayers++;
        const retired=this.retiredSurfaces.findIndex(e=>e.layer.name===layer.name&&e.layer.kind===layer.kind&&e.layer.mask===layer.mask);
        if(retired>=0){stage.prepared.set(layer.name,this.retiredSurfaces.splice(retired,1)[0].pair);continue;}
        const cached=!layer.labelSet?cachedSurface(layer.mask,this.affine,layer.kind,next.shape):undefined,s=layer.surface;
        const buffers=cached??(s?{positions:s.source,normals:s.sourceNormals,smoothPositions:s.positions,smoothNormals:s.normals,indices:s.indices}:undefined);
        if(buffers){const pair=buffers.positions.length?this.makeSurface(layer,buffers):null;if(buffers.positions.length&&!pair)throw Error('Invalid prepared modeled surface');stage.prepared.set(layer.name,pair);}
        else stage.jobs.push({layer,indices,values});
      }
      for(const old of this.volume.compartments)if(!next.compartments.some(l=>l.name===old.name)){stage.prepared.set(old.name,null);stage.changedLayers++;}
      if(candidates){
        const indices:number[]=[],values:number[]=[],current=this.labelTexture.image.data as Uint8Array;
        for(const index of candidates){const row=[0,0,0,0];next.compartments.forEach((layer,i)=>{const [channel,flag]=this.labelCodes[i];if(flag>=0)throw Error('Modeled label layout requires independent source masks');if(layer.mask[index])row[channel]|=-flag;});
          if(row.some((value,c)=>value!==current[index*4+c])){indices.push(index);values.push(...row);}}
        stage.patch={indices:Uint32Array.from(indices),values:Uint8Array.from(values)};
      }else if(next.packedLabels){stage.packed=next.packedLabels;stage.pack=false;}
      this.onSurfaceStatus(stage.jobs.length+Number(stage.pack));
      this.sendModelJob();
      return true;
    }catch(cause){this.failModelStage(cause instanceof Error?cause.message:String(cause),interaction,next);return false;}
  }
  private failModelStage(message:string,interaction=this.modelStage?.interaction??null,next=this.modelStage?.next){
    if(next)this.failedModelState={next,target:interaction?.preview??interaction?.view??null};
    this.modelWorker?.terminate();this.modelWorker=null;this.modelWorkerStates.clear();this.clearModelStage();this.onSurfaceStatus(0);
    const detail=`Modeled display retained its previous accepted state: ${message}`;
    if(interaction?.onDisplayError)interaction.onDisplayError(interaction.preview??interaction.view,detail);
    else this.onError(detail);
    this.requestRender();
  }
  private sendModelJob(){
    const stage=this.modelStage;if(!stage)return;
    const job=stage.jobs.shift();if(!job&&!stage.pack){stage.ready=true;this.requestRender();return;}
    if(!this.modelWorker){this.modelWorker=new Worker(new URL('./incrementalSurface.worker.ts',import.meta.url),{type:'module'});this.displayWork.modelWorkerBuilds++;}
    const worker=this.modelWorker;
    worker.onerror=()=>{
      if(this.disposed||this.modelWorker!==worker)return;
      if(this.modelStage!==stage){
        // An idle worker can fail after its last successful response. Release
        // its lost resident state; the next request admits a fresh worker.
        if(!this.modelStage){worker.terminate();this.modelWorker=null;this.modelWorkerStates.clear();}
        return;
      }
      this.failModelStage('surface worker failed');
    };
    worker.onmessage=(event:MessageEvent<PreparedSurface&{jobId:number;layerId:string;identity:string;resident?:Array<{layerId:string;identity:string}>;work?:Record<string,number|boolean>;task?:'pack';packed?:{data:Uint8Array;codes:Array<[number,number]>};error?:string}>)=>{
      if(this.disposed||this.modelStage!==stage||event.data.jobId!==stage.generation)return;
      if(event.data.error){this.failModelStage(event.data.error);return;}
      if(event.data.task==='pack'){
        const packed=event.data.packed;if(!packed||packed.data.length!==stage.next.mri.length*4){this.failModelStage('invalid modeled label buffers');return;}stage.packed=packed;
      }else{
        const layer=stage.waiting;if(!layer||layer==='pack'||layer.name!==event.data.layerId){this.failModelStage('unexpected modeled surface');return;}
        const buffers=event.data;if(!buffers.positions||!buffers.normals||!buffers.smoothPositions||!buffers.smoothNormals||buffers.positions.length!==buffers.normals.length||buffers.smoothPositions.length!==buffers.positions.length||buffers.smoothNormals.length!==buffers.positions.length){this.failModelStage('inconsistent modeled surface buffers');return;}
        retainSurface(layer.mask,this.affine,layer.kind,buffers,stage.next.shape);
        const pair=buffers.positions.length?this.makeSurface(layer,buffers):null;if(buffers.positions.length&&!pair){this.failModelStage('invalid modeled surface');return;}
        stage.prepared.set(layer.name,pair);this.modelWorkerStates=new Map((event.data.resident??[]).map(e=>[e.layerId,e.identity]));
        const work=event.data.work;this.displayWork.modelRemeshedLayers++;this.displayWork.modelRemeshedChunks+=Number(work?.remeshedChunks??0);
        this.canvas.dispatchEvent(new CustomEvent('anatomy-work',{bubbles:true,detail:{kind:'modeled-surface-preparation',name:layer.name,...work}}));
      }
      stage.waiting=undefined;this.onSurfaceStatus(stage.jobs.length+Number(stage.pack));this.sendModelJob();
    };
    if(job){
      stage.waiting=job.layer;const oldLayer=this.volume.compartments.find(l=>l.name===job.layer.name),previousIdentity=oldLayer?this.modelIdentity(oldLayer.mask):'',identity=this.modelIdentity(job.layer.mask),incremental=job.indices&&this.modelWorkerStates.get(job.layer.name)===previousIdentity;
      const request:IncrementalSurfaceRequest={jobId:stage.generation,layerId:job.layer.name,previousIdentity,identity,shape:stage.next.shape,affine:this.affine,kind:job.layer.kind,retainChunks:!stage.initialMapping};
      if(incremental){request.indices=job.indices;request.values=job.values;this.displayWork.workerInputTransferBytes+=job.indices!.byteLength+job.values!.byteLength;worker.postMessage(request,[job.indices!.buffer,job.values!.buffer]);}
      else {request.mask=materializeLayerMask(stage.next,job.layer);this.displayWork.workerInputCloneBytes+=request.mask.byteLength;worker.postMessage(request);}
    }else{
      stage.pack=false;stage.waiting='pack';
      this.displayWork.workerInputCloneBytes+=stage.next.compartments.reduce((sum,l)=>sum+l.mask.byteLength,0);
      worker.postMessage({task:'pack',jobId:stage.generation,volume:{shape:stage.next.shape,compartments:stage.next.compartments.map(({name,mask,labelSet,labelValue})=>({name,mask,labelSet,labelValue})),labelSets:stage.next.labelSets}});
    }
  }
  private commitModelStage(stage:ModelStage){
    if(this.disposed||this.modelStage!==stage)return;
    const oldVolume=this.volume;
    // Native pre-mapping pairs do not match modeled baseline/undo masks. Keep
    // their CPU surface cache for source restoration, but release GPU/BVH here.
    for(const [name,pair]of stage.prepared){const old=oldVolume.compartments.find(l=>l.name===name);if(old)this.retireSurface(old,!stage.initialMapping);
      this.preparedSourceLayers.add(name);if(pair){pair.mesh.geometry=this.surfaceSmoothing?pair.smooth:pair.source;pair.mesh.visible=this.visible[name]!==false;this.surfaceGeometry.set(name,{source:pair.source,smooth:pair.smooth});this.surfaces.set(name,pair.mesh);this.anatomy.add(pair.mesh);}}
    this.volume=stage.next;
    if(stage.packed){this.labelTexture.image.data=stage.packed.data;this.labelCodes=stage.packed.codes;}
    if(stage.patch){const data=this.labelTexture.image.data as Uint8Array;stage.patch.indices.forEach((index,i)=>data.set(stage.patch!.values.subarray(i*4,i*4+4),index*4));}
    this.labelTextureUpdates.update(this.renderer,this.labelTexture,this.volume.shape,stage.patch,!!stage.packed);
    const bits=this.volume.compartments.reduce((value,layer,i)=>value|(this.visible[layer.name]!==false?1<<i:0),0);
    this.materials().forEach(shader=>{shader.uniforms.uLayerCount.value=this.volume.compartments.length;this.labelCodes.forEach((code,i)=>shader.uniforms.uLayerCodes.value[i].set(...code));
      shader.uniforms.uVisibleBits.value=bits;this.volume.compartments.forEach((layer,i)=>shader.uniforms.uLayerEnabled.value[i]=this.visible[layer.name]!==false?1:0);});
    this.scanVolume.setLayers(this.volume.compartments,this.labelCodes);this.appearanceSignature=null;this.visibilitySignature=null;this.setWorkspaceSettings(this.workspaceSettings);
    this.modelStage=null;this.displayWork.modelCommits++;this.onSurfaceStatus(0);this.setSurgical(stage.interaction);this.requestRender();
    this.queueSurgicalPublication(stage.interaction);
  }

  private materials(): THREE.ShaderMaterial[] {
    return [
      this.sourcePlane.material,
      ...Array.from(this.slices.values(), (slice) => slice.material),
    ];
  }

  update(
    cursor: Point3,
    visible: Record<string, boolean>,
    opacity: number,
    routes: ViewerRoute[],
    cameraMode: "anatomy" | "instruments",
  ): void {
    const cursorChanged = cursor.some((value, axis) => value !== this.cursor[axis]);
    const visibilitySignature = this.volume.compartments.map(layer => visible[layer.name] !== false ? '1' : '0').join('');
    const visibilityChanged = visibilitySignature !== this.visibilitySignature;
    if(visibilityChanged&&this.visibilitySignature!=null){this.beginInteraction();this.endInteraction();}
    const nextOpacity = Math.max(0, Math.min(1, opacity));
    const opacityChanged = nextOpacity !== this.opacity;
    this.visible = visible;
    this.opacity = nextOpacity;
    if (visibilityChanged) {
      this.visibilitySignature = visibilitySignature;
      this.scanVolume.labels(this.labelTexture,visible,this.workspaceSettings,this.extraClipPlanes);
      let bits = 0;
      this.volume.compartments.forEach((layer, index) => {
        if (visible[layer.name] !== false) bits |= 1 << index;
        const mesh = this.surfaces.get(layer.name);
        if (mesh) mesh.visible = visible[layer.name] !== false;
      });
      for (const group of [this.replayGroup, this.pendingReplayGroup])
        group?.children.forEach(mesh => this.applyReplaySurfaceAppearance(mesh));
      this.materials().forEach(shader => {
        shader.uniforms.uVisibleBits.value = bits;
        this.volume.compartments.forEach((layer,i) => { shader.uniforms.uLayerEnabled.value[i] = visible[layer.name] !== false ? 1 : 0; });
      });
    }
    if (cursorChanged) {
      this.cursor = [...cursor];
      this.materials().forEach(shader => shader.uniforms.uCursor.value.set(...cursor));
      this.cursorObject.position.set(...cursor);
      this.updateSourcePlane();
    }
    if (opacityChanged) this.materials().forEach(shader => { shader.uniforms.uOverlay.value = this.opacity; });
    const signature = JSON.stringify(routes);
    const routesChanged = signature !== this.routeSignature;
    if (routesChanged) {
      this.routeSignature = signature;
      this.updateRoutes(routes);
    }
    const modeChanged = cameraMode !== this.mode;
    if (modeChanged) {
      this.mode = cameraMode;
      this.fitCamera(cameraMode);
    }
    if (cursorChanged || visibilityChanged || opacityChanged || routesChanged || modeChanged) this.requestRender();
  }

  setWindow(low: number, high: number): void {
    if (!Number.isFinite(low) || !Number.isFinite(high) || high <= low) return;
    this.materials().forEach((shader) =>
      shader.uniforms.uWindow.value.set(low, high),
    );
    this.requestRender();
  }

  setWorkspaceSettings(settings: ViewerWorkspaceSettings): void {
    this.workspaceSettings = settings;
    const framingSignature=JSON.stringify(settings.sliceFraming);
    if(framingSignature!==this.sliceFramingSignature){this.sliceFramingSignature=framingSignature;this.requestRender();}
    // Camera persistence and toolbar/layout choices do not change anatomy
    // materials. Compare values so cloned or in-place edited settings work too.
    const signature = JSON.stringify([settings.layerOpacity, settings.layerColor,
      settings.clipping, settings.clipBox, settings.obliqueClip,
      settings.fusion?.opacity, settings.volumeRendering,settings.skullMode,settings.sliceOverlay,settings.contextStyle]);
    if (signature === this.appearanceSignature) return;
    if(this.appearanceSignature!=null){this.beginInteraction();this.endInteraction();}
    this.appearanceSignature = signature;
    const clipping = clippingForBounds(settings.clipping,this.bounds);
    const normal = new THREE.Vector3();
    normal.setComponent({x:0,y:1,z:2}[clipping.axis],clipping.flipped ? -1 : 1);
    this.clipPlane = clipping.enabled ? new THREE.Plane(normal,clipping.flipped ? clipping.value : -clipping.value) : null;
    this.extraClipPlanes=[];
    const box=settings.clipBox;
    if(box?.enabled && [...box.min,...box.max].every(Number.isFinite))for(let i=0;i<3;i++){
      const n=new THREE.Vector3().setComponent(i,1);
      this.extraClipPlanes.push(new THREE.Plane(n,-box.min[i]),new THREE.Plane(n.clone().negate(),box.max[i]));
    }
    const oblique=settings.obliqueClip;
    if(oblique?.enabled && [...oblique.normal,oblique.offset].every(Number.isFinite) && Math.hypot(...oblique.normal)>1e-5){
      const n=new THREE.Vector3(...oblique.normal);this.extraClipPlanes.push(new THREE.Plane(n,oblique.offset).normalize());
    }
    this.volume.compartments.forEach((layer,index)=>{
      const appearance=layerAppearance(layer.name,layer.color,settings,layer.kind);
      const mesh=this.surfaces.get(layer.name);
      if(mesh){
        const material=mesh.material as THREE.MeshStandardMaterial;
        material.color.set(appearance.color);material.opacity=appearance.opacity;
        applySourceContextMaterial(material,settings.contextStyle==='shell'&&['brain','head-envelope','bone','skull'].includes(layer.kind??''));
        const transparent = appearance.opacity < 1;
        if (material.transparent !== transparent) {
          material.transparent = transparent;
          material.needsUpdate = true;
        }
        material.depthWrite = !transparent;
        material.clippingPlanes=[...this.extraClipPlanes,...(this.clipPlane ? [this.clipPlane] : [])];
      }
      this.materials().forEach(shader=>{
        shader.uniforms.uColors.value[index].set(appearance.color).convertLinearToSRGB();
        shader.uniforms.uLayerOpacities.value[index]=appearance.opacity;
      });
    });
    // Derived residuals and explicitly colored effects follow cutaway changes
    // even while the worker is preparing the rest of the snapshot.
    for (const group of [this.replayGroup, this.pendingReplayGroup])
      group?.children.forEach(mesh => this.applyReplaySurfaceAppearance(mesh));
    this.materials().forEach(shader=>{
      shader.uniforms.uSliceContours.value=settings.sliceOverlay==='contour'?1:0;
      shader.uniforms.uClipEnabled.value=clipping.enabled ? 1 : 0;
      shader.uniforms.uClipPlane.value.set(normal.x,normal.y,normal.z,clipping.flipped ? clipping.value : -clipping.value);
      shader.uniforms.uFusionOpacity.value=Number.isFinite(settings.fusion?.opacity) ? Math.max(0,Math.min(1,settings.fusion!.opacity)) : .5;
    });
    this.scanVolume?.configure(settings.volumeRendering, this.volumeModality, this.clipPlane);
    this.scanVolume.labels(this.labelTexture,this.visible,settings,this.extraClipPlanes);
    this.requestRender();
  }

  /** Display-only styling for derived anatomy/effects. Instrument groups never
   * enter this path: their complete safety geometry stays visible and unclipped. */
  private applyReplaySurfaceAppearance(object: THREE.Object3D): void {
    if (!(object instanceof THREE.Mesh)) return;
    const material = object.material as THREE.MeshStandardMaterial;
    const layer = this.volume.compartments.find(item => item.name === object.userData.compartment);
    if (layer) {
      const appearance = layerAppearance(layer.name, layer.color, this.workspaceSettings,layer.kind);
      material.color.set(appearance.color);
      material.opacity = appearance.opacity;
      object.visible = this.visible[layer.name] !== false;
    } else {
      // Effect overlays retain their explicit evidence colors and opacity.
      object.visible = true;
    }
    const transparent = material.opacity < 1;
    const clippingPlanes = [...this.extraClipPlanes, ...(this.clipPlane ? [this.clipPlane] : [])];
    if (material.transparent !== transparent || material.clippingPlanes?.length !== clippingPlanes.length)
      material.needsUpdate = true;
    material.transparent = transparent;
    material.depthWrite = !transparent;
    material.clippingPlanes = clippingPlanes;
  }

  setVolumeRendering(settings: VolumeRenderingSettings | undefined, modality: string): void {
    this.volumeModality = modality;
    this.scanVolume.configure(settings, modality, this.clipPlane);
    this.requestRender();
  }

  /** Reversible mesh-only display; masks, slice textures and metrics are untouched. */
  setSurfaceSmoothing(enabled: boolean): void {
    this.surfaceSmoothing = enabled;
    this.surfaces.forEach((mesh, name) => {
      const pair = this.surfaceGeometry.get(name);
      if (pair) mesh.geometry = enabled ? pair.smooth : pair.source;
    });
    for (const group of [this.replayGroup, this.pendingReplayGroup])
      group?.children.forEach(mesh => {
        const pair = mesh.userData.replayGeometry as {source:THREE.BufferGeometry;smooth:THREE.BufferGeometry}|undefined;
        if (mesh instanceof THREE.Mesh && pair) mesh.geometry = enabled ? pair.smooth : pair.source;
      });
    this.requestRender();
  }

  setRegisteredFusion(fusion: ViewerRegisteredFusion|null): void {
    // Clear before validation so a rejected replacement never leaves stale fusion.
    this.materials().forEach(shader=>{shader.uniforms.uFusionActive.value=0;});
    this.fusionTexture.dispose();
    this.fusionTexture=dataTexture(new Float32Array(1),[1,1,1]);
    this.materials().forEach(shader=>{shader.uniforms.uFusion.value=this.fusionTexture;});
    this.requestRender();
    if(!fusion)return;
    const worldToFusion=checkedFusionTransform(this.volume,fusion);
    const gl=this.renderer.getContext() as WebGL2RenderingContext;
    if(fusion.volume.shape.some(n=>n>gl.getParameter(gl.MAX_3D_TEXTURE_SIZE)))throw new Error('Fusion image exceeds the GPU texture limit.');
    const window=intensityWindow(fusion.volume.mri);
    this.fusionTexture.dispose();this.fusionTexture=dataTexture(fusion.volume.mri,fusion.volume.shape);
    this.materials().forEach(shader=>{
      shader.uniforms.uFusion.value=this.fusionTexture;
      shader.uniforms.uFusionShape.value.set(...fusion.volume.shape);
      shader.uniforms.uWorldToFusion.value.copy(matrix(worldToFusion));
      shader.uniforms.uFusionWindow.value.set(...window);
      shader.uniforms.uFusionActive.value=1;
    });
    this.requestRender();
  }

  getCameraState(): ViewerCameraState {
    return {position:this.camera.position.toArray() as Point3,target:this.controls.target.toArray() as Point3,up:this.camera.up.toArray() as Point3};
  }

  private surgicalGroup: THREE.Group | null = null;
  private surgicalGhost: THREE.Group | null = null;
  private surgicalGizmo: TransformControls | null = null;
  private surgicalInteraction: SurgicalInteraction | null = null;
  private surgicalEffectsView: SurgicalInteraction['view'] = null;
  private surgicalEffectsOutcome: SurgicalInteraction['outcome'];
  private surgicalEffectsPreview = false;
  private surgicalEffects: THREE.Group | null = null;
  private surgicalSlots = new Map<string,{key:string;group:THREE.Group}>();
  private surgicalGhostKey: string | null = null;
  private surgicalPath: THREE.Line | null = null;
  private surgicalPathSource: SurgicalInteraction['path'] | null = null;
  private readonly displayWork = {surgicalUpdates:0,toolBuilds:0,effectBuilds:0,pathBufferBuilds:0,previewStateInstalls:0,modelWorkerBuilds:0,modelRemeshedLayers:0,modelRemeshedChunks:0,modelCommits:0,
    sourceWorkerBuilds:0,replayWorkerBuilds:0,sourceRemeshedLayers:0,replayRemeshedLayers:0,workerInputCloneBytes:0,workerInputTransferBytes:0,workerResultTransferBytes:0};
  setSurgical(interaction: SurgicalInteraction | null): void {
    this.surgicalInteraction=interaction;
    if(!this.surgicalGroup){this.surgicalGroup=new THREE.Group();this.scene.add(this.surgicalGroup);}
    // Pose editing is independent of the accepted tissue/effect state. Keep its
    // geometry resident while the ghost and recorded tools move by transforms.
    if(!interaction?.view){
      disposeObject(this.surgicalGroup);this.surgicalGroup.clear();this.surgicalSlots?.clear();
      this.surgicalEffects=null;this.surgicalEffectsView=null;this.surgicalEffectsOutcome=undefined;
      this.surgicalPath=null;this.surgicalPathSource=null;
      this.surgicalGizmo?.detach();if(this.surgicalGhost)this.surgicalGhost.visible=false;
      this.controls.enabled=true;this.requestRender();return;
    }
    // Lazily initialize for the CPU-only renderer seams used by regression tests.
    this.surgicalSlots??=new Map();
    this.displayWork.surgicalUpdates++;
    const {pose,path,selectedSlot,mode}=interaction;const view=interaction.preview??interaction.view;
    if(this.surgicalEffectsView!==view||this.surgicalEffectsOutcome!==interaction.outcome||this.surgicalEffectsPreview!==!!interaction.preview){
      if(interaction.preview)this.displayWork.previewStateInstalls++;
      if(this.surgicalEffects){disposeObject(this.surgicalEffects);this.surgicalGroup.remove(this.surgicalEffects);}
      this.surgicalEffects=surgicalEffectGroup(view,!!interaction.preview,interaction.outcome);
      this.surgicalGroup.add(this.surgicalEffects);this.displayWork.effectBuilds++;
      this.surgicalEffectsView=view;this.surgicalEffectsOutcome=interaction.outcome;this.surgicalEffectsPreview=!!interaction.preview;
    }
    const retained=new Set<string>();
    for(const [name,recordedSlot] of Object.entries(view.slots)){
      const effective=interaction.effectiveActivation?.[name];
      const slot=typeof effective==='boolean'?{...recordedSlot,active:effective}:recordedSlot;
      const tool=view.catalog.find(t=>t.id===slot.instrument);if(!tool)continue;
      retained.add(name);const key=surgicalToolGeometryKey(tool,slot),old=this.surgicalSlots.get(name);
      if(old?.key===key)updateSurgicalToolMesh(old.group,slot);
      else {
        if(old){disposeObject(old.group);this.surgicalGroup.remove(old.group);}
        const group=surgicalToolMesh(tool,slot);this.surgicalSlots.set(name,{key,group});this.surgicalGroup.add(group);this.displayWork.toolBuilds++;
      }
    }
    for(const [name,old] of this.surgicalSlots)if(!retained.has(name)){
      disposeObject(old.group);this.surgicalGroup.remove(old.group);this.surgicalSlots.delete(name);
    }
    if(!this.surgicalGhost){
      this.surgicalGhost=new THREE.Group();this.scene.add(this.surgicalGhost);
      this.surgicalGizmo=new TransformControls(this.camera,this.panes.anatomy);
      this.scene.add(this.surgicalGizmo.getHelper());
      this.surgicalGizmo.addEventListener('dragging-changed',event=>{this.controls.enabled=!event.value;if(event.value)this.beginInteraction();else this.endInteraction();});
      this.surgicalGizmo.addEventListener('objectChange',()=>{if(this.surgicalGhost&&this.surgicalInteraction){this.surgicalInteraction.onPose({position:this.surgicalGhost.position.toArray(),quaternion:this.surgicalGhost.quaternion.toArray()});this.requestRender();}});
    }
    this.surgicalGhost.visible=!interaction.readOnly;
    this.surgicalGhost.position.set(...pose.position);this.surgicalGhost.quaternion.set(...pose.quaternion);
    const selected=view.slots[selectedSlot],tool=selected?view.catalog.find(t=>t.id===selected.instrument):undefined;
    const ghostSlot=selected?{...selected,parameters:{...selected.parameters,...interaction.parameters},position:[0,0,0] as Point3,quaternion:[0,0,0,1] as [number,number,number,number]}:null;
    const ghostKey=tool&&ghostSlot?surgicalToolGeometryKey(tool,ghostSlot,true):null;
    if(ghostKey!==this.surgicalGhostKey){
      disposeObject(this.surgicalGhost);this.surgicalGhost.clear();this.surgicalGhostKey=ghostKey;
      if(tool&&ghostSlot){this.surgicalGhost.add(surgicalToolMesh(tool,ghostSlot,true));this.displayWork.toolBuilds++;}
    }
    if(mode==='translate'||mode==='rotate'){this.surgicalGizmo!.attach(this.surgicalGhost);this.surgicalGizmo!.setMode(mode);}else this.surgicalGizmo!.detach();
    this.controls.enabled=mode!=='draw';
    if(path!==this.surgicalPathSource){
      this.surgicalPathSource=path;
      if(path.length>1){
        const capacity=this.surgicalPath?.geometry.getAttribute('position').count??0;
        if(capacity<path.length){
          if(this.surgicalPath){disposeObject(this.surgicalPath);this.surgicalGroup.remove(this.surgicalPath);}
          const geometry=new THREE.BufferGeometry(),size=2**Math.ceil(Math.log2(Math.max(16,path.length)));
          geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(size*3),3).setUsage(THREE.DynamicDrawUsage));
          this.surgicalPath=new THREE.Line(geometry,new THREE.LineBasicMaterial({color:'#8cffe0'}));
          this.surgicalPath.frustumCulled=false;this.surgicalGroup.add(this.surgicalPath);this.displayWork.pathBufferBuilds++;
        }
        const attribute=this.surgicalPath!.geometry.getAttribute('position') as THREE.BufferAttribute;
        path.forEach((p,i)=>attribute.setXYZ(i,...p.position));attribute.needsUpdate=true;
        this.surgicalPath!.geometry.setDrawRange(0,path.length);this.surgicalPath!.visible=true;
      }else if(this.surgicalPath)this.surgicalPath.visible=false;
    }
    this.requestRender();
  }

  setNavigationMode(mode: 'orbit'|'pan'): void {
    this.controls.mouseButtons.LEFT=mode==='pan' ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE;
    this.controls.touches.ONE=mode==='pan' ? THREE.TOUCH.PAN : THREE.TOUCH.ROTATE;
  }

  beginInteraction():void { this.interaction.start(); }
  endInteraction():void { this.refinementStarted=performance.now();this.interaction.end(); }
  retryGraphics():void {
    if(this.disposed)return;
    const gl=this.renderer.getContext() as WebGL2RenderingContext;
    if(!gl.isContextLost()){this.onGraphicsStatus?.('ready');return;}
    const recovery=this.graphicsRecovery;
    if(recovery)recovery.restoreContext();
    else this.onError('Graphics recovery is unavailable in this session. Restart the application to restore the source-image views.');
  }

  restoreCamera(state: ViewerCameraState|null|undefined): void {
    if(!validCameraState(state))return;
    this.cameraRestored=true;this.pendingFit=null;
    const unchanged = this.camera.position.equals(new THREE.Vector3(...state.position)) &&
      this.controls.target.equals(new THREE.Vector3(...state.target)) &&
      this.camera.up.equals(new THREE.Vector3(...state.up).normalize());
    const distance = new THREE.Vector3(...state.position).distanceTo(new THREE.Vector3(...state.target));
    const near = Math.max(.1,distance/1000), far = Math.max(1000,distance*8);
    if (unchanged && this.camera.near === near && this.camera.far === far) return;
    this.camera.position.set(...state.position);this.controls.target.set(...state.target);this.camera.up.set(...state.up).normalize();
    this.camera.near=near;this.camera.far=far;
    this.camera.updateProjectionMatrix();this.controls.update();this.requestRender(false);
  }

  cameraPreset(preset: 'oblique'|'anterior'|'left'|'superior'): void {
    this.fitCamera(this.mode,preset);
    this.cameraRestored=true;
    this.onCameraChange?.(this.getCameraState());
  }

  setSourcePlane(plane: SlicePlane | null): void {
    this.sourcePlane.visible = plane !== null;
    if (plane !== null) this.activePlane = plane;
    this.updateSourcePlane();
    this.requestRender();
  }

  /** Declared task marker only; never tissue, removal, route or measurement. */
  setPublicGoal(input: ViewerPublicGoal | null): void {
    disposeObject(this.publicGoalObject); this.publicGoalObject.clear();
    const goal = checkedPublicGoal(this.volume, input);
    if (goal) {
      const marker = new THREE.Mesh(new THREE.SphereGeometry(.38, 12, 8), new THREE.MeshBasicMaterial({color: PUBLIC_GOAL_COLOR, wireframe: true, depthTest: false}));
      marker.position.set(...goal.rasMm); marker.renderOrder = 10;
      marker.userData.publicGoal = goal; this.publicGoalObject.add(marker);
    }
    this.requestRender();
  }

  /** A certified effect overlays the original MRI; source labels remain immutable. */
  setReplay(replay: ViewerReplay | null): void {
    // A requested mode switch clears an unexecuted tool even if replay rejects.
    if (replay) {
      this.setInspectionTool(null);
      this.setDiagnosticLayer(null, null);
    }
    if (this.replayActive) this.onSurfaceStatus(0);
    const generation = ++this.replayGeneration;
    this.replayWorker?.terminate();
    this.replayWorker = null;
    if (this.pendingReplayGroup) disposeObject(this.pendingReplayGroup);
    this.pendingReplayGroup = null;
    disposeObject(this.replayGroup);
    this.replayGroup.clear();
    this.replayActive = false;
    this.recordedDisplay = null;
    this.passiveActive = false;
    this.passiveDisplays = [];
    this.rehearsalActive = false;
    this.rehearsalDisplays = [];
    this.replayBounds = null;
    disposeObject(this.recordedTools);
    this.recordedTools.clear();
    this.anatomy.visible = true;
    this.instrumentDisplay.setReplay(false);
    this.syncInstrumentDisplay();
    this.materials().forEach((shader) => {
      shader.uniforms.uReplayActive.value = 0;
    });
    this.requestRender();
    if (!replay) return;
    validateReplay(this.volume, replay);
    const recorded = recordedToolDisplay(this.volume, replay);
    const cameraBounds = checkedRecordedEpisodeBounds(this.volume, replay);
    this.setStructuralProposal(null);
    this.setPriorLayer(null);

    // Legacy certificates retain their original validation before entering the
    // shared display-only surface builder.
    const removed = replay.removedMask.slice();
    this.installRemovalDisplay(removed, removed, recorded ? [recorded] : [], cameraBounds,
      "executed-generated-episode", [{id:"modeled-removal",name:"Modeled removal",color:"#b7f0d8",mask:removed,removed:true}]);
  }

  /** Separate patient-source research gate; never calls generated replay acceptance. */
  setRehearsal(state: ViewerRehearsalState | null): void {
    if (!state) { if (this.rehearsalActive) this.setReplay(null); return; }
    // Withhold an invalid replacement before retaining any older effect.
    let checked: ReturnType<typeof checkedRehearsalState>;
    try { checked = checkedRehearsalState(this.volume, state); }
    catch (cause) { this.setReplay(null); throw cause; }
    if (this.rehearsalActive) {
      ++this.replayGeneration;
      // A busy worker cannot process a cancellation message while extracting a
      // surface. Replace it safely; an idle worker survives completed seeks.
      if(this.pendingReplayGroup){this.replayWorker?.terminate();this.replayWorker=null;}
      if (this.pendingReplayGroup) disposeObject(this.pendingReplayGroup);
      this.pendingReplayGroup=null;
      disposeObject(this.recordedTools);this.recordedTools.clear();
    } else this.setReplay(null);
    this.setInspectionTool(null);
    this.setDiagnosticLayer(null, null);
    this.setStructuralProposal(null);
    this.setPriorLayer(null);
    this.rehearsalActive = true;
    this.rehearsalDisplays = checked.displays;
    this.installRemovalDisplay(checked.voidMask, checked.effectMask, checked.displays,
      checked.bounds, state.scope, [
        {id:"assumed-exposure",name:"Assumed initial exposure",color:REHEARSAL_COLORS.exposure,mask:state.assumedExposureMask.slice(),removed:true},
        {id:"rehearsal-removal",name:"Modeled episode removal",color:REHEARSAL_COLORS.removal,mask:state.removedMask.slice(),removed:true},
        {id:"rehearsal-field-contact",name:"Modeled field-management contact",color:REHEARSAL_COLORS.field_management,mask:residualMask(state.fieldContactMask,checked.voidMask),removed:true},
      ]);
  }

  /** Builds derived meshes and textures only. Original source arrays never change. */
  private installRemovalDisplay(removed: Uint8Array, effects: Uint8Array,
    displays: RecordedToolDisplay[], cameraBounds: RecordedEpisodeBounds | null,
    scope: string, overlays: Array<{id:string;name:string;color:string;mask:Uint8Array;removed:boolean}>): void {
    const generation = this.replayGeneration;
    const texture = dataTexture(effects, this.volume.shape);
    this.removedTexture.dispose();
    this.removedTexture = texture;
    this.materials().forEach((shader) => {
      shader.uniforms.uRemoved.value = texture;
      shader.uniforms.uReplayActive.value = 1;
      shader.uniforms.uRouteCount.value = 0;
    });
    this.replayActive = true;
    this.instrumentDisplay.setReplay(true);
    this.recordedDisplay = this.rehearsalActive ? null : (displays[0] ?? null);
    this.replayBounds = cameraBounds;
    for (const display of displays) this.recordedTools.add(instrumentCapsuleMeshes(display, scope, display.identity));
    disposeObject(this.inspectionTools);
    this.inspectionTools.clear();
    this.syncInstrumentDisplay();
    this.anatomy.visible = false;
    if (this.mode === "instruments" && cameraBounds && this.fittedReplayId !== cameraBounds.episodeId)
      this.fitCamera("instruments");
    this.tools.visible = false;
    const pending = new THREE.Group();
    this.pendingReplayGroup = pending;
    const layers = this.volume.compartments.map((layer, index) => ({
      id: `residual-${index}`,name: layer.name,color: layer.color,removed: false,
      kind:layer.kind,mask: residualMask(materializeLayerMask(this.volume,layer), removed),
    }));
    layers.push(...overlays.map(layer=>({...layer,kind:undefined})));
    const waiting = new Map(layers.map((layer) => [layer.id, layer]));
    this.onSurfaceStatus(waiting.size);
    const worker = this.replayWorker??new Worker(new URL("./surface.worker.ts", import.meta.url), {type: "module"});
    if(!this.replayWorker&&this.displayWork)this.displayWork.replayWorkerBuilds++;
    this.replayWorker = worker;
    let nextLayer=0,requestStarted=0;
    const sendNext=()=>{
      const layer=layers[nextLayer++];if(!layer)return;
      requestStarted=performance.now();
      // Legacy removal uses the same derived array for texture and overlay.
      // Transfer a disposable copy in that case, never the resident texture.
      const mask=layer.mask.buffer===effects.buffer?layer.mask.slice():layer.mask;
      if(this.displayWork){this.displayWork.replayRemeshedLayers++;this.displayWork.workerInputTransferBytes+=mask.byteLength;}
      // Residuals/overlays are disposable derived arrays. Source masks, MRI and
      // effect-texture storage stay owned by the renderer and are never detached.
      worker.postMessage({name:layer.id,mask,shape:this.volume.shape,affine:this.affine,kind:layer.kind,
        displaySmoothing:!layer.removed,requestGeneration:generation},[mask.buffer]);
    };
    const failed = (message: string) => {
      if (this.disposed || generation !== this.replayGeneration) return;
      this.setReplay(null);
      this.onSurfaceStatus(0);
      this.onReplayError(`Modeled anatomy could not be displayed: ${message}. The original source anatomy has been restored.`);
    };
    worker.onerror = () => failed("surface preparation failed");
    worker.onmessage = (event: MessageEvent<{name:string;positions?:Float32Array;normals?:Float32Array;smoothPositions?:Float32Array;smoothNormals?:Float32Array;indices?:Uint32Array;error?:string;requestGeneration?:number;diagnostics?:SurfaceDiagnostics}>) => {
      if (this.disposed || generation !== this.replayGeneration) return;
      if(event.data.requestGeneration!==undefined&&event.data.requestGeneration!==generation)return;
      if (event.data.error) {failed(event.data.error);return;}
      const layer = waiting.get(event.data.name);
      if (!layer) {failed("unexpected surface response");return;}
      waiting.delete(event.data.name);
      const positions = event.data.positions, normals = event.data.normals;
      if (!positions || !normals || normals.length !== positions.length) {failed("missing or inconsistent source-grid surface");return;}
      const resultBytes=[positions,normals,event.data.indices,event.data.smoothPositions,event.data.smoothNormals].reduce((sum,array)=>sum+(array?.byteLength??0),0);
      if(this.displayWork)this.displayWork.workerResultTransferBytes+=resultBytes;
      this.canvas?.dispatchEvent(new CustomEvent('anatomy-work',{bubbles:true,detail:{kind:'surface-preparation',scope:'rehearsal',name:event.data.name,
        requestToResultMs:performance.now()-requestStarted,preparationMs:event.data.diagnostics?.preparationMs??null,resultTransferBytes:resultBytes}}));
      if (positions.length) {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position",new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute("normal",new THREE.BufferAttribute(normals, 3));
        if (event.data.indices) geometry.setIndex(new THREE.BufferAttribute(event.data.indices,1));
        let smooth = geometry;
        if (!layer.removed) {
          const {smoothPositions,smoothNormals} = event.data;
          if (!smoothPositions || !smoothNormals || smoothPositions.length !== positions.length || smoothNormals.length !== positions.length) {
            geometry.dispose();failed("smoothed residual surface buffers do not match source geometry");return;
          }
          smooth = new THREE.BufferGeometry();
          smooth.setAttribute("position",new THREE.BufferAttribute(smoothPositions,3));
          smooth.setAttribute("normal",new THREE.BufferAttribute(smoothNormals,3));
          if (event.data.indices) smooth.setIndex(new THREE.BufferAttribute(event.data.indices,1));
        }
        const mesh = new THREE.Mesh(this.surfaceSmoothing ? smooth : geometry,new THREE.MeshStandardMaterial({
          color: layer.color,roughness: 0.42,metalness: 0,transparent: true,
          opacity: layer.removed ? 0.78 : 0.66,side: THREE.DoubleSide,depthWrite: false,wireframe: layer.removed,
        }));
        placeInSourceFrame(mesh, this.affine);
        this.picking?.install(mesh,geometry);
        if (!layer.removed) {
          mesh.userData.compartment = layer.name;
          mesh.userData.replayGeometry = {source:geometry,smooth};
        }
        this.applyReplaySurfaceAppearance(mesh);
        pending.add(mesh);
      }
      this.onSurfaceStatus(waiting.size);
      if (!waiting.size) {
        disposeObject(this.replayGroup);this.replayGroup.clear();
        for (const child of [...pending.children]) this.replayGroup.add(child);
        this.pendingReplayGroup = null;this.requestRender();
      }else sendNext();
    };
    sendNext();
    this.requestRender();
  }

  /** Separate generated passive replay: two persistent capsules, no removal mask.
   * Acceptance is source-bound and fails closed before any pose is displayed. */
  setPassiveReplay(replay: ViewerPassiveReplay | null): void {
    if (!replay) {
      if (this.passiveActive) this.setReplay(null);
      return;
    }
    // Clear any prior legacy effect and both old capsules before validation.
    this.setReplay(null);
    this.setInspectionTool(null);
    this.setDiagnosticLayer(null, null);
    this.setStructuralProposal(null);
    this.setPriorLayer(null);
    const checked = checkedPassiveRecordedTools(this.volume, replay);
    this.passiveActive = true;
    this.replayActive = true;
    this.instrumentDisplay.setReplay(true);
    this.passiveDisplays = checked.displays;
    this.replayBounds = checked.bounds;
    for (const display of checked.displays)
      this.recordedTools.add(instrumentCapsuleMeshes(display, replay.scope, display.identity));
    // Generated passive motion has no removal effect; labels and MRI remain the
    // unchanged source. The existing two capsule shader slots cover every MPR.
    this.materials().forEach((shader) => { shader.uniforms.uReplayActive.value = 0; });
    this.syncInstrumentDisplay();
    if (this.mode === "instruments" && this.fittedReplayId !== replay.episodeId)
      this.fitCamera("instruments");
    this.requestRender();
  }

  /** Estimates are isolated MRI contour textures, never source or target masks. */
  setStructuralProposal(proposal: ViewerStructuralProposal | null): void {
    // Clear first so rejected or superseded evidence cannot leave a stale contour.
    this.materials().forEach((shader) => {
      shader.uniforms.uProposalActive.value = 0;
    });
    this.proposalTexture.dispose();
    this.proposalTexture = dataTexture(new Uint8Array(1), [1, 1, 1]);
    this.materials().forEach((shader) => {
      shader.uniforms.uProposal.value = this.proposalTexture;
    });
    this.requestRender();
    if (!proposal || this.replayActive) return;
    validateStructuralProposal(this.volume, proposal);
    this.setDiagnosticLayer(null, null);
    this.setPriorLayer(null);
    const snapshot = proposal.mask.slice();
    this.proposalTexture.dispose();
    this.proposalTexture = dataTexture(snapshot, this.volume.shape);
    this.materials().forEach((shader) => {
      shader.uniforms.uProposal.value = this.proposalTexture;
      shader.uniforms.uProposalActive.value = 1;
    });
    this.requestRender();
  }

  /** A separate MRI-plane diagnostic; no route, replay, or planning data. */
  setDiagnosticLayer(layer: LoadedDiagnosticLayer | null,
                     selectedSourceSha256: string | null): CheckedDiagnosticLayer | null {
    this.materials().forEach((shader) => { shader.uniforms.uDiagnosticActive.value = 0; });
    this.diagnosticStateTexture.dispose();
    this.diagnosticCoverageTexture.dispose();
    this.diagnosticStateTexture = dataTexture(new Uint8Array(1), [1, 1, 1]);
    this.diagnosticCoverageTexture = dataTexture(new Uint8Array(1), [1, 1, 1]);
    this.materials().forEach((shader) => {
      shader.uniforms.uDiagnosticState.value = this.diagnosticStateTexture;
      shader.uniforms.uDiagnosticCoverage.value = this.diagnosticCoverageTexture;
    });
    this.requestRender();
    if (!layer) return null;
    // A rejected diagnostic must not leave an earlier planning overlay visible.
    // Keep the null path narrow so hiding this layer does not alter other modes.
    this.setReplay(null);
    this.setStructuralProposal(null);
    this.setPriorLayer(null);
    this.setInspectionTool(null);
    this.setPublicGoal(null);
    this.routeSignature = "[]";
    this.updateRoutes([]);
    if (!selectedSourceSha256) throw new Error("The displayed diagnostic source SHA is unavailable.");
    const checked = validateDiagnosticLayer(layer, this.volume, selectedSourceSha256);
    const positive = new Uint8Array(layer.stateXYZ.length);
    for (let i = 0; i < positive.length; i++) positive[i] = layer.stateXYZ[i] === 1 ? 1 : 0;
    this.diagnosticStateTexture.dispose();
    this.diagnosticCoverageTexture.dispose();
    this.diagnosticStateTexture = dataTexture(positive, checked.shape);
    this.diagnosticCoverageTexture = dataTexture(layer.coverageXYZ.slice(), checked.shape);
    this.materials().forEach((shader) => {
      shader.uniforms.uDiagnosticState.value = this.diagnosticStateTexture;
      shader.uniforms.uDiagnosticCoverage.value = this.diagnosticCoverageTexture;
      shader.uniforms.uDiagnosticActive.value = 1;
    });
    this.requestRender();
    return checked;
  }

  /** Population maps have independent values, FOV coverage, and physical frame. */
  setPriorLayer(layer: ViewerPriorLayer | null): ViewerPriorLayer | null {
    this.materials().forEach((shader) => {
      shader.uniforms.uPriorActive.value = 0;
    });
    this.priorTexture.dispose();
    this.priorCoverageTexture.dispose();
    this.priorTexture = dataTexture(new Float32Array(1), [1, 1, 1]);
    this.priorCoverageTexture = dataTexture(new Uint8Array(1), [1, 1, 1]);
    this.materials().forEach((shader) => {
      shader.uniforms.uPrior.value = this.priorTexture;
      shader.uniforms.uPriorCoverage.value = this.priorCoverageTexture;
    });
    this.requestRender();
    if (!layer || this.replayActive) return null;
    validatePriorLayer(this.volume, layer);
    this.setDiagnosticLayer(null, null);
    const snapshot: ViewerPriorLayer = {
      ...layer,
      values: layer.values.slice(),
      coverage: layer.coverage.slice(),
      shape: [...layer.shape],
      affine: layer.affine.map((row) => [...row]),
    };
    // A selected prior and an estimated envelope are separate inspection modes.
    this.setStructuralProposal(null);
    this.priorTexture.dispose();
    this.priorCoverageTexture.dispose();
    this.priorTexture = dataTexture(snapshot.values, snapshot.shape);
    this.priorCoverageTexture = dataTexture(snapshot.coverage, snapshot.shape);
    const inverse = inverseAffine(snapshot.affine);
    const tolerance = priorSamplingTolerance(snapshot)!;
    this.materials().forEach((shader) => {
      shader.uniforms.uPrior.value = this.priorTexture;
      shader.uniforms.uPriorCoverage.value = this.priorCoverageTexture;
      shader.uniforms.uPriorActive.value = 1;
      shader.uniforms.uPriorKind.value =
        snapshot.mapKind === "structural_mask" ? 1 : 0;
      shader.uniforms.uWorldToPrior.value.copy(matrix(inverse));
      shader.uniforms.uPriorShape.value.set(...snapshot.shape);
      shader.uniforms.uPriorVoxelTolerance.value.set(...tolerance);
    });
    this.requestRender();
    return snapshot;
  }

  private updateSourcePlane(): void {
    const [a, b, c] = PLANE_AXES[this.activePlane],
      values = this.sourcePlane.geometry.attributes.position
        .array as Float32Array;
    [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 1],
    ].forEach(([u, v], index) => {
      const point = [...this.cursor];
      point[a] = this.bounds[u][a];
      point[b] = this.bounds[v][b];
      point[c] = this.cursor[c];
      values.set(point, index * 3);
    });
    this.sourcePlane.geometry.attributes.position.needsUpdate = true;
    this.sourcePlane.geometry.computeBoundingSphere();
  }

  private sourceToRas(point: number[]): THREE.Vector3 {
    const sign = this.volume.frame.startsWith("LPS") ? -1 : 1;
    return new THREE.Vector3(point[0] * sign, point[1] * sign, point[2]);
  }

  private updateRoutes(routes: ViewerRoute[]): void {
    disposeObject(this.tools);
    this.tools.clear();
    const shown = routes.slice(0, 2);
    const capsules: InstrumentCapsuleDisplay[] = [];
    shown.forEach((route, index) => {
      const entry = this.sourceToRas(route.entry_mm),
        tip = this.sourceToRas(route.target_mm),
        axis = tip.clone().sub(entry).normalize();
      const shaftStart = tip
          .clone()
          .addScaledVector(axis, -route.tool.working_length_mm),
        shaftEnd = tip.clone().addScaledVector(axis, -route.tool.tip_length_mm);
      const { color, slot } = routeAppearance(route, index);
      const addPart = (part: THREE.Object3D) => {
        part.userData.comparisonSlot = slot;
        part.userData.routeId = route.route_id;
        this.tools.add(part);
      };
      const capsule = (
        a: THREE.Vector3,
        b: THREE.Vector3,
        radius: number,
        active: boolean,
      ) => {
        const mesh = new THREE.Mesh(
          new THREE.CapsuleGeometry(radius, a.distanceTo(b), 5, 16),
          new THREE.MeshStandardMaterial({
            color,
            metalness: active ? 0.2 : 0.62,
            roughness: active ? 0.3 : 0.23,
            transparent: true,
            opacity: active ? 1 : 0.9,
          }),
        );
        mesh.position.copy(a).add(b).multiplyScalar(0.5);
        mesh.quaternion.setFromUnitVectors(
          new THREE.Vector3(0, 1, 0),
          b.clone().sub(a).normalize(),
        );
        addPart(mesh);
      };
      capsule(shaftStart, shaftEnd, route.tool.shaft_radius_mm, false);
      capsule(shaftEnd, tip, route.tool.tip_radius_mm, true);
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([entry, tip]),
        new THREE.LineDashedMaterial({
          color,
          transparent: true,
          opacity: 0.65,
          dashSize: 2,
          gapSize: 1.4,
          depthTest: false,
        }),
      );
      line.computeLineDistances();
      line.renderOrder = 4;
      addPart(line);
      if (route.window) {
        const ring = new THREE.Mesh(
          new THREE.RingGeometry(
            Math.max(0.1, route.window.radius_mm - 0.24),
            route.window.radius_mm + 0.24,
            48,
          ),
          new THREE.MeshBasicMaterial({
            color,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.95,
          }),
        );
        ring.position.copy(this.sourceToRas(route.window.center_mm));
        ring.quaternion.setFromUnitVectors(
          new THREE.Vector3(0, 0, 1),
          this.sourceToRas(route.window.normal_inward).normalize(),
        );
        addPart(ring);
      }
      const failure = route.geometry?.failures?.[0];
      if (failure) {
        const marker = new THREE.Mesh(
          new THREE.SphereGeometry(1.5, 16, 12),
          new THREE.MeshBasicMaterial({
            color: FAILURE_COLOR,
            depthTest: false,
          }),
        );
        marker.position.copy(this.sourceToRas(failure.position_mm));
        marker.renderOrder = 6;
        marker.userData.failure = failure.reason;
        addPart(marker);
      }
      capsules.push({ shaftStart: shaftStart.toArray(), shaftEnd: shaftEnd.toArray(), tip: tip.toArray(),
        shaftRadius: route.tool.shaft_radius_mm, tipRadius: route.tool.tip_radius_mm, color });
    });
    this.instrumentDisplay.setRoutes(capsules);
    this.syncInstrumentDisplay();
    if (this.mode === "instruments") this.fitCamera("instruments");
  }

  /** Canonical RAS initial pose only; no route conversion, motion or tissue effect. */
  setInspectionTool(input: ViewerInspectionTool | null): InspectionToolDisplay | null {
    disposeObject(this.inspectionTools);
    this.inspectionTools.clear();
    try {
      const display = this.instrumentDisplay.setInspection(this.volume, input);
      if (!display) return null;
      this.inspectionTools.add(inspectionToolMeshes(display));
      return display;
    } catch (cause) {
      this.instrumentDisplay.setInspection(this.volume, null);
      disposeObject(this.inspectionTools);
      this.inspectionTools.clear();
      throw cause;
    } finally {
      this.syncInstrumentDisplay();
      this.requestRender();
    }
  }

  /** Shader slots describe visible capsules, independently of A/B route identity. */
  private syncInstrumentDisplay(): void {
    this.tools.visible = this.instrumentDisplay.routeVisible;
    this.inspectionTools.visible = this.instrumentDisplay.inspected !== null;
    this.recordedTools.visible = this.replayActive && (this.recordedDisplay !== null || this.passiveDisplays.length > 0 || this.rehearsalDisplays.length > 0);
    const shown = this.rehearsalActive ? this.rehearsalDisplays : this.passiveActive ? this.passiveDisplays :
      this.recordedTools.visible ? [this.recordedDisplay!] : this.instrumentDisplay.displayed;
    this.materials().forEach((shader) => {
      shader.uniforms.uRouteCount.value = shown.length;
      shown.forEach((capsule, index) => {
        const set = (name: string, point: readonly number[]) => shader.uniforms[name].value[index].set(point[0], point[1], point[2]);
        set("uShaftStart", capsule.shaftStart);
        set("uShaftEnd", capsule.shaftEnd);
        set("uTipEnd", capsule.tip);
        shader.uniforms.uRadii.value[index].set(capsule.shaftRadius, capsule.tipRadius);
        shader.uniforms.uRouteColors.value[index].set(capsule.color).convertLinearToSRGB();
      });
    });
  }

  fitCamera(mode: "anatomy" | "instruments" = this.mode, preset: 'oblique'|'anterior'|'left'|'superior' = 'oblique'): void {
    const pane = this.panes.anatomy.getBoundingClientRect();
    if (pane.width < 1 || pane.height < 1) {
      this.pendingFit = mode;
      return;
    }
    this.pendingFit = null;
    let box = physicalBounds(this.anatomy);
    if (box.isEmpty())
      box = new THREE.Box3(
        new THREE.Vector3(...this.bounds[0]),
        new THREE.Vector3(...this.bounds[1]),
      );
    else box.expandByScalar(12);
    if(this.scanVolume?.active&&this.scanFramingPoints.length){const support=new THREE.Box3().setFromArray(this.scanFramingPoints);box.union(support.expandByScalar(3));}
    const visibleTools = this.recordedTools.visible ? this.recordedTools : this.instrumentDisplay.inspected ? this.inspectionTools : this.tools;
    if (mode === "instruments" && this.replayActive && this.replayBounds) {
      const [low, high] = this.replayBounds.boundsRasMm;
      box.union(new THREE.Box3(new THREE.Vector3(...low), new THREE.Vector3(...high))).expandByScalar(8);
      this.fittedReplayId = this.replayBounds.episodeId;
    } else if (mode === "instruments" && visibleTools.visible && !new THREE.Box3().setFromObject(visibleTools).isEmpty())
      box.union(new THREE.Box3().setFromObject(visibleTools)).expandByScalar(8);
    const centre = box.getCenter(new THREE.Vector3()),
      size = box.getSize(new THREE.Vector3());
    const aspect = Math.max(0.2, pane.width / Math.max(pane.height, 1));
    const vertical = THREE.MathUtils.degToRad(this.camera.fov),
      horizontal = 2 * Math.atan(Math.tan(vertical / 2) * aspect);
    const directions = {oblique:[.88,-1.65,.8],anterior:[0,1,0],left:[-1,0,0],superior:[0,0,1]};
    const direction = new THREE.Vector3(...directions[preset] as [number,number,number]).normalize();
    this.camera.up.set(...(preset==='superior' ? [0,1,0] : [0,0,1]) as [number,number,number]);
    const right = new THREE.Vector3()
      .crossVectors(this.camera.up, direction)
      .normalize();
    const up = new THREE.Vector3().crossVectors(direction, right).normalize();
    let distance = 40;
    for (const x of [box.min.x, box.max.x])
      for (const y of [box.min.y, box.max.y])
        for (const z of [box.min.z, box.max.z]) {
          const relative = new THREE.Vector3(x, y, z).sub(centre),
            depth = relative.dot(direction);
          distance = Math.max(
            distance,
            depth + Math.abs(relative.dot(right)) / Math.tan(horizontal / 2),
            depth + Math.abs(relative.dot(up)) / Math.tan(vertical / 2),
          );
        }
    distance *= 1.06;
    // Rounded patient anatomy leaves most expanded AABB corners empty. Fit
    // its actual unchanged vertices; recorded instrument/replay envelopes keep
    // the conservative complete-bounds calculation above.
    const surfaceDistance=mode==='anatomy' && !this.replayActive && !this.passiveActive
      ? surfaceCameraDistance(this.anatomy,centre,direction,right,up,vertical,aspect) : null;
    if(surfaceDistance!==null)distance=surfaceDistance;
    if(this.scanVolume?.active&&this.scanFramingPoints.length){
      let scanDistance=40;const p=new THREE.Vector3();
      for(let i=0;i<this.scanFramingPoints.length;i+=3){p.fromArray(this.scanFramingPoints,i).sub(centre);const depth=p.dot(direction);scanDistance=Math.max(scanDistance,depth+(Math.abs(p.dot(right))+3)/Math.tan(horizontal/2),depth+(Math.abs(p.dot(up))+3)/Math.tan(vertical/2));}
      distance=Math.max(distance,scanDistance*1.08);
    }
    this.camera.position.copy(centre).addScaledVector(direction, distance);
    this.panes.anatomy.dataset.cameraFit = JSON.stringify({
      mode,
      centre: centre.toArray(),
      size: size.toArray(),
      distanceMm: distance,
      fitGeometry: surfaceDistance===null ? 'complete-bounds' : 'source-surface-vertices',
      surfaces: this.surfaces.size,
    });
    this.controls.target.copy(centre);
    this.camera.near = Math.max(0.1, distance / 1000);
    this.camera.far = Math.max(1000, distance * 8);
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.requestRender();
  }

  /** A click on displayed source MRI/surfaces updates the same RAS cursor. */
  pick(clientX: number, clientY: number): Point3 | null {
    const started=performance.now();
    const rect = this.panes.anatomy.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      (-(clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.pickRay.setFromCamera(ndc, this.camera);
    // Cursor picking remains on the original isosurface in either display mode.
    // Raycasting is synchronous; restore display geometry before any rendering.
    const displayed = new Map<THREE.Mesh, THREE.BufferGeometry>();
    this.surfaces.forEach((mesh, name) => {
      const pair = this.surfaceGeometry.get(name);
      if (pair) {
        displayed.set(mesh, mesh.geometry);
        mesh.geometry = pair.source;
      }
    });
    for(const object of this.replayGroup.children){
      const mesh=object as THREE.Mesh,pair=mesh.userData.replayGeometry as {source:THREE.BufferGeometry}|undefined;
      if(pair){displayed.set(mesh,mesh.geometry);mesh.geometry=pair.source;}
    }
    let hits: THREE.Intersection[];
    try { hits = this.pickRay.intersectObjects(
      [
        ...(this.sourcePlane.visible ? [this.sourcePlane] : []),
        ...(this.anatomy.visible
          ? Array.from(this.surfaces.values()).filter((mesh) => mesh.visible && (mesh.material as THREE.MeshStandardMaterial).opacity > 0)
          : []),
        ...this.replayGroup.children.filter((mesh) => mesh.visible &&
          (!(mesh instanceof THREE.Mesh)||Array.isArray(mesh.material)||mesh.material.opacity>0)),
      ],
      false,
    ); } finally { displayed.forEach((geometry, mesh) => { mesh.geometry = geometry; }); }
    const planes=[...(this.extraClipPlanes??[]),...(this.clipPlane?[this.clipPlane]:[])];
    const hit=hits.find(item=>{
      const visiblePlanes=item.object===this.sourcePlane?(this.clipPlane?[this.clipPlane]:[]):planes;
      if(!visiblePlanes.every(plane=>plane.distanceToPoint(item.point)>=0))return false;
      if(item.object!==this.sourcePlane)return true;
      // The source-plane shader discards empty/out-of-field pixels. A quad
      // intersection there must allow the next visible anatomy hit through.
      const point=item.point.clone().applyMatrix4(this.sourcePlane.material.uniforms.uWorldToVoxel.value).toArray() as Point3;
      if(point.some((v,a)=>v<-.0001||v>this.volume.shape[a]-.9999))return false;
      const voxel=point.map((v,a)=>Math.max(0,Math.min(this.volume.shape[a]-1,v))) as Point3;
      const signal=sampleTrilinear(this.volume.mri,this.volume.shape,voxel);
      return Number.isFinite(signal)&&Math.abs(signal)>=.000001;
    });
    this.canvas?.dispatchEvent(new CustomEvent('anatomy-pick',{bubbles:true,detail:{milliseconds:performance.now()-started,
      measurement:'CPU-source-surface-raycast-and-clip',intersections:hits.length,selected:!!hit}}));
    return hit
      ? [hit.point.x, hit.point.y, hit.point.z]
      : null;
  }

  private estimateAllocation(){
    const geometries=new Set<THREE.BufferGeometry>();
    const gather=(root:THREE.Object3D)=>root.traverse(object=>{const geometry=(object as THREE.Mesh).geometry;if(geometry)geometries.add(geometry);});
    gather(this.scene);this.slices.forEach(slice=>gather(slice.scene));
    this.surfaceGeometry.forEach(pair=>{geometries.add(pair.source);geometries.add(pair.smooth);});
    if(this.pendingReplayGroup)gather(this.pendingReplayGroup);
    for(const entry of this.retiredSurfaces){geometries.add(entry.pair.source);geometries.add(entry.pair.smooth);}
    if(this.modelStage)for(const pair of this.modelStage.prepared.values())if(pair){geometries.add(pair.source);geometries.add(pair.smooth);}
    const textures=[this.mriTexture,this.labelTexture,this.fusionTexture,this.removedTexture,this.proposalTexture,
      this.diagnosticStateTexture,this.diagnosticCoverageTexture,this.priorTexture,this.priorCoverageTexture];
    const geometryBytes=geometryBufferBytes(geometries),textureBytes=[...new Set(textures)].reduce((sum,texture)=>{
      const data=texture.image?.data as ArrayBufferView|undefined;return sum+(data?.byteLength??0);
    },0);
    const targetBytes=this.scanVolume.estimatedTargetBytes+this.surfaceInteraction.estimatedTargetBytes+renderTargetBytes(this.environmentTarget)+[...this.sliceCaches.values()].reduce((sum,cache)=>sum+cache.estimatedTargetBytes,0);
    return {geometryBytes,textureBytes,targetBytes,totalKnownBytes:geometryBytes+textureBytes+targetBytes,
      meaning:'Uncompressed buffers and known render attachments; excludes driver, programs, default-framebuffer and allocation overhead'};
  }

  private requestRender(slicesChanged=true): void {
    this.scanVolume?.invalidate?.();
    if(slicesChanged)this.sliceRevision++;
    if(this.disposed||this.graphicsLost)return;
    if(this.rendering)return;
    if(this.gpuSync){this.needsFrame=true;return;}
    if(this.frame)return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      this.controls.update();
      this.render();
    });
  }

  private render(): void {
    if (this.disposed||this.graphicsLost) return;
    const frameStarted=performance.now();
    if(!this.paneBounds)this.paneBounds={root:this.container.getBoundingClientRect(),panes:new Map(Object.values(this.panes).map(pane=>[pane,pane.getBoundingClientRect()]))};
    const root = this.paneBounds.root,
      width = Math.round(root.width),
      height = Math.round(root.height);
    if (width < 1 || height < 1) return;
    this.renderer.info.reset();
    const serial=++this.renderSerial;
    this.gpuTimer?.begin(serial);
    this.rendering=true;
    if(this.modelStage?.ready)this.commitModelStage(this.modelStage);
    this.labelTextureUpdates?.sync(this.renderer,this.labelTexture);
    // Interaction quality changes only offscreen 3D targets. The shared canvas
    // and linked slices retain their display resolution throughout a gesture.
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    if (pixelRatio !== this.renderer.getPixelRatio()) this.renderer.setPixelRatio(pixelRatio);
    if (width !== this.width || height !== this.height) {
      this.width = width;
      this.height = height;
      this.renderer.setSize(width, height, false);
    }
    this.renderer.setScissorTest(false);
    this.renderer.clear();
    this.renderer.setScissorTest(true);
    const viewport = (element: HTMLElement) => {
      const rect = paneViewport(root, this.paneBounds!.panes.get(element)!);
      if (!rect) return null;
      this.renderer.setViewport(
        rect.left,
        rect.bottom,
        rect.width,
        rect.height,
      );
      this.renderer.setScissor(rect.left, rect.bottom, rect.width, rect.height);
      return { w: rect.width, h: rect.height };
    };
    const main = viewport(this.panes.anatomy);
    let surfaceQuality:{adaptive:boolean;width:number;height:number}|null=null;
    if (main) {
      if (this.pendingFit) this.fitCamera(this.pendingFit);
      this.camera.aspect = main.w / main.h;
      this.camera.updateProjectionMatrix();
      if (this.replayActive || this.passiveActive||!this.scanVolume.active)
        surfaceQuality=this.surfaceInteraction.render(this.renderer,this.scene,this.camera,main.w,main.h,this.interacting,this.budget.longEdge,()=>{viewport(this.panes.anatomy);});
      else this.scanVolume.render(this.scene,this.camera,main.w,main.h,this.interacting,()=>{viewport(this.panes.anatomy);},this.budget.longEdge);
    }
    let sliceDraws=0;
    for (const plane of SLICE_PLANES) {
      const size = viewport(this.panes[plane]);
      if (!size) continue;
      const slice = this.slices.get(plane)!,
        view = mprView(this.bounds,this.sliceSupportBounds,plane,size.w,size.h,this.workspaceSettings.sliceFraming),
        rect = view.rect;
      slice.material.uniforms.uLow.value.set(...view.bounds[0]);
      slice.material.uniforms.uHigh.value.set(...view.bounds[1]);
      slice.material.uniforms.uRect.value.set(
        rect.left / size.w,
        1 - (rect.top + rect.height) / size.h,
        rect.width / size.w,
        rect.height / size.h,
      );
      slice.material.uniforms.uResolution.value.set(size.w, size.h);
      let cached=this.sliceCaches.get(plane);
      if(!cached){cached=new SliceFrameCache();this.sliceCaches.set(plane,cached);}
      if(cached.render(this.renderer,slice.scene,slice.camera,size.w,size.h,this.sliceRevision,()=>{viewport(this.panes[plane]);}))sliceDraws++;
    }
    this.renderer.setScissorTest(false);
    this.gpuTimer?.end();
    this.rendering=false;
    const interactive=this.interacting,cpuMs=performance.now()-frameStarted;
    this.canvas.dispatchEvent(new CustomEvent('anatomy-frame',{bubbles:true,detail:{serial,at:frameStarted,cpuMs,cpuMeasurement:'CPU-frame-submission',interactive,volume:this.scanVolume.active,sliceDraws,longEdge:interactive?this.budget.longEdge:1300,
      resources:{...this.renderer.info.memory},drawCalls:this.renderer.info.render.calls,drawStatistics:{...this.renderer.info.render},gpuTimingSupported:this.gpuTimer?.supported??false,
      estimatedAllocation:this.estimateAllocation(),displayWork:{...this.displayWork},workers:{source:this.sourceWorkerActive?1:0,replay:this.replayWorker?1:0,modeled:this.modelWorker?1:0,picking:this.picking?.stats().workerLive??0},surfaceQuality,
      surfaceCache:{budgetBytes:RETIRED_SURFACE_BUDGET_BYTES,retiredBytes:this.retiredSurfaces.reduce((sum,entry)=>sum+entry.bytes,0),retiredEntries:this.retiredSurfaces.length},
      labelUploads:this.labelTextureUpdates?.work??null,
      sourceCaseHash:this.volume.caseHash,surgicalStateId:(this.surgicalInteraction?.preview??this.surgicalInteraction?.view)?.stateId??null,picking:this.picking?.stats()??null}}));
    this.finishSurgicalFrame();
    const gl=this.renderer.getContext() as WebGL2RenderingContext;
    const sync=gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE,0);
    if(!sync)return;
    this.gpuSync=sync;gl.flush();
    const refinement=!interactive?this.refinementStarted:0;
    const poll=()=>{
      this.gpuPoll=0;
      if(this.disposed)return;
      const status=gl.clientWaitSync(sync,0,0);
      if(status===gl.TIMEOUT_EXPIRED){this.gpuPoll=requestAnimationFrame(poll);return;}
      gl.deleteSync(sync);this.gpuSync=null;
      const elapsed=performance.now()-frameStarted;
      if(interactive&&status!==gl.WAIT_FAILED)this.budget.sample(elapsed);
      this.canvas.dispatchEvent(new CustomEvent('anatomy-gpu',{bubbles:true,detail:{serial,completed:status!==gl.WAIT_FAILED,milliseconds:elapsed,completionPollMs:elapsed,measurement:'submission-to-fence-poll-upper-bound'}}));
      if(!interactive&&!this.replayActive&&!this.passiveActive&&this.scanVolume.needsRefinement)this.needsFrame=true;
      if(refinement&&this.refinementStarted===refinement&&!this.interacting&&!this.needsFrame){
        this.refinementStarted=0;
        this.canvas.dispatchEvent(new CustomEvent('anatomy-refined',{bubbles:true,detail:{completed:status!==gl.WAIT_FAILED,milliseconds:performance.now()-refinement,volume:this.scanVolume.active}}));
      }
      if(this.needsFrame){this.needsFrame=false;this.controls.update();this.render();}
    };
    this.gpuPoll=requestAnimationFrame(poll);
  }

  dispose(): void {
    this.picking?.dispose();
    if (this.disposed) return;
    this.disposed = true;
    this.pendingSurgicalPublication=null;
    if(this.surgicalGizmo){this.scene.remove(this.surgicalGizmo.getHelper());this.surgicalGizmo.dispose();}
    this.gpuTimer?.dispose();
    cancelAnimationFrame(this.frame);
    cancelAnimationFrame(this.gpuPoll);
    if(this.gpuSync)(this.renderer.getContext() as WebGL2RenderingContext).deleteSync(this.gpuSync);
    this.gpuSync=null;
    this.interaction.dispose();
    this.sliceCaches.forEach(cache=>cache.dispose());this.sliceCaches.clear();
    window.removeEventListener('scroll',this.onScroll,true);
    window.removeEventListener('blur',this.onBlur);
    this.panes.anatomy.removeEventListener('pointercancel',this.onControlsEnd);
    this.observer.disconnect();
    this.worker.terminate();
    this.modelWorker?.terminate();this.modelWorker=null;this.clearModelStage();
    for(const entry of this.retiredSurfaces)this.disposeSurfacePair(entry.pair);this.retiredSurfaces.length=0;
    this.replayWorker?.terminate();
    if (this.pendingReplayGroup) disposeObject(this.pendingReplayGroup);
    this.controls.removeEventListener("change", this.onControlsChange);
    this.controls.removeEventListener("start", this.onControlsStart);
    this.controls.removeEventListener("end", this.onControlsEnd);
    this.controls.dispose();
    this.canvas.removeEventListener("webglcontextlost", this.onContextLost);
    this.canvas.removeEventListener('webglcontextrestored',this.onContextRestored);
    this.surfaceGeometry.forEach((pair, name) => {
      const displayed = this.surfaces.get(name)?.geometry;
      if (pair.source !== displayed) pair.source.dispose();
      if (pair.smooth !== displayed) pair.smooth.dispose();
    });
    this.surfaceGeometry.clear();
    disposeObject(this.scene);
    this.slices.forEach((slice) => disposeObject(slice.scene));
    this.mriTexture.dispose();
    this.scanVolume.dispose();
    this.surfaceInteraction.dispose();
    this.environmentTarget.dispose();
    this.fusionTexture.dispose();
    this.labelTexture.dispose();
    this.removedTexture.dispose();
    this.proposalTexture.dispose();
    this.diagnosticStateTexture.dispose();
    this.diagnosticCoverageTexture.dispose();
    this.priorTexture.dispose();
    this.priorCoverageTexture.dispose();
    this.renderer.dispose();
  }
}
