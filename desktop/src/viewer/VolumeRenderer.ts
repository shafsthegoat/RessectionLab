import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  PLANE_AXES,
  inverseAffine,
  rasAffine,
  sliceRect,
  transformPoint,
  volumeBounds,
} from "./coordinates";
import type { Affine, Bounds3, Point3, SlicePlane } from "./coordinates";
import type {
  ViewerReplay,
  ViewerPriorLayer,
  ViewerRoute,
  ViewerStructuralProposal,
  ViewerVolume,
  ViewerInspectionTool,
} from "./contracts";
import { InstrumentDisplayState } from "./inspectionTool";
import type { InspectionToolDisplay, InstrumentCapsuleDisplay } from "./inspectionTool";
import { recordedToolDisplay } from "./recordedTool";
import type { RecordedToolDisplay } from "./recordedTool";
import { instrumentCapsuleMeshes, inspectionToolMeshes } from "./inspectionToolGeometry";
import { fragmentShader, vertexShader } from "./shaders";
import { physicalBounds, placeInSourceFrame } from "./sceneGeometry";
import { paneViewport } from "./layout";
import {
  PRIOR_COLORS,
  priorSamplingTolerance,
  validatePriorLayer,
} from "./priorLayer";
import { residualMask, validateReplay } from "./replay";
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
const SLICE_PLANES: SlicePlane[] = ["axial", "coronal", "sagittal"];

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
    if (child instanceof THREE.Mesh || child instanceof THREE.Line) {
      child.geometry.dispose();
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
  readonly defaultWindow: [number, number];
  readonly affine: Affine;
  private readonly renderer: THREE.WebGLRenderer;
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
  private readonly instrumentDisplay = new InstrumentDisplayState();
  private readonly replayGroup = new THREE.Group();
  private replayWorker: Worker | null = null;
  private replayGeneration = 0;
  private pendingReplayGroup: THREE.Group | null = null;
  private replayActive = false;
  private removedTexture: THREE.Data3DTexture;
  private proposalTexture: THREE.Data3DTexture;
  private priorTexture: THREE.Data3DTexture;
  private priorCoverageTexture: THREE.Data3DTexture;
  private readonly surfaces = new Map<string, THREE.Mesh>();
  private readonly mriTexture: THREE.Data3DTexture;
  private readonly labelTexture: THREE.Data3DTexture;
  private readonly sourcePlane: THREE.Mesh<
    THREE.BufferGeometry,
    THREE.ShaderMaterial
  >;
  private readonly cursorObject: THREE.LineSegments;
  private readonly observer: ResizeObserver;
  private readonly worker: Worker;
  private disposed = false;
  private frame = 0;
  private width = 0;
  private height = 0;
  private cursor: Point3;
  private visible: Record<string, boolean> = {};
  private opacity = 0.4;
  private activePlane: SlicePlane = "axial";
  private routeSignature = "";
  private mode: "anatomy" | "instruments" = "anatomy";
  private pendingFit: "anatomy" | "instruments" | null = null;
  private readonly pickRay = new THREE.Raycaster();
  private readonly onControlsChange = () => this.requestRender();
  private readonly onContextLost = (event: Event) => {
    event.preventDefault();
    this.onError(
      "The graphics context was lost. Reopen this case to restore the source-image views.",
    );
  };

  constructor(
    private readonly container: HTMLElement,
    private readonly canvas: HTMLCanvasElement,
    private readonly panes: Panes,
    private readonly volume: ViewerVolume,
    private readonly onError: (message: string) => void,
    private readonly onSurfaceStatus: (remaining: number) => void,
    private readonly onReplayError: (message: string) => void,
  ) {
    const count = volume.shape.reduce((a, b) => a * b, 1);
    if (
      volume.shape.some((x) => !Number.isInteger(x) || x < 2) ||
      volume.mri.length !== count
    )
      throw new Error("MRI array dimensions do not match the source grid.");
    if (volume.compartments.length > 8)
      throw new Error(
        "This viewer supports up to eight independently visible source compartments.",
      );
    if (volume.compartments.some((layer) => layer.mask.length !== count))
      throw new Error("A source compartment does not match the MRI grid.");
    this.affine = rasAffine(volume.affine, volume.frame);
    const inverse = inverseAffine(this.affine);
    this.bounds = volumeBounds(volume.shape, this.affine);
    this.cursor = this.bounds[0].map(
      (x, i) => (x + this.bounds[1][i]) / 2,
    ) as Point3;
    this.defaultWindow = intensityWindow(volume.mri);
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });
    const gl = this.renderer.getContext() as WebGL2RenderingContext;
    const maximum = gl.getParameter(gl.MAX_3D_TEXTURE_SIZE) as number;
    if (volume.shape.some((x) => x > maximum)) {
      this.renderer.dispose();
      throw new Error(
        `The source grid exceeds this GPU's ${maximum}-voxel 3-D texture limit.`,
      );
    }
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(0x05080d, 1);
    this.renderer.autoClear = false;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.debug.onShaderError = () =>
      onError(
        "The GPU could not compile the source-image renderer. No image or route geometry has been substituted.",
      );
    this.canvas.addEventListener("webglcontextlost", this.onContextLost);
    this.mriTexture = dataTexture(volume.mri, volume.shape);
    const packed = new Uint8Array(count);
    volume.compartments.forEach((layer, index) => {
      const bit = 1 << index;
      for (let i = 0; i < count; i++) if (layer.mask[i]) packed[i] |= bit;
      this.visible[layer.name] = true;
    });
    this.labelTexture = dataTexture(packed, volume.shape);
    this.removedTexture = dataTexture(new Uint8Array(1), [1, 1, 1]);
    this.proposalTexture = dataTexture(new Uint8Array(1), [1, 1, 1]);
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
          uLabels: { value: this.labelTexture },
          uRemoved: { value: this.removedTexture },
          uReplayActive: { value: 0 },
          uProposal: { value: this.proposalTexture },
          uProposalActive: { value: 0 },
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
          uThreeD: { value: threeD ? 1 : 0 },
          uVisibleBits: { value: (1 << volume.compartments.length) - 1 },
          uColors: {
            value: Array.from({ length: 8 }, (_, i) =>
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
    this.scene.add(new THREE.HemisphereLight(0xb7d6df, 0x142738, 2.1));
    const centre = new THREE.Vector3(...this.cursor);
    const key = new THREE.DirectionalLight(0xe8f3f3, 3.1);
    key.position.copy(centre).add(new THREE.Vector3(170, -220, 240));
    key.target.position.copy(centre);
    const rim = new THREE.DirectionalLight(0x77bdb4, 1.6);
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
    this.scene.add(this.cursorObject);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, panes.anatomy);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.rotateSpeed = 0.65;
    this.controls.zoomSpeed = 0.8;
    this.controls.minDistance = 5;
    this.controls.maxDistance = 2200;
    this.controls.addEventListener("change", this.onControlsChange);
    this.observer = new ResizeObserver(() => this.requestRender());
    this.observer.observe(container);
    Object.values(panes).forEach((pane) => this.observer.observe(pane));
    this.worker = new Worker(new URL("./surface.worker.ts", import.meta.url), {
      type: "module",
    });
    let remaining = volume.compartments.length;
    onSurfaceStatus(remaining);
    this.worker.onmessage = (
      event: MessageEvent<{
        name: string;
        positions?: Float32Array;
        normals?: Float32Array;
        error?: string;
      }>,
    ) => {
      if (this.disposed) return;
      remaining--;
      if (!this.replayActive) onSurfaceStatus(remaining);
      if (event.data.error) {
        onError(`Source surface unavailable: ${event.data.error}`);
        return;
      }
      const layer = volume.compartments.find(
          (item) => item.name === event.data.name,
        ),
        positions = event.data.positions,
        normals = event.data.normals;
      if (!layer || !positions?.length) return;
      if (!normals || normals.length !== positions.length) {
        onError("Source surface normals do not match its unchanged geometry.");
        return;
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.BufferAttribute(positions, 3),
      );
      geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
      const mesh = new THREE.Mesh(
        geometry,
        new THREE.MeshStandardMaterial({
          color: layer.color,
          roughness: 0.34,
          metalness: 0.08,
          transparent: true,
          opacity: 0.66,
          side: THREE.DoubleSide,
          depthWrite: false,
        }),
      );
      placeInSourceFrame(mesh, this.affine);
      mesh.visible = this.visible[layer.name] !== false;
      mesh.userData.compartment = layer.name;
      this.surfaces.set(layer.name, mesh);
      this.anatomy.add(mesh);
      if (remaining === 0 && this.mode === "anatomy") this.fitCamera("anatomy");
      this.requestRender();
    };
    this.worker.onerror = () =>
      onError(
        "A source surface could not be prepared. The original MRI and labels remain available in the linked slices.",
      );
    volume.compartments.forEach((layer) =>
      this.worker.postMessage({
        name: layer.name,
        mask: layer.mask,
        shape: volume.shape,
      }),
    );
    this.updateSourcePlane();
    this.fitCamera("anatomy");
    this.requestRender();
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
    this.cursor = [...cursor];
    this.visible = visible;
    this.opacity = Math.max(0, Math.min(1, opacity));
    let bits = 0;
    this.volume.compartments.forEach((layer, index) => {
      if (visible[layer.name] !== false) bits |= 1 << index;
      const mesh = this.surfaces.get(layer.name);
      if (mesh) mesh.visible = visible[layer.name] !== false;
    });
    this.replayGroup.children.forEach((mesh) => {
      mesh.visible =
        mesh.userData.compartment === undefined ||
        visible[mesh.userData.compartment] !== false;
    });
    this.materials().forEach((shader) => {
      shader.uniforms.uCursor.value.set(...cursor);
      shader.uniforms.uVisibleBits.value = bits;
      shader.uniforms.uOverlay.value = this.opacity;
    });
    this.cursorObject.position.set(...cursor);
    this.updateSourcePlane();
    const signature = JSON.stringify(routes);
    if (signature !== this.routeSignature) {
      this.routeSignature = signature;
      this.updateRoutes(routes);
    }
    if (cameraMode !== this.mode) {
      this.mode = cameraMode;
      this.fitCamera(cameraMode);
    }
    this.requestRender();
  }

  setWindow(low: number, high: number): void {
    if (!Number.isFinite(low) || !Number.isFinite(high) || high <= low) return;
    this.materials().forEach((shader) =>
      shader.uniforms.uWindow.value.set(low, high),
    );
    this.requestRender();
  }

  setSourcePlane(plane: SlicePlane | null): void {
    this.sourcePlane.visible = plane !== null;
    if (plane !== null) this.activePlane = plane;
    this.updateSourcePlane();
    this.requestRender();
  }

  /** A certified effect overlays the original MRI; source labels remain immutable. */
  setReplay(replay: ViewerReplay | null): void {
    // A requested mode switch clears an unexecuted tool even if replay rejects.
    if (replay) this.setInspectionTool(null);
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
    this.setStructuralProposal(null);
    this.setPriorLayer(null);

    // Snapshot the accepted effect so later UI updates cannot mutate this replay.
    const removed = replay.removedMask.slice();
    const texture = dataTexture(removed, this.volume.shape);
    this.removedTexture.dispose();
    this.removedTexture = texture;
    this.materials().forEach((shader) => {
      shader.uniforms.uRemoved.value = texture;
      shader.uniforms.uReplayActive.value = 1;
      shader.uniforms.uRouteCount.value = 0;
    });
    this.replayActive = true;
    this.instrumentDisplay.setReplay(true);
    this.recordedDisplay = recorded;
    if (recorded) this.recordedTools.add(instrumentCapsuleMeshes(recorded, "executed-generated-episode", recorded.identity));
    disposeObject(this.inspectionTools);
    this.inspectionTools.clear();
    this.syncInstrumentDisplay();
    this.anatomy.visible = false;
    // Route candidates show terminal approach poses, not certified replay motions.
    this.tools.visible = false;
    const pending = new THREE.Group();
    this.pendingReplayGroup = pending;
    const layers = this.volume.compartments.map((layer, index) => ({
      id: `residual-${index}`,
      name: layer.name,
      color: layer.color,
      removed: false,
      mask: residualMask(layer.mask, removed),
    }));
    layers.push({
      id: "modeled-removal",
      name: "Modeled removal",
      color: "#b7f0d8",
      removed: true,
      mask: removed,
    });
    const waiting = new Map(layers.map((layer) => [layer.id, layer]));
    this.onSurfaceStatus(waiting.size);
    const worker = new Worker(new URL("./surface.worker.ts", import.meta.url), {
      type: "module",
    });
    this.replayWorker = worker;
    const failed = (message: string) => {
      if (this.disposed || generation !== this.replayGeneration) return;
      this.setReplay(null);
      this.onSurfaceStatus(0);
      this.onReplayError(
        `Modeled anatomy could not be displayed: ${message}. The original source anatomy has been restored.`,
      );
    };
    worker.onerror = () => failed("surface preparation failed");
    worker.onmessage = (
      event: MessageEvent<{
        name: string;
        positions?: Float32Array;
        normals?: Float32Array;
        error?: string;
      }>,
    ) => {
      if (this.disposed || generation !== this.replayGeneration) return;
      if (event.data.error) {
        failed(event.data.error);
        return;
      }
      const layer = waiting.get(event.data.name);
      if (!layer) {
        failed("unexpected surface response");
        return;
      }
      waiting.delete(event.data.name);
      const positions = event.data.positions;
      if (!positions) {
        failed("missing source-grid surface");
        return;
      }
      const normals = event.data.normals;
      if (!normals || normals.length !== positions.length) {
        failed("surface normals do not match source-grid geometry");
        return;
      }
      if (positions.length) {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute(
          "position",
          new THREE.BufferAttribute(positions, 3),
        );
        geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
        const mesh = new THREE.Mesh(
          geometry,
          new THREE.MeshStandardMaterial({
            color: layer.color,
            roughness: 0.34,
            metalness: 0.08,
            transparent: true,
            opacity: layer.removed ? 0.78 : 0.66,
            side: THREE.DoubleSide,
            depthWrite: false,
            wireframe: layer.removed,
          }),
        );
        placeInSourceFrame(mesh, this.affine);
        if (!layer.removed) mesh.userData.compartment = layer.name;
        mesh.visible = layer.removed || this.visible[layer.name] !== false;
        pending.add(mesh);
      }
      this.onSurfaceStatus(waiting.size);
      if (!waiting.size) {
        for (const child of [...pending.children]) this.replayGroup.add(child);
        this.pendingReplayGroup = null;
        worker.terminate();
        this.replayWorker = null;
        this.requestRender();
      }
    };
    for (const layer of layers) {
      worker.postMessage({
        name: layer.id,
        mask: layer.mask,
        shape: this.volume.shape,
      });
    }
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
    this.recordedTools.visible = this.replayActive && this.recordedDisplay !== null;
    const shown = this.recordedTools.visible ? [this.recordedDisplay!] : this.instrumentDisplay.displayed;
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

  fitCamera(mode: "anatomy" | "instruments" = this.mode): void {
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
    const visibleTools = this.recordedTools.visible ? this.recordedTools : this.instrumentDisplay.inspected ? this.inspectionTools : this.tools;
    if (mode === "instruments" && visibleTools.visible && !new THREE.Box3().setFromObject(visibleTools).isEmpty())
      box.union(new THREE.Box3().setFromObject(visibleTools)).expandByScalar(8);
    const centre = box.getCenter(new THREE.Vector3()),
      size = box.getSize(new THREE.Vector3());
    const aspect = Math.max(0.2, pane.width / Math.max(pane.height, 1));
    const vertical = THREE.MathUtils.degToRad(this.camera.fov),
      horizontal = 2 * Math.atan(Math.tan(vertical / 2) * aspect);
    const direction = new THREE.Vector3(0.88, -1.65, 0.8).normalize();
    const right = new THREE.Vector3()
      .crossVectors(new THREE.Vector3(0, 0, 1), direction)
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
    this.camera.position.copy(centre).addScaledVector(direction, distance);
    this.panes.anatomy.dataset.cameraFit = JSON.stringify({
      mode,
      centre: centre.toArray(),
      size: size.toArray(),
      distanceMm: distance,
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
    const rect = this.panes.anatomy.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      (-(clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.pickRay.setFromCamera(ndc, this.camera);
    const hits = this.pickRay.intersectObjects(
      [
        ...(this.sourcePlane.visible ? [this.sourcePlane] : []),
        ...(this.anatomy.visible
          ? Array.from(this.surfaces.values()).filter((mesh) => mesh.visible)
          : []),
        ...this.replayGroup.children.filter((mesh) => mesh.visible),
      ],
      false,
    );
    return hits.length
      ? [hits[0].point.x, hits[0].point.y, hits[0].point.z]
      : null;
  }

  private requestRender(): void {
    if (this.disposed || this.frame) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      this.controls.update();
      this.render();
    });
  }

  private render(): void {
    if (this.disposed) return;
    const root = this.container.getBoundingClientRect(),
      width = Math.round(root.width),
      height = Math.round(root.height);
    if (width < 1 || height < 1) return;
    if (width !== this.width || height !== this.height) {
      this.width = width;
      this.height = height;
      this.renderer.setSize(width, height, false);
    }
    this.renderer.setScissorTest(false);
    this.renderer.clear();
    this.renderer.setScissorTest(true);
    const viewport = (element: HTMLElement) => {
      const rect = paneViewport(root, element.getBoundingClientRect());
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
    if (main) {
      if (this.pendingFit) this.fitCamera(this.pendingFit);
      this.camera.aspect = main.w / main.h;
      this.camera.updateProjectionMatrix();
      this.renderer.render(this.scene, this.camera);
    }
    for (const plane of SLICE_PLANES) {
      const size = viewport(this.panes[plane]);
      if (!size) continue;
      const slice = this.slices.get(plane)!,
        rect = sliceRect(this.bounds, plane, size.w, size.h);
      slice.material.uniforms.uRect.value.set(
        rect.left / size.w,
        1 - (rect.top + rect.height) / size.h,
        rect.width / size.w,
        rect.height / size.h,
      );
      slice.material.uniforms.uResolution.value.set(size.w, size.h);
      this.renderer.render(slice.scene, slice.camera);
    }
    this.renderer.setScissorTest(false);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    cancelAnimationFrame(this.frame);
    this.observer.disconnect();
    this.worker.terminate();
    this.replayWorker?.terminate();
    if (this.pendingReplayGroup) disposeObject(this.pendingReplayGroup);
    this.controls.removeEventListener("change", this.onControlsChange);
    this.controls.dispose();
    this.canvas.removeEventListener("webglcontextlost", this.onContextLost);
    disposeObject(this.scene);
    this.slices.forEach((slice) => disposeObject(slice.scene));
    this.mriTexture.dispose();
    this.labelTexture.dispose();
    this.removedTexture.dispose();
    this.proposalTexture.dispose();
    this.priorTexture.dispose();
    this.priorCoverageTexture.dispose();
    this.renderer.dispose();
  }
}
