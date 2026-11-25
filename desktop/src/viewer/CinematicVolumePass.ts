import * as THREE from 'three';
import type {Point3} from './coordinates';
import {ctaCutKey} from './ctaVesselLight';
import {cinematicDenoiseShader, cinematicFragmentShader, cinematicSurfaceShader} from './cinematicVolumeShaders';
import {volumeVertexShader} from './volumeRaycastShaders';
import {renderTargetBytes} from './renderResourceEstimate';
import {cinematicFieldSteps, cinematicMajorantSteps, cinematicMaterial, cinematicSignalSteps, cinematicTagSteps,
  type CinematicField, type CinematicMaterialInput, type CinematicSignalStats, type CinematicTagLayer, type CinematicTags} from './cinematicTags.ts';

/** Samples after which the running mean is no longer refined. */
export const CINEMATIC_SAMPLE_CAP = 1024;
/** Sample count at which the image shown on a 2x panel is within 40 dB, and 1% of pixels beyond 8 levels, of the capped mean on the three fixed CTA views. 64 samples miss on the cut skull base. */
export const CINEMATIC_SETTLED_SAMPLES = 72;
/** Hidden captures may name a fixed still sample count on the page. It is read
 * when an accumulation starts, so a count named later applies after the next restart. */
export function cinematicSampleLimit(value: unknown = (globalThis as {resectionCinematicStillSamples?: unknown}).resectionCinematicStillSamples) {
  return typeof value === 'number' && Number.isInteger(value) && value >= 1 && value <= CINEMATIC_SAMPLE_CAP ? value : CINEMATIC_SAMPLE_CAP;
}
/** Sample counts between which the variance-guided filter is blended away at rest. */
export const CINEMATIC_FILTER_SAMPLES: [number, number] = [32, 128];
/** Weight of the unfiltered mean in the displayed image after a number of samples. */
export function cinematicFilterBlend(samples: number) {
  const [start, end] = CINEMATIC_FILTER_SAMPLES;
  return Math.max(0, Math.min(1, (samples - start) / (end - start)));
}
/** Least faint-matter step rate (events per mm) where there is any: glass is
 * stepped by its majorant, MR context fog is broad and smooth. */
export const CINEMATIC_FAINT_FLOOR = {glass: .1, fog: .2};
/** Peel layers of translucent meshes that can be composited over the traced image. */
export const CINEMATIC_SURFACE_LAYERS = 5;
const MOTION_FILTER_STEPS = [1, 2, 4];
const REST_FILTER_STEPS = [1, 2];

type Job = {key: string; steps: Generator<void, void>; timer: ReturnType<typeof setTimeout> | null};
/** floatLinear: whether float textures may be linearly filtered (OES_texture_float_linear). */
export interface CinematicSource {signal: Float32Array; shape: Point3; texture: THREE.Data3DTexture; worldToVoxel: THREE.Matrix4; voxelMm: number; ct: boolean; floatLinear: boolean}
export interface CinematicLighting {vesselColors: [THREE.Color, THREE.Color]; boneColor: THREE.Color}
/** Translucent mesh layers front to back: premultiplied colour and the depth of their front. */
export interface CinematicSurfaceLayers {count: number; colors: THREE.Texture[]; depths: Array<THREE.Texture | null>}
/** Uniforms shared with the standard ray caster: one set of physical cuts. */
export interface CinematicSharedUniforms {
  uClipEnabled: THREE.IUniform<boolean>; uClipPlane: THREE.IUniform<THREE.Vector4>;
  uExtraClipCount: THREE.IUniform<number>; uExtraClips: THREE.IUniform<THREE.Vector4[]>;
  uBackground: THREE.IUniform<THREE.Color>;
}

const keyView = new THREE.Vector3(...ctaCutKey).normalize();
const fillView = new THREE.Vector3(.7, -.2, .68).normalize();
const linear = (hex: string) => new THREE.Color(hex);

/** Colour and guide. The displayed colour is linearly filtered when its type allows. */
function multipleTarget(type: THREE.TextureDataType, linearColor: boolean) {
  const target = new THREE.WebGLRenderTarget(1, 1, {count: 2, type, depthBuffer: false});
  target.textures[1].type = THREE.HalfFloatType;
  for (const texture of target.textures) texture.minFilter = texture.magFilter = THREE.NearestFilter;
  if (linearColor) target.textures[0].minFilter = target.textures[0].magFilter = THREE.LinearFilter;
  return target;
}
/** Trace output: faint radiance and transmittance, and the dense collision distance. */
function traceTarget() {
  const target = new THREE.WebGLRenderTarget(1, 1, {count: 2, type: THREE.FloatType, depthBuffer: false});
  for (const texture of target.textures) texture.minFilter = texture.magFilter = THREE.NearestFilter;
  return target;
}
function colorTarget() {
  return new THREE.WebGLRenderTarget(1, 1, {type: THREE.HalfFloatType, depthBuffer: false});
}
function fit(target: THREE.WebGLRenderTarget, w: number, h: number) {
  if (target.width !== w || target.height !== h) target.setSize(w, h);
}
function volumeTexture(data: Uint8Array | Float32Array, shape: Point3, format: THREE.PixelFormat, type: THREE.TextureDataType, filter: THREE.MagnificationTextureFilter) {
  const [x, y, z] = shape;
  const texture = new THREE.Data3DTexture(data, z, y, x);
  texture.format = format;
  texture.type = type;
  texture.minFilter = texture.magFilter = filter;
  texture.unpackAlignment = 1;
  texture.needsUpdate = true;
  return texture;
}

/** Progressive path-traced scalar volume. Owns its tags, glass field,
 * majorants and attachments; the acquired R32F texture is shared and never
 * modified. Tags, optical constants and majorants are prepared in time
 * slices and installed together, so the image never mixes two of them. */
export class CinematicVolumePass {
  samples = 0;
  readonly uniforms: Record<string, THREE.IUniform>;
  private readonly rest: [THREE.ShaderMaterial, THREE.ShaderMaterial];
  private readonly motion: [THREE.ShaderMaterial, THREE.ShaderMaterial];
  private readonly filter: THREE.ShaderMaterial;
  private readonly surfaces: THREE.ShaderMaterial;
  private readonly quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2));
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.Camera();
  private readonly accumulation: THREE.WebGLRenderTarget[];
  private readonly trace = traceTarget();
  private motionTrace: THREE.WebGLRenderTarget | null = null;
  private motionTarget: THREE.WebGLRenderTarget | null = null;
  private motionFiltered: THREE.WebGLRenderTarget[] = [];
  private restFiltered: THREE.WebGLRenderTarget[] = [];
  private surfaceTarget: THREE.WebGLRenderTarget | null = null;
  private tagTexture: THREE.Data3DTexture | null = null;
  private majorantTexture: THREE.Data3DTexture | null = null;
  private fieldTexture: THREE.Data3DTexture | null = null;
  private signalStats?: CinematicSignalStats;
  private field?: CinematicField;
  private tags: CinematicTags | null = null;
  /** Installed layer-set key and its material key ([layer key, material input]). */
  private preparedKey: string | null = null;
  private materialKey = '';
  private requestedKey: string | null = null;
  private requestedInput: CinematicMaterialInput | null = null;
  private job: Job | null = null;
  /** A layer set that could not be prepared is not retried by this pass. */
  private failure: {key: string; message: string} | null = null;
  private limit = CINEMATIC_SAMPLE_CAP;
  private guide: THREE.Texture | null = null;
  private motionCompiled = false;

  private readonly source: CinematicSource;
  private readonly onReady?: () => void;

  constructor(source: CinematicSource, shared: CinematicSharedUniforms, onReady?: () => void) {
    this.source = source;
    this.onReady = onReady;
    this.accumulation = [multipleTarget(THREE.FloatType, source.floatLinear), multipleTarget(THREE.FloatType, source.floatLinear)];
    const [x, y, z] = source.shape;
    this.uniforms = {...shared,
      uVolume: {value: source.texture}, uTags: {value: null}, uMajorant: {value: null}, uField: {value: null},
      uScene: {value: null}, uDepth: {value: null}, uPrevious: {value: null}, uTraceVeil: {value: null}, uTraceHit: {value: null},
      uWorldToVoxel: {value: source.worldToVoxel}, uCameraWorld: {value: new THREE.Matrix4()}, uInverseProjection: {value: new THREE.Matrix4()},
      uShape: {value: new THREE.Vector3(x, y, z)}, uBrickShape: {value: new THREE.Vector3(1, 1, 1)},
      uResolution: {value: new THREE.Vector2(1, 1)}, uSample: {value: 0}, uAccumulate: {value: false}, uJitter: {value: false},
      uTagChannels: {value: 0}, uSingleClass: {value: 0}, uTagGate: {value: new THREE.Vector2()}, uTagFloor: {value: new THREE.Vector3()},
      uVesselRamp: {value: new THREE.Vector4(1e30, 2e30, 1e30, 2e30)}, uVesselSigma: {value: 0},
      uVesselColor0: {value: linear('#a32e22')}, uVesselColor1: {value: linear('#325988')},
      uBoneRamp: {value: new THREE.Vector3(1e30, 2e30, 0)}, uBoneColor: {value: linear('#e3d6bf')},
      uGlass: {value: false}, uFaint: {value: new THREE.Vector4()}, uSoftColor: {value: linear('#c4a196')},
      uGradientScale: {value: 1}, uVoxelMm: {value: source.voxelMm},
      uKeyDirection: {value: new THREE.Vector3()}, uKeyCosine: {value: .97}, uKeyIrradiance: {value: new THREE.Vector3(3, 2.91, 2.76)},
      uFillDirection: {value: new THREE.Vector3()}, uFillRadiance: {value: new THREE.Vector3(.22, .26, .32)},
      uSkyUp: {value: new THREE.Vector3()}, uSkyColor: {value: new THREE.Vector3(.3, .32, .36)}, uGroundColor: {value: new THREE.Vector3(.05, .045, .04)},
      uBounceMm: {value: 20}, uFaintFloor: {value: 0},
    };
    const kernel = (rest: boolean, pass: number) => new THREE.ShaderMaterial({glslVersion: THREE.GLSL3, vertexShader: volumeVertexShader,
      fragmentShader: cinematicFragmentShader, defines: {CINEMATIC_REST: rest ? 1 : 0, CINEMATIC_PASS: pass},
      depthTest: false, depthWrite: false, toneMapped: false, uniforms: this.uniforms});
    this.rest = [kernel(true, 0), kernel(true, 1)];
    this.motion = [kernel(false, 0), kernel(false, 1)];
    this.filter = new THREE.ShaderMaterial({glslVersion: THREE.GLSL3, vertexShader: volumeVertexShader, fragmentShader: cinematicDenoiseShader,
      depthTest: false, depthWrite: false, toneMapped: false,
      uniforms: {uColor: {value: null}, uGuide: {value: null}, uOriginal: {value: null}, uStep: {value: 1}, uBlend: {value: 0}, uSamples: {value: 0}}});
    this.surfaces = new THREE.ShaderMaterial({glslVersion: THREE.GLSL3, vertexShader: volumeVertexShader, fragmentShader: cinematicSurfaceShader,
      depthTest: false, depthWrite: false, toneMapped: false,
      uniforms: {uImage: {value: null}, uGuide: {value: null}, uInverseProjection: this.uniforms.uInverseProjection, uSurfaceCount: {value: 0},
        ...Object.fromEntries(Array.from({length: CINEMATIC_SURFACE_LAYERS}, (_, i) => [[`uSurfaceColor${i}`, {value: null}], [`uSurfaceDepth${i}`, {value: null}]]).flat())}});
    this.quad.frustumCulled = false;
    this.scene.add(this.quad);
  }

  get ready() { return this.preparedKey !== null && this.error === null; }
  get pending() { return this.job !== null; }
  /** Why the requested layer set cannot be path traced, or null. */
  get error() { return this.failure !== null && this.failure.key === this.requestedKey ? this.failure.message : null; }
  get tagStats() { return this.tags; }
  /** The layer-set key of the installed tags. */
  get key() { return this.preparedKey; }
  get sampleLimit() { return this.limit; }
  get needsRefinement() { return this.ready && this.samples > 0 && this.samples < this.limit; }
  get settled() { return this.samples >= CINEMATIC_SETTLED_SAMPLES; }
  invalidate() { this.samples = 0; }

  /** Request tags for a vessel layer set (named by key) and optical constants
   * for a material input. Until both are prepared the installed ones stay. */
  prepare(key: string, layers: readonly CinematicTagLayer[], input: CinematicMaterialInput, synchronous = false) {
    this.requestedKey = key;
    this.requestedInput = input;
    if (this.failure?.key === key) { this.cancel(); return; }
    if (this.job) {
      // A job for this layer set builds the material of the latest request.
      if (this.job.key === key) return;
      this.cancel();
    }
    if (key === this.preparedKey) {
      if (JSON.stringify([key, input]) !== this.materialKey) this.start(key, null, synchronous);
      return;
    }
    this.start(key, layers, synchronous);
  }

  private *build(key: string, layers: readonly CinematicTagLayer[] | null): Generator<void, void> {
    const source = this.source;
    this.signalStats ??= yield* cinematicSignalSteps(source.signal, source.shape);
    if (source.ct && !this.field) {
      const e = source.worldToVoxel.elements;
      this.field = yield* cinematicFieldSteps(source.signal, source.shape, [e[0], e[4], e[8], e[1], e[5], e[9], e[2], e[6], e[10]]);
      this.fieldTexture = volumeTexture(this.field.data, source.shape, THREE.RGBAFormat, THREE.UnsignedByteType, THREE.LinearFilter);
    }
    const tags = layers ? yield* cinematicTagSteps(source.signal, source.shape, layers, this.signalStats.range) : this.tags;
    // An input requested while the majorants were built is built again
    // before anything is installed.
    for (;;) {
      const input = this.requestedInput!, materialKey = JSON.stringify([key, input]);
      const material = cinematicMaterial(tags, input), glass = material.faint.glass && !!this.field;
      const majorants = yield* cinematicMajorantSteps(this.signalStats, tags, material, glass ? this.field! : null);
      if (JSON.stringify([key, this.requestedInput]) !== materialKey) continue;
      this.install(key, tags, input, material, glass, majorants, materialKey);
      return;
    }
  }

  private start(key: string, layers: readonly CinematicTagLayer[] | null, synchronous: boolean) {
    const task: Job = {key, steps: this.build(key, layers), timer: null};
    this.job = task;
    const advance = () => {
      if (this.job !== task) return;
      const deadline = performance.now() + 5;
      try {
        do { if (task.steps.next().done) { this.job = null; break; } } while (synchronous || performance.now() < deadline);
      } catch (error) {
        this.job = null;
        this.failure = {key, message: error instanceof Error ? error.message : String(error)};
        console.warn('Cinematic volume tags unavailable; using the standard volume.', error);
      }
      if (this.job === task) task.timer = setTimeout(advance, 0);
      else if (!synchronous) this.onReady?.();
    };
    if (synchronous) advance();
    else task.timer = setTimeout(advance, 0);
  }

  private cancel() {
    if (this.job?.timer != null) clearTimeout(this.job.timer);
    this.job = null;
  }

  private install(key: string, tags: CinematicTags | null, input: CinematicMaterialInput, material: ReturnType<typeof cinematicMaterial>,
    glass: boolean, majorants: Float32Array, materialKey: string) {
    const u = this.uniforms;
    if (tags !== this.tags) {
      this.tagTexture?.dispose();
      this.tagTexture = tags?.data ? volumeTexture(tags.data, this.source.shape, tags.channels === 2 ? THREE.RGFormat : THREE.RedFormat,
        THREE.UnsignedByteType, THREE.LinearFilter) : null;
      this.tags = tags;
      u.uTags.value = this.tagTexture;
      u.uTagChannels.value = tags?.channels ?? 0;
      u.uSingleClass.value = tags?.channels === 1 ? Math.max(0, tags.channelOf.indexOf(0)) : 0;
    }
    u.uField.value = this.fieldTexture;
    u.uTagGate.value.set(...material.gate);
    u.uTagFloor.value.set(...material.floor);
    u.uVesselRamp.value.set(...material.ramps[0], ...material.ramps[1]);
    u.uVesselSigma.value = material.vesselSigma;
    u.uBoneRamp.value.set(...material.bone.ramp, material.bone.dense ? material.bone.sigma : 0);
    u.uGlass.value = glass;
    u.uFaint.value.set(material.faint.soft, material.faint.bone, ...material.faint.ramp);
    u.uSoftColor.value.copy(linear(input.ct ? '#cbd2d8' : '#9da5ad'));
    u.uGradientScale.value = material.gradientScale;
    u.uFaintFloor.value = glass ? CINEMATIC_FAINT_FLOOR.glass : CINEMATIC_FAINT_FLOOR.fog;
    const [bx, by, bz] = this.signalStats!.bricks;
    this.majorantTexture?.dispose();
    this.majorantTexture = volumeTexture(majorants, [bx, by, bz], THREE.RGBAFormat, THREE.FloatType, THREE.NearestFilter);
    u.uMajorant.value = this.majorantTexture;
    u.uBrickShape.value.set(bx, by, bz);
    this.preparedKey = key;
    this.materialKey = materialKey;
    this.samples = 0;
  }

  /** Light and material colours; they do not bound extinction, so they apply at once. */
  setLighting(lighting: CinematicLighting) {
    const u = this.uniforms;
    u.uVesselColor0.value.copy(lighting.vesselColors[0]);
    u.uVesselColor1.value.copy(lighting.vesselColors[1]);
    u.uBoneColor.value.copy(lighting.boneColor);
  }

  private draw(renderer: THREE.WebGLRenderer, material: THREE.Material, target: THREE.WebGLRenderTarget) {
    this.quad.material = material;
    renderer.setRenderTarget(target);
    renderer.render(this.scene, this.camera);
  }

  private sample(renderer: THREE.WebGLRenderer, kernels: [THREE.ShaderMaterial, THREE.ShaderMaterial], trace: THREE.WebGLRenderTarget, target: THREE.WebGLRenderTarget) {
    this.draw(renderer, kernels[0], trace);
    this.uniforms.uTraceVeil.value = trace.textures[0];
    this.uniforms.uTraceHit.value = trace.textures[1];
    this.draw(renderer, kernels[1], target);
    this.guide = target.textures[1];
  }

  private denoise(renderer: THREE.WebGLRenderer, color: THREE.Texture, guide: THREE.Texture, targets: THREE.WebGLRenderTarget[], blend: number,
    steps: number[], samples = 0) {
    const u = this.filter.uniforms;
    u.uGuide.value = guide;
    u.uOriginal.value = color;
    u.uSamples.value = samples;
    let input = color;
    steps.forEach((step, i) => {
      const output = targets[i % 2];
      u.uColor.value = input;
      u.uStep.value = step;
      u.uBlend.value = i === steps.length - 1 ? blend : 0;
      this.draw(renderer, this.filter, output);
      input = output.texture;
    });
    return input;
  }

  /** One sample. Interactive frames are filtered single samples at the drag
   * resolution; settled frames extend the running mean and fade the filter. */
  render(renderer: THREE.WebGLRenderer, camera: THREE.PerspectiveCamera, scene: THREE.WebGLRenderTarget,
    w: number, h: number, interactive: boolean): THREE.Texture {
    const u = this.uniforms;
    camera.updateMatrixWorld();
    u.uCameraWorld.value.copy(camera.matrixWorld);
    u.uInverseProjection.value.copy(camera.projectionMatrixInverse);
    const rotation = new THREE.Matrix3().setFromMatrix4(camera.matrixWorld);
    u.uKeyDirection.value.copy(keyView).applyMatrix3(rotation).normalize();
    u.uFillDirection.value.copy(fillView).applyMatrix3(rotation).normalize();
    u.uSkyUp.value.set(0, 1, 0).applyMatrix3(rotation).normalize();
    u.uScene.value = scene.texture;
    u.uDepth.value = scene.depthTexture;
    u.uResolution.value.set(w, h);
    if (interactive) {
      this.samples = 0;
      this.motionTarget ??= multipleTarget(THREE.HalfFloatType, true);
      this.motionTrace ??= traceTarget();
      if (!this.motionFiltered.length) this.motionFiltered = [colorTarget(), colorTarget()];
      for (const target of [this.motionTarget, this.motionTrace, ...this.motionFiltered]) fit(target, w, h);
      u.uAccumulate.value = false;
      u.uJitter.value = false;
      u.uSample.value = 0;
      this.sample(renderer, this.motion, this.motionTrace, this.motionTarget);
      return this.denoise(renderer, this.motionTarget.textures[0], this.motionTarget.textures[1], this.motionFiltered, 0, MOTION_FILTER_STEPS);
    }
    if (this.samples === 0) this.limit = cinematicSampleLimit();
    const current = this.accumulation[this.samples % 2], previous = this.accumulation[(this.samples + 1) % 2];
    fit(current, w, h); fit(previous, w, h);
    u.uPrevious.value = previous.textures[0];
    u.uAccumulate.value = true;
    u.uJitter.value = true;
    u.uSample.value = this.samples;
    fit(this.trace, w, h);
    this.sample(renderer, this.rest, this.trace, current);
    this.samples++;
    if (!this.motionCompiled) {
      // Compile the moving-frame kernels in the background so the first drag
      // does not stall on them.
      this.motionCompiled = true;
      for (const material of this.motion) {
        this.quad.material = material;
        renderer.compileAsync?.(this.scene, this.camera)?.catch(() => {});
      }
    }
    const blend = cinematicFilterBlend(this.samples);
    if (blend >= 1) {
      // Release the full-resolution filter targets once the mean stands alone.
      if (this.restFiltered.length) { this.restFiltered.forEach(target => target.dispose()); this.restFiltered = []; }
      return current.textures[0];
    }
    if (!this.restFiltered.length) this.restFiltered = [colorTarget(), colorTarget()];
    this.restFiltered.forEach(target => fit(target, w, h));
    return this.denoise(renderer, current.textures[0], current.textures[1], this.restFiltered, blend, REST_FILTER_STEPS, this.samples);
  }

  /** Composite translucent mesh layers over the last rendered image where
   * they lie in front of its first dense collision. */
  overlay(renderer: THREE.WebGLRenderer, image: THREE.Texture, layers: CinematicSurfaceLayers, w: number, h: number): THREE.Texture {
    const u = this.surfaces.uniforms;
    u.uImage.value = image;
    u.uGuide.value = this.guide;
    u.uSurfaceCount.value = Math.min(layers.count, CINEMATIC_SURFACE_LAYERS);
    for (let i = 0; i < CINEMATIC_SURFACE_LAYERS; i++) {
      u[`uSurfaceColor${i}`].value = layers.colors[i] ?? null;
      u[`uSurfaceDepth${i}`].value = layers.depths[i] ?? null;
    }
    this.surfaceTarget ??= colorTarget();
    fit(this.surfaceTarget, w, h);
    this.draw(renderer, this.surfaces, this.surfaceTarget);
    return this.surfaceTarget.texture;
  }

  /** Filter weight of the displayed image: zero once the mean stands alone. */
  get filterWeight() { return this.samples ? 1 - cinematicFilterBlend(this.samples) : 1; }

  get bytes() {
    const targets = [...this.accumulation, this.trace, ...[this.motionTarget, this.motionTrace, this.surfaceTarget].filter(target => !!target),
      ...this.motionFiltered, ...this.restFiltered];
    return targets.reduce((sum, target) => sum + renderTargetBytes(target), 0) + (this.tagTexture?.image.data?.byteLength ?? 0) +
      (this.majorantTexture?.image.data?.byteLength ?? 0) + (this.fieldTexture?.image.data?.byteLength ?? 0);
  }

  dispose() {
    this.cancel();
    this.tagTexture?.dispose();
    this.majorantTexture?.dispose();
    this.fieldTexture?.dispose();
    this.accumulation.forEach(target => target.dispose());
    this.trace.dispose();
    this.motionTarget?.dispose();
    this.motionTrace?.dispose();
    this.surfaceTarget?.dispose();
    [...this.motionFiltered, ...this.restFiltered].forEach(target => target.dispose());
    this.quad.geometry.dispose();
    [...this.rest, ...this.motion, this.filter, this.surfaces].forEach(material => material.dispose());
  }
}
