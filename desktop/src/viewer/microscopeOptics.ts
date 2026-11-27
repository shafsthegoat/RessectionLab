import * as THREE from 'three';
import {renderTargetBytes} from './renderResourceEstimate.ts';
import {renderAnatomyScene, type VesselSlabContext} from './vesselSlabContext.ts';

/** Illustrative operating-microscope optics for the surgeon view only: depth
 * of field about the current focus plane and a mild falloff toward the edge of
 * the circular field. Geometry, picking and measurements are unaffected; the
 * blur is a display convention, not a model of a particular microscope. */
export interface MicroscopeOpticsParameters {
  /** Camera-to-focus distance along the view axis, in mm. */
  focusDistanceMm: number;
  /** Width of the field at the focus plane, in mm. */
  fieldMm: number;
  /** The craniotomy (patient RAS centre and radius in mm): the periphery beyond
   * it falls darker, as under a microscope lamp, so the eye goes to the opening. */
  opening?: {centerMm: [number, number, number]; radiusMm: number} | null;
}

/** Circle-of-confusion radius in pixels for a surface `distanceMm` from the
 * camera. Within `toleranceMm` of the focus plane the image is sharp; beyond it
 * the radius grows with the defocus, scaled by the field's pixels per mm, up
 * to `maxPx`. */
export function defocusRadiusPx(distanceMm: number, p: MicroscopeOpticsParameters, heightPx: number,
  apertureRatio = 0.02, toleranceMm = 5, maxPx = 6): number {
  const defocus = Math.max(0, Math.abs(distanceMm - p.focusDistanceMm) - toleranceMm);
  return Math.min(maxPx, defocus * apertureRatio * heightPx / Math.max(1e-6, p.fieldMm));
}

/** Relative brightness at a normalised radius from the field centre (1 at the
 * inscribed circle of the pane): a mild optical falloff, not a mask. */
export function fieldFalloff(radius: number, strength = 0.3): number {
  const t = Math.min(1, Math.max(0, (radius - 0.62) / (1.12 - 0.62)));
  return 1 - strength * t * t * (3 - 2 * t);
}

/** Relative brightness at `distance` (any unit) from the opening centre for an
 * opening of `radius`: full inside the opening and its bone rim, falling
 * smoothly to `1 - strength` at about two opening radii (scalp and drapes). */
export const OPENING_FALLOFF = {start: 1.08, span: 0.9, strength: 0.6};
export function openingFalloff(distance: number, radius: number, strength = OPENING_FALLOFF.strength): number {
  if (!(radius > 0)) return 1;
  const t = Math.min(1, Math.max(0, (distance / radius - OPENING_FALLOFF.start) / OPENING_FALLOFF.span));
  return 1 - strength * t * t * (3 - 2 * t);
}

/** Centre and radius of the bone flap outline in patient RAS mm, from the
 * engine's bone body cells; null without one. */
export function openingFromCells(centres: ReadonlyArray<readonly [number, number, number]>, outward?: readonly [number, number, number] | null) {
  if (!centres.length) return null;
  const c = [0, 0, 0] as [number, number, number];
  for (const p of centres) for (let k = 0; k < 3; k++) c[k] += p[k] / centres.length;
  const n = outward ? Math.hypot(...outward) : 0, u = n ? outward!.map((v) => v / n) : null;
  let radius = 0;
  for (const p of centres) {
    const d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]], along = u ? d[0] * u[0] + d[1] * u[1] + d[2] * u[2] : 0;
    radius = Math.max(radius, Math.hypot(d[0] - along * (u?.[0] ?? 0), d[1] - along * (u?.[1] ?? 0), d[2] - along * (u?.[2] ?? 0)));
  }
  return {centerMm: c, radiusMm: radius};
}

const fragmentShader = `
#include <packing>
uniform sampler2D uColor;
uniform sampler2D uDepth;
uniform vec2 uSize;
uniform float uNear;
uniform float uFar;
uniform float uFocus;
uniform float uPixelsPerMm;
uniform float uAperture;
uniform float uTolerance;
uniform float uMaxRadius;
uniform float uFalloff;
uniform vec3 uOpening;
uniform float uOpeningStrength;
varying vec2 vUv;
float distanceAt(vec2 uv){
  float d=texture2D(uDepth,uv).x;
  return d>=1.?uFar:-perspectiveDepthToViewZ(d,uNear,uFar);
}
float cocAt(vec2 uv){
  return min(uMaxRadius,max(0.,abs(distanceAt(uv)-uFocus)-uTolerance)*uAperture*uPixelsPerMm);
}
void main(){
  float coc=cocAt(vUv);
  vec4 sum=texture2D(uColor,vUv);float weight=1.;
  if(coc>.5){
    // Golden-angle disk; a sample only spreads onto this pixel when it is
    // itself defocused, so sharp tissue in front does not bleed backwards.
    for(int i=1;i<24;i++){
      float r=sqrt(float(i)/24.)*coc,a=float(i)*2.39996;
      vec2 uv=vUv+vec2(cos(a),sin(a))*r/uSize;
      float c=cocAt(uv),w=smoothstep(r-1.,r+.5,c)+.02;
      sum+=texture2D(uColor,uv)*w;weight+=w;
    }
  }
  gl_FragColor=sum/weight;
  vec2 centred=(vUv-.5)*uSize/(.5*min(uSize.x,uSize.y));
  float t=clamp((length(centred)-.62)/.5,0.,1.);
  gl_FragColor.rgb*=1.-uFalloff*t*t*(3.-2.*t);
  if(uOpening.z>0.){
    // Lamp spot on the opening (uOpening: centre in pixels, radius in pixels).
    float o=clamp((length(vUv*uSize-uOpening.xy)/uOpening.z-${OPENING_FALLOFF.start.toFixed(3)})/${OPENING_FALLOFF.span.toFixed(3)},0.,1.);
    gl_FragColor.rgb*=1.-uOpeningStrength*o*o*(3.-2.*o);
  }
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

export class MicroscopeOpticsPass {
  private target: THREE.WebGLRenderTarget | null = null;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private readonly geometry = new THREE.PlaneGeometry(2, 2);
  readonly material = new THREE.ShaderMaterial({
    vertexShader: 'varying vec2 vUv;void main(){vUv=uv;gl_Position=vec4(position.xy,0.,1.);}',
    fragmentShader,
    depthTest: false,
    depthWrite: false,
    toneMapped: true,
    uniforms: {
      uColor: {value: null}, uDepth: {value: null}, uSize: {value: new THREE.Vector2(1, 1)},
      uNear: {value: 0.1}, uFar: {value: 1000}, uFocus: {value: 160}, uPixelsPerMm: {value: 10},
      uAperture: {value: 0.02}, uTolerance: {value: 5}, uMaxRadius: {value: 6}, uFalloff: {value: 0.3},
      uOpening: {value: new THREE.Vector3()}, uOpeningStrength: {value: OPENING_FALLOFF.strength},
    },
  });
  constructor() {
    this.scene.add(new THREE.Mesh(this.geometry, this.material));
  }
  get estimatedTargetBytes() {
    return this.target ? renderTargetBytes(this.target) : 0;
  }
  render(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.PerspectiveCamera, width: number, height: number,
    optics: MicroscopeOpticsParameters, restore: () => void, context?: VesselSlabContext) {
    const ratio = renderer.getPixelRatio(),
      w = Math.max(1, Math.round(width * ratio)),
      h = Math.max(1, Math.round(height * ratio));
    if (!this.target) {
      this.target = new THREE.WebGLRenderTarget(w, h, {type: THREE.HalfFloatType, depthBuffer: true, samples: 4});
      this.target.depthTexture = new THREE.DepthTexture(w, h, THREE.UnsignedIntType);
    } else if (this.target.width !== w || this.target.height !== h) this.target.setSize(w, h);
    try {
      renderer.setRenderTarget(this.target);
      renderer.setScissorTest(false);
      renderer.setViewport(0, 0, w / ratio, h / ratio);
      renderer.clear();
      renderAnatomyScene(renderer, scene, camera, context);
    } finally {
      renderer.setRenderTarget(null);
      renderer.setScissorTest(true);
      restore();
    }
    const u = this.material.uniforms;
    u.uColor.value = this.target.texture;
    u.uDepth.value = this.target.depthTexture;
    u.uSize.value.set(w, h);
    u.uNear.value = camera.near;
    u.uFar.value = camera.far;
    u.uFocus.value = optics.focusDistanceMm;
    u.uPixelsPerMm.value = h / Math.max(1e-6, optics.fieldMm);
    if (optics.opening) {
      const centre = new THREE.Vector3(...optics.opening.centerMm).project(camera),
        distance = camera.position.distanceTo(new THREE.Vector3(...optics.opening.centerMm)),
        pixelsPerMm = h / (2 * distance * Math.tan((camera.fov * Math.PI) / 360));
      u.uOpening.value.set((centre.x * 0.5 + 0.5) * w, (centre.y * 0.5 + 0.5) * h, centre.z < 1 ? optics.opening.radiusMm * pixelsPerMm : 0);
    } else u.uOpening.value.set(0, 0, 0);
    renderer.render(this.scene, this.camera);
    return {adaptive: false, width: w, height: h, optics: 'microscope-depth-of-field'};
  }
  dispose() {
    this.target?.dispose();
    this.material.dispose();
    this.geometry.dispose();
  }
}
