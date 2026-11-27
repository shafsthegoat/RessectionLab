import * as THREE from 'three';
import { renderTargetBytes } from './renderResourceEstimate.ts';

const localDepthShading = /* glsl */`
  vec4 localEyeDepth(vec4 depth, vec2 range) {
    return range.x * range.y / mix(vec4(range.y), vec4(range.x), depth);
  }
  float localDepthShade(vec4 nearer, float eye, float reach) {
    vec4 difference = eye - nearer;
    vec4 weight = smoothstep(vec4(0.9), vec4(3.5), difference)
      * (1.0 - smoothstep(vec4(reach * 0.5), vec4(reach), difference));
    return dot(weight, vec4(0.20)) + 0.20 * max(max(weight.x, weight.y), max(weight.z, weight.w));
  }
`;

type VesselMesh = THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>;

/** Nearest opaque vessel wall at the current camera. This depth field
 * affects radiance only. The installed meshes still handle visibility,
 * clipping, source identity and picking. */
export class VesselDepthSeparation {
  private readonly depthScene = new THREE.Scene();
  private readonly depthMaterial = new THREE.MeshDepthMaterial({
    side: THREE.DoubleSide,
    colorWrite: false,
  });
  private readonly projection = new THREE.Matrix4();
  private readonly uniforms = {
    uLocalVesselDepth: { value: null as THREE.Texture | null },
    uLocalVesselCamera: { value: new THREE.Matrix4() },
    uLocalVesselRange: { value: new THREE.Vector2() },
    uLocalVesselScale: { value: new THREE.Vector2() },
    uLocalVesselEnabled: { value: false },
  };
  private target?: THREE.WebGLRenderTarget;
  private signature = '';
  get bytes() { return this.target ? renderTargetBytes(this.target) : 0; }
  get active() { return this.uniforms.uLocalVesselEnabled.value; }

  install(material: THREE.MeshStandardMaterial) {
    const previous = material.onBeforeCompile;
    const key = material.customProgramCacheKey();
    material.onBeforeCompile = (shader, renderer) => {
      previous.call(material, shader, renderer);
      Object.assign(shader.uniforms, this.uniforms);
      shader.vertexShader = 'varying vec3 vLocalVesselPosition;\n' + shader.vertexShader
        .replace('#include <project_vertex>', `#include <project_vertex>
          vLocalVesselPosition = (modelMatrix * vec4(transformed, 1.0)).xyz;`);
      shader.fragmentShader = /* glsl */`
        varying vec3 vLocalVesselPosition;
        uniform sampler2D uLocalVesselDepth;
        uniform mat4 uLocalVesselCamera;
        uniform vec2 uLocalVesselRange;
        uniform vec2 uLocalVesselScale;
        uniform bool uLocalVesselEnabled;
        ${localDepthShading}
        float localVesselShade(vec3 world) {
          vec4 camera = uLocalVesselCamera * vec4(world, 1.0);
          if (camera.w <= uLocalVesselRange.x) return 0.0;
          vec2 uv = 0.5 + 0.5 * camera.xy / camera.w;
          if (any(lessThanEqual(uv, vec2(0.0))) || any(greaterThanEqual(uv, vec2(1.0)))) return 0.0;
          vec2 radius = max(uLocalVesselScale * 1.8 / camera.w,
            1.3 / vec2(textureSize(uLocalVesselDepth, 0)));
          vec4 depth = vec4(texture2D(uLocalVesselDepth, uv + vec2(radius.x, 0.0)).r,
                            texture2D(uLocalVesselDepth, uv - vec2(radius.x, 0.0)).r,
                            texture2D(uLocalVesselDepth, uv + vec2(0.0, radius.y)).r,
                            texture2D(uLocalVesselDepth, uv - vec2(0.0, radius.y)).r);
          return localDepthShade(localEyeDepth(depth, uLocalVesselRange), camera.w, 30.0);
        }
      ` + shader.fragmentShader.replace('#include <opaque_fragment>', `
        if (uLocalVesselEnabled) {
          float contact = localVesselShade(vLocalVesselPosition);
          outgoingLight *= 1.0 - 0.72 * contact;
        }
        #include <opaque_fragment>`);
    };
    material.customProgramCacheKey = () => key + '/local-vessel-depth';
  }

  capture(renderer: THREE.WebGLRenderer, camera: THREE.PerspectiveCamera, meshes: VesselMesh[],
    width: number, height: number, longEdge: number) {
    this.uniforms.uLocalVesselEnabled.value = meshes.length > 0;
    if (!this.active) return;
    const scale = Math.min(1, longEdge / Math.max(width, height));
    const w = Math.max(1, Math.round(width * scale));
    const h = Math.max(1, Math.round(height * scale));
    camera.updateMatrixWorld();
    this.projection.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    const planes = meshes[0].material.clippingPlanes;
    const signature = JSON.stringify([w, h, this.projection.elements,
      planes?.map(plane => [...plane.normal.toArray(), plane.constant]),
      meshes.map(mesh => [mesh.geometry.id, ...mesh.matrixWorld.elements])]);
    if (this.signature === signature) return;
    if (!this.target) {
      this.target = new THREE.WebGLRenderTarget(w, h);
      const depth = new THREE.DepthTexture(w, h, THREE.UnsignedIntType);
      depth.magFilter = depth.minFilter = THREE.NearestFilter;
      this.target.depthTexture = depth;
      this.uniforms.uLocalVesselDepth.value = depth;
    }
    if (this.target.width !== w || this.target.height !== h) this.target.setSize(w, h);
    this.depthMaterial.clippingPlanes = planes;
    for (const original of meshes) {
      const mesh = new THREE.Mesh(original.geometry, this.depthMaterial);
      mesh.matrixAutoUpdate = false;
      mesh.matrix.copy(original.matrixWorld);
      this.depthScene.add(mesh);
    }
    const previous = renderer.getRenderTarget();
    const viewport = renderer.getViewport(new THREE.Vector4());
    const scissor = renderer.getScissor(new THREE.Vector4());
    const scissorTest = renderer.getScissorTest();
    try {
      renderer.setRenderTarget(this.target);
      renderer.clear(false, true, false);
      renderer.render(this.depthScene, camera);
      this.signature = signature;
      this.uniforms.uLocalVesselCamera.value.copy(this.projection);
      this.uniforms.uLocalVesselRange.value.set(camera.near, camera.far);
      this.uniforms.uLocalVesselScale.value.set(camera.projectionMatrix.elements[0] * 0.5, camera.projectionMatrix.elements[5] * 0.5);
    } finally {
      this.depthScene.clear();
      renderer.setRenderTarget(previous);
      renderer.setViewport(viewport);
      renderer.setScissor(scissor);
      renderer.setScissorTest(scissorTest);
    }
  }

  dispose() {
    this.depthMaterial.dispose();
    this.target?.dispose();
  }
}
