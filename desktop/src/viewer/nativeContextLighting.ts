import {Vector3} from 'three';
import type {MeshStandardMaterial} from 'three';
import type {Bounds3} from './coordinates';

const updateByMaterial = new WeakMap<MeshStandardMaterial, (enabled: boolean) => void>();

export function setNativeContextLighting(material: MeshStandardMaterial, enabled: boolean): void {
  updateByMaterial.get(material)?.(enabled);
}

/** Presentation for smoothed native context. The standard material retains the scene lights,
 * roughness and highlights. A restrained falloff gives farther geometry less emphasis
 * using its physical view depth. The volume compositor reuses this material. */
export function installNativeContextLighting(material: MeshStandardMaterial, bounds: Bounds3, enabled: boolean): void {
  const low = new Vector3(...bounds[0]), high = new Vector3(...bounds[1]);
  const center = {value: low.clone().lerp(high, .5)};
  const radius = {value: Math.max(1, low.distanceTo(high) / 2)};
  const previous = material.onBeforeCompile;
  const previousKey = material.customProgramCacheKey;
  const lightingKey = () => `native-context-lighting-v2-${previousKey.call(material)}`;
  const compileLighting: MeshStandardMaterial['onBeforeCompile'] = (shader, renderer) => {
    previous.call(material, shader, renderer);
    shader.uniforms.uNativeContextCenter = center;
    shader.uniforms.uNativeContextRadius = radius;
    shader.fragmentShader = `uniform vec3 uNativeContextCenter;
      uniform float uNativeContextRadius;
    ` + shader.fragmentShader.replace('#include <opaque_fragment>', `
        float centerDepth = -(viewMatrix * vec4(uNativeContextCenter, 1.0)).z;
        float depth = (vViewPosition.z - centerDepth) / uNativeContextRadius;
        float attenuation = mix(1.0, 0.78, smoothstep(-0.55, 0.60, depth));
        // Keep the broad camera-relative light for consistent inspection from
        // either side. Blend in the scene's colored light and roughness response.
        float key = max(0.0, dot(normal, normalize(vec3(-0.46, 0.55, 0.70))));
        float fill = max(0.0, dot(normal, normalize(vec3(0.72, -0.22, 0.66))));
        vec3 inspectionLight = diffuseColor.rgb * (0.13 + 0.72 * key + 0.12 * fill);
        vec3 sceneLight = outgoingLight - totalEmissiveRadiance;
        outgoingLight = mix(inspectionLight, sceneLight, 0.32) * attenuation + totalEmissiveRadiance;
      #include <opaque_fragment>`);
  };
  const update = (active: boolean) => {
    if ((material.onBeforeCompile === compileLighting) === active) return;
    // Compile the original GLSL verbatim. A disabled branch can still alter
    // floating-point code generation on the GPU.
    material.onBeforeCompile = active ? compileLighting : previous;
    material.customProgramCacheKey = active ? lightingKey : previousKey;
    material.needsUpdate = true;
  };
  updateByMaterial.set(material, update);
  update(enabled);
}
