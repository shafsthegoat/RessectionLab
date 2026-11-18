import * as THREE from 'three';
import type { OperativeField } from '../operative-field';
import type { FieldLayer } from './operativeFieldModel';

type Signal = NonNullable<OperativeField['intensity']>;

/** The second channel carries acquisition coverage through GPU interpolation.
 * A dark measured voxel and a sample outside the source have different codes. */
export function operativeSignalTexture(signal: Signal) {
  const data = new Uint8Array(signal.voxels.length * 2);
  for (let i = 0; i < signal.voxels.length; i++) {
    const value = signal.voxels[i];
    data[2 * i] = value;
    data[2 * i + 1] = value === signal.unknownValue ? 0 : 255;
  }
  const [x, y, z] = signal.shape;
  const texture = new THREE.Data3DTexture(data, z, y, x);
  texture.format = THREE.RGFormat;
  texture.minFilter = texture.magFilter = THREE.LinearFilter;
  texture.unpackAlignment = 1;
  texture.needsUpdate = true;
  return texture;
}

const tissues = {
  bone: { color: '#e5d8bc', roughness: 0.68, contrast: 0.48, section: 0.2 },
  dura: { color: '#d7b2a0', roughness: 0.46, contrast: 0.64, section: 0.32 },
  brain: { color: '#c68788', roughness: 0.6, contrast: 1.4, section: 0.83 },
  tumor: { color: '#b67cae', roughness: 0.55, contrast: 1.02, section: 0.64 },
  vessel: { color: '#be4349', roughness: 0.38, contrast: 0.4, section: 0.16 },
};

export function operativeTissueMaterial(
  name: FieldLayer,
  field: OperativeField,
  texture: THREE.Data3DTexture | null,
  restOffset: THREE.Vector3,
  sectionY: THREE.IUniform<number>
) {
  const tissue = tissues[name === 'boneFlap' ? 'bone' : name === 'duraFlap' ? 'dura' : name];
  const material = new THREE.MeshStandardMaterial({ color: tissue.color, roughness: tissue.roughness });
  const signal = field.intensity;
  if (!signal || !texture) return material;

  const origin = new THREE.Vector3(...signal.origin);
  const scale = new THREE.Vector3(...signal.shape).multiply(new THREE.Vector3(...signal.spacingMm));
  const half = field.spacingMm / 2;
  const min = new THREE.Vector3(...field.origin).addScalar(-half);
  const max = min.clone().addScaledVector(new THREE.Vector3(...field.shape), field.spacingMm);
  // The min corner is half a texel before the first sample centre.
  origin.addScaledVector(new THREE.Vector3(...signal.spacingMm), -0.5);
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, {
      uFieldSignal: { value: texture },
      uFieldMin: { value: origin },
      uFieldSize: { value: scale },
      uRestOffset: { value: restOffset },
      uFieldContrast: { value: tissue.contrast },
      uCropMin: { value: min },
      uCropMax: { value: max },
      uSectionY: sectionY,
      uBoundaryWidth: { value: half },
      uSectionMix: { value: tissue.section },
    });
    shader.vertexShader =
      `uniform vec3 uFieldMin, uFieldSize, uRestOffset;
      varying vec3 vFieldUVW, vRestPosition;
    ` +
      shader.vertexShader.replace(
        '#include <begin_vertex>',
        `#include <begin_vertex>
      vRestPosition=position+uRestOffset;
      vFieldUVW=((vRestPosition-uFieldMin)/uFieldSize).zyx;`
      );
    // Sample rest coordinates so lifted bone, reflected dura and displaced
    // brain carry their original scan contrast as the geometry moves.
    shader.fragmentShader =
      `uniform highp sampler3D uFieldSignal;
      uniform float uFieldContrast, uSectionY, uBoundaryWidth, uSectionMix;
      uniform vec3 uCropMin, uCropMax;
      varying vec3 vFieldUVW, vRestPosition;
    ` +
      shader.fragmentShader.replace(
        '#include <color_fragment>',
        `#include <color_fragment>
      if (all(greaterThanEqual(vFieldUVW,vec3(0.))) && all(lessThanEqual(vFieldUVW,vec3(1.)))) {
        vec2 signal=texture(uFieldSignal,vFieldUVW).rg;
        if(signal.g>.999){
          float brightness=clamp((255.*signal.r-1.)/254.,0.,1.);
          diffuseColor.rgb *= exp2((brightness-.5)*uFieldContrast*1.85);
          // Use measured grayscale on the cut faces, with enough of the
          // layer color to identify the tissue across the cut.
          vec3 edge=min(abs(vRestPosition-uCropMin),abs(vRestPosition-uCropMax));
          float distanceToFace=min(min(edge.x,edge.y),min(edge.z,abs(vRestPosition.y-uSectionY)));
          float section=1.-smoothstep(.02*uBoundaryWidth,uBoundaryWidth,distanceToFace);
          vec3 tint=mix(vec3(1.),diffuse,0.42);
          vec3 scanColor=(.014+.68*pow(brightness,2.2))*tint;
          diffuseColor.rgb=mix(diffuseColor.rgb,scanColor,uSectionMix*section);
        }
      }`
      );
  };
  material.customProgramCacheKey = () => 'operative-acquired-section-v2';
  return material;
}
