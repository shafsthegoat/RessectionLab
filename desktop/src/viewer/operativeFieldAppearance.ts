import * as THREE from 'three';
import type { OperativeField,FieldLayer as SourceLayer } from '../operative-field';
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

export const tissueColors = {
  grayMatter:'#b78491',whiteMatter:'#ece0c9',csf:'#8bbfdb',ventricles:'#5b92c4',
  artery:'#cd5954',vein:'#7283ca',sinus:'#5769ad',enhancing:'#ce77a5',nonenhancingCore:'#9571b3',
  perilesionalSignal:'#cb9e60',tumorCore:'#b289bd',wholeAbnormality:'#a686ad',lesionContext:'#a487ad',
};
const tissues = {
  bone: { color: '#e5d8bc', roughness: 0.76, contrast: 0.68, section: 0.2 },
  dura: { color: '#d7b2a0', roughness: 0.46, contrast: 0.64, section: 0.32 },
  brain: { color: '#c68788', roughness: 0.6, contrast: 1.4, section: 0.83 },
  tumor: { color: '#65bfd3', roughness: 0.55, contrast: 1.02, section: 0.64 },
  vessel: { color: '#be4349', roughness: 0.38, contrast: 0.4, section: 0.16 },
};

/** Combined typed bits render in the generic mesh. Include their provenance
 * whenever a retained cell has two overlapping vessel classes. */
export function unclassifiedVesselAppearance(layers:SourceLayer[],conflictingVoxels:number){
  const candidates=layers.filter(l=>l.role==='vessels'&&l.sampledVoxels>0&&(!l.detail||conflictingVoxels>0));
  const estimated=candidates.length>0&&candidates.every(l=>!!l.model&&(l.vascularContextOnly===true||l.evidence==='Source-derived estimate'));
  return estimated?{label:'Vascular estimate',color:'#947486',roughness:.79}:
    {label:'Unclassified vessel label',color:'#ad7783',roughness:.67};
}

export function operativeTissueMaterial(
  name: FieldLayer,
  field: OperativeField,
  texture: THREE.Data3DTexture | null,
  restOffset: THREE.Vector3,
  sectionY: THREE.IUniform<number>,
  conflictingVoxels=0
) {
  const fluid=name==='csf'||name==='ventricles';
  const base=name==='boneFlap'?'bone':name==='duraFlap'?'dura':name==='artery'||name==='vein'||name==='sinus'?'vessel':
    name==='whiteMatter'||name==='grayMatter'||fluid?'brain':name in tissueColors?'tumor':name;
  const tissue=tissues[base as keyof typeof tissues];
  const generic=name==='vessel'?unclassifiedVesselAppearance(field.layers,conflictingVoxels):null;
  const color=generic?.color??(name in tissueColors?tissueColors[name as keyof typeof tissueColors]:tissue.color);
  const material = new THREE.MeshStandardMaterial({ color, roughness:generic?.roughness??tissue.roughness, transparent:fluid, opacity:fluid ? .42 : 1, depthWrite:!fluid });
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
      uSectionContrast: { value: tissue.section },
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
      uniform float uFieldContrast, uSectionY, uBoundaryWidth, uSectionContrast;
      uniform vec3 uCropMin, uCropMax;
      varying vec3 vFieldUVW, vRestPosition;
    ` +
      shader.fragmentShader.replace(
        '#include <color_fragment>',
        `#include <color_fragment>
      // The closed mesh has flat caps exactly on the cut/crop planes. Use
      // geometric rest-space normals, since fitted vertex normals are smoothed
      // across their rims and moving flaps carry their own acquisition.
      vec3 faceCross=cross(dFdx(vRestPosition),dFdy(vRestPosition));
      vec3 faceAlignment=abs(faceCross)/max(length(faceCross),1e-12);
      vec3 edge=min(abs(vRestPosition-uCropMin),abs(vRestPosition-uCropMax));
      edge.y=min(edge.y,abs(vRestPosition.y-uSectionY));
      vec3 onPlane=(1.-smoothstep(vec3(.005),vec3(.1),edge/uBoundaryWidth))
        *smoothstep(vec3(.97),vec3(.999),faceAlignment);
      float section=step(uCropMin.y,uSectionY)*max(onPlane.x,max(onPlane.y,onPlane.z));
      if (all(greaterThanEqual(vFieldUVW,vec3(0.))) && all(lessThanEqual(vFieldUVW,vec3(1.)))) {
        vec2 signal=texture(uFieldSignal,vFieldUVW).rg;
        if(signal.g>.999){
          float brightness=clamp((255.*signal.r-1.)/254.,0.,1.);
          // Scan variation belongs in albedo. Keeping lighting downstream
          // preserves self-shadow and the color of residual target at the lip.
          float exteriorContrast=1.85*uFieldContrast;
          diffuseColor.rgb*=exp2((brightness-.5)*exteriorContrast);
          // A section through a label is an acquired gray-value slice with
          // tissue tint. Use the scan's grayscale range on the cut face while
          // retaining standard illumination, roughness and instrument shadows.
          vec3 tint=diffuse/max(max(max(diffuse.r,diffuse.g),diffuse.b),.001);
          vec3 sampledAlbedo=pow(vec3(brightness),vec3(2.2))*mix(vec3(1.),tint,.48);
          diffuseColor.rgb=mix(diffuseColor.rgb,sampledAlbedo,section*mix(.72,.90,uSectionContrast));
        }
      }`
      );
  };
  material.customProgramCacheKey = () => 'operative-acquired-albedo-v5';
  return material;
}
