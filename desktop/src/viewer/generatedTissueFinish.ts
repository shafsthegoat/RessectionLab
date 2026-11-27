import * as THREE from 'three';

// Illustrative material finish in source RAS millimetres. The fixed seed and
// rest coordinates make the same surface travel with its native material.
const profiles: Record<string, [number, number, number]> = {
  scalp: [0.22, 0.023, 0],
  galea: [0.13, 0.014, 1],
  skull: [0.12, 0.012, 0],
  diploe: [0.23, 0.024, 0],
  dura: [0.12, 0.012, 1],
  grayMatter: [0.15, 0.012, 0],
  whiteMatter: [0.11, 0.01, 1],
  // A published tract bundle is a named membership of white matter; it looks
  // like the white matter around it.
  tract: [0.11, 0.01, 1],
  // Other extracranial head tissue: muscle, fat and fascia between scalp and bone.
  extracranial: [0.17, 0.016, 1],
  tumor: [0.12, 0.012, 0],
  artery: [0.065, 0.005, 0],
  vein: [0.065, 0.005, 0],
};
const noise = `
float tissueHash(vec3 q) {
  q=fract(q*.1031);q+=dot(q,q.yzx+33.33);return fract((q.x+q.y)*q.z);
}
float tissueGrain(vec3 p) {
  vec3 c=floor(p),f=fract(p);f=f*f*(3.-2.*f);
  vec3 dx=vec3(1,0,0),dy=vec3(0,1,0),dz=vec3(0,0,1);
  return mix(mix(mix(tissueHash(c),tissueHash(c+dx),f.x),mix(tissueHash(c+dy),tissueHash(c+dx+dy),f.x),f.y),
    mix(mix(tissueHash(c+dz),tissueHash(c+dx+dz),f.x),mix(tissueHash(c+dy+dz),tissueHash(c+dx+dy+dz),f.x),f.y),f.z);
}`;
/** Strength of the moist film highlight under the coaxial microscope lamp.
 * Illustrative: brain, tumour and dura look wet; dry layers have none. */
export const wetFilmStrength: Record<string, number> = {
  grayMatter: 0.34, whiteMatter: 0.28, tract: 0.28, tumor: 0.3, dura: 0.42, galea: 0.12, extracranial: 0.1,
};
export function generatedTissueFinish(material: THREE.MeshStandardMaterial, kind: string, mobile: boolean) {
  const profile = profiles[kind];
  if (!profile) return;
  const previous = material.onBeforeCompile,
    key = material.customProgramCacheKey(),
    active = {value: true},
    wet = {value: 0, strength: wetFilmStrength[kind] ?? 0};
  material.userData.tissueFinish = active;
  material.userData.wetFilm = wet;
  material.onBeforeCompile = (shader, renderer) => {
    previous.call(material, shader, renderer);
    shader.uniforms.uGeneratedFinish = active;
    shader.uniforms.uFinishProfile = {value: new THREE.Vector3(...profile)};
    shader.uniforms.uWetFilm = wet;
    shader.vertexShader =
      'varying vec3 vFinishRas;varying float vFinishUnknown;\n' +
      (mobile ? 'attribute vec3 exposureRest;\n' : 'attribute vec3 finishRestOffset;attribute float finishRestUnknown;\n') +
      shader.vertexShader.replace(
        '#include <begin_vertex>',
        '#include <begin_vertex>\nvFinishRas=' + (mobile ? 'exposureRest;vFinishUnknown=0.;' : '(modelMatrix*vec4(position+finishRestOffset,1.)).xyz;vFinishUnknown=finishRestUnknown;'),
      );
    shader.fragmentShader =
      'varying vec3 vFinishRas;varying float vFinishUnknown;uniform bool uGeneratedFinish;uniform vec3 uFinishProfile;uniform float uWetFilm;\n' +
      noise +
      '\n' +
      shader.fragmentShader
        .replace(
          '#include <color_fragment>',
          `#include <color_fragment>
        float finishRelief=0.;
        if(uGeneratedFinish&&vFinishUnknown<.0001){
          vec3 p=vFinishRas;
          float broad=tissueGrain(p*.39)+.55*tissueGrain(p*.87+14.)-.775;
          float grain=tissueGrain(p*(uFinishProfile.z>.5?vec3(2.8,.65,3.):vec3(2.7))) - .5;
          float pores=tissueGrain(p*6.2+2.7)-.5;
          diffuseColor.rgb*=exp2((broad+.55*grain+.28*pores)*uFinishProfile.x*vec3(.70,1.30,1.55));
          finishRelief=(grain+.35*pores)*uFinishProfile.y;
        }`,
        )
        .replace(
          '#include <normal_fragment_maps>',
          `#include <normal_fragment_maps>
        if(uGeneratedFinish&&vFinishUnknown<.0001){
          vec3 dx=dFdx(-vViewPosition),dy=dFdy(-vViewPosition);
          vec3 a=cross(dy,normal),b=cross(normal,dx);float det=dot(dx,a);
          normal=normalize(max(abs(det),1e-12)*normal-sign(det)*(dFdx(finishRelief)*a+dFdy(finishRelief)*b));
        }`,
        )
        .replace(
          '#include <opaque_fragment>',
          `if(uWetFilm>0.){
          // Moist film under a lamp on the viewing axis: a tight glint where the
          // surface faces the eyepiece and a faint broad sheen around it.
          float facing=max(dot(normal,normalize(vViewPosition)),0.);
          outgoingLight+=uWetFilm*(.38*pow(facing,120.)+.05*pow(facing,14.))*vec3(1.,.97,.94);
        }
        #include <opaque_fragment>`,
        );
  };
  material.customProgramCacheKey = () => key + '/generated-finish-' + mobile;
  return active;
}
