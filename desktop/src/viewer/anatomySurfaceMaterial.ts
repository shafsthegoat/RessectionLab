import * as THREE from 'three';
import type {Bounds3} from './coordinates';

/** Physical-frame depth cue on vessel surfaces. Only radiance is changed:
 * opacity and the depth used by volume compositing remain authoritative. */
export function applyVesselDepthMaterial(material:THREE.MeshStandardMaterial,bounds:Bounds3) {
  const center=new THREE.Vector3(...bounds[0].map((v,i)=>(v+bounds[1][i])/2));
  const radius=Math.max(1,new THREE.Vector3(...bounds[1]).distanceTo(new THREE.Vector3(...bounds[0]))*.5);
  const previous=material.onBeforeCompile,previousKey=material.customProgramCacheKey();
  material.onBeforeCompile=(shader,renderer)=>{
    previous.call(material,shader,renderer);
    shader.uniforms.uAnatomyCenter={value:center};shader.uniforms.uAnatomyRadius={value:radius};
    shader.fragmentShader='uniform vec3 uAnatomyCenter;\nuniform float uAnatomyRadius;\n'+shader.fragmentShader;
    shader.fragmentShader=shader.fragmentShader.replace('#include <opaque_fragment>',`
      float anatomyDepth=(length(vViewPosition)-length((viewMatrix*vec4(uAnatomyCenter,1.0)).xyz))/uAnatomyRadius;
      outgoingLight*=mix(1.08,0.56,smoothstep(-0.65,0.65,anatomyDepth));
      #include <opaque_fragment>`);
  };
  material.customProgramCacheKey=()=>`anatomy-vessel-depth-v1-${previousKey}`;
  material.needsUpdate=true;
}
