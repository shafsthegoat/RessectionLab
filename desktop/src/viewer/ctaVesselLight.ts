import type * as THREE from 'three';

export const ctaCutKey = [-0.61,0.48,0.63] as const;
export const ctaCutKeyView = `vec3(${ctaCutKey.join(',')})`;

type CtaVesselLightUniforms = {
  uCtaVesselLight: THREE.IUniform<boolean>;
  uCtaVesselSignal: THREE.IUniform<THREE.Data3DTexture>;
  uCtaVesselBone: THREE.IUniform<THREE.Data3DTexture | null>;
  uCtaVesselWorldToVoxel: THREE.IUniform<THREE.Matrix4>;
  uCtaVesselShape: THREE.IUniform<THREE.Vector3>;
  uCtaVesselCut: THREE.IUniform<THREE.Vector4>;
};

export function installCtaVesselLight(material: THREE.MeshStandardMaterial, uniforms: CtaVesselLightUniforms) {
  const previous=material.onBeforeCompile, key=material.customProgramCacheKey();
  material.onBeforeCompile=(shader,renderer)=>{
    previous.call(material,shader,renderer);
    Object.assign(shader.uniforms,uniforms);
    shader.fragmentShader=`
      uniform bool uCtaVesselLight;
      uniform highp sampler3D uCtaVesselSignal;
      uniform highp sampler3D uCtaVesselBone;
      uniform mat4 uCtaVesselWorldToVoxel;
      uniform vec3 uCtaVesselShape;
      uniform vec4 uCtaVesselCut;
      float ctaVesselKeyExtinction(vec3 point){
        if(dot(uCtaVesselCut.xyz,point)+uCtaVesselCut.w<0.0)return 0.0;
        vec3 p=(uCtaVesselWorldToVoxel*vec4(point,1.0)).xyz;
        if(any(lessThan(p,vec3(0.0)))||any(greaterThan(p,uCtaVesselShape-1.0)))return 0.0;
        vec3 uv=((p+0.5)/uCtaVesselShape).zyx;
        float coverage=texture(uCtaVesselBone,uv).r;
        if(coverage<=0.0)return 0.0;
        float signal=texture(uCtaVesselSignal,uv).r;
        return coverage*smoothstep(160.0,430.0,signal);
      }
    `+shader.fragmentShader.replace('#include <lights_fragment_end>',`
      #include <lights_fragment_end>
      if(uCtaVesselLight){
        vec3 keyView=normalize(${ctaCutKeyView});
        mat3 viewToWorld=transpose(mat3(viewMatrix));
        vec3 worldPosition=cameraPosition-viewToWorld*vViewPosition;
        vec3 keyWorld=viewToWorld*keyView;
        float depth=ctaVesselKeyExtinction(worldPosition+1.8*keyWorld)
          +ctaVesselKeyExtinction(worldPosition+5.2*keyWorld);
        float visibility=exp(-1.5*depth);
        reflectedLight.directDiffuse*=mix(0.43,1.0,visibility);
        reflectedLight.directSpecular*=mix(0.33,1.0,visibility);
      }
    `);
  };
  material.customProgramCacheKey=()=>key+'/cta-vessel-key';
}
