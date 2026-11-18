import type * as THREE from 'three';

const uniforms=new WeakMap<THREE.MeshStandardMaterial,{shell:{value:boolean};faceOpacity:{value:number}}>();

/** Shell is an explicit display style on the existing source mesh. Facing
 * translucent context recedes; grazing source-surface normals retain an
 * outline. Source geometry, labels and the configured layer opacity are
 * unchanged. Opaque layers always keep their solid presentation. */
export function applySourceContextMaterial(material:THREE.MeshStandardMaterial,enabled:boolean,boneContext=false) {
  const existing=uniforms.get(material);
  // A broad bone surface needs transmitted front-face shading to show its
  // curvature; very thin translucent envelopes keep a rim-dominant response.
  const floor=boneContext?.48:.06;
  if(existing){existing.shell.value=enabled;existing.faceOpacity.value=floor;return;}
  const shell={value:enabled},faceOpacity={value:floor};uniforms.set(material,{shell,faceOpacity});
  const previous=material.onBeforeCompile,previousKey=material.customProgramCacheKey();
  material.onBeforeCompile=(shader,renderer)=>{
    previous.call(material,shader,renderer);
    shader.uniforms.uSourceContextShell=shell;
    shader.uniforms.uSourceFacingOpacity=faceOpacity;
    shader.fragmentShader='uniform bool uSourceContextShell;\nuniform float uSourceFacingOpacity;\n'+shader.fragmentShader;
    shader.fragmentShader=shader.fragmentShader.replace('#include <opaque_fragment>',`
      if(uSourceContextShell && opacity<0.999){
        float contextFacing=abs(dot(normal,normalize(vViewPosition)));
        diffuseColor.a*=mix(uSourceFacingOpacity,1.0,pow(1.0-contextFacing,2.5));
      }
      #include <opaque_fragment>`);
  };
  material.customProgramCacheKey=()=>`source-context-shell-v1-${previousKey}`;
  material.needsUpdate=true;
}

/** CPU counterpart for checking the presentation response independently of GL. */
export function sourceContextOpacity(opacity:number,viewCosine:number,boneContext=false) {
  const floor=boneContext?.48:.06;
  return opacity>=.999?opacity:opacity*(floor+(1-floor)*(1-Math.min(1,Math.abs(viewCosine)))**2.5);
}
