import type * as THREE from 'three';

const uniforms=new WeakMap<THREE.MeshStandardMaterial,{value:boolean}>();

/** Shell is an explicit display style on the existing source mesh. Facing
 * translucent context recedes; grazing source-surface normals retain an
 * outline. Source geometry, labels and the configured layer opacity are
 * unchanged. Opaque layers always keep their solid presentation. */
export function applySourceContextMaterial(material:THREE.MeshStandardMaterial,enabled:boolean) {
  const existing=uniforms.get(material);
  if(existing){existing.value=enabled;return;}
  const shell={value:enabled};uniforms.set(material,shell);
  const previous=material.onBeforeCompile,previousKey=material.customProgramCacheKey();
  material.onBeforeCompile=(shader,renderer)=>{
    previous.call(material,shader,renderer);
    shader.uniforms.uSourceContextShell=shell;
    shader.fragmentShader='uniform bool uSourceContextShell;\n'+shader.fragmentShader;
    shader.fragmentShader=shader.fragmentShader.replace('#include <opaque_fragment>',`
      if(uSourceContextShell && opacity<0.999){
        float contextFacing=abs(dot(normal,normalize(vViewPosition)));
        diffuseColor.a*=0.06+0.94*pow(1.0-contextFacing,2.5);
      }
      #include <opaque_fragment>`);
  };
  material.customProgramCacheKey=()=>`source-context-shell-v1-${previousKey}`;
  material.needsUpdate=true;
}

/** CPU counterpart for checking the presentation response independently of GL. */
export function sourceContextOpacity(opacity:number,viewCosine:number) {
  return opacity>=.999?opacity:opacity*(.06+.94*(1-Math.min(1,Math.abs(viewCosine)))**2.5);
}
