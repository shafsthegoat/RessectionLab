import * as THREE from 'three';
import type {ViewerWorkspaceSettings} from './workspaceSettings';
const surfaceUniforms=new WeakMap<THREE.MeshStandardMaterial,{value:boolean}>();

/** Illustrative optical appearance only. No anatomical detail or geometry is
 * generated, and source-label colours remain unchanged in the linked MRI. */
export function operativeSurfaceStyle(kind:string|undefined,settings:ViewerWorkspaceSettings){
  if(settings.operativeView!=='surgeon'||settings.microscope?.tissueStyle==='labels')return null;
  if(['brain','soft_tissue','tissue','head-envelope'].includes(kind??''))return {color:'#b77b79',roughness:.6};
  if(kind==='tumor')return {color:'#cfad75',roughness:.78};
  if(['bone','skull'].includes(kind??''))return {color:'#ded0b1',roughness:.8};
  if(['membrane','dura'].includes(kind??''))return {color:'#c8aaa0',roughness:.57};
  return null;
}
export function applyOperativeSurface(material:THREE.MeshStandardMaterial,kind:string|undefined,settings:ViewerWorkspaceSettings){
  const style=operativeSurfaceStyle(kind,settings);
  const existing=surfaceUniforms.get(material);
  if(existing)existing.value=!!style;
  else{
    const enabled={value:!!style};surfaceUniforms.set(material,enabled);
    const previous=material.onBeforeCompile,key=material.customProgramCacheKey();
    material.onBeforeCompile=(shader,renderer)=>{
      previous.call(material,shader,renderer);shader.uniforms.uOperativeSurface=enabled;
      shader.vertexShader='varying vec3 vOperativeWorld;\n'+shader.vertexShader.replace('#include <worldpos_vertex>','#include <worldpos_vertex>\nvOperativeWorld=(modelMatrix*vec4(transformed,1.0)).xyz;');
      shader.fragmentShader=`uniform bool uOperativeSurface;varying vec3 vOperativeWorld;
        float opHash(vec3 p){return fract(sin(dot(p,vec3(12.9898,78.233,37.719)))*43758.5453);}
        float opNoise(vec3 p){vec3 i=floor(p),f=fract(p);f=f*f*(3.0-2.0*f);return mix(mix(mix(opHash(i),opHash(i+vec3(1,0,0)),f.x),mix(opHash(i+vec3(0,1,0)),opHash(i+vec3(1,1,0)),f.x),f.y),mix(mix(opHash(i+vec3(0,0,1)),opHash(i+vec3(1,0,1)),f.x),mix(opHash(i+vec3(0,1,1)),opHash(i+vec3(1,1,1)),f.x),f.y),f.z);}
      `+shader.fragmentShader;
      // Restrained illustrative albedo only: no bump/displacement, anatomical
      // folds or vessel-like patterns, and no time-varying visual noise.
      shader.fragmentShader=shader.fragmentShader.replace('#include <color_fragment>','#include <color_fragment>\nif(uOperativeSurface)diffuseColor.rgb*=0.95+0.065*opNoise(vOperativeWorld*.85)+0.035*opNoise(vOperativeWorld*3.0);');
    };
    material.customProgramCacheKey=()=>`operative-optics-v1-${key}`;material.needsUpdate=true;
  }
  if(style){material.color.set(style.color);material.roughness=style.roughness;material.metalness=0;}
  else material.roughness=kind==='tumor'?.48:['vessel','vessels','artery','vein','sinus'].includes(kind??'')?.32:.52;
}
