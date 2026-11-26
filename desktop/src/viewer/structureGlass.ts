import type * as THREE from 'three';
import type {ViewerWorkspaceSettings} from './workspaceSettings';

/** Display style of one structure surface. Geometry, masks, measurements and
 * picking are unchanged; only the mesh's facing alpha and rim radiance differ. */
export type StructureSurfaceStyle='glass'|'solid';
export type StructureTier='target'|'structure'|'outer-shell'|'inner-shell'|'cortex'|'vessel'|'other';

const TIERS:Record<Exclude<StructureTier,'other'>,ReadonlySet<string>>={
  target:new Set(['tumor','tumour','lesion','target','enhancing-tumor','tumor-core','tumor-whole','nonenhancing-tumor','edema','lesion-context']),
  'outer-shell':new Set(['head-envelope','outer-head-tissue','scalp','skin','soft_tissue','tissue','bone','skull']),
  'inner-shell':new Set(['brain','dura','csf']),
  cortex:new Set(['gray-matter','cerebral-cortex','cerebellar-cortex','white-matter','cerebellar-white-matter']),
  vessel:new Set(['vessel','vessels','artery','vein','sinus']),
  structure:new Set(['ventricles','brain-region','brainstem','deep-gray','nucleus','structure']),
};

export function structureTier(kind:string|undefined):StructureTier {
  for(const [tier,kinds] of Object.entries(TIERS))if(kinds.has(kind??''))return tier as StructureTier;
  return 'other';
}

/** Lesions read as bright-rimmed glass. Context shells get a subtle glass rim.
 * Vessels and folded cortex keep their solid shading; unknown kinds stay solid. */
export function defaultStructureSurfaceStyle(kind:string|undefined):StructureSurfaceStyle {
  return ['target','structure','outer-shell','inner-shell'].includes(structureTier(kind))?'glass':'solid';
}

/** A saved choice wins; a missing or unrecognized entry means the kind's default. */
export function layerSurfaceStyle(name:string,kind:string|undefined,settings:ViewerWorkspaceSettings={}):StructureSurfaceStyle {
  const chosen=settings.layerSurfaceStyle?.[name];
  return chosen==='glass'||chosen==='solid'?chosen:defaultStructureSurfaceStyle(kind);
}

export function withLayerSurfaceStyle(settings:ViewerWorkspaceSettings,name:string,style:StructureSurfaceStyle):ViewerWorkspaceSettings {
  return {...settings,layerSurfaceStyle:{...settings.layerSurfaceStyle,[name]:style}};
}

interface GlassProfile {
  /** Facing alpha as a fraction of the layer opacity. */
  interior:number;
  /** Silhouette alpha as a multiple of the layer opacity, clamped to one. */
  edge:number;
  /** Fresnel exponent applied to 1-|n·v|. */
  exponent:number;
  /** Rim radiance in the structure's own colour (linear units, before blending). */
  glow:number;
}

export const GLASS_PROFILES:Record<'target'|'structure'|'shell',GlassProfile>={
  target:{interior:.22,edge:1.5,exponent:2.5,glow:1.1},
  structure:{interior:.5,edge:1.35,exponent:2.5,glow:.45},
  shell:{interior:.6,edge:1.25,exponent:2.5,glow:.28},
};

/** Shader-only lighting for non-vessel surfaces, added to the scene's fixed
 * key, hemisphere and environment: a camera-relative key from the upper left,
 * a camera-relative rim light, and curvature cavity darkening. Vessels keep
 * their established finish. */
const LIGHTING:Record<StructureTier,{key:number;cavity:number;rimLight:number}>={
  target:{key:.32,cavity:.28,rimLight:.3},
  structure:{key:.3,cavity:.32,rimLight:.3},
  'outer-shell':{key:.22,cavity:.22,rimLight:.22},
  'inner-shell':{key:.26,cavity:.3,rimLight:.25},
  cortex:{key:.2,cavity:.18,rimLight:.2},
  vessel:{key:0,cavity:0,rimLight:0},
  other:{key:.26,cavity:.25,rimLight:.25},
};

/** Transparent draw order. Context shells composite first, so their rear walls
 * never paint over a target inside them; the target follows. Within one tier,
 * Three's depth sort and back-then-front faces of double-sided meshes apply. */
const RENDER_ORDER:Record<StructureTier,number>={
  'outer-shell':-3,'inner-shell':-2,cortex:-2,structure:-1,target:0,vessel:0,other:0,
};

export interface StructureSurfaceDisplay {
  tier:StructureTier;
  /** Engine-modeled tissue: never glass or extra light. */
  modeled:boolean;
  style:StructureSurfaceStyle;
  /** Whether glass changes this surface at the current opacity. */
  glass:boolean;
  /** Alpha of a surface facing the viewer, written to material.opacity. */
  materialOpacity:number;
  /** Alpha at the silhouette. */
  edgeAlpha:number;
  exponent:number;
  glow:number;
  key:number;
  cavity:number;
  rimLight:number;
  renderOrder:number;
}

/**
 * Glass on a target uses the configured opacity as its rim alpha and lets the
 * facing wall fall to `interior`, so an opaque lesion becomes a glass shell.
 * Other structures follow the existing context rule: a layer set to full
 * opacity (for example an opaque skull exterior) keeps its solid presentation.
 * `contextShell` marks layers whose facing alpha the source context shell
 * already reduces; glass then adds only its rim. `nativeLit` marks surfaces
 * whose native context lighting already supplies a camera-relative key.
 * Operative views and engine-modeled tissue (`modeled`) keep their own
 * finish: no glass or extra light.
 */
export function structureSurfaceDisplay(name:string,kind:string|undefined,opacity:number,settings:ViewerWorkspaceSettings={},
  {contextShell=false,modeled=false,nativeLit=false}:{contextShell?:boolean;modeled?:boolean;nativeLit?:boolean}={}):StructureSurfaceDisplay {
  const tier=structureTier(kind),style=layerSurfaceStyle(name,kind,settings);
  const operative=!!settings.operativeView||modeled;
  const profile=tier==='target'?GLASS_PROFILES.target:tier==='outer-shell'||tier==='inner-shell'?GLASS_PROFILES.shell:GLASS_PROFILES.structure;
  const glass=style==='glass'&&!operative&&opacity>0&&(tier==='target'||opacity<1);
  const interior=contextShell&&profile===GLASS_PROFILES.shell?1:profile.interior;
  const materialOpacity=glass?opacity*interior:opacity;
  const lighting=operative?{key:0,cavity:0,rimLight:0}:{...LIGHTING[tier],...(nativeLit?{key:0}:{})};
  return {tier,modeled,style,glass,materialOpacity,edgeAlpha:glass?Math.min(1,Math.max(materialOpacity,opacity*profile.edge)):materialOpacity,
    exponent:profile.exponent,glow:glass?profile.glow:0,...lighting,renderOrder:RENDER_ORDER[tier]};
}

/** Fresnel rim factor, identical for front and back faces. */
export function glassRimFactor(viewCosine:number,exponent=2.5):number {
  return (1-Math.min(1,Math.abs(viewCosine)))**exponent;
}

/** CPU counterpart of the fragment alpha, for checking the response without GL. */
export function structureSurfaceAlpha(display:StructureSurfaceDisplay,viewCosine:number):number {
  if(!display.glass)return display.materialOpacity;
  const rim=glassRimFactor(viewCosine,display.exponent);
  return display.materialOpacity+(display.edgeAlpha-display.materialOpacity)*rim;
}

interface ShadingUniforms {
  uGlassActive:{value:boolean};
  uGlassEdgeAlpha:{value:number};
  uGlassExponent:{value:number};
  uGlassGlow:{value:number};
  uStructureKey:{value:number};
  uStructureCavity:{value:number};
  uStructureRimLight:{value:number};
}
const installed=new WeakMap<THREE.MeshStandardMaterial,ShadingUniforms>();

/** Display-only fragment terms, inserted after the material's emissive input
 * and before lighting. Later stages (context lighting, cranial depth peels,
 * the shell alpha) keep their own behaviour and see the updated values. The
 * uniforms are shared with the volume compositor's peel clones. */
function install(material:THREE.MeshStandardMaterial):ShadingUniforms {
  const uniforms:ShadingUniforms={uGlassActive:{value:false},uGlassEdgeAlpha:{value:1},uGlassExponent:{value:2.5},
    uGlassGlow:{value:0},uStructureKey:{value:0},uStructureCavity:{value:0},uStructureRimLight:{value:0}};
  installed.set(material,uniforms);
  const previous=material.onBeforeCompile,previousKey=material.customProgramCacheKey;
  material.onBeforeCompile=(shader,renderer)=>{
    previous.call(material,shader,renderer);
    Object.assign(shader.uniforms,uniforms);
    shader.fragmentShader=`uniform bool uGlassActive;
      uniform float uGlassEdgeAlpha;
      uniform float uGlassExponent;
      uniform float uGlassGlow;
      uniform float uStructureKey;
      uniform float uStructureCavity;
      uniform float uStructureRimLight;
    `+shader.fragmentShader.replace('#include <emissivemap_fragment>',`#include <emissivemap_fragment>
      {
        // Curvature from screen-space derivatives of the view normal, in 1/mm.
        // Concave folds and pits darken the albedo: cheap cavity occlusion.
        vec3 structureDx=dFdx(-vViewPosition),structureDy=dFdy(-vViewPosition);
        float structureCurvature=dot(dFdx(normal),structureDx)/max(dot(structureDx,structureDx),1e-8)
          +dot(dFdy(normal),structureDy)/max(dot(structureDy,structureDy),1e-8);
        diffuseColor.rgb*=1.0-uStructureCavity*smoothstep(0.04,0.5,-structureCurvature);
        float structureEdge=1.0-min(1.0,abs(dot(normal,normalize(vViewPosition))));
        // Camera-relative key (upper left, wrapped) keeps form legible from every orbit.
        float structureKeyCos=dot(normal,vec3(-0.36,0.5,0.787));
        totalEmissiveRadiance+=diffuseColor.rgb*(uStructureKey*clamp((structureKeyCos+0.3)/1.3,0.0,1.0));
        vec3 structureRimColor=mix(diffuseColor.rgb,vec3(1.0),0.22);
        // Camera-relative rim light from above the viewer separates silhouettes from the black field.
        totalEmissiveRadiance+=structureRimColor*(uStructureRimLight*pow(structureEdge,3.0)*(0.6+0.4*normal.y));
        if(uGlassActive){
          float glassRim=pow(structureEdge,uGlassExponent);
          diffuseColor.a=mix(diffuseColor.a,uGlassEdgeAlpha,glassRim);
          totalEmissiveRadiance+=structureRimColor*(uGlassGlow*glassRim);
        }
      }`);
  };
  material.customProgramCacheKey=()=>`structure-glass-v1-${previousKey.call(material)}`;
  material.needsUpdate=true;
  return uniforms;
}

/** Apply a computed display to an anatomy material. Solid vessels and
 * engine-modeled tissue keep their original program verbatim. Other surfaces
 * install the uniform-gated terms on creation, before native context
 * lighting wraps the material, even while an operative view zeroes them. */
export function applyStructureSurface(material:THREE.MeshStandardMaterial,display:StructureSurfaceDisplay):void {
  let uniforms=installed.get(material);
  if(!uniforms){
    if(!display.glass&&(display.tier==='vessel'||display.modeled))return;
    uniforms=install(material);
  }
  uniforms.uGlassActive.value=display.glass;
  uniforms.uGlassEdgeAlpha.value=display.edgeAlpha;
  uniforms.uGlassExponent.value=display.exponent;
  uniforms.uGlassGlow.value=display.glow;
  uniforms.uStructureKey.value=display.key;
  uniforms.uStructureCavity.value=display.cavity;
  uniforms.uStructureRimLight.value=display.rimLight;
  // A smoother finish gives glass a coherent highlight; solid keeps the
  // roughness set by the operative/tissue finish.
  if(display.glass)material.roughness=Math.min(material.roughness,.34);
}

export function structureSurfaceUniforms(material:THREE.MeshStandardMaterial):Readonly<ShadingUniforms>|undefined {
  return installed.get(material);
}
