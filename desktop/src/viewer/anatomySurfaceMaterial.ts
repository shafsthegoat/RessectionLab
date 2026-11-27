import * as THREE from 'three';
import type {Bounds3,Point3,Shape3} from './coordinates';
import type {VesselDisplayFilter,ViewerVesselTrace} from './contracts';
import type {ViewerWorkspaceSettings} from './workspaceSettings';
import {isCtAngiography} from './volumeTransfer.ts';

interface VesselLabelGrid {
  texture:THREE.Data3DTexture;
  shape:readonly [number,number,number];
  worldToVoxel:THREE.Matrix4;
  code:readonly [number,number];
}

interface VesselUniforms {
  uVesselStyle:{value:number};
  uVesselColor:{value:THREE.Color};
  uVesselLabels:{value:THREE.Data3DTexture};
  uVesselShape:{value:THREE.Vector3};
  uVesselWorldToVoxel:{value:THREE.Matrix4};
  uVesselLabelCode:{value:THREE.Vector2};
  uVesselFocusPath:{value:THREE.DataTexture|null};
  uVesselFocusCount:{value:number};
  uVesselFocusMin:{value:THREE.Vector3};
  uVesselFocusMax:{value:THREE.Vector3};
  uVesselFocusRadius2:{value:number};
}

export interface VesselInspectionFocus {
  path:THREE.DataTexture;
  bounds:THREE.Box3;
  radius2:number;
  points:Point3[];
}

export function createVesselInspectionFocus(trace:ViewerVesselTrace):VesselInspectionFocus|undefined {
  const radius=trace.isolationRadiusMm;
  if(radius===undefined||!Number.isFinite(radius)||radius<.5||radius>8)return undefined;
  const points=trace.points,pathData=new Float32Array(points.length*4),bounds=new THREE.Box3();
  for(const [i,point] of points.entries()){
    pathData.set(point,i*4);
    bounds.expandByPoint(new THREE.Vector3(...point));
  }
  const path=new THREE.DataTexture(pathData,points.length,1,THREE.RGBAFormat,THREE.FloatType);
  path.needsUpdate=true;
  return {path,bounds:bounds.expandByScalar(radius),radius2:radius*radius,points};
}

/** Same source-neighborhood test as the fragment shader, for visible picks. */
export function vesselInspectionContains(focus:VesselInspectionFocus,point:THREE.Vector3):boolean {
  if(!focus.bounds.containsPoint(point))return false;
  for(let i=1;i<focus.points.length;i++){
    const [ax,ay,az]=focus.points[i-1],[bx,by,bz]=focus.points[i];
    const dx=bx-ax,dy=by-ay,dz=bz-az,px=point.x-ax,py=point.y-ay,pz=point.z-az;
    const t=THREE.MathUtils.clamp((px*dx+py*dy+pz*dz)/Math.max(dx*dx+dy*dy+dz*dz,1e-8),0,1);
    if((px-t*dx)**2+(py-t*dy)**2+(pz-t*dz)**2<=focus.radius2)return true;
  }
  return false;
}

export function setVesselInspectionFocus(material:THREE.MeshStandardMaterial,focus?:VesselInspectionFocus) {
  const style=material.userData.vesselStyle as VesselUniforms|undefined;
  if(!style)return;
  style.uVesselFocusCount.value=focus?.points.length??0;
  style.uVesselFocusPath.value=focus?.path??null;
  if(focus){
    style.uVesselFocusMin.value.copy(focus.bounds.min);
    style.uVesselFocusMax.value.copy(focus.bounds.max);
    style.uVesselFocusRadius2.value=focus.radius2;
  }
}

const shadingValue={source:0,balanced:1,large:2,fine:3,'cta-contrast':0};
export const vesselSourceBoundaryColor='#747c81';

export const vesselWallFinish = {
  specularIntensity:.46,
  clearcoat:.22,
  clearcoatRoughness:.36,
};

/** One vessel colour for every vessel kind; classes are not told apart by colour. */
export const ctaVesselColors = {
  artery:'#a32e22',
  vein:'#a32e22',
  sinus:'#a32e22',
  vessel:'#a32e22',
  vessels:'#a32e22',
} as const;

const ctaVesselFinish = {
  specularIntensity:.24,
  clearcoat:.11,
  clearcoatRoughness:.44,
};

/** Use the declared layer kind only. The material remains a display choice
 * and leaves linked slice colors and the acquired optical path intact. */
export function ctaVesselColor(kind:string|undefined,settings:ViewerWorkspaceSettings,modality:string|undefined) {
  if(settings.vesselShading!=='cta-contrast'||settings.operativeView||
    !['segmentation','vascular'].includes(settings.presentation??'')||!isCtAngiography(modality??''))return undefined;
  return Object.hasOwn(ctaVesselColors,kind??'')?ctaVesselColors[kind as keyof typeof ctaVesselColors]:undefined;
}

/** Kind used for the CTA vessel colour. A derived part is named after its class, an
 * unclassified layer's connected tree is arterial, and the amber review overlay keeps its
 * own colour. */
export function ctaDisplayKind(layer:{name:string;kind?:string;color?:string;displayFilter?:{scope?:string}}) {
  const derived=/^__vessel_(veins|arteries|review):/.exec(layer.name)?.[1];
  if(derived)return derived==='veins'?'vein':derived==='arteries'?'artery':undefined;
  if(!['rooted','rooted-review'].includes(layer.displayFilter?.scope??'')||!['vessel','vessels'].includes(layer.kind??''))return layer.kind;
  return 'artery';
}

export function applyCtaVesselFinish(material:THREE.MeshStandardMaterial,kind:string|undefined,
  settings:ViewerWorkspaceSettings,modality:string|undefined) {
  if(!(material instanceof THREE.MeshPhysicalMaterial))return;
  const color=ctaVesselColor(kind,settings,modality);
  if(color){
    material.color.set(color);
    Object.assign(material,ctaVesselFinish);
    material.roughness=.47;
    material.userData.ctaVesselFinish=true;
  }else if(material.userData.ctaVesselFinish){
    Object.assign(material,vesselWallFinish);
    delete material.userData.ctaVesselFinish;
  }
}

/** Distinguish the padded source closure while keeping it opaque and pickable. */
function shadeVesselSourceBoundary(shader:Parameters<THREE.MeshStandardMaterial['onBeforeCompile']>[0],shape:Shape3){
  shader.uniforms.uVesselSourceLastSample={value:new THREE.Vector3(shape[0]-1,shape[1]-1,shape[2]-1)};
  shader.uniforms.uVesselSourceBoundaryColor={value:new THREE.Color(vesselSourceBoundaryColor)};
  shader.vertexShader='attribute vec3 vesselRestNative;\nvarying vec3 vVesselRestNative;\n'+shader.vertexShader;
  shader.vertexShader=shader.vertexShader.replace('#include <begin_vertex>',`
    #include <begin_vertex>
    vVesselRestNative=vesselRestNative;`);
  shader.fragmentShader=`
    uniform vec3 uVesselSourceLastSample;
    uniform vec3 uVesselSourceBoundaryColor;
    varying vec3 vVesselRestNative;
  `+shader.fragmentShader;
  shader.fragmentShader=shader.fragmentShader.replace('#include <roughnessmap_fragment>',`
    vec3 sourceExcess=max(-vVesselRestNative,vVesselRestNative-uVesselSourceLastSample);
    float sourceBoundary=smoothstep(0.00001,0.035,max(sourceExcess.x,max(sourceExcess.y,sourceExcess.z)));
    diffuseColor.rgb=mix(diffuseColor.rgb,uVesselSourceBoundaryColor,sourceBoundary);
    #include <roughnessmap_fragment>
    roughnessFactor=mix(roughnessFactor,0.72,sourceBoundary);`);
  shader.fragmentShader=shader.fragmentShader.replace('#include <lights_physical_fragment>',`
    #include <lights_physical_fragment>
    #ifdef USE_CLEARCOAT
      material.clearcoat*=1.0-sourceBoundary;
    #endif
    #ifdef PHYSICAL
      material.specularColor=mix(material.specularColor,vec3(0.018),sourceBoundary);
    #endif`);
}

/** Only radiance changes. Depth, source triangles, masks and picks remain authoritative. */
export function applyVesselDepthMaterial(material:THREE.MeshStandardMaterial,bounds:Bounds3,grid?:VesselLabelGrid,sourceShape?:Shape3) {
  const center=new THREE.Vector3(...bounds[0].map((v,i)=>(v+bounds[1][i])/2));
  const radius=Math.max(1,new THREE.Vector3(...bounds[1]).distanceTo(new THREE.Vector3(...bounds[0]))*.5);
  const previous=material.onBeforeCompile,previousKey=material.customProgramCacheKey();
  const style:VesselUniforms|undefined=grid?{
    uVesselStyle:{value:0},uVesselColor:{value:new THREE.Color()},
    uVesselLabels:{value:grid.texture},uVesselShape:{value:new THREE.Vector3(...grid.shape)},
    uVesselWorldToVoxel:{value:grid.worldToVoxel},uVesselLabelCode:{value:new THREE.Vector2(...grid.code)},
    uVesselFocusPath:{value:null},uVesselFocusCount:{value:0},uVesselFocusMin:{value:new THREE.Vector3()},
    uVesselFocusMax:{value:new THREE.Vector3()},uVesselFocusRadius2:{value:0},
  }:undefined;
  if(style)material.userData.vesselStyle=style;
  material.onBeforeCompile=(shader,renderer)=>{
    previous.call(material,shader,renderer);
    shader.uniforms.uAnatomyCenter={value:center};shader.uniforms.uAnatomyRadius={value:radius};
    shader.fragmentShader='uniform vec3 uAnatomyCenter;\nuniform float uAnatomyRadius;\n'+shader.fragmentShader;
    shader.fragmentShader=shader.fragmentShader.replace('#include <opaque_fragment>',`
      float anatomyDepth=(vViewPosition.z+(viewMatrix*vec4(uAnatomyCenter,1.0)).z)/uAnatomyRadius;
      float farCue=smoothstep(-0.55,0.56,anatomyDepth);
      float grazing=1.0-max(0.0,dot(normal,normalize(vViewPosition)));
      outgoingLight*=mix(1.09,0.49,farCue)*(1.0-0.18*grazing*grazing);
      #include <opaque_fragment>`);
    if(sourceShape)shadeVesselSourceBoundary(shader,sourceShape);
    if(!style)return;
    Object.assign(shader.uniforms,style);
    shader.vertexShader=`
      uniform highp sampler3D uVesselLabels;
      uniform vec3 uVesselShape;
      uniform mat4 uVesselWorldToVoxel;
      uniform vec2 uVesselLabelCode;
      uniform float uVesselStyle;
      varying float vVesselWidth;
      varying vec3 vVesselWorld;

      bool vesselAt(vec3 point) {
        ivec3 cell=ivec3(floor(point+0.5));
        if(any(lessThan(cell,ivec3(0)))||any(greaterThanEqual(cell,ivec3(uVesselShape))))return false;
        int value=int(round(texelFetch(uVesselLabels,cell.zyx,0)[int(uVesselLabelCode.x)]*255.0));
        int code=int(uVesselLabelCode.y);
        return code<0?(value&(-code))!=0:code>0&&value==code;
      }
      // Low-cost local thickness cue in physical millimetres. A chord sampled
      // inward from the surface is neither a centerline diameter nor a vessel
      // type classification. Stop at the first gap, even in dense networks.
      float vesselSupportWidth(vec3 point, vec3 inward) {
        float width=0.0;
        for(int i=0;i<8;i++){
          float d=0.38+0.62*float(i);
          if(!vesselAt(point+inward*d))break;
          width=d;
        }
        return width;
      }
    `+shader.vertexShader;
    shader.vertexShader=shader.vertexShader.replace('#include <project_vertex>',`
      #include <project_vertex>
      vVesselWidth=0.0;
      vVesselWorld=(modelMatrix*vec4(transformed,1.0)).xyz;
      if(uVesselStyle>0.5){
        vec3 sourcePoint=(uVesselWorldToVoxel*vec4(vVesselWorld,1.0)).xyz;
        vec3 worldInward=-normalize((vec4(transformedNormal,0.0)*viewMatrix).xyz);
        vVesselWidth=vesselSupportWidth(sourcePoint,mat3(uVesselWorldToVoxel)*worldInward);
      }`);
    shader.fragmentShader=`
      uniform float uVesselStyle;
      uniform vec3 uVesselColor;
      varying float vVesselWidth;
      varying vec3 vVesselWorld;
      uniform highp sampler2D uVesselFocusPath;
      uniform int uVesselFocusCount;
      uniform vec3 uVesselFocusMin;
      uniform vec3 uVesselFocusMax;
      uniform float uVesselFocusRadius2;
      bool vesselInspectionContains(vec3 point){
        if(any(lessThan(point,uVesselFocusMin))||any(greaterThan(point,uVesselFocusMax)))return false;
        vec3 a=texelFetch(uVesselFocusPath,ivec2(0,0),0).xyz;
        for(int i=1;i<uVesselFocusCount;i++){
          vec3 b=texelFetch(uVesselFocusPath,ivec2(i,0),0).xyz;
          vec3 d=b-a,offset=point-a;
          vec3 difference=offset-d*clamp(dot(offset,d)/max(dot(d,d),1e-8),0.0,1.0);
          if(dot(difference,difference)<=uVesselFocusRadius2)return true;
          a=b;
        }
        return false;
      }
    `+shader.fragmentShader;
    shader.fragmentShader=shader.fragmentShader.replace('#include <clipping_planes_fragment>',`
      #include <clipping_planes_fragment>
      if(uVesselFocusCount>1&&!vesselInspectionContains(vVesselWorld))discard;
    `);
    shader.fragmentShader=shader.fragmentShader.replace('#include <color_fragment>',`
      #include <color_fragment>
      if(uVesselStyle>0.5){
        float broad=smoothstep(0.9,4.3,vVesselWidth);
        float emphasis=uVesselStyle>2.5?mix(1.1,0.59,broad):uVesselStyle>1.5?mix(0.24,1.08,broad):mix(0.68,1.0,broad);
        diffuseColor.rgb=uVesselColor*emphasis;
      }`);
  };
  material.customProgramCacheKey=()=>`anatomy-vessel-depth-v4-${!!style}-${!!sourceShape}-${previousKey}`;
  material.needsUpdate=true;
}

export function setVesselDepthAppearance(material:THREE.MeshStandardMaterial,settings:ViewerWorkspaceSettings,
  layer:{displayFilter?:VesselDisplayFilter},grid?:VesselLabelGrid) {
  const style=material.userData.vesselStyle as VesselUniforms|undefined;
  if(!style||!grid)return;
  const explicitScope=layer.displayFilter?.scope;
  const mode=settings.operativeView||(explicitScope&&!['exterior-air','rooted','rooted-review'].includes(explicitScope))?'source':
    settings.vesselShading??(settings.presentation==='vascular'?'balanced':'source');
  style.uVesselStyle.value=shadingValue[mode];
  style.uVesselColor.value.copy(material.color);
  style.uVesselLabels.value=grid.texture;
  style.uVesselShape.value.set(...grid.shape);
  style.uVesselWorldToVoxel.value=grid.worldToVoxel;
  style.uVesselLabelCode.value.set(...grid.code);
}
