import * as THREE from 'three';
import type {Bounds3,Point3} from './coordinates';
import type {ViewerVesselTrace} from './contracts';
import type {ViewerWorkspaceSettings} from './workspaceSettings';

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

const vesselColors:Record<string,string>={artery:'#b33e32',vein:'#3e688c',sinus:'#4b5a88'};
const shadingValue={source:0,balanced:1,large:2,fine:3};

/** Only radiance changes. Depth, source triangles, masks and picks remain authoritative. */
export function applyVesselDepthMaterial(material:THREE.MeshStandardMaterial,bounds:Bounds3,grid?:VesselLabelGrid) {
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
      float anatomyDepth=(length(vViewPosition)-length((viewMatrix*vec4(uAnatomyCenter,1.0)).xyz))/uAnatomyRadius;
      outgoingLight*=mix(1.08,0.56,smoothstep(-0.65,0.65,anatomyDepth));
      #include <opaque_fragment>`);
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
        float emphasis=uVesselStyle>2.5?mix(1.1,0.59,broad):uVesselStyle>1.5?mix(0.24,1.08,broad):mix(0.53,1.0,broad);
        diffuseColor.rgb=uVesselColor*emphasis;
      }`);
    shader.fragmentShader=shader.fragmentShader.replace(
      'outgoingLight*=mix(1.08,0.56,smoothstep(-0.65,0.65,anatomyDepth));',`
      if(uVesselStyle>0.5){
        // Use camera-axis depth so equally deep branches do not darken just
        // because they lie near an edge of the screen. Keep a soft shoulder
        // at both ends so a close-up stays legible throughout an orbit.
        float physicalDepth=(vViewPosition.z+(viewMatrix*vec4(uAnatomyCenter,1.0)).z)/uAnatomyRadius;
        float farCue=smoothstep(-0.42,0.48,physicalDepth);
        float grazing=1.0-max(0.0,dot(normal,normalize(vViewPosition)));
        outgoingLight*=mix(vec3(1.13,1.08,1.01),vec3(0.47,0.49,0.59),farCue);
        outgoingLight*=1.0-0.36*grazing*grazing;
      }else outgoingLight*=mix(1.08,0.56,smoothstep(-0.65,0.65,anatomyDepth));
    `);
  };
  material.customProgramCacheKey=()=>`anatomy-vessel-depth-v2-${!!style}-${previousKey}`;
  material.needsUpdate=true;
}

export function setVesselDepthAppearance(material:THREE.MeshStandardMaterial,settings:ViewerWorkspaceSettings,
  layer:{name:string;kind?:string;displayFilter?:unknown},grid?:VesselLabelGrid) {
  const style=material.userData.vesselStyle as VesselUniforms|undefined;
  if(!style||!grid)return;
  const mode=settings.operativeView||layer.displayFilter?'source':settings.vesselShading??(settings.presentation==='vascular'?'balanced':'source');
  style.uVesselStyle.value=shadingValue[mode];
  style.uVesselColor.value.set(settings.layerColor?.[layer.name]??vesselColors[layer.kind??'']??'#b64f43');
  style.uVesselLabels.value=grid.texture;
  style.uVesselShape.value.set(...grid.shape);
  style.uVesselWorldToVoxel.value=grid.worldToVoxel;
  style.uVesselLabelCode.value.set(...grid.code);
  if(mode!=='source'){
    material.color.copy(style.uVesselColor.value);
    material.roughness=.4;
  }
}
