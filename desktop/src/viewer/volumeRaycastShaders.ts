import {labelMembershipShader} from './labelPacking';
export const volumeVertexShader = `
out vec2 vUv;
void main(){vUv=uv;gl_Position=vec4(position.xy,0.0,1.0);}
`;
export const volumeFragmentShader = `
#ifndef CONTEXT_SHADOWS
#define CONTEXT_SHADOWS 0
#endif
#ifndef DIRECTIONAL_OPTICS
#define DIRECTIONAL_OPTICS 0
#endif
#ifndef SCALAR_ONLY
#define SCALAR_ONLY 0
#endif
precision highp float;
precision highp sampler3D;
in vec2 vUv;
out vec4 outColor;
uniform sampler3D uVolume;
uniform sampler3D uAuxiliary;
uniform bool uUseHeadSupport;
uniform int uContextCoverageLayer;
uniform sampler2D uTransfer;
uniform vec2 uOpticalSize;
uniform vec3 uOpticalGrid;
uniform int uOpticalColumns;
uniform int uOpticalDepthRow;
uniform vec3 uOpticalLight;
uniform bool uOpticalCache;
uniform float uClipOpticalWidth;
uniform float uExtraOpticalWidths[7];
uniform sampler2D uScene;
uniform sampler2D uDepth;
uniform mat4 uWorldToVoxel;
uniform mat4 uLabelWorldToVoxel;
uniform vec3 uLabelShape;
uniform mat4 uCameraWorld;
uniform mat4 uInverseProjection;
uniform vec3 uShape;
uniform vec2 uRange;
uniform float uZeroExtinctionLevel;
uniform float uDensity;
uniform bool uDrawScalar;
uniform int uSkullMode;
uniform bool uAcquiredBoneCut;
uniform bool uCt;
uniform bool uScanPresentation;
uniform bool uGradientOpacity;
uniform float uStep;
uniform bool uLinear;
uniform bool uClipEnabled;
uniform vec4 uClipPlane;
uniform vec3 uBackground;
uniform sampler3D uLabels;
${labelMembershipShader}
uniform int uLabelBits;
uniform int uVisibleBits;
uniform vec4 uLayerColor[128];
uniform int uActiveLayerCount;
uniform float uActiveLayers[128];
uniform int uContextLayerCount;
uniform float uContextLayers[128];
uniform int uExtraClipCount;
uniform vec4 uExtraClips[7];
uniform bool uShadows;
uniform bool uIsolate;
uniform bool uInteractive;
uniform float uSampleIndex;
uniform bool uExport;
uniform vec2 uSampleOrigin;
uniform int uSurfaceCount;
uniform sampler2D uSurfaceColor0;
uniform sampler2D uSurfaceDepth0;
uniform sampler2D uSurfaceColor1;
uniform sampler2D uSurfaceDepth1;
uniform sampler2D uSurfaceColor2;
uniform sampler2D uSurfaceDepth2;
uniform sampler2D uSurfaceColor3;
uniform sampler2D uSurfaceDepth3;
uniform sampler2D uSurfaceColor4;
uniform sampler2D uSurfaceDepth4;
vec4 boundaryColor(int i){
  if(i==0)return texture(uSurfaceColor0,vUv);
  if(i==1)return texture(uSurfaceColor1,vUv);
  if(i==2)return texture(uSurfaceColor2,vUv);
  if(i==3)return texture(uSurfaceColor3,vUv);
  if(i==4)return texture(uSurfaceColor4,vUv);
  return vec4(0.0);
}
float boundaryDepth(int i){
  if(i==0)return texture(uSurfaceDepth0,vUv).r;
  if(i==1)return texture(uSurfaceDepth1,vUv).r;
  if(i==2)return texture(uSurfaceDepth2,vUv).r;
  if(i==3)return texture(uSurfaceDepth3,vUv).r;
  if(i==4)return texture(uSurfaceDepth4,vUv).r;
  return 1.0;
}

float scalar(vec3 p){
  if(uLinear)return texture(uVolume,((p+0.5)/uShape).zyx).r;
  ivec3 a=ivec3(floor(p));vec3 f=fract(p);
  float result=0.0;
  for(int x=0;x<2;x++)for(int y=0;y<2;y++)for(int z=0;z<2;z++){
    ivec3 index=clamp(a+ivec3(x,y,z),ivec3(0),ivec3(uShape)-1);
    vec3 w=mix(vec3(1.0)-f,f,vec3(x,y,z));
    result+=w.x*w.y*w.z*texelFetch(uVolume,index.zyx,0).r;
  }
  return result;
}
float transferLevel(float value){
  return clamp((value-uRange.x)/(uRange.y-uRange.x),0.0,1.0);
}
vec4 transferAtLevel(float level){
  return textureLod(uTransfer,vec2(level,0.5/uOpticalSize.y),0.0);
}
bool supported(vec3 p){
  return !uUseHeadSupport || texture(uAuxiliary,((p+0.5)/uShape).zyx).r>0.0;
}
ivec4 labelsAt(vec3 labelPoint){
  vec3 labelUv=(labelPoint+0.5)/uLabelShape;
  if(all(greaterThanEqual(labelUv,vec3(0.0)))&&all(lessThan(labelUv,vec3(1.0))))
    return ivec4(round(texture(uLabels,labelUv.zyx)*255.0));
  return ivec4(0);
}
bool selectedLabel(ivec4 packed){
  for(int slot=0;slot<128;slot++){
    if(slot>=uActiveLayerCount)break;
    if(hasLabel(packed,int(uActiveLayers[slot])))return true;
  }
  return false;
}
bool requiresLabels(){
  return uIsolate||(uCt&&(!uScanPresentation||uSkullMode==1)&&uContextLayerCount>0&&uContextCoverageLayer<0);
}
vec3 contextTint(vec3 acquired,vec3 color,float value){
  float mineral=smoothstep(180.0,1600.0,value);
  vec3 tint=mix(vec3(0.68,0.53,0.40),vec3(1.13,1.11,1.06),mineral);
  return mix(color*tint,acquired,0.24);
}
float contextCoverage(vec3 p){
  return texture(uAuxiliary,((p+0.5)/uShape).zyx).r;
}
void applyContext(inout vec4 tissue,float value,ivec4 packed,bool selected,vec3 p){
  tissue.a*=uDensity;
  if(uIsolate&&!selected){tissue.a=0.0;return;}
  if(!uCt || (uScanPresentation&&uSkullMode!=1))return;
  if(uContextCoverageLayer>=0){
    int layer=uContextCoverageLayer;
    float coverage=contextCoverage(p);
    vec4 context=uLayerColor[layer];
    float clearWeight=1.0-coverage;
    float skullWeight=coverage*context.a*uLayerEnabled[layer];
    float weight=clearWeight+skullWeight;
    vec3 tint=contextTint(tissue.rgb,context.rgb,value);
    tissue.rgb=(clearWeight*tissue.rgb+skullWeight*tint)/max(weight,0.000001);
    tissue.a*=weight;
    return;
  }
  // Contrast and mineralized tissue overlap in HU. A source context label,
  // when present, alone receives the skull controls and its chosen color.
  if(uContextLayerCount>0){
    vec4 context=vec4(0.0,0.0,0.0,-1.0);
    for(int slot=0;slot<128;slot++){
      if(slot>=uContextLayerCount)break;
      int layer=int(uContextLayers[slot]);
      if(!hasLabel(packed,layer))continue;
      if(uLayerEnabled[layer]>0.5)context=uLayerColor[layer].a>context.a?uLayerColor[layer]:context;
      else context.a=max(context.a,0.0);
    }
    if(context.a>=0.0){
      // Keep the configured hue while letting marrow, trabecular and compact
      // regions receive distinct color from the underlying acquired value.
      tissue.rgb=contextTint(tissue.rgb,context.rgb,value);
      if(context.a<=0.0)tissue.a=0.0;
      else if(uSkullMode==1){
        // In a physical CT section, let mineral signal supply the wall.
        // A categorical context label can include marrow or background.
        tissue.a=uAcquiredBoneCut?max(tissue.a,2.2*smoothstep(160.0,430.0,value)):10000.0;
      }
      else tissue.a*=context.a;
    }
  }else if(value>300.0){
    if(uSkullMode==1)tissue.a=10000.0;
    if(uSkullMode==2)tissue.a*=0.08;
  }
}
vec3 scalarNormal(vec3 p){
  vec3 gradient=vec3(scalar(p+vec3(1,0,0))-scalar(p-vec3(1,0,0)),
    scalar(p+vec3(0,1,0))-scalar(p-vec3(0,1,0)),scalar(p+vec3(0,0,1))-scalar(p-vec3(0,0,1)));
  return -0.5*transpose(mat3(uWorldToVoxel))*gradient;
}
float normalizedGradient(vec3 derivative){
  return length(derivative)/max(uRange.y-uRange.x,0.000001);
}
float boundaryExtinction(float gradient){
  return mix(0.06,1.0,smoothstep(0.01,0.075,gradient));
}
vec4 opticalSlice(int slice,vec2 yz,int row){
  vec2 tile=vec2(float(slice%uOpticalColumns)*uOpticalGrid.z,
    float(row)+float(slice/uOpticalColumns)*uOpticalGrid.y);
  return textureLod(uTransfer,(tile+yz.yx+0.5)/uOpticalSize,0.0);
}
vec4 opticalField(vec3 p,int row){
  vec3 cell=clamp((p+0.5)*uOpticalGrid/uShape-0.5,vec3(0.0),uOpticalGrid-1.0);
  int lo=int(cell.x);
  int hi=min(lo+1,int(uOpticalGrid.x)-1);
  return mix(opticalSlice(lo,cell.yz,row),opticalSlice(hi,cell.yz,row),fract(cell.x));
}
vec2 activeOpticalChannel(vec4 density){
#if CONTEXT_SHADOWS > 0
  // Average total and bone-associated extinction independently. The
  // transparent-layer opacity can then change without rebuilding the atlas.
  float opacity=uLayerColor[uContextCoverageLayer].a;
  return max(vec2(0.0),density.rg-(1.0-opacity)*density.ba);
#else
  return uGradientOpacity?density.ba:density.rg;
#endif
}
float cachedExtinction(vec3 p,float footprint){
  vec2 scale=activeOpticalChannel(opticalField(p,1));
  float extinction=mix(scale.x,scale.y,footprint)*uDensity;
#if CONTEXT_SHADOWS > 0
  return min(extinction,1.5);
#else
  return extinction;
#endif
}
float keyVisibility(vec3 p){
  vec2 depth=activeOpticalChannel(opticalField(p,uOpticalDepthRow))*uDensity;
  // Average the visibility of the two emitter directions. Averaging depth
  // before exponentiation would make penumbras systematically too dark.
  return dot(exp(-depth),vec2(0.5));
}
float opticalProbe(vec3 p,vec3 world,float footprint){
  if(any(lessThan(p,vec3(-0.5)))||any(greaterThanEqual(p,uShape-0.5)))return 0.0;
#if CONTEXT_SHADOWS == 1 || DIRECTIONAL_OPTICS == 1
  // Both variants have a ready atlas and no active clipping half-spaces.
  return cachedExtinction(p,footprint);
#else
  // Test the same retained half-spaces as the eye ray. Ease the wider cache
  // footprint as it approaches a cut so removed tissue cannot darken it.
  float clearance=1e6;
#if CONTEXT_SHADOWS == 2
  float cellClearance=1e6;
#endif
  if(uClipEnabled){
    clearance=dot(uClipPlane.xyz,world)+uClipPlane.w;
#if CONTEXT_SHADOWS == 2
    cellClearance=clearance/max(uClipOpticalWidth,1e-6);
#endif
  }
  for(int i=0;i<7;i++){
    if(i>=uExtraClipCount)break;
    vec4 plane=uExtraClips[i];
    float distance=dot(plane.xyz,world)+plane.w;
    clearance=min(clearance,distance);
#if CONTEXT_SHADOWS == 2
    cellClearance=min(cellClearance,distance/max(uExtraOpticalWidths[i],1e-6));
#endif
  }
  if(clearance<0.0)return 0.0;
#if CONTEXT_SHADOWS == 2
  // Trilinear interpolation can reach 1.5 cells from the probe. A nonzero
  // coarse contribution adds one blur cell. Compare with the conservative
  // projection onto each clipping normal before using the prefiltered field.
  if(cellClearance>=(footprint>0.0?2.5:1.5))return cachedExtinction(p,footprint);
#else
  if(uOpticalCache)return cachedExtinction(p,footprint)*smoothstep(0.0,mix(2.0,7.0,footprint),clearance);
#endif
  // Label-isolated and explicit skull modes require per-probe membership.
  // Do not use an unmasked density field to cast shadows from hidden anatomy.
  if(!supported(p))return 0.0;
  float value=scalar(p);
  float level=transferLevel(value);
#if SCALAR_ONLY == 1
  if(level<uZeroExtinctionLevel)return 0.0;
  vec4 tissue=transferAtLevel(level);
  tissue.a*=uDensity;
#else
  vec4 tissue=transferAtLevel(level);
  ivec4 packed=ivec4(0);
  if(requiresLabels())packed=labelsAt((uLabelWorldToVoxel*vec4(world,1.0)).xyz);
  applyContext(tissue,value,packed,!uIsolate||selectedLabel(packed),p);
#endif
  if(uGradientOpacity&&tissue.a>0.0)tissue.a*=boundaryExtinction(normalizedGradient(scalarNormal(p)));
  return min(tissue.a,1.5);
#endif
}
float opticalSegment(vec3 p,vec3 world,vec3 offset,float width,float footprint){
  return width*opticalProbe(p+mat3(uWorldToVoxel)*offset,world+offset,footprint);
}
vec2 volumeIllumination(vec3 p,vec3 world,vec3 light,vec3 ambient){
#if DIRECTIONAL_OPTICS == 1
  float key=keyVisibility(p);
  if(uInteractive){
    float ambientDepth=opticalSegment(p,world,ambient*4.5,4.0,0.0);
    return vec2(key,exp(-ambientDepth));
  }
  vec3 right=uCameraWorld[0].xyz;
  vec3 tangent=normalize(cross(ambient,abs(dot(right,ambient))<0.9?right:uCameraWorld[1].xyz));
  vec3 left=normalize(ambient-tangent*0.52);
  right=normalize(ambient+tangent*0.52);
  float leftDepth=opticalSegment(p,world,left*4.8,4.5,0.0);
  float rightDepth=opticalSegment(p,world,right*4.8,4.5,0.0);
  // Near rays sample distinct parts of the hemisphere. The shared wide
  // cone bounds the number of lookups for longer-range ambient shading.
  float broadDepth=opticalSegment(p,world,ambient*12.0,8.0,0.8);
  return vec2(key,0.5*(exp(-leftDepth)+exp(-rightDepth))*exp(-broadDepth));
#else
#if CONTEXT_SHADOWS > 0
  if(!uInteractive){
    float keyDepth=opticalSegment(p,world,light*4.75,6.5,0.0);
    keyDepth+=opticalSegment(p,world,light*14.0,12.0,0.38);
    keyDepth+=opticalSegment(p,world,light*29.5,19.0,1.0);
    vec3 right=uCameraWorld[0].xyz;
    vec3 lateral=(right-ambient*dot(right,ambient))*2.8;
    float leftDepth=opticalSegment(p,world,ambient*4.8-lateral,4.0,0.0);
    float rightDepth=opticalSegment(p,world,ambient*4.8+lateral,4.0,0.0);
    float broadDepth=opticalSegment(p,world,ambient*12.0,8.0,1.0);
    // Average the visibility of two near-field directions. A longer shared
    // cone gives macroscopic depth without averaging the visible scalar ray.
    float ambientVisibility=0.5*(exp(-leftDepth)+exp(-rightDepth))*exp(-broadDepth);
    return vec2(exp(-keyDepth),ambientVisibility);
  }
#endif
  float keyDepth=opticalSegment(p,world,light*4.0,5.0,0.0);
  float ambientDepth=opticalSegment(p,world,ambient*4.5,4.0,0.16);
  if(!uInteractive){
    keyDepth+=opticalSegment(p,world,light*11.0,9.0,0.42);
    keyDepth+=opticalSegment(p,world,light*24.0,16.0,1.0);
    ambientDepth+=opticalSegment(p,world,ambient*12.0,8.0,1.0);
  }
  return exp(-vec2(keyDepth,ambientDepth));
#endif
}
vec3 unproject(vec3 ndc){
  vec4 p=uInverseProjection*vec4(ndc,1.0);p/=p.w;
  return (uCameraWorld*p).xyz;
}
void main(){
  vec4 scene=texture(uScene,vUv);
  scene.rgb+=(1.0-scene.a)*uBackground;
  if(!uDrawScalar){
    vec4 layers=vec4(0.);
    for(int i=0;i<5;i++){
      if(i>=uSurfaceCount)break;
      layers+=(1.-layers.a)*boundaryColor(i);
    }
    outColor=vec4(layers.rgb+(1.-layers.a)*scene.rgb,1.);
    return;
  }
  vec3 origin=uCameraWorld[3].xyz;
  vec3 direction=normalize(unproject(vec3(vUv*2.0-1.0,1.0))-origin);
  vec3 voxelOrigin=(uWorldToVoxel*vec4(origin,1.0)).xyz;
  vec3 voxelDirection=mat3(uWorldToVoxel)*direction;
  vec3 labelOrigin=(uLabelWorldToVoxel*vec4(origin,1.0)).xyz;
  vec3 labelDirection=mat3(uLabelWorldToVoxel)*direction;
  float entry=0.0,exit=1e10;
  for(int axis=0;axis<3;axis++){
    if(abs(voxelDirection[axis])<1e-8){
      if(voxelOrigin[axis]<-0.5 || voxelOrigin[axis]>uShape[axis]-0.5){exit=-1.0;}
    }else{
      float a=(-0.5-voxelOrigin[axis])/voxelDirection[axis];
      float b=(uShape[axis]-0.5-voxelOrigin[axis])/voxelDirection[axis];
      entry=max(entry,min(a,b));exit=min(exit,max(a,b));
    }
  }
  if(uClipEnabled){
    float d=dot(uClipPlane.xyz,origin)+uClipPlane.w;
    float slope=dot(uClipPlane.xyz,direction);
    if(abs(slope)<1e-8){if(d<0.0){exit=-1.0;}}
    else if(slope>0.0)entry=max(entry,-d/slope);else exit=min(exit,-d/slope);
  }
  for(int i=0;i<7;i++){
    if(i>=uExtraClipCount)break;
    vec4 clip=uExtraClips[i];float d=dot(clip.xyz,origin)+clip.w;float slope=dot(clip.xyz,direction);
    if(abs(slope)<1e-8){if(d<0.0){exit=-1.0;}}
    else if(slope>0.0)entry=max(entry,-d/slope);else exit=min(exit,-d/slope);
  }
  float depth=texture(uDepth,vUv).r;
  if(depth<0.999999){
    vec3 opaque=unproject(vec3(vUv*2.0-1.0,depth*2.0-1.0));
    exit=min(exit,dot(opaque-origin,direction));
  }
  // Surface boundaries outside the scalar field still composite correctly.
  float surfaceT[5];
  for(int j=0;j<5;j++){
    float d=j<uSurfaceCount?boundaryDepth(j):1.0;
    surfaceT[j]=d<0.999999?dot(unproject(vec3(vUv*2.0-1.0,d*2.0-1.0))-origin,direction):1e10;
  }
  // Reserve loop slots for split steps and mesh boundaries, including large fields.
  float stepMm=max(uStep,(exit-entry)/(uScanPresentation&&!uExport?639.0:1519.0));
  float jitter=fract(52.9829189*fract(dot(gl_FragCoord.xy+uSampleOrigin,vec2(0.06711056,0.00583715)))+uSampleIndex*0.61803398875);
  float t=entry;
  int boundary=0;
  vec4 accumulated=vec4(0.0);
#if SCALAR_ONLY == 0
  ivec4 previousLabels=ivec4(0);
  bool previousSelected=false;
  bool useLabels=requiresLabels();
#endif
#if DIRECTIONAL_OPTICS == 1
  vec3 light=uOpticalLight;
#else
  // The eye-relative fill keeps the clipped and unsupported paths legible
  // without using shadows prefiltered across a cut or hidden context.
  vec3 light=uScanPresentation?normalize(mat3(uCameraWorld)*vec3(-0.48,0.62,0.72)):
    normalize(vec3(-0.4,-0.5,0.75)-direction*0.7);
#endif
  vec3 sky=normalize(light-0.65*direction);
  for(int i=0;i<1536;i++){
    if(accumulated.a>0.995)break;
    if(boundary<uSurfaceCount && surfaceT[boundary]<=t+0.000001){
      vec4 color=boundaryColor(boundary);
      accumulated.rgb+=(1.0-accumulated.a)*color.rgb;
      accumulated.a+=(1.0-accumulated.a)*color.a;
      boundary++;continue;
    }
    if(t>=exit)break;
    float dt=min(stepMm,exit-t);
    if(boundary<uSurfaceCount)dt=min(dt,max(0.0,surfaceT[boundary]-t));
    float sampleT=t+(uExport?jitter:0.25+0.5*jitter)*dt;
    vec3 p=voxelOrigin+voxelDirection*sampleT;
    // Source support lives on the selected scalar acquisition's grid.
    if(!supported(p)){
      t+=dt;continue;
    }
    float value=scalar(p);
    float level=transferLevel(value);
#if SCALAR_ONLY == 1
    // No label can add extinction on an ordinary non-CT scalar ray.
    if(level<uZeroExtinctionLevel){
      t+=dt;continue;
    }
    vec4 tissue=transferAtLevel(level);
    tissue.a*=uDensity;
#else
    vec4 tissue=transferAtLevel(level);
    ivec4 packed=ivec4(0);
    if(useLabels)packed=labelsAt(labelOrigin+labelDirection*sampleT);
    bool selected=previousSelected;
    if(uIsolate&&any(notEqual(packed,previousLabels)))selected=selectedLabel(packed);
    previousSelected=selected;previousLabels=packed;
    applyContext(tissue,value,packed,selected,p);
#endif
    if(tissue.a>0.0){
      // Central differences and the affine covector transform give signal/mm,
      // not signal/voxel. Automatic MRI context emphasizes acquired boundaries
      // instead of integrating uniform interior signal into a gray veil.
      vec3 derivative=scalarNormal(p);
      float gradient=normalizedGradient(derivative);
      if(uGradientOpacity)tissue.a*=boundaryExtinction(gradient);
      float alpha=1.0-exp(-tissue.a*dt);
      float confidence=uCt?smoothstep(0.008,0.085,gradient):smoothstep(0.01,0.12,gradient);
      vec3 normal=derivative/max(length(derivative),0.000001);
      float facing=dot(normal,-direction);
      float diffuse=max(dot(normal,light),0.0);
      float glint=pow(max(dot(reflect(-light,normal),-direction),0.0),28.0);
      float key=mix(0.33,0.68*diffuse+0.08*glint,confidence);
      float fill=mix(0.30,0.24+0.12*abs(facing),confidence);
      vec2 visibility=vec2(1.0);
      if(uShadows&&alpha>0.0001){
        // Weak gradients use coherent camera-relative lighting; reliable
        // gradients steer the ambient probe out of the boundary.
        vec3 ambient=normalize(sky+normal*confidence*0.80);
        visibility=volumeIllumination(p,origin+direction*sampleT,light,ambient);
      }
      // A bounded colored fill approximates light scattered within tissue.
      // It modifies radiance only; Beer–Lambert visibility still uses the
      // authored per-mm transfer extinction and exact world-space step.
      float scatter=uCt?mix(0.19,0.09,smoothstep(200.0,1300.0,value)):0.18;
      float scattered=scatter*(1.0-visibility.x)*(0.4+0.6*visibility.y);
      float ambientFloor=uCt?0.46:0.60;
      vec3 lighting=vec3(key*(0.15+0.85*visibility.x)+fill*mix(ambientFloor,1.0,visibility.y));
      lighting+=scattered*vec3(1.0,0.89,0.76);
      accumulated.rgb+=(1.0-accumulated.a)*alpha*tissue.rgb*lighting;
      accumulated.a+=(1.0-accumulated.a)*alpha;
    }
    t+=dt;
  }
  for(int j=0;j<5;j++){
    if(j<boundary||j>=uSurfaceCount||surfaceT[j]>=1e9)continue;
    vec4 color=boundaryColor(j);accumulated.rgb+=(1.0-accumulated.a)*color.rgb;
    accumulated.a+=(1.0-accumulated.a)*color.a;
  }
  outColor=vec4(accumulated.rgb+(1.0-accumulated.a)*scene.rgb,1.0);
}
`;
