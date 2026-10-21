import {labelMembershipShader} from './labelPacking';
export const volumeVertexShader = `
out vec2 vUv;
void main(){vUv=uv;gl_Position=vec4(position.xy,0.0,1.0);}
`;
export const volumeFragmentShader = `
precision highp float;
precision highp sampler3D;
in vec2 vUv;
out vec4 outColor;
uniform sampler3D uVolume;
uniform sampler2D uTransfer;
uniform sampler2D uScene;
uniform sampler2D uDepth;
uniform mat4 uWorldToVoxel;
uniform mat4 uCameraWorld;
uniform mat4 uInverseProjection;
uniform vec3 uShape;
uniform vec2 uRange;
uniform float uDensity;
uniform int uSkullMode;
uniform bool uCt;
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
vec3 unproject(vec3 ndc){
  vec4 p=uInverseProjection*vec4(ndc,1.0);p/=p.w;
  return (uCameraWorld*p).xyz;
}
void main(){
  vec4 scene=texture(uScene,vUv);
  scene.rgb+=(1.0-scene.a)*uBackground;
  vec3 origin=uCameraWorld[3].xyz;
  vec3 direction=normalize(unproject(vec3(vUv*2.0-1.0,1.0))-origin);
  vec3 voxelOrigin=(uWorldToVoxel*vec4(origin,1.0)).xyz;
  vec3 voxelDirection=mat3(uWorldToVoxel)*direction;
  float entry=0.0,exit=1e10;
  for(int axis=0;axis<3;axis++){
    if(abs(voxelDirection[axis])<1e-8){
      if(voxelOrigin[axis]<-0.5 || voxelOrigin[axis]>uShape[axis]-0.5){outColor=scene;return;}
    }else{
      float a=(-0.5-voxelOrigin[axis])/voxelDirection[axis];
      float b=(uShape[axis]-0.5-voxelOrigin[axis])/voxelDirection[axis];
      entry=max(entry,min(a,b));exit=min(exit,max(a,b));
    }
  }
  if(uClipEnabled){
    float d=dot(uClipPlane.xyz,origin)+uClipPlane.w;
    float slope=dot(uClipPlane.xyz,direction);
    if(abs(slope)<1e-8){if(d<0.0){outColor=scene;return;}}
    else if(slope>0.0)entry=max(entry,-d/slope);else exit=min(exit,-d/slope);
  }
  for(int i=0;i<7;i++){
    if(i>=uExtraClipCount)break;
    vec4 clip=uExtraClips[i];float d=dot(clip.xyz,origin)+clip.w;float slope=dot(clip.xyz,direction);
    if(abs(slope)<1e-8){if(d<0.0){outColor=scene;return;}}
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
  float stepMm=max(uStep,(exit-entry)/1519.0);
  float jitter=fract(52.9829189*fract(dot(gl_FragCoord.xy,vec2(0.06711056,0.00583715)))+uSampleIndex*0.61803398875);
  float t=entry;
  int boundary=0;
  vec4 accumulated=vec4(0.0);
  ivec4 previousLabels=ivec4(0);
  bool previousSelected=false;
  vec3 light=normalize(vec3(-0.4,-0.5,0.75)-direction*0.7);
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
    vec3 p=voxelOrigin+voxelDirection*(t+(0.25+0.5*jitter)*dt);
    float value=scalar(p);
    vec4 tissue=texture(uTransfer,vec2(clamp((value-uRange.x)/(uRange.y-uRange.x),0.0,1.0),0.5));
    ivec4 packed=ivec4(0);
    if(uActiveLayerCount>0||uContextLayerCount>0)packed=ivec4(round(texture(uLabels,((p+0.5)/uShape).zyx)*255.0));
    bool selected=previousSelected;
    if(any(notEqual(packed,previousLabels))){
      selected=false;
      for(int slot=0;slot<128;slot++){if(slot>=uActiveLayerCount)break;
        int j=int(uActiveLayers[slot]);if(hasLabel(packed,j)){selected=true;break;}
      }
    }
    previousSelected=selected;previousLabels=packed;
    if(uIsolate && !selected)tissue.a=0.0;
    tissue.a*=uDensity;
    // CTA contrast and high-attenuation context overlap in intensity. When a
    // source context label exists, only its exact packed membership receives
    // skull presentation and its configured color. This includes categorical
    // labels above index 31; it never relies on a 32-bit layer visibility mask.
    if(uCt){
      if(uContextLayerCount>0){
        bool contextMember=false;
        vec4 context=vec4(0.0);
        for(int slot=0;slot<128;slot++){
          if(slot>=uContextLayerCount)break;
          int j=int(uContextLayers[slot]);
          if(!hasLabel(packed,j))continue;
          contextMember=true;
          if(uLayerEnabled[j]>0.5&&uLayerColor[j].a>context.a)context=uLayerColor[j];
        }
        if(contextMember){
          tissue.rgb=context.rgb;
          if(context.a<=0.0)tissue.a=0.0;
          else if(!uIsolate||selected){
            if(uSkullMode==1)tissue.a=10000.0;
            else tissue.a*=context.a;
          }
        }
      }else if(value>300.0&&(!uIsolate||selected)){
        // Legacy intensity-only CT has no supplied context separation. Retain
        // its existing threshold behavior rather than inventing a label.
        if(uSkullMode==1)tissue.a=10000.0;
        if(uSkullMode==2)tissue.a*=0.08;
      }
    }
    if(tissue.a>0.0){
      vec3 gradient=vec3(scalar(p+vec3(1,0,0))-scalar(p-vec3(1,0,0)),
        scalar(p+vec3(0,1,0))-scalar(p-vec3(0,1,0)),scalar(p+vec3(0,0,1))-scalar(p-vec3(0,0,1)));
      // Central differences and the affine covector transform give signal/mm,
      // not signal/voxel. Automatic MRI context emphasizes acquired boundaries
      // instead of integrating uniform interior signal into a gray veil.
      vec3 normal=-0.5*transpose(mat3(uWorldToVoxel))*gradient;
      if(uGradientOpacity){
        float boundary=smoothstep(0.01,0.075,length(normal)/max(uRange.y-uRange.x,0.000001));
        tissue.a*=mix(0.06,1.0,boundary);
      }
      float alpha=1.0-exp(-tissue.a*dt);
      float lighting=0.8;
      if(length(normal)>0.0001){
        normal=normalize(normal);
        lighting=0.28+0.66*max(dot(normal,light),0.0)+0.15*max(dot(normal,-direction),0.0);
        lighting+=0.12*pow(max(dot(reflect(-light,normal),-direction),0.0),24.0);
        // A short density probe supplies soft local occlusion in bone creases.
        if(uShadows && alpha>0.0001){
          vec3 ahead=p+mat3(uWorldToVoxel)*light*2.0;
          float obstruction=0.0;
          if(all(greaterThanEqual(ahead,vec3(-.5)))&&all(lessThan(ahead,uShape-.5)))obstruction=texture(uTransfer,vec2(clamp((scalar(ahead)-uRange.x)/(uRange.y-uRange.x),0.0,1.0),0.5)).a;
          float shadow=obstruction*2.0;
          if(!uInteractive)for(int j=1;j<5;j++){
            vec3 probe=p+mat3(uWorldToVoxel)*light*(2.0+float(j)*3.0);
            if(all(greaterThanEqual(probe,vec3(0)))&&all(lessThan(probe,uShape)))
              shadow+=3.0*texture(uTransfer,vec2(clamp((scalar(probe)-uRange.x)/(uRange.y-uRange.x),0.0,1.0),0.5)).a;
          }
          lighting*=0.3+0.7*exp(-shadow*uDensity);
        }
      }
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
