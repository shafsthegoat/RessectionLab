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
uniform int uExtraClipCount;
uniform vec4 uExtraClips[7];
uniform bool uShadows;
uniform bool uIsolate;
uniform bool uInteractive;

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
  if(exit<=entry || uDensity<=0.0){outColor=scene;return;}
  // The loop covers the complete interval even for large physical fields.
  float stepMm=max(uStep,(exit-entry)/1535.0);
  float jitter=fract(52.9829189*fract(dot(gl_FragCoord.xy,vec2(0.06711056,0.00583715))));
  float t=entry+(0.25+0.5*jitter)*stepMm;
  vec4 accumulated=vec4(0.0);
  ivec4 previousLabels=ivec4(0);
  vec3 light=normalize(vec3(-0.4,-0.5,0.75)-direction*0.7);
  for(int i=0;i<1536;i++){
    if(t>=exit || accumulated.a>0.995)break;
    vec3 p=voxelOrigin+voxelDirection*t;
    float value=scalar(p);
    vec4 tissue=texture(uTransfer,vec2(clamp((value-uRange.x)/(uRange.y-uRange.x),0.0,1.0),0.5));
    ivec4 packed=ivec4(round(texture(uLabels,((p+0.5)/uShape).zyx)*255.0));
    bool selected=false;vec3 labelColor=vec3(0.0);float extinction=0.0;
    for(int j=0;j<128;j++){if(j>=uLayerCount)break;
      if(uLayerEnabled[j]<0.5 || !hasLabel(packed,j))continue;
      selected=true;
      float opacity=uLayerColor[j].a;
      if(opacity<=0.0 || opacity>=1.0 || hasLabel(previousLabels,j))continue;
      float e=-log(max(0.001,1.0-opacity))/min(stepMm,exit-t);
      labelColor+=uLayerColor[j].rgb*e;extinction+=e;
    }
    previousLabels=packed;
    if(uIsolate && !selected)tissue.a=0.0;
    tissue.a*=uDensity;
    if(extinction>0.0){tissue.rgb=(tissue.rgb*tissue.a+labelColor)/(tissue.a+extinction);tissue.a+=extinction;}
    float alpha=1.0-exp(-tissue.a*min(stepMm,exit-t));
    if(alpha>0.0001){
      vec3 gradient=vec3(scalar(p+vec3(1,0,0))-scalar(p-vec3(1,0,0)),
        scalar(p+vec3(0,1,0))-scalar(p-vec3(0,1,0)),scalar(p+vec3(0,0,1))-scalar(p-vec3(0,0,1)));
      vec3 normal=-transpose(mat3(uWorldToVoxel))*gradient;
      float lighting=0.8;
      if(length(normal)>0.0001){
        normal=normalize(normal);
        lighting=0.32+0.62*max(dot(normal,light),0.0)+0.18*max(dot(normal,-direction),0.0);
        // A short density probe supplies soft local occlusion in bone creases.
        vec3 ahead=p+mat3(uWorldToVoxel)*light*2.0;
        float obstruction=texture(uTransfer,vec2(clamp((scalar(ahead)-uRange.x)/(uRange.y-uRange.x),0.0,1.0),0.5)).a;
        if(uShadows){
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
    t+=stepMm;
  }
  outColor=vec4(accumulated.rgb+(1.0-accumulated.a)*scene.rgb,1.0);
}
`;
