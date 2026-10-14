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
  vec3 light=normalize(vec3(-0.4,-0.5,0.75)-direction*0.7);
  for(int i=0;i<1536;i++){
    if(t>=exit || accumulated.a>0.995)break;
    vec3 p=voxelOrigin+voxelDirection*t;
    float value=scalar(p);
    vec4 tissue=texture(uTransfer,vec2(clamp((value-uRange.x)/(uRange.y-uRange.x),0.0,1.0),0.5));
    float alpha=1.0-exp(-tissue.a*uDensity*min(stepMm,exit-t));
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
        lighting*=exp(-obstruction*uDensity*0.7);
      }
      accumulated.rgb+=(1.0-accumulated.a)*alpha*tissue.rgb*lighting;
      accumulated.a+=(1.0-accumulated.a)*alpha;
    }
    t+=stepMm;
  }
  outColor=vec4(accumulated.rgb+(1.0-accumulated.a)*scene.rgb,1.0);
}
`;
