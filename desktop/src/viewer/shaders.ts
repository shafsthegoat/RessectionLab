// WebGL2 samples the unresampled source grid. Manual trilinear interpolation
// avoids depending on optional float-linear-texture support on the host GPU.
export const vertexShader = `
out vec2 vUv;
out vec3 vWorld;
void main() {
  vUv = uv;
  vWorld = (modelMatrix * vec4(position, 1.0)).xyz;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

export const fragmentShader = `
precision highp float;
precision highp sampler3D;
in vec2 vUv;
in vec3 vWorld;
out vec4 outColor;
uniform sampler3D uMri;
uniform sampler3D uLabels;
uniform mat4 uWorldToVoxel;
uniform vec3 uShape;
uniform vec3 uLow;
uniform vec3 uHigh;
uniform vec3 uCursor;
uniform ivec3 uAxes;
uniform vec4 uRect;
uniform vec2 uResolution;
uniform vec2 uWindow;
uniform float uOverlay;
uniform int uThreeD;
uniform int uVisibleBits;
uniform vec3 uColors[8];
uniform int uRouteCount;
uniform vec3 uShaftStart[2];
uniform vec3 uShaftEnd[2];
uniform vec3 uTipEnd[2];
uniform vec2 uRadii[2];
uniform vec3 uRouteColors[2];

float scalar(ivec3 index) {
  // NumPy xyz C order has z fastest; the GL texture uses z as its width.
  return texelFetch(uMri, clamp(index, ivec3(0), ivec3(uShape)-1).zyx, 0).r;
}
float trilinear(vec3 voxel) {
  ivec3 a = ivec3(floor(voxel));
  vec3 f = fract(voxel);
  float c00=mix(scalar(a),scalar(a+ivec3(1,0,0)),f.x);
  float c01=mix(scalar(a+ivec3(0,0,1)),scalar(a+ivec3(1,0,1)),f.x);
  float c10=mix(scalar(a+ivec3(0,1,0)),scalar(a+ivec3(1,1,0)),f.x);
  float c11=mix(scalar(a+ivec3(0,1,1)),scalar(a+ivec3(1,1,1)),f.x);
  return mix(mix(c00,c10,f.y),mix(c01,c11,f.y),f.z);
}
int labels(ivec3 voxel) {
  if (any(lessThan(voxel,ivec3(0))) || any(greaterThanEqual(voxel,ivec3(uShape)))) return 0;
  return int(round(texelFetch(uLabels,voxel.zyx,0).r*255.0));
}
float capsuleDistance(vec3 p,vec3 a,vec3 b) {
  vec3 v=b-a; float along=clamp(dot(p-a,v)/max(dot(v,v),0.00001),0.0,1.0);
  return length(p-a-v*along);
}
void main() {
  vec3 world=vWorld;
  vec2 point=(vUv-uRect.xy)/uRect.zw;
  if (uThreeD==0) {
    if (any(lessThan(point,vec2(0.0))) || any(greaterThan(point,vec2(1.0)))) {outColor=vec4(0.019,0.031,0.043,1.0);return;}
    world=uCursor;
    world[uAxes.x]=mix(uLow[uAxes.x],uHigh[uAxes.x],point.x);
    world[uAxes.y]=mix(uLow[uAxes.y],uHigh[uAxes.y],point.y);
  }
  vec3 voxel=(uWorldToVoxel*vec4(world,1.0)).xyz;
  if(any(lessThan(voxel,vec3(-0.0001)))||any(greaterThan(voxel,uShape-vec3(0.9999)))) {
    if(uThreeD==1) discard;
    outColor=vec4(0.019,0.031,0.043,1.0);return;
  }
  voxel=clamp(voxel,vec3(0.0),uShape-1.0);
  float signal=trilinear(voxel);
  if (uThreeD==1 && abs(signal)<0.000001) discard;
  float gray=clamp((signal-uWindow.x)/max(uWindow.y-uWindow.x,0.000001),0.0,1.0);
  vec3 color=vec3(gray);
  ivec3 nearest=ivec3(floor(voxel+0.5));
  int bits=labels(nearest)&uVisibleBits;
  for(int i=0;i<8;i++) if((bits & (1<<i))!=0) {
    int neighbors=labels(nearest+ivec3(1,0,0))&labels(nearest-ivec3(1,0,0))&labels(nearest+ivec3(0,1,0))&labels(nearest-ivec3(0,1,0))&labels(nearest+ivec3(0,0,1))&labels(nearest-ivec3(0,0,1));
    float edge=(neighbors&(1<<i))==0?min(0.9,uOverlay+0.3):uOverlay;
    color=mix(color,uColors[i],edge);
  }
  for(int i=0;i<2;i++) if(i<uRouteCount) {
    float shaft=capsuleDistance(world,uShaftStart[i],uShaftEnd[i])-uRadii[i].x;
    float tip=capsuleDistance(world,uShaftEnd[i],uTipEnd[i])-uRadii[i].y;
    if(min(shaft,tip)<=0.0) color=mix(color,uRouteColors[i],0.82);
  }
  if (uThreeD==0) {
    vec2 cross=vec2((uCursor[uAxes.x]-uLow[uAxes.x])/(uHigh[uAxes.x]-uLow[uAxes.x]),(uCursor[uAxes.y]-uLow[uAxes.y])/(uHigh[uAxes.y]-uLow[uAxes.y]));
    vec2 delta=abs(point-cross)*uRect.zw*uResolution;
    if((delta.x<0.65 && delta.y>5.0)||(delta.y<0.65 && delta.x>5.0)) color=mix(color,vec3(0.50,0.85,0.80),0.70);
  }
  outColor=vec4(color,uThreeD==1?0.86:1.0);
}`;
