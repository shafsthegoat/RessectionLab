import {labelMembershipShader} from './labelPacking';
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
uniform sampler3D uFusion;
uniform int uFusionActive;
uniform float uFusionOpacity;
uniform vec3 uFusionShape;
uniform mat4 uWorldToFusion;
uniform vec2 uFusionWindow;
uniform sampler3D uLabels;
${labelMembershipShader}
uniform sampler3D uRemoved;
uniform int uReplayActive;
uniform sampler3D uProposal;
uniform int uProposalActive;
uniform sampler3D uDiagnosticState;
uniform sampler3D uDiagnosticCoverage;
uniform int uDiagnosticActive;
uniform vec3 uProposalColor;
uniform sampler3D uPrior;
uniform sampler3D uPriorCoverage;
uniform int uPriorActive;
uniform int uPriorKind;
uniform mat4 uWorldToPrior;
uniform vec3 uPriorShape;
uniform vec3 uPriorVoxelTolerance;
uniform vec3 uPriorColors[3];
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
uniform int uSliceContours;
uniform float uLayerOpacities[128];
uniform int uClipEnabled;
uniform vec4 uClipPlane;
uniform int uThreeD;
uniform int uVisibleBits;
uniform vec3 uColors[128];
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
float fusionScalar(ivec3 index) {
  return texelFetch(uFusion,clamp(index,ivec3(0),ivec3(uFusionShape)-1).zyx,0).r;
}
float fusionSample(vec3 voxel) {
  ivec3 a=ivec3(floor(voxel));vec3 f=fract(voxel);
  float c00=mix(fusionScalar(a),fusionScalar(a+ivec3(1,0,0)),f.x);
  float c01=mix(fusionScalar(a+ivec3(0,0,1)),fusionScalar(a+ivec3(1,0,1)),f.x);
  float c10=mix(fusionScalar(a+ivec3(0,1,0)),fusionScalar(a+ivec3(1,1,0)),f.x);
  float c11=mix(fusionScalar(a+ivec3(0,1,1)),fusionScalar(a+ivec3(1,1,1)),f.x);
  return mix(mix(c00,c10,f.y),mix(c01,c11,f.y),f.z);
}
ivec4 labels(ivec3 voxel) {
  if (any(lessThan(voxel,ivec3(0))) || any(greaterThanEqual(voxel,ivec3(uShape)))) return ivec4(0);
  int effect=int(round(texelFetch(uRemoved,voxel.zyx,0).r*255.0));
  if (uReplayActive==1 && (effect==1 || effect==2)) return ivec4(0);
  return ivec4(round(texelFetch(uLabels,voxel.zyx,0)*255.0));
}
float capsuleDistance(vec3 p,vec3 a,vec3 b) {
  vec3 v=b-a; float along=clamp(dot(p-a,v)/max(dot(v,v),0.00001),0.0,1.0);
  return length(p-a-v*along);
}
bool proposalAt(vec3 voxel) {
  ivec3 index=ivec3(floor(voxel+0.5));
  if(any(lessThan(index,ivec3(0)))||any(greaterThanEqual(index,ivec3(uShape)))) return false;
  return texelFetch(uProposal,index.zyx,0).r>0.0;
}
int diagnosticAt(vec3 voxel) {
  ivec3 index=ivec3(floor(voxel+0.5));
  if(any(lessThan(index,ivec3(0)))||any(greaterThanEqual(index,ivec3(uShape))))return -1;
  if(texelFetch(uDiagnosticCoverage,index.zyx,0).r<=0.0)return -1;
  return texelFetch(uDiagnosticState,index.zyx,0).r>0.0?1:0;
}
bool priorAt(vec3 world,out float result) {
  vec3 voxel=(uWorldToPrior*vec4(world,1.0)).xyz;
  result=0.0;
  // Abstain at numerically ambiguous outer faces instead of extending coverage.
  if(any(lessThanEqual(abs(voxel+0.5),uPriorVoxelTolerance))||any(lessThanEqual(abs(voxel-(uPriorShape-0.5)),uPriorVoxelTolerance)))return false;
  if(any(lessThan(voxel,vec3(-0.5)))||any(greaterThanEqual(voxel,uPriorShape-0.5)))return false;
  float grid=uPriorKind==1?2.0:1.0;
  vec3 anchor=floor(voxel*grid+0.5)/grid;
  voxel=mix(voxel,anchor,lessThanEqual(abs(voxel-anchor),uPriorVoxelTolerance));
  voxel=clamp(voxel,vec3(0.0),uPriorShape-1.0);
  if(uPriorKind==1) {
    ivec3 index=ivec3(floor(voxel+0.5));
    if(texelFetch(uPriorCoverage,index.zyx,0).r<=0.0)return false;
    result=texelFetch(uPrior,index.zyx,0).r;return true;
  }
  ivec3 base=ivec3(floor(voxel));vec3 f=fract(voxel);
  for(int x=0;x<=1;x++)for(int y=0;y<=1;y++)for(int z=0;z<=1;z++) {
    float weight=(x==1?f.x:1.0-f.x)*(y==1?f.y:1.0-f.y)*(z==1?f.z:1.0-f.z);
    if(weight<=0.0)continue;
    ivec3 index=min(base+ivec3(x,y,z),ivec3(uPriorShape)-1);
    if(texelFetch(uPriorCoverage,index.zyx,0).r<=0.0)return false;
    result+=weight*texelFetch(uPrior,index.zyx,0).r;
  }
  return true;
}
void main() {
  vec3 world=vWorld;
  if(uThreeD==1 && uClipEnabled==1 && dot(uClipPlane.xyz,world)+uClipPlane.w<0.0) discard;
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
  if(uFusionActive==1) {
    vec3 fusionVoxel=(uWorldToFusion*vec4(world,1.0)).xyz;
    // Outside the second acquisition's coverage the original image is retained.
    if(all(greaterThanEqual(fusionVoxel,vec3(0.0)))&&all(lessThanEqual(fusionVoxel,uFusionShape-1.0))) {
      float fusionGray=clamp((fusionSample(fusionVoxel)-uFusionWindow.x)/max(uFusionWindow.y-uFusionWindow.x,.000001),0.0,1.0);
      color=mix(color,vec3(fusionGray),uFusionOpacity);
    }
  }
  if(uThreeD==0 && uDiagnosticActive==1 && uReplayActive==0) {
    int candidate=diagnosticAt(voxel);
    if(candidate<0) {
      // Unknown is hatching, never a negative prediction.
      bool hatch=mod(floor(gl_FragCoord.x)+floor(gl_FragCoord.y),15.0)<1.2;
      if(hatch)color=mix(color,vec3(0.58,0.64,0.68),uOverlay*0.42);
    } else if(candidate==1) {
      vec3 stepA=vec3(0.0),stepB=vec3(0.0);
      stepA[uAxes.x]=(uHigh[uAxes.x]-uLow[uAxes.x])/max(uRect.z*uResolution.x,1.0)*1.25;
      stepB[uAxes.y]=(uHigh[uAxes.y]-uLow[uAxes.y])/max(uRect.w*uResolution.y,1.0)*1.25;
      vec3 a=(uWorldToVoxel*vec4(stepA,0.0)).xyz;
      vec3 b=(uWorldToVoxel*vec4(stepB,0.0)).xyz;
      bool edge=diagnosticAt(voxel+a)!=1||diagnosticAt(voxel-a)!=1||
                diagnosticAt(voxel+b)!=1||diagnosticAt(voxel-b)!=1;
      bool dash=mod(floor(gl_FragCoord.x)+floor(gl_FragCoord.y),9.0)<6.0;
      if(edge&&dash)color=mix(color,vec3(0.98,0.48,0.42),uOverlay*0.92);
      else color=mix(color,vec3(0.98,0.48,0.42),uOverlay*0.08);
    }
  }
  if(uThreeD==0 && uPriorActive==1 && uReplayActive==0) {
    float priorValue;
    bool covered=priorAt(world,priorValue);
    if(covered) {
      if(uPriorKind==0 || priorValue>0.5) {
        vec3 tint=priorValue<0.5?mix(uPriorColors[0],uPriorColors[1],priorValue*2.0):mix(uPriorColors[1],uPriorColors[2],priorValue*2.0-1.0);
        color=mix(color,tint,min(uOverlay,0.65));
      }
    } else {
      // Missing atlas support is a separate neutral hatch, never scalar zero.
      bool hatch=mod(floor(gl_FragCoord.x)-floor(gl_FragCoord.y),13.0)<1.2;
      if(hatch)color=mix(color,vec3(0.62,0.66,0.69),0.32);
    }
  }
  // Display-only estimate contour in the physical MPR plane. Neighbours are
  // one screen pixel apart in RAS, transformed into the unchanged source grid.
  // Tumor label colors are composited afterwards and retain their own identity.
  if(uThreeD==0 && uProposalActive==1 && uReplayActive==0 && proposalAt(voxel)) {
    vec3 stepA=vec3(0.0),stepB=vec3(0.0);
    stepA[uAxes.x]=(uHigh[uAxes.x]-uLow[uAxes.x])/max(uRect.z*uResolution.x,1.0)*1.25;
    stepB[uAxes.y]=(uHigh[uAxes.y]-uLow[uAxes.y])/max(uRect.w*uResolution.y,1.0)*1.25;
    vec3 a=(uWorldToVoxel*vec4(stepA,0.0)).xyz;
    vec3 b=(uWorldToVoxel*vec4(stepB,0.0)).xyz;
    bool edge=!proposalAt(voxel+a)||!proposalAt(voxel-a)||!proposalAt(voxel+b)||!proposalAt(voxel-b);
    bool dash=mod(floor(gl_FragCoord.x)+floor(gl_FragCoord.y),9.0)<6.0;
    if(edge && dash) color=mix(color,uProposalColor,0.94);
  }
  ivec3 nearest=ivec3(floor(voxel+0.5));
  ivec4 packed=labels(nearest);
  bool contours=uSliceContours==1 && uThreeD==0;
  ivec4 contourA=packed,contourB=packed,contourC=packed,contourD=packed;
  if(contours){
    // Keep the stroke in the displayed RAS slice plane, including oblique
    // source affines. A screen-space stroke stays thin when the MPR zooms.
    vec3 stepA=vec3(0.0),stepB=vec3(0.0);
    stepA[uAxes.x]=(uHigh[uAxes.x]-uLow[uAxes.x])/max(uRect.z*uResolution.x,1.0)*1.25;
    stepB[uAxes.y]=(uHigh[uAxes.y]-uLow[uAxes.y])/max(uRect.w*uResolution.y,1.0)*1.25;
    vec3 a=(uWorldToVoxel*vec4(stepA,0.0)).xyz;
    vec3 b=(uWorldToVoxel*vec4(stepB,0.0)).xyz;
    contourA=labels(ivec3(floor(voxel+a+0.5)));
    contourB=labels(ivec3(floor(voxel-a+0.5)));
    contourC=labels(ivec3(floor(voxel+b+0.5)));
    contourD=labels(ivec3(floor(voxel-b+0.5)));
  }
  int effect=uReplayActive==1?int(round(texelFetch(uRemoved,nearest.zyx,0).r*255.0)):0;
  bool removed=effect==1 || effect==2;
  for(int i=0;i<128;i++) { if(i>=uLayerCount)break;
    if(removed || uLayerEnabled[i]<0.5 || !hasLabel(packed,i))continue;
    float layerAlpha=clamp(uOverlay*uLayerOpacities[i]/0.66,0.0,1.0);
    if(contours){
      bool interior=hasLabel(contourA,i)&&hasLabel(contourB,i)&&hasLabel(contourC,i)&&hasLabel(contourD,i);
      if(!interior)color=mix(color,uColors[i],min(0.95,layerAlpha*1.65));
      continue;
    }
    bool interior=hasLabel(labels(nearest+ivec3(1,0,0)),i)&&hasLabel(labels(nearest-ivec3(1,0,0)),i)&&hasLabel(labels(nearest+ivec3(0,1,0)),i)&&hasLabel(labels(nearest-ivec3(0,1,0)),i)&&hasLabel(labels(nearest+ivec3(0,0,1)),i)&&hasLabel(labels(nearest-ivec3(0,0,1)),i);
    float edge=!interior?min(0.9,layerAlpha+0.3*uLayerOpacities[i]/0.66):layerAlpha;
    color=mix(color,uColors[i],edge);
  }
  if(effect==1) color=mix(color,vec3(0.66,0.94,0.82),0.70);
  if(effect==2) color=mix(color,vec3(0.72,0.65,0.91),0.58);
  if(effect==3) color=mix(color,vec3(0.94,0.71,0.40),0.66);
  if(effect==4) color=mix(color,vec3(0.93,0.80,0.54),0.36);
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
