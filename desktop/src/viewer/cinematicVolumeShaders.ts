import {CINEMATIC_BRICK} from './cinematicTags.ts';

/** One path-traced sample per pixel in two passes: CINEMATIC_PASS 0 traces
 * the first dense collision and the faint matter in front of it, pass 1
 * lights that collision and composites. Splitting keeps each pass small
 * enough for the GPU to keep many rays in flight. Dense matter (tagged vessel
 * lumen and opaque bone) is delta tracked over the brick majorant grid; faint
 * matter (CT glass interfaces and see-through bone, or MR context) is
 * absorbed at regular shifted steps of its own brick bound. Shadow and bounce
 * rays see dense matter only. CINEMATIC_REST selects cubic B-spline
 * intensity, a shadow ray toward the key light and one short bounce. Moving
 * frames use trilinear intensity, unshadowed key light with ambient fill,
 * fewer faint steps and half the dense extinction. */
export const cinematicFragmentShader = `
#ifndef CINEMATIC_REST
#define CINEMATIC_REST 0
#endif
#ifndef CINEMATIC_PASS
#define CINEMATIC_PASS 0
#endif
#if CINEMATIC_REST == 1
#define PRIMARY_STEPS 640
#define SHADOW_STEPS 256
#define DENSE_SCALE 1.0
#define FAINT_STEPS 2.0
#else
#define PRIMARY_STEPS 160
#define SHADOW_STEPS 0
// Moving frames halve dense extinction: half the null collisions per brick,
// at the cost of slightly softer walls until the view rests.
#define DENSE_SCALE 0.5
#define FAINT_STEPS 1.0
#endif
#define BOUNCE_STEPS 96
// Glass: share of faint extinction left face on, and inside bone away from its surfaces.
#define GLASS_FACE 0.03
#define GLASS_INTERIOR 0.04
precision highp float;
precision highp int;
precision highp sampler3D;
in vec2 vUv;
layout(location=0) out vec4 outColor;
layout(location=1) out vec4 outGuide;
uniform sampler3D uVolume;
uniform sampler3D uTags;
uniform sampler3D uMajorant;
uniform sampler3D uField;
uniform sampler2D uScene;
uniform sampler2D uDepth;
uniform sampler2D uPrevious;
uniform sampler2D uTraceVeil;
uniform sampler2D uTraceHit;
uniform mat4 uWorldToVoxel;
uniform mat4 uCameraWorld;
uniform mat4 uInverseProjection;
uniform vec3 uShape;
uniform ivec3 uBrickShape;
uniform vec2 uResolution;
uniform float uSample;
uniform bool uAccumulate;
uniform bool uJitter;
uniform bool uClipEnabled;
uniform vec4 uClipPlane;
uniform int uExtraClipCount;
uniform vec4 uExtraClips[7];
uniform vec3 uBackground;
uniform int uTagChannels;
uniform int uSingleClass;
uniform vec2 uTagGate;
uniform vec3 uTagFloor;
uniform vec4 uVesselRamp;
uniform float uVesselSigma;
uniform vec3 uVesselColor0;
uniform vec3 uVesselColor1;
uniform vec3 uBoneRamp;
uniform vec3 uBoneColor;
uniform vec4 uVesselFinish;
uniform vec4 uBoneFinish;
uniform vec3 uSubsurface;
uniform vec4 uMarrow;
uniform bool uGlass;
uniform vec4 uFaint;
uniform vec3 uSoftColor;
uniform float uGradientScale;
uniform float uVoxelMm;
uniform vec3 uKeyDirection;
uniform float uKeyCosine;
uniform vec3 uKeyIrradiance;
uniform vec3 uFillDirection;
uniform vec3 uFillRadiance;
uniform vec3 uSkyUp;
uniform vec3 uSkyColor;
uniform vec3 uGroundColor;
uniform float uBounceMm;
uniform float uFaintFloor;
const float PI=3.14159265;

uint rngState;
uint pcg(uint v){uint s=v*747796405u+2891336453u;uint w=((s>>((s>>28u)+4u))^s)*277803737u;return (w>>22u)^w;}
float rnd(){rngState=pcg(rngState);return float(rngState>>8u)*(1.0/16777216.0);}
// Smooth integrals (pixel area, key light disc, bounce hemisphere) take one
// eight-dimensional R-sequence point per sample with a per-pixel rotation, so
// their error falls faster than independent random numbers would allow.
vec3 gRotation;
vec2 stratified(int pair){
  const vec2 alpha[4]=vec2[4](vec2(0.9215993196,0.8493453059),vec2(0.7827560561,0.7213874487),
    vec2(0.6648301820,0.6127070434),vec2(0.5646703943,0.5203998512));
  vec2 rotation=fract(gRotation.xy*float(pair*7+1)+gRotation.z*vec2(0.31,0.71)*float(pair));
  return fract(rotation+uSample*alpha[pair]);
}

float ramp(float lo,float hi,float x){float t=clamp((x-lo)/max(hi-lo,1e-6),0.0,1.0);return t*t*(3.0-2.0*t);}
vec3 voxelUv(vec3 p){return ((p+0.5)/uShape).zyx;}
float linearValue(vec3 p){return textureLod(uVolume,voxelUv(p),0.0).r;}
// Cubic B-spline from eight trilinear fetches. Every fetch lies within two
// voxels of the sample's brick, which the majorant maxima cover.
float cubicValue(vec3 p){
  vec3 i=floor(p),f=p-i,g=1.0-f,f2=f*f,f3=f2*f;
  vec3 w0=g*g*g/6.0,w1=(3.0*f3-6.0*f2+4.0)/6.0,w2=(-3.0*f3+3.0*f2+3.0*f+1.0)/6.0,w3=f3/6.0;
  vec3 g0=w0+w1,g1=w2+w3;
  vec3 h0=i-1.0+w1/g0,h1=i+1.0+w3/g1;
  float a0=g0.x*linearValue(vec3(h0.x,h0.y,h0.z))+g1.x*linearValue(vec3(h1.x,h0.y,h0.z));
  float a1=g0.x*linearValue(vec3(h0.x,h1.y,h0.z))+g1.x*linearValue(vec3(h1.x,h1.y,h0.z));
  float a2=g0.x*linearValue(vec3(h0.x,h0.y,h1.z))+g1.x*linearValue(vec3(h1.x,h0.y,h1.z));
  float a3=g0.x*linearValue(vec3(h0.x,h1.y,h1.z))+g1.x*linearValue(vec3(h1.x,h1.y,h1.z));
  return g0.z*(g0.y*a0+g1.y*a1)+g1.z*(g0.y*a2+g1.y*a3);
}
#if CINEMATIC_REST == 1
#define PRIMARY_VALUE(p) cubicValue(p)
#else
#define PRIMARY_VALUE(p) linearValue(p)
#endif
vec2 tagsAt(vec3 p){
  if(uTagChannels==0)return vec2(0.0);
  vec4 t=textureLod(uTags,voxelUv(p),0.0);
  return uTagChannels==2?t.rg:uSingleClass==0?vec2(t.r,0.0):vec2(0.0,t.r);
}
struct Medium{float dense;float vessel;float bone;vec2 gate;};
// A confident tag core keeps a floor under the intensity ramp, so a lumen
// stays connected where partial volume dims it; outside tags nothing changes.
Medium medium(float v,vec2 tags){
  Medium m;
  m.gate=vec2(ramp(uTagGate.x,uTagGate.y,tags.x),ramp(uTagGate.x,uTagGate.y,tags.y));
  vec2 lumen=max(vec2(ramp(uVesselRamp.x,uVesselRamp.y,v),ramp(uVesselRamp.z,uVesselRamp.w,v)),
    uTagFloor.z*vec2(ramp(uTagFloor.x,uTagFloor.y,tags.x),ramp(uTagFloor.x,uTagFloor.y,tags.y)));
  m.vessel=DENSE_SCALE*uVesselSigma*max(m.gate.x*lumen.x,m.gate.y*lumen.y);
  m.bone=DENSE_SCALE*uBoneRamp.z*ramp(uBoneRamp.x,uBoneRamp.y,v)*(1.0-max(m.gate.x,m.gate.y));
  m.dense=m.vessel+m.bone;
  return m;
}
vec3 vesselColor(vec2 gate){return mix(uVesselColor0,uVesselColor1,gate.y/max(gate.x+gate.y,1e-4));}

// Faint matter at a point. CT glass reads the precomputed field: interfaces
// and see-through bone are nearly clear face on and dense at their rims, so
// a see-through skull reads as a thin shell. Tagged vessels carry none.
struct Faint{float sigma;vec3 normal;float surface;float bone;};
Faint faintAt(vec3 p,vec3 direction,bool tagged){
  Faint f;f.sigma=0.0;f.normal=-direction;f.surface=0.0;f.bone=0.0;
  if(!uGlass){f.sigma=uFaint.x*ramp(uFaint.z,uFaint.w,linearValue(p));return f;}
  vec4 field=textureLod(uField,voxelUv(p),0.0);
  vec3 g=(field.rgb*255.0-127.0)/127.0;
  float length_=length(g),s=min(length_,1.0);
  float soft=uFaint.x*s*max(0.0,1.0-4.0*field.a),bone=uFaint.y*field.a*mix(GLASS_INTERIOR,1.0,s);
  if(soft+bone<=0.0)return f;
  vec2 tags=tagged?tagsAt(p):vec2(0.0);
  float open=1.0-max(ramp(uTagGate.x,uTagGate.y,tags.x),ramp(uTagGate.x,uTagGate.y,tags.y));
  if(s>0.02){f.normal=-g/length_;if(dot(f.normal,direction)>0.0)f.normal=-f.normal;}
  float edge=s>0.02?1.0-abs(dot(f.normal,direction)):0.0;
  f.sigma=(soft+bone)*open*mix(GLASS_FACE,1.0,edge*edge*edge);
  f.surface=s;f.bone=bone/(soft+bone);
  return f;
}

// Retained interval of a world ray: volume cells and every clip half-space.
float gEntryT;
vec4 gEntryPlane;
bool gEntryCut;
vec2 rayInterval(vec3 origin,vec3 direction,vec3 vo,vec3 vd){
  float entry=0.0,exit=1e10;
  gEntryCut=false;
  for(int axis=0;axis<3;axis++){
    if(abs(vd[axis])<1e-8){
      if(vo[axis]<-0.5||vo[axis]>uShape[axis]-0.5)exit=-1.0;
    }else{
      float a=(-0.5-vo[axis])/vd[axis],b=(uShape[axis]-0.5-vo[axis])/vd[axis];
      entry=max(entry,min(a,b));exit=min(exit,max(a,b));
    }
  }
  for(int i=0;i<8;i++){
    vec4 plane;
    if(i==0){if(!uClipEnabled)continue;plane=uClipPlane;}
    else{if(i-1>=uExtraClipCount)break;plane=uExtraClips[i-1];}
    float d=dot(plane.xyz,origin)+plane.w,slope=dot(plane.xyz,direction);
    if(abs(slope)<1e-8){if(d<0.0)exit=-1.0;}
    else if(slope>0.0){
      float t=-d/slope;
      if(t>entry){entry=t;gEntryCut=true;gEntryPlane=plane;}
    }else exit=min(exit,-d/slope);
  }
  gEntryT=entry;
  return vec2(entry,exit);
}

// A walk over bricks that skips cubes of bricks without dense matter. A
// primary walk also reads the faint bound of its brick or skipped cube.
struct Walk{ivec3 cell;ivec3 dir;vec3 next;vec3 delta;vec3 origin;vec3 slope;float dense;float faint;int skip;bool all;};
void fetchBrick(inout Walk w){
  vec4 m=texelFetch(uMajorant,w.cell.zyx,0);
  w.dense=m.x*DENSE_SCALE;w.skip=int(m.y);w.faint=w.all?m.w:0.0;
}
Walk beginWalk(vec3 vo,vec3 vd,float t,bool all){
  Walk w;w.all=all;
  w.origin=(vo+0.5)/${CINEMATIC_BRICK}.0;w.slope=vd/${CINEMATIC_BRICK}.0;
  vec3 db=w.slope=vec3(abs(w.slope.x)<1e-9?1e-9:w.slope.x,abs(w.slope.y)<1e-9?1e-9:w.slope.y,abs(w.slope.z)<1e-9?1e-9:w.slope.z);
  w.cell=clamp(ivec3(floor(w.origin+db*t)),ivec3(0),uBrickShape-1);
  w.dir=ivec3(db.x>0.0?1:-1,db.y>0.0?1:-1,db.z>0.0?1:-1);
  vec3 inv=1.0/db;
  w.next=(vec3(w.cell)+max(vec3(w.dir),0.0)-w.origin)*inv;
  w.delta=abs(inv);
  fetchBrick(w);
  return w;
}
bool walkEmpty(Walk w){return w.dense<=0.0&&w.skip>=1;}
// The end of the current segment: the brick boundary, or the far side of the
// empty cube of bricks around an empty brick.
float walkBoundary(Walk w){
  if(!walkEmpty(w))return min(w.next.x,min(w.next.y,w.next.z));
  vec3 r=vec3(w.skip);
  vec3 bound=vec3(w.cell)+mix(-r,r+1.0,greaterThan(vec3(w.dir),vec3(0.0)));
  vec3 t=(bound-w.origin)/w.slope;
  return min(t.x,min(t.y,t.z));
}
bool walkAdvance(inout Walk w,vec3 vo,vec3 vd,float t){
  if(walkEmpty(w)){w=beginWalk(vo,vd,t+1e-3,w.all);return true;}
  if(w.next.x<=w.next.y&&w.next.x<=w.next.z){w.cell.x+=w.dir.x;w.next.x+=w.delta.x;}
  else if(w.next.y<=w.next.z){w.cell.y+=w.dir.y;w.next.y+=w.delta.y;}
  else{w.cell.z+=w.dir.z;w.next.z+=w.delta.z;}
  if(any(lessThan(w.cell,ivec3(0)))||any(greaterThanEqual(w.cell,uBrickShape)))return false;
  fetchBrick(w);
  return true;
}
// Free paths are drawn as optical depth and spent across bricks, so a brick
// crossing costs no random number or logarithm.
float depth(){return -log(1.0-rnd());}
float eventAt(float t,float tau,float m){return m>1e-7?t+tau/m:1e30;}

// Tetrahedral central difference: four samples instead of six.
vec3 gradientWorld(vec3 p,bool cubic){
  const vec3 a=vec3(0.5,0.5,0.5),b=vec3(0.5,-0.5,-0.5),c=vec3(-0.5,0.5,-0.5),d=vec3(-0.5,-0.5,0.5);
  vec3 g;
  if(cubic)g=a*cubicValue(p+a)+b*cubicValue(p+b)+c*cubicValue(p+c)+d*cubicValue(p+d);
  else g=a*linearValue(p+a)+b*linearValue(p+b)+c*linearValue(p+c)+d*linearValue(p+d);
  return transpose(mat3(uWorldToVoxel))*g;
}
// A vessel wall is shaded by the gradient of whatever makes it opaque there:
// filtered intensity where it clears its ramp, the smoothed tag where the
// floor does. Both are read before their ramps, so the normal stays smooth
// and agrees with the drawn silhouette on thin vessels that intensity alone
// would leave without a usable gradient.
float lumenLevel(vec3 p,int kind,bool cubic){
  float v=cubic?cubicValue(p):linearValue(p);
  vec2 tags=tagsAt(p);
  vec2 edge=kind==0?uVesselRamp.xy:uVesselRamp.zw;
  float tag=kind==0?tags.x:tags.y;
  return max(clamp((v-edge.x)/max(edge.y-edge.x,1e-6),-2.0,3.0),(tag-uTagFloor.x)/max(uTagFloor.y-uTagFloor.x,1e-6));
}
// At rest the lumen stencil spans two voxels: a vessel one or two voxels
// across is then shaded as a round tube rather than by its staircase.
vec3 lumenGradientWorld(vec3 p,int kind,bool cubic){
  float r=cubic?1.0:0.5;
  vec3 a=vec3(r,r,r),b=vec3(r,-r,-r),c=vec3(-r,r,-r),d=vec3(-r,-r,r);
  vec3 g=a*lumenLevel(p+a,kind,cubic)+b*lumenLevel(p+b,kind,cubic)+c*lumenLevel(p+c,kind,cubic)+d*lumenLevel(p+d,kind,cubic);
  return transpose(mat3(uWorldToVoxel))*g*(0.25/(r*r));
}
// Cosine-convolved environment: the unoccluded mean of the bounce estimator.
vec3 ambient(vec3 n){
  vec3 sky=0.5*(uGroundColor+uSkyColor)+(uSkyColor-uGroundColor)*(dot(n,uSkyUp)/3.0);
  float f=0.5+0.5*dot(n,uFillDirection);
  return sky+0.5*uFillRadiance*f*f;
}
vec3 basis(vec3 n){return normalize(abs(n.z)<0.9?cross(n,vec3(0,0,1)):cross(n,vec3(1,0,0)));}
vec3 sampleCone(vec3 axis,float cosMax,vec2 xi){
  float c=1.0-xi.x*(1.0-cosMax),s=sqrt(max(0.0,1.0-c*c)),phi=2.0*PI*xi.y;
  vec3 u=basis(axis),v=cross(axis,u);
  return normalize(u*(s*cos(phi))+v*(s*sin(phi))+axis*c);
}
vec3 sampleCosine(vec3 n,vec2 xi){
  float r1=xi.x,r2=xi.y,r=sqrt(r1);
  vec3 u=basis(n),v=cross(n,u);
  return normalize(u*(r*cos(2.0*PI*r2))+v*(r*sin(2.0*PI*r2))+n*sqrt(max(0.0,1.0-r1)));
}
vec3 sampleSphere(vec2 xi){
  float z=1.0-2.0*xi.x,r=sqrt(max(0.0,1.0-z*z)),phi=2.0*PI*xi.y;
  return vec3(r*cos(phi),r*sin(phi),z);
}
float ggx(vec3 n,vec3 v,vec3 l,float rough){
  vec3 hv=v+l;
  float hl=dot(hv,hv);
  if(hl<1e-12)return 0.0;
  vec3 h=hv*inversesqrt(hl);
  float nh=max(dot(n,h),0.0),nl=max(dot(n,l),1e-4),nv=max(dot(n,v),1e-4);
  float a2=rough*rough*rough*rough,dd=nh*nh*(a2-1.0)+1.0;
  float k=(rough+1.0)*(rough+1.0)/8.0;
  float G=nl/(nl*(1.0-k)+k)*nv/(nv*(1.0-k)+k);
  float F=0.04+0.96*pow(1.0-max(dot(h,v),0.0),5.0);
  return a2/(PI*dd*dd)*G*F/(4.0*nl*nv);
}

// Ratio-tracked transmittance of dense matter, with trilinear intensity.
float transmittance(vec3 world,vec3 direction,float limit){
  vec3 vo=(uWorldToVoxel*vec4(world,1.0)).xyz,vd=mat3(uWorldToVoxel)*direction;
  vec2 range=rayInterval(world,direction,vo,vd);
  float t=max(range.x,0.0),end=min(range.y,limit);
  if(t>=end)return 1.0;
  Walk w=beginWalk(vo,vd,t,false);
  float T=1.0,tau=depth();
  for(int i=0;i<SHADOW_STEPS;i++){
    float m=w.dense,boundary=min(walkBoundary(w),end),next=eventAt(t,tau,m);
    if(next<boundary){
      t=next;vec3 p=vo+vd*t;
      T*=max(0.0,1.0-medium(linearValue(p),tagsAt(p)).dense/m);
      if(T<0.05){if(rnd()<0.5)return 0.0;T*=2.0;}
      tau=depth();
    }else{
      tau=max(0.0,tau-m*(boundary-t));
      if(boundary>=end)break;
      t=boundary;
      if(!walkAdvance(w,vo,vd,t))break;
    }
  }
  return T;
}

// wet: the vessel share of the hit, which takes the wet lumen finish; the rest is bone.
struct Hit{bool found;vec3 world;vec3 albedo;vec3 normal;float surface;float rough;float wet;};
Hit noHit(){return Hit(false,vec3(0.0),vec3(0.0),vec3(0.0,0.0,1.0),0.0,0.5,0.0);}
// Bone albedo follows the acquired density: thin and cancellous bone just
// past the bone ramp is darker and warmer than dense cortex.
vec3 boneAlbedo(float v){return uBoneColor*mix(uMarrow.rgb,vec3(1.0),ramp(uBoneRamp.x,uBoneRamp.y+uMarrow.w,v));}
Hit surfaceAt(vec3 world,vec3 p,vec3 direction,Medium s,float v,bool cubic){
  Hit h=noHit();h.found=true;h.world=world;
  float total=max(s.vessel+s.bone,1e-6);
  h.albedo=(s.vessel*vesselColor(s.gate)+s.bone*boneAlbedo(v))/total;
  h.wet=s.vessel/total;
  h.rough=mix(uBoneFinish.x,uVesselFinish.x,h.wet);
  bool lumen=s.vessel>s.bone;
  vec3 g=lumen?lumenGradientWorld(p,s.gate.y>s.gate.x?1:0,cubic):gradientWorld(p,cubic);
  float magnitude=length(g);
  h.normal=magnitude>1e-6?-g/magnitude:-direction;
  // A normal just past the silhouette is turned edge on, not mirrored: a
  // mirrored one points into the lumen, away from the light, and rims the
  // silhouette with dark dots.
  float away=dot(h.normal,direction);
  if(away>0.0)h.normal=normalize(h.normal-direction*(away+0.02));
  // A lumen level rises by one across its ramp, so its slope per voxel says
  // directly how wall-like the hit is.
  // The two-voxel rest stencil reads a gentler slope across a thin lumen.
  h.surface=lumen?(cubic?ramp(0.08,0.4,magnitude*uVoxelMm):ramp(0.15,0.6,magnitude*uVoxelMm)):ramp(0.12,0.5,magnitude*uGradientScale);
  return h;
}
// Delta-track dense matter for a short bounce ray, trilinear.
Hit bounceHit(vec3 world,vec3 direction,float limit){
  Hit h=noHit();
  vec3 vo=(uWorldToVoxel*vec4(world,1.0)).xyz,vd=mat3(uWorldToVoxel)*direction;
  vec2 range=rayInterval(world,direction,vo,vd);
  float t=max(range.x,0.0),end=min(range.y,limit);
  if(t>=end)return h;
  Walk w=beginWalk(vo,vd,t,false);
  float tau=depth();
  for(int i=0;i<BOUNCE_STEPS;i++){
    float m=w.dense,boundary=min(walkBoundary(w),end),next=eventAt(t,tau,m);
    if(next<boundary){
      t=next;vec3 p=vo+vd*t;
      Medium s=medium(linearValue(p),tagsAt(p));
      if(rnd()*m<s.dense){
        // The bounce point faces back along the ray; no gradient is needed.
        h.found=true;h.world=world+direction*t;h.normal=-direction;
        h.albedo=(s.vessel*vesselColor(s.gate)+s.bone*uBoneColor)/max(s.dense,1e-6);
        return h;
      }
      tau=depth();
    }else{
      tau=max(0.0,tau-m*(boundary-t));
      if(boundary>=end)break;
      t=boundary;
      if(!walkAdvance(w,vo,vd,t))break;
    }
  }
  return h;
}
// The key light is a disc: diffuse light and the shadow ray sample it, while
// its highlights come from the disc centre with roughness widened by the
// disc's angular radius, which removes glint noise at no cost. Vessels add a
// sharper wet coat lobe; bone wraps diffuse light past the terminator with a
// warm tint, a cheap stand-in for light scattered under its surface.
vec3 keyLight(Hit h,vec3 view){
  vec3 l=sampleCone(uKeyDirection,uKeyCosine,stratified(1));
  float nl=dot(h.normal,l),nc=max(dot(h.normal,uKeyDirection),0.0);
  float disc=0.5*sqrt(1.0-uKeyCosine*uKeyCosine)/uKeyCosine;
  float wrap=uBoneFinish.y*(1.0-h.wet),lambert=max(nl,0.0),wrapped=max(nl+wrap,0.0)/(1.0+wrap);
  vec3 diffuse=h.albedo*(lambert+(wrapped-lambert)*uSubsurface)/PI;
  float specular=ggx(h.normal,view,uKeyDirection,sqrt(min(1.0,h.rough*h.rough+disc)))+
    h.wet*uVesselFinish.z*ggx(h.normal,view,uKeyDirection,sqrt(min(1.0,uVesselFinish.y*uVesselFinish.y+0.25*disc)));
  vec3 f=mix(h.albedo*(0.25/PI),diffuse+vec3(specular*nc),h.surface);
  if(max(f.r,max(f.g,f.b))<=0.0)return vec3(0.0);
  // The shadow ray starts off the surface and a little back along the eye
  // ray, which was just crossed: a smoothed normal alone can leave it inside
  // a thin lumen, which then shadows itself in dark seams.
  vec3 origin=h.world+(h.normal*(0.6*h.surface)+view*0.4)*uVoxelMm;
#if CINEMATIC_REST == 1
  return uKeyIrradiance*f*transmittance(origin,l,1e10);
#else
  // Moving frames light without shadow rays; shadows return at rest.
  return uKeyIrradiance*f;
#endif
}
// The studio seen in a surface: Fresnel-weighted, from the cosine-convolved
// environment so it adds no noise. Wet vessels take more of it than bone.
vec3 reflection(Hit h,vec3 view){
  float nv=max(dot(h.normal,view),0.0),F=0.04+0.96*pow(1.0-nv,5.0);
  return mix(uBoneFinish.z,uVesselFinish.w,h.wet)*h.surface*F*ambient(reflect(-view,h.normal));
}
// Faint matter is lit without a shadow ray. Glass adds a sharp key highlight.
vec3 veilRadiance(vec3 p,vec3 direction,Faint f){
  if(!uGlass){
    // MR context: shade with the intensity gradient where there is one.
    vec3 g=gradientWorld(p,false);
    float magnitude=length(g);
    vec3 n=magnitude>1e-6?-g/magnitude:-direction;
    if(dot(n,direction)>0.0)n=-n;
    f.normal=n;f.surface=ramp(0.12,0.5,magnitude*uGradientScale);
  }
  vec3 albedo=mix(uSoftColor,uBoneColor,f.bone);
  float nl=max(dot(f.normal,uKeyDirection),0.0);
  vec3 key=uKeyIrradiance*mix(vec3(0.25/PI),vec3(nl/PI),f.surface);
  vec3 radiance=albedo*(key+mix(0.5*ambient(-direction),ambient(f.normal),f.surface));
  // See-through bone reads as glass: mostly its highlight and rim, with only
  // a share of its diffuse light, so stacked shells do not fog what lies behind.
  if(uGlass)radiance=radiance*mix(1.0,uBoneFinish.w,f.bone)+uKeyIrradiance*(f.surface*ggx(f.normal,-direction,uKeyDirection,0.32)*nl);
  return radiance;
}

struct Ray{vec3 origin;vec3 direction;vec3 vo;vec3 vd;vec2 range;vec3 behind;float limit;};
Ray primaryRay(vec2 pixel){
  Ray r;
  vec2 jitter=uJitter?stratified(0):vec2(0.5);
  vec2 ndc=(floor(pixel)+jitter)/uResolution*2.0-1.0;
  vec4 far=uInverseProjection*vec4(ndc,1.0,1.0);far/=far.w;
  r.origin=uCameraWorld[3].xyz;
  r.direction=normalize((uCameraWorld*far).xyz-r.origin);
  vec4 scene=texture(uScene,vUv);
  r.behind=scene.rgb+(1.0-scene.a)*uBackground;
  float sceneDepth=texture(uDepth,vUv).r;
  r.limit=1e10;
  if(sceneDepth<0.999999){
    vec4 q=uInverseProjection*vec4(vUv*2.0-1.0,sceneDepth*2.0-1.0,1.0);q/=q.w;
    r.limit=dot((uCameraWorld*q).xyz-r.origin,r.direction);
  }
  r.vo=(uWorldToVoxel*vec4(r.origin,1.0)).xyz;r.vd=mat3(uWorldToVoxel)*r.direction;
  r.range=rayInterval(r.origin,r.direction,r.vo,r.vd);
  r.range.y=min(r.range.y,r.limit);
  return r;
}

void main(){
  vec2 pixel=gl_FragCoord.xy;
  rngState=pcg(uint(pixel.x)+pcg(uint(pixel.y)));
  gRotation=vec3(rnd(),rnd(),rnd());
  rngState=pcg(rngState+pcg(uint(uSample)+uint(CINEMATIC_PASS)*2654435769u));
  Ray ray=primaryRay(pixel);
#if CINEMATIC_PASS == 0
  // Trace: the first dense collision and the faint matter in front of it.
  vec3 veil=vec3(0.0);
  float veilWeight=0.0,veilT=1.0,hitT=-1.0,keptT=0.0;
  if(ray.range.x<ray.range.y){
    vec3 vo=ray.vo,vd=ray.vd;
    // Delta-track dense matter and step through faint matter in one brick walk.
    Walk w=beginWalk(vo,vd,ray.range.x,true);
    float t=ray.range.x;
    // Faint matter is absorbed at regular, per-sample shifted steps of its
    // brick majorant's optical depth (FAINT_STEPS per unit, and at least
    // uFaintFloor per mm): a stratified quadrature with far less noise than
    // random flights, and no steps where there is no faint matter.
    // The first free path is stratified across samples too, so a pixel that
    // grazes a surface converges on its coverage as fast as its jitter does.
    vec2 phase=stratified(3);
    float faint=phase.x,tau=-log(1.0-phase.y);
    bool finished=false;
    for(int i=0;i<PRIMARY_STEPS;i++){
      float m=w.dense,mf=w.faint>0.0?max(w.faint*FAINT_STEPS,uFaintFloor):0.0;
      float boundary=min(walkBoundary(w),ray.range.y),dense=eventAt(t,tau,m),next=eventAt(t,faint,mf);
      if(dense<boundary&&dense<=next){
        faint-=mf*(dense-t);
        t=dense;vec3 p=vo+vd*t;
        if(rnd()*m<medium(PRIMARY_VALUE(p),tagsAt(p)).dense){hitT=t;finished=true;break;}
        tau=depth();
      }else if(next<boundary){
        tau=max(0.0,tau-m*(next-t));
        t=next;faint=1.0;
        float sigma=faintAt(vo+vd*t,ray.direction,w.dense>0.0).sigma;
        if(sigma>0.0){
          // A step stands for 1/mf of path, which absorbs 1-exp(-sigma/mf)
          // where sigma is constant over it, so opacity does not depend on
          // the step rate. Keep one scattering event in proportion to its
          // weight and shade it after the walk, scaled by the summed weight:
          // the choice of event adds no bias.
          float r=1.0-exp(-sigma/mf),weight=veilT*r;
          veilWeight+=weight;
          if(rnd()*veilWeight<weight)keptT=t;
          veilT*=1.0-r;
          // What remains behind a veil this opaque is under one display
          // level. Ending here is noise free; roulette would trade that for
          // several levels of speckle on every pixel it touched.
          if(veilT<0.004){veilT=0.0;finished=true;break;}
        }
      }else{
        tau=max(0.0,tau-m*(boundary-t));
        faint=max(0.0,faint-mf*(boundary-t));
        if(boundary>=ray.range.y){finished=true;break;}
        t=boundary;
        if(!walkAdvance(w,vo,vd,t)){finished=true;break;}
      }
    }
    // A walk that runs out of steps ends in absorption, as a path tracer's
    // depth cutoff does: whatever lies beyond darkens the pixel instead of
    // letting the background show through matter that was never traced.
    if(!finished)veilT=0.0;
    if(veilWeight>0.0){
      vec3 p=vo+vd*keptT;
      veil=veilWeight*veilRadiance(p,ray.direction,faintAt(p,ray.direction,true));
    }
  }
  outColor=vec4(veil,veilT);
  outGuide=vec4(hitT,0.0,0.0,0.0);
#else
  // Shade: light the traced collision and composite the faint matter over it.
  vec4 traced=texelFetch(uTraceVeil,ivec2(pixel),0);
  float hitT=texelFetch(uTraceHit,ivec2(pixel),0).r;
  vec3 forward=-normalize(uCameraWorld[2].xyz);
  vec3 color=traced.a>0.0?ray.behind:vec3(0.0);
  vec4 guide=vec4(0.0,0.0,0.0,ray.limit<1e9?ray.limit*dot(ray.direction,forward):1e4);
  if(hitT>=0.0){
    vec3 direction=ray.direction,world=ray.origin+direction*hitT,p=ray.vo+ray.vd*hitT;
    float v=PRIMARY_VALUE(p);
    Medium s=medium(v,tagsAt(p));
    Hit h=surfaceAt(world,p,direction,s,v,CINEMATIC_REST==1);
    // A cut through dense matter is shaded as the flat cut face.
    if(gEntryCut&&hitT-gEntryT<1.5*uVoxelMm){h.normal=-normalize(gEntryPlane.xyz);h.surface=1.0;h.rough=0.7;}
    vec3 view=-direction;
#if CINEMATIC_REST == 1
    vec3 radiance=keyLight(h,view);
    vec3 origin2=world+(h.normal*(0.6*h.surface)+view*0.4)*uVoxelMm;
    vec2 u=stratified(2);
    vec3 bounce=rnd()<h.surface?sampleCosine(h.normal,u):sampleSphere(u);
    Hit b=bounceHit(origin2,bounce,uBounceMm);
    // A short bounce: an open direction returns the studio's cosine-weighted
    // mean, so open surfaces carry no ambient noise; nearby matter returns
    // its own colour under half the ambient light, weighted by closeness so
    // that the estimate is continuous. Crevices darken and adjacent walls
    // tint each other without a second shadow ray.
    vec3 open=mix(ambient(-direction),ambient(h.normal),h.surface);
    float near=b.found?1.0-distance(b.world,origin2)/uBounceMm:0.0;
    // Nearby matter returns its colour under half the studio light and a
    // quarter of an unshadowed key: bone under a vessel lights its underside warmly
    // instead of leaving a dark rim at the silhouette.
    float occluded=near*(2.0-near);
    vec3 nearby=0.5*b.albedo*(ambient(b.normal)+0.25*uKeyIrradiance*max(dot(b.normal,uKeyDirection),0.0)/PI);
    radiance+=h.albedo*mix(open,nearby,occluded)+(1.0-occluded)*reflection(h,view);
#else
    vec3 radiance=keyLight(h,view)+0.8*h.albedo*mix(ambient(view),ambient(h.normal),h.surface)+reflection(h,view);
#endif
    color=radiance;
    guide=vec4(h.normal,hitT*dot(direction,forward));
  }
  color=traced.rgb+traced.a*color;
  // One non-finite sample would poison the running mean for good.
  if(any(isnan(color))||any(isinf(color)))color=vec3(0.0);
  // Clamp rare specular fireflies so the running mean settles promptly.
  float luminance=dot(color,vec3(0.2126,0.7152,0.0722));
  if(luminance>6.0){color*=6.0/luminance;luminance=6.0;}
  // Alpha keeps the running mean of squared luminance for the filter's
  // per-pixel variance.
  vec4 sampled=vec4(color,luminance*luminance);
  if(uAccumulate){
    vec4 previous=texelFetch(uPrevious,ivec2(pixel),0);
    sampled=uSample<0.5?sampled:previous+(sampled-previous)/(uSample+1.0);
  }
  outColor=sampled;
  outGuide=guide;
#endif
}
`;

/** Edge-aware a-trous step guided by first-hit view depth and normal. With
 * uSamples set, the accumulated mean's luminance standard error also stops
 * the filter: differences of a few errors are noise, larger ones are kept,
 * so the filter fades as the mean converges. The last step blends back
 * toward the unfiltered estimate by uBlend. */
export const cinematicDenoiseShader = `
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uColor;
uniform sampler2D uGuide;
uniform sampler2D uOriginal;
uniform int uStep;
uniform float uBlend;
uniform float uSamples;
const vec3 LUMA=vec3(0.2126,0.7152,0.0722);
void main(){
  ivec2 pixel=ivec2(gl_FragCoord.xy),size=textureSize(uColor,0);
  vec4 g=texelFetch(uGuide,pixel,0);
  bool hasNormal=dot(g.xyz,g.xyz)>0.5;
  float depthScale=0.006*max(g.w,1.0)*float(uStep);
  float stop=1e30,centre=dot(texelFetch(uColor,pixel,0).rgb,LUMA);
  if(uSamples>0.0){
    // Standard error of the mean from a 3x3 neighbourhood of its moments.
    float variance=0.0;
    for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++){
      vec4 m=texelFetch(uOriginal,clamp(pixel+ivec2(x,y),ivec2(0),size-1),0);
      float l=dot(m.rgb,LUMA);variance+=max(m.a-l*l,0.0);
    }
    stop=3.0*sqrt(variance/(9.0*uSamples))/sqrt(float(uStep))+1e-4;
  }
  vec3 sum=vec3(0.0);float total=0.0;
  for(int y=-2;y<=2;y++)for(int x=-2;x<=2;x++){
    ivec2 q=clamp(pixel+ivec2(x,y)*uStep,ivec2(0),size-1);
    vec4 gq=texelFetch(uGuide,q,0);
    vec3 c=texelFetch(uColor,q,0).rgb;
    float k=(x==0?0.375:abs(x)==1?0.25:0.0625)*(y==0?0.375:abs(y)==1?0.25:0.0625);
    float w=k*exp(-abs(gq.w-g.w)/depthScale-abs(dot(c,LUMA)-centre)/stop);
    bool other=dot(gq.xyz,gq.xyz)>0.5;
    w*=hasNormal&&other?pow(max(dot(g.xyz,gq.xyz),0.0),uSamples>0.0?8.0:24.0):hasNormal==other?1.0:0.0;
    sum+=c*w;total+=w;
  }
  vec3 filtered=sum/max(total,1e-6);
  outColor=vec4(mix(filtered,texelFetch(uOriginal,pixel,0).rgb,uBlend),1.0);
}
`;

/** The running mean of composited samples: colour and the mean of squared
 * luminance (for the filter's variance), with the sample's guide passed on. */
export const cinematicAccumulateShader = `
precision highp float;
in vec2 vUv;
layout(location=0) out vec4 outColor;
layout(location=1) out vec4 outGuide;
uniform sampler2D uSampleColor;
uniform sampler2D uSampleGuide;
uniform sampler2D uPrevious;
uniform float uSample;
void main(){
  ivec2 pixel=ivec2(gl_FragCoord.xy);
  vec3 color=texelFetch(uSampleColor,pixel,0).rgb;
  float luminance=dot(color,vec3(0.2126,0.7152,0.0722));
  vec4 sampled=vec4(color,luminance*luminance);
  vec4 previous=texelFetch(uPrevious,pixel,0);
  outColor=uSample<0.5?sampled:previous+(sampled-previous)/(uSample+1.0);
  outGuide=texelFetch(uSampleGuide,pixel,0);
}
`;

/** Translucent mesh layers (premultiplied, front to back, as the surface
 * peels leave them) over the traced image wherever a layer's front lies
 * before the image's first dense collision, given as view depth in the guide. */
export const cinematicSurfaceShader = `
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uImage;
uniform sampler2D uGuide;
uniform mat4 uInverseProjection;
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
vec4 layerColor(int i){
  if(i==0)return texture(uSurfaceColor0,vUv);
  if(i==1)return texture(uSurfaceColor1,vUv);
  if(i==2)return texture(uSurfaceColor2,vUv);
  if(i==3)return texture(uSurfaceColor3,vUv);
  return texture(uSurfaceColor4,vUv);
}
float layerDepth(int i){
  if(i==0)return texture(uSurfaceDepth0,vUv).r;
  if(i==1)return texture(uSurfaceDepth1,vUv).r;
  if(i==2)return texture(uSurfaceDepth2,vUv).r;
  if(i==3)return texture(uSurfaceDepth3,vUv).r;
  return texture(uSurfaceDepth4,vUv).r;
}
void main(){
  ivec2 pixel=ivec2(gl_FragCoord.xy);
  vec3 image=texelFetch(uImage,pixel,0).rgb;
  float traced=texelFetch(uGuide,pixel,0).w;
  vec4 front=vec4(0.0);
  for(int i=0;i<5;i++){
    if(i>=uSurfaceCount)break;
    float d=layerDepth(i);
    if(d>=0.999999)continue;
    vec4 q=uInverseProjection*vec4(vUv*2.0-1.0,d*2.0-1.0,1.0);
    if(-q.z/q.w>=traced)continue;
    vec4 c=layerColor(i);
    front.rgb+=(1.0-front.a)*c.rgb;
    front.a+=(1.0-front.a)*c.a;
  }
  outColor=vec4(front.rgb+(1.0-front.a)*image,1.0);
}
`;
