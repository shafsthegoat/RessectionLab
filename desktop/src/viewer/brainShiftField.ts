/** Display of the engine's opt-in brain-shift scenario (`brainShiftV1`).
 *
 * The engine moves tissue by an analytic field that it describes compactly in
 * its view: opening frame, decay lengths, current peak and up to 512 lattice
 * samples. The field is
 *
 *   d(p) = peak · 2^(-max(depth(p) - surface, 0) / halfDepth)
 *               · exp(-max(lateral(p) - radius, 0) / lateralDecay)
 *               · taper(p) · gravity,
 *
 * with depth and lateral measured from the dural opening (`surgical_brain_shift.py`,
 * `_Geometry.evaluate`). The taper toward rigid supports and the engine's
 * pull-back away from rigid cells depend on the full label grid, so the renderer
 * evaluates the analytic part exactly and recovers taper·pull-back from the
 * samples (their ratio to the analytic value), interpolated with a Gaussian
 * kernel that falls to zero beyond the sampled domain.
 *
 * Display only: the drawn brain moves; source cells, picking, counts and the
 * linked slices (the preoperative image) are unchanged. */
import * as THREE from 'three';
import type {Affine, Point3} from './coordinates';
import {inverseAffine} from './coordinates.ts';

export interface BrainShiftSample {positionRasMm:Point3;displacementMm:Point3}
export interface BrainShiftViewV1 {
  schema:string;scenario:string;evidenceClass:string;
  declaration:{gravityRasUnit:Point3|string;peakSagMm:number;depthHalfLengthMm:number|string;lateralDecayMm:number;boundaryTaperMm:number;
    settlingSeconds:number;settlingSteps:number;onset:string;limits?:string[]};
  resolved:{depthHalfLengthMm:number;openingCenterRasMm:Point3;openingInwardRas:Point3;openingRadiusMm:number};
  state:{gravityRasUnit:Point3;level:number;settlingSteps:number;peakSagMm:number;onsetSeconds:number;directionSource?:string}|null;
  started:boolean;currentPeakMm:number;latticeSamples:BrainShiftSample[];
  plannedTargetRasMm:Point3|null;shiftedTargetRasMm:Point3|null;
  /** Added by the desktop bridge from the engine's cached field geometry. */
  display?:{regionIds:number[];surfaceDepthMm:number;cellMm:number};
}

export interface AnalyticShift {
  center:Point3;inward:Point3;radius:number;surface:number;halfDepth:number;lateralDecay:number;peak:number;direction:Point3;
}

const dot=(a:readonly number[],b:readonly number[])=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];

/** Analytic parameters of the active field, or null before onset, at level 0 or without the bridge's display geometry. */
export function analyticShift(view:BrainShiftViewV1|undefined|null):AnalyticShift|null {
  if(!view?.started||!view.state||!view.display||!(view.currentPeakMm>0))return null;
  const direction=view.state.gravityRasUnit,inward=view.resolved.openingInwardRas;
  if(!Array.isArray(direction)||direction.length!==3||!direction.every(Number.isFinite))return null;
  return {center:view.resolved.openingCenterRasMm,inward,radius:view.resolved.openingRadiusMm,surface:view.display.surfaceDepthMm,
    halfDepth:view.resolved.depthHalfLengthMm,lateralDecay:view.declaration.lateralDecayMm,peak:view.currentPeakMm,direction};
}

/** Unit-peak analytic amplitude at a RAS point, before the rigid-support taper. */
export function analyticAmplitude(shift:AnalyticShift,point:readonly number[]):number {
  const offset=[point[0]-shift.center[0],point[1]-shift.center[1],point[2]-shift.center[2]];
  const depth=dot(offset,shift.inward);
  const lateral=Math.hypot(offset[0]-depth*shift.inward[0],offset[1]-depth*shift.inward[1],offset[2]-depth*shift.inward[2]);
  return 2**(-Math.max(depth-shift.surface,0)/shift.halfDepth)*Math.exp(-Math.max(lateral-shift.radius,0)/shift.lateralDecay);
}

/** Taper·pull-back of each sample: its displacement over the analytic value, in [0, 1]. */
export function sampleRatios(shift:AnalyticShift,samples:readonly BrainShiftSample[]):number[] {
  return samples.map(s=>{
    const expected=analyticAmplitude(shift,s.positionRasMm)*shift.peak;
    const along=dot(s.displacementMm,shift.direction);
    return expected>1e-9?Math.min(1,Math.max(0,along/expected)):0;
  });
}

/** Typical spacing of the engine's sample lattice (its stride in millimetres). */
export function sampleSpacingMm(samples:readonly BrainShiftSample[]):number {
  let best=Infinity;
  const n=Math.min(samples.length,512);
  for(let i=0;i<n;i++)for(let j=i+1;j<n;j++){
    const a=samples[i].positionRasMm,b=samples[j].positionRasMm,d=Math.hypot(a[0]-b[0],a[1]-b[1],a[2]-b[2]);
    if(d>1e-6&&d<best)best=d;
  }
  return Number.isFinite(best)?best:4;
}

/** Interpolated taper ratio at a point: Gaussian-weighted mean of the sample
 * ratios, fading to zero where fewer samples support the point than one at the
 * lattice spacing would. */
export class TaperInterpolator {
  private readonly sigma:number;private readonly cell:number;private readonly buckets=new Map<string,number[]>();
  private readonly samples:readonly BrainShiftSample[];private readonly ratios:readonly number[];
  constructor(samples:readonly BrainShiftSample[],ratios:readonly number[],spacingMm=sampleSpacingMm(samples)){
    this.samples=samples;this.ratios=ratios;
    this.sigma=Math.max(spacingMm,1e-3);this.cell=2.5*this.sigma;
    samples.forEach((s,i)=>{const key=this.key(s.positionRasMm);(this.buckets.get(key)??this.buckets.set(key,[]).get(key)!).push(i);});
  }
  private key(p:readonly number[]){return p.map(v=>Math.floor(v/this.cell)).join(',');}
  at(point:readonly number[]):number {
    const base=point.map(v=>Math.floor(v/this.cell)),limit=this.cell*this.cell,inv=-.5/(this.sigma*this.sigma);
    let sum=0,weight=0;
    for(let dx=-1;dx<=1;dx++)for(let dy=-1;dy<=1;dy++)for(let dz=-1;dz<=1;dz++){
      const list=this.buckets.get(`${base[0]+dx},${base[1]+dy},${base[2]+dz}`);if(!list)continue;
      for(const i of list){
        const p=this.samples[i].positionRasMm,d2=(p[0]-point[0])**2+(p[1]-point[1])**2+(p[2]-point[2])**2;
        if(d2>limit)continue;const w=Math.exp(d2*inv);sum+=w*this.ratios[i];weight+=w;
      }
    }
    return weight>0?sum/Math.max(weight,Math.exp(-.5)):0;
  }
}

/** Displacement (RAS mm) of the displayed tissue at a point. */
export function shiftAt(shift:AnalyticShift,taper:TaperInterpolator,point:readonly number[]):Point3 {
  const magnitude=shift.peak*analyticAmplitude(shift,point)*taper.at(point);
  return [shift.direction[0]*magnitude,shift.direction[1]*magnitude,shift.direction[2]*magnitude];
}

export interface BrainShiftGrid {
  /** RGBA8, displacement in source voxel units encoded as (d/scale + 1)/2. */
  data:Uint8Array;size:[number,number,number];origin:Point3;step:number;scale:number;maxMm:number;
}

/** Displacement on a coarse lattice in source voxel space, covering the samples
 * with a margin; trilinear texture interpolation reproduces the smooth field. */
export function brainShiftGrid(view:BrainShiftViewV1,affine:Affine,step=2):BrainShiftGrid|null {
  const shift=analyticShift(view);
  if(!shift||!view.latticeSamples.length)return null;
  const samples=view.latticeSamples,ratios=sampleRatios(shift,samples),spacing=sampleSpacingMm(samples);
  const taper=new TaperInterpolator(samples,ratios,spacing),inverse=inverseAffine(affine);
  const toIjk=(p:readonly number[])=>[0,1,2].map(a=>inverse[a][0]*p[0]+inverse[a][1]*p[1]+inverse[a][2]*p[2]+inverse[a][3]);
  const voxelMm=Math.max(...[0,1,2].map(c=>Math.hypot(affine[0][c],affine[1][c],affine[2][c])));
  const margin=Math.ceil((2.5*spacing+shift.peak)/voxelMm);
  const low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];
  for(const s of samples){const ijk=toIjk(s.positionRasMm);for(let a=0;a<3;a++){low[a]=Math.min(low[a],ijk[a]);high[a]=Math.max(high[a],ijk[a]);}}
  const origin=low.map(v=>Math.floor(v)-margin) as Point3;
  const size=high.map((v,a)=>Math.min(160,Math.ceil((Math.ceil(v)+margin-origin[a])/step)+1)) as [number,number,number];
  const count=size[0]*size[1]*size[2],vectors=new Float32Array(count*3);
  // Linear part of the inverse affine maps a RAS displacement to voxel units.
  const toVoxel=(d:Point3)=>[0,1,2].map(a=>inverse[a][0]*d[0]+inverse[a][1]*d[1]+inverse[a][2]*d[2]);
  let scale=0,maxMm=0;
  for(let x=0;x<size[0];x++)for(let y=0;y<size[1];y++)for(let z=0;z<size[2];z++){
    const ijk=[origin[0]+x*step,origin[1]+y*step,origin[2]+z*step];
    const ras=[0,1,2].map(a=>affine[a][0]*ijk[0]+affine[a][1]*ijk[1]+affine[a][2]*ijk[2]+affine[a][3]);
    const d=shiftAt(shift,taper,ras),v=toVoxel(d),index=((x*size[1]+y)*size[2]+z)*3;
    vectors.set(v,index);
    scale=Math.max(scale,Math.abs(v[0]),Math.abs(v[1]),Math.abs(v[2]));maxMm=Math.max(maxMm,Math.hypot(...d));
  }
  if(!(scale>0))return null;
  const data=new Uint8Array(count*4);
  for(let i=0;i<count;i++){
    for(let c=0;c<3;c++)data[i*4+c]=Math.round(255*(vectors[i*3+c]/scale+1)/2);
    data[i*4+3]=255;
  }
  return {data,size,origin,step,scale,maxMm};
}

/** Shared uniforms for every shifted material; one texture per accepted state. */
export class BrainShiftDisplay {
  readonly uniforms={uShiftOn:{value:false},uShiftField:{value:null as THREE.Data3DTexture|null},uShiftOrigin:{value:new THREE.Vector3()},
    uShiftStep:{value:1},uShiftScale:{value:0}};
  private readonly active=this.uniforms.uShiftOn;
  private readonly field=this.uniforms.uShiftField;
  private readonly origin=this.uniforms.uShiftOrigin;
  private readonly step=this.uniforms.uShiftStep;
  private readonly scale=this.uniforms.uShiftScale;
  private readonly installed=new WeakSet<THREE.Material>();
  private key='';
  grid:BrainShiftGrid|null=null;

  /** Rebuild the lattice when the field changes; returns whether anything is displaced. */
  update(view:BrainShiftViewV1|undefined|null,affine:Affine):boolean {
    const key=view?JSON.stringify([view.state,view.currentPeakMm,view.latticeSamples.length,view.latticeSamples[0],view.latticeSamples.at(-1),view.display,affine]):'';
    if(key!==this.key){
      this.key=key;this.field.value?.dispose();this.field.value=null;
      this.grid=view?brainShiftGrid(view,affine):null;
      if(this.grid){
        const [nx,ny,nz]=this.grid.size,texture=new THREE.Data3DTexture(this.grid.data,nz,ny,nx);
        texture.format=THREE.RGBAFormat;texture.type=THREE.UnsignedByteType;texture.unpackAlignment=1;
        texture.minFilter=texture.magFilter=THREE.LinearFilter;texture.needsUpdate=true;
        this.field.value=texture;this.origin.value.set(...this.grid.origin);this.step.value=this.grid.step;this.scale.value=this.grid.scale;
      }
    }
    this.active.value=!!this.field.value;
    return this.active.value;
  }

  /** Displace a source-voxel-space surface and its shadow caster by the field. */
  installMesh(mesh:THREE.Mesh){
    for(const material of Array.isArray(mesh.material)?mesh.material:[mesh.material])this.install(material);
    if(!mesh.customDepthMaterial){
      mesh.customDepthMaterial=new THREE.MeshDepthMaterial({depthPacking:THREE.RGBADepthPacking});
      mesh.userData.brainShiftDepthMaterial=true;
    }
    if(mesh.userData.brainShiftDepthMaterial)this.install(mesh.customDepthMaterial);
  }

  /** Displace a source-voxel-space surface material by the field (vertex stage only). */
  install(material:THREE.Material){
    if(this.installed.has(material))return;
    this.installed.add(material);
    const previous=material.onBeforeCompile,key=material.customProgramCacheKey();
    material.onBeforeCompile=(shader,renderer)=>{
      previous.call(material,shader,renderer);
      Object.assign(shader.uniforms,this.uniforms);
      shader.vertexShader='uniform bool uShiftOn;uniform highp sampler3D uShiftField;uniform vec3 uShiftOrigin;uniform float uShiftStep;uniform float uShiftScale;\n'+
        shader.vertexShader.replace('#include <project_vertex>',`if(uShiftOn){
          vec3 uvw=((transformed-uShiftOrigin)/uShiftStep+.5).zyx/vec3(textureSize(uShiftField,0));
          transformed+=(texture(uShiftField,uvw).xyz*2.-1.)*uShiftScale;
        }
        #include <project_vertex>`);
    };
    material.customProgramCacheKey=()=>key+'/brain-shift-scenario-v1';material.needsUpdate=true;
  }

  dispose(){this.field.value?.dispose();this.field.value=null;this.active.value=false;this.key='';this.grid=null;}
}

/** Readout values: the scenario peak and how far the target centroid moved. */
export function brainShiftSummary(view:BrainShiftViewV1|undefined|null) {
  if(!view)return null;
  const planned=view.plannedTargetRasMm,shifted=view.shiftedTargetRasMm;
  const moved=planned&&shifted?Math.hypot(shifted[0]-planned[0],shifted[1]-planned[1],shifted[2]-planned[2]):null;
  return {started:view.started,peakMm:view.currentPeakMm,declaredPeakMm:view.declaration.peakSagMm,targetMovedMm:moved,
    level:view.state?.level??0,steps:view.state?.settlingSteps??view.declaration.settlingSteps};
}
