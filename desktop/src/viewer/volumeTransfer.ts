import type { Affine, Point3 } from './coordinates';
import { inverseAffine, rasAffine } from './coordinates.ts';

export type VolumePreset = 'auto' | 'ct-bone' | 'ct-tissue' | 'ct-angiography' | 'mri';
export interface VolumeRenderingSettings { enabled?: boolean; preset?: VolumePreset; density?: number; window?: [number,number]; shadows?: boolean; isolateLabels?: boolean; showCtExteriorSignal?: boolean; showCtBoneOverlay?: boolean; balanceTofContext?: boolean; points?: Array<[number,number,number,number,number]> }
export const VOLUME_TRANSFER_WIDTH = 1024;
/** One texel inside the zero-extinction prefix of the linearly filtered LUT. */
export function zeroExtinctionLevel(data:Float32Array) {
  let first=0;
  while(first<VOLUME_TRANSFER_WIDTH&&data[4*first+3]===0)first++;
  return Math.max(0,(first-1.5)/VOLUME_TRANSFER_WIDTH);
}
export function intensityWindow(values: Float32Array): [number, number] {
  const sampled:number[]=[];
  const stride=Math.max(1,Math.floor(values.length/16000));
  for(let i=0;i<values.length;i+=stride)
    if(Number.isFinite(values[i])&&values[i]!==0)sampled.push(values[i]);
  sampled.sort((a,b)=>a-b);
  if(!sampled.length)return [0,1];
  const low=sampled[Math.floor(sampled.length*.005)];
  const high=sampled[Math.min(sampled.length-1,Math.floor(sampled.length*.995))];
  return [low,Math.max(low+1e-6,high)];
}

export const isCtModality=(modality:string)=>/^\s*(CT|CTA)(\b|[-_\s])/i.test(modality);
export const isCtAngiography=(modality:string)=>/^\s*(?:CTA\b|CT[-_\s]+angiogra(?:phy|m)\b)/i.test(modality)&&
  !/segmentation only|analytic signal|generated image signal|diagnostic/i.test(modality);
export function usesAcquiredCtBone(input:VolumeRenderingSettings|undefined,modality:string) {
  if(!isCtModality(modality)||input?.showCtBoneOverlay!==false||input.isolateLabels)return false;
  const {enabled,density}=volumeSettings(input,modality);
  return enabled&&density>0;
}
const MR_ANGIOGRAPHY=/\b(?:MRA|TOF)\b|\bMR(?:I)?[-_\s]+angiogra(?:phy|m)\b/i;
export const isMrAngiography=(modality:string)=>MR_ANGIOGRAPHY.test(modality.replaceAll('_',' '))&&
  !isCtModality(modality)&&!/segmentation only|analytic signal|generated image signal|diagnostic/i.test(modality);

export function canBalanceTofContext(input:VolumeRenderingSettings|undefined, modality:string) {
  return isMrAngiography(modality)&&(input?.preset===undefined||input.preset==='auto')&&
    !input?.isolateLabels&&input?.window===undefined&&input?.points===undefined;
}

export interface VolumePresetOption { value: VolumePreset; label: string; disabled: boolean; title: string }
/** Task-named choices over the existing transfer functions; stored preset ids
 * are unchanged. HU presets are offered only for CT. On MR angiography the
 * sequence-aware MR transfer is its bright-blood (vessel) presentation. */
export function volumePresetOptions(modality: string): VolumePresetOption[] {
  const ct=isCtModality(modality), brightBlood=!ct&&MR_ANGIOGRAPHY.test(modality);
  const ctOnly='Uses CT Hounsfield units; available for CT and CTA.';
  return [
    {value:'auto',label:'Automatic',disabled:false,title:`Choose from the acquisition (${modality})`},
    brightBlood
      ? {value:'mri',label:'Vessels',disabled:false,title:'Bright-blood MR signal. Bright voxels are not vessel classifications.'}
      : {value:'ct-angiography',label:'Vessels',disabled:!ct,title:ct?'Contrast CT: iodine and mineral emphasis':ctOnly},
    {value:'ct-bone',label:'Bone',disabled:!ct,title:ct?'CT bone':ctOnly},
    {value:'ct-tissue',label:'Soft tissue',disabled:!ct,title:ct?'CT soft tissue':ctOnly},
    ...(brightBlood?[]:[{value:'mri' as const,label:'Brain',disabled:false,
      title:ct?'Intensity-windowed tissue boundaries; not a calibrated CT brain window':'MR tissue signal'}]),
  ];
}
/** The option shown for a stored preset. A CT preset saved on MR signal is
 * rendered with the MR transfer (see volumeSettings), so it shows that option. */
export function volumePresetChoice(input: VolumeRenderingSettings|undefined, modality: string): VolumePreset {
  const preset=input?.preset;
  if(!preset||preset==='auto')return 'auto';
  return preset.startsWith('ct-')&&!isCtModality(modality)?'mri':preset;
}

export function volumeSettings(input: VolumeRenderingSettings = {}, modality = 'MRI') {
  const ct=isCtModality(modality), acquired=!/segmentation only|analytic signal|generated image signal/i.test(modality);
  const explicit=input.preset && ['ct-bone','ct-tissue','ct-angiography','mri'].includes(input.preset)?input.preset:null;
  // Stored presets stay readable, but HU thresholds must never be applied to
  // arbitrary MRI signal (including a saved CT preset opened on a T1CE scan).
  const safeExplicit=explicit?.startsWith('ct-')&&!ct?'mri':explicit;
  const preset = safeExplicit??(ct?(isCtAngiography(modality)?'ct-angiography':'ct-bone'):'mri');
  return { enabled: acquired && input.enabled !== false, preset: preset as Exclude<VolumePreset,'auto'>,
    density: Number.isFinite(input.density) ? Math.max(0,Math.min(2,input.density!)) : .45 };
}

/** Display LUT only. Alpha stores optical extinction per millimetre, not a
 * segmentation probability. Colors are linear RGB for physical compositing. */
export function volumeTransfer(preset: Exclude<VolumePreset,'auto'>, window: [number,number], custom?: VolumeRenderingSettings,modality='MRI') {
  type Knot = [number, number, number, number, number];
  let knots: Knot[] = preset === 'ct-bone' ? [
    [0, .52,.30,.18,0], [150,.65,.45,.28,0], [230,.75,.57,.38,.04],
    [400,.91,.78,.56,.27], [900,.98,.91,.76,.65], [1800,1,.98,.91,.95], [3000,1,1,.99,1],
  ] : preset === 'ct-angiography' ? [
    // CTA signal can be iodine or mineral. A low opacity shoulder limits
    // soft-tissue haze. The warmer mid-HU and cooler high-HU shades reveal
    // intensity variation while remaining distinct from vessel label colors.
    [-1000,.08,.09,.10,0], [110,.38,.36,.35,0], [170,.53,.48,.43,.008],
    [235,.67,.60,.51,.073], [365,.82,.75,.62,.33], [620,.90,.85,.76,.49],
    [1000,.95,.93,.87,.62], [1800,.96,.97,.98,.85], [3000,.99,1,1,1],
  ] : preset === 'ct-tissue' ? [
    [-1000,.15,.10,.09,0], [-500,.42,.20,.13,0], [-120,.62,.38,.24,.004],
    [20,.73,.43,.28,.045], [90,.81,.55,.37,.07], [180,.84,.60,.42,.045],
    [450,.93,.80,.62,.25], [1800,1,.98,.91,.9], [3000,1,1,.99,1],
  ] : [
    [0,.05,.06,.07,0], [.08,.16,.18,.20,0], [.2,.30,.33,.36,.008],
    [.4,.49,.51,.53,.025], [.65,.72,.73,.73,.07], [.85,.91,.90,.87,.16], [1,1,.98,.94,.28],
  ];
  if(preset==='mri'&&!custom?.points){
    // MRI has arbitrary intensity units. Sequence changes presentation only;
    // signal never creates a vessel/tumor label or a missing scan.
    const brightBlood=MR_ANGIOGRAPHY.test(modality);
    if(brightBlood){
      // A declared angiographic acquisition supports high-signal emphasis.
      // Neutral ivory preserves its signal provenance: bright voxels are not
      // artery/vein classifications, and no vessel geometry is synthesized.
      knots=[
        [0,.05,.06,.07,0], [.25,.18,.20,.22,0], [.45,.40,.43,.45,.002],
        [.60,.69,.71,.72,.016], [.75,.92,.92,.89,.13],
        [.90,1,.98,.93,.40], [1,1,.99,.96,.65],
      ];
    }else if(/T2|FLAIR/i.test(modality))knots=knots.map(([x,r,g,b,a])=>[x,r*.94,g*.98,Math.min(1,b*1.04),a*(x<.65?.7:1)] as Knot);
  }
  const range: [number,number] = preset === 'mri' ? [Math.min(0,window[0]),Math.max(1,window[1])] : [knots[0][0],3000];
  if(custom?.window && custom.window.every(Number.isFinite) && custom.window[1]>custom.window[0])range.splice(0,2,...custom.window);
  if(custom?.points && custom.points.length>=2 && custom.points.length<=32 && custom.points.every(p=>p.length===5&&p.every(Number.isFinite)&&p.every(v=>v>=0&&v<=1)) && custom.points.every((p,i)=>i===0||p[0]>custom.points![i-1][0]))knots=custom.points.map(p=>[preset==='mri'?p[0]:range[0]+p[0]*(range[1]-range[0]),...p.slice(1)] as Knot);
  // Extinction values are deliberately small in transparent context views.
  // Byte quantization erased values below 1/510 and produced visible opacity
  // bands; retain the actual transfer in floating point through compositing.
  const data = new Float32Array(VOLUME_TRANSFER_WIDTH*4);
  const linear = (n: number) => n <= .04045 ? n/12.92 : ((n+.055)/1.055)**2.4;
  for(let i=0;i<VOLUME_TRANSFER_WIDTH;i++) {
    const normalized=i/(VOLUME_TRANSFER_WIDTH-1);
    const value = preset === 'mri' ? normalized : range[0]+normalized*(range[1]-range[0]);
    let k=0; while(k<knots.length-2 && value>knots[k+1][0])k++;
    const a=knots[k],b=knots[k+1],t=Math.max(0,Math.min(1,(value-a[0])/(b[0]-a[0])));
    for(let channel=0;channel<4;channel++) {
      const v=a[channel+1]*(1-t)+b[channel+1]*t;
      data[i*4+channel]=channel===3?v:linear(v);
    }
  }
  return { data, range, gradientOpacity: preset==='mri'&&!custom?.points };
}

/** Display-only boundary emphasis for automatic MRI context. A physical
 * gradient is normalized to the selected signal range, so the same boundary
 * has the same extinction on mirrored/anisotropic source grids. Uniform
 * interiors recede while source-supported boundaries retain the chosen LUT.
 * Custom transfer points and CT presets retain their authored extinction. */
export function mriBoundaryExtinction(normalizedGradientPerMm: number) {
  const t=Math.max(0,Math.min(1,(normalizedGradientPerMm-.01)/.065));
  return .06+.94*t*t*(3-2*t);
}

/** World-distance ray bounds for the complete voxel-cell field. Used by the
 * shader too: affine scale/shear must never turn ray steps into voxel units. */
export function volumeRayInterval(origin: Point3, direction: Point3, shape: Point3, affine: Affine, clip?: [number,number,number,number]) {
  const inverse=inverseAffine(rasAffine(affine,'RAS+'));
  let near=0,far=Infinity;
  for(let axis=0;axis<3;axis++) {
    const o=inverse[axis].slice(0,3).reduce((s,v,i)=>s+v*origin[i],inverse[axis][3]);
    const d=inverse[axis].slice(0,3).reduce((s,v,i)=>s+v*direction[i],0);
    if(Math.abs(d)<1e-12){if(o<-.5 || o>shape[axis]-.5)return null;continue;}
    const a=(-.5-o)/d,b=(shape[axis]-.5-o)/d;
    near=Math.max(near,Math.min(a,b));far=Math.min(far,Math.max(a,b));
  }
  if(clip) {
    const distance=clip.slice(0,3).reduce((s,v,i)=>s+v*origin[i],clip[3]);
    const slope=clip.slice(0,3).reduce((s,v,i)=>s+v*direction[i],0);
    if(Math.abs(slope)<1e-12){if(distance<0)return null;}
    else if(slope>0)near=Math.max(near,-distance/slope);else far=Math.min(far,-distance/slope);
  }
  return far>near ? [near,far] : null;
}
export const opticalAlpha = (extinction: number, density: number, stepMm: number) => 1-Math.exp(-extinction*density*stepMm);
