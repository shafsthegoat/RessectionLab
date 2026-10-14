import type { Affine, Point3 } from './coordinates';
import { inverseAffine, rasAffine } from './coordinates.ts';

export type VolumePreset = 'auto' | 'ct-bone' | 'ct-tissue' | 'mri';
export interface VolumeRenderingSettings { enabled?: boolean; preset?: VolumePreset; density?: number }
export function volumeSettings(input: VolumeRenderingSettings = {}, modality = 'MRI') {
  const preset = input.preset && ['ct-bone','ct-tissue','mri'].includes(input.preset)
    ? input.preset : /^CT/i.test(modality) ? 'ct-bone' : 'mri';
  return { enabled: input.enabled !== false, preset: preset as Exclude<VolumePreset,'auto'>,
    density: Number.isFinite(input.density) ? Math.max(0,Math.min(2,input.density!)) : .45 };
}

/** Display LUT only. Alpha stores optical extinction per millimetre, not a
 * segmentation probability. Colors are linear RGB for physical compositing. */
export function volumeTransfer(preset: Exclude<VolumePreset,'auto'>, window: [number,number]) {
  type Knot = [number, number, number, number, number];
  const knots: Knot[] = preset === 'ct-bone' ? [
    [0, .52,.30,.18,0], [150,.65,.45,.28,0], [230,.75,.57,.38,.04],
    [400,.91,.78,.56,.27], [900,.98,.91,.76,.65], [1800,1,.98,.91,.95], [3000,1,1,.99,1],
  ] : preset === 'ct-tissue' ? [
    [-1000,.15,.10,.09,0], [-500,.42,.20,.13,0], [-120,.62,.38,.24,.004],
    [20,.73,.43,.28,.045], [90,.81,.55,.37,.07], [180,.84,.60,.42,.045],
    [450,.93,.80,.62,.25], [1800,1,.98,.91,.9], [3000,1,1,.99,1],
  ] : [
    [0,.05,.06,.07,0], [.08,.16,.18,.20,0], [.2,.30,.33,.36,.008],
    [.4,.49,.51,.53,.025], [.65,.72,.73,.73,.07], [.85,.91,.90,.87,.16], [1,1,.98,.94,.28],
  ];
  const range: [number,number] = preset === 'mri' ? [Math.min(0,window[0]),Math.max(1,window[1])] : [knots[0][0],3000];
  const data = new Uint8Array(1024*4);
  const linear = (n: number) => n <= .04045 ? n/12.92 : ((n+.055)/1.055)**2.4;
  for(let i=0;i<1024;i++) {
    const value = preset === 'mri' ? i/1023 : range[0]+i/1023*(range[1]-range[0]);
    let k=0; while(k<knots.length-2 && value>knots[k+1][0])k++;
    const a=knots[k],b=knots[k+1],t=Math.max(0,Math.min(1,(value-a[0])/(b[0]-a[0])));
    for(let channel=0;channel<4;channel++) {
      const v=a[channel+1]*(1-t)+b[channel+1]*t;
      data[i*4+channel]=Math.round(255*(channel===3?v:linear(v)));
    }
  }
  return { data, range };
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
