import type { Bounds3, Point3 } from './coordinates';

export interface ViewerCameraState { position: Point3; target: Point3; up: Point3 }
export interface ViewerClipping { axis: 'x'|'y'|'z'; value: number; enabled: boolean; flipped?: boolean }
/** Display choices only. No setting changes source arrays, masks or registration. */
export interface ViewerWorkspaceSettings {
  skullMode?: 'exterior'|'transparent'|'cutaway';
  volumeRendering?: import('./volumeTransfer').VolumeRenderingSettings;
  layout?: 'mri-review'|'3d-focus';
  activePlane?: 'axial'|'coronal'|'sagittal';
  showPlane?: boolean;
  contrast?: number;
  layerOpacity?: Record<string, number>;
  layerColor?: Record<string, string>;
  camera?: ViewerCameraState | null;
  clipping?: ViewerClipping;
  clipBox?: {enabled:boolean;min:Point3;max:Point3};
  obliqueClip?: {enabled:boolean;normal:Point3;offset:number};
  fusion?: { seriesId: string|null; opacity: number };
}
export function layerAppearance(name: string, color: string, settings: ViewerWorkspaceSettings = {}) {
  const opacity = settings.layerOpacity?.[name];
  const chosenColor = settings.layerColor?.[name];
  return { opacity: typeof opacity === 'number' && Number.isFinite(opacity) ? Math.min(1, Math.max(0, opacity)) : .66,
    color: chosenColor && /^#[\da-f]{6}$/i.test(chosenColor) ? chosenColor : color };
}
export function clippingForBounds(settings: ViewerClipping|undefined, bounds: Bounds3): ViewerClipping {
  const axis = settings && ['x','y','z'].includes(settings.axis) ? settings.axis : 'z';
  const index = {x:0,y:1,z:2}[axis];
  const value = settings && Number.isFinite(settings.value) ? settings.value : (bounds[0][index]+bounds[1][index])/2;
  return {axis, value: Math.max(bounds[0][index], Math.min(bounds[1][index],value)), enabled: settings?.enabled === true, flipped: settings?.flipped === true};
}
export function validCameraState(input: ViewerCameraState|null|undefined): input is ViewerCameraState {
  if (!input || ![input.position,input.target,input.up].every(p=>Array.isArray(p)&&p.length===3&&p.every(n=>Number.isFinite(n)&&Math.abs(n)<1e6))) return false;
  const direction=input.position.map((n,i)=>n-input.target[i]);
  const distance=Math.hypot(...direction),upLength=Math.hypot(...input.up);
  if(distance<.1 || distance>1e5 || upLength<.01) return false;
  const cosine=Math.abs(direction.reduce((sum,n,i)=>sum+n*input.up[i],0)/(distance*upLength));
  return cosine<.9999;
}
