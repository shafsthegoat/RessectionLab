import type {Point3} from './coordinates.ts';
import type {ViewerRoute} from './contracts.ts';
import type {ViewerCameraState,ViewerWorkspaceSettings} from './workspaceSettings.ts';

function unit(p:Point3):Point3 {const n=Math.hypot(...p);return p.map(v=>v/n) as Point3;}
function cross(a:Point3,b:Point3):Point3{return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];}
function approachEndpoints(route:ViewerRoute,frame:string):[Point3,Point3] {
  const sign=frame.startsWith('LPS')?-1:1;
  const point=(p:number[]):Point3=>[p[0]*sign,p[1]*sign,p[2]];
  return [point(route.entry_mm),point(route.target_mm)];
}

/** Temporary visual boundary around the recorded corridor and nearby tissue. */
export function approachClipBox(route:ViewerRoute,frame:string):ViewerWorkspaceSettings['clipBox'] {
  const [entry,target]=approachEndpoints(route,frame);
  const margin=Math.max(10,route.tool.shaft_radius_mm+6,route.tool.tip_radius_mm+6,(route.window?.radius_mm??0)+4);
  if(![...entry,...target,margin].every(Number.isFinite))return undefined;
  return {enabled:true,min:entry.map((n,i)=>Math.min(n,target[i])-margin) as Point3,
    max:entry.map((n,i)=>Math.max(n,target[i])+margin) as Point3};
}

/** Optical framing for an illustrative corridor; source geometry stays unchanged. */
export function approachCamera(route:ViewerRoute,frame:string,mode:'along'|'oblique',aspect=1):ViewerCameraState|null {
  const [entry,target]=approachEndpoints(route,frame),delta=entry.map((n,i)=>n-target[i]) as Point3,length=Math.hypot(...delta);
  if(![...entry,...target,length].every(Number.isFinite)||length<.01)return null;
  const outward=unit(delta),reference:Point3=Math.abs(outward[2])>.9?[0,1,0]:[0,0,1],right=unit(cross(reference,outward)),up=unit(cross(outward,right));
  const direction=mode==='along'?outward:unit(outward.map((n,i)=>n*.55+right[i]*.8+up[i]*.32) as Point3);
  const radius=Math.max(route.tool.shaft_radius_mm,route.tool.tip_radius_mm,2);
  const centre=entry.map((n,i)=>(n+target[i])/2) as Point3;
  const fov=34*Math.PI/180,halfFov=Math.min(fov/2,Math.atan(Math.tan(fov/2)*Math.max(.2,aspect)));
  // The geometric axis extends 18 mm outside the entry. Keep it in the oblique
  // frame together with the corridor and a useful local anatomical context.
  const patchRadius=route.window?.radius_mm??0;
  const visibleExtent=mode==='along'?Math.max(28,radius*3,patchRadius*1.4):Math.max(38,length*.55+18+radius*2,patchRadius*1.5);
  const distance=Math.max(length*.65+40,visibleExtent/Math.tan(halfFov));
  return {position:centre.map((n,i)=>n+direction[i]*distance) as Point3,target:centre,up};
}
