import type {Point3} from './coordinates.ts';
import type {ViewerRoute} from './contracts.ts';
import type {ViewerCameraState} from './workspaceSettings.ts';

function unit(p:Point3):Point3 {const n=Math.hypot(...p);return p.map(v=>v/n) as Point3;}
function cross(a:Point3,b:Point3):Point3{return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];}
/** Optical framing for an illustrative corridor; source geometry stays unchanged. */
export function approachCamera(route:ViewerRoute,frame:string,mode:'along'|'oblique',aspect=1):ViewerCameraState|null {
  const sign=frame.startsWith('LPS')?-1:1;
  const point=(p:number[]):Point3=>[p[0]*sign,p[1]*sign,p[2]];
  const entry=point(route.entry_mm),target=point(route.target_mm),delta=entry.map((n,i)=>n-target[i]) as Point3,length=Math.hypot(...delta);
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
