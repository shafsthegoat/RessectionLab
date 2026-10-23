import type {FullCaseRehearsal,SurgicalView} from './surgical-types';
import type {Point3} from './viewer/coordinates';
import {inverseAffine,rasAffine,transformPoint} from './viewer/coordinates.ts';
import type {ViewerVolume} from './viewer/contracts';
import type {ViewerWorkspaceSettings} from './viewer/workspaceSettings';

const validPoint=(point:unknown):point is Point3=>Array.isArray(point)&&point.length===3&&point.every(Number.isFinite);
/** Acquired voxel support in RAS, including oblique and LPS source grids. */
export function sourceContainsPoint(source:Pick<ViewerVolume,'shape'|'affine'|'frame'>|null,point:unknown):point is Point3 {
  if(!source||!validPoint(point))return false;
  try{
    if(source.shape.length!==3||!source.shape.every(n=>Number.isInteger(n)&&n>0))return false;
    const inverse=inverseAffine(rasAffine(source.affine,source.frame));
    return transformPoint(inverse,point).every((v,i)=>Number.isFinite(v)&&v>=0&&v<=source.shape[i]-1);
  }catch{return false;}
}

/** An explicit display command, detached from the source and recorded engine. */
export function operativeFieldFrame(view:SurgicalView,targetCenter:Point3|null,settings:ViewerWorkspaceSettings,source?:Pick<ViewerVolume,'shape'|'affine'|'frame'>|null) {
  if(source===null)return null;
  const inSource=(point:unknown):point is Point3=>source===undefined?validPoint(point):sourceContainsPoint(source,point);
  const inserted=Object.values(view.slots).filter(slot=>slot.inserted&&inSource(slot.position));
  const tip=inserted.find(slot=>slot.active)??inserted[0];
  const center=tip?.position??targetCenter;if(!inSource(center))return null;
  const target=[...center] as Point3;
  let direction:Point3=[35,-45,40],up:Point3=[0,0,1];
  const quaternion=tip?.quaternion;
  if(quaternion?.length===4&&quaternion.every(Number.isFinite)&&Math.hypot(...quaternion)>1e-8){
    const length=Math.hypot(...quaternion),[x,y,z,w]=quaternion.map(n=>n/length);
    // Engine tools point inward along local +Z. View into the corridor from
    // its outside, with the orthogonal tool-local +Y as the camera up vector.
    direction=[-2*(x*z+w*y),-2*(y*z-w*x),-(1-2*(x*x+y*y))];
    up=[2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)];
  }
  const scale=115/Math.hypot(...direction),position=direction.map((n,i)=>target[i]+n*scale) as Point3;
  return {center:target,anchor:tip?'instrument' as const:'target' as const,settings:{...settings,layout:'3d-focus' as const,camera:{target,position,up}}};
}

/** Accepted instruments only: an observed activation overrides the stored slot flag. */
export function acceptedCloseTool(view:SurgicalView|null,selectedSlot:string,effectiveActivation?:Record<string,boolean>){
  const inserted=Object.entries(view?.slots??{}).filter(([,slot])=>slot.inserted&&Array.isArray(slot.position)&&slot.position.length===3&&slot.position.every(Number.isFinite));
  const chosen=inserted.find(([name,slot])=>(effectiveActivation?.[name]??slot.active)===true)??inserted.find(([name])=>name===selectedSlot)??inserted[0];
  return chosen?{name:chosen[0],slot:chosen[1]}:null;
}

/** Display alpha by declared material; no anatomy is hidden, removed or relabeled. */
export function closeToolPresentation(view:SurgicalView,settings:ViewerWorkspaceSettings):ViewerWorkspaceSettings {
  const alpha:Record<string,number>={soft_tissue:.045,bone:.55,membrane:.14,tumor:.9,vessel:1,unknown:.08};
  return {...settings,layerOpacity:{...settings.layerOpacity,...Object.fromEntries(view.scene.regions.map(region=>[region.name,region.material==='vessel'?1:region.target?.9:alpha[region.material]??.14]))}};
}

export type CloseToolWindowClipping={
  clipping:NonNullable<ViewerWorkspaceSettings['clipping']>;
  clipBox:NonNullable<ViewerWorkspaceSettings['clipBox']>;
  obliqueClip:NonNullable<ViewerWorkspaceSettings['obliqueClip']>;
};

/** Change only display clipping, using the window saved at explicit focus.
 * Disabling restores original key presence as well as values; neither toggle
 * changes the camera, linked cursor or the latest recorded instrument pose. */
export function closeToolWindowPresentation(settings:ViewerWorkspaceSettings,window:CloseToolWindowClipping,original:ViewerWorkspaceSettings,enabled:boolean):ViewerWorkspaceSettings {
  const next={...settings};
  for(const key of ['clipping','clipBox','obliqueClip'] as const)delete next[key];
  if(enabled)return {...next,...structuredClone(window)};
  for(const key of ['clipping','clipBox','obliqueClip'] as const)
    if(Object.hasOwn(original,key))Object.assign(next,{[key]:structuredClone(original[key])});
  return next;
}

/** Explicit, pinned microscope-scale view of an exact accepted tip in RAS mm. */
export function closeToolFrame(view:SurgicalView,selectedSlot:string,effectiveActivation:Record<string,boolean>|undefined,settings:ViewerWorkspaceSettings,translucent=true,cutaway=false){
  const chosen=acceptedCloseTool(view,selectedSlot,effectiveActivation);if(!chosen)return null;
  const tool=view.catalog.find(item=>item.id===chosen.slot.instrument),radius=chosen.slot.parameters?.radiusMm??tool?.radiusMm??2;
  const fieldMm=Math.max(18,Math.min(42,Number.isFinite(radius)&&radius>0?radius*12:24));
  const center=[...chosen.slot.position] as Point3;
  let direction:Point3=[.45,0,-1],up:Point3=[0,1,0];
  const quaternion=chosen.slot.quaternion;
  if(quaternion?.length===4&&quaternion.every(Number.isFinite)&&Math.hypot(...quaternion)>1e-8){
    const length=Math.hypot(...quaternion),[x,y,z,w]=quaternion.map(value=>value/length);
    const axis:Point3=[2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)],right:Point3=[1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)];
    // A slight oblique view exposes the working end instead of looking straight
    // down its shaft. Orbit remains free; later states never move this camera.
    direction=axis.map((value,i)=>-value+.45*right[i]) as Point3;
    up=[2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)];
  }
  // Match the existing renderer's34degree vertical field of view.
  const distance=fieldMm/(2*Math.tan(17*Math.PI/180)),scale=distance/Math.hypot(...direction);
  const camera={target:center,position:direction.map((value,i)=>center[i]+value*scale) as Point3,up};
  // Keep the exact tip and a bounded native-world neighborhood. Positive plane
  // distance retains the working end and tissue behind it, while removing
  // camera-facing context. This plane stays pinned when the camera is orbited.
  const towardTip=center.map((value,i)=>value-camera.position[i]),normalLength=Math.hypot(...towardTip);
  const normal=towardTip.map(value=>value/normalLength) as Point3;
  const halfWidth=.75*fieldMm;
  const windowClipping:CloseToolWindowClipping={
    clipping:{axis:'z',value:center[2],enabled:false},
    clipBox:{enabled:true,min:center.map(value=>value-halfWidth) as Point3,max:center.map(value=>value+halfWidth) as Point3},
    obliqueClip:{enabled:true,normal,offset:-normal.reduce((sum,value,i)=>sum+value*center[i],0)+fieldMm/12},
  };
  const appearance=translucent?closeToolPresentation(view,settings):settings;
  const framed={...appearance,layout:'3d-focus' as const,showPlane:false,volumeRendering:{...settings.volumeRendering,enabled:false},camera};
  return {center,name:chosen.name,instrument:tool?.name??chosen.slot.instrument,fieldMm,stateId:view.stateId,windowClipping,
    settings:cutaway?closeToolWindowPresentation(framed,windowClipping,settings,true):framed};
}

/** Read quantities from the selected recorded state, including during replay. */
export function fullCaseStateSummary(view:SurgicalView) {
  const finite=(value:unknown):number|null=>typeof value==='number'&&Number.isFinite(value)?value:null;
  const initial=finite(view.scene.environment?.target.initialVolumeMm3);
  const removed=finite(view.metrics.targetRemovedMm3);
  const sources=view.sources?.filter(source=>source.kind==='modeled-vessel-wound'||source.modelVersion===2);
  const wounds=view.vesselsV2?.wounds??view.vesselWounds;
  const faces=view.vesselsV2?.faces??view.vesselConnections;
  const spatialFluid=Object.values(view.fluid).reduce((a,b)=>a+b,0);
  const reservoirs=view.sources?.reduce((total,source)=>total+(finite(source.reservoirMm3)??0),0)??null;
  const byLabel=view.metrics.removedByLabelMm3;
  const normalMaterials=new Set(['soft_tissue','membrane','functional_region']);
  const sumLabels=(regions:SurgicalView['scene']['regions'])=>regions.length&&byLabel&&typeof byLabel==='object'?regions.reduce((total,region)=>total+(finite(byLabel[region.name])??0),0):null;
  const exposure=view.metrics.exposureByMaterialCellSeconds;
  const sumExposure=(materials:string[])=>materials.length&&exposure&&typeof exposure==='object'?materials.reduce((total,material)=>total+(finite(exposure[material])??0),0):null;
  const normalRegions=view.scene.regions.filter(region=>normalMaterials.has(region.material));
  const unknownRegions=view.scene.regions.filter(region=>region.material==='unknown');
  const vesselMetrics=view.metrics.vesselPhysicsV2;
  return {initialTargetMm3:initial,targetRemovedMm3:removed,
    targetRemainingMm3:initial!==null&&removed!==null?Math.max(0,initial-removed):null,
    targetRemovedPercent:initial!==null&&initial>0&&removed!==null?100*removed/initial:null,
    boneRemovedMm3:sumLabels(view.scene.regions.filter(region=>region.material==='bone')),
    normalRemovedMm3:sumLabels(normalRegions.filter(region=>!region.target)),unknownRemovedMm3:sumLabels(unknownRegions),
    normalExposureCellSeconds:sumExposure([...new Set(normalRegions.map(region=>region.material))]),
    unknownExposureCellSeconds:sumExposure(unknownRegions.length?['unknown']:[]),
    modeledBloodGeneratedMm3:finite(view.metrics.modeledVesselFluidGeneratedMm3)??(vesselMetrics&&typeof vesselMetrics==='object'?finite(vesselMetrics.generatedMm3):null),
    fluidSuctionedMm3:finite(view.metrics.fluidRemovedMm3),elapsedSeconds:finite(view.metrics.elapsedSeconds),
    woundCount:wounds?Object.keys(wounds).length:null,
    activeWounds:sources&&(wounds||sources.length>0)?sources.filter(source=>source.bleedingActive===true).length:null,
    nonpatentFaces:faces?Object.values(faces).filter(face=>face.patent===false).length:null,
    spatialFluidMm3:spatialFluid,reservoirFluidMm3:reservoirs};
}

export function recordedFullCasePhase(fullCase:FullCaseRehearsal|undefined|null,position:number) {
  // A phase begins after its starting snapshot. At a shared boundary show the
  // phase whose effects have actually completed, not the next phase's promise.
  return fullCase?.phases.find(phase=>position>phase.startPosition&&position<=phase.endPosition)??
    (position===0?fullCase?.phases.find(phase=>phase.startPosition===0):undefined);
}
