import type {FullCaseRehearsal,SurgicalView} from './surgical-types';
import type {Point3} from './viewer/coordinates';
import type {ViewerWorkspaceSettings} from './viewer/workspaceSettings';

/** An explicit display command, detached from the source and recorded engine. */
export function operativeFieldFrame(view:SurgicalView,targetCenter:Point3|null,settings:ViewerWorkspaceSettings) {
  const valid=(point:unknown):point is Point3=>Array.isArray(point)&&point.length===3&&point.every(Number.isFinite);
  const inserted=Object.values(view.slots).filter(slot=>slot.inserted&&valid(slot.position));
  const tip=inserted.find(slot=>slot.active)??inserted[0];
  const center=tip?.position??targetCenter;if(!valid(center))return null;
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
  return {center:target,settings:{...settings,layout:'3d-focus' as const,camera:{target,position,up}}};
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
