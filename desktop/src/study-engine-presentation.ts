import {Vector3} from 'three';
import type {NativeApproach} from './native-approaches';
import type {VerifiedStudyReplay} from './types';
import type {ViewerRoute} from './viewer/contracts';
import {microscopeCamera} from './viewer/microscope.ts';
import type {MicroscopeSettings,ViewerWorkspaceSettings} from './viewer/workspaceSettings';

type StudySelection = {
  collectionId:string;patientId:string;snapshotId:string;targetLayerId:string;
  route:NativeApproach|null;
};

export function recordingSelectionKey(selection:StudySelection|null){
  if(!selection)return null;
  const {collectionId,patientId,snapshotId,targetLayerId,route}=selection;
  return JSON.stringify([collectionId,patientId,snapshotId,targetLayerId,route?.id,route?.parameterHash,route?.measurementHash]);
}

export function recordingMatchesStudy(receipt:VerifiedStudyReplay|undefined,caseHash:string,selection:StudySelection){
  if(!receipt||receipt.caseHash!==caseHash||receipt.patientId!==selection.patientId||
    receipt.sourceSnapshotId!==selection.snapshotId||receipt.targetLayerId!==selection.targetLayerId)return false;
  const recorded=receipt.recordedApproach;
  if(!recorded)return true; // Older source-verified episodes do not certify a selected approach.
  const route=selection.route;
  return !!route&&route.sourceSnapshotId===selection.snapshotId&&route.targetLayerId===selection.targetLayerId&&
    route.assessmentStatus==='current'&&route.id===recorded.id&&route.parameterHash===recorded.parameterHash&&
    route.measurementHash===recorded.measurementHash&&route.radiusMm===recorded.radiusMm&&
    route.entry.every((value,i)=>value===recorded.entry[i])&&route.target.every((value,i)=>value===recorded.target[i]);
}

export function recordedOverviewRoute(recorded:NonNullable<VerifiedStudyReplay['recordedApproach']>):ViewerRoute{
  return {route_id:recorded.id,comparisonSlot:'A',scope:'geometric-approach',label:recorded.name,selected:true,
    entry_mm:recorded.entry,target_mm:recorded.target,
    tool:{working_length_mm:Math.hypot(...recorded.entry.map((v,i)=>v-recorded.target[i])),
      tip_length_mm:0,shaft_radius_mm:recorded.radiusMm,tip_radius_mm:recorded.radiusMm}};
}

/** Fixed optical presets from the physically bound route. The longitudinal
 * inspection plane passes through both endpoints and keeps the far side. */
export function recordedCorridorView(recorded:NonNullable<VerifiedStudyReplay['recordedApproach']>,settings:ViewerWorkspaceSettings,section:boolean):ViewerWorkspaceSettings {
  const entry=new Vector3(...recorded.entry),target=new Vector3(...recorded.target);
  const outward=entry.clone().sub(target),length=outward.length();
  if(!(length>0))return settings;
  outward.normalize();
  // The exterior opening is toward the entry. Allow room for the projected
  // entry-to-target separation when the user tilts the microscope.
  const field=section?length*1.15+recorded.radiusMm*4:Math.max(18,recorded.radiusMm*13,length*.28+recorded.radiusMm*7);
  const center=target.clone().lerp(entry,section?.44:.65);
  const microscope:MicroscopeSettings={...settings.microscope,target:center.toArray(),outward:outward.toArray(),
    fieldMm:Math.max(12,Math.min(120,field)),angleDeg:section?62:7,azimuthDeg:0,workingDistanceMm:160,
    focusOffsetMm:0,light:settings.microscope?.light??2.5,inspectionCutaway:section,inspectionTransparency:false};
  const camera=microscopeCamera(microscope);
  let obliqueClip:ViewerWorkspaceSettings['obliqueClip'];
  if(section){
    const eye=new Vector3(...camera.position).sub(center);
    const normal=eye.addScaledVector(outward,-eye.dot(outward)).normalize().negate();
    obliqueClip={enabled:true,normal:normal.toArray(),offset:-normal.dot(target)};
  }
  return {...settings,operativeView:'surgeon',microscope,camera,
    externalCamera:settings.operativeView==='surgeon'?settings.externalCamera:settings.camera,
    layout:'3d-focus',contextStyle:'uniform',showPlane:false,clipping:undefined,clipBox:undefined,obliqueClip,
    volumeRendering:{...settings.volumeRendering,enabled:false}};
}

export function recordedMicroscope(recorded:NonNullable<VerifiedStudyReplay['recordedApproach']>,settings:ViewerWorkspaceSettings):ViewerWorkspaceSettings{
  return settings.operativeView?settings:recordedCorridorView(recorded,settings,false);
}
