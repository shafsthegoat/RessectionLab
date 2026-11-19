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

export function recordedMicroscope(recorded:NonNullable<VerifiedStudyReplay['recordedApproach']>,settings:ViewerWorkspaceSettings):ViewerWorkspaceSettings{
  if(settings.operativeView)return settings;
  const outward=recorded.entry.map((v,i)=>v-recorded.target[i]) as MicroscopeSettings['outward'];
  const length=Math.hypot(...outward);
  if(!(length>0))return settings;
  const microscope:MicroscopeSettings={target:recorded.target,outward,fieldMm:Math.max(18,Math.min(42,length*1.3)),
    angleDeg:7,azimuthDeg:0,workingDistanceMm:160,focusOffsetMm:0,light:2.5};
  return {...settings,operativeView:'surgeon',microscope,camera:microscopeCamera(microscope),externalCamera:settings.camera,
    layout:'3d-focus',contextStyle:'uniform',showPlane:false,clipping:undefined,clipBox:undefined,obliqueClip:undefined,
    volumeRendering:{...settings.volumeRendering,enabled:false}};
}
