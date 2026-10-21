import type {RehearsalEpisode, RehearsalResponse, RehearsalTool} from './rehearsal-types.ts';
import type {ViewerRehearsalState, ViewerVolume} from './viewer/contracts.ts';
import type {Bounds3, Point3} from './viewer/coordinates.ts';
import {rasAffine} from './viewer/coordinates.ts';
import {checkedRehearsalTool} from './viewer/rehearsal.ts';
function need(value: unknown, reason: string): asserts value {if(!value)throw Error(`Research rehearsal withheld: ${reason}`);}
const equal=(a:unknown,b:unknown)=>JSON.stringify(a)===JSON.stringify(b);
function indices(input: unknown, shape: readonly number[], label: string): number[] {
  const count=shape.reduce((a,b)=>a*b,1);
  need(Array.isArray(input),`${label} is not a native cell list`);
  const flat=input.map(value=>{
    if(Array.isArray(value)){
      need(value.length===3&&value.every((n,i)=>Number.isInteger(n)&&n>=0&&n<shape[i]),`${label} contains an invalid native coordinate`);
      return (value[0]*shape[1]+value[1])*shape[2]+value[2];
    }
    need(Number.isInteger(value)&&value>=0&&value<count,`${label} contains an invalid source index`);
    return value as number;
  });
  need(new Set(flat).size===flat.length,`${label} contains duplicate source cells`);
  return flat;
}
function normalizeTool(tool:RehearsalTool):RehearsalTool {
  need(['aspiration','field_management'].includes(tool.mode),'unknown instrument behavior');
  return {...tool};
}
function withGeometry(tool:RehearsalTool, registry:RehearsalTool[]):RehearsalTool {
  const fixed=registry.find(t=>t.toolId===tool.toolId);
  need(fixed,'recorded instrument is absent from the source-bound tool registry');
  // A missing recorded pose must never inherit the registry’s final pose.
  return normalizeTool({...tool});
}
export function checkedRehearsalEpisode(volume:ViewerVolume,raw:RehearsalEpisode):RehearsalEpisode {
  const episode=structuredClone(raw),binding=episode?.sourceBinding;
  need(episode?.schema==='resectionlab.source-bound-rehearsal.v2','unsupported episode mechanism version');
  need(binding?.caseHash===volume.caseHash,'episode belongs to another source case');
  need(equal(binding.shape,volume.shape)&&equal(rasAffine(binding.affine,binding.frame),rasAffine(volume.affine,volume.frame)),
    'native dimensions or physical frame differ from the original source');
  need(typeof episode.initialStateId==='string'&&episode.initialStateId.length>0&&typeof episode.finalStateId==='string'&&
    Array.isArray(episode.replayFrames)&&episode.replayFrames.length>0&&Array.isArray(episode.history)&&
    Array.isArray(episode.tools)&&episode.tools.length===2,'episode state chain or tool registry is missing');
  episode.tools=episode.tools.map(normalizeTool);
  episode.assumedExposureIndicesNative=indices(episode.assumedExposureIndicesNative,volume.shape,'initial exposure');
  episode.finalRemovedIndicesNative=indices(episode.finalRemovedIndicesNative,volume.shape,'final removal');
  const removed=new Set<number>(),exposure=new Set(episode.assumedExposureIndicesNative);
  const affine=rasAffine(volume.affine,volume.frame),[a,b,c]=affine;
  const voxelVolume=Math.abs(a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0]));
  need(Number.isFinite(voxelVolume)&&voxelVolume>0,'source grid has no physical cell volume');
  let previousState=episode.initialStateId,previousTime=0,previousAction=-1;
  const ids=episode.tools.map(t=>t.toolId);
  need(new Set(ids).size===2,'persistent instrument identities coincide');
  for(const [index,frame] of episode.replayFrames.entries()) {
    need(frame.frameIndex===index&&Number.isInteger(frame.actionIndex)&&frame.actionIndex>=previousAction&&
      typeof frame.kind==='string'&&Number.isFinite(frame.elapsedSeconds)&&frame.elapsedSeconds>=previousTime&&
      frame.stateBefore===previousState&&typeof frame.stateAfter==='string'&&frame.stateAfter.length>0,
      'frame order, action clock or state ancestry changed');
    need(Array.isArray(frame.tools)&&frame.tools.length===2&&equal(frame.tools.map(t=>t.toolId).sort(),[...ids].sort()),'persistent tool identity changed');
    frame.tools=frame.tools.map(t=>withGeometry(t,episode.tools));
    frame.tools.forEach(t=>checkedRehearsalTool(t,`${index}`));
    for(const [key,label] of [['removedIndicesNative','removal'],['contactIndicesNative','contact'],['fieldContactIndicesNative','field management']] as const)frame[key]=indices(frame[key],volume.shape,label);
    for(const cell of frame.removedIndicesNative){need(!removed.has(cell)&&!exposure.has(cell),'irreversible removal was counted twice');removed.add(cell);}
    const metrics=frame.metrics;
    need(metrics&&[metrics.targetRemovedMm3,metrics.nonTargetRemovedMm3,metrics.residualTargetMm3,metrics.fieldBurden].every(n=>Number.isFinite(n)&&n>=0)&&
      metrics.motorRisk===null&&metrics.languageRisk===null&&metrics.vascularEvents===null,'effect accounting or unknown vascular/functional evidence is invalid');
    need(Math.abs(metrics.targetRemovedMm3+metrics.nonTargetRemovedMm3-removed.size*voxelVolume)<1e-5*Math.max(1,removed.size*voxelVolume),
      'removed volume differs from cumulative native cells');
    const geometryFields=['workingLengthMm','tipLengthMm','shaftRadiusMm','tipRadiusMm','mode'] as const;
    for(const tool of frame.tools){const fixed=episode.tools.find(t=>t.toolId===tool.toolId)!;need(geometryFields.every(k=>tool[k]===fixed[k]),'persistent tool geometry changed');}
    if(index===0)need(frame.stateAfter===episode.initialStateId&&frame.removedIndicesNative.length===0,'initial state already contains episode removal');
    previousState=frame.stateAfter;previousTime=frame.elapsedSeconds;previousAction=frame.actionIndex;
  }
  need(previousState===episode.finalStateId,'final state is disconnected from replay');
  need(removed.size===episode.finalRemovedIndicesNative.length&&episode.finalRemovedIndicesNative.every(n=>removed.has(n)),
    'final removal disagrees with frame deltas');
  return episode;
}
export function checkedRehearsalResponse(volume:ViewerVolume,raw:RehearsalResponse):RehearsalResponse {
  need(raw.caseHash===volume.caseHash,'response belongs to another case');
  const episode=checkedRehearsalEpisode(volume,raw.episode);
  need(raw.stateId===episode.finalStateId&&Array.isArray(raw.suggestedActions),'live state differs from replay');
  const result={...raw,episode};
  if(raw.comparison) {
    const baseIndex=episode.replayFrames.findIndex(f=>f.stateAfter===raw.comparison!.snapshotId);
    need(baseIndex>=0&&raw.comparison.alternatives.length===2,'comparison snapshot is absent from live history');
    const basePrefix=episode.replayFrames.slice(0,baseIndex+1);
    const alternatives=raw.comparison.alternatives.map(branch=>({...branch,episode:checkedRehearsalEpisode(volume,branch.episode)}));
    need(new Set(alternatives.map(a=>a.id)).size===2,'alternative identities coincide');
    for(const branch of alternatives) {
      const prefix=branch.episode.replayFrames.slice(0,basePrefix.length);
      need(equal(prefix,basePrefix)&&equal(branch.episode.assumedExposureIndicesNative,episode.assumedExposureIndicesNative),
        'alternative changed the saved source state or its previous effects');
    }
    result.comparison={...raw.comparison,alternatives};
  }
  return result;
}
/** Materialize only the selected cumulative frame, never one full volume per frame. */
export function rehearsalViewerState(volume:ViewerVolume,episode:RehearsalEpisode,frameIndex:number):ViewerRehearsalState {
  need(Number.isInteger(frameIndex)&&frameIndex>=0&&frameIndex<episode.replayFrames.length,'selected frame is outside the episode');
  const count=volume.shape.reduce((a,b)=>a*b,1),removedMask=new Uint8Array(count),contactMask=new Uint8Array(count),fieldContactMask=new Uint8Array(count),assumedExposureMask=new Uint8Array(count);
  for(const i of episode.assumedExposureIndicesNative)assumedExposureMask[i]=1;
  for(const frame of episode.replayFrames.slice(0,frameIndex+1)) {
    for(const i of frame.removedIndicesNative)removedMask[i]=1;
    for(const i of frame.contactIndicesNative)contactMask[i]=1;
    for(const i of frame.fieldContactIndicesNative)fieldContactMask[i]=1;
  }
  const bounds:Bounds3=[[Infinity,Infinity,Infinity],[-Infinity,-Infinity,-Infinity]];
  for(const frame of episode.replayFrames)for(const tool of frame.tools){
    const display=checkedRehearsalTool(tool,'bounds');
    for(const [point,radius] of [[display.shaftStart,display.shaftRadius],[display.shaftEnd,display.shaftRadius],[display.shaftEnd,display.tipRadius],[display.tip,display.tipRadius]] as const)
      for(let a=0;a<3;a++){bounds[0][a]=Math.min(bounds[0][a],point[a]-radius);bounds[1][a]=Math.max(bounds[1][a],point[a]+radius);}
  }
  const frame=episode.replayFrames[frameIndex],episodeId=episode.episodeId??`${episode.initialStateId}:${episode.finalStateId}`;
  const state:ViewerRehearsalState={scope:'source-bound-post-exposure-research',caseHash:volume.caseHash,episodeId,stateId:frame.stateAfter,
    frameIndex,stepCount:episode.replayFrames.length-1,shape:[...volume.shape],affine:rasAffine(volume.affine,volume.frame),frame:'RAS+',
    elapsedSeconds:frame.elapsedSeconds,kind:frame.kind,tools:frame.tools.map(t=>({...t,tipRasMm:[...t.tipRasMm] as Point3,axis:[...t.axis] as Point3})),
    removedMask,contactMask,fieldContactMask,assumedExposureMask,metrics:{...frame.metrics},recordedEpisodeBounds:{caseHash:volume.caseHash,episodeId,boundsRasMm:bounds}};
  return state;
}
