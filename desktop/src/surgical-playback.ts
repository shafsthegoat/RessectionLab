import type {SurgicalPlayback,SurgicalResponse,SurgicalView} from './surgical-types';
import {checkedSurgicalFlowDisplay} from './surgical-flow-display.ts';
/** Elapsed time recorded by the authoritative action result, never inferred
 * from frame index or requested duration (partial actions may stop early). */
export function surgicalPlaybackDelay(timeline:readonly {elapsedSeconds?:number}[],position:number,speed=1):{milliseconds:number;recorded:boolean} {
  const rate=Number.isFinite(speed)&&speed>0?speed:1;
  const elapsed=timeline[position]?.elapsedSeconds;
  if(typeof elapsed==='number'&&Number.isFinite(elapsed)&&elapsed>=0)return {milliseconds:elapsed*1000/rate,recorded:true};
  // Historical results without timing remain explicit state-by-state review.
  return {milliseconds:600/rate,recorded:false};
}

/** A requested interval includes time already spent fetching/preparing its
 * exact predecessor. Falling behind removes idle delay, never observations. */
export class SurgicalPlaybackClock {
  private anchor:{observedSeconds:number;wallTime:number;speed:number}|null=null;
  reset(){this.anchor=null;}
  delay(observedSeconds:number,intervalMs:number,speed:number,now:number):number {
    const rate=Number.isFinite(speed)&&speed>0?speed:1;
    if(!Number.isFinite(observedSeconds)||!Number.isFinite(now)){this.reset();return intervalMs;}
    if(!this.anchor||this.anchor.speed!==rate||observedSeconds<this.anchor.observedSeconds)
      this.anchor={observedSeconds,wallTime:now,speed:rate};
    return Math.max(0,this.anchor.wallTime+(observedSeconds-this.anchor.observedSeconds)*1000/rate+intervalMs-now);
  }
}

export const surgicalCursorCommand=(command:string)=>['seek','seekFrame','seekPrefix','seekPrefixFrame','inspectArchive','selectBranch','inspect'].includes(command);
export const updatesEditableSurgicalCursor=(command:string)=>!['preview','seekFrame','seekPrefix','seekPrefixFrame','inspectArchive'].includes(command);
export function checkedSurgicalPlayback(response:SurgicalResponse,request:{branchId?:unknown;position?:unknown;frameIndex?:unknown}) {
  const p=response.playback;
  const need=(condition:unknown)=>{if(!condition)throw new Error('Recorded microstep response differs from the requested source-bound frame.');};
  need(p?.readOnly===true&&p.branchId===request.branchId&&p.position===request.position&&p.actionIndex===p.position&&p.frameIndex===request.frameIndex);
  need(p&&Number.isInteger(p.position)&&p.position>=0&&Number.isInteger(p.frameIndex)&&p.frameIndex>=0&&Number.isInteger(p.frameCount)&&p.frameCount>=2&&p.frameIndex<p.frameCount);
  if(!p)throw new Error('Recorded microstep metadata is unavailable.');
  need(Array.isArray(p.frames)&&p.frames.length===p.frameCount&&[p.observedTimeSeconds,p.actionElapsedSeconds,p.actionDurationSeconds].every(n=>Number.isFinite(n)&&n>=0));
  let previous=-Infinity;
  for(const [index,frame] of p.frames.entries()){need(frame.frameIndex===index&&Number.isFinite(frame.observedTimeSeconds)&&frame.observedTimeSeconds>=previous&&typeof frame.kind==='string');previous=frame.observedTimeSeconds;}
  need(p.frames[p.frameIndex].observedTimeSeconds===p.observedTimeSeconds&&typeof p.isActionBoundary==='boolean');
  need(Math.abs((p.observedTimeSeconds-p.frames[0].observedTimeSeconds)-p.actionElapsedSeconds)<1e-7&&p.actionElapsedSeconds<=p.actionDurationSeconds+1e-7);
  need(Math.abs((p.frames.at(-1)!.observedTimeSeconds-p.frames[0].observedTimeSeconds)-p.actionDurationSeconds)<1e-7);
  need(p.frameIndex!==p.frameCount-1||p.isActionBoundary);
  need(Math.abs(Number(response.view.metrics.elapsedSeconds)-p.observedTimeSeconds)<1e-7);
  need(!p.effectiveActivation||Object.entries(p.effectiveActivation).every(([slot,active])=>slot in response.view.slots&&typeof active==='boolean'));
  checkedSurgicalFlowDisplay(p.flowDisplay,response.view,p);
  return p;
}

/** Prefetched wire may omit immutable scene context. Check it against the same
 * source/branch and acknowledged timeline generation without hydrating future
 * field values, changing stream acknowledgments or publishing the receipt. */
export function checkedSurgicalPlaybackWire(response:SurgicalResponse,request:{branchId?:unknown;position?:unknown;frameIndex?:unknown},resident:{caseHash:string;branchId:string;timelineToken?:string;scene:SurgicalView['scene']}){
  if(!response.timelineOmitted)return checkedSurgicalPlayback(response,request);
  if(!resident.timelineToken||response.timelineToken!==resident.timelineToken||response.caseHash!==resident.caseHash||response.branchId!==resident.branchId||Object.hasOwn(response.view,'scene'))
    throw Error('Compact recorded microstep differs from its resident source context.');
  return checkedSurgicalPlayback({...response,view:{...response.view,scene:resident.scene}},request);
}

/** Advance only to observed successful states. Never interpolate effects or time. */
export function nextSurgicalPlaybackFrame(response:SurgicalResponse,steps:number,speed=1) {
  const p=response.playback,rate=Number.isFinite(speed)&&speed>0?speed:1;
  if(!p)return response.position<steps?{branchId:response.branchId,position:response.position,frameIndex:0,milliseconds:0}:null;
  if(p.frameIndex+1<p.frameCount){const next=p.frames[p.frameIndex+1];return {branchId:p.branchId,position:p.position,frameIndex:next.frameIndex,milliseconds:Math.max(0,(next.observedTimeSeconds-p.observedTimeSeconds)*1000/rate)};}
  return p.position+1<steps?{branchId:p.branchId,position:p.position+1,frameIndex:0,milliseconds:0}:null;
}
export function surgicalPlaybackPosition(p:SurgicalPlayback|undefined|null,position:number){return p?p.position+(p.frameIndex===p.frameCount-1?1:0):position;}
