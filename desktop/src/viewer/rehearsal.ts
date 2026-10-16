import type {ViewerRehearsalState, ViewerVolume} from './contracts.ts';
import type {RehearsalTool} from '../rehearsal-types.ts';
import type {RecordedToolDisplay} from './recordedTool.ts';
import {rasAffine} from './coordinates.ts';

export const REHEARSAL_COLORS = {aspiration:'#67d7d1',field_management:'#efb566',removal:'#b7f0d8',exposure:'#b7a7e8',contact:'#edcd8a'};
function need(value: unknown, message: string): asserts value {
  if (!value) throw Error(`Research rehearsal withheld: ${message}`);
}
export function checkedRehearsalTool(tool: RehearsalTool, identity: string): RecordedToolDisplay {
  need(tool && typeof tool.toolId === 'string' && tool.toolId.length > 0 &&
    ['aspiration','field_management'].includes(tool.mode) && typeof tool.active === 'boolean' &&
    [tool.tipRasMm,tool.axis].every(point => Array.isArray(point) && point.length === 3 && point.every(Number.isFinite)) &&
    Math.abs(Math.hypot(...tool.axis)-1)<1e-6 &&
    [tool.workingLengthMm,tool.tipLengthMm,tool.shaftRadiusMm,tool.tipRadiusMm].every(value => Number.isFinite(value) && value > 0) &&
    tool.workingLengthMm > tool.tipLengthMm, 'invalid persistent instrument capsule');
  return {identity:`${identity}:${tool.toolId}`,shaftStart:tool.tipRasMm.map((n,i)=>n-tool.axis[i]*tool.workingLengthMm),
    shaftEnd:tool.tipRasMm.map((n,i)=>n-tool.axis[i]*tool.tipLengthMm),tip:[...tool.tipRasMm],
    shaftRadius:tool.shaftRadiusMm,tipRadius:tool.tipRadiusMm,color:REHEARSAL_COLORS[tool.mode]};
}
/** Validate the dedicated source grid and all visible geometry before display.
 * This is a display guard, not a physiological or independent safety certificate. */
export function checkedRehearsalState(volume: ViewerVolume, state: ViewerRehearsalState) {
  need(state.scope==='source-bound-post-exposure-research' && state.caseHash===volume.caseHash,
    'case identity differs from the displayed source');
  need(state.frame==='RAS+' && Array.isArray(state.shape) && state.shape.length===3 &&
    state.shape.every((n,i)=>n===volume.shape[i]),'source frame or dimensions differ');
  const affine=rasAffine(volume.affine,volume.frame);
  need(Array.isArray(state.affine) && state.affine.length===4 && state.affine.every((row,r)=>
    Array.isArray(row) && row.length===4 && row.every((n,c)=>Number.isFinite(n)&&Math.abs(n-affine[r][c])<1e-7)),
    'source affine differs');
  need(Number.isInteger(state.frameIndex)&&Number.isInteger(state.stepCount)&&state.frameIndex>=0&&state.frameIndex<=state.stepCount&&
    typeof state.stateId==='string'&&state.stateId.length>0&&typeof state.episodeId==='string'&&state.episodeId.length>0&&
    Number.isFinite(state.elapsedSeconds)&&state.elapsedSeconds>=0,'recorded frame is invalid');
  const count=volume.shape.reduce((a,b)=>a*b,1);
  for (const mask of [state.removedMask,state.assumedExposureMask,state.fieldContactMask,state.contactMask])
    need(mask instanceof Uint8Array&&mask.length===count,'effect grid differs from the source');
  const effectMask=new Uint8Array(count),voidMask=new Uint8Array(count);
  for(let i=0;i<count;i++) {
    need(state.removedMask[i]<=1&&state.assumedExposureMask[i]<=1&&state.fieldContactMask[i]<=1&&state.contactMask[i]<=1,'effect mask is not binary');
    need(!(state.removedMask[i]&&state.assumedExposureMask[i]),'episode removal overlaps assumed initial exposure');
    // Byte codes retain original MRI intensity and distinguish assumptions from effects.
    effectMask[i]=state.removedMask[i]?1:state.assumedExposureMask[i]?2:state.fieldContactMask[i]?3:state.contactMask[i]?4:0;
    voidMask[i]=state.removedMask[i]||state.assumedExposureMask[i]?1:0;
  }
  need(Array.isArray(state.tools)&&state.tools.length===2&&new Set(state.tools.map(t=>t.toolId)).size===2&&
    new Set(state.tools.map(t=>t.mode)).size===2,'two distinct persistent tool behaviors are required');
  const displays=state.tools.map(t=>checkedRehearsalTool(t,`${state.episodeId}:${state.frameIndex}`));
  const bounds=state.recordedEpisodeBounds;
  need(bounds?.caseHash===state.caseHash&&bounds?.episodeId===state.episodeId&&Array.isArray(bounds.boundsRasMm)&&bounds.boundsRasMm.length===2&&
    bounds.boundsRasMm.every(p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite))&&
    bounds.boundsRasMm[0].every((n,i)=>n<=bounds.boundsRasMm[1][i]),'episode camera envelope is invalid');
  for(const display of displays)for(const [point,radius] of [[display.shaftStart,display.shaftRadius],[display.shaftEnd,display.shaftRadius],[display.shaftEnd,display.tipRadius],[display.tip,display.tipRadius]] as const)
    for(let a=0;a<3;a++)need(point[a]-radius>=bounds.boundsRasMm[0][a]-1e-6&&point[a]+radius<=bounds.boundsRasMm[1][a]+1e-6,'camera envelope omits a recorded capsule');
  return {displays,bounds,effectMask,voidMask};
}
