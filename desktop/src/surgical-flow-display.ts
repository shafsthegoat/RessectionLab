import type {SurgicalFlowDisplay,SurgicalPlayback,SurgicalView} from './surgical-types';

export const MAX_FLOW_OPERATIONS=256;
export const MAX_FLOW_DISPLAY_BYTES=64*1024;
const hash=(value:unknown)=>typeof value==='string'&&/^sha256:[0-9a-f]{64}$/.test(value);
const finite=(value:unknown):value is number=>typeof value==='number'&&Number.isFinite(value)&&value>=0;
const keys=(value:unknown,expected:string[]):value is Record<string,unknown>=>!!value&&typeof value==='object'&&!Array.isArray(value)&&Object.keys(value).length===expected.length&&expected.every(key=>Object.hasOwn(value,key));
const envelope=['schema','sceneId','actionIndex','frameIndex','beforeStateId','stateId','startSeconds','endSeconds','truncated','operations'];
const operation=['kind','field','provenance','fromCell','toCell','volumeMm3','startSeconds','endSeconds'];

/** Optional legacy traces remain unavailable. A supplied trace must match the
 * demanded endpoint, never the UI's previously displayed (possibly sought) state.
 * Historical FREE occupancy is authenticated by the writer at the actual write;
 * an endpoint snapshot cannot reconstruct intermediate occupancy. */
export function checkedSurgicalFlowDisplay(raw:unknown,view:SurgicalView,p:Pick<SurgicalPlayback,'actionIndex'|'frameIndex'|'observedTimeSeconds'|'frames'>):SurgicalFlowDisplay|undefined {
  const flow=checkedFlowDisplayForView(raw,view);if(!flow)return undefined;
  const prior=p.frames[Math.max(0,p.frameIndex-1)];
  if(flow.actionIndex!==p.actionIndex||flow.frameIndex!==p.frameIndex||!prior||flow.startSeconds!==prior.observedTimeSeconds||flow.endSeconds!==p.observedTimeSeconds)
    throw Error('Recorded field transfers differ from the source-bound observation.');
  return flow;
}

/** Defense at the GPU boundary, also used by the source-bound wire check. */
export function checkedFlowDisplayForView(raw:unknown,view:SurgicalView):SurgicalFlowDisplay|undefined {
  if(raw===undefined)return undefined;
  const need=(condition:unknown)=>{if(!condition)throw Error('Recorded field transfers differ from the source-bound observation.');};
  need(keys(raw,envelope));const flow=raw as SurgicalFlowDisplay;
  need(flow.schema==='resectionlab.recorded-field-transfers.v1'&&hash(flow.sceneId)&&flow.sceneId===view.scene.sceneId&&hash(flow.beforeStateId)&&hash(flow.stateId)&&flow.stateId===view.stateId);
  need(Number.isSafeInteger(flow.actionIndex)&&flow.actionIndex>=0&&Number.isSafeInteger(flow.frameIndex)&&flow.frameIndex>=0);
  need(finite(flow.startSeconds)&&finite(flow.endSeconds)&&flow.endSeconds>=flow.startSeconds&&flow.endSeconds===view.metrics.elapsedSeconds&&typeof flow.truncated==='boolean');
  need(Array.isArray(flow.operations)&&flow.operations.length<=MAX_FLOW_OPERATIONS);
  need(flow.startSeconds!==flow.endSeconds||flow.operations.length===0);
  const shape=view.scene.shape;need(shape.length===3&&shape.every(n=>Number.isSafeInteger(n)&&n>0));
  const count=shape[0]*shape[1]*shape[2];need(Number.isSafeInteger(count));
  const cell=(id:unknown):id is number=>typeof id==='number'&&Number.isSafeInteger(id)&&id>=0&&id<count;
  const coordinates=(id:number)=>[Math.floor(id/(shape[1]*shape[2])),Math.floor(id/shape[2])%shape[1],id%shape[2]];
  let previous=-Infinity,previousEnd=-Infinity;
  for(const op of flow.operations){
    need(keys(op,operation)&&['transfer','uptake','deposit','escape'].includes(op.kind)&&['fluid','debris'].includes(op.field));
    need(op.field==='debris'?op.provenance==='debris':['mixed','irrigation','wound','declared-source'].includes(op.provenance));
    need(finite(op.volumeMm3)&&op.volumeMm3>0&&finite(op.startSeconds)&&finite(op.endSeconds)&&op.startSeconds>=flow.startSeconds&&op.endSeconds<=flow.endSeconds&&op.endSeconds>=op.startSeconds&&op.startSeconds>=previous&&op.endSeconds>=previousEnd);previous=op.startSeconds;previousEnd=op.endSeconds;
    if(op.kind==='transfer'){
      need(cell(op.fromCell)&&cell(op.toCell));
      const left=coordinates(op.fromCell!),right=coordinates(op.toCell!);need(left.reduce((sum,n,i)=>sum+Math.abs(n-right[i]),0)===1);
      need(op.provenance===(op.field==='fluid'?'mixed':'debris'));
    }else if(op.kind==='deposit')need(op.fromCell===null&&cell(op.toCell));
    else need((cell(op.fromCell)||(op.kind==='escape'&&op.fromCell===null))&&op.toCell===null&&op.provenance===(op.field==='fluid'?'mixed':'debris'));
  }
  need(flow.frameIndex!==0||(flow.beforeStateId===flow.stateId&&flow.operations.length===0&&flow.startSeconds===flow.endSeconds));
  need(new TextEncoder().encode(JSON.stringify(flow)).byteLength<=MAX_FLOW_DISPLAY_BYTES);
  return flow;
}
