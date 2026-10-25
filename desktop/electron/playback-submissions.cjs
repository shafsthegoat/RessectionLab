'use strict';

/** Match returned frames to actual CPU submissions, never to speculative IPC.
 * Main-process response completion is translated onto the renderer clock.
 * Refinement draws collapse only when consecutive; returning to an older state
 * remains visible in the sequence and fails the benchmark's exact-order check.
 */
function submittedPlayback(requests,frames,{position,mainToRendererOffset,startedRenderer,command='seekFrame'}){
  const delivered=requests.filter(row=>row.command===command&&!row.error&&row.position===position&&
    Number.isInteger(row.frameIndex)&&typeof row.stateId==='string'&&Number.isFinite(row.started)&&Number.isFinite(row.ms));
  const result=[];
  for(const frame of frames){
    if(!Number.isFinite(frame.at)||frame.at<startedRenderer)continue;
    const identity=frame.surgicalFrame;
    if(!identity||identity.position!==position)continue;
    const response=delivered.find(row=>row.branchId===identity.branchId&&row.position===identity.position&&
      row.frameIndex===identity.frameIndex&&row.stateId===frame.surgicalStateId&&
      row.started+row.ms-mainToRendererOffset<=frame.at);
    const previous=result.at(-1);
    if(!response||(previous?.branchId===response.branchId&&previous.position===response.position&&previous.frameIndex===response.frameIndex&&previous.stateId===response.stateId))continue;
    result.push({...response,firstSubmittedAt:frame.at,submissionSerial:frame.serial});
  }
  return result;
}

module.exports={submittedPlayback};
