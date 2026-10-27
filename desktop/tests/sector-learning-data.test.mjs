import test from 'node:test';
import assert from 'node:assert/strict';
import {learnedSectorDecisionEvidence,learnedSectorExecutions} from '../src/sector-learning-data.ts';
const row={schema:'resectionlab.patient-sector-learning.v1',sourceRole:'DEVELOPMENT',executionSourceSceneId:'scene',executionAccessPlanSeal:'sha256:'+'e'.repeat(64),executionAccessMethod:'superior-access',branchId:'actor',checkpointFileSha256:'a'.repeat(64),actorSha256:'b'.repeat(64),trainingReceiptSha256:'sha256:'+'c'.repeat(64),observationSchema:'published-surgical-rl-v2-plus-sector-ledger-v1',optimizerUpdates:2,nativeActions:14,prefixActions:0,exactNativeReplay:true,finalStateId:'sha256:'+'d'.repeat(64),status:'success',taskGateMet:true,cleanupComplete:true,deltaTargetMm3:84,deltaNonTargetMm3:3,totalNonTargetMm3:138,perSectorRemovedMm3:{near:12,far:19},highLevelDecisions:['corridor:left','far:left','near:left','STOP']};
const response={view:{scene:{sceneId:'scene'}},branches:[{id:'actor',steps:14}]};
test('learned hierarchy display binds source, native actions and unchanged sector task gate',()=>{
  assert.equal(learnedSectorExecutions({learnedSectorExecutions:[row]},response).length,1);
  for(const delta of [{executionSourceSceneId:'other'},{nativeActions:13},{perSectorRemovedMm3:{near:9,far:19}},{taskGateMet:false},{totalNonTargetMm3:201},{cleanupComplete:false},{highLevelDecisions:['script']},{checkpointFileSha256:'label'}])assert.deepEqual(learnedSectorExecutions({learnedSectorExecutions:[{...row,...delta}]},response),[]);
});
test('failed task and failed cleanup remain separate recorded facts',()=>{
  const failed={...row,status:'incomplete',taskGateMet:false,cleanupComplete:false,deltaTargetMm3:22,deltaNonTargetMm3:16,perSectorRemovedMm3:{near:0,far:2},highLevelDecisions:['corridor:left','far:left']};
  const displayed=learnedSectorExecutions({learnedSectorExecutions:[failed]},response)[0];
  assert.equal(displayed.taskGateMet,false);assert.equal(displayed.cleanupComplete,false);
  assert.deepEqual(learnedSectorExecutions({learnedSectorExecutions:[{...failed,status:'success',taskGateMet:true}]},response),[]);
});
test('saved refusal probes stay distinct from actor observations and forced cleanup',()=>{
  const selected=['corridor:left','far:left'];
  const decisions=[
    {index:0,boundary:'intact_before_access',optionId:selected[0],observationHash:'sha256:'+'1'.repeat(64),nativeOption:{id:selected[0],status:'accepted',reason:null}},
    {index:1,boundary:'post_corridor',optionId:selected[1],observationHash:'sha256:'+'2'.repeat(64),nativeOption:{id:selected[1],status:'refused',reason:'ESCAPE_PREFLIGHT_CLEANUP_INCOMPLETE',escapePreflight:{status:'refused',previewCalls:9,committedCalls:2}}},
    {index:2,boundary:'forced_cleanup',optionId:'STOP',nativeOption:{id:'STOP',status:'accepted',reason:null}},
  ];
  const source={...row,highLevelDecisions:selected,policyDecisions:decisions};
  const evidence=learnedSectorDecisionEvidence(source);
  assert.equal(evidence.length,3);assert.equal(evidence[1].exitStatus,'refused');
  assert.equal(evidence[1].discardedPreviewCalls+evidence[1].discardedCommittedCalls,11);
  assert.equal(evidence[2].forcedCleanup,true);assert.equal(evidence[2].observationHash,null);
  assert.deepEqual(learnedSectorDecisionEvidence({...source,highLevelDecisions:['corridor:right','far:left']}),[]);
  for(const bad of [{observationHash:'future-preview'},{nativeOption:{...decisions[1].nativeOption,escapePreflight:{status:'refused',previewCalls:-1,committedCalls:2}}}]){
    const altered=structuredClone(decisions);altered[1]={...altered[1],...bad};
    assert.deepEqual(learnedSectorDecisionEvidence({...source,policyDecisions:altered}),[]);
  }
  assert.deepEqual(learnedSectorDecisionEvidence(row),[]); // Historical archives remain readable.
});
