'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const {frozenCusaArgs,validateFrozenResult,validateMatchingRuntimeAdmission,loadFrozenWorkspace}=require('./frozen-cusa.cjs');
const output='/repo/testcases/run/execution';
const valid=()=>({status:'evaluated',optimizerUpdates:0,task:{role:'DEVELOPMENT'},
  source:{unchanged:true},exactReplay:{exactNativeReplay:true},
  derived:{reopenedExact:true,workspacePath:output+'/frozen-transfer.ressectionlab',sha256:'a'.repeat(64)},
  execution:{status:'abstained',taskGateMet:false,cleanupComplete:true,
    deltaTargetMm3:10,deltaNonTargetMm3:2,admittedNonStopActions:1}});
test('desktop runner invokes only frozen DEVELOPMENT inference with separate argv',()=>{
  const args=frozenCusaArgs({repo:'/repo',source:'/repo/a case',checkpoint:'/repo/model.pt',branchId:'access-a',output});
  assert.deepEqual(args,['/repo/scripts/run_patient_cusa_learning.py','--source','/repo/a case',
    '--branch','access-a','--source-role','DEVELOPMENT','--frozen-model','/repo/model.pt','--output',output]);
  assert.throws(()=>frozenCusaArgs({branchId:''}),/saved access/);
});
test('hierarchical execution uses the selected access method and its cold companion, without fitting',()=>{
  const args=frozenCusaArgs({repo:'/repo',source:'/repo/source',checkpoint:'/repo/model.pt',branchId:'axis0-access',output,kind:'sector'});
  assert.deepEqual(args,['/repo/scripts/execute_patient_sector_policy.py','--source','/repo/source','--access-method','axis0-access','--source-role','DEVELOPMENT','--checkpoint','/repo/model.pt','--output',output,'--allow-matching-runtime']);
  assert.equal(args.includes('--fit'),false);
});
test('explicit archived runtime admission binds commit, checkpoint, loaded producer and executor',()=>{
  const producer={runnerSourceSha256:'a'.repeat(64),authoritativeEngineFingerprint:'b'.repeat(64)};
  const runtime='/repo/testcases/outputs/integrated-goal-20261026/joint-frozen-runtime-v1/runtime/source';
  const archive='bbaeaa5098c918b566523b4c098db96c742454f593a772fa84e597de1d51b3e2';
  const manifestSha='ee0dfb0e4cff704d5f42b2669efb3a24db0da615f01cc4ee631b525cb6d12ccb';
  const commit='01e5a365abb7cf9b158a27d49f1e1bfa83e593a2';
  const exe='c'.repeat(64),checkpointSha='d'.repeat(64),actorSha='e'.repeat(64);
  const manifest={schema:'resectionlab.frozen-matching-runtime-files.v1',gitCommit:commit,
    sourceArchiveSha256:archive,producerIdentity:producer,
    files:{'scripts/execute_patient_sector_policy.py':exe}};
  const admission={schema:'resectionlab.frozen-matching-runtime-admission.v1',
    reason:'current_producer_differs_from_frozen_checkpoint',sourceArchiveSha256:archive,
    gitCommit:commit,manifestSha256:manifestSha,runtimeSource:runtime,
    canonicalTestcases:'/repo/testcases',checkpointPath:'/repo/model.pt',
    checkpointFileSha256:checkpointSha,checkpointActorSha256:actorSha,
    executorSourceSha256:exe,currentProducerIdentity:{changed:true},matchingProducerIdentity:producer};
  const receipt={producerIdentity:producer,executorSourceSha256:exe,
    model:{path:'/repo/model.pt',fileSha256:checkpointSha,actorSha256:actorSha}};
  const context={repo:'/repo',checkpoint:'/repo/model.pt',checkpointSha};
  assert.equal(validateMatchingRuntimeAdmission(admission,receipt,manifest,context).gitCommit,commit);
  for(const [badAdmission,badReceipt,badManifest] of [
    [{...admission,gitCommit:'other'},receipt,manifest],
    [{...admission,matchingProducerIdentity:{changed:true}},receipt,manifest],
    [{...admission,checkpointFileSha256:'wrong'},receipt,manifest],
    [admission,{...receipt,executorSourceSha256:'wrong'},manifest],
    [admission,{...receipt,model:{...receipt.model,actorSha256:'wrong'}},manifest],
    [admission,receipt,{...manifest,files:{'scripts/execute_patient_sector_policy.py':'wrong'}}]])
    assert.throws(()=>validateMatchingRuntimeAdmission(badAdmission,badReceipt,badManifest,context),/runtime admission/);
});
test('hierarchical success additionally requires both declared sectors and total exposure cap',()=>{
  const receipt=valid();receipt.execution={...receipt.execution,status:'success',taskGateMet:true,
    deltaTargetMm3:84,deltaNonTargetMm3:3,admittedNonStopActions:2,totalNonTargetMm3:138,
    perSectorRemovedMm3:{near:12,far:19}};
  assert.equal(validateFrozenResult(receipt,output,'sector').taskGateMet,true);
  receipt.execution.perSectorRemovedMm3.near=9;
  assert.throws(()=>validateFrozenResult(receipt,output,'sector'),/sector execution/);
});
test('an unpassed task with completed cleanup stays unpassed at desktop admission',()=>{
  const result=validateFrozenResult(valid(),output);
  assert.equal(result.status,'abstained');assert.equal(result.taskGateMet,false);assert.equal(result.cleanupComplete,true);
  const incorrect=valid();incorrect.execution.status='success';incorrect.execution.taskGateMet=true;
  assert.throws(()=>validateFrozenResult(incorrect,output),/inconsistent task gate/);
});
test('desktop rejects updates, unverified replay, reserved roles and escaping export paths',()=>{
  for(const mutate of [r=>r.optimizerUpdates=1,r=>r.task.role='TEST',
    r=>r.exactReplay.exactNativeReplay=false,r=>r.source.unchanged=false,
    r=>r.derived.reopenedExact=false,r=>r.derived.workspacePath='/other/export.ressectionlab',
    r=>r.execution.taskGateMet=true,r=>delete r.execution.deltaTargetMm3,
    r=>r.execution.deltaNonTargetMm3=NaN,r=>r.execution.admittedNonStopActions=undefined]){
    const receipt=valid();mutate(receipt);assert.throws(()=>validateFrozenResult(receipt,output));
  }
});
test('cancellation during a derived load restores the source before rejecting admission',async()=>{
  const controller=new AbortController(),loaded=[];
  const load=async filename=>{loaded.push(filename);if(filename==='derived')controller.abort();return filename;};
  await assert.rejects(loadFrozenWorkspace(load,'source','derived',controller.signal),{name:'AbortError'});
  assert.deepEqual(loaded,['derived','source']);
});
test('a failed derived load restores the source and a successful load needs no rollback',async()=>{
  const loaded=[];
  await assert.rejects(loadFrozenWorkspace(async filename=>{loaded.push(filename);if(filename==='derived')throw Error('asset failure');},'source','derived'),/asset failure/);
  assert.deepEqual(loaded,['derived','source']);
  assert.equal(await loadFrozenWorkspace(async filename=>filename,'source','derived'),'derived');
});
