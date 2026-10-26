'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const {frozenCusaArgs,validateFrozenResult,loadFrozenWorkspace}=require('./frozen-cusa.cjs');
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
  assert.deepEqual(args,['/repo/scripts/execute_patient_sector_policy.py','--source','/repo/source','--access-method','axis0-access','--source-role','DEVELOPMENT','--checkpoint','/repo/model.pt','--output',output]);
  assert.equal(args.includes('--fit'),false);
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
