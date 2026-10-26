import test from 'node:test';
import assert from 'node:assert/strict';
import {learnedExecution} from '../src/learned-execution-data.ts';
import {learnedBoundaryRequest} from '../src/learned-boundary-request.ts';
const response={branches:[{id:'trained',steps:5}]};
const report={schema:'resectionlab.patient-rescue-ppo-bc-audit.v1',method:'ppo_initialized_from_behavior_cloning',branchId:'trained',prefixActions:0,finalStateId:'sha256:'+'b'.repeat(64),checkpointSha256:'a'.repeat(64),observationSchema:'resectionlab.prepared-rescue-observation.v2',samePreparedInjuryOnly:true,promoted:false,auditPassed:true,actualPpoOptionTransitions:64,actualActorChoices:['coagulate_center','aspirate_pool','finish_verify']};
test('learned playback requires a real available branch, task scope and training provenance',()=>{
  assert.equal(learnedExecution(report,response).actualPpoOptionTransitions,64);
  for(const delta of [{branchId:'missing'},{actualPpoOptionTransitions:0},{checkpointSha256:'label only'},{samePreparedInjuryOnly:false},{promoted:true},{actualActorChoices:['scripted_sequence']},{observationSchema:'unknown'}])assert.equal(learnedExecution({...report,...delta},response),null);
  assert.equal(learnedExecution(report,null),null);
});
test('behavior cloning stays distinct from PPO and can retain an unpassed audit',()=>{
  const bc={...report,schema:'resectionlab.patient-rescue-bc.v1',method:'frozen_behavior_cloning',actualPpoOptionTransitions:undefined,actualRlOptionTransitions:0,actualSupervisedOptimizerSteps:1000,auditPassed:false};
  assert.equal(learnedExecution(bc,response).method,'frozen_behavior_cloning');
  assert.equal(learnedExecution(bc,response).auditPassed,false);
  assert.equal(learnedExecution({...bc,actualRlOptionTransitions:64},response),null);
  assert.equal(learnedExecution({...bc,method:report.method},response),null);
});
test('imported learned playback reconstructs origin before seeking its policy boundary',()=>{
  const r={branchId:'trained',position:5,branches:[{id:'trained',steps:5}],verification:{readOnly:true,status:'unverified-import'}};
  assert.deepEqual(learnedBoundaryRequest(r,'trained',3),{command:'seekPrefix',branchId:'trained',position:0});
  const prefix={...r,position:0,verification:{readOnly:true,status:'reconstructed-prefix'}};
  assert.deepEqual(learnedBoundaryRequest(prefix,'trained',3),{command:'seekPrefix',branchId:'trained',position:3});
  assert.equal(learnedBoundaryRequest({...prefix,position:3},'trained',3),null);
  assert.throws(()=>learnedBoundaryRequest(prefix,'missing',3));
});
