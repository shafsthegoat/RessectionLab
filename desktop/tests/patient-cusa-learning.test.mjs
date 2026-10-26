import test from 'node:test';
import assert from 'node:assert/strict';
import {patientCusaExecutions} from '../src/patient-cusa-learning-data.ts';
const row={schema:'resectionlab.patient-cusa-learning.v1',method:'ppo',branchId:'policy',parentPosition:9,sourceSceneId:'scene',sourceStateId:'sha256:'+'a'.repeat(64),modelSha256:'b'.repeat(64),checkpointFileSha256:'c'.repeat(64),trainingReceiptSha256:'sha256:'+'d'.repeat(64),observationSchema:'surgical-rl.v2-tool-frame-task-v1',optimizerUpdates:2,actorDecisions:['inward-2','inward-2','STOP'],deltaTargetMm3:36,deltaNonTargetMm3:0,cleanupComplete:true,admittedNonStopActions:2,status:'success'};
const response={branches:[{id:'policy',steps:13}],view:{scene:{sceneId:'scene'}}};
test('fresh CUSA display binds branch, source, genuine fitting and the declared success gate',()=>{
  assert.equal(patientCusaExecutions({learnedExecutions:[row]},response).length,1);
  for(const delta of [{sourceSceneId:'other'},{branchId:'missing'},{optimizerUpdates:0},{checkpointFileSha256:'label'},{cleanupComplete:false},{deltaTargetMm3:5},{admittedNonStopActions:1},{deltaNonTargetMm3:21},{actorDecisions:['prewritten']}])assert.deepEqual(patientCusaExecutions({learnedExecutions:[{...row,...delta}]},response),[]);
  assert.deepEqual(patientCusaExecutions({learnedExecutions:[row,row]},response),[]);
});
test('a retained failed gate stays incomplete and cannot become a success label',()=>{
  const failed={...row,status:'incomplete',cleanupComplete:false,deltaTargetMm3:0,admittedNonStopActions:0};
  assert.equal(patientCusaExecutions({learnedExecutions:[failed]},response)[0].status,'incomplete');
});
