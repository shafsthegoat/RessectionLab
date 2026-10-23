import test from 'node:test';
import assert from 'node:assert/strict';
import {planReplayPosition,planDecisionReason,planDecisionAtPosition} from '../src/research-planning-data.ts';

test('aligns decision replay while rejected commands do not advance native history',()=>{
  const plan={prefixActions:12,stopReason:'exact_execution_rejected',decisions:[{actionId:'z-',status:'accepted'},{actionId:'z-',status:'rejected',reason:'SHAFT_COLLISION'}]};
  assert.equal(planReplayPosition(plan,0),12);assert.equal(planReplayPosition(plan,1),13);
  assert.equal(planReplayPosition(plan,2),13);assert.equal(planReplayPosition(plan,6),13);
  assert.match(planDecisionReason(plan,6),/Execution refused: SHAFT_COLLISION/);
  assert.throws(()=>planReplayPosition(plan,-1));assert.throws(()=>planReplayPosition(plan,1.5));
});
test('STOP stays distinct from task completion and truncated limits',()=>{
  const stopped={prefixActions:3,stopReason:'policy_stop',decisions:[{actionId:'STOP',status:'accepted'}]};
  assert.match(planDecisionReason(stopped,0),/Shared/);assert.match(planDecisionReason(stopped,1),/Residual target remains unassessed/);
  assert.equal(planReplayPosition(stopped,6),4);
  const capped={...stopped,stopReason:'decision_limit',decisions:[{actionId:'apply_here',status:'accepted'}]};
  assert.match(planDecisionReason(capped,6),/decision limit/);
});

test('long executable strategies replay their full accepted command history',()=>{
  const plan={prefixActions:48,stopReason:'sequence_complete',decisions:Array.from({length:29},(_,index)=>({actionId:'rescue '+index,status:index===20?'rejected':'accepted'}))};
  assert.equal(planReplayPosition(plan,29),76);
  assert.equal(planReplayPosition(plan,20),68);
  assert.throws(()=>planReplayPosition(plan,30));
});

test('reopened endpoints and external scrubs recover the accepted decision annotation',()=>{
  const plan={prefixActions:48,decisions:Array.from({length:25},()=>({status:'accepted'}))};
  assert.equal(planDecisionAtPosition(plan,73),25);
  assert.equal(planDecisionAtPosition(plan,52),4);
  assert.equal(planDecisionAtPosition(plan,47),0);
  const refused={prefixActions:48,decisions:[{status:'accepted'},{status:'rejected'},{status:'accepted'}]};
  assert.equal(planDecisionAtPosition(refused,49),2);
  assert.equal(planDecisionAtPosition(refused,50),3);
});
