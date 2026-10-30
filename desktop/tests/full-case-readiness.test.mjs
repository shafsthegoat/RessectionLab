import test from 'node:test';
import assert from 'node:assert/strict';
import {boundFullCaseReadiness,fullCaseScenarioAvailable} from '../src/full-case-readiness.ts';
const response=()=>({caseHash:'patient',view:{stateId:'state',scene:{sceneId:'scene'}},fullCaseReadiness:{schema:'resectionlab.full-case-readiness.v1',caseHash:'patient',sceneId:'scene',stateId:'state',status:'ready',availableActions:['inspect','primary','vascular'],blockers:[],recovery:null,protectionContractId:'contract'}});
test('full-case eligibility is bound to the accepted case, scene and state',()=>{
  assert(fullCaseScenarioAvailable(response(),'patient','primary'));
  assert.equal(boundFullCaseReadiness(response(),'another-patient'),null);
  for(const field of ['caseHash','sceneId','stateId','schema']){
    const r=response();r.fullCaseReadiness[field]='stale';
    assert.equal(boundFullCaseReadiness(r,'patient'),null,field);
    assert.equal(fullCaseScenarioAvailable(r,'patient','primary'),false,field);
  }
  assert.equal(fullCaseScenarioAvailable({...response(),fullCaseReadiness:undefined},'patient','primary'),false);
});
test('blocked, inconsistent ready, missing action and read-only receipts cannot authorize computation',()=>{
  const blocked=response();blocked.fullCaseReadiness.status='blocked';blocked.fullCaseReadiness.availableActions=['inspect','attachProtection'];
  assert(boundFullCaseReadiness(blocked,'patient'),'blocked evidence remains available for recovery');
  assert.equal(fullCaseScenarioAvailable(blocked,'patient','primary'),false);
  const r=response();r.fullCaseReadiness.availableActions=['primary'];
  assert(fullCaseScenarioAvailable(r,'patient','primary'));assert.equal(fullCaseScenarioAvailable(r,'patient','vascular'),false);
  r.fullCaseReadiness.blockers=[{code:'missing',message:'Missing evidence',details:[]}];assert.equal(fullCaseScenarioAvailable(r,'patient','primary'),false);
  for(const flags of [{readOnly:true},{playback:{readOnly:true}},{verification:{readOnly:true}}])assert.equal(fullCaseScenarioAvailable({...response(),...flags},'patient','primary'),false);
});
