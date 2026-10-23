import test from 'node:test';
import assert from 'node:assert/strict';
import {rescueLearningReport,learningEndpointVerified} from '../src/rescue-learning-data.ts';

const stateId='sha256:'+('a'.repeat(64));
const row={branchId:'rescue-trained-seed-17',method:'trained',seed:17,success:false,stateId,
  reward:-.2,decisions:12,reason:'option_decision_limit',modeledField:[1.,1.,.1,2.]};
const response={branchId:row.branchId,branches:[{id:row.branchId,steps:87}],position:87,
  view:{stateId,metrics:{fluidPresentMm3:1.,activeVesselWounds:1}}};
const report={schema:'resectionlab.patient-rescue-learning.v1',assessments:[row],limitations:['Development injury only'],settings:{train_decisions:64,search_decisions:64}};

test('saved experiment descriptions never authorize an imported or partial endpoint',()=>{
  assert.ok(rescueLearningReport(report,response));
  assert.equal(learningEndpointVerified(row,response),true);
  for(const changed of [{verification:{readOnly:true}},{readOnly:true},{position:86},
      {view:{stateId:'sha256:'+('b'.repeat(64)),metrics:response.view.metrics}},{branchId:'another'},
      {view:{stateId,metrics:{...response.view.metrics,fluidPresentMm3:2.}}}])
    assert.equal(learningEndpointVerified(row,{...response,...changed}),false);
});

test('learning rows require current branches, finite quantities and unique identities',()=>{
  assert.equal(rescueLearningReport(report,{...response,branches:[]}),null);
  assert.equal(rescueLearningReport({...report,assessments:[row,row]},response),null);
  assert.ok(rescueLearningReport({...report,assessments:[row,{...row,method:'untrained'}]},response));
  assert.equal(rescueLearningReport({...report,assessments:[{...row,modeledField:[NaN,0,0,0]}]},response),null);
  assert.equal(rescueLearningReport({...report,assessments:[{...row,success:'yes'}]},response),null);
});
