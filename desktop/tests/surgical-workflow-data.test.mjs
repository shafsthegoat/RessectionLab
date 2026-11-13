import test from 'node:test';
import assert from 'node:assert/strict';
import {workflowReportCurrent,workflowPlannerOptions,preferredWorkflowPlanner,workflowOriginalTarget,workflowCoverage,preparationWorkflowAlternative,strategyWorkflowAlternative,workflowRefusalReason,approachWorkflowAlternative} from '../src/surgical-workflow-data.ts';
const response=()=>({branchId:'main',position:4,view:{stateId:'state',scene:{sceneId:'scene',environment:{target:{initialVolumeMm3:20}}}},preparationPlanningAvailability:{available:true},strategyPlanningAvailability:{available:true}});
const binding=()=>({current:true,sourceSceneId:'scene',startStateId:'state',prefixActions:4,sourcePosition:4});
const alternative=(id='treatment')=>({id,label:id,feasible:true,taskProgressMet:true,withdrawal:'complete',summary:{additionalTargetMm3:2,residualTargetMm3:8,additionalNonTargetMm3:0,elapsedSeconds:3}});
test('current report identity requires scene, state, prefix and optional source position',()=>{
 const r=response(),report=binding();assert(workflowReportCurrent(r,report));
 for(const patch of [{current:false},{sourceSceneId:'other'},{startStateId:'other'},{prefixActions:3},{sourcePosition:3}])assert.equal(workflowReportCurrent(r,{...report,...patch}),false);
 assert.equal(workflowReportCurrent(null,report),false);
});
test('current or historical meaningful treatment comparison outranks incidental preparation eligibility',()=>{
 const r=response();r.strategyPlanning={...binding(),publicPlan:{alternatives:[alternative()]}};
 assert.equal(preferredWorkflowPlanner(r),'strategy');
 r.strategyPlanning.current=false;assert.equal(preferredWorkflowPlanner(r),'strategy');
 assert.deepEqual(workflowPlannerOptions(r),['preparation','strategy']);
 r.preparationPlanning={...binding(),controls:{},rows:[{preparedAdmission:{},preparation:{schema:'resectionlab.source-exposure-preparation.v2',publicPlan:{decision:{operation:'PREPARE',selectedIndex:0},attempts:[{complete:true,failure:null,summary:{additionalTargetMm3:0,additionalNonTargetMm3:12,elapsedSeconds:7,withdrawn:true}}]}},comparison:{publicPlan:{decision:{operation:'COMPARE',selectedId:'treatment'},alternatives:[alternative()]}}}]};
 assert.equal(preferredWorkflowPlanner(r),'preparation');
});
test('original source fraction stays unknown without its denominator and rejects narrower/impossible substitutes',()=>{
 const r=response();assert.equal(workflowOriginalTarget(r),null);
 r.sequenceOutcome={sceneId:'scene',originalTarget:{sourceVolumeMm3:null}};assert.equal(workflowOriginalTarget(r),null);
 r.sequenceOutcome.originalTarget.sourceVolumeMm3=100;assert.equal(workflowOriginalTarget(r),100);r.sequenceOutcome.sceneId='other';assert.equal(workflowOriginalTarget(r),null);r.sequenceOutcome.sceneId='scene';
 assert.equal(workflowCoverage(25,100),.75);assert.equal(workflowCoverage(101,100),null);assert.equal(workflowCoverage(0,0),null);
});
test('complete preparation alternative includes access costs once and never mixes owned residual with original coverage',()=>{
 const suffix=alternative();const row={id:'p',widthMm:1,reason:null,preparedAdmission:{},preparation:{schema:'resectionlab.source-exposure-preparation.v2',publicPlan:{decision:{operation:'PREPARE',selectedIndex:0},attempts:[{complete:true,failure:null,summary:{additionalTargetMm3:0,additionalNonTargetMm3:12,elapsedSeconds:7,withdrawn:true}}]}},comparison:{publicPlan:{decision:{operation:'COMPARE',selectedId:suffix.id,reason:'MIN_TIME'},alternatives:[suffix],approaches:{},rejections:[]}}};
 const report={controls:{},policies:{completePlanAware:{selectedPreparationId:'p'}},baselineBurden:{targetRemovedMm3:0}};
 const card=preparationWorkflowAlternative(row,report);assert.equal(card.nonTargetMm3,12);assert.equal(card.seconds,10);assert.equal(card.removedMm3,2);assert.equal(card.residualMm3,8);assert.equal(card.fractionRemoved,null);assert(card.recommended);
 row.comparison.publicPlan.decision.selectedId='stop';const stop=preparationWorkflowAlternative(row,report);assert.equal(stop.recommended,false);assert.equal(stop.status,'No complete treatment continuation');assert.equal(stop.nonTargetMm3,12);assert.equal(stop.residualMm3,null,'no original-source residual inferred from owned-label baseline');
 row.preparation.publicPlan.attempts[0].complete=false;const rejected=preparationWorkflowAlternative(row,report);assert.equal(rejected.removedMm3,null);assert.equal(rejected.fractionRemoved,null);assert.match(rejected.unknowns,/rejected previews receive no treatment credit/);
});
test('STOP and constraint refusals do not become treatment completion or removed-volume credit',()=>{
 const stop=strategyWorkflowAlternative(alternative('stop'),'stop');assert.match(stop.reason,/does not establish completed treatment/);assert.match(stop.residualLocation,/not recorded/);
 const rejected=strategyWorkflowAlternative({...alternative(),feasible:false},'stop');assert.equal(rejected.fractionRemoved,null);assert.equal(rejected.recommended,false);assert.match(rejected.reason,/cannot be installed/);
});

test('refusal text distinguishes missing evidence, bounded search, controller limits and exhausted budget',()=>{
 assert.match(workflowRefusalReason('REQUIRED_EVIDENCE_UNAVAILABLE'),/evidence is missing/);
 assert.match(workflowRefusalReason('NO_COMPLETE_TOOL_ACCESS'),/bounded candidates/);
 assert.match(workflowRefusalReason('PREPARATION_GENERATOR_REFUSAL'),/candidate generator/);
 assert.match(workflowRefusalReason('COMPUTATION_BUDGET'),/feasibility remains unresolved/);
 assert.match(workflowRefusalReason('ANATOMICAL_CONTEXT_LIMIT'),/declared source-context constraint/);
});

test('historical approach remains discoverable after replay changes the accepted scene identity',()=>{
 const r=response();r.interactivePlanning={...binding(),current:false,routes:[{id:'saved-route'}]};
 assert.equal(preferredWorkflowPlanner(r),'approach','incidental preparation eligibility must not hide the saved checkpoint');
});

test('inactive routes use only the exact accepted original source ledger for coverage and residual location',()=>{
 const r=response(),route={id:'inactive',label:'Inactive route',metrics:{pathLengthMm:10,actionCount:2}};r.sequenceOutcome={sceneId:'scene',stateId:'state',originalTarget:{sourceVolumeMm3:100,residualMm3:75,residualCentroidRasMm:[1,2,3]}};
 const card=approachWorkflowAlternative(route,r);assert.equal(card.fractionRemoved,.25);assert.equal(card.residualMm3,75);assert.match(card.residualLocation,/RAS 1.0 \/ 2.0 \/ 3.0 mm/);assert.equal(card.removedMm3,0);
 r.sequenceOutcome.stateId='different';const stale=approachWorkflowAlternative(route,r);assert.equal(stale.fractionRemoved,null);assert.match(stale.residualLocation,/not recorded/);
});
