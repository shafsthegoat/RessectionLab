import {fixture} from './cranial-workflow-fixture.mjs';
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {cranialPlanningPresent,cranialAvailability,cranialGenerationIssue,cranialComparison,cranialWorkflowAlternative} from '../src/cranial-workflow-data.ts';
import {workflowPlannerOptions,preferredWorkflowPlanner} from '../src/surgical-workflow-data.ts';
import {SurgicalComparisonContext} from '../src/surgical-comparison-context.ts';

test('cranial owns primary task including blocked skull; exposed legacy stays unchanged',()=>{
 const r=fixture();assert.deepEqual(workflowPlannerOptions(r),['cranial']);assert.equal(preferredWorkflowPlanner(r),'cranial');delete r.cranialPlanningAvailability;delete r.cranialPlanning;assert(cranialPlanningPresent(r));assert.match(cranialGenerationIssue(r),/qualified opening registry/);assert.deepEqual(workflowPlannerOptions(r),['cranial']);r.view.scene.regions=[];r.cranialPlanning=null;r.cranialPlanningAvailability=null;assert(!cranialPlanningPresent(r));assert.deepEqual(workflowPlannerOptions(r),['preparation']);
});
test('qualification and read-only blockers remain despite acknowledgement',()=>{
 const r=fixture();assert.equal(cranialGenerationIssue(r),null);r.cranialPlanningAvailability.available=false;r.cranialPlanningAvailability.blockers=[{code:'UNKNOWN_INTERSECTION',message:'Required cover intersects unknown source tissue'}];assert.match(cranialGenerationIssue(r),/UNKNOWN_INTERSECTION/);r.cranialPlanningAvailability.available=true;r.playback={readOnly:true};assert.match(cranialGenerationIssue(r),/editable/);
});
test('comparison binds branches/root scenes/endpoints/complete calls',()=>{
 assert(cranialComparison(fixture()));for(const change of [r=>r.branches.pop(),r=>r.branches[1].steps=11,r=>r.cranialPlanning.alternatives[0].rootId='invented',r=>r.cranialPlanning.alternatives[0].sceneId='other',r=>r.cranialPlanning.alternatives[0].sequenceOutcome.stateId='other',r=>r.cranialPlanning.cost.physicalCalls++,r=>r.cranialPlanning.sourceStateId='other']){const r=fixture();change(r);assert.equal(cranialComparison(r),null);}
});
test('bone-only, unfinished exit, over-budget or malformed credit cannot be complete',()=>{
 for(const change of [x=>x.wholeExitComplete=false,x=>x.sequenceOutcome.withdrawalComplete=false,x=>x.ledger.withinLimits=false,x=>x.goalAssessment.creditedOriginalTargetCells=0,x=>x.goalAssessment.originalTargetTreatmentFraction=.99,x=>x.ledger.wholeNonTargetRemovedMm3=2,x=>x.goalAssessment.originalTargetVolumeMm3=10]){const r=fixture();change(r.cranialPlanning.alternatives[0]);assert.equal(cranialComparison(r),null);}
 const r=fixture(),row=r.cranialPlanning.alternatives[0];Object.assign(row.goalAssessment,{creditedOriginalTargetCells:0,removedOriginalTargetCells:0,residualOriginalTargetCells:100,residualOriginalTargetMm3:12.5,originalTargetFractionRemoved:0,originalTargetRemovedMm3:0,originalTargetTreatmentFraction:0,originalTargetTreatmentMm3:0,completedFrozenTask:false,reasons:['NO_TARGET_WORK']});row.completedFrozenTask=false;r.cranialPlanning.selectedBranchId=null;assert(cranialComparison(r));const card=cranialWorkflowAlternative(row,null);assert.match(card.status,/task incomplete/);assert.equal(card.removedMm3,0);assert.equal(card.residualMm3,12.5);assert.equal(card.nonTargetMm3,25);assert.equal(card.fractionRemoved,0);assert(!card.recommended);
});
test('cards preserve credited-vs-removed quantities, full denominator and missing evidence',()=>{
 const r=fixture(),row=r.cranialPlanning.alternatives[0],g=row.goalAssessment;g.creditedOriginalTargetCells=40;g.originalTargetTreatmentFraction=.4;g.originalTargetTreatmentMm3=5;assert(cranialComparison(r));const card=cranialWorkflowAlternative(row,'actual-branch');assert.equal(card.removedMm3,5);assert.equal(card.fractionRemoved,.4);assert.equal(card.residualMm3,2.5);assert.equal(card.nonTargetMm3,25);assert.equal(card.quantityScope,'whole-sequence');assert.match(card.unknowns,/motor mapping/);assert.match(card.residualLocation,/20 \/ 100/);
});
test('root IDs are unique; frame metadata is source-bound and explicit null clears',()=>{
 const r=fixture();r.cranialPlanningAvailability.roots.push({...r.cranialPlanningAvailability.roots[0]});assert.equal(cranialAvailability(r),null);const source={},context=new SurgicalComparisonContext(),original=fixture();context.accept(original,source,false);const frame={...original,playback:{readOnly:true}};delete frame.cranialPlanning;delete frame.cranialPlanningAvailability;const h=context.accept(frame,source,true);assert.equal(h.cranialPlanning,original.cranialPlanning);assert.equal(h.cranialPlanningAvailability,original.cranialPlanningAvailability);assert.equal(context.accept({...frame,cranialPlanning:null},source,true).cranialPlanning,null);assert.equal(context.accept({...frame,caseHash:'another'},source,true).cranialPlanning,undefined);
});

test('versioned policy preserves selected partial treatment and rejects changed harm evidence',()=>{
 const r=fixture(),c=r.cranialPlanning,row=c.alternatives[0];
 const policy={schema:'resectionlab.supratentorial-procedure-policy.v1',policyId:'declared-task',policyHash:'bound-policy',available:true,startStage:'post-scalp',requiredCoverage:['brainstem','cerebellum','intracranial'],scenarios:['nominal'],limitations:['Nominal only']};
 c.schema='resectionlab.cranial-strategy-comparison.v2';c.procedurePolicy=policy;r.cranialPlanningAvailability.procedurePolicy=structuredClone(policy);
 c.selectionRule='max_treatment_within_harm_bounds';c.stopReason='BOUNDED_INVENTORY_OR_HORIZON_STOP_WITH_RESIDUAL';
 row.sequenceOutcome.schema='resectionlab.sequence-outcome.v2';row.ledger.neuralRemovedMm3=0;
 row.sequenceOutcome.dimensions.neuralRemovedMm3={unit:'mm3',status:'source_count',cumulative:0};
 row.completedFrozenTask=row.goalAssessment.completedFrozenTask=false;row.goalAssessment.reasons=['ORIGINAL_SPATIAL_PROGRESS_BELOW_GOAL'];
 assert(cranialComparison(r));assert.equal(c.selectedBranchId,'actual-branch');
 c.procedurePolicy={...policy,policyHash:'changed'};assert.equal(cranialComparison(r),null);
 c.procedurePolicy=policy;row.sequenceOutcome.dimensions.neuralRemovedMm3.cumulative=null;assert.equal(cranialComparison(r),null);
});

function canonical(){
 const r=fixture(),c=r.cranialPlanning,row=c.alternatives[0];c.schema='resectionlab.cranial-strategy-comparison.v3';
 const e={schema:'resectionlab.sequential-task-config.v1',configHash:'shared-config',classification:'successful-treatment',completedTask:true,completeOriginalTargetRemoval:false,validWithdrawal:true,cleanupComplete:true,pending:[],fluidMm3:0,debrisMm3:0,terminated:true,truncated:false};
 Object.assign(row,{candidateKey:'checked-program',validStop:true,recommendationEligible:true,endpointAssessment:e});row.goalAssessment.endpointAssessment=structuredClone(e);
 c.endpointContract={schema:'resectionlab.canonical-cranial-endpoint.v1',configHash:'shared-config'};
 c.selectionEvidence={schema:'resectionlab.opening-forest-priority.v1',candidateCount:1,eligibleCount:1,selectedCandidateKey:'checked-program',reason:'max_treatment_within_harm_bounds',candidates:[{candidateKey:'checked-program',rootId:row.rootId,sceneId:row.sceneId,finalStateId:row.finalStateId,eligible:true}]};return r;
}
test('canonical endpoint and complete-forest certificate bind adapter recommendation',()=>{
 const r=canonical();assert(cranialComparison(r));assert(cranialWorkflowAlternative(r.cranialPlanning.alternatives[0],'actual-branch').recommended);
 for(const change of [c=>delete c.endpointContract,c=>c.selectionEvidence.candidateCount++,c=>c.selectionEvidence.selectedCandidateKey='other',c=>c.selectionEvidence.candidates[0].eligible=false,c=>c.alternatives[0].candidateKey='other']){const altered=canonical();change(altered.cranialPlanning);assert.equal(cranialComparison(altered),null);}
});
test('dirty or truncated canonical endpoint cannot retain an old completed flag',()=>{
 for(const kind of ['debris','fluid','truncated']){const r=canonical(),row=r.cranialPlanning.alternatives[0],e=row.endpointAssessment;e.completedTask=false;e.classification=kind==='truncated'?'truncated':'in-progress';e.truncated=kind==='truncated';e.terminated=false;if(kind!=='truncated'){e.cleanupComplete=false;e.validWithdrawal=false;e.pending=[kind];e[kind+'Mm3']=1;}assert.equal(cranialComparison(r),null);}
});
test('clean canonical STOP with residual can be selected without becoming completed treatment',()=>{
 const r=canonical(),row=r.cranialPlanning.alternatives[0];row.completedFrozenTask=row.goalAssessment.completedFrozenTask=false;row.goalAssessment.reasons=['ORIGINAL_SPATIAL_PROGRESS_BELOW_GOAL'];row.endpointAssessment.completedTask=false;row.endpointAssessment.classification='valid-stop-with-residual';
 assert(cranialComparison(r));assert(cranialWorkflowAlternative(row,'actual-branch').recommended);
 r.cranialPlanning.selectedBranchId=null;r.cranialPlanning.decision='STOP';r.cranialPlanning.selectionEvidence.selectedCandidateKey=null;r.cranialPlanning.selectionEvidence.reason='unresolved_exposure_tradeoff_stop_with_residual';assert(cranialComparison(r));
});
test('historical comparisons retain inspection quantities and never imply current recommendation',()=>{
 const original=fixture(),c=cranialComparison(original);assert(c.historicalInspectionOnly);assert.equal(c.selectedBranchId,null);assert.equal(c.historicalSelectedBranchId,'actual-branch');assert(!c.alternatives[0].completedFrozenTask);assert(!cranialWorkflowAlternative(c.alternatives[0],'actual-branch').recommended);assert.equal(c.alternatives[0].goalAssessment.originalTargetTreatmentMm3,10);assert.equal(original.cranialPlanning.selectedBranchId,'actual-branch');
});
