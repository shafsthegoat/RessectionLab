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
