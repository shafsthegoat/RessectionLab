import {test} from 'node:test';
import assert from 'node:assert/strict';
import {fixture} from './cranial-workflow-fixture.mjs';
import {checkedCranialGeometry,cranialRecordedOverlay} from '../src/cranial-recorded-geometry.ts';
import {planningRoutePoints,planningSliceSegments,planningSliceMemberships} from '../src/interactive-planning-data.ts';
function recorded(){
 const r=fixture(),c=r.cranialPlanning,row=c.alternatives[0];
 const policy={schema:'resectionlab.supratentorial-procedure-policy.v1',policyId:'declared',policyHash:'bound-policy',available:true,startStage:'post-scalp',requiredCoverage:['brainstem','cerebellum','intracranial'],scenarios:['nominal'],limitations:[]};
 Object.assign(c,{schema:'resectionlab.cranial-strategy-comparison.v2',sourceBranchId:'main',procedurePolicy:policy,selectionRule:'max_treatment_within_harm_bounds',stopReason:'STOP_WITH_RESIDUAL'});r.cranialPlanningAvailability.procedurePolicy=structuredClone(policy);
 row.sequenceOutcome.schema='resectionlab.sequence-outcome.v2';row.ledger.neuralRemovedMm3=0;row.sequenceOutcome.dimensions.neuralRemovedMm3={unit:'mm3',status:'source_count',cumulative:0};
 row.sequenceOutcome.dimensions.structureRemovedMm3={motor:{cumulative:0}};row.sequenceOutcome.dimensions.structureCutSourceMm3={motor:{cumulative:0}};
 row.nativeHistory=Array.from({length:12},(_,i)=>({stateBefore:i===0?'root':`state-${i}`,stateAfter:i===11?'final':`state-${i+1}`}));
 row.nativeHistory[0].stateAfter='prefix';row.nativeHistory[1].stateBefore='prefix';
 row.recordedGeometry={schema:'resectionlab.recorded-cranial-geometry.v1',caseHash:r.caseHash,sourceSceneId:c.sourceSceneId,sourceStateId:c.sourceStateId,rootSourceStateId:'root',sceneId:row.sceneId,policyHash:policy.policyHash,branchId:row.branchId,finalStateId:row.finalStateId,nativeActions:12,actionHistoryHash:'history',geometrySeal:'geometry',acceptedStateIds:['root',...row.nativeHistory.map(e=>e.stateAfter)],acceptedActions:row.nativeHistory.map((e,i)=>({actionIndex:i,operation:'apply',slot:'left',instrument:'cusa',status:'accepted',elapsedSeconds:.1,...e})),cleanup:{debrisPresentMm3:0,debrisRemovedMm3:236,fluidPresentMm3:0,activeBleedingSources:0},paths:[{actionIndex:1,operation:'apply',slot:'left',instrument:'cusa',status:'accepted',stateBefore:'prefix',stateAfter:'state-2',pointsRasMm:[[0,0,0],[1,1,1],[2,1,2]]}],omittedPartialActions:[],structures:[{id:'motor',label:'Generated motor',kind:'neural',evidenceKind:'generated-engineering',reviewStatus:'generated-engineering',provenance:'Independent declaration',sourceCells:[1],pointsRasMm:[[1,1,1]],displayTruncated:false,removedMm3:0,cutSourceMm3:0}],targetSourceCells:Array.from({length:100},(_,i)=>i),targetPointsRasMm:Array.from({length:100},(_,i)=>[i,1,2]),residualSourceCells:Array.from({length:20},(_,i)=>i),residualPointsRasMm:Array.from({length:20},(_,i)=>[i,1,2]),targetDisplayTruncated:false,scope:'Recorded accepted paths'};
 return r;
}
test('recorded guides join case, source, policy, branch, endpoint and structure accounting',()=>{
 const r=recorded(),row=r.cranialPlanning.alternatives[0];assert(checkedCranialGeometry(r,row));
 for(const key of ['caseHash','sourceSceneId','sourceStateId','sceneId','policyHash','branchId','finalStateId']){const altered=structuredClone(r);altered.cranialPlanning.alternatives[0].recordedGeometry[key]='other';assert.equal(checkedCranialGeometry(altered,altered.cranialPlanning.alternatives[0]),null,key);}
 const altered=structuredClone(r);altered.cranialPlanning.alternatives[0].recordedGeometry.structures[0].removedMm3=1;assert.equal(checkedCranialGeometry(altered,altered.cranialPlanning.alternatives[0]),null);
});
test('exact polylines and source memberships survive slice clipping without requested-endpoint shortcuts',()=>{
 const r=recorded(),before=structuredClone(r),o=cranialRecordedOverlay(r,r.view);assert(o?.recorded);assert.deepEqual(planningRoutePoints(o.routes[0]),[[0,0,0],[1,1,1],[2,1,2]]);assert.deepEqual(planningSliceSegments(o,2,1,.1).map(s=>s.points),[[[.9,.9,1],[1,1,1]],[[1,1,1],[1.1,1,1]]]);assert.equal(planningSliceMemberships(o,2,1,.1).length,1);assert.equal(planningSliceMemberships(o,2,2,.1).length,0);assert.deepEqual(r,before);
});
test('recorded branch and certified replay prefix retain guides; edits and unaccepted views clear them',()=>{
 const r=recorded(),row=r.cranialPlanning.alternatives[0];r.branchId=row.branchId;r.position=12;r.view={...r.view,stateId:'final',scene:{...r.view.scene,sceneId:'root-scene'}};assert(cranialRecordedOverlay(r,r.view));r.position=1;r.view.stateId='prefix';assert(cranialRecordedOverlay(r,r.view));r.view.stateId='edited';assert.equal(cranialRecordedOverlay(r,r.view),undefined);r.position=13;r.view.stateId='final';assert.equal(cranialRecordedOverlay(r,r.view),undefined);assert.equal(cranialRecordedOverlay(r,structuredClone(r.view)),undefined);
});
