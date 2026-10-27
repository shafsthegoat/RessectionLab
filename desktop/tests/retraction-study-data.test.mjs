import test from 'node:test';
import assert from 'node:assert/strict';
import {retractionStudyDisplay} from '../src/retraction-study-data.ts';
const seal=n=>'sha256:'+n.repeat(64);
const fixture=()=>({caseHash:seal('0'),view:{scene:{sceneId:seal('1')}},
  branches:[{id:'paired',steps:8}],retractionStudy:{
    schema:'resectionlab.retractor-native-study.v1',sourceSceneId:seal('1'),sourceStateId:seal('2'),
    comparisonSeal:seal('3'),sourceRole:'DEVELOPMENT',optimizerUpdates:0,clinicalOutcomes:'unavailable',
    physicalCalls:8,wallSeconds:3,variants:[{branchId:'paired',assisted:true,status:'completed_native_program',failureReason:null,
      finalStateId:seal('4'),nativeActions:8,exactNativeReplay:true,
      outcome:{wholeTargetRemovedMm3:10,wholeNonTargetRemovedMm3:100,residualTargetMm3:1000,
        cleanupComplete:true,taskGateMet:false,failedGates:['ratio','wholeTarget']},
      phases:[{label:'Native coupled hold acquired',position:4}],
      finalMetrics:{brainMaxDisplacementMm:.005,contactCellSeconds:30,elapsedSeconds:10},
      sampledMechanicalPeaks:{brainMaxDisplacementMm:.9},outsideRecoveryOffsetMm:4,
      publicCalls:8,costMode:'recorded_program_replay',originalPlanningAndReplayCalls:90}]}});
test('paired study keeps failed task, replay cost and residual displacement visible',()=>{
  const data=retractionStudyDisplay(fixture());
  assert.equal(data.variants[0].outcome.taskGateMet,false);
  assert.equal(data.variants[0].costMode,'recorded_program_replay');
  assert.equal(data.variants[0].originalPlanningAndReplayCalls,90);
  assert.equal(data.variants[0].finalMetrics.brainMaxDisplacementMm,.005);
});
test('paired study cannot cross scenes, branches, positions, costs or task result bindings',()=>{
  for(const mutate of [
    x=>x.view.scene.sceneId=seal('9'),x=>x.branches[0].steps=7,
    x=>x.retractionStudy.sourceRole='FROZEN_ASSESSMENT',x=>x.retractionStudy.optimizerUpdates=1,
    x=>x.retractionStudy.variants[0].phases[0].position=9,
    x=>x.retractionStudy.physicalCalls=7,
    x=>x.retractionStudy.variants[0].outcome.taskGateMet=true,
    x=>x.retractionStudy.variants[0].originalPlanningAndReplayCalls=null]){
    const x=fixture();mutate(x);assert.equal(retractionStudyDisplay(x),null);
  }
});
