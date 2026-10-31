import test from 'node:test';
import assert from 'node:assert/strict';
import {consequenceComparisonData} from '../src/consequence-comparison-data.ts';

const hash=n=>'sha256:'+String(n).repeat(64);
function fixture(){
  const scenarios=[{id:'nominal',sourceSceneId:hash(1),relaxationTimeScale:1,interpretation:'Declared scenario'},
    {id:'slow',sourceSceneId:hash(2),relaxationTimeScale:2,interpretation:'Declared scenario'}];
  const samples=scenarios.flatMap(s=>['continue','STOP'].map((branchId,i)=>({branchId,scenarioId:s.id,
    vector:{sourceSceneId:s.sourceSceneId,stateId:hash(3+i),nativeActions:1,cleanupComplete:true,clinicalValidation:false,
      residualTargetMm3:20+i,dimensions:{targetRemovedMm3:{unit:'mm3',direction:'max',status:'source_count',cumulative:8-i,incremental:8-i,reason:null},
        motorEncounterMm3:{unit:'mm3',direction:'min',status:'unavailable',cumulative:null,incremental:null,reason:'Missing source evidence'}}}})));
  return {view:{scene:{sceneId:hash(1)}},branches:[{id:'continue',steps:1},{id:'STOP',steps:1}],
    consequenceComparison:{schema:'resectionlab.surgical-consequence-comparison.v1',sourceSceneId:hash(1),comparisonSeal:hash(5),
      sourceRole:'DEVELOPMENT',baselineBranchId:'prefix',stopBranchId:'STOP',optimizerUpdates:0,physicalValidationPassed:false,
      physiologicalRL:'hold',scenarios,samples,limitations:['No clinical prediction'],decisions:[{
        policy:{id:'source-model',basis:'declared_research_task_not_clinical_threshold',objectives:['targetRemovedMm3'],requiredMetrics:[]},
        selectedBranchId:'continue',stopFallback:false,stopExecutable:true,taskSatisfied:true,pairedScenarioParetoFront:['continue'],violations:{continue:[],STOP:[]},reason:'unique_paired_scenario_frontier'}]}};
}

test('typed bridge-bound comparison retains paired worlds and unknown evidence',()=>{
  const value=consequenceComparisonData(fixture());assert(value);assert.equal(value.scenarios.length,2);
  assert.equal(value.samples[0].vector.dimensions.motorEncounterMm3.cumulative,null);
});

for(const [name,mutate] of [
  ['foreign source',r=>r.consequenceComparison.sourceSceneId=hash(9)],
  ['nominal native branch mismatch',r=>r.branches[0].steps=2],
  ['missing paired sample',r=>r.consequenceComparison.samples.pop()],
  ['duplicate paired sample',r=>r.consequenceComparison.samples.push(structuredClone(r.consequenceComparison.samples[0]))],
  ['missing whole world',r=>r.consequenceComparison.samples=r.consequenceComparison.samples.filter(s=>s.scenarioId==='nominal')],
  ['foreign scenario scene',r=>r.consequenceComparison.samples[2].vector.sourceSceneId=hash(9)],
  ['null coerced to zero',r=>r.consequenceComparison.samples[0].vector.dimensions.motorEncounterMm3.cumulative=0],
  ['negative cost',r=>r.consequenceComparison.samples[0].vector.dimensions.targetRemovedMm3.cumulative=-1],
  ['string cleanup',r=>r.consequenceComparison.samples[0].vector.cleanupComplete='false'],
  ['clinical promotion',r=>r.consequenceComparison.physicalValidationPassed=true],
  ['foreign selected branch',r=>r.consequenceComparison.decisions[0].selectedBranchId='invented'],
  ['foreign Pareto branch',r=>r.consequenceComparison.decisions[0].pairedScenarioParetoFront=['invented']],
  ['non-array violations',r=>r.consequenceComparison.decisions[0].violations.continue={}],
  ['untyped violation',r=>r.consequenceComparison.decisions[0].violations.continue=[{scenario:'nominal',reason:3}]],
  ['foreign violation scenario',r=>r.consequenceComparison.decisions[0].violations.continue=[{scenario:'invented',reason:'unknown_required_consequence'}]],
  ['missing branch violations',r=>delete r.consequenceComparison.decisions[0].violations.STOP],
  ['role promotion',r=>r.consequenceComparison.sourceRole='MEASUREMENT_EVAL'],
])test(`rejects ${name}`,()=>{const r=fixture();mutate(r);assert.equal(consequenceComparisonData(r),null);});
