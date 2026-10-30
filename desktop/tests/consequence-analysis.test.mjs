import test from 'node:test';
import assert from 'node:assert/strict';
import {consequenceAnalysisData,defaultConsequencePolicyInput,consequenceComparisonData,validConsequencePolicyInput} from '../src/consequence-comparison-data.ts';
import {fixture,hash} from './consequence-editor-fixture.mjs';
test('analysis binds exact accepted checkpoint, paired programs and nonclinical policy',()=>{
  const r=fixture(),data=consequenceAnalysisData(r);assert(data);assert.equal(data.latest.decision.feasibleContinuations.length,2);
  assert.deepEqual(defaultConsequencePolicyInput(consequenceComparisonData(r)),data.latest.policyInput);
});
for(const [name,change] of [
  ['case',r=>r.consequenceAnalysis.caseHash='old'],['scene',r=>r.consequenceAnalysis.sourceSceneId=hash(3)],['seal',r=>r.consequenceAnalysis.comparisonSeal=hash(3)],
  ['context branch',r=>r.consequenceAnalysis.context.branchId='STOP'],['context position',r=>r.consequenceAnalysis.context.position=0],['context state',r=>r.consequenceAnalysis.context.stateId=hash(3)],
  ['latest seal',r=>r.consequenceAnalysis.latest.comparisonSeal=hash(3)],['binding baseline',r=>r.consequenceAnalysis.latest.binding.baseline.stateId=hash(3)],
  ['duplicate program',r=>r.consequenceAnalysis.latest.binding.programs[1]=r.consequenceAnalysis.latest.binding.programs[0]],['missing program',r=>r.consequenceAnalysis.latest.binding.programs.pop()],
  ['untyped binding hash',r=>r.consequenceAnalysis.latest.binding.samplesHash='bad'],['wrong archive encoding',r=>r.consequenceAnalysis.latest.binding.sourceArchiveSha256=hash(3)],
  ['new actions',r=>r.consequenceAnalysis.latest.newPhysicalActions=1],['optimization',r=>r.consequenceAnalysis.latest.optimizerUpdates=1],['clinical promotion',r=>r.consequenceAnalysis.latest.clinicalValidation=true],
  ['negative task',r=>r.consequenceAnalysis.latest.policyInput.minimumAdditionalTargetMm3=-1],['unsupported evidence',r=>r.consequenceAnalysis.latest.policyInput.requiredMetrics=['invented']],
  ['extra policy input',r=>r.consequenceAnalysis.latest.policyInput.objectives=[]],['policy/input drift',r=>r.consequenceAnalysis.latest.policyInput.maximumCumulativeNonTargetMm3=1],
  ['objective substitution',r=>r.consequenceAnalysis.latest.decision.policy.objectives=['brainStrainSquaredMm3S']],['foreign frontier',r=>r.consequenceAnalysis.latest.decision.pairedScenarioParetoFront=['invented']],
  ['fabricated unique winner',r=>r.consequenceAnalysis.latest.decision.pairedScenarioParetoFront=['continue-a','continue-b']],
  ['unjustified STOP fallback',r=>Object.assign(r.consequenceAnalysis.latest.decision,{selectedBranchId:'STOP',stopFallback:true,taskSatisfied:false})],
  ['missing numeric constraint actual',r=>delete r.consequenceAnalysis.latest.decision.violations.STOP[0].actual],
  ['stale revision',r=>r.consequenceAnalysis.revisionCount=2],
])test(`rejects analysis ${name}`,()=>{const r=fixture();change(r);assert.equal(consequenceAnalysisData(r),null);});
test('branch changes retain a historical record but cannot promote it as current',()=>{
  const r=fixture();r.branchId='STOP';r.position=2;r.view.stateId=hash(5);Object.assign(r.consequenceAnalysis,{context:{branchId:r.branchId,position:2,stateId:r.view.stateId},available:false,current:false,reason:'common_checkpoint_required'});
  assert(consequenceAnalysisData(r));r.consequenceAnalysis.current=true;assert.equal(consequenceAnalysisData(r),null);
});
test('read-only playback cannot be a reassessable accepted common checkpoint',()=>{
  const r=fixture();r.playback={readOnly:true};assert.equal(consequenceAnalysisData(r),null);
});
test('strict numeric inputs reject null coercion, infinity and duplicate evidence',()=>{
  const input=fixture().consequenceAnalysis.latest.policyInput;
  for(const patch of [{minimumAdditionalTargetMm3:null},{maximumCumulativeNonTargetMm3:Infinity},{requiredMetrics:['motorEncounterMm3','motorEncounterMm3']},{maxBrainStrainSquaredMm3S:-1}])assert.equal(validConsequencePolicyInput({...input,...patch}),false);
  assert(validConsequencePolicyInput({...input,minimumAdditionalTargetMm3:0,maxBrainStressSquaredPa2Mm3S:0,requiredMetrics:['motorEncounterMm3']}));
});
