import assert from 'node:assert/strict';
import {test} from 'node:test';
import {caseImageEvidence,caseLayerEvidence} from '../src/data-case-evidence.ts';

test('acquired CT and both healthy source contracts retain acquired image identity with synthetic geometry',()=>{
 for(const [classification,observations] of [
  ['synthetic_scenario_on_acquired_host',{primary_image:'baseline_acquired_background_geometry_only'}],
  ['synthetic_scenario_on_acquired_host',{primary_T1:'baseline_acquired_background_geometry_only'}],
  ['synthetic_scenario_on_acquired_healthy_host',{T1:'acquired_unmodified_baseline',tumor:'synthetic_geometry_only'}]]){
  assert.deepEqual(caseImageEvidence({is_synthetic:true,data_case:{classification,observations}}),{synthetic:true,onAcquiredHost:true,geometryOnly:true,generatedImage:false});
 }
});
test('supplied vascular unions retain component evidence inside synthetic cases without inventing unspecified provenance',()=>{
 const metadata={is_synthetic:true,data_case:{components:{arteries:{provenance:'supplied_annotation',coverage:'landmark arteries only'},veins_sinuses:{provenance:'supplied_annotation'}}},anatomy_layers:{supplied_arterial_annotation_union:{kind:'vessels'},brain:{provenance:'estimated'}}};
 for(const name of ['supplied_arterial_annotation_union','supplied_venous_sinus_annotation_union']){
  assert.equal(caseLayerEvidence(metadata,name).provenance,'source-provided');
  assert.match(caseLayerEvidence(metadata,name).coverage,/Derived union.*incomplete/);
 }
 assert.equal(caseLayerEvidence({is_synthetic:true},'supplied_arterial_annotation_union'),undefined);
 assert.equal(caseLayerEvidence(metadata,'brain').provenance,'estimated');
 assert.equal(caseLayerEvidence({...metadata,anatomy_layers:{supplied_arterial_annotation_union:{provenance:'estimated'}}},'supplied_arterial_annotation_union').provenance,'estimated');
});
test('generated image variants and analytic fixtures remain visibly generated',()=>{
 assert.equal(caseImageEvidence({is_synthetic:true,data_case:{classification:'synthetic_scenario_on_acquired_host',observations:{primary_image:'synthetic_appearance'}}}).generatedImage,true);
 assert.deepEqual(caseImageEvidence({is_synthetic:true}),{synthetic:true,onAcquiredHost:false,geometryOnly:false,generatedImage:true});
 assert.equal(caseImageEvidence(undefined).generatedImage,false);
});
