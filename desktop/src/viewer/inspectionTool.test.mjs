import assert from "node:assert/strict";
import test from "node:test";
import * as THREE from "three";
import { InspectionSelectionGate, InstrumentDisplayState, validateInspectionTool } from "./inspectionTool.ts";
import { inspectionToolMeshes } from "./inspectionToolGeometry.ts";

export function inspectionFixture() {
  const hash = (digit) => `sha256:${digit.repeat(64)}`;
  const volume = {caseId:"test",caseHash:hash("a"),planningHash:hash("b"),frame:"RAS+",
    affine:[[1,0,0,10],[0,1,0,20],[0,0,1,30],[0,0,0,1]],shape:[3,3,3],mri:new Float32Array(27),compartments:[]};
  const tool = {tool_id:"explicit-test-tool",tip_radius_mm:1.25,shaft_radius_mm:.45,working_length_mm:20,
    tip_length_mm:2,max_access_angle_deg:30,parameter_source:"analytic fixture"};
  const entry=[11,21,29.5],tip=[11,21,32],proposalId="proposal-1",actionId="AXISv1:example:primary";
  const binding = {case_hash:volume.caseHash,planning_hash:volume.planningHash,binding_hash:hash("c"),
    mode:"experimental_axis_columns",geometry_frame:"RAS+",source_shape:volume.shape.slice(),native_affine_ras_mm:structuredClone(volume.affine),
    native_config_hash:hash("d"),decision_model_hash:hash("e"),proposal_model_hash:hash("f"),proposal_rule_hash:hash("1"),tools:[tool]};
  const proposal={proposal_id:proposalId,tool_id:tool.tool_id,entry_mm:entry.slice(),primary_target_mm:tip.slice(),fallback_target_mm:[11,21,31],fallback_condition:"primary_preview_rejected"};
  const attempt={status:"complete",feasible:true,proposal_id:proposalId,tool_id:tool.tool_id,entry_mm:entry.slice(),tip_mm:tip.slice(),phase:"primary",reason:"NATIVE_CONNECTED_STROKE"};
  const action={action_id:actionId,kind:"native_stroke",geometry:{frame:"RAS+",tool_id:tool.tool_id,entry_mm:entry.slice(),tip_mm:tip.slice(),axis_unit:[0,0,1],insertion_distance_mm:2.5},
    native_preview:{scope:"native_engine_preview_only",independent_history_checked:false,proposal_id:proposalId,phase:"primary",reason:attempt.reason,
      source_hash:volume.caseHash,source_state_hash:hash("2"),native_config_hash:binding.native_config_hash,
      geometry_unknowns:["tool_geometry_outside_image_unassessed"],native_footprint:"fully-contained-native-cells",unexecuted_contained_cell_count:2,unexecuted_contact_cell_count:4}};
  const report={version:"native-axis-inspection-v1",role:"inspection",status:"ready",inventory_complete:true,binding,
    candidate_eligible:false,removal_authorized:false,clinical_deficit_probability:null,initial_cavity_state_hash:hash("2"),inspection_hash:hash("3"),legal_non_stop_actions:1,
    accounting:{gradient_steps:0,executed_transitions:0,native_commits:0,simulated_removed_volume_mm3:0},
    inventory:{status:"complete",terminated:false,certified_action_ids:[actionId],attempts:[attempt],batch:{unsupported_reason:null,geometry_certified:false,removal_authorized:false,
      source_hash:volume.caseHash,engine_model_hash:binding.native_config_hash,proposal_model_hash:binding.proposal_model_hash,rule_hash:binding.proposal_rule_hash,cavity_state_hash:hash("2"),slot_count:1,
      proposals:[proposal],ledger:[{proposal_id:proposalId,tool_id:tool.tool_id,reason:"PROPOSED_UNCERTIFIED"}]}},
    actions:[{action_id:"STOP",kind:"stop",geometry:null,native_preview:null},action],unknowns:["tool_geometry_outside_image_unassessed"]};
  const input={scope:"unexecuted-native-axis-inspection",caseHash:volume.caseHash,planningHash:volume.planningHash,bindingHash:binding.binding_hash,inspectionHash:report.inspection_hash,actionId,report};
  return {volume,input,report,action,tool,attempt,proposal};
}

test("full shaft and active tip use exact canonical RAS endpoints in 3D and MRI display state",()=>{
  const {volume,input}=inspectionFixture(),display=validateInspectionTool(volume,input);
  assert.deepEqual(display.shaftStart,[11,21,12]);assert.deepEqual(display.shaftEnd,[11,21,30]);assert.deepEqual(display.tip,[11,21,32]);
  const meshes=inspectionToolMeshes(display);assert.equal(meshes.children.length,2);
  assert.deepEqual(meshes.children.map(m=>m.userData.part),["shaft","active-tip"]);
  const bounds=new THREE.Box3().setFromObject(meshes);assert.ok(bounds.min.z<12);assert.ok(bounds.max.z>32);
  for(const mesh of meshes.children){assert.equal(mesh.userData.routeId,undefined);assert.equal(mesh.userData.comparisonSlot,undefined);mesh.geometry.dispose();mesh.material.dispose();}
  const state=new InstrumentDisplayState();state.setInspection(volume,input);assert.deepEqual(state.displayed,[display]);
});
test("LPS source grid converts once while already-RAS tool endpoints stay identical",()=>{
  const {volume,input}=inspectionFixture(),ras=validateInspectionTool(volume,input);
  const lps={...volume,frame:"LPS+",affine:volume.affine.map((row,i)=>row.map(value=>i<2?-value:value))};
  assert.deepEqual(validateInspectionTool(lps,input),ras);
});
test("oblique mirrored source grid accepts exact RAS geometry without resampling",()=>{
  const {volume,input}=inspectionFixture();volume.affine=[[-.8,-.6,0,10],[-.6,.8,0,20],[0,0,2,30],[0,0,0,1]];
  input.report.binding.native_affine_ras_mm=structuredClone(volume.affine);
  assert.deepEqual(validateInspectionTool(volume,input).tip,[11,21,32]);
});
test("display snapshot is detached and deeply immutable",()=>{
  const {volume,input,action,tool}=inspectionFixture(),display=validateInspectionTool(volume,input);
  action.geometry.tip_mm[0]=900;tool.working_length_mm=1000;input.report.unknowns.push("new");
  assert.deepEqual(display.tip,[11,21,32]);assert.deepEqual(display.shaftStart,[11,21,12]);assert.equal(display.unknowns.length,1);
  assert.throws(()=>{display.tip[0]=7;},TypeError);assert.throws(()=>{display.unknowns.push("x");},TypeError);
});
test("A/B state updates while inspection is shown, then restores without route mutation",()=>{
  const {volume,input}=inspectionFixture(),state=new InstrumentDisplayState();
  const a={shaftStart:[0,0,0],shaftEnd:[0,0,8],tip:[0,0,10],shaftRadius:.5,tipRadius:1,color:"#a3e5d3"};
  const b={...a,tip:[1,0,10],color:"#e5c598"};state.setRoutes([a,b]);
  state.setInspection(volume,input);assert.equal(state.displayed.length,1);assert.equal(state.routeVisible,false);
  state.setRoutes([b]);assert.equal(state.displayed[0].tip[0],11);state.setInspection(volume,null);
  assert.deepEqual(state.displayed,[b]);assert.equal(state.routeVisible,true);
});
test("invalid replacement clears old tool; replay suppresses it without resurrection",()=>{
  const {volume,input}=inspectionFixture(),state=new InstrumentDisplayState();state.setInspection(volume,input);
  assert.throws(()=>state.setInspection({...volume,caseHash:"stale"},input),/stale/);assert.equal(state.inspected,null);
  state.setInspection(volume,input);state.setReplay(true);assert.equal(state.inspected,null);assert.deepEqual(state.displayed,[]);
  assert.equal(state.setInspection(volume,input),null);state.setReplay(false);assert.equal(state.inspected,null);
});
test("selection suppression survives cloned wrappers until explicit clear/reselect",()=>{
  const {input}=inspectionFixture(),gate=new InspectionSelectionGate();
  assert.equal(gate.select(input,false),input);gate.clear(input);
  assert.equal(gate.select(structuredClone(input),false),null);
  gate.select(null,false);assert.equal(gate.select(input,false),input);
  assert.equal(gate.select(input,true),null);assert.equal(gate.select(structuredClone(input),false),null);
  const different={...input,actionId:"another"};assert.equal(gate.select(different,false),different);
});
for(const [name,mutate] of [
  ["wrong planning source", f=>f.input.planningHash=`sha256:${"0".repeat(64)}`],
  ["unavailable planning source", f=>delete f.volume.planningHash],
  ["wrong binding", f=>f.input.bindingHash=`sha256:${"0".repeat(64)}`],
  ["changed affine", f=>f.volume.affine[0][3]+=.001],
  ["wrong source shape", f=>f.report.binding.source_shape[0]=4],
  ["partial inventory", f=>f.report.inventory_complete=false],
  ["interrupted preview", f=>f.attempt.status="cancelled"],
  ["STOP geometry", f=>f.input.actionId="STOP"],
  ["rejected attempt", f=>f.attempt.feasible=false],
  ["moved accepted endpoint", f=>f.action.geometry.tip_mm[0]+=1],
  ["different proposal endpoint", f=>f.proposal.primary_target_mm[2]+=1],
  ["different instrument", f=>f.action.geometry.tool_id="other"],
  ["duplicate catalog", f=>f.report.binding.tools.push({...f.tool})],
  ["invalid shaft size", f=>f.tool.shaft_radius_mm=NaN],
  ["zero direction", f=>f.action.geometry.axis_unit=[0,0,0]],
  ["mismatched direction", f=>f.action.geometry.axis_unit=[1,0,0]],
  ["stale preview cavity", f=>f.action.native_preview.source_state_hash=`sha256:${"0".repeat(64)}`],
  ["clinical authority", f=>f.report.clinical_deficit_probability=.1],
  ["removal authority", f=>f.report.removal_authorized=true],
  ["executed cells", f=>f.report.accounting.simulated_removed_volume_mm3=1],
  ["independent certificate claim", f=>f.action.native_preview.independent_history_checked=true],
  ["lost uncertainty", f=>f.report.unknowns=[]],
])test(`withholds ${name}`,()=>{const f=inspectionFixture();mutate(f);assert.throws(()=>validateInspectionTool(f.volume,f.input),/withheld/);});

test("fallback geometry must follow its rejected primary and exact declared endpoint",()=>{
  const f=inspectionFixture();f.attempt.feasible=false;
  const accepted={...f.attempt,phase:"fallback",feasible:true,tip_mm:f.proposal.fallback_target_mm.slice()};
  f.report.inventory.attempts.push(accepted);f.action.native_preview.phase="fallback";f.action.geometry.tip_mm=accepted.tip_mm.slice();f.action.geometry.insertion_distance_mm=1.5;
  assert.deepEqual(validateInspectionTool(f.volume,f.input).tip,[11,21,31]);
  f.attempt.feasible=true;assert.throws(()=>validateInspectionTool(f.volume,f.input),/fallback/);
});
