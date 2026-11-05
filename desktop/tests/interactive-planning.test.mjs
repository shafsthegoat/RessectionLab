import test from 'node:test';
import assert from 'node:assert/strict';
import {planningBinding,currentPlanningReport,currentSourcePreflight,selectedPlanningRoute,planningDraftIssue,planningConstraintKey,planningOverlayForView,planningSliceSegments,planningSphereSliceRadius} from '../src/interactive-planning-data.ts';
const constraints=()=>({target:{centerMm:[10,10,10],radiusMm:2},entry:{centerMm:[10,0,10],radiusMm:4},tool:{instrument:'cusa',radiusMm:1.5},requiredEvidence:[],candidateCount:3,nonTargetClearanceMm:0});
const pose=position=>({position,quaternion:[0,0,0,1]});
function fixture(){
  const draft=constraints(),route={id:'route-1',label:'Route 1',outsidePose:pose([0,0,-10]),entryPose:pose([0,0,0]),targetPose:pose([0,0,10]),metrics:{pathLengthMm:20},actionProgram:[],scope:'Inactive checked route'};
  const report={schema:'resectionlab.interactive-planning.v1',sourceSceneId:'scene',startStateId:'state',prefixActions:7,constraints:draft,routes:[route],evidence:{},rejections:[],reportSeal:'seal'};
  const response={caseHash:'patient',branchId:'source',position:7,view:{stateId:'state',scene:{sceneId:'scene'}},interactivePlanning:report};
  const submitted={binding:planningBinding(response),revision:4};return {draft,response,report,route,submitted};
}
test('source readiness is shown only for its verified native response and grants no route authority',()=>{
  const {response}=fixture();
  const report={schema:'resectionlab.surgical-source-preflight.v1',sourceSceneId:'scene',startStateId:'state',prefixActions:7,planningAdmitted:false,checks:[],sourceEvidence:null,scope:'Read-only'};
  const bound={...response,interactivePlanning:null,interactivePlanningAvailability:{available:false,defaults:null,sourcePreflight:report}};
  assert.equal(currentSourcePreflight(bound),report);
  assert.equal(bound.interactivePlanningAvailability.available,false);
  assert.equal(currentPlanningReport(bound,constraints(),null,0),null);
  for(const [key,value] of [['schema','unknown'],['sourceSceneId','other'],['startStateId','other'],['prefixActions',8]]){
    assert.equal(currentSourcePreflight({...bound,interactivePlanningAvailability:{...bound.interactivePlanningAvailability,sourcePreflight:{...report,[key]:value}}}),null);
  }
  for(const flags of [{readOnly:true},{playback:{readOnly:true}},{verification:{readOnly:true}}])assert.equal(currentSourcePreflight({...bound,...flags}),null);
  assert.equal(currentSourcePreflight(null),null);
});
test('current report requires the exact source/checkpoint and submitted draft revision',()=>{
  const f=fixture();assert.equal(currentPlanningReport(f.response,f.draft,f.submitted,4),f.report);
  for(const [key,value] of [['caseHash','another-patient'],['branchId','another-branch'],['position',8]]){
    assert.equal(currentPlanningReport({...f.response,[key]:value},f.draft,f.submitted,4),null,key);
  }
  for(const key of ['sourceSceneId','startStateId','prefixActions']){
    const report={...f.report,[key]:key==='prefixActions'?8:'stale'};
    assert.equal(currentPlanningReport({...f.response,interactivePlanning:report},f.draft,f.submitted,4),null,key);
  }
  assert.equal(currentPlanningReport(f.response,f.draft,null,4),null);
  assert.equal(currentPlanningReport({...f.response,interactivePlanning:{...f.report,current:false}},f.draft,f.submitted,4),null,'historical saved reports cannot authorize a route');
  assert.equal(currentPlanningReport(f.response,f.draft,f.submitted,5),null,'editing then reverting still invalidates the old generation');
  for(const flags of [{readOnly:true},{playback:{readOnly:true}},{verification:{readOnly:true}}])assert.equal(currentPlanningReport({...f.response,...flags},f.draft,f.submitted,4),null);
});
test('target, entry, geometry and evidence edits invalidate selection authority',()=>{
  const f=fixture();
  for(const change of [d=>d.target.centerMm[1]++,d=>d.target.radiusMm++,d=>d.entry.centerMm[2]++,d=>d.entry.radiusMm++,d=>d.tool.radiusMm++,d=>d.requiredEvidence.push('vessels'),d=>d.candidateCount++,d=>d.nonTargetClearanceMm=.5]){
    const edited=structuredClone(f.draft);change(edited);assert.equal(currentPlanningReport(f.response,edited,f.submitted,4),null);
  }
  assert.equal(selectedPlanningRoute(null,'route-1'),null);
  assert.equal(selectedPlanningRoute(f.report,'old-route'),null);
  assert.equal(selectedPlanningRoute(f.report,'route-1'),f.route);
  assert.equal(planningConstraintKey(f.draft),planningConstraintKey(Object.fromEntries(Object.entries(f.draft).reverse())),'canonical backend fields can have different insertion order');
  assert.equal(planningConstraintKey({...f.draft,requiredEvidence:['vessels','shift']}),planningConstraintKey({...f.draft,requiredEvidence:['shift','vessels']}));
});
test('invalid numeric drafts refuse generation and overlays cannot cross source or display admission',()=>{
  const f=fixture();assert.equal(planningDraftIssue(f.draft),null);assert.equal(planningDraftIssue({...f.draft,nonTargetClearanceMm:2}),null);
  assert.equal(planningDraftIssue({...f.draft,target:{...f.draft.target,radiusMm:.5},entry:{...f.draft.entry,radiusMm:30},tool:{instrument:'cusa',radiusMm:6}}),null);
  for(const change of [d=>d.target.centerMm[0]=NaN,d=>d.entry.radiusMm=0,d=>d.target.radiusMm=.4,d=>d.entry.radiusMm=30.1,d=>d.tool.radiusMm=.09,d=>d.tool.radiusMm=6.1,d=>d.tool.radiusMm=Infinity,d=>d.candidateCount=1,d=>d.nonTargetClearanceMm=-.1,d=>d.nonTargetClearanceMm=2.1,d=>d.nonTargetClearanceMm=NaN]){const d=structuredClone(f.draft);change(d);assert(planningDraftIssue(d));}
  const overlay={binding:planningBinding(f.response),constraints:f.draft,routes:[f.route],selectedId:f.route.id,limitingStructures:[]};
  assert.equal(planningOverlayForView(overlay,f.response,f.response.view),overlay);
  assert.equal(planningOverlayForView(overlay,{...f.response,caseHash:'other'},f.response.view),undefined);
  assert.equal(planningOverlayForView(overlay,f.response,{...f.response.view}),undefined,'withhold overlay until exact anatomy object is admitted');
});
test('MRI trajectories clip at slice slab intersections and sphere circles respect RAS depth',()=>{
  const f=fixture(),overlay={binding:'',constraints:f.draft,routes:[f.route],selectedId:f.route.id,limitingStructures:[]};
  const segments=planningSliceSegments(overlay,2,5,1);assert.equal(segments.length,1);
  assert.deepEqual(segments[0].points,[[0,0,5],[0,0,5]]);assert.equal(segments[0].selected,true);
  assert.equal(planningSliceSegments(overlay,2,30,1).length,0);
  assert.equal(planningSliceSegments(overlay,2,5,-1).length,0);
  assert.equal(planningSphereSliceRadius([0,0,5],5,2,5),5);
  assert.equal(planningSphereSliceRadius([0,0,5],5,2,8),4);
  assert.equal(planningSphereSliceRadius([0,0,5],5,2,11),null);
});
