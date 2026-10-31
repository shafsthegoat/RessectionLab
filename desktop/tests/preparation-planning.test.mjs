import test from 'node:test';
import assert from 'node:assert/strict';
import {planningBinding} from '../src/interactive-planning-data.ts';
import {PREPARATION_SCHEMA,currentPreparationReport,currentPreparationSuggestion,preparationActions,
  preparationControlKey,preparationDraftIssue,preparationLocked,preparationPhaseAvailable,
  preparationPreviewMatches,preparationBaselineText,preparationCost,preparationIncomplete,snapPreparationPoint,
  preparationPointIsCell} from '../src/preparation-planning-data.ts';

function fixture(){
  const controls={sourceRay:{surfaceMm:[3,3,3],sourceTargetMm:[3,3,10]},target:{centerMm:[3,3,10],radiusMm:6},
    widthsMm:[0,.3,1],maxNonTargetMm3:200,maxActions:128,maxSeconds:180,entryRadiusMm:.5,
    shortSeconds:.25,longSeconds:1,minimumTargetMm3:1,requiredEvidence:['brain','target'],encounter:null};
  const attempt={complete:true,failure:null,summary:{additionalTargetMm3:0,additionalNonTargetMm3:43,elapsedSeconds:23.8,withdrawn:true},actionProgram:[{operation:'apply'}],physicalStateHashes:['physical']};
  const row={id:'width-0',widthMm:0,preparation:{schema:'resectionlab.source-exposure-preparation.v2',preparationSeal:'preparation',publicPlan:{attempts:[attempt],decision:{operation:'PREPARE',selectedIndex:0},scope:'Research'}},preparedAdmission:{admissionSeal:'admitted'},comparison:{publicPlan:{decision:{operation:'COMPARE',selectedId:'short'},alternatives:[{id:'short',feasible:true,taskProgressMet:true,withdrawal:'complete',summary:{additionalTargetMm3:1,additionalNonTargetMm3:0,elapsedSeconds:2}}],rejections:[],approaches:{routes:[{}],rejections:[]}}},reason:null};
  const report={schema:PREPARATION_SCHEMA,reportSeal:'report',sourceSceneId:'scene',startStateId:'state',
    sourceBranchId:'protected-analysis-copy',sourcePosition:7,prefixActions:7,current:true,revision:2,
    controls:structuredClone(controls),rows:[row],policies:{corridorFirst:{selectedPreparationId:'width-0',suffixId:'short'},completePlanAware:{selectedPreparationId:'width-0',suffixId:'short'}},scope:'Finite offline comparison'};
  const response={caseHash:'case',branchId:'main',position:7,view:{stateId:'state',scene:{sceneId:'scene',shape:[24,24,24],affine:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}},preparationPlanning:report};
  const submitted={binding:planningBinding(response),revision:4};
  const displayed={stateId:'preview'};const preview={kind:'preparation',phaseId:'prepared',candidateId:row.id,reportSeal:report.reportSeal,view:displayed};
  return {controls,attempt,row,report,response,submitted,displayed,preview};
}

test('current preparation report uses accepted source cursor and explicit submitted draft revision',()=>{
  const f=fixture();assert.equal(currentPreparationReport(f.response,f.controls,f.submitted,4),f.report);
  assert.notEqual(f.report.sourceBranchId,f.response.branchId,'protected analysis copy is not the editable cursor branch');
  for(const [key,value] of [['caseHash','other'],['branchId','equivalent-copy'],['position',8]]){
    assert.equal(currentPreparationReport({...f.response,[key]:value},f.controls,f.submitted,4),null,key);
  }
  for(const [key,value] of [['schema','unknown'],['current',false],['sourceSceneId','other'],['startStateId','other'],['sourcePosition',8],['prefixActions',8]]){
    assert.equal(currentPreparationReport({...f.response,preparationPlanning:{...f.report,[key]:value}},f.controls,f.submitted,4),null,key);
  }
  assert.equal(currentPreparationReport(f.response,f.controls,null,4),null);
  assert.equal(currentPreparationReport(f.response,f.controls,f.submitted,5),null,'edit and revert cannot revive a prior generation');
});

test('every meaningful draft control invalidates a saved comparison while evidence order is canonical',()=>{
  const f=fixture(),key=preparationControlKey(f.controls);
  const edits=[c=>c.sourceRay.surfaceMm[0]++,c=>c.sourceRay.sourceTargetMm[1]++,c=>c.target.centerMm[2]++,
    c=>c.target.radiusMm++,c=>c.widthsMm.reverse(),c=>c.maxNonTargetMm3--,c=>c.maxActions--,c=>c.maxSeconds--,
    c=>c.entryRadiusMm++,c=>c.shortSeconds=.5,c=>c.longSeconds=1.5,c=>c.minimumTargetMm3=2,
    c=>c.requiredEvidence.push('functional'),c=>c.encounter={layerId:'context',marginMm:0,maxNearMm3:1000}];
  for(const edit of edits){const changed=structuredClone(f.controls);edit(changed);assert.notEqual(preparationControlKey(changed),key);assert.equal(currentPreparationReport(f.response,changed,f.submitted,4),null);}
  const reordered=Object.fromEntries(Object.entries(structuredClone(f.controls)).reverse());reordered.requiredEvidence.reverse();
  assert.equal(preparationControlKey(reordered),key);
  const withContext={...f.controls,encounter:{layerId:'source-a',marginMm:0,maxNearMm3:1000}};
  for(const edit of [{layerId:'source-b'},{marginMm:3},{maxNearMm3:0}])assert.notEqual(preparationControlKey({...withContext,encounter:{...withContext.encounter,...edit}}),preparationControlKey(withContext));
});

test('all read-only modes and pending work lock generation/preview/installation authority',()=>{
  const f=fixture();assert.equal(preparationLocked(f.response),false);
  for(const flags of [{readOnly:true},{playback:{readOnly:true}},{verification:{readOnly:true}}]){
    const response={...f.response,...flags};assert.equal(preparationLocked(response),true);
    assert.equal(currentPreparationReport(response,f.controls,f.submitted,4),null);
    const actions=preparationActions({response,draft:f.controls,submitted:f.submitted,revision:4,candidateId:f.row.id,
      phase:'prepared',preview:f.preview,displayed:f.displayed,pending:false,acknowledged:true});
    assert.equal(actions.previewAllowed,false);assert.equal(actions.installAllowed,false);
  }
  assert.equal(preparationLocked(f.response,true),true);assert.equal(preparationLocked(null),true);
});

test('source-grid snapping handles reflection, obliquity and submillimeter spacing without clamping',()=>{
  const scene={shape:[6,7,8],affine:[[-1,0,0,12],[0,0,-2,20],[0,.5,0,-3],[0,0,0,1]]};
  assert.deepEqual(snapPreparationPoint([9.8,11.2,-1.4],scene),[10,12,-1.5]);
  assert.equal(preparationPointIsCell([10,12,-1.5],scene),true);
  assert.equal(preparationPointIsCell([9.8,11.2,-1.4],scene),false);
  assert.equal(snapPreparationPoint([1000,0,0],scene),null);
  assert.equal(snapPreparationPoint([NaN,0,0],scene),null);
  assert.equal(snapPreparationPoint([0,0,0],{...scene,shape:[0,7,8]}),null);
  assert.equal(snapPreparationPoint([0,0,0],{...scene,affine:[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]}),null);
});

test('suggestions are accepted only from the current editable source and remain independently editable',()=>{
  const f=fixture();const suggestion={sourceSceneId:'scene',startStateId:'state',sourceRay:structuredClone(f.controls.sourceRay),target:structuredClone(f.controls.target),entryRadiusMm:.5};
  const response={...f.response,preparationSuggestion:suggestion};assert.equal(currentPreparationSuggestion(response),suggestion);
  const detached=structuredClone(currentPreparationSuggestion(response));detached.sourceRay.surfaceMm[0]++;
  assert.deepEqual(suggestion.sourceRay.surfaceMm,[3,3,3]);
  for(const change of [{sourceSceneId:'other'},{startStateId:'other'},{entryRadiusMm:NaN},{target:{centerMm:[0,0,0],radiusMm:16}}])assert.equal(currentPreparationSuggestion({...response,preparationSuggestion:{...suggestion,...change}}),null);
  assert.equal(currentPreparationSuggestion({...response,readOnly:true}),null);
});

test('invalid points, widths, budgets, durations and required evidence cannot authorize comparison',()=>{
  const f=fixture();assert.equal(preparationDraftIssue(f.controls),null);
  const edits=[c=>c.sourceRay=null,c=>c.sourceRay.surfaceMm[0]=NaN,c=>c.sourceRay.sourceTargetMm=c.sourceRay.surfaceMm,
    c=>c.target=null,c=>c.target.radiusMm=16,c=>c.widthsMm=[],c=>c.widthsMm=[0,0],c=>c.widthsMm=[.6],
    c=>c.maxNonTargetMm3=0,c=>c.maxNonTargetMm3=201,c=>c.maxActions=5,c=>c.maxActions=128.5,
    c=>c.maxSeconds=181,c=>c.maxSeconds=Infinity,c=>c.entryRadiusMm=.4,c=>c.shortSeconds=2,
    c=>c.longSeconds=.1,c=>c.minimumTargetMm3=9,c=>c.requiredEvidence=['target'],c=>c.requiredEvidence=['brain','target','brain']];
  for(const edit of edits){const changed=structuredClone(f.controls);edit(changed);assert(preparationDraftIssue(changed));assert.equal(currentPreparationReport(f.response,changed,f.submitted,4),null);}
  const context={layerId:'context',marginMm:0,maxNearMm3:1000};const c={...f.controls,encounter:context};
  assert.equal(preparationDraftIssue(c,[{layerId:'context'}]),null);
  assert(preparationDraftIssue(c,[]));assert(preparationDraftIssue({...c,encounter:{...context,marginMm:6}}));
  assert(preparationDraftIssue({...c,encounter:{...context,maxNearMm3:-1}}));
});

test('a complete preparation may be inspected despite full-plan STOP, but never installed as a full sequence',()=>{
  const f=fixture();assert.equal(preparationPhaseAvailable(f.row,'prepared'),true);assert.equal(preparationPhaseAvailable(f.row,'withdraw'),true);
  f.row.comparison.publicPlan.decision={operation:'STOP',selectedId:'stop'};
  assert.equal(preparationPhaseAvailable(f.row,'prepared'),true);assert.equal(preparationPhaseAvailable(f.row,'withdraw'),false);
  assert.equal(preparationCost(f.row).additionalNonTargetMm3,43);
  assert.equal(preparationCost(f.row).elapsedSeconds,23.8);
});

test('generator/evidence refusals and incomplete costly attempts have no preview or installation capability',()=>{
  const f=fixture();const variants=[{...f.row,preparation:null,preparedAdmission:null,missingEvidence:['functional']},
    {...f.row,preparation:{...f.row.preparation,publicPlan:{...f.row.preparation.publicPlan,attempts:[],decision:{operation:'STOP',selectedIndex:null}}},preparedAdmission:null}];
  for(const row of variants){assert.equal(preparationPhaseAvailable(row,'prepared'),false);assert.equal(preparationPhaseAvailable(row,'withdraw'),false);assert.equal(preparationCost(row),null);}
  const failed=structuredClone(f.row);failed.preparation.publicPlan.decision={operation:'STOP',selectedIndex:null};failed.preparation.publicPlan.attempts[0].complete=false;failed.preparation.publicPlan.attempts[0].failure='PREPARATION_ACTION_OR_TIME_BUDGET';failed.preparedAdmission=null;
  assert.equal(preparationCost(failed).additionalNonTargetMm3,43,'failed virtual attempt cost remains visible');
  assert.equal(preparationPhaseAvailable(failed,'prepared'),false);
});

test('source admission and complete/withdrawn/positive cost flags are all necessary for preparation actions',()=>{
  const f=fixture();const edits=[r=>r.preparedAdmission=null,r=>r.preparation.schema='resectionlab.source-exposure-preparation.v1',
    r=>r.preparation.publicPlan.decision.operation='STOP',r=>r.preparation.publicPlan.attempts[0].complete=1,
    r=>r.preparation.publicPlan.attempts[0].failure='failed',r=>r.preparation.publicPlan.attempts[0].summary.withdrawn=false,
    r=>r.preparation.publicPlan.attempts[0].summary.additionalTargetMm3=1,r=>r.preparation.publicPlan.attempts[0].summary.additionalNonTargetMm3=0,
    r=>r.preparation.publicPlan.attempts[0].summary.elapsedSeconds=NaN];
  for(const edit of edits){const row=structuredClone(f.row);edit(row);assert.equal(preparationPhaseAvailable(row,'prepared'),false);}
});

test('full-sequence phase requires exact feasible progress and withdrawal on the selected suffix',()=>{
  const f=fixture();const edits=[r=>r.comparison=null,r=>r.comparison.publicPlan.decision.selectedId='missing',r=>r.comparison.publicPlan.decision.operation='STOP',
    r=>r.comparison.publicPlan.alternatives[0].feasible=false,r=>r.comparison.publicPlan.alternatives[0].taskProgressMet=false,
    r=>r.comparison.publicPlan.alternatives[0].taskProgressMet=1,r=>r.comparison.publicPlan.alternatives[0].withdrawal='incomplete'];
  for(const edit of edits){const row=structuredClone(f.row);edit(row);assert.equal(preparationPhaseAvailable(row,'withdraw'),false);assert.equal(preparationPhaseAvailable(row,'prepared'),true);}
});

test('preview identity includes source report, candidate, phase, kind and exact displayed anatomy object',()=>{
  const f=fixture();assert.equal(preparationPreviewMatches(f.report,f.row,'prepared',f.preview,f.displayed),true);
  for(const patch of [{kind:'strategy'},{kind:undefined},{phaseId:'withdraw'},{candidateId:'width-1'},{reportSeal:'old'},
    {view:{...f.displayed}}])assert.equal(preparationPreviewMatches(f.report,f.row,'prepared',{...f.preview,...patch},f.displayed),false);
  assert.equal(preparationPreviewMatches(f.report,f.row,'withdraw',f.preview,f.displayed),false);
  assert.equal(preparationPreviewMatches(f.report,f.row,'prepared',f.preview,null),false);
  assert.equal(preparationPreviewMatches(null,f.row,'prepared',f.preview,f.displayed),false);
});

test('install requires acknowledgement, unlocked exact draft, selected row and matching accepted preview',()=>{
  const f=fixture();const input={response:f.response,draft:f.controls,submitted:f.submitted,revision:4,candidateId:f.row.id,phase:'prepared',preview:f.preview,displayed:f.displayed,pending:false,acknowledged:true};
  assert.equal(preparationActions(input).installAllowed,true);
  for(const patch of [{acknowledged:false},{pending:true},{candidateId:'width-1'},{phase:'withdraw'},{revision:5},{submitted:null},
    {preview:{...f.preview,reportSeal:'old'}},{displayed:{...f.displayed}}])assert.equal(preparationActions({...input,...patch}).installAllowed,false);
  const fresh=preparationActions({...input,preview:null,displayed:null});assert.equal(fresh.previewAllowed,true);assert.equal(fresh.installAllowed,false);
  assert.equal(preparationActions({...input,preview:{...f.preview,kind:'strategy'}}).previewAllowed,false,'cancel a foreign preview first');
});

test('budget exhaustion is distinguished from finite generator refusals or ordinary sequence STOP',()=>{
  const f=fixture();assert.equal(preparationIncomplete(f.row),false);
  assert.equal(preparationIncomplete({...f.row,reason:'COMPUTATION_BUDGET'}),true);
  const changed=structuredClone(f.row);changed.comparison.publicPlan.approaches.rejections=[{reason:'GEOMETRY_WALL_BUDGET'}];
  assert.equal(preparationIncomplete(changed),true);
  assert.equal(preparationIncomplete({...f.row,reason:'SECONDARY_TOOL_COLLISION'}),false);
});

test('action authority also refuses a missing context choice or points between source cells',()=>{
  const f=fixture();const input={response:f.response,draft:f.controls,submitted:f.submitted,revision:4,candidateId:f.row.id,phase:'prepared',preview:f.preview,displayed:f.displayed,pending:false,acknowledged:true};
  f.controls.encounter={layerId:'no-longer-available',marginMm:0,maxNearMm3:1000};
  f.report.controls=structuredClone(f.controls);
  assert.equal(preparationActions({...input,contexts:[]}).installAllowed,false);
  assert.equal(preparationActions({...input,contexts:[{layerId:'no-longer-available'}]}).installAllowed,true);
  f.controls.sourceRay.surfaceMm[0]+=.25;f.report.controls=structuredClone(f.controls);
  assert.equal(preparationActions({...input,contexts:[{layerId:'no-longer-available'}]}).installAllowed,false);
});

test('iterative source burdens stay separate from additional preparation cost and suffix progress',()=>{
  const f=fixture();f.report.baselineBurden={targetRemovedMm3:2,nonTargetRemovedMm3:101,elapsedSeconds:60.2};
  assert.equal(preparationBaselineText(f.report),'Prior recorded burdens: 2.00 mm³ target and 101.00 mm³ non-target removed; 60.20 s elapsed. These persist before the additional preparation and sequence costs below.');
  assert.equal(preparationCost(f.row).additionalNonTargetMm3,43);
  assert.equal(f.row.comparison.publicPlan.alternatives[0].summary.additionalTargetMm3,1);
  assert.equal(preparationBaselineText({...f.report,baselineBurden:undefined}),null,'old saved reports do not invent a zero history');
  assert.equal(preparationBaselineText({...f.report,baselineBurden:{targetRemovedMm3:0,nonTargetRemovedMm3:0,elapsedSeconds:0}})?.includes('0.00 mm³'),true);
  assert.equal(preparationBaselineText({...f.report,baselineBurden:{...f.report.baselineBurden,elapsedSeconds:NaN}}),null);
});
