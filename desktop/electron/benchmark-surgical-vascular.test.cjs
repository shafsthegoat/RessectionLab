'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {COUNTS,expectedSequence,submittedVascularPlayback,exactSequence,checkedVascularAuthority,hardBudgetStop,isVascularBenchmarkEntry,vascularPreparedOnly,vascularStartButton,vascularPreparationChecks,qualifiedFlowCapture}=require('./benchmark-surgical-vascular.cjs');
const options={startedRenderer:1000};
const response=(position,frameIndex,stateId='shared')=>({command:'seekPrefixFrame',branchId:'full-vascular',position,frameIndex,stateId,started:1100+frameIndex,ms:10});
const frame=(position,frameIndex,at=1200,stateId='shared')=>({at,serial:frameIndex+1,surgicalStateId:stateId,surgicalFrame:{branchId:'full-vascular',position,frameIndex}});
const received=(requests)=>requests.map((row,index)=>({command:row.command,branchId:row.branchId,position:row.position,frameIndex:row.frameIndex,stateId:row.stateId,notificationSerial:index+1,receivedAt:1030+index}));
const submitted=(requests,frames,extra={})=>submittedVascularPlayback(requests,frames,{...options,rendererReceipts:received(requests),...extra});
test('flow screenshots require a stable nonempty drawn interval across asynchronous capture',()=>{
  const state='sha256:'+'a'.repeat(64),before={frame:{branchId:'full-vascular',position:49,frameIndex:1},stateId:state,panelStateId:state,at:100,flow:{startSeconds:190,endSeconds:190.2,operations:3,glyphInstances:4,truncated:false}};
  const after={...structuredClone(before),at:101};assert.equal(qualifiedFlowCapture(before,after),true);
  for(const change of [r=>r.frame.position++,r=>r.frame.frameIndex++,r=>r.stateId='other',r=>r.panelStateId='other',r=>r.at=99,r=>r.flow.operations=0,r=>r.flow.glyphInstances=0,r=>r.flow.endSeconds++,r=>r.flow=null]){
    const altered=structuredClone(after);change(altered);assert.equal(qualifiedFlowCapture(before,altered),false);
  }
  assert.equal(qualifiedFlowCapture(undefined,after),false);assert.equal(qualifiedFlowCapture(before,null),false);
});
test('predeclared full vascular sequence includes all starts/microsteps/finalizations',()=>{
  const rows=expectedSequence();assert.equal(COUNTS.length,73);assert.equal(rows.length,1149);
  assert.deepEqual(rows[0],{branchId:'full-vascular',position:0,frameIndex:0});assert.deepEqual(rows.at(-1),{branchId:'full-vascular',position:72,frameIndex:2});
});
test('all-action matcher preserves identical hashes on different observed ordinals and boundaries',()=>{
  const r=[response(0,1),response(0,2),response(1,0)];const f=[frame(0,1),frame(0,2),frame(0,2,1201),frame(1,0,1202)];
  const shown=submitted(r,f);assert.deepEqual(shown.map(x=>[x.position,x.frameIndex]),[[0,1],[0,2],[1,0]]);assert.equal(shown.length,3);
});
test('unmatched physical state/cpu draw before completed receipt remains an explicit failure',()=>{
  const rows=submitted([response(0,0)],[frame(0,0,1001),frame(0,0,1200)]);
  assert.equal(rows[0].unmatched,true);assert.equal(exactSequence(rows,[{branchId:'full-vascular',position:0,frameIndex:0}]),false);
});
test('ordered returns/refinement and serial refetch do not mask backwards CPU admissions',()=>{
  const r=[response(0,0),response(0,1),{...response(0,1),started:1110}];const f=[frame(0,0),frame(0,1,1201),frame(0,0,1202)];
  const rows=submitted(r,f);assert.deepEqual(rows.map(x=>x.frameIndex),[0,1,0]);assert.equal(exactSequence(rows,[{branchId:'full-vascular',position:0,frameIndex:0},{branchId:'full-vascular',position:0,frameIndex:1}]),false);
});
test('full exact sequence also requires the independently audited fresh state identities',()=>{
  const expected=expectedSequence().map((r,i)=>({...r,stateId:'state'+i}));const requests=expected.map(r=>({...r,command:'seekPrefixFrame',started:1100,ms:10}));
  const frames=expected.map((r,i)=>({at:1200+i,serial:i,surgicalStateId:r.stateId,surgicalFrame:r}));
  const rows=submitted(requests,frames);assert.equal(exactSequence(rows,expected),true);
  assert.equal(exactSequence(rows.map((r,i)=>i===1100?{...r,stateId:'altered'}:r),expected),false);
});
test('lookahead does not count, wrong authority does not match, and foreign branch draws fail',()=>{
  assert.deepEqual(submitted([response(0,0),response(0,1)],[frame(0,0,999)]),[]);
  const rows=submitted([{...response(0,0),command:'seekFrame'}],[frame(0,0)]);assert.equal(rows[0].unmatched,true);
  const foreign=frame(0,0);foreign.surgicalFrame.branchId='other';assert.equal(submitted([response(0,0)],[foreign])[0].unmatched,true);
});

test('receipt causality uses renderer timestamps despite unrelated main clock origins',()=>{
  const requests=[{...response(0,0),started:900000,ms:5000}];
  const rows=submitted(requests,[frame(0,0)]);
  assert.equal(rows[0].unmatched,undefined);assert.equal(rows[0].rendererReceivedAt,1030);
  assert.equal(rows[0].receiptNotificationSerial,1);
});
test('missing, late, wrong-state and ambiguous renderer receipt proofs fail',()=>{
  const requests=[response(0,0)],frames=[frame(0,0)];
  for(const rendererReceipts of [[],[{...received(requests)[0],receivedAt:1200.001}],
    [{...received(requests)[0],stateId:'other'}],received([...requests,...requests])]){
    assert.equal(submitted(requests,frames,{rendererReceipts})[0].unmatched,true);
  }
  assert.equal(submitted([...requests,...requests],frames)[0].unmatched,true);
  assert.equal(submitted(requests,frames,{rendererReceipts:[{...received(requests)[0],receivedAt:1200}]})[0].unmatched,undefined);
});
test('renderer notification sequence cannot move backwards or reuse a serial',()=>{
  const requests=[response(0,0),response(0,1)],frames=[frame(0,0),frame(0,1,1201)];
  for(const patch of [{notificationSerial:1},{receivedAt:1029},{notificationSerial:0}]){
    const proof=received(requests);proof[1]={...proof[1],...patch};
    assert(submitted(requests,frames,{rendererReceipts:proof}).every(row=>row.unmatched));
  }
});

test('realistic imported endpoint can retain original read-only director metadata',()=>{
  const imported={branchId:'full-vascular',position:73,changed:false,fullCase:{branchId:'full-vascular',actions:73,phases:[{id:'injury',startPosition:45,endPosition:49}]},verification:{status:'unverified-import',readOnly:true,canonical:false,physicsActionsExecuted:0}};
  assert.equal(checkedVascularAuthority('inspectArchive',imported),imported);
  for(const patch of [{changed:true},{verification:{...imported.verification,readOnly:false}},{verification:{...imported.verification,canonical:true}},{verification:{...imported.verification,status:'verified'}}])assert.throws(()=>checkedVascularAuthority('inspectArchive',{...imported,...patch}));
});
test('prefix frames require separate read-only authority and no editable director',()=>{
  const prefix={branchId:'full-vascular',position:1,changed:false,readOnly:true,fullCase:null,verification:{status:'reconstructed-prefix',readOnly:true,totalActions:73,verifiedThrough:1}};
  assert.equal(checkedVascularAuthority('seekPrefixFrame',prefix),prefix);
  for(const patch of [{fullCase:{actions:73}},{branchId:'full-primary'},{readOnly:false},{verification:{...prefix.verification,status:'verified'}},{verification:{...prefix.verification,readOnly:false}}])assert.throws(()=>checkedVascularAuthority('seekPrefixFrame',{...prefix,...patch}));
  assert.throws(()=>checkedVascularAuthority('step',prefix));
});

test('hard wall watchdog kills and exits even if receipt or kill fails',()=>{
  for(const failing of ['write','kill']){const calls=[];assert.throws(()=>hardBudgetStop(()=>{calls.push('write');if(failing==='write')throw Error('generated disk failure');},()=>{calls.push('kill');if(failing==='kill')throw Error('generated kill failure');},code=>calls.push('exit'+code)));assert.deepEqual(calls,['write','kill','exit1']);}
});

test('Electron browser argv entry starts despite require.main being false',()=>{
  const path=require('node:path'),filename=path.resolve(__dirname,'benchmark-surgical-vascular.cjs');
  assert.equal(isVascularBenchmarkEntry({versions:{electron:'44.5.1'},type:'browser',argv:['Electron',filename]},filename,false),true);
  assert.equal(isVascularBenchmarkEntry({versions:{electron:'44.5.1'},type:'browser',argv:['Electron',path.relative(process.cwd(),filename)]},filename,false),true);
});
test('mere imports, foreign browser entry and Electron renderers never start the harness',()=>{
  const filename='/workspace/benchmark-surgical-vascular.cjs';
  assert.equal(isVascularBenchmarkEntry({versions:{node:'26'},argv:['node','/workspace/tests.cjs']},filename,false),false);
  assert.equal(isVascularBenchmarkEntry({versions:{node:'26'},argv:['node',filename]},filename,false),false);
  assert.equal(isVascularBenchmarkEntry({versions:{electron:'44'},type:'browser',argv:['Electron','/workspace/main.cjs']},filename,false),false);
  assert.equal(isVascularBenchmarkEntry({versions:{electron:'44'},type:'renderer',argv:['Electron',filename]},filename,true),false);
  assert.equal(isVascularBenchmarkEntry({versions:{electron:'44'},type:'browser',argv:['Electron']},filename,true),false);
  assert.equal(isVascularBenchmarkEntry({versions:{node:'26'},argv:['node',filename]},filename,true),true);
});

test('prepared-only mode is explicit and selects the actual dedicated UI button',()=>{
  assert.equal(vascularPreparedOnly(undefined),false);assert.equal(vascularPreparedOnly('1'),true);
  for(const value of ['', '0','true',true,1,null])assert.throws(()=>vascularPreparedOnly(value));
  assert.equal(vascularStartButton(false),'Replay from start · read only');
  assert.equal(vascularStartButton(true),'Replay prepared observations · read only');
});
test('prepared and fresh qualification retain opposite step/load/publication requirements',()=>{
  const make=prepared=>({qualificationMode:prepared?'prepared-only':'fresh',failure:null,permittedSteps:Array(prepared?0:73).fill({}),forbiddenSteps:[],publicStepAttempts:{},ownership:{prefixStatistics:{preparedCache:{mode:prepared?'only':'prepare',loadedActions:prepared?73:0,publishedActions:prepared?0:73,misses:0,lastSkip:null}}}});
  for(const prepared of [false,true]){
    const g=make(prepared);assert.deepEqual(vascularPreparationChecks(g,prepared),{stepAuthority:true,preparedCache:true});
    assert.equal(vascularPreparationChecks(g,!prepared).stepAuthority,false);
    for(const patch of [{loadedActions:prepared?72:1},{publishedActions:prepared?1:72},{misses:1},{lastSkip:'missing action'},{mode:'disabled'}]){
      const changed=structuredClone(g);Object.assign(changed.ownership.prefixStatistics.preparedCache,patch);assert.equal(vascularPreparationChecks(changed,prepared).preparedCache,false);
    }
    for(const mutate of [x=>x.permittedSteps.push({}),x=>x.forbiddenSteps.push({}),x=>x.publicStepAttempts.unexpected=1,x=>x.failure={message:'failed'}]){
      const changed=structuredClone(g);mutate(changed);assert.equal(vascularPreparationChecks(changed,prepared).stepAuthority,false);
    }
  }
  assert.deepEqual(vascularPreparationChecks(undefined,true),{stepAuthority:false,preparedCache:false});
});
test('wire prefix cache mode must remain fixed while canonical and imported authority remain separate',()=>{
  const response={branchId:'full-vascular',changed:false,readOnly:true,fullCase:null,verification:{status:'reconstructed-prefix',readOnly:true,totalActions:73,preparedCache:{mode:'only'}}};
  assert.equal(checkedVascularAuthority('seekPrefixFrame',response,true),response);
  assert.throws(()=>checkedVascularAuthority('seekPrefixFrame',response,false),/prepared-only mode/);
});
