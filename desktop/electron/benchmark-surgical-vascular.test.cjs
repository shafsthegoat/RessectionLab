'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {COUNTS,expectedSequence,submittedVascularPlayback,exactSequence,checkedVascularAuthority,hardBudgetStop,isVascularBenchmarkEntry}=require('./benchmark-surgical-vascular.cjs');
const options={mainToRendererOffset:100,startedRenderer:1000};
const response=(position,frameIndex,stateId='shared')=>({command:'seekPrefixFrame',branchId:'full-vascular',position,frameIndex,stateId,started:1100+frameIndex,ms:10});
const frame=(position,frameIndex,at=1200,stateId='shared')=>({at,serial:frameIndex+1,surgicalStateId:stateId,surgicalFrame:{branchId:'full-vascular',position,frameIndex}});
test('predeclared full vascular sequence includes all starts/microsteps/finalizations',()=>{
  const rows=expectedSequence();assert.equal(COUNTS.length,73);assert.equal(rows.length,1149);
  assert.deepEqual(rows[0],{branchId:'full-vascular',position:0,frameIndex:0});assert.deepEqual(rows.at(-1),{branchId:'full-vascular',position:72,frameIndex:2});
});
test('all-action matcher preserves identical hashes on different observed ordinals and boundaries',()=>{
  const r=[response(0,1),response(0,2),response(1,0)];const f=[frame(0,1),frame(0,2),frame(0,2,1201),frame(1,0,1202)];
  const shown=submittedVascularPlayback(r,f,options);assert.deepEqual(shown.map(x=>[x.position,x.frameIndex]),[[0,1],[0,2],[1,0]]);assert.equal(shown.length,3);
});
test('unmatched physical state/cpu draw before completed receipt remains an explicit failure',()=>{
  const rows=submittedVascularPlayback([response(0,0)],[frame(0,0,1001),frame(0,0,1200)],options);
  assert.equal(rows[0].unmatched,true);assert.equal(exactSequence(rows,[{branchId:'full-vascular',position:0,frameIndex:0}]),false);
});
test('ordered returns/refinement and serial refetch do not mask backwards CPU admissions',()=>{
  const r=[response(0,0),response(0,1),{...response(0,1),started:1110}];const f=[frame(0,0),frame(0,1,1201),frame(0,0,1202)];
  const rows=submittedVascularPlayback(r,f,options);assert.deepEqual(rows.map(x=>x.frameIndex),[0,1,0]);assert.equal(exactSequence(rows,[{branchId:'full-vascular',position:0,frameIndex:0},{branchId:'full-vascular',position:0,frameIndex:1}]),false);
});
test('full exact sequence also requires the independently audited fresh state identities',()=>{
  const expected=expectedSequence().map((r,i)=>({...r,stateId:'state'+i}));const requests=expected.map(r=>({...r,command:'seekPrefixFrame',started:1100,ms:10}));
  const frames=expected.map((r,i)=>({at:1200+i,serial:i,surgicalStateId:r.stateId,surgicalFrame:r}));
  const rows=submittedVascularPlayback(requests,frames,options);assert.equal(exactSequence(rows,expected),true);
  assert.equal(exactSequence(rows.map((r,i)=>i===1100?{...r,stateId:'altered'}:r),expected),false);
});
test('lookahead does not count, wrong authority does not match, and foreign branch draws fail',()=>{
  assert.deepEqual(submittedVascularPlayback([response(0,0),response(0,1)],[frame(0,0,999)],options),[]);
  const rows=submittedVascularPlayback([{...response(0,0),command:'seekFrame'}],[frame(0,0)],options);assert.equal(rows[0].unmatched,true);
  const foreign=frame(0,0);foreign.surgicalFrame.branchId='other';assert.equal(submittedVascularPlayback([response(0,0)],[foreign],options)[0].unmatched,true);
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
