'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {submittedPlayback}=require('./playback-submissions.cjs');
const options={position:2,mainToRendererOffset:100,startedRenderer:1000};
const response=(index,extra={})=>({command:'seekFrame',branchId:'primary',position:2,frameIndex:index,stateId:`state${index}`,started:1100+index*100,ms:20,...extra});
const frame=(index,at=1030+index*100)=>({surgicalStateId:`state${index}`,surgicalFrame:{branchId:'primary',position:2,frameIndex:index},at,serial:index+1});

test('speculative responses never count before a CPU-submitted frame',()=>{
  const rows=submittedPlayback([response(0),response(1),response(2)],[frame(0),frame(1)],options);
  assert.deepEqual(rows.map(row=>row.frameIndex),[0,1]);
  assert.equal(rows[1].firstSubmittedAt,1130);
});
test('an earlier boundary draw cannot stand in for the later observed receipt',()=>{
  assert.deepEqual(submittedPlayback([response(0)],[frame(0,1010)],options),[]);
  assert.equal(submittedPlayback([response(0)],[frame(0,1010),frame(0,1030)],options).length,1);
});
test('refinement duplicates collapse but an out-of-order return stays detectable',()=>{
  const rows=submittedPlayback([response(0),response(1)],[frame(0),frame(0,1040),frame(1),frame(0,1150)],options);
  assert.deepEqual(rows.map(row=>row.frameIndex),[0,1,0]);
});
test('failed, different-action and pre-play submissions are not admitted',()=>{
  const rows=submittedPlayback([response(0,{error:'failed'}),response(1,{position:3}),response(2)],
    [frame(0),frame(1),frame(2,990)],options);
  assert.deepEqual(rows,[]);
});
test('serial refetch of a released lookahead counts exactly one submitted state',()=>{
  const rows=submittedPlayback([response(0),response(0,{started:1150,ms:10})],[frame(0,1170)],options);
  assert.deepEqual(rows.map(row=>row.frameIndex),[0]);
});
test('distinct observations with an identical physical state hash remain distinct',()=>{
  const rows=submittedPlayback([response(0,{stateId:'same'}),response(1,{stateId:'same'})],
    [{...frame(0),surgicalStateId:'same'},{...frame(1),surgicalStateId:'same'}],options);
  assert.deepEqual(rows.map(row=>row.frameIndex),[0,1]);
});
test('a different branch or absent recorded identity cannot stand in for playback',()=>{
  const wrong={...frame(0),surgicalFrame:{branchId:'different',position:2,frameIndex:0}};
  assert.deepEqual(submittedPlayback([response(0)],[wrong,{...frame(0),surgicalFrame:null}],options),[]);
});
test('same-ordinal identity changes cannot disappear behind refinement deduplication',()=>{
  const rows=submittedPlayback([response(0,{stateId:'A'}),response(0,{stateId:'B'})],
    [{...frame(0,1050),surgicalStateId:'A'},{...frame(0,1060),surgicalStateId:'B'},{...frame(0,1070),surgicalStateId:'A'}],options);
  assert.deepEqual(rows.map(row=>row.stateId),['A','B','A']);
});

 test('prefix observations require the explicit prefix command and retain exact tuples',()=>{
  const rows=[response(0),response(0,{command:'seekPrefixFrame'}),response(1,{command:'seekPrefixFrame',stateId:'state0'})];
  const frames=[frame(0),{...frame(1),surgicalStateId:'state0'}];
  const shown=submittedPlayback(rows,frames,{...options,command:'seekPrefixFrame'});
  assert.deepEqual(shown.map(row=>[row.command,row.frameIndex]),[['seekPrefixFrame',0],['seekPrefixFrame',1]]);
  assert.deepEqual(submittedPlayback(rows,frames,options).map(row=>row.frameIndex),[0]);
});
