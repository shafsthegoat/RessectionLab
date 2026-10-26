import test from 'node:test';
import assert from 'node:assert/strict';
import {SurgicalComparisonContext} from '../src/surgical-comparison-context.ts';
const full=()=>({caseHash:'case',view:{scene:{sceneId:'scene'}},planning:{alternatives:['baseline']},sectorPlanning:{alternatives:['near','far']}});
const frame=()=>({caseHash:'case',view:{scene:{sceneId:'scene'}},playback:{readOnly:true}});
test('comparison evidence stays visible during read-only frames without changing frame authority',()=>{
  const cache=new SurgicalComparisonContext(),source={},initial=full();cache.accept(initial,source,false);
  const wire=frame(),display=cache.accept(wire,source,true);
  assert.equal(display.planning,initial.planning);assert.equal(display.sectorPlanning,initial.sectorPlanning);
  assert.equal(display.playback,wire.playback);assert.equal(Object.hasOwn(wire,'planning'),false);
});
test('another source installation, scene or editable response cannot inherit comparison receipts',()=>{
  for(const [source,change] of [[{},{}],[null,{caseHash:'other'}],[null,{view:{scene:{sceneId:'other'}}}],[null,{playback:undefined}]]){
    const cache=new SurgicalComparisonContext(),owner={};cache.accept(full(),owner,false);
    assert.equal(cache.accept({...frame(),...change},source??owner,true).planning,undefined);
  }
});
test('ordinary responses replace prior evidence and explicit null is respected',()=>{
  const cache=new SurgicalComparisonContext(),source={};cache.accept(full(),source,false);
  assert.equal(cache.accept({...frame(),planning:null},source,true).planning,null);
  assert.equal(cache.accept(frame(),source,true).planning,null);
  cache.accept({...full(),planning:null,sectorPlanning:null},source,false);
  assert.equal(cache.accept(frame(),source,true).sectorPlanning,null);
});
