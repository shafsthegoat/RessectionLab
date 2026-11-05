import test from 'node:test';
import assert from 'node:assert/strict';
import {SurgicalComparisonContext} from '../src/surgical-comparison-context.ts';
const full=()=>({caseHash:'case',view:{scene:{sceneId:'scene'}},planning:{alternatives:['baseline']},sectorPlanning:{alternatives:['near','far']},jointPlanning:{jointSeal:'sha256:joint'},jointBaselineHistory:[{comparison:'classic'}],stagedPlanning:{comparisonSeal:'sha256:staged'},precisionPlanning:{comparisonSeal:'sha256:precision'},retractionStudy:{comparisonSeal:'sha256:bimanual'}});
const frame=()=>({caseHash:'case',view:{scene:{sceneId:'scene'}},playback:{readOnly:true}});
test('comparison evidence stays visible during read-only frames without changing frame authority',()=>{
  const cache=new SurgicalComparisonContext(),source={},initial=full();cache.accept(initial,source,false);
  const wire=frame(),display=cache.accept(wire,source,true);
  assert.equal(display.planning,initial.planning);assert.equal(display.sectorPlanning,initial.sectorPlanning);
  assert.equal(display.jointPlanning,initial.jointPlanning);
  assert.equal(display.jointBaselineHistory,initial.jointBaselineHistory);
  assert.equal(display.stagedPlanning,initial.stagedPlanning);
  assert.equal(display.retractionStudy,initial.retractionStudy);
  assert.equal(display.playback,wire.playback);assert.equal(Object.hasOwn(wire,'planning'),false);
});
test('another source installation, scene or editable response cannot inherit comparison receipts',()=>{
  for(const [source,change] of [[{},{}],[null,{caseHash:'other'}],[null,{view:{scene:{sceneId:'other'}}}],[null,{playback:undefined}]]){
    const cache=new SurgicalComparisonContext(),owner={};cache.accept(full(),owner,false);
    assert.equal(cache.accept({...frame(),...change},source??owner,true).planning,undefined);
    assert.equal(cache.accept({...frame(),...change},source??owner,true).jointPlanning,undefined);
    assert.equal(cache.accept({...frame(),...change},source??owner,true).stagedPlanning,undefined);
    assert.equal(cache.accept({...frame(),...change},source??owner,true).retractionStudy,undefined);
  }
});
test('ordinary responses replace prior evidence and explicit null is respected',()=>{
  const cache=new SurgicalComparisonContext(),source={};cache.accept(full(),source,false);
  assert.equal(cache.accept({...frame(),planning:null},source,true).planning,null);
  assert.equal(cache.accept(frame(),source,true).planning,null);
  assert.equal(cache.accept({...frame(),jointPlanning:null,jointBaselineHistory:null},source,true).jointPlanning,null);
  assert.equal(cache.accept(frame(),source,true).jointBaselineHistory,null);
  assert.equal(cache.accept({...frame(),stagedPlanning:null},source,true).stagedPlanning,null);
  assert.equal(cache.accept(frame(),source,true).stagedPlanning,null);
  assert.equal(cache.accept({...frame(),retractionStudy:null},source,true).retractionStudy,null);
  assert.equal(cache.accept(frame(),source,true).retractionStudy,null);
  cache.accept({...full(),planning:null,sectorPlanning:null,jointPlanning:null,jointBaselineHistory:null,stagedPlanning:null},source,false);
  assert.equal(cache.accept(frame(),source,true).sectorPlanning,null);
  assert.equal(cache.accept(frame(),source,true).jointPlanning,null);
  assert.equal(cache.accept(frame(),source,true).stagedPlanning,null);
});
test('legacy responses without joint fields leave static artifact fallback available',()=>{
  const cache=new SurgicalComparisonContext(),source={};
  const legacy=full();delete legacy.jointPlanning;delete legacy.jointBaselineHistory;delete legacy.stagedPlanning;
  cache.accept(legacy,source,false);
  const compact=cache.accept(frame(),source,true);
  assert.equal(Object.hasOwn(compact,'jointPlanning'),false);
  assert.equal(Object.hasOwn(compact,'jointBaselineHistory'),false);
  assert.equal(Object.hasOwn(compact,'stagedPlanning'),false);
});

test('precision receipts survive compact frames and explicit null revokes their fallback',()=>{
  const cache=new SurgicalComparisonContext(),source={},initial=full();cache.accept(initial,source,false);
  assert.equal(cache.accept(frame(),source,true).precisionPlanning,initial.precisionPlanning);
  assert.equal(cache.accept({...frame(),precisionPlanning:null},source,true).precisionPlanning,null);
  assert.equal(cache.accept(frame(),source,true).precisionPlanning,null);
  cache.accept(initial,source,false);
  assert.equal(cache.accept(frame(),{},true).precisionPlanning,undefined);
  cache.accept(initial,source,false);
  assert.equal(cache.accept({...frame(),view:{scene:{sceneId:'other'}}},source,true).precisionPlanning,undefined);
});

test('consequence policy evidence survives observed frames only as explicitly historical',()=>{
  const source={},cache=new SurgicalComparisonContext();
  const initial={...full(),branchId:'baseline',position:15,view:{stateId:'baseline-state',scene:{sceneId:'scene'}},
    consequenceComparison:{comparisonSeal:'frozen'},consequenceAnalysis:{caseHash:'case',comparisonSeal:'frozen',available:true,current:true,
      context:{branchId:'baseline',position:15,stateId:'baseline-state'},latest:{revision:1,policyInput:{minimumAdditionalTargetMm3:8},decision:{selectedBranchId:'short'}}}};
  cache.accept(initial,source,false);
  const wire={...frame(),branchId:'short',position:16,view:{stateId:'observed-frame',scene:{sceneId:'scene'}}};
  const display=cache.accept(wire,source,true);
  assert.equal(display.consequenceComparison,initial.consequenceComparison);
  assert.equal(display.consequenceAnalysis.latest,initial.consequenceAnalysis.latest);
  assert.equal(display.consequenceAnalysis.current,false);assert.equal(display.consequenceAnalysis.available,false);
  assert.deepEqual(display.consequenceAnalysis.context,{branchId:'short',position:16,stateId:'observed-frame'});
  assert.equal(initial.consequenceAnalysis.current,true);assert.equal(wire.consequenceAnalysis,undefined);
  assert.equal(cache.accept(wire,{},true).consequenceAnalysis,undefined,'new source installation loses saved analysis context');
  cache.accept(initial,source,false);cache.accept({...wire,consequenceAnalysis:null,consequenceComparison:null},source,true);
  const revoked=cache.accept(wire,source,true);assert.equal(revoked.consequenceAnalysis,null);assert.equal(revoked.consequenceComparison,null);
});

test('two-tool evidence survives observed frames as historical and cannot cross source identity',()=>{
  const cache=new SurgicalComparisonContext(),source={},strategy={current:true,reportSeal:'sealed',publicPlan:{decision:{selectedId:'long'}}};
  cache.accept({...full(),strategyPlanning:strategy},source,false);
  const wire=frame(),display=cache.accept(wire,source,true);
  assert.equal(display.strategyPlanning.current,false);assert.equal(display.strategyPlanning.publicPlan,strategy.publicPlan);
  assert.equal(strategy.current,true);assert.equal(wire.strategyPlanning,undefined);
  assert.equal(cache.accept(frame(),{},true).strategyPlanning,undefined);
  cache.accept({...full(),strategyPlanning:strategy},source,false);
  assert.equal(cache.accept({...frame(),strategyPlanning:null},source,true).strategyPlanning,null);
});
