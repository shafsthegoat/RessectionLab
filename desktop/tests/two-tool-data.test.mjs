import test from 'node:test';
import assert from 'node:assert/strict';
import {strategyControls,strategyControlKey} from '../src/two-tool-data.ts';

const controls={shortSeconds:.25,longSeconds:1,minimumTargetMm3:1,geometry:{candidateCount:3,requiredEvidence:['brain','target']}};
test('saved public context controls recover their native layer only at the UI boundary',()=>{
 const publicControls={...structuredClone(controls),encounter:{marginMm:3,maxNearMm3:4}};
 const saved={publicPlan:{constraints:publicControls},contextBinding:{layerId:'native-source-layer'}};
 const before=structuredClone(saved),draft=strategyControls(saved);
 assert.deepEqual(draft.encounter,{layerId:'native-source-layer',marginMm:3,maxNearMm3:4});
 draft.encounter.marginMm=2;draft.geometry.requiredEvidence.push('functional');
 assert.deepEqual(saved,before);
 assert.throws(()=>strategyControls({publicPlan:{constraints:publicControls}}),/source binding/);
});
test('each source-context edit invalidates the submitted comparison key',()=>{
 const draft={...structuredClone(controls),encounter:{layerId:'first',marginMm:3,maxNearMm3:4}};
 const key=strategyControlKey(draft);
 for(const edit of [{layerId:'second'},{marginMm:2},{maxNearMm3:0}]){
  assert.notEqual(strategyControlKey({...draft,encounter:{...draft.encounter,...edit}}),key);
 }
 const reordered=structuredClone(draft);reordered.geometry.requiredEvidence.reverse();
 assert.equal(strategyControlKey(reordered),key);
 assert.notEqual(strategyControlKey(controls),key);
});
test('historical comparisons without context retain ordinary independent drafts',()=>{
 const saved={publicPlan:{constraints:structuredClone(controls)}};
 const draft=strategyControls(saved);assert.deepEqual(draft,controls);assert(!('encounter' in draft));
 draft.shortSeconds=.5;assert.equal(saved.publicPlan.constraints.shortSeconds,.25);
});
