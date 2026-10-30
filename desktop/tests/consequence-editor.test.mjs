import test,{before,after} from 'node:test';
import assert from 'node:assert/strict';
import * as React from 'react';
import {createServer} from 'vite';
import {mkdtemp,mkdir,rm} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {fixture,hash} from './consequence-editor-fixture.mjs';
let server,cache,Panel,fromDraft,toDraft;
before(async()=>{
  const output=fileURLToPath(new URL('../../testcases/outputs/consequence-editor-tests/',import.meta.url));await mkdir(output,{recursive:true});cache=await mkdtemp(path.join(output,'vite-'));
  server=await createServer({root:fileURLToPath(new URL('../',import.meta.url)),cacheDir:cache,logLevel:'error',optimizeDeps:{noDiscovery:true,include:[]},server:{middlewareMode:true,hmr:false,ws:false},appType:'custom'});
  ({ConsequenceComparisonPanel:Panel,consequencePolicyFromDraft:fromDraft,consequencePolicyDraft:toDraft}=await server.ssrLoadModule('/src/ConsequenceComparisonPanel.tsx'));
});
after(async()=>{await server?.close();if(cache)await rm(cache,{recursive:true,force:true});});
function* walk(node){if(Array.isArray(node)){for(const child of node)yield*walk(child);}else if(node?.props){yield node;yield*walk(node.props.children);}}
const text=node=>Array.isArray(node)?node.map(text).join(''):node?.props?text(node.props.children):typeof node==='string'||typeof node==='number'?String(node):'';
const find=(tree,type,label)=>[...walk(tree)].find(node=>node.type===type&&(node.props['aria-label']===label||text(node)===label));
function setup(result=fixture()){
  const calls=[],c={result,pending:false,run:async(command,args)=>{calls.push({command,...args});return true;}},states=[];let key,cursor;
  function render(){const outer=Panel({c,pause(){calls.push('pause');}}),editor=[...walk(outer)].find(node=>node.type?.name==='ConsequenceEditor');assert(editor);if(editor.key!==key){states.length=0;key=editor.key;}cursor=0;const internals=React.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,previous=internals.H;
    internals.H={useState(initial){const index=cursor++;if(!(index in states))states[index]=typeof initial==='function'?initial():initial;return [states[index],value=>{states[index]=typeof value==='function'?value(states[index]):value;}];}};
    try{return {outer,editor:editor.type(editor.props)};}finally{internals.H=previous;}}
  return {c,calls,render};
}
test('draft edits stay local, invalidate result, and submit exact bound reassessment only on click',()=>{
  const h=setup();let {editor}=h.render();assert(text(editor).includes('Current reassessment · revision 1'));
  find(editor,'input','Minimum additional target removal mm³').props.onChange({target:{value:'9'}});
  ({editor}=h.render());assert.equal(h.calls.length,0);assert(text(editor).includes('Draft changed'));assert(!text(editor).includes('Current reassessment'));
  const rows=[...walk(editor)].filter(node=>node.props['data-consequence-analysis-branch']);assert(rows.every(row=>text(row).includes('Reassessment required')));
  find(editor,'input','Require Motor encounter evidence').props.onChange({target:{checked:true}});({editor}=h.render());assert.equal(h.calls.length,0);
  find(editor,'button','Reassess checked alternatives').props.onClick();assert.deepEqual(h.calls,['pause',{command:'reevaluateConsequences',expectedComparisonSeal:hash(7),policyInput:{minimumAdditionalTargetMm3:9,maximumCumulativeNonTargetMm3:4940,requiredMetrics:['motorEncounterMm3'],maxBrainStrainSquaredMm3S:null,maxBrainStressSquaredPa2Mm3S:null}}]);
});
test('invalid draft cannot dispatch and advanced caps do not silently coerce blank required inputs',()=>{
  const h=setup();let {editor}=h.render();find(editor,'input','Cumulative non-target removal budget mm³').props.onChange({target:{value:''}});({editor}=h.render());assert.equal(find(editor,'button','Reassess checked alternatives').props.disabled,true);find(editor,'button','Reassess checked alternatives').props.onClick();assert.deepEqual(h.calls,[]);
  const input=h.c.result.consequenceAnalysis.latest.policyInput;assert.equal(fromDraft({...toDraft(input),minimumAdditionalTargetMm3:''}),null);assert.equal(fromDraft({...toDraft(input),maxBrainStressSquaredPa2Mm3S:'-1'}),null);assert.equal(fromDraft({...toDraft(input),maxBrainStressSquaredPa2Mm3S:'0'}).maxBrainStressSquaredPa2Mm3S,0);
});
test('view changes preserve historical analysis and only offer canonical common checkpoint recovery',()=>{
  const h=setup();h.render();Object.assign(h.c.result,{branchId:'STOP',position:2});h.c.result.view.stateId=hash(5);Object.assign(h.c.result.consequenceAnalysis,{context:{branchId:'STOP',position:2,stateId:hash(5)},available:false,current:false,reason:'common_checkpoint_required'});
  const {editor}=h.render();assert(text(editor).includes('historical at this view'));assert.equal(find(editor,'button','Reassess checked alternatives').props.disabled,true);find(editor,'button','Open common checkpoint').props.onClick();assert.deepEqual(h.calls,['pause',{command:'selectBranch',branchId:'prefix'}]);
  find(editor,'button','Inspect continue-a').props.onClick();assert.deepEqual(h.calls.at(-1),{command:'selectBranch',branchId:'continue-a',position:2});
});
test('cold reopen initializes saved policy, while new comparison identity resets unsaved edits',()=>{
  const result=fixture();result.consequenceAnalysis.latest.policyInput.minimumAdditionalTargetMm3=9;result.consequenceAnalysis.latest.policy.constraints[0].value=9;result.consequenceAnalysis.latest.decision.policy.constraints[0].value=9;
  const h=setup(result);let {editor}=h.render();assert.equal(find(editor,'input','Minimum additional target removal mm³').props.value,'9');
  find(editor,'input','Minimum additional target removal mm³').props.onChange({target:{value:'70'}});h.c.result=fixture();h.c.result.caseHash='other-case';h.c.result.consequenceAnalysis.caseHash='other-case';h.c.result.consequenceAnalysis.latest.caseHash='other-case';({editor}=h.render());assert.equal(find(editor,'input','Minimum additional target removal mm³').props.value,'8');
});
test('frozen report remains read-only and retains original decision attributes',()=>{
  const h=setup(),{outer}=h.render();const frozen=[...walk(outer)].find(node=>node.props['data-consequence-frozen-report']);assert(frozen);assert(text(frozen).includes('Original frozen experiment'));assert([...walk(frozen)].some(node=>node.props['data-consequence-decision']==='user-consequence-analysis'));assert(![...walk(frozen)].some(node=>['input','button'].includes(node.type)));assert(text(frozen).includes('Frozen declaration'));assert.equal(h.calls.length,0);
});
test('each compact alternative retains all cost dimensions and its canonical replay action',()=>{
  const h=setup(),{editor}=h.render(),cards=[...walk(editor)].filter(node=>node.props['data-consequence-analysis-branch']);
  assert.equal(cards.length,3);
  for(const card of cards){
    assert.equal(card.type,'article');
    for(const label of ['Additional target · mm³','Residual target · mm³','Cumulative non-target · mm³','Modeled duration · s'])assert([...walk(card)].some(node=>node.type==='dt'&&text(node)===label));
    assert([...walk(card)].some(node=>node.type==='button'&&node.props['data-consequence-replay']===card.props['data-consequence-analysis-branch']));
  }
  assert.equal(h.calls.length,0);
});
