import assert from 'node:assert/strict';
import {test,before,after} from 'node:test';
import * as React from 'react';
import {createServer} from 'vite';
import {mkdtemp,rm,mkdir} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {checkedPrefixResponse,checkedPrefixWireResponse,checkedPrefixStartMode,replayAuthorityKey,surgicalAuthorityIssue,surgicalReplayRoutes} from '../src/surgical-authority.ts';
let server,cache,SurgicalPanel;
before(async()=>{
  const output=fileURLToPath(new URL('../../testcases/outputs/rl-readiness-20261022/',import.meta.url));await mkdir(output,{recursive:true});cache=await mkdtemp(path.join(output,'prefix-ui-vite-'));
  server=await createServer({root:fileURLToPath(new URL('../',import.meta.url)),cacheDir:cache,logLevel:'error',optimizeDeps:{noDiscovery:true,include:[]},server:{middlewareMode:true,hmr:false,ws:false},appType:'custom'});
  ({SurgicalPanel}=await server.ssrLoadModule('/src/SurgicalPanel.tsx'));
});
after(async()=>{await server?.close();if(cache)await rm(cache,{recursive:true,force:true});});
const prefix=()=>({readOnly:true,status:'reconstructed-prefix',verifiedThrough:4,totalActions:10,archiveEndpointVerified:false,originSnapshotHash:'origin',actionLogHash:'actions',implementation:'current',protectionContractHash:'guard',prefixImplementation:'prefix-v1'});
function controller(verification=prefix()){
  const calls=[],pose={position:[0,0,0],quaternion:[0,0,0,1]},view={schema:'test',stateId:'prefix4',scene:{sceneId:'scene',shape:[2,2,2],resolutionMm:[1,1,1],affine:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],regions:[],approximations:[]},catalog:[],slots:{left:{...pose,parameters:{},active:false,inserted:false,grip:[]}},metrics:{elapsedSeconds:0},removed:[],moved:[],cuts:[],contact:[],thermal:{},fluid:{},history:[]};
  const result={caseHash:'case',branchId:'main',branches:[{id:'main',steps:10},{id:'other',steps:8}],position:4,requestedPosition:9,budgetStopped:true,verification,timeline:Array.from({length:4},()=>({action:{operation:'inspect'},status:'accepted',reason:null,events:[]})),view,outcome:null};
  const noop=()=>{};return {calls,result,view,preview:null,slot:'left',pose,path:[],parameters:{},pending:false,mode:'navigate',playbackVersion:0,run:async(command,extra)=>{calls.push({command,...extra});return true;},cancel:noop,pauseFullCase:noop,stopPlayback:noop,startPlayback:()=>true,runPlayback:async()=>true,setMode:noop,setPose:noop,selectSlot:noop,setParameters:noop,clearPreview:noop,action:()=>({}),create:noop};
}
function render(c){
  const internals=React.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,previous=internals.H;
  internals.H={useRef:value=>({current:value}),useState:value=>[typeof value==='function'?value():value,()=>{}],useEffect:()=>{}};
  try{return SurgicalPanel({c,available:true,onFixture:()=>{}});}finally{internals.H=previous;}
}
function* walk(node){if(Array.isArray(node)){for(const child of node)yield*walk(child);}else if(node&&typeof node==='object'&&node.props){yield node;yield*walk(node.props.children);}}
const text=node=>Array.isArray(node)?node.map(text).join(''):node?.props?text(node.props.children):typeof node==='string'||typeof node==='number'?String(node):'';
const find=(tree,type,label)=>[...walk(tree)].find(node=>node.type===type&&(node.props['aria-label']===label||text(node)===label));

test('imported endpoint offers explicit read-only start and retains full verification while ordinary replay stays blocked',async()=>{
  const c=controller({readOnly:true,status:'unverified-import',causalReplayVerified:false,limitations:[]}),tree=render(c);
  assert(text(tree).includes('Imported endpoint · unverified · read only'));assert.equal(find(tree,'button','Play replay').props.disabled,true);
  assert.equal(find(tree,'input','Surgical timeline').props.disabled,true);
  find(tree,'button','Replay from start · read only').props.onClick();await Promise.resolve();assert.deepEqual(c.calls.at(-1),{command:'seekPrefix',position:0,branchId:'main'});
  find(tree,'button','Replay prepared observations · read only').props.onClick();await Promise.resolve();assert.deepEqual(c.calls.at(-1),{command:'seekPrefix',position:0,branchId:'main',cacheOnly:true});
  find(tree,'button','Verify replay for planning').props.onClick();assert.deepEqual(c.calls.at(-1),{command:'inspect',branchId:'main'});
});
test('prepared-only prefix explains misses and offers an explicit fresh restart without enabling edits',async()=>{
  const c=controller({...prefix(),preparedCache:{mode:'only',loadedActions:4,publishedActions:0,misses:1,lastSkip:'missing action'},failure:{actionIndex:4,reason:'Prepared observations unavailable'}}),tree=render(c);
  assert(text(tree).includes('Prepared observations · read only'));assert(text(tree).includes('Missing observations stop replay without reconstructing actions.'));
  assert(text(tree).includes('Start a fresh read-only reconstruction to prepare missing actions.'));
  assert.equal(find(tree,'button','Branch').props.disabled,true);
  find(tree,'button','Replay from start · read only').props.onClick();await Promise.resolve();assert.deepEqual(c.calls.at(-1),{command:'seekPrefix',position:0,branchId:'main'});
});
test('prefix labels progress and budget stop; boundary, timeline and branch controls remain in their authority domains',()=>{
  const c=controller(),tree=render(c);assert(text(tree).includes('Reconstructed prefix · read only'));assert(text(tree).includes('Reconstructed 4 / 10 recorded actions'));assert(text(tree).includes('boundary 4 while requesting 9'));assert(text(tree).includes('Full archive planning verification is still required'));
  assert.equal(find(tree,'button','Play replay').props.disabled,false);assert.equal(find(tree,'input','Surgical timeline').props.max,10);
  find(tree,'button','Next recorded state').props.onClick();assert.deepEqual(c.calls.at(-1),{command:'seekPrefix',branchId:'main',position:5});
  find(tree,'input','Surgical timeline').props.onChange({target:{value:'9'}});assert.deepEqual(c.calls.at(-1),{command:'seekPrefix',branchId:'main',position:9});
  find(tree,'select','Surgical branch').props.onChange({target:{value:'other'}});assert.deepEqual(c.calls.at(-1),{command:'inspectArchive',branchId:'other'});
  const timeline=[...walk(tree)].find(node=>node.props.summary==='Action timeline').props.children();const rows=[...walk(timeline)].filter(node=>node.type==='button');assert.equal(rows.length,4);rows[0].props.onClick();assert.deepEqual(c.calls.at(-1),{command:'seekPrefix',branchId:'main',position:1});
  const instrument=[...walk(tree)].find(node=>node.type==='fieldset'&&[...walk(node)].some(child=>child.props['aria-label']==='Instrument interaction'));assert.equal(instrument.props.disabled,true);
});
test('prefix microstep controls select prefix frames and never canonical seeks',()=>{
  const c=controller();c.result.position=0;c.result.playback={readOnly:true,branchId:'main',position:0,actionIndex:0,frameIndex:1,frameCount:3,observedTimeSeconds:.2,actionElapsedSeconds:.2,actionDurationSeconds:.4,isActionBoundary:false,frames:[0,1,2].map(i=>({frameIndex:i,observedTimeSeconds:i*.2,kind:'microstep'}))};
  const tree=render(c);find(tree,'button','Next microstep').props.onClick();assert.deepEqual(c.calls.at(-1),{command:'seekPrefixFrame',branchId:'main',position:0,frameIndex:2});
  find(tree,'button','Return to recorded state 0').props.onClick();assert.deepEqual(c.calls.at(-1),{command:'seekPrefix',branchId:'main',position:0});
});
test('malformed prefix progress and authority drift cannot become canonical or authorize mutations',()=>{
  const r=controller().result;assert.equal(checkedPrefixResponse(r).verifiedThrough,4);
  for(const mutate of [x=>x.verification.verifiedThrough=11,x=>x.timeline.pop(),x=>x.position=9,x=>x.verification.originSnapshotHash='',x=>x.requestedPosition=11]){const copy=structuredClone(r);mutate(copy);assert.throws(()=>checkedPrefixResponse(copy),/prefix authority/);}
  const drift=structuredClone(r);drift.verification.protectionContractHash='other';assert.notEqual(replayAuthorityKey(drift),replayAuthorityKey(r));
  const prepared=structuredClone(r);prepared.verification.preparedCache={mode:'only',loadedActions:4,publishedActions:0,misses:0,lastSkip:null};assert.notEqual(replayAuthorityKey(prepared),replayAuthorityKey(r));assert.equal(replayAuthorityKey(prepared,false),replayAuthorityKey(r,false));
  for(const cache of [null,{mode:'other',loadedActions:0,publishedActions:0,misses:0,lastSkip:null},{mode:'only',loadedActions:-1,publishedActions:0,misses:0,lastSkip:null},{mode:'only',loadedActions:0,publishedActions:0,misses:0,lastSkip:1}])assert.throws(()=>checkedPrefixResponse({...r,verification:{...r.verification,preparedCache:cache}}),/prefix authority/);
  for(const command of ['step','plan','branch','fullCase','seek','seekFrame','selectBranch'])assert(surgicalAuthorityIssue(r,command,{position:0}));
  assert.deepEqual(surgicalReplayRoutes(r),{prefix:true,imported:true,canReplay:true,boundary:'seekPrefix',frame:'seekPrefixFrame',branch:'inspectArchive'});
});
test('compact prefix wire authority requires explicit omission and a real generation, while display validation requires restored receipts',()=>{
  const full=controller().result,wire={...full,timelineToken:'prefix-replay.v1/owner/4',timelineOmitted:true};delete wire.timeline;
  assert.equal(checkedPrefixWireResponse(wire).verifiedThrough,4);assert.throws(()=>checkedPrefixResponse(wire),/prefix authority/);
  assert.equal(checkedPrefixResponse({...wire,timeline:full.timeline}).verifiedThrough,4);
  for(const change of [{timelineOmitted:undefined},{timelineOmitted:1},{timelineToken:undefined},{timelineToken:''},{timelineToken:42},{timeline:full.timeline}])assert.throws(()=>checkedPrefixWireResponse({...wire,...change}),/prefix authority/);
});

function freshPipelineResponse(){
  const c=controller({...prefix(),preparedCache:{mode:'only',loadedActions:4,publishedActions:0,misses:0,lastSkip:null}}),r=c.result;
  r.readOnly=true;r.budgetStopped=false;r.view.scene.sceneId='sha256:'+'a'.repeat(64);
  r.prefixOrigin={schema:'resectionlab.prefix-origin.v1',kind:'fresh-reconstruction',activationId:'123e4567-e89b-42d3-a456-426614174000',
    caseHash:r.caseHash,branchId:r.branchId,sourceArchiveSha256:'b'.repeat(64),sceneId:r.view.scene.sceneId};
  return c;
}
test('fresh main origin preserves real consumer-only counters and explicit prepared-only start distinction',()=>{
  const r=freshPipelineResponse().result;
  assert.equal(checkedPrefixStartMode(r,false).preparedCache.mode,'only');
  assert.throws(()=>checkedPrefixStartMode(r,true),/prepared-only mode/);
  const unmarked=structuredClone(r);delete unmarked.prefixOrigin;
  assert.throws(()=>checkedPrefixStartMode(unmarked,false),/prepared-only mode/);
  assert.equal(checkedPrefixStartMode(unmarked,true).preparedCache.loadedActions,4);
  assert.notEqual(replayAuthorityKey(r),replayAuthorityKey(unmarked));
  assert.equal(replayAuthorityKey(r,false),replayAuthorityKey(unmarked,false));
});
test('fresh origin validates exact fields and full scene while compact authority pins its prior origin',()=>{
  const r=freshPipelineResponse().result;
  for(const mutate of [x=>x.prefixOrigin.extra='unexpected',x=>delete x.prefixOrigin.sceneId,x=>x.prefixOrigin=null,
    x=>x.prefixOrigin.activationId='other',x=>x.prefixOrigin.sourceArchiveSha256='bad',x=>x.prefixOrigin.caseHash='foreign',
    x=>x.prefixOrigin.branchId='other',x=>x.prefixOrigin.sceneId='sha256:'+'c'.repeat(64),
    x=>x.verification.preparedCache.publishedActions=1,x=>x.readOnly=false]){
    const copy=structuredClone(r);mutate(copy);assert.throws(()=>checkedPrefixResponse(copy),/prefix authority/);
  }
  const compact=structuredClone(r);compact.timelineToken='owner';compact.timelineOmitted=true;delete compact.timeline;delete compact.view.scene;
  assert.equal(checkedPrefixWireResponse(compact).verifiedThrough,4);
  assert.throws(()=>checkedPrefixResponse({...compact,timeline:r.timeline}),/prefix authority/);
  assert.equal(checkedPrefixResponse({...compact,timeline:r.timeline,view:{...compact.view,scene:r.view.scene}}).verifiedThrough,4);
  const changed=structuredClone(compact);changed.prefixOrigin.activationId='123e4567-e89b-42d3-a456-426614174001';
  assert.notEqual(replayAuthorityKey(changed),replayAuthorityKey(r));
});
test('fresh origin labels reconstruction while retaining read-only edits and actual consumer counters',()=>{
  const c=freshPipelineResponse(),tree=render(c);
  assert(text(tree).includes('Reconstructed prefix · read only'));assert(text(tree).includes('Reconstructed 4 / 10 recorded actions'));
  assert(text(tree).includes('Reconstructs the saved actions during playback.'));
  assert.equal(text(tree).includes('previously prepared observations'),false);
  assert.equal(text(tree).includes('Prepared observations · read only'),false);
  assert.equal(find(tree,'button','Branch').props.disabled,true);
  assert.equal(c.result.verification.preparedCache.mode,'only');
});

test('checked withdrawal sends a bounded recovery command and stays unavailable during read-only playback',()=>{
  const c=controller();delete c.result.verification;delete c.result.readOnly;
  Object.assign(c.view.slots.left,{instrument:'cusa',inserted:true,active:true,grip:[]});
  c.view.holds={};c.view.metrics.nonTargetRemovedMm3=10;
  let tree=render(c);const button=find(tree,'button','Checked withdrawal');
  assert(button);assert.equal(button.props.disabled,false);
  button.props.onClick();assert.deepEqual(c.calls.at(-1),{command:'recoverExit'});
  c.result.readOnly=true;tree=render(c);
  const instrument=[...walk(tree)].find(node=>node.type==='fieldset'&&[...walk(node)].some(child=>child.props['aria-label']==='Instrument interaction'));
  assert.equal(instrument.props.disabled,true);
  delete c.result.readOnly;c.result.exitRecovery={sourceStateId:c.view.stateId};
  assert.equal(find(render(c),'button','Checked withdrawal').props.disabled,true);
});
