import test from 'node:test';
import assert from 'node:assert/strict';
import * as React from 'react';
import {useSurgical} from '../src/use-surgical.ts';

// Exercise the actual hook's effects and ordered async requests without a DOM.
function hookHarness() {
  const slots=[],effects=[];let index=0,dirty=false;
  const internals=React.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
  const previous=internals.H;
  const same=(a,b)=>a&&b&&a.length===b.length&&a.every((v,i)=>Object.is(v,b[i]));
  const dispatcher={
    useRef(value){const i=index++;return slots[i]??=( {current:value});},
    useState(value){const i=index++;if(!(i in slots))slots[i]=typeof value==='function'?value():value;return [slots[i],v=>{slots[i]=typeof v==='function'?v(slots[i]):v;dirty=true;}];},
    useMemo(fn,deps){const i=index++;if(!slots[i]||!same(slots[i].deps,deps))slots[i]={value:fn(),deps};return slots[i].value;},
    useEffect(fn,deps){const i=index++;if(!slots[i]||!same(slots[i].deps,deps)){const old=slots[i];slots[i]={deps,cleanup:old?.cleanup};effects.push(()=>{old?.cleanup?.();slots[i].cleanup=fn();});}},
  };
  return {flushEffects(){for(const effect of effects.splice(0))effect();},render(source,api,{passive=true,initialArchivePreview=false}={}){let result;do{dirty=false;index=0;internals.H=dispatcher;try{result=useSurgical(source,api,true,initialArchivePreview);}finally{internals.H=previous;}if(passive)for(const effect of effects.splice(0))effect();}while(dirty);return result;},close(){for(const item of slots)item?.cleanup?.();internals.H=previous;}};
}
test('reopening the same patient invalidates in-flight rehearsal state and inspects the new workspace',async()=>{
  const harness=hookHarness(),calls=[];let resolveOld;
  const old=new Promise(resolve=>{resolveOld=resolve;});
  const source={caseHash:'same-patient'},reopened={caseHash:'same-patient'};
  const api={async surgicalRehearsal(request){calls.push(request);return calls.length===1?old:null;}};
  try {
    harness.render(source,api);
    assert.equal(calls.length,1);
    harness.render(reopened,api);
    resolveOld({caseHash:'same-patient',view:{stateId:'old-workspace-state',scene:{shape:[1,1,1]}}});
    await new Promise(resolve=>setImmediate(resolve));
    const result=harness.render(reopened,api);
    assert.equal(calls.length,2,'same hash must still inspect the newly opened workspace');
    assert.equal(calls[1].expectedStateId,undefined,'old editable identity cannot enter new workspace request');
    assert.equal(result.result,null);
    assert.equal(result.volume,null,'old response cannot become newly reopened anatomy');
    assert.equal(result.error,null);
  }finally{harness.close();}
});
test('hook acknowledges authenticated fields only after full receipt and hydrates compact anatomy before display admission',async()=>{
  const harness=hookHarness(),calls=[],affine=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],hash='a'.repeat(64);
  const regions=[{id:1,name:'brain',material:'soft_tissue',color:'#ccc',target:false}];
  const source={caseHash:'case',shape:[2,2,2],affine,frame:'RAS+',mri:new Float32Array(8),compartments:[{name:'brain',kind:'soft_tissue',color:'#ccc',mask:new Uint8Array(8).fill(1)}]};
  const field=[{cells:[1,2],separable:null}],base={stateId:'editable',scene:{shape:source.shape,affine,regions},catalog:[],slots:{left:{position:[0,0,0],quaternion:[0,0,0,1],parameters:{}}},metrics:{elapsedSeconds:0},removed:[],moved:[],history:[],flapComponents:field};
  const initial={caseHash:'case',branchId:'main',position:0,branches:[{id:'main',steps:1}],view:base,outcome:null};
  const playback=index=>({readOnly:true,branchId:'main',position:0,actionIndex:0,frameIndex:index,frameCount:2,observedTimeSeconds:index*.2,actionElapsedSeconds:index*.2,actionDurationSeconds:.2,isActionBoundary:true,frames:[{frameIndex:0,observedTimeSeconds:0,kind:'initial'},{frameIndex:1,observedTimeSeconds:.2,kind:'action_boundary'}]});
  const first={...initial,timelineToken:'generation',viewFieldTokens:{flapComponents:hash},timeline:[{action:{operation:'inspect'}}],playback:playback(0),view:{...base,stateId:'frame0'}};
  const wire={...initial,timelineToken:'generation',timelineOmitted:true,viewFieldTokens:{flapComponents:hash},viewFieldsOmitted:['flapComponents'],playback:playback(1),view:{stateId:'frame1',slots:base.slots,metrics:{elapsedSeconds:.2},removed:[0],moved:[],history:[]}};
  const api={async surgicalRehearsal(request){calls.push(request);return calls.length===1?initial:calls.length===2?first:wire;}};
  const flush=()=>new Promise(resolve=>setImmediate(resolve));
  const admit=()=>{let c=harness.render(source,api);c.interaction.onDisplayReady(c.interaction.view);return harness.render(source,api);};
  try{
    harness.render(source,api);await flush();let c=admit();assert.equal(c.view.stateId,'editable');
    await c.run('seekFrame',{branchId:'main',position:0,frameIndex:0});c=admit();
    assert.equal(calls[1].timelineToken,undefined);assert.equal(calls[1].viewFieldTokens,undefined);
    await c.run('seekFrame',{branchId:'main',position:0,frameIndex:1});c=admit();
    assert.equal(calls[2].timelineToken,'generation');assert.deepEqual(calls[2].viewFieldTokens,{flapComponents:hash});
    assert.equal(calls[2].expectedStateId,'editable','read-only acknowledgments cannot replace the editable cursor');
    assert.equal(c.view.stateId,'frame1');assert.equal(c.view.flapComponents,field);assert.equal(c.volume.compartments[0].mask[0],0);
    assert.equal(c.error,null);assert.equal(Object.hasOwn(wire.view,'scene'),false);assert.equal(Object.hasOwn(wire.view,'flapComponents'),false);
  }finally{harness.close();}
});

function replayFixture(intercept=()=>undefined,options={}){
  const affine=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],regions=[{id:1,name:'brain',material:'soft_tissue',color:'#ccc',target:false}];
  const source={caseHash:'case',shape:[2,2,2],affine,frame:'RAS+',mri:new Float32Array(8),compartments:[{name:'brain',kind:'soft_tissue',color:'#ccc',mask:new Uint8Array(8).fill(1)}]};
  const view={stateId:'editable',scene:{shape:source.shape,affine,regions},catalog:[],slots:{left:{position:[0,0,0],quaternion:[0,0,0,1],parameters:{}}},metrics:{elapsedSeconds:0},removed:[],moved:[],history:[],flapComponents:[]};
  const timeline=[{action:{operation:'inspect'}},{action:{operation:'inspect'}}],base={caseHash:'case',branchId:'main',position:0,branches:[{id:'main',steps:2},{id:'other',steps:2}],view,outcome:null};
  const frame=request=>{
    const {position,frameIndex,branchId}=request,time=position*.4+frameIndex*.2;
    const playback={readOnly:true,branchId,position,actionIndex:position,frameIndex,frameCount:3,observedTimeSeconds:time,actionElapsedSeconds:frameIndex*.2,actionDurationSeconds:.4,isActionBoundary:frameIndex===0||frameIndex===2,frames:[0,1,2].map(i=>({frameIndex:i,observedTimeSeconds:position*.4+i*.2,kind:i===2?'action_boundary':'microstep'}))};
    const full=!request.timelineToken,wireView={stateId:`${branchId}:${position}:${frameIndex}`,slots:view.slots,metrics:{elapsedSeconds:time},removed:Array.from({length:position*2+frameIndex},(_,i)=>i),moved:[],history:[]};
    return {...base,branchId,timelineToken:'generation',...(full?{timeline}: {timelineOmitted:true}),playback,view:full?{...view,...wireView}:wireView};
  };
  const calls=[],api={async surgicalRehearsal(request){calls.push(request);if(request.command==='seekFrame')assert.deepEqual(Object.keys(request).filter(key=>!['caseHash','command','expectedStateId','branchId','position','frameIndex','timelineToken','viewFieldTokens'].includes(key)),[],'seekFrame must respect the production IPC whitelist');const intercepted=intercept(request,calls);if(intercepted!==undefined)return intercepted;return request.command==='seekFrame'?frame(request):{...base,branchId:request.branchId??'main',view:{...view,stateId:request.command==='inspect'?'editable':`editable:${request.command}`}};}};
  const harness=hookHarness(),render=(s=source)=>harness.render(s,api,options),flush=()=>new Promise(resolve=>setImmediate(resolve));
  const admit=()=>{const c=render();c.interaction.onDisplayReady(c.interaction.view);return render();};
  return {source,api,calls,frame,harness,render,flush,admit,async open(){render();await flush();return admit();}};
}
const selection=(frameIndex,position=0,branchId='main')=>({branchId,position,frameIndex});
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};

test('automatic replay fetches ahead but admits every exact frame FIFO, with no extra volume until display acknowledgment',async()=>{
  const f=replayFixture();try{
    let c=await f.open();assert(c.startPlayback());
    await c.runPlayback(selection(0));await f.flush();c=f.render();
    assert.deepEqual(f.calls.filter(q=>q.command==='seekFrame').map(q=>q.frameIndex),[0,1]);
    assert.equal(c.view.stateId,'editable');assert.equal(c.interaction.view.stateId,'main:0:0');assert(c.displayPending);
    const activeVolume=c.volume;await f.flush();c=f.render();assert.equal(c.volume,activeVolume,'a prefetched receipt cannot create a modeled volume');
    assert.equal(await c.runPlayback(selection(1)),false,'successor cannot supersede active anatomy');
    const admitted=[];
    for(const [frameIndex,position] of [[0,0],[1,0],[2,0],[0,1]]){
      c=f.admit();admitted.push(c.view.stateId);
      if(position===1)break;
      const next=frameIndex===2?selection(0,1):selection(frameIndex+1);
      await c.runPlayback(next);await f.flush();c=f.render();
    }
    assert.deepEqual(admitted,['main:0:0','main:0:1','main:0:2','main:1:0']);
    assert(f.calls.filter(q=>q.command==='seekFrame').every(q=>q.expectedStateId==='editable'));
    assert.equal(c.error,null);
  }finally{f.harness.close();}
});
test('pause discards only lookahead and preserves active display; resume refetches the exact next frame',async()=>{
  const pending=deferred();let delayed=true;
  const f=replayFixture(q=>q.command==='seekFrame'&&q.frameIndex===1&&delayed?pending.promise:undefined);
  try{
    let c=await f.open();assert(c.startPlayback());await c.runPlayback(selection(0));await f.flush();c=f.render();
    const active=c.interaction.view,volume=c.volume;c.stopPlayback();pending.resolve(f.frame({...selection(1),timelineToken:'generation'}));await f.flush();
    c=f.render();assert.equal(c.interaction.view,active);assert.equal(c.volume,volume);c=f.admit();assert.equal(c.view.stateId,'main:0:0');assert.equal(c.pending,false);
    delayed=false;assert(c.startPlayback());await c.runPlayback(selection(1));c=f.admit();assert.equal(c.view.stateId,'main:0:1');
    assert.deepEqual(f.calls.filter(q=>q.command==='seekFrame').slice(0,3).map(q=>q.frameIndex),[0,1,1]);
    c.stopPlayback();const before=f.calls.length;assert.equal(await c.run('step',{action:{operation:'apply'}}),false);assert.equal(f.calls.length,before,'read-only replay cannot authorize editing');
    await c.run('seek',{position:0});c=f.admit();await c.run('step',{action:{operation:'apply'}});
    assert.equal(f.calls.at(-1).expectedStateId,'editable:seek','only the canonical seek can replace editable identity');
  }finally{f.harness.close();}
});
test('manual branch commands wait for lookahead transport, invalidate it and preserve ordinary queue barriers',async()=>{
  const pending=deferred(),f=replayFixture(q=>q.command==='seekFrame'&&q.frameIndex===1?pending.promise:undefined);
  try{
    let c=await f.open();assert(c.startPlayback());await c.runPlayback(selection(0));await f.flush();c=f.admit();
    const barrier=c.run('selectBranch',{branchId:'other'});await f.flush();assert.equal(f.calls.at(-1).command,'seekFrame');
    pending.resolve(f.frame({...selection(1),timelineToken:'generation'}));await barrier;c=f.admit();
    assert.equal(c.result.branchId,'other');assert.equal(c.view.stateId,'editable:selectBranch');
    assert.equal(f.calls.at(-1).expectedStateId,'editable');assert.equal(c.startPlayback(),true);
    assert.equal(await c.runPlayback(selection(1)),false,'old branch/ordinal cannot enter a restarted lane');
    assert.equal(c.view.stateId,'editable:selectBranch');
  }finally{f.harness.close();}
});
test('source generations and late prefetch failures cannot publish an obsolete view or suppress an active frame',async()=>{
  const pending=deferred(),f=replayFixture(q=>q.command==='seekFrame'&&q.frameIndex===1?pending.promise:undefined);
  try{
    let c=await f.open();assert(c.startPlayback());await c.runPlayback(selection(0));await f.flush();c=f.render();
    const active=c.interaction.view;c.interaction.onDisplayReady(active);c=f.render();assert.equal(c.view.stateId,'main:0:0');
    const reopened={...f.source};c=f.render(reopened);assert.equal(c.view,null);pending.reject(Error('obsolete prefetch failed'));await f.flush();
    c=f.render(reopened);if(c.interaction.view)c.interaction.onDisplayReady(c.interaction.view);c=f.render(reopened);
    assert.equal(c.view.stateId,'editable');assert.equal(c.error,null);assert.equal(f.calls.at(-1).expectedStateId,undefined);
  }finally{f.harness.close();}
});
test('demanded prefetch errors retain accepted anatomy and allow manual recovery without editable cursor pollution',async()=>{
  const pending=deferred(),f=replayFixture(q=>q.command==='seekFrame'&&q.frameIndex===1?pending.promise:undefined);
  try{
    let c=await f.open();assert(c.startPlayback());await c.runPlayback(selection(0));await f.flush();c=f.render();
    pending.reject(Error('recorded frame unavailable'));await f.flush();c=f.render();assert.equal(c.error,null,'future transport failure cannot reject the active display');
    c=f.admit();const accepted=c.volume;assert.equal(await c.runPlayback(selection(1)),false);c=f.render();
    assert.equal(c.view.stateId,'main:0:0');assert.equal(c.volume,accepted);assert.match(c.error,/recorded frame unavailable/);assert.equal(c.pending,false);
    await c.run('seek',{position:0});c=f.admit();assert.equal(c.view.stateId,'editable:seek');assert.equal(c.error,null);assert.equal(f.calls.at(-1).expectedStateId,'editable');
  }finally{f.harness.close();}
});
test('renderer rejection stops lookahead and retains the last accepted anatomy',async()=>{
  const f=replayFixture();try{
    let c=await f.open();assert(c.startPlayback());await c.runPlayback(selection(0));await f.flush();c=f.render();
    c.interaction.onDisplayError(c.interaction.view,'mesh failed');c=f.render();assert.equal(c.view.stateId,'editable');assert.match(c.error,/mesh failed/);
    assert.equal(await c.runPlayback(selection(1)),false);assert.equal(await c.run('step',{action:{operation:'apply'}}),false);
    assert.equal(f.calls.filter(q=>q.command==='step').length,0);
  }finally{f.harness.close();}
});
test('pause during the demanded transport releases pending state and never installs an unadmitted receipt',async()=>{
  const pending=deferred(),f=replayFixture(q=>q.command==='seekFrame'&&q.frameIndex===0?pending.promise:undefined);
  try{
    let c=await f.open();const accepted=c.volume;assert(c.startPlayback());const demanded=c.runPlayback(selection(0));await f.flush();c=f.render();assert(c.pending);
    c.stopPlayback();pending.resolve(f.frame(selection(0)));assert.equal(await demanded,false);c=f.render();
    assert.equal(c.view.stateId,'editable');assert.equal(c.interaction.view.stateId,'editable');assert.equal(c.volume,accepted);assert.equal(c.pending,false);assert.equal(c.error,null);
    assert.equal(f.calls.filter(q=>q.command==='seekFrame').length,1,'pause cannot schedule a successor');
  }finally{f.harness.close();}
});
test('pause or restart between transport resolution and hook admission invalidates the receipt lease',async()=>{
  for(const restart of [false,true]){
    const pending=deferred(),f=replayFixture(q=>q.command==='seekFrame'&&q.frameIndex===0?pending.promise:undefined);
    try{
      let c=await f.open();assert(c.startPlayback());const demanded=c.runPlayback(selection(0));await f.flush();
      pending.resolve(f.frame(selection(0)));for(let i=0;i<5;i++)await Promise.resolve();
      c=f.render();assert.equal(c.interaction.view.stateId,'editable');c.stopPlayback();if(restart)assert(c.startPlayback());
      assert.equal(await demanded,false);c=f.render();assert.equal(c.interaction.view.stateId,'editable');assert.equal(c.view.stateId,'editable');assert.equal(c.error,null);
      assert.equal(f.calls.filter(q=>q.command==='seekFrame').length,1);
    }finally{f.harness.close();}
  }
});
test('manual barrier at settled transport prevents replay admission and preserves manual pending state',async()=>{
  const pending=deferred(),manual=deferred(),f=replayFixture(q=>q.command==='seekFrame'?pending.promise:q.command==='seek'?manual.promise:undefined);
  try{
    let c=await f.open();assert(c.startPlayback());const demanded=c.runPlayback(selection(0));await f.flush();
    pending.resolve(f.frame(selection(0)));for(let i=0;i<5;i++)await Promise.resolve();c=f.render();assert.equal(c.interaction.view.stateId,'editable');
    const barrier=c.run('seek',{position:0});assert.equal(await demanded,false);await f.flush();c=f.render();
    assert.equal(c.interaction.view.stateId,'editable');assert.equal(c.pending,true);assert.equal(c.simulationPending,true);assert.equal(c.pendingCommand,'seek');
    manual.resolve({...f.frame(selection(0)),playback:undefined,view:{...c.view,stateId:'manual'}});await barrier;c=f.admit();assert.equal(c.view.stateId,'manual');assert.equal(c.pending,false);
  }finally{f.harness.close();}
});
test('equal state IDs at microstep and finalization remain separate ordered recorded admissions',async()=>{
  const f=replayFixture(q=>{
    if(q.command!=='seekFrame')return;
    const r=f.frame(q),times=[0,.2,.4,.4],frameIndex=q.frameIndex;
    r.playback={...r.playback,frameCount:4,observedTimeSeconds:q.position*.4+times[frameIndex],actionElapsedSeconds:times[frameIndex],isActionBoundary:frameIndex===0||frameIndex===3,frames:times.map((t,i)=>({frameIndex:i,observedTimeSeconds:q.position*.4+t,kind:i===3?'action_boundary':'microstep'}))};
    r.view.metrics.elapsedSeconds=r.playback.observedTimeSeconds;
    if(frameIndex===3){r.view.stateId=`${q.branchId}:${q.position}:2`;r.view.removed=[0,1];}
    return r;
  });
  try{
    let c=await f.open();assert(c.startPlayback());const admitted=[];let previous;
    for(let i=0;i<4;i++){
      await c.runPlayback(selection(i));await f.flush();c=f.render();
      assert.deepEqual(c.interaction.recordedFrame,selection(i));
      if(i===3)assert.equal(c.volume,previous,'identical geometry may reuse the mesh while retaining observation identity');
      previous=c.volume;c=f.admit();admitted.push({stateId:c.view.stateId,frameIndex:c.result.playback.frameIndex});
    }
    assert.deepEqual(admitted,[{stateId:'main:0:0',frameIndex:0},{stateId:'main:0:1',frameIndex:1},{stateId:'main:0:2',frameIndex:2},{stateId:'main:0:2',frameIndex:3}]);
    assert.deepEqual(f.calls.filter(q=>q.command==='seekFrame').slice(0,4).map(q=>[q.position,q.frameIndex]),[[0,0],[0,1],[0,2],[0,3]]);
  }finally{f.harness.close();}
});
test('queued lookahead keeps its playback owner across a source render before passive invalidation',async()=>{
  for(const caseHash of ['case','reopened-case']){
    const f=replayFixture();try{
      let c=await f.open();assert(c.startPlayback());const demanded=c.runPlayback(selection(0));
      let queued=false;
      for(let tick=0;tick<20;tick++){
        await Promise.resolve();c=f.render();
        if(c.interaction.view?.stateId==='main:0:0'&&f.calls.filter(q=>q.command==='seekFrame').length===1){queued=true;break;}
      }
      assert(queued,'the hook has installed frame0 with its successor queued but not dispatched');
      const count=f.calls.length,reopened={...f.source,caseHash};
      c=f.harness.render(reopened,f.api,{passive:false});assert.equal(c.view,null);
      await f.flush();assert.equal(await demanded,true);
      assert.equal(f.calls.length,count,'queued old-source lookahead cannot dispatch against the new source before effects run');
      c.stopPlayback();assert.equal(c.startPlayback(),false,'an old accepted receipt cannot start a new-source lane before inspection');
      f.harness.flushEffects();await f.flush();
      assert.equal(f.calls.at(-1).command,'inspect');assert.equal(f.calls.at(-1).caseHash,caseHash);assert.equal(f.calls.at(-1).expectedStateId,undefined);
    }finally{f.harness.close();}
  }
});

function prefixFixture(intercept=()=>undefined,{stream=false,transform=response=>response}={}){
  let through=0;const prefixResponses=[];
  const f=replayFixture(q=>{
    const overridden=intercept(q);if(overridden!==undefined)return overridden;
    if(q.command==='inspectArchive'){
      const r=f.frame({...selection(0),branchId:q.branchId??'main'});return {...r,playback:null,timelineToken:undefined,position:10,verification:{readOnly:true,status:'unverified-import',causalReplayVerified:false,limitations:[]},view:{...r.view,stateId:'saved:endpoint'}};
    }
    if(q.command==='seekPrefix'||q.command==='seekPrefixFrame'){
      assert.deepEqual(Object.keys(q).filter(key=>!['caseHash','command','branchId','position',...(q.command==='seekPrefixFrame'?['frameIndex','prefixToken','prefixFieldTokens']:[])].includes(key)),[],'prefix requests cannot carry canonical identity or compact acknowledgments');
      let r,position,budgetStopped=false;
      if(q.command==='seekPrefix'){
        const actual=Math.min(q.position,through+4);through=Math.max(through,actual);position=actual;budgetStopped=actual<q.position;
        r=f.frame({...selection(0),branchId:q.branchId??'main'});r={...r,playback:null,view:{...r.view,stateId:`prefix:${actual}`,removed:Array.from({length:Math.min(actual,8)},(_,i)=>i)}};
      }else {r=f.frame(q);through=Math.max(through,q.position+1);position=q.position+Number(q.frameIndex===2);}
      let result={...r,timelineToken:undefined,position,requestedPosition:q.command==='seekPrefix'?q.position:q.position+1,budgetStopped,
        branches:r.branches.map(b=>({...b,steps:10})),timeline:Array.from({length:through},()=>({action:{operation:'inspect'},status:'accepted',reason:null,events:[]})),
        verification:{readOnly:true,status:'reconstructed-prefix',verifiedThrough:through,totalActions:10,archiveEndpointVerified:false,originSnapshotHash:'origin',actionLogHash:'actions',protectionContractHash:'protection',implementation:'current',prefixImplementation:'prefix-v1'}};
      if(stream&&q.command==='seekPrefixFrame'){
        const token=`prefix-replay.v1/owner-main/${through}`,hash=(q.frameIndex===2?'b':'a').repeat(64);
        result={...result,timelineToken:token,fullCase:null,viewFieldTokens:{flapComponents:hash},view:{...result.view,history:[],flapComponents:[{cells:[q.frameIndex===2?2:1],separable:q.frameIndex===2}]}};
        if(q.prefixToken===token){
          result={...result,timelineOmitted:true,view:{...result.view}};delete result.timeline;delete result.fullCase;
          for(const field of ['scene','catalog','protection'])delete result.view[field];
          if(q.prefixFieldTokens?.flapComponents===hash){delete result.view.flapComponents;result.viewFieldsOmitted=['flapComponents'];}
        }
      }
      result=transform(result,q);prefixResponses.push({request:q,response:result});return result;
    }
  },{initialArchivePreview:true});f.prefixResponses=prefixResponses;return f;
}
test('unverified endpoints enter prefix0 explicitly; prefix requests cannot enter editable or canonical cursor domains',async()=>{
  const f=prefixFixture();try{
    let c=await f.open();assert.equal(c.result.verification.status,'unverified-import');assert.equal(c.startPlayback(),false);
    for(const command of ['seek','seekFrame','seekPrefixFrame','step','plan','branch','fullCase']){
      const before=f.calls.length;assert.equal(await c.run(command,{...selection(1)}),false,command);assert.equal(f.calls.length,before);
    }
    assert.equal(await c.run('seekPrefix',{position:1,branchId:'main'}),false,'initial reconstruction must start at0');
    await c.run('seekPrefix',{position:0,branchId:'main',expectedStateId:'must-not-cross'});c=f.admit();assert.equal(c.view.stateId,'prefix:0');assert.equal(c.interaction.readOnly,true);
    for(const command of ['seek','seekFrame','selectBranch','step','preview','plan','branch','fullCase','reset','create']){
      const before=f.calls.length;assert.equal(await c.run(command,{...selection(0)}),false,command);assert.equal(f.calls.length,before);
    }
    c.startFullCase('primary');c=f.render();assert.equal(c.computingFullCase,null,'prefix cannot start computation through the direct controller');
    await c.run('inspect');c=f.admit();assert.equal(f.calls.at(-1).expectedStateId,undefined,'prefix/import views never replace the editable cursor');
    assert.equal(c.result.verification,undefined);const before=f.calls.length;assert.equal(await c.run('seekPrefix',{position:0}),false);assert.equal(f.calls.length,before,'canonical session cannot jump into prefix authority');
  }finally{f.harness.close();}
});
test('prefix automatic FIFO admits each full receipt once with no future volume before CPU acknowledgment',async()=>{
  const f=prefixFixture();try{
    let c=await f.open();await c.run('seekPrefix',{position:0,branchId:'main'});c=f.admit();assert(c.startPlayback());
    const admitted=[];
    for(let i=0;i<3;i++){
      assert(await c.runPlayback(selection(i)));await f.flush();c=f.render();assert.equal(c.view.stateId,i===0?'prefix:0':`main:0:${i-1}`);
      const activeVolume=c.volume;await f.flush();c=f.render();assert.equal(c.volume,activeVolume,'a retained full prefix lookahead cannot prepare future anatomy');
      if(i<2)assert.equal(await c.runPlayback(selection(i+1)),false,'the current anatomy must be acknowledged before its successor is demanded');
      c=f.admit();admitted.push(c.result.playback.frameIndex);assert.equal(c.result.verification.status,'reconstructed-prefix');assert.equal(c.interaction.readOnly,true);
    }
    assert.deepEqual(admitted,[0,1,2]);const frames=f.calls.filter(q=>q.command==='seekPrefixFrame');
    assert.deepEqual(frames.filter(q=>q.position===0).map(q=>q.frameIndex),[0,1,2],'each full prefix observation is fetched exactly once');assert(frames.every(q=>!Object.hasOwn(q,'expectedStateId')&&!Object.hasOwn(q,'timelineToken')));
    c.stopPlayback();await c.run('inspectArchive',{branchId:'other'});c=f.admit();assert.equal(c.result.branchId,'other');assert.equal(c.startPlayback(),false,'branch switch returns to imported authority');
  }finally{f.harness.close();}
});
test('budget-stopped prefix seek displays actual reached progress and preserves original archive size',async()=>{
  const f=prefixFixture();try{
    let c=await f.open();await c.run('seekPrefix',{position:0});c=f.admit();await c.run('seekPrefix',{position:9});c=f.admit();
    assert.equal(c.result.position,4);assert.equal(c.result.requestedPosition,9);assert.equal(c.result.budgetStopped,true);assert.equal(c.result.verification.verifiedThrough,4);assert.equal(c.result.verification.totalActions,10);assert.equal(c.result.branches[0].steps,10);assert.equal(c.result.timeline.length,4);
    await c.run('seekPrefix',{position:9});c=f.admit();assert.equal(c.result.position,8);assert.equal(c.result.verification.verifiedThrough,8);
  }finally{f.harness.close();}
});
test('prefix frame streams hydrate exact full-to-compact FIFO, changed fields and growing history generations only on demand',async()=>{
  const f=prefixFixture(undefined,{stream:true});try{
    let c=await f.open();await c.run('seekPrefix',{position:0});c=f.admit();assert(c.startPlayback());
    await c.runPlayback(selection(0));await f.flush();c=f.render();const active=c.volume;
    const first=f.prefixResponses.find(row=>row.request.command==='seekPrefixFrame'&&row.request.frameIndex===0).response;
    const second=f.prefixResponses.find(row=>row.request.frameIndex===1).response;
    assert.equal(second.timelineOmitted,true);assert.equal(Object.hasOwn(second,'timeline'),false);assert.equal(Object.hasOwn(second.view,'scene'),false);assert.equal(Object.hasOwn(second.view,'flapComponents'),false);
    assert.equal(c.volume,active);assert.equal(c.view.stateId,'prefix:0');c=f.admit();
    assert(await c.runPlayback(selection(1)));await f.flush();c=f.admit();
    assert.equal(c.view.scene,first.view.scene);assert.equal(c.view.flapComponents,first.view.flapComponents);assert.deepEqual(c.view.history,[]);assert.equal(c.result.timeline.length,1);
    assert(await c.runPlayback(selection(2)));await f.flush();c=f.admit();
    assert.deepEqual(c.view.flapComponents,[{cells:[2],separable:true}]);assert.deepEqual(c.view.history,[]);
    const final=f.prefixResponses.find(row=>row.request.position===0&&row.request.frameIndex===2).response;
    assert.equal(final.timelineOmitted,true);assert.equal(final.viewFieldsOmitted,undefined,'changed field travels in full despite stable timeline token');
    assert(await c.runPlayback(selection(0,1)));await f.flush();c=f.admit();assert.equal(c.result.timeline.length,2);assert.equal(c.result.verification.verifiedThrough,2);assert.deepEqual(c.view.history,[]);
    const next=f.prefixResponses.find(row=>row.request.position===1&&row.request.frameIndex===0).response;
    assert.notEqual(next.timelineToken,first.timelineToken);assert.equal(next.timelineOmitted,undefined);assert(Array.isArray(next.timeline));assert(Object.hasOwn(next.view,'scene'));
    const frames=f.calls.filter(q=>q.command==='seekPrefixFrame');assert.deepEqual(frames.map(q=>[q.position,q.frameIndex]),[[0,0],[0,1],[0,2],[1,0],[1,1]]);
    assert(frames.every(q=>!Object.hasOwn(q,'expectedStateId')&&!Object.hasOwn(q,'timelineToken')&&!Object.hasOwn(q,'viewFieldTokens')));
    assert.equal(frames[1].prefixToken,first.timelineToken);assert.deepEqual(frames[1].prefixFieldTokens,{flapComponents:'a'.repeat(64)});
    c.stopPlayback();assert.equal(await c.run('seekPrefixFrame',selection(1,1)),true);c=f.admit();
    assert.equal(f.calls.at(-1).prefixToken,next.timelineToken);assert.equal(f.prefixResponses.at(-1).response.timelineOmitted,true);assert.deepEqual(c.view.history,[]);assert.equal(c.result.timeline.length,2);
    await c.run('seekPrefix',{position:1,prefixToken:'supplied-token',prefixFieldTokens:{flapComponents:'f'.repeat(64)}});c=f.admit();
    assert.deepEqual(Object.keys(f.calls.at(-1)),['caseHash','command','branchId','position'],'prefix boundary never sends stream tokens');
    await c.run('seekPrefixFrame',selection(1));c=f.admit();assert.equal(f.calls.at(-1).prefixToken,undefined,'boundary invalidates prior stream acknowledgment');assert.equal(c.result.timeline.length,2);
  }finally{f.harness.close();}
});
test('compact prefix provenance or source substitution fails before future anatomy is admitted',async()=>{
  for(const substitute of [r=>({...r,caseHash:'other'}),r=>({...r,verification:{...r.verification,originSnapshotHash:'other'}}),r=>({...r,timelineToken:undefined}),r=>({...r,verification:undefined})]){
    const f=prefixFixture(undefined,{stream:true,transform:(r,q)=>q.command==='seekPrefixFrame'&&q.frameIndex===1?substitute(r):r});try{
      let c=await f.open();await c.run('seekPrefix',{position:0});c=f.admit();assert(c.startPlayback());await c.runPlayback(selection(0));await f.flush();c=f.admit();
      assert.equal(await c.runPlayback(selection(1)),false);c=f.render();assert.equal(c.view.stateId,'main:0:0');assert.equal(c.interaction.view,c.view);assert(c.error);
    }finally{f.harness.close();}
  }
});
test('compact prefix hydration rejects missing generations, stale fields and incomplete reconstructed timeline before display',async()=>{
  for(const substitute of [r=>({...r,timelineToken:'prefix-replay.v1/missing'}),r=>({...r,viewFieldTokens:{flapComponents:'f'.repeat(64)}}),r=>({...r,verification:{...r.verification,verifiedThrough:2}})]){
    let corrupt=true;
    const f=prefixFixture(undefined,{stream:true,transform:(r,q)=>corrupt&&q.command==='seekPrefixFrame'&&q.frameIndex===1?substitute(r):r});try{
      let c=await f.open();await c.run('seekPrefix',{position:0});c=f.admit();assert(c.startPlayback());await c.runPlayback(selection(0));await f.flush();c=f.admit();
      assert.equal(await c.runPlayback(selection(1)),false);c=f.render();assert.equal(c.view.stateId,'main:0:0');assert(c.error);
      corrupt=false;assert.equal(await c.run('seekPrefixFrame',selection(1)),true);c=f.admit();
      assert.equal(f.calls.at(-1).prefixToken,undefined,'failed hydration clears resident acknowledgments for a complete refresh');assert.equal(f.calls.at(-1).prefixFieldTokens,undefined);assert.equal(c.view.stateId,'main:0:1');assert.deepEqual(c.view.history,[]);
    }finally{f.harness.close();}
  }
});
test('prefix owner source changes and authority substitution fail closed before future display',async()=>{
  const pending=deferred();let blocked=false;
  const f=prefixFixture(q=>q.command==='seekPrefixFrame'&&q.frameIndex===1&&blocked?pending.promise:undefined);
  try{
    let c=await f.open();await c.run('seekPrefix',{position:0});c=f.admit();assert(c.startPlayback());blocked=true;await c.runPlayback(selection(0));await f.flush();c=f.admit();
    const reopened={...f.source};c=f.harness.render(reopened,f.api,{initialArchivePreview:true});assert.equal(c.view,null);
    pending.resolve({...f.frame(selection(1)),verification:undefined});await f.flush();c=f.harness.render(reopened,f.api,{initialArchivePreview:true});
    if(c.interaction.view)c.interaction.onDisplayReady(c.interaction.view);c=f.harness.render(reopened,f.api,{initialArchivePreview:true});assert.equal(c.result.verification.status,'unverified-import');assert.equal(c.error,null);assert.equal(f.calls.at(-1).command,'inspectArchive');assert.equal(f.calls.at(-1).expectedStateId,undefined);
  }finally{f.harness.close();}
});
test('a prefix frame response cannot substitute canonical authority or pollute editable state',async()=>{
  const pending=deferred();let block=false;const f=prefixFixture(q=>q.command==='seekPrefixFrame'&&q.frameIndex===1&&block?pending.promise:undefined);
  try{
    let c=await f.open();await c.run('seekPrefix',{position:0});c=f.admit();assert(c.startPlayback());block=true;await c.runPlayback(selection(0));await f.flush();c=f.admit();
    pending.resolve(f.frame(selection(1)));await f.flush();assert.equal(await c.runPlayback(selection(1)),false);c=f.render();assert.equal(c.view.stateId,'main:0:0');assert.match(c.error,/prefix authority/);
    await c.run('inspect');c=f.admit();assert.equal(f.calls.at(-1).expectedStateId,undefined);assert.equal(c.result.verification,undefined);
  }finally{f.harness.close();}
});
test('manual prefix commands cannot borrow stale authority after a source render before passive effects',async()=>{
  for(const caseHash of ['case','replacement-source']){
    const f=prefixFixture();try{
      let c=await f.open();await c.run('seekPrefix',{position:0});c=f.admit();assert.equal(c.result.verification.status,'reconstructed-prefix');
      const replacement={...f.source,caseHash};c=f.harness.render(replacement,f.api,{initialArchivePreview:true,passive:false});assert.equal(c.view,null);
      const before=f.calls.length;
      for(const [command,extra] of [['seekPrefix',{position:4}],['seekPrefixFrame',selection(0)]]){
        assert.equal(await c.run(command,extra),false,command);assert.equal(f.calls.length,before,'no old-prefix cursor or editable identity may dispatch to the replacement source');
      }
      // Inspection remains legal before the replacement source has a bound view.
      assert.equal(await c.run('inspectArchive',{branchId:'main'}),true);assert.equal(f.calls.at(-1).command,'inspectArchive');assert.equal(f.calls.at(-1).caseHash,caseHash);
      c=f.harness.render(replacement,f.api,{initialArchivePreview:true,passive:false});
      assert.equal(c.result.verification.status,'reconstructed-prefix','old accepted prefix remains until the replacement receipt is displayed');assert.equal(c.view,null,'an unadmitted replacement receipt cannot expose stale same-hash anatomy');
      const received=f.calls.length;
      for(const [command,extra] of [['seekPrefix',{position:4}],['seekPrefixFrame',selection(0)]]){
        assert.equal(await c.run(command,extra),false,command);assert.equal(f.calls.length,received,'an unadmitted replacement receipt cannot bind old prefix authority');
      }
      assert.equal(await c.run('inspect'),true);assert.equal(f.calls.at(-1).command,'inspect');assert.equal(f.calls.at(-1).caseHash,caseHash);
      f.harness.flushEffects();await f.flush();assert.equal(f.calls.at(-1).command,'inspectArchive');assert.equal(f.calls.at(-1).expectedStateId,undefined);
    }finally{f.harness.close();}
  }
});
