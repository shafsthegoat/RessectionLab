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
  return {render(source,api){let result;do{dirty=false;index=0;internals.H=dispatcher;try{result=useSurgical(source,api,true);}finally{internals.H=previous;}for(const effect of effects.splice(0))effect();}while(dirty);return result;},close(){for(const item of slots)item?.cleanup?.();internals.H=previous;}};
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
