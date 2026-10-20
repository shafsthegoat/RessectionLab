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
