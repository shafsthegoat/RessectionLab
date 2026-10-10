import assert from'node:assert/strict';import{test}from'node:test';import fs from'node:fs';
const source=fs.readFileSync(new URL('../src/use-diagnostic-layer.ts',import.meta.url),'utf8');
const body=source.split('const load=useCallback(async()=>{')[1].split('},[selected,api,pending]);')[0];
const deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return{promise,resolve}};
function setup(){const selected={descriptor:{referenceCaseHash:'case',seriesId:'series'}},request=deferred(),hydration=deferred(),writes=[];
 const current={current:selected},generation={current:0},mounted={current:true};let hydrated=0;
 const scope={selected,api:{importDiagnosticLayer:()=>request.promise},pending:false,current,generation,mounted,
 setPending:x=>writes.push(['pending',x]),setError:x=>writes.push(['error',x]),setView:x=>writes.push(['view',x]),hydrateDiagnostic:()=>{hydrated++;return hydration.promise}};
 const load=Function(...Object.keys(scope),`return async()=>{${body}}`)(...Object.values(scope));return{selected,request,hydration,writes,current,mounted,load,get hydrated(){return hydrated}};}
test('source change before response refuses hydration and cannot install old diagnostic',async()=>{const x=setup(),job=x.load();x.current.current={};x.request.resolve({});await job;assert.equal(x.hydrated,0);assert(!x.writes.some(([k,v])=>k==='view'&&v));});
test('source change or unmount during hydration refuses late diagnostic installation',async()=>{for(const reason of ['source','unmount']){const x=setup(),job=x.load();x.request.resolve({});await Promise.resolve();assert.equal(x.hydrated,1);if(reason==='source')x.current.current={};else x.mounted.current=false;x.hydration.resolve({layer:'late'});await job;assert(!x.writes.some(([k,v])=>k==='view'&&v));}});
