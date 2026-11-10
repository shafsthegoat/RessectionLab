import {before,after,test} from 'node:test';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import React from 'react';
import {fixture} from './cranial-workflow-fixture.mjs';
let server,Workflow,Panel;
before(async()=>{server=await createServer({root:fileURLToPath(new URL('../',import.meta.url)),logLevel:'error',server:{middlewareMode:true,hmr:false,ws:false},appType:'custom'});({SurgicalWorkflow:Workflow}=await server.ssrLoadModule('/src/SurgicalWorkflow.tsx'));({CranialStrategiesPanel:Panel}=await server.ssrLoadModule('/src/CranialStrategiesPanel.tsx'));});
after(async()=>{await server?.close();});
function nodes(tree){if(!tree||typeof tree!=='object')return [];if(Array.isArray(tree))return tree.flatMap(nodes);return [tree,...nodes(tree.props?.children)];}
const find=(tree,p)=>nodes(tree).find(p);
function harness(Component){
 const slots=[],effects=[];let index=0,dirty=false;const internals=React.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
 const same=(a,b)=>a&&b&&a.length===b.length&&a.every((x,i)=>Object.is(x,b[i]));
 const dispatcher={useState(value){const i=index++;if(!(i in slots))slots[i]=typeof value==='function'?value():value;return [slots[i],next=>{slots[i]=typeof next==='function'?next(slots[i]):next;dirty=true;}];},useEffect(fn,deps){const i=index++;if(!slots[i]||!same(slots[i].deps,deps)){slots[i]={deps};effects.push(fn);}}};
 return {render(props){let result;do{dirty=false;index=0;const previous=internals.H;internals.H=dispatcher;try{result=Component(props);}finally{internals.H=previous;}for(const run of effects.splice(0))run();}while(dirty);return result;}};
}
const controller=r=>({result:r,view:r.view,pending:false,preview:null,clearPreview(){},run:async()=>true});
test('workflow mounts only cranial task even if legacy preparation appears available',()=>{
 const r=fixture(),h=harness(Workflow),tree=h.render({c:controller(r),stage:'plan',onStage(){},pause(){}});assert(find(tree,n=>n.type?.name==='CranialStrategiesPanel'));assert(!find(tree,n=>['PreparationPlanningPanel','StagedStrategiesPanel','InteractivePlanningPanel'].includes(n.type?.name)));
});
test('Generate dispatches complete horizon and Compare/Rehearse use actual native branch',async()=>{
 const r=fixture(),calls=[],events=[],c=controller(r);c.run=async(command,args)=>{calls.push({command,...args});return true;};const h=harness(Panel),props={c,pause(){},play:id=>events.push(['play',id]),stage:'plan',onCompared:()=>events.push('compare'),onInstalled:()=>events.push('rehearse')};let tree=h.render(props);
 let generate=find(tree,n=>n.type==='button'&&n.props.children==='Generate strategies');assert(generate.props.disabled);find(tree,n=>n.props?.['aria-label']==='Acknowledge cranial research context').props.onChange({target:{checked:true}});tree=h.render(props);generate=find(tree,n=>n.type==='button'&&n.props.children==='Generate strategies');assert.equal(generate.props.disabled,false);generate.props.onClick();await new Promise(setImmediate);assert.deepEqual(calls[0],{command:'plan',opponent:'anatomy-cranial-strategy',acknowledgeResearchUnknowns:true,planningHorizon:5,beamWidth:3});assert.equal(events[0],'compare');
 tree=h.render({...props,stage:'compare'});const cards=find(tree,n=>n.type?.name==='WorkflowAlternatives');assert.equal(cards.props.rows[0].nonTargetMm3,25);cards.props.onSelect('actual-branch');await new Promise(setImmediate);assert.deepEqual(calls[1],{command:'selectBranch',branchId:'actual-branch',position:12});assert(events.includes('rehearse'));find(tree,n=>n.type==='button'&&n.props.children?.[0]==='Replay ').props.onClick();assert.deepEqual(events.at(-1),['play','actual-branch']);
});
test('acknowledgement never enables UNKNOWN-blocked generation',()=>{
 const r=fixture();r.cranialPlanningAvailability.available=false;r.cranialPlanningAvailability.blockers=[{code:'UNKNOWN',message:'Unknown source cover overlaps'}];const h=harness(Panel),props={c:controller(r),pause(){},play(){},stage:'plan'};let tree=h.render(props);find(tree,n=>n.props?.['aria-label']==='Acknowledge cranial research context').props.onChange({target:{checked:true}});tree=h.render(props);assert(find(tree,n=>n.type==='button'&&n.props.children==='Generate strategies').props.disabled);assert(find(tree,n=>n.props?.role==='status'&&String(n.props.children).includes('UNKNOWN')));
});
