import assert from 'node:assert/strict';
import {before,after,test} from 'node:test';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {fixture as stagedFixture,seal} from './staged-strategy-fixture.mjs';
let server,Workflow,Preparation,Alternatives,SurgicalPanel,Staged;
before(async()=>{server=await createServer({root:fileURLToPath(new URL('../',import.meta.url)),logLevel:'error',server:{middlewareMode:true,hmr:false,ws:false},appType:'custom'});({SurgicalWorkflow:Workflow}=await server.ssrLoadModule('/src/SurgicalWorkflow.tsx'));({PreparationPlanningPanel:Preparation}=await server.ssrLoadModule('/src/PreparationPlanningPanel.tsx'));({WorkflowAlternatives:Alternatives}=await server.ssrLoadModule('/src/WorkflowAlternatives.tsx'));({SurgicalPanel}=await server.ssrLoadModule('/src/SurgicalPanel.tsx'));({StagedStrategiesPanel:Staged}=await server.ssrLoadModule('/src/StagedStrategiesPanel.tsx'));});
after(async()=>{await server?.close();});
function nodes(tree){if(!tree||typeof tree!=='object')return [];if(Array.isArray(tree))return tree.flatMap(nodes);return [tree,...nodes(tree.props?.children)];}
const find=(tree,predicate)=>nodes(tree).find(predicate);
function harness(Component){
 const slots=[],effects=[];let index=0,dirty=false;const internals=React.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
 const same=(a,b)=>a&&b&&a.length===b.length&&a.every((x,i)=>Object.is(x,b[i]));
 const dispatcher={useState(value){const i=index++;if(!(i in slots))slots[i]=typeof value==='function'?value():value;return [slots[i],next=>{slots[i]=typeof next==='function'?next(slots[i]):next;dirty=true;}];},useRef(value){return slots[index++]??={current:value};},useEffect(fn,deps){const i=index++;if(!slots[i]||!same(slots[i].deps,deps)){const previous=slots[i];slots[i]={deps,cleanup:previous?.cleanup};effects.push(()=>{previous?.cleanup?.();slots[i].cleanup=fn();});}}};
 return {render(props){let result;do{dirty=false;index=0;const previous=internals.H;internals.H=dispatcher;try{result=Component(props);}finally{internals.H=previous;}for(const run of effects.splice(0))run();}while(dirty);return result;},close(){for(const item of slots)item?.cleanup?.();}};
}
function fixture(){
 const controls={sourceRay:{surfaceMm:[3,3,3],sourceTargetMm:[3,3,10]},target:{centerMm:[3,3,10],radiusMm:6},widthsMm:[0],maxNonTargetMm3:100,maxActions:128,maxSeconds:180,entryRadiusMm:.5,shortSeconds:.25,longSeconds:1,minimumTargetMm3:1,requiredEvidence:['brain','target'],encounter:null};
 const suffix={id:'short',feasible:true,taskProgressMet:true,withdrawal:'complete',summary:{additionalTargetMm3:1,additionalNonTargetMm3:0,elapsedSeconds:2,residualTargetMm3:9}};
 const row={id:'width-0',widthMm:0,reason:null,preparedAdmission:{},preparation:{schema:'resectionlab.source-exposure-preparation.v2',publicPlan:{decision:{operation:'PREPARE',selectedIndex:0},attempts:[{complete:true,failure:null,summary:{additionalTargetMm3:0,additionalNonTargetMm3:12,elapsedSeconds:7,withdrawn:true}}]}},comparison:{publicPlan:{decision:{operation:'COMPARE',selectedId:'short'},alternatives:[suffix],approaches:{},rejections:[]}}};
 const report={schema:'resectionlab.source-preparation-comparison.v1',reportSeal:'seal',sourceSceneId:'scene',startStateId:'state',sourceBranchId:'analysis',sourcePosition:0,prefixActions:0,current:true,revision:1,controls,rows:[row],policies:{completePlanAware:{selectedPreparationId:'width-0'},corridorFirst:{selectedPreparationId:'width-0'}}};
 const view={stateId:'state',scene:{sceneId:'scene',shape:[24,24,24],affine:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],regions:[{target:true,name:'Supplied target'}]}};
 const result={caseHash:'case',branchId:'main',position:0,branches:[{id:'main',steps:0}],view,preparationPlanningAvailability:{available:true,defaults:controls},preparationPlanning:report};
 const calls=[],c={result,view,pending:false,preview:null,approachPreview:null,clearPreview(){c.preview=null;c.approachPreview=null;},setPlanningOverlay(){},async run(command,input){calls.push({command,...input});return true;}};
 return {c,calls,controls,report,row};
}
test('stage navigation exposes one planning owner and preserves its component identity across Plan and Compare',()=>{
 const f=fixture(),events=[],h=harness(Workflow),props={c:f.c,stage:'plan',onStage:value=>events.push(value),pause:()=>events.push('pause')};
 try{let tree=h.render(props);const draft=find(tree,node=>node.type?.name==='PreparationPlanningPanel');assert(draft);assert.equal(nodes(tree).filter(node=>['PreparationPlanningPanel','TwoToolStrategyPanel','InteractivePlanningPanel'].includes(node.type?.name)).length,1);
 const compare=find(tree,node=>node.type==='button'&&node.props.children?.at?.(-1)==='Compare');compare.props.onClick();assert.deepEqual(events,['pause','compare']);
 tree=h.render({...props,stage:'compare'});const next=find(tree,node=>node.type?.name==='PreparationPlanningPanel');assert.equal(next.type,draft.type);assert.equal(next.key,draft.key);assert.equal(next.props.stage,'compare');assert.equal(next.props.c,draft.props.c);
 }finally{h.close();}
});
test('an acknowledged preparation draft survives stage changes; editing and reverting requires regeneration',()=>{
 const f=fixture(),props={c:f.c,stage:'plan',pause(){},onCompared(){}};
 const shell=Preparation(props),element=find(shell,node=>node.type?.name==='PreparationDraft'),h=harness(element.type);
 try{let tree=h.render(element.props);find(tree,node=>node.props?.['aria-label']==='Acknowledge preparation research scope').props.onChange({target:{checked:true}});
 tree=h.render({...element.props,stage:'compare'});assert.equal(find(tree,node=>node.props?.['aria-label']==='Acknowledge preparation research scope').props.checked,true);assert(find(tree,node=>node.type?.name==='WorkflowAlternatives'));
 const region=find(tree,node=>node.type?.name==='RegionEditor');region.props.onChange({...f.controls.target,radiusMm:5});tree=h.render({...element.props,stage:'compare'});assert.equal(find(tree,node=>node.type?.name==='WorkflowAlternatives'),undefined);assert(nodes(tree).some(node=>node.props?.role==='status'&&String(node.props.children).includes('No current comparison')));
 find(tree,node=>node.type?.name==='RegionEditor').props.onChange(f.controls.target);tree=h.render({...element.props,stage:'compare'});assert.equal(find(tree,node=>node.type?.name==='WorkflowAlternatives'),undefined,'edit/revert must not revive accepted report');
 }finally{h.close();}
});
test('complete native preview installation forwards the bound seal and advances only on successful apply',async()=>{
 const f=fixture(),installed=[];const shell=Preparation({c:f.c,stage:'compare',pause(){},onInstalled:()=>installed.push('rehearse')});const element=find(shell,node=>node.type?.name==='PreparationDraft'),h=harness(element.type);
 try{let tree=h.render(element.props);find(tree,node=>node.props?.['aria-label']==='Acknowledge preparation research scope').props.onChange({target:{checked:true}});tree=h.render(element.props);
 const phase=find(tree,node=>node.props?.['aria-label']==='Preparation checkpoint phase');assert.equal(phase.props.value,'withdraw');
 const preview={stateId:'preview'};f.c.preview=preview;f.c.approachPreview={kind:'preparation',phaseId:'withdraw',candidateId:'width-0',reportSeal:'seal',view:preview};tree=h.render(element.props);
 const button=find(tree,node=>node.type==='button'&&node.props.children==='Install complete plan branch');assert.equal(button.props.disabled,false);button.props.onClick();await new Promise(resolve=>setImmediate(resolve));assert.deepEqual(f.calls.at(-1),{command:'applyPreparation',candidateId:'width-0',expectedPlanSeal:'seal',phase:'withdraw'});assert.deepEqual(installed,['rehearse']);f.c.run=async()=>false;button.props.onClick();await new Promise(resolve=>setImmediate(resolve));assert.deepEqual(installed,['rehearse'],'failed installation stays in Compare');
 }finally{h.close();}
});
test('cards render unknown fractions, recorded residual limits and STOP without clinical success claims',()=>{
 const html=renderToStaticMarkup(React.createElement(Alternatives,{rows:[{id:'stop',label:'STOP',status:'STOP · retains current state',removedMm3:0,residualMm3:10,fractionRemoved:null,nonTargetMm3:0,seconds:0,reason:'STOP does not establish completed treatment.',unknowns:'Functional outcomes unassessed.',residualLocation:'Residual location not recorded.',recommended:true}],onSelect(){}}));
 assert.match(html,/Target coverage/);assert.match(html,/Fraction of the original source target removed/);assert.match(html,/Unassessed/);assert.match(html,/Residual location not recorded/);assert.match(html,/does not establish completed treatment/);assert.doesNotMatch(html,/clinical success|0.0%/);
});
test('Case surfaces missing mapped anatomy alongside critical-structure uncertainty',()=>{
 const f=fixture();f.c.view.scene.environment={anatomy:{brain:'supplied',skull:'missing',vessels:'partial'}};f.c.view.protection={configured:false,unknownStructures:['brainstem coverage unassessed'],missingRequired:[]};
 const h=harness(Workflow);try{const tree=h.render({c:f.c,stage:'case',onStage(){},pause(){}}),section=find(tree,node=>node.props?.['aria-label']==='Case anatomy and evidence');const values=nodes(section).filter(node=>node.type==='dd').map(node=>String(node.props.children));assert(values.some(value=>value.includes('skull: missing')&&value.includes('vessels: partial')&&value.includes('brainstem coverage unassessed')));assert(values.includes('Supplied target'));}finally{h.close();}
});

test('advanced staged playback and inspection expose the existing rehearsal controls',async()=>{
 const f=fixture();Object.assign(f.c.view.scene,{resolutionMm:[1,1,1]});Object.assign(f.c.view,{catalog:[],slots:{},metrics:{elapsedSeconds:0},history:[]});Object.assign(f.c,{slot:'left',pose:{position:[0,0,0],quaternion:[0,0,0,1]},parameters:{},path:[],setPlaybackDisplay(){},stopPlayback(){},pausePlayback(){},pauseFullCase(){},canStartFullCase(){return false;}});
 const h=harness(SurgicalPanel),props={c:f.c,onFixture(){},available:true};try{let tree=h.render(props);assert.equal(find(tree,node=>node.type?.name==='SurgicalWorkflow').props.stage,'case');
 find(tree,node=>node.type?.name==='StagedStrategiesPanel').props.play('saved-staged-branch');await new Promise(resolve=>setImmediate(resolve));tree=h.render(props);assert.equal(find(tree,node=>node.type?.name==='SurgicalWorkflow').props.stage,'rehearse');assert.equal(find(tree,node=>node.props?.['aria-label']==='Rehearse and edit branch').props.hidden,false);
 find(tree,node=>node.type?.name==='SurgicalWorkflow').props.onStage('case');tree=h.render(props);find(tree,node=>node.type?.name==='JointStrategiesPanel').props.pause();tree=h.render(props);assert.equal(find(tree,node=>node.type?.name==='SurgicalWorkflow').props.stage,'rehearse');
 }finally{h.close();}
});

test('eligible paired source uses the single primary staged draft, preserves edits across stages and opens recorded branches',async()=>{
 const f=stagedFixture();f.response.view.scene.regions=[];delete f.artifact.stagedSummary;const calls=[],events=[],c={result:f.response,view:f.response.view,pending:false};c.run=async(command,args)=>{calls.push({command,...args});if(command==='plan'){f.response.stagedPlanning={...f.summary,controllerMode:'bounded-sequence-lookahead',sequenceSearch:{horizon:3,beamWidth:3,maxExpansionsPerSearch:24,baseline:'Immediate baseline',preference:'Native constraints then Pareto outcomes'}};}return true;};
 const owner=harness(Workflow),props={c,artifact:f.artifact,stage:'plan',onStage(){},pause(){}};const tree=owner.render(props);assert(find(tree,node=>node.type?.name==='StagedStrategiesPanel'));assert.equal(nodes(tree).filter(node=>['PreparationPlanningPanel','TwoToolStrategyPanel','InteractivePlanningPanel','StagedStrategiesPanel'].includes(node.type?.name)).length,1);owner.close();
 const h=harness(Staged),draft={c,artifact:f.artifact,stage:'plan',pause(){},play(){},onCompared:()=>events.push('compare'),onInstalled:()=>events.push('rehearse')};
 try{let panel=h.render(draft);find(panel,node=>node.props?.['aria-label']==='Staged planning depth').props.onChange({target:{value:'3'}});find(panel,node=>node.props?.['aria-label']==='Acknowledge staged source assumptions').props.onChange({target:{checked:true}});panel=h.render({...draft,stage:'compare'});assert(nodes(panel).some(node=>node.props?.role==='status'&&String(node.props.children).includes('No current staged comparison')));panel=h.render(draft);assert.equal(find(panel,node=>node.props?.['aria-label']==='Staged planning depth').props.value,3);
 find(panel,node=>node.type==='button'&&node.props.children==='Generate staged source strategies').props.onClick();await new Promise(resolve=>setImmediate(resolve));assert.equal(calls[0].planningHorizon,3);assert.equal(calls[0].beamWidth,3);assert.deepEqual(events,['compare']);panel=h.render({...draft,stage:'compare'});const cards=find(panel,node=>node.type?.name==='WorkflowAlternatives');assert(cards);assert.equal(cards.props.rows[0].nonTargetMm3,80);assert.equal(cards.props.rows[0].fractionRemoved,null);cards.props.onSelect(f.alternative.branchId);await new Promise(resolve=>setImmediate(resolve));assert.deepEqual(calls.at(-1),{command:'selectBranch',branchId:f.alternative.branchId,position:5});assert.deepEqual(events,['compare','rehearse']);
 panel=h.render(draft);assert(nodes(panel).some(node=>node.props?.role==='status'&&String(node.props.children).includes('current branch edits do not recalculate')));f.response.branchId='edited-continuation';f.response.position=1;f.response.view.stateId=seal('d');panel=h.render({...draft,stage:'compare'});assert(nodes(panel).some(node=>node.props?.role==='status'&&String(node.props.children).includes('Edits have not been recomputed')));f.response.view.scene.sceneId=seal('b');panel=h.render({...draft,stage:'compare'});assert(!find(panel,node=>node.type?.name==='WorkflowAlternatives'),'anatomy/source edits cannot keep the recorded comparison eligible');
 }finally{h.close();}
});
