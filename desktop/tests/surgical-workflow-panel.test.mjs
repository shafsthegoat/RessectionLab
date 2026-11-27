import assert from 'node:assert/strict';
import {before,after,test} from 'node:test';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {fixture as stagedFixture,seal} from './staged-strategy-fixture.mjs';
let server,Workflow,Preparation,Alternatives,SurgicalPanel,Staged,HoldControls;
before(async()=>{server=await createServer({root:fileURLToPath(new URL('../',import.meta.url)),logLevel:'error',server:{middlewareMode:true,hmr:false,ws:false},appType:'custom'});({SurgicalWorkflow:Workflow}=await server.ssrLoadModule('/src/SurgicalWorkflow.tsx'));({PreparationPlanningPanel:Preparation}=await server.ssrLoadModule('/src/PreparationPlanningPanel.tsx'));({WorkflowAlternatives:Alternatives}=await server.ssrLoadModule('/src/WorkflowAlternatives.tsx'));({SurgicalPanel}=await server.ssrLoadModule('/src/SurgicalPanel.tsx'));({PracticeHoldControls:HoldControls}=await server.ssrLoadModule('/src/PracticeHoldControls.tsx'));({StagedStrategiesPanel:Staged}=await server.ssrLoadModule('/src/StagedStrategiesPanel.tsx'));});
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

test('manual legacy CUSA controls omit unsupported duty while V2 retains it',()=>{
 for(const v2 of [false,true]){
  const f=fixture(),pose={position:[0,0,0],quaternion:[0,0,0,1]};
  Object.assign(f.c.view.scene,{resolutionMm:[1,1,1]});
  Object.assign(f.c.view,{catalog:[{id:'cusa',name:'CUSA',category:'Research',effect:'aspirate',geometry:'capsule',radiusMm:1.5,materials:['tumor']}],slots:{left:{instrument:'cusa',inserted:false,active:false,grip:[]}},metrics:{elapsedSeconds:0},history:[],physics:{cusaV3:{enabled:true,profile:{schema:'resectionlab.cusa-contact.v3'}},...(v2?{profile:{schema:'resectionlab.tissue-physics.v2'}}:{})}});
  Object.assign(f.c,{slot:'left',pose,parameters:{power:.8,duty:.5},path:[],setPlaybackDisplay(){},stopPlayback(){},pausePlayback(){},pauseFullCase(){},canStartFullCase(){return false;},action(operation,parameters,duration){return {slot:'left',operation,parameters,duration};}});
  const h=harness(SurgicalPanel),props={c:f.c,onFixture(){},available:true};
  try{
   let tree=h.render(props);assert.equal(!!find(tree,node=>node.props?.label==='CUSA duty fraction'),v2);
   find(tree,node=>node.props?.['aria-label']==='Instrument action').props.onChange({target:{value:'extract'}});tree=h.render(props);
   find(tree,node=>node.type==='button'&&node.props.children==='Apply').props.onClick();
   assert.deepEqual(f.calls.at(-1),{command:'step',action:{slot:'left',operation:'extract',parameters:{radiusMm:1.5,power:.8,...(v2?{duty:.5}:{})},duration:1}});
  }finally{h.close();}
 }
});

test('accepted cranial root display preserves Inspect destination and resets only for a new source',()=>{
 const f=fixture();Object.assign(f.c.view.scene,{resolutionMm:[1,1,1]});Object.assign(f.c.view,{catalog:[],slots:{},metrics:{elapsedSeconds:0},history:[]});Object.assign(f.c,{slot:'left',pose:{position:[0,0,0],quaternion:[0,0,0,1]},parameters:{},path:[],setPlaybackDisplay(){},stopPlayback(){},pausePlayback(){},pauseFullCase(){},canStartFullCase(){return false;}});
 f.c.result.cranialPlanningAvailability={sourceSceneId:'common-source'};
 const h=harness(SurgicalPanel),props={c:f.c,onFixture(){},available:true};
 try{
  let tree=h.render(props),workflow=find(tree,node=>node.type?.name==='SurgicalWorkflow');assert.equal(workflow.props.stage,'case');
  // Inspect resolves before the renderer commits the selected native root.
  workflow.props.onStage('rehearse');tree=h.render(props);
  f.c.view.scene={...f.c.view.scene,sceneId:'opening-root'};f.c.result.view=f.c.view;
  tree=h.render(props);workflow=find(tree,node=>node.type?.name==='SurgicalWorkflow');
  assert.equal(workflow.props.stage,'rehearse');assert.equal(workflow.key,'common-source');assert.equal(find(tree,node=>node.props?.['aria-label']==='Rehearse and edit branch').props.hidden,false);
  workflow.props.onStage('compare');tree=h.render(props);
  f.c.result.cranialPlanningAvailability={sourceSceneId:'changed-anatomy'};f.c.view.scene={...f.c.view.scene,sceneId:'changed-root'};
  tree=h.render(props);workflow=find(tree,node=>node.type?.name==='SurgicalWorkflow');assert.equal(workflow.props.stage,'case');assert.equal(workflow.key,'changed-anatomy');
  workflow.props.onStage('rehearse');tree=h.render(props);delete f.c.result.cranialPlanningAvailability;f.c.view.scene={...f.c.view.scene,sceneId:'legacy-new-scene'};
  tree=h.render(props);assert.equal(find(tree,node=>node.type?.name==='SurgicalWorkflow').props.stage,'case');
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

test('held practice submits only from a displayed receipt and ignores late replies after release or focus loss',async()=>{
  globalThis.window=new EventTarget();
  globalThis.document=new EventTarget();
  const calls=[],pose={position:[0,0,0],quaternion:[0,0,0,1]};
  const view={stateId:'start',slots:{left:{...pose,instrument:'cusa'}}};
  const origin={view,branchId:'child',caseHash:'case',position:3,practiceEpisode:{episodeId:'child',parent:{checkpointSeal:'seal'}}};
  const c={result:origin,view,slot:'left',pending:false,setPose(){},cancel(){},
    action(operation,parameters,duration){return {slot:this.slot,operation,parameters,duration};},
    runPracticeAction(action,from,signal){return new Promise(resolve=>calls.push({action,from,signal,resolve}));}};
  const h=harness(HoldControls),props={c,params:{},blocked:false,onRequest(){}};
  const key=(s,repeat=false)=>({key:s,repeat,preventDefault(){}});
  const tick=()=>new Promise(resolve=>setImmediate(resolve));
  const interval=()=>new Promise(resolve=>setTimeout(resolve,115));
  const button=(label)=>find(h.render({...props,blocked:c.pending}),node=>node.type==='button'&&node.props.children===label);
  try {
    button('Hold to advance').props.onKeyDown(key(' '));
    button('Hold to advance').props.onKeyDown(key(' ',true));
    assert.equal(calls.length,1);assert.equal(calls[0].from,origin);
    assert.equal(calls[0].action.duration,.1);assert.deepEqual(calls[0].action.path[0].position,[0,0,.5]);
    const accepted={...origin,position:4,view:{stateId:'first',slots:{left:{...view.slots.left,position:[0,0,.5]}}},
      outcome:{status:'accepted',reason:null,events:[],completedFraction:1}};
    c.pending=true;calls[0].resolve(accepted);await tick();h.render({...props,blocked:true});
    await interval();assert.equal(calls.length,1,'backend completion cannot race GPU acknowledgement');
    c.pending=false;c.result=accepted;c.view=accepted.view;h.render(props);await interval();
    assert.equal(calls.length,2);assert.equal(calls[1].from,accepted);assert.deepEqual(calls[1].action.path[0].position,[0,0,1]);
    button('Hold to advance').props.onKeyUp(key(' '));assert(calls[1].signal.aborted);
    button('Hold to apply').props.onKeyDown(key('Enter'));assert.equal(calls.length,3);
    calls[1].resolve({...accepted,position:5});await tick();h.render(props);
    await interval();assert.equal(calls.length,3,'released gesture cannot rearm its successor');
    window.dispatchEvent(new Event('blur'));assert(calls[2].signal.aborted);
    calls[2].resolve(null);await tick();assert.equal(h.render(props).props['data-practice-hold'],'idle');
    const pointer={pointerId:7,isPrimary:true,button:0,currentTarget:{focus(){},setPointerCapture(){}}};
    button('Hold to retract').props.onPointerDown(pointer);assert.equal(calls.length,4);
    button('Hold to retract').props.onLostPointerCapture(pointer);assert(calls[3].signal.aborted);
    calls[3].resolve(null);await tick();assert.equal(h.render(props).props['data-practice-hold'],'idle');
    button('Hold to apply').props.onPointerDown(pointer);
    const partial={...accepted,position:5,view:{...accepted.view,stateId:'partial'},outcome:{...accepted.outcome,status:'partial',completedFraction:.3}};
    c.result=partial;c.view=partial.view;calls[4].resolve(partial);await tick();h.render(props);
    await interval();assert.equal(calls.length,5);assert.equal(h.render(props).props['data-practice-hold'],'idle');
  } finally {h.close();delete globalThis.window;delete globalThis.document;}
});

test('boundary-99 practice task follows exact stage evidence, original target, complete collection and exterior exit',async()=>{
  const {CranialPracticeTask}=await server.ssrLoadModule('/src/CranialPracticeTask.tsx');
  const parent={position:99,caseHash:'case',sceneId:'scene',checkpointSeal:'sha256:b54fff661810b8798c961181d57ac699d4759a3e4465949426adcfeec70afbe3'};
  const membrane={retainedRimIntact:true,removedMembraneMm3:49,actualNativeExtractionEvents:1};
  const stage={stateId:'endpoint',sceneId:'scene',contractHash:'sha256:00e861fa8802c8dacd44ab13d0f55402ab94b335c55f26a3b5402da2aeaa820d',
    assessmentCurrent:true,bonePrefixVerified:true,boundaryCutComplete:true,patchExtracted:true,workEntrySupported:true,wholeExitComplete:true,membrane};
  const metrics={nonTargetRemovedMm3:334,targetRemovedMm3:27,debrisMm3:0,debrisRemovedMm3:236.0000000000002,fluidPresentMm3:0};
  const result={position:198,branchId:'practice',practiceEpisode:{episodeId:'practice',parent},caseHash:'case',practiceMetrics:{originalTargetVolumeMm3:27},
    view:{stateId:'endpoint',scene:{sceneId:'scene'},metrics,originalTargetResidualMm3:0,cranialStages:stage}};
  const html=(r=result)=>renderToStaticMarkup(React.createElement(CranialPracticeTask,{result:r}));
  assert.match(html(),/Modeled endpoint reached/);
  const atView=patch=>({...result,view:{...result.view,...patch}});
  assert.match(html(atView({cranialStages:{...stage,patchExtracted:false,patchHeld:true}})),/Transport the held patch/);
  assert.match(html(atView({originalTargetResidualMm3:1})),/Aspirate the remaining original target/);
  assert.match(html(atView({metrics:{...metrics,debrisRemovedMm3:230}})),/Collect the remaining kerf debris/);
  assert.match(html(atView({metrics:{...metrics,fluidPresentMm3:.5}})),/Clear the remaining modeled fluid/);
  assert.match(html(atView({cranialStages:{...stage,wholeExitComplete:false}})),/Release and withdraw/);
  assert.match(html(atView({metrics:{...metrics,nonTargetRemovedMm3:335}})),/Review retained rim and non-target/);
  assert.match(html(atView({cranialStages:{...stage,membrane:{...membrane,retainedRimIntact:false}}})),/Review retained rim and non-target/);
  assert.match(html(atView({cranialStages:{...stage,stateId:'previous'}})),/Current task evidence unavailable/);
  assert.match(html(atView({originalTargetResidualMm3:undefined})),/Current task evidence unavailable/);
  assert.equal(html({...result,practiceEpisode:{...result.practiceEpisode,parent:{...parent,checkpointSeal:'other'}}}),'');
  assert.equal(html({...result,position:98}),'');
  assert.equal(html({...result,playback:{}}),'');
});
