import test,{before,after} from 'node:test';
import assert from 'node:assert/strict';
import * as React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {createServer} from 'vite';
import {mkdtemp,mkdir,rm} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
let server,cache,Panel,FullCase,Outcome,inspectionNotice;
before(async()=>{
  const output=fileURLToPath(new URL('../../testcases/outputs/rehearse-disclosure-20261030/',import.meta.url));await mkdir(output,{recursive:true});cache=await mkdtemp(path.join(output,'test-vite-'));
  server=await createServer({root:fileURLToPath(new URL('../',import.meta.url)),cacheDir:cache,logLevel:'error',optimizeDeps:{noDiscovery:true,include:[]},server:{middlewareMode:true,hmr:false,ws:false},appType:'custom'});
  ({SurgicalPanel:Panel}=await server.ssrLoadModule('/src/SurgicalPanel.tsx'));
  ({FullCaseRehearsalPanel:FullCase}=await server.ssrLoadModule('/src/FullCaseRehearsalPanel.tsx'));
  ({SurgicalOutcomeSummary:Outcome}=await server.ssrLoadModule('/src/SurgicalOutcomeSummary.tsx'));
  ({inspectionNotice}=await server.ssrLoadModule('/src/SurgeonViewControls.tsx'));
});
after(async()=>{await server?.close();if(cache)await rm(cache,{recursive:true,force:true});});
function* walk(node,parents=[]){if(Array.isArray(node)){for(const child of node)yield*walk(child,parents);}else if(node?.props){yield {node,parents};yield*walk(node.props.children,[...parents,node]);}}
const nodes=tree=>[...walk(tree)].map(x=>x.node);
const text=node=>Array.isArray(node)?node.map(text).join(''):node?.props?text(node.props.children):typeof node==='string'||typeof node==='number'?String(node):'';
const find=(tree,type,label)=>nodes(tree).find(n=>n.type===type&&(n.props['aria-label']===label||text(n)===label));
function controller(){
  const calls=[],slot={position:[0,0,-5],quaternion:[0,0,0,1],parameters:{},instrument:'cusa',inserted:false,active:false,grip:[]};
  const view={stateId:'accepted',scene:{sceneId:'scene',shape:[2,2,2],resolutionMm:[1,1,1],affine:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],regions:[],approximations:[],environment:{target:{initialVolumeMm3:10}}},catalog:[{id:'cusa',name:'CUSA / ultrasonic aspirator',materials:[],radiusMm:1.5}],slots:{left:slot,right:{...slot}},metrics:{elapsedSeconds:0,targetRemovedMm3:0,nonTargetRemovedMm3:0},removed:[],moved:[],cuts:[],contact:[],thermal:{},fluid:{},history:[]};
  const result={caseHash:'case',branchId:'main',branches:[{id:'main',steps:0}],position:0,view,outcome:null};
  const c={view,result,calls,preview:null,slot:'left',pose:slot,path:[],parameters:{},pending:false,mode:'navigate',playbackVersion:0,
    fullCaseReadiness:{status:'ready',availableActions:['primary','vascular'],blockers:[],recovery:null,protectionContractId:'contract'},canStartFullCase:()=>true,
    run:async(command,extra)=>{calls.push({command,...extra});return true;},cancel:()=>calls.push('cancel'),pauseFullCase:()=>calls.push('pause'),stopPlayback(){},startPlayback:()=>true,setMode(){},setPose(){},selectSlot(){},setParameters(){},clearPreview(){calls.push('clear-preview');},action:()=>({}),create(){}};
  return c;
}
function render(c){
  const internals=React.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,previous=internals.H;
  internals.H={useRef:value=>({current:value}),useState:value=>[typeof value==='function'?value():value,()=>{}],useEffect:()=>{}};
  try{return Panel({c,available:true,onFixture(){},inspectionNotice:'Inspection only · cutaway'});}finally{internals.H=previous;}
}
test('primary/manual/replay controls precede mounted advanced workflows and accounting',()=>{
  const c=controller(),tree=render(c),all=nodes(tree);
  const manual=find(tree,'section','Manual instrument actions'),replay=find(tree,'section','Replay controls');
  assert(all.indexOf(manual)<all.indexOf(replay));
  const expected=['RescueLearningPanel','ConsequenceComparisonPanel','LearnedExecutionPanel','LearnedSectorExecutionPanel','PatientCusaLearningPanel','FrozenCusaExecutionPanel','JointStrategiesPanel','RetractionStudyPanel','StagedStrategiesPanel','PrecisionContinuationPanel','ResearchPlanningPanel','SectorStrategiesPanel'];
  for(const name of expected){const row=[...walk(tree)].find(({node})=>node.type?.name===name);assert(row,name);assert(all.indexOf(row.node)>all.indexOf(replay),name);assert(row.parents.some(p=>p.type==='details'&&text(p.props.children[0])==='Advanced research workflows'),name+' stays mounted in a native disclosure');}
  for(const label of ['Pose, stroke and tool settings','Advanced research workflows','Environment and source setup']){
    const detail=nodes(tree).find(n=>n.type==='details'&&text(n.props.children[0])===label);assert(detail,label);assert.equal(detail.props.open,undefined);assert.equal(detail.props.onToggle,undefined,'toggle cannot dispatch or unmount controls');assert(detail.props.children.length>1);
  }
  assert.equal(c.calls.length,0);assert.equal(nodes(tree).filter(n=>n.type?.name==='SurgicalOutcomeSummary').length,1);
});
test('preview, read-only, inspection and partial/refused states remain explicit without opening details',()=>{
  const c=controller();c.preview={...c.view,stateId:'preview',metrics:{...c.view.metrics,targetRemovedMm3:1}};c.previewOutcome={status:'accepted',reason:null,events:[]};
  let tree=render(c);assert(text(find(tree,'section','Current rehearsal state')).includes('Preview · uncommitted'));assert(text(tree).includes('Inspection only · cutaway'));assert.equal(find(tree,'button','Apply').props.disabled,false);find(tree,'button','Cancel preview').props.onClick();assert.deepEqual(c.calls,['cancel']);
  c.preview=null;c.result.playback={readOnly:true,branchId:'main',position:0,actionIndex:0,frameIndex:0,frameCount:2,observedTimeSeconds:0,actionElapsedSeconds:0,actionDurationSeconds:0,frames:[{frameIndex:0,observedTimeSeconds:0},{frameIndex:1,observedTimeSeconds:.2}]};
  tree=render(c);assert(text(find(tree,'section','Current rehearsal state')).includes('Read-only replay'));assert.equal(find(tree,'button','Apply').props.disabled,true);assert(find(tree,'button','Return to recorded state 0'));
  delete c.result.playback;c.result.outcome={status:'rejected',reason:'WITHDRAW_BEFORE_EXCHANGE',events:[]};c.error='rejected: WITHDRAW_BEFORE_EXCHANGE';tree=render(c);assert(!nodes(tree).some(n=>n.props.role==='alert'),'the rendered outcome replaces its duplicate error');
  let result=nodes(tree).find(n=>n.type?.name==='ActionResult');assert(result);const displayed=result.type(result.props);assert(text(displayed).includes('Withdraw this instrument'));assert(find(displayed,'button','Set action to withdraw'));assert.equal(find(displayed,'details','Action events and technical reasonWITHDRAW_BEFORE_EXCHANGENo effect events were reported.').props.open,undefined);
  c.result.outcome={status:'partial',reason:'SHAFT_COLLISION',events:[],completedFraction:.25};tree=render(c);result=nodes(tree).find(n=>n.type?.name==='ActionResult');assert(text(result.type(result.props)).includes('25% completed · accepted partial state is displayed.'));
});
test('pending cancellation, running pause, stopped reason and actual readiness recovery stay exposed',()=>{
  const c=controller();c.pending=true;c.simulationPending=true;c.pendingCommand='step';let tree=render(c);assert.equal(find(tree,'button','Apply').props.disabled,true);assert.equal(find(tree,'button','Cancel').props.disabled,undefined);assert(text(tree).includes('Computing instrument action'));
  c.computingFullCase='primary';let full=FullCase({c,pause(){}});assert(find(full,'button','Pause computation'));assert.equal(find(full,'button','Pause computation').props.disabled,undefined);
  c.pending=false;c.computingFullCase=null;c.result.fullCase={scenario:'primary',status:'stopped',phases:[],stopReason:'ENGINE_MODEL_LIMIT'};full=FullCase({c,pause(){}});const stopped=[...walk(full)].find(({node})=>node.type==='p'&&text(node)==='ENGINE MODEL LIMIT');assert(stopped);assert(!stopped.parents.some(p=>p.type==='details'));c.simulationPending=false;c.result.fullCase.status='running';full=FullCase({c,pause(){}});assert(text(full).includes('computation paused'),'recorded running status must not imply active computation');
  c.fullCaseReadiness={status:'blocked',availableActions:['attachProtection'],blockers:[{code:'missing',message:'Matching evidence missing',details:[]}],recovery:{action:'attachProtection',label:'Attach matching protection'},protectionContractId:null};c.canStartFullCase=()=>false;full=FullCase({c,pause(){}});assert.equal(find(full,'button','Resume primary rehearsal').props.disabled,true);find(full,'button','Attach matching protection').props.onClick();assert.deepEqual(c.calls.at(-1),{command:'attachProtection'});
});
test('one outcome readout distinguishes unknown from measured zero and follows preview anatomy',()=>{
  const c=controller();let tree=Outcome({c});assert.equal(text(find(tree,'strong','0.00')),'0.00');let fluid=nodes(tree).find(n=>n.props['data-surgical-metric']==='fluidPresentMm3');assert.equal(text(fluid),'Unknown');
  c.preview={...c.view,stateId:'preview',metrics:{...c.view.metrics,targetRemovedMm3:1.5,fluidPresentMm3:0}};tree=Outcome({c});assert(text(tree).includes('Preview modeled outcome'));assert.equal(text(nodes(tree).find(n=>n.props['data-surgical-metric']==='targetRemovedMm3')),'1.50');assert.equal(text(nodes(tree).find(n=>n.props['data-surgical-metric']==='fluidPresentMm3')),'0.00');assert.equal(c.view.metrics.targetRemovedMm3,0);
});
test('inspection notice reflects active optical state and remains outside settings disclosures',()=>{
  assert.equal(inspectionNotice({}),null);assert.match(inspectionNotice({layerOpacity:{brain:.1}}),/transparent anatomy/);assert.match(inspectionNotice({clipping:{enabled:true}}),/cutaway/);
  assert.equal(inspectionNotice({operativeView:'surgeon',layerOpacity:{brain:.1},microscope:{}}),null,'opaque microscope ignores external alpha');assert.match(inspectionNotice({operativeView:'surgeon',microscope:{inspectionTransparency:true,inspectionCutaway:true}}),/transparent anatomy and cutaway/);
});

// Render every mounted child, including closed native disclosures. A component
// count alone missed the accepted-state cards hidden in environment setup.
test('entire rendered toolbox has one workflow and authoritative displayed-state measurements',()=>{
  const c=controller();
  const html=()=>renderToStaticMarkup(React.createElement(Panel,{c,available:true,onFixture(){}}));
  const check=(markup,target,fluid)=>{
    for(const [pattern,label] of [
      [/aria-label="Instrument action"/g,'manual action'],
      [/aria-label="Instrument interaction"/g,'pose controls'],
      [/aria-label="Search instruments"/g,'exchange'],
      [/Research Planning · compare approaches/g,'planning'],
      [/aria-label="Acknowledge research planning assumptions"/g,'planning acknowledgment'],
      [/Action timeline/g,'timeline'],
      [/>Undo<\/button>/g,'undo'],
      [/>Apply<\/button>/g,'apply'],
      [/>Preview<\/button>/g,'preview'],
    ])assert.equal([...markup.matchAll(pattern)].length,1,label);
    for(const [key,value] of [['targetRemovedMm3',target],['nonTargetRemovedMm3','0.00'],['fluidPresentMm3',fluid],['contactCellSeconds','Unknown']]){
      const readings=[...markup.matchAll(new RegExp(`data-surgical-metric="${key}"[^>]*>([^<]*)<`,'g'))].map(m=>m[1]);
      assert.deepEqual(readings,[value],key);
    }
    assert.equal([...markup.matchAll(/Target removed · mm³/g)].length,1);
    assert.equal([...markup.matchAll(/Other removed · mm³/g)].length,1);
    assert(!markup.includes('NaN'));
  };
  check(html(),'0.00','Unknown');
  c.preview={...c.view,stateId:'preview',metrics:{...c.view.metrics,targetRemovedMm3:1.5,fluidPresentMm3:0}};
  const preview=html();check(preview,'1.50','0.00');assert.match(preview,/Preview · uncommitted/);assert.match(preview,/Preview modeled outcome/);
  assert(!preview.includes('Target remaining: 10.00'),'no contradictory accepted-state remainder');
  c.preview={...c.preview,metrics:{nonTargetRemovedMm3:0}};
  check(html(),'Unknown','Unknown');
  assert.equal(c.view.metrics.targetRemovedMm3,0);assert.deepEqual(c.calls,[]);
});
