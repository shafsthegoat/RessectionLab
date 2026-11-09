'use strict';
// Exercise the real consolidated workflow offscreen. Never show or focus a window.
const {app,ipcMain,dialog,BrowserWindow}=require('electron');
// Install before filesystem preflight or assertions: Electron's default fatal
// handling can display a native dialog even when no BrowserWindow was created.
let hiddenFatalHandled=false;
const exitHiddenHarness=(error,kind)=>{
 if(!hiddenFatalHandled){hiddenFatalHandled=true;try{console.error('HIDDEN_WORKFLOW_FATAL '+JSON.stringify({kind,error:String(error),browserWindows:BrowserWindow.getAllWindows().length}));}catch{}}
 try{app.exit(1);}finally{process.exit(1);}
};
process.prependListener('uncaughtException',error=>exitHiddenHarness(error,'uncaughtException'));
process.prependListener('unhandledRejection',error=>exitHiddenHarness(error,'unhandledRejection'));
dialog.showErrorBox=(title,content)=>exitHiddenHarness(`${title}: ${content}`,'blocked-native-error-box');
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..'),base=path.join(root,'testcases');
const input=fs.realpathSync(process.env.WORKFLOW_CASE||path.join(base,'outputs/interactive-planning-20261031/bridge-v4-head-physics/interactive-planning.ressectionlab'));
const output=path.resolve(process.env.WORKFLOW_OUTPUT||path.join(base,'decision-system-v1/workflow-v1/hidden-v1'));
const externalSourceRoot='/Volumes/shafrir/old coding projects/RessectionLabData';
assert(input.startsWith(base+path.sep)||(process.env.WORKFLOW_RECORDED_ONLY==='1'&&input.startsWith(externalSourceRoot+path.sep)),'Use an ignored testcase or explicitly requested recorded external source');
assert(output.startsWith(base+path.sep),'Keep generated evidence in ignored testcases');
assert(!fs.existsSync(output),'Preserve previous evidence');fs.mkdirSync(output,{recursive:true});
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex'),original=sha(input),saved=path.join(output,'workflow-saved.ressectionlab');
const backendFiles=['surgical_staged_skills.py','surgical_sequence_search.py','surgical_staged_planning.py',
 'surgical_engine.py','surgical_scene.py','surgical_effects.py','surgical_tools.py','surgical_clock.py',
 'surgical_craniotomy.py','surgical_bone_transport.py','surgical_cranial_guards.py','surgical_craniotomy_skills.py',
 'surgical_milling.py','surgical_transition_meter.py','surgical_drilling_v2.py','surgical_physics_v2.py','surgical_task_admission.py'].map(name=>path.join(root,'src/resectionlab',name));
const backendSourceHashes=()=>Object.fromEntries(backendFiles.map(file=>[path.basename(file),sha(file)]));
const report={input,output,sourceSha256:original,backendSourceSha256:backendSourceHashes(),rendererIndexSha256:sha(path.join(root,'desktop/dist/index.html')),windowEverVisible:false,requests:[],checks:[],captures:[],passed:false};let last;
let millingCheckpoints,millingCheckpointPath;
if(process.env.WORKFLOW_MILLING==='1'){
 assert.equal(process.env.WORKFLOW_RECORDED_ONLY,'1','Milling harness verifies recorded native controls only');
 assert.notEqual(process.env.WORKFLOW_CRANIOTOMY,'1','Select one native opening method');
 assert(process.env.WORKFLOW_MILLING_CHECKPOINTS,'Use the formal dynamic native checkpoint file');
 millingCheckpointPath=path.resolve(process.env.WORKFLOW_MILLING_CHECKPOINTS);
 assert(millingCheckpointPath.startsWith(base+path.sep),'Keep checkpoint provenance in ignored testcases');
 millingCheckpoints=JSON.parse(fs.readFileSync(millingCheckpointPath,'utf8'));
 assert.equal(millingCheckpoints.schema,'resectionlab.generated-milling-checkpoints.v1');
 assert.equal(millingCheckpoints.checkpoints.length,6);
 report.millingCheckpointPath=millingCheckpointPath;report.millingCheckpointSha256=sha(millingCheckpointPath);
}
app.setPath('userData',path.join(output,'hidden-profile'));process.argv.push('--case',input,'--automation-hidden');
dialog.showSaveDialog=async()=>({canceled:false,filePath:saved});
const handle=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(channel,handler)=>handle(channel,async(...args)=>{const started=Date.now();try{const value=await handler(...args);if(channel==='research:surgicalRehearsal'){last=value;report.requests.push({command:args[1]?.command,...(args[1]?.command==='plan'?{settings:{opponent:args[1].opponent,planningHorizon:args[1].planningHorizon,beamWidth:args[1].beamWidth,initialAccessRadiusMm:args[1].initialAccessRadiusMm,cumulativeReentry:args[1].cumulativeReentry,exploreExteriorFaces:args[1].exploreExteriorFaces}}:{}),seconds:(Date.now()-started)/1000,stateId:value?.view?.stateId,branchId:value?.branchId,position:value?.position});}return value;}catch(error){if(channel==='research:surgicalRehearsal')report.requests.push({command:args[1]?.command,seconds:(Date.now()-started)/1000,error:String(error)});throw error;}});
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
app.on('browser-window-created',(_event,window)=>{
 window.on('show',()=>{report.windowEverVisible=true;window.hide();});
 window.webContents.once('did-finish-load',async()=>{
  window.webContents.setBackgroundThrottling(false);window.setContentSize(1440,900);
  const js=async code=>{const value=await window.webContents.executeJavaScript(`(()=>{try{return (${code})}catch(error){return {automationError:String(error)}}})()`,true);if(value?.automationError)throw Error(value.automationError);return value;};
  const wait=async(expression,label,seconds=240)=>{const started=Date.now();while(Date.now()-started<seconds*1000){if(await js(expression))return;await delay(80);}throw Error('Timeout '+label+'\n'+await js('document.body.innerText'));};
  const ready=()=>wait(`!!document.querySelector('.surgical-workflow')&&document.querySelector('.rl-viewer')?.dataset.surfacesPending==='0'&&!document.querySelector('.viewer-loading')&&![...document.querySelectorAll('.surgical-panel [role="status"]')].some(e=>/^(Computing instrument action|Preparing an exact recorded state|Preparing accepted display|Verifying saved source-bound rehearsal)/.test(e.textContent.trim()))`,'accepted display');
  const click=async(label,scope='document')=>{await js(`(()=>{const root=${scope},e=[...root.querySelectorAll('button,input[type=checkbox]')].find(e=>e.textContent.trim()===${JSON.stringify(label)}||e.getAttribute('aria-label')===${JSON.stringify(label)});if(!e||e.matches(':disabled')||!e.checkVisibility())throw Error('Missing visible enabled '+${JSON.stringify(label)});e.click()})()`);await delay(100);};
  const set=async(label,value)=>{await js(`(()=>{const e=document.querySelector('[aria-label='+${JSON.stringify(JSON.stringify(label))}+']');if(!e||e.matches(':disabled')||!e.checkVisibility())throw Error('Missing visible input '+${JSON.stringify(label)});Object.getOwnPropertyDescriptor(e.tagName==='SELECT'?HTMLSelectElement.prototype:HTMLInputElement.prototype,'value').set.call(e,${JSON.stringify(String(value))});e.dispatchEvent(new Event(e.tagName==='SELECT'?'change':'input',{bubbles:true}));})()`);await delay(100);};
  const stage=async label=>{await click(label+' stage');await wait(`document.querySelector('.surgical-workflow')?.dataset.workflowStage===${JSON.stringify(label==='Rehearse/edit'?'rehearse':label.toLowerCase())}`,'stage '+label);};
  const capture=async(name,selector='.surgical-workflow')=>{await ready();await js(`document.querySelector(${JSON.stringify(selector)})?.scrollIntoView({block:'start'})`);await delay(120);assert(!window.isVisible());fs.writeFileSync(path.join(output,name+'.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());report.captures.push(name);};
  const state=()=>js(`({stateId:document.querySelector('.surgical-panel')?.dataset.stateId,branchId:document.querySelector('[aria-label="Surgical branch"]')?.value,position:document.querySelector('[aria-label="Surgical timeline"]')?.value})`);
  const request=async(label,command)=>{const start=report.requests.length;await click(label);const started=Date.now();while(!report.requests.slice(start).some(row=>row.command===command)){if(Date.now()-started>240000)throw Error('Missing result '+command);await delay(80);}await ready();assert(!report.requests.slice(start).find(row=>row.command===command)?.error);};
  try{
   await wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`,'source load');await click('Rehearse');await ready();
   if(await js(`[...document.querySelectorAll('button')].some(e=>e.textContent.trim()==='Verify saved replay'&&!e.disabled&&e.checkVisibility())`)){await request('Verify saved replay','inspect');}
   report.initial=await state();if(process.env.WORKFLOW_EXPECT_INITIAL_STATE)assert.equal(last.view.stateId,process.env.WORKFLOW_EXPECT_INITIAL_STATE);report.patient=last?.interactivePlanningAvailability?.admission?.sourceEvidence?.patientId;
   await capture('01-case');assert(await js(`document.querySelector('[aria-label="Case anatomy and evidence"]').checkVisibility()`));
   const start=report.requests.length,baseline=await state();
   for(const name of ['Plan','Compare','Rehearse/edit','Case'])await stage(name);
   assert.equal(report.requests.length,start,'stage navigation must not request native simulation');assert.deepEqual(await state(),baseline);
   report.checks.push('Case → Plan → Compare → Rehearse/edit navigation preserves accepted state without native calls');
   if(process.env.WORKFLOW_RECORDED_ONLY==='1'){
    await stage('Rehearse/edit');report.recordedInitialBranch=last.branchId;await capture('02-recorded-endpoint','.workflow-rehearsal');
    const replayStart=report.requests.length;await click('Play replay');const playStarted=Date.now();while(!report.requests.slice(replayStart).some(row=>row.command==='seekFrame')){if(Date.now()-playStarted>120000)throw Error('Missing actual recorded replay frame');await delay(80);}
    await click('Pause replay');await ready();if(await js(`[...document.querySelectorAll('button')].some(e=>e.textContent.trim()==='Return to recorded boundary'&&e.checkVisibility())`))await request('Return to recorded boundary','seek');
    const endpoint=Number(process.env.WORKFLOW_EXPECT_END_POSITION??last.branches.find(row=>row.id===last.branchId).steps),seekStart=report.requests.length;await set('Surgical timeline',endpoint);const seekStarted=Date.now();while(!report.requests.slice(seekStart).some(row=>row.command==='seek')){if(Date.now()-seekStarted>120000)throw Error('Missing actual recorded endpoint seek');await delay(80);}await ready();assert.equal(last.position,endpoint);assert.equal(last.branchId,report.recordedInitialBranch);
    const byLabel=last.view.metrics.removedByLabelMm3;
    const removed=material=>last.view.scene.regions.filter(region=>region.material===material).reduce((total,region)=>total+Number(byLabel[region.name]??0),0);
    report.recordedOutcome={position:last.position,stateId:last.view.stateId,boneRemovedMm3:removed('bone'),brainRemovedMm3:removed('soft_tissue'),targetRemovedMm3:last.view.metrics.targetRemovedMm3};
    if(process.env.WORKFLOW_EXPECT_BONE_MM3!==undefined)assert(Math.abs(report.recordedOutcome.boneRemovedMm3-Number(process.env.WORKFLOW_EXPECT_BONE_MM3))<1e-5);
    if(process.env.WORKFLOW_EXPECT_NO_BRAIN_TARGET==='1'){assert.equal(report.recordedOutcome.brainRemovedMm3,0);assert.equal(report.recordedOutcome.targetRemovedMm3,0);}
    if(process.env.WORKFLOW_MILLING==='1'){
     assert.equal(last.view.scene.sceneId,millingCheckpoints.sourceSceneId);
     assert.equal(last.position,millingCheckpoints.finalPosition);
     assert.equal(last.view.stateId,millingCheckpoints.finalStateId);
     report.millingCheckpoints=[];
     for(const item of millingCheckpoints.checkpoints){
      const since=report.requests.length;await set('Surgical timeline',item.position);
      const started=Date.now();while(!report.requests.slice(since).some(row=>row.command==='seek')){if(Date.now()-started>120000)throw Error('Missing milling checkpoint seek');await delay(80);}await ready();
      assert.equal(last.position,item.position);assert.equal(last.view.stateId,item.stateId);
      const opening=last.view.craniotomy,proof=opening.milling;
      assert.equal(opening.method,'milling');assert.equal(opening.phase,item.phase);
      assert.equal(opening.openingComplete,item.openingComplete);assert.equal(opening.apertureEvacuated,item.apertureEvacuated);
      assert.equal(opening.milledBoneMm3,item.milledBoneMm3);assert.equal(last.view.metrics.nonTargetRemovedMm3,item.nonTargetRemovedMm3);
      assert.equal(last.view.metrics.targetRemovedMm3,0);assert.equal(last.view.moved.length,0);
      assert.equal(opening.flapEvacuatedMm3,0);assert.equal(proof.treatmentEntrySupported,false);
      const nativeEvent=last.view.history?.at(-1);
      if(item.position===0&&Array.isArray(last.view.history))assert.equal(last.view.history.length,0);
      if(item.position>0&&nativeEvent){assert.equal(nativeEvent.status,item.lastNativeStatus);assert.equal(nativeEvent.reason??null,item.lastNativeReason??null);}
      for(const key of ['apertureRemoved','historyVerified','debrisConserved','generatedBoneDebrisMm3','debrisRemovedMm3','debrisPresentMm3','debrisEscapedMm3'])assert.equal(proof[key],item[key],key);
      await capture('milling-'+item.position+'-'+item.name,'.workflow-rehearsal');
      await stage('Compare');await js(`(()=>{const d=document.querySelector('.workflow-checkpoint-burdens');if(!d.open)d.querySelector('summary').click()})()`);
      assert(await js(`document.querySelector('[aria-label="Native bone opening"]')?.checkVisibility()`));
      const displayed=await js(`document.querySelector('[aria-label="Native bone opening"]').innerText`);
      assert(displayed.includes('Milled bone')&&displayed.includes('Bone debris collected')&&displayed.includes('Bone debris remaining'));
      assert(displayed.includes('Dural opening and tumor entry remain unsupported.'));
      assert(displayed.includes('Opening with both hands withdrawn: '+(item.openingComplete?'Complete':'Incomplete')));
      assert(!await js(`!!document.querySelector('[aria-label="Native bone opening"] [data-craniotomy-metric="flapEvacuatedMm3"]')`));
      for(const [key,value]of [['milledBoneMm3',item.milledBoneMm3],['debrisRemovedMm3',item.debrisRemovedMm3],['debrisPresentMm3',item.debrisPresentMm3]]){
       const text=await js(`document.querySelector('[aria-label="Native bone opening"] [data-craniotomy-metric="${key}"]').textContent`);
       assert.equal(Number(text),Number(Number(value).toFixed(2)),key+' displayed native quantity');
      }
      if(item.name==='bone-removed-debris-retained')assert(displayed.includes('debris collection pending'));
      if(item.name==='debris-collected-withdrawal-pending')assert(displayed.includes('withdrawal pending'));
      if(item.name==='complete-native-withdrawal')assert(displayed.includes('instruments withdrawn'));
      report.millingCheckpoints.push({position:last.position,stateId:last.view.stateId,opening,displayed,
       nativeHistoryAvailable:!!nativeEvent,lastNativeStatus:nativeEvent?.status??null,lastNativeReason:nativeEvent?.reason??null});
      await capture('milling-summary-'+item.position,'.workflow-checkpoint-burdens');await stage('Rehearse/edit');
     }
     assert.equal(last.position,millingCheckpoints.finalPosition);assert.equal(last.view.stateId,millingCheckpoints.finalStateId);
     report.millingDisplayed=report.millingCheckpoints.at(-1).displayed;
     report.checks.push('Dynamic actual native milling checkpoints preserve capacity partials, wholeA81 removal, global debris collection, true exterior withdrawal and unsupported dural/tumor entry; recorded playback supplies no new patient plan');
    }
    if(process.env.WORKFLOW_CRANIOTOMY==='1'){
     report.craniotomyCheckpoints=[];
     for(const item of [{position:8,phase:'KERF_IN_PROGRESS',removed:236,moved:0},
       {position:11,phase:'KERF_IN_PROGRESS',removed:236,moved:0},
       {position:15,phase:'GRASPED_OR_TRANSPORTED',removed:236,moved:0},
       {position:16,phase:'GRASPED_OR_TRANSPORTED',removed:236,moved:49},
       {position:17,phase:'EVACUATED',removed:285,moved:0},
       {position:18,phase:'HOLDER_WITHDRAWN',removed:285,moved:0}]){
      const since=report.requests.length;await set('Surgical timeline',item.position);
      const started=Date.now();while(!report.requests.slice(since).some(row=>row.command==='seek')){if(Date.now()-started>120000)throw Error('Missing craniotomy checkpoint seek');await delay(80);}await ready();
      assert.equal(last.position,item.position);assert.equal(last.view.craniotomy.phase,item.phase);
      assert.equal(last.view.metrics.nonTargetRemovedMm3,item.removed);assert.equal(last.view.metrics.targetRemovedMm3,0);assert.equal(last.view.moved.length,item.moved);
      assert.equal(last.view.craniotomy.openingComplete,item.position===18);
      if(item.position>=11)assert(last.view.separatedAttachments.includes('idealized-periosteal'));
      report.craniotomyCheckpoints.push({position:last.position,stateId:last.view.stateId,opening:last.view.craniotomy,movedCells:last.view.moved.length,removedCells:last.view.removed.length});
      await capture('craniotomy-'+item.position,'.workflow-rehearsal');
     }
     await stage('Compare');await js(`(()=>{const d=document.querySelector('.workflow-checkpoint-burdens');if(!d.open)d.querySelector('summary').click()})()`);
     assert(await js(`document.querySelector('[aria-label="Native bone opening"]')?.checkVisibility()`));
     report.craniotomyDisplayed=await js(`document.querySelector('[aria-label="Native bone opening"]').innerText`);
     assert(report.craniotomyDisplayed.includes('285')&&report.craniotomyDisplayed.includes('49')&&report.craniotomyDisplayed.includes('236'));
     await capture('craniotomy-native-summary','.workflow-checkpoint-burdens');await stage('Rehearse/edit');
     report.checks.push('Generated bone-only replay shows actual cutting, declared attachment release,49-cell lift, exterior evacuation, empty-holder withdrawal and native complete phase; whole285mm3 remains non-target and no tumor-task pass is claimed');
    }
    await capture('03-recorded-endpoint-after-play','.workflow-rehearsal');report.checks.push('Recorded-only native replay plays actual frames, pauses, seeks the exact endpoint and preserves declared bone/brain/target quantities; no plan is generated');
   }else if(process.env.WORKFLOW_STAGED_ONLY==='1'){
    await stage('Plan');assert.equal(await js(`document.querySelector('.surgical-workflow').dataset.workflowPlanner`),'staged','A genuine paired source must expose the existing staged scope without eligibility changes');
    const before=await state();report.stagedInitial={branchId:last.branchId,position:last.position,stateId:last.view.stateId,sceneId:last.view.scene.sceneId,physicsSchema:last.view.physics?.profile?.schema};
    if(process.env.WORKFLOW_REOPEN!=='1'){
     report.stagedInputs=await js(`({planningDepth:document.querySelector('[aria-label="Staged planning depth"]')?.value,resectionSequence:document.querySelector('[aria-label="Staged resection sequence"]')?.value,initialAccessConfiguration:document.querySelector('[aria-label="Staged initial access configuration"]')?.value,exploreEntrySurfaces:document.querySelector('[aria-label="Explore additional entry surfaces"]')?.checked})`);
     assert.equal(report.stagedInputs.planningDepth,'2');assert.equal(report.stagedInputs.resectionSequence,'local');assert.equal(report.stagedInputs.initialAccessConfiguration,'saved');assert.equal(report.stagedInputs.exploreEntrySurfaces,false);
     await capture('02-staged-default-plan');await click('Acknowledge staged source assumptions');await request('Generate staged source strategies','plan');report.stagedRequest=report.requests.find(row=>row.command==='plan').settings;assert.equal(report.stagedRequest.opponent,'anatomy-staged-strategy');assert.equal(report.stagedRequest.planningHorizon,2);assert.equal(report.stagedRequest.beamWidth,3);assert.equal(report.stagedRequest.initialAccessRadiusMm,undefined);assert(!report.stagedRequest.cumulativeReentry&&!report.stagedRequest.exploreExteriorFaces);await wait(`document.querySelector('.surgical-workflow').dataset.workflowStage==='compare'`,'staged generated Compare');assert.equal(last.view.stateId,before.stateId);assert.equal(last.position,0);
    }else{await stage('Compare');assert(last.stagedPlanning,'A cold native save must retain the staged report');}
    const summary=last.stagedPlanning;assert(summary);assert.equal(summary.maxContactReplansPerSkill,8);assert.equal(summary.controllerMode,'bounded-sequence-lookahead');assert.equal(summary.sequenceSearch.horizon,2);assert.equal(summary.sequenceSearch.beamWidth,3);assert(summary.cost.physicalCalls<=summary.cost.maxPhysicalCalls&&summary.cost.maxPhysicalCalls<=4000);assert(summary.wallSeconds<=240);
    if(process.env.WORKFLOW_EXPECT_STAGED_SEAL)assert.equal(summary.comparisonSeal,process.env.WORKFLOW_EXPECT_STAGED_SEAL);
    report.stagedSummary=summary;report.stagedWorkingConfigurations=[...new Set(summary.alternatives.flatMap(row=>Object.keys(row.outcome.configurationTargetRemovedMm3)))];report.stagedModel=summary.cusaModel;assert.equal(summary.cusaModel,'legacy-instantaneous-with-v2-mechanics','This existing patient source must not be relabeled as v3');
    await js(`(()=>{const d=[...document.querySelectorAll('.staged-strategies details')].find(d=>d.querySelector('summary')?.textContent==='Recorded stages, costs and task criteria');if(!d)throw Error('Missing recorded staged disclosure');if(!d.open)d.querySelector('summary').click()})()`);
    assert.equal(await js(`document.querySelector('.surgical-workflow').dataset.workflowPlanner`),'staged');assert(await js(`[...document.querySelectorAll('.staged-strategies p')].some(p=>p.checkVisibility()&&p.textContent.includes('at most 8 finite-aspiration clearance and resume cycles'))`));assert(await js(`[...document.querySelectorAll('.staged-strategies [role="status"]')].some(p=>p.checkVisibility()&&p.textContent.includes('Recorded native comparison'))`));report.displayedStagedBudget=await js(`(()=>{const p=[...document.querySelectorAll('.staged-strategies p')].find(p=>p.checkVisibility()&&p.textContent.includes('at most 8 finite-aspiration clearance and resume cycles'));p.dataset.stagedBudgetProof='true';return p.textContent})()`);await capture('03-staged-recorded-budget','[data-staged-budget-proof]');
    report.checks.push('Actual default-depth staged Generate on a predeclared paired DEVELOPMENT source preserves the intact source cursor and displays its native max8 recorded budget, bounded search receipt and honest legacy model');
    const row=summary.alternatives.find(row=>row.nativeActions>0);
    if(row){await click('Inspect '+row.sourceMethod+' · '+row.hand+' hand');await ready();assert.equal(last.branchId,row.branchId);assert.equal(last.position,row.nativeActions);assert.equal(last.view.stateId,row.finalStateId);await wait(`document.querySelector('.surgical-workflow').dataset.workflowStage==='rehearse'`,'recorded staged branch rehearsal');await capture('04-staged-native-branch','.workflow-rehearsal');report.stagedSelectedBranch={branchId:row.branchId,nativeActions:row.nativeActions,finalStateId:row.finalStateId,status:row.status};report.checks.push('Selecting a staged card opens its existing checked native branch in Rehearse/edit; it does not generate a new patient plan');}
   }else{
   await stage('Plan');
   const task=process.env.WORKFLOW_TASK||'approach';
   if(await js(`!!document.querySelector('[aria-label="Supported planning task"]')`))await set('Supported planning task',task);
   assert.equal(await js(`document.querySelector('.surgical-workflow').dataset.workflowPlanner`),task);
   assert.equal(await js(`[...document.querySelectorAll('.surgical-workflow>.preparation-planning,.surgical-workflow>.two-tool-planning,.surgical-workflow>.approach-planning')].filter(e=>e.checkVisibility()).length`),1);
   await capture('02-plan');await stage('Compare');
   if(process.env.WORKFLOW_REOPEN==='1'){
    assert(last?.interactivePlanning);report.coldInitialBranch=last.branchId;
    await stage('Rehearse/edit');await capture('03-cold-installed-branch','.workflow-rehearsal');
    if(Number(last?.branches?.find(row=>row.id===last.branchId)?.steps)>0){
     const replayStart=report.requests.length;await click('Play replay');const started=Date.now();while(!report.requests.slice(replayStart).some(row=>row.command==='seekFrame')){if(Date.now()-started>120000)throw Error('Missing actual recorded replay frame');await delay(80);}
     await click('Pause replay');await ready();if(await js(`[...document.querySelectorAll('button')].some(e=>e.textContent.trim()==='Return to recorded boundary'&&e.checkVisibility())`))await request('Return to recorded boundary','seek');
     report.checks.push('Fresh process plays actual native recorded frames on the selected saved branch and pauses');
    }
    await stage('Compare');await js(`(()=>{const d=document.querySelector('.approach-history');if(d&&!d.open)d.querySelector('summary').click()})()`);if(await js(`[...document.querySelectorAll('button')].some(e=>e.textContent.trim()==='Open planning checkpoint'&&e.checkVisibility()&&!e.disabled)`)){await request('Open planning checkpoint','selectBranch');await stage('Compare');}
    assert(last.interactivePlanning.current);await capture('04-cold-comparison');
    report.checks.push('Cold reopen retains saved comparison and returns to its exact source checkpoint without a new plan');
   }else if(task==='approach'){
    await js(`(()=>{const d=document.querySelector('.approach-history');if(d&&!d.open)d.querySelector('summary').click()})()`);if(await js(`[...document.querySelectorAll('button')].some(e=>e.textContent.trim()==='Open planning checkpoint'&&e.checkVisibility()&&!e.disabled)`))await request('Open planning checkpoint','selectBranch');
    await stage('Plan');await js(`document.querySelector('.approach-constraints').open=true`);
    const label='Allowable entry region radius',originalValue=await js(`document.querySelector('[aria-label="'+${JSON.stringify(label)}+'"]').value`);
    await js(`window.workflowDraftNode=document.querySelector('[aria-label="'+${JSON.stringify(label)}+'"]')`);
    await set(label,Number(originalValue)+.1);await stage('Compare');
    assert.equal(await js(`document.querySelector('[data-planning-status]')?.dataset.planningStatus`),'draft');
    await stage('Plan');assert(await js(`window.workflowDraftNode===document.querySelector('[aria-label="'+${JSON.stringify(label)}+'"]')`));await set(label,originalValue);await stage('Compare');assert.equal(await js(`document.querySelector('[data-planning-status]')?.dataset.planningStatus`),'draft');
    report.checks.push('Real edit/revert invalidates prior comparison; stage navigation retains the identical draft input');
    await stage('Plan');if(!await js(`document.querySelector('[aria-label="Acknowledge interactive planning unknowns"]').checked`))await click('Acknowledge interactive planning unknowns');
    const accepted=await state(),beforeCancel=report.requests.length;await click('Generate approaches');await wait(`[...document.querySelectorAll('button')].some(e=>e.textContent.trim()==='Cancel computation'&&!e.disabled&&e.checkVisibility())`,'owned cancellation');await click('Cancel computation');
    const started=Date.now();while(!report.requests.slice(beforeCancel).some(row=>row.command==='generateApproaches')){if(Date.now()-started>120000)throw Error('Missing cancelled terminal result');await delay(80);}assert(report.requests.slice(beforeCancel).some(row=>/cancelled/i.test(row.error??'')));await ready();assert.deepEqual(await state(),accepted);
    report.checks.push('Actual planner cancellation reaches its owned backend request and preserves accepted anatomy');
    await request('Generate approaches','generateApproaches');await wait(`document.querySelector('.surgical-workflow').dataset.workflowStage==='compare'`,'generated comparison stage');await capture('03-compare');
    report.generatedRoutes=last.interactivePlanning.routes.length;report.decision=last.interactivePlanning.decision;if(process.env.WORKFLOW_EXPECT_ROUTES!==undefined)assert.equal(report.generatedRoutes,Number(process.env.WORKFLOW_EXPECT_ROUTES));
    if(report.generatedRoutes){await click('Inspect '+last.interactivePlanning.routes[0].label);await request('Preview checked approach','previewApproach');assert.equal(last.view.stateId,accepted.stateId);await capture('04-native-preview');await request('Install checked branch','applyApproach');await wait(`document.querySelector('.surgical-workflow').dataset.workflowStage==='rehearse'`,'installed rehearsal stage');assert.notEqual(last.branchId,accepted.branchId);await capture('05-installed-rehearsal','.workflow-rehearsal');report.checks.push('Regeneration enters Compare; sealed native preview/install creates a replayable branch and enters Rehearse/edit');}
    else{assert(last.interactivePlanning.decision.operation==='STOP');report.checks.push('Native STOP remains a refusal with no claimed treatment completion');}
   }
   }
   if(process.env.WORKFLOW_EVIDENCE==='1'){
    const acceptedState=last.view.stateId;await stage('Plan');await js(`document.querySelector('.approach-constraints').open=true`);
    if(!await js(`document.querySelector('[aria-label="Acknowledge interactive planning unknowns"]').checked`))await click('Acknowledge interactive planning unknowns');
    if(!await js(`document.querySelector('[aria-label="Require functional evidence"]').checked`))await click('Require functional evidence');
    await request('Generate approaches','generateApproaches');assert.equal(last.view.stateId,acceptedState);assert.equal(last.interactivePlanning.routes.length,0);assert(last.interactivePlanning.missingRequiredEvidence.includes('functional'));report.functionalRefusal=last.interactivePlanning.decision;await capture('07-required-evidence-refusal');
    await stage('Plan');await js(`document.querySelector('.approach-constraints').open=true`);await click('Require functional evidence');await request('Generate approaches','generateApproaches');assert.equal(last.view.stateId,acceptedState);if(process.env.WORKFLOW_EXPECT_ROUTES!==undefined)assert.equal(last.interactivePlanning.routes.length,Number(process.env.WORKFLOW_EXPECT_ROUTES));await capture('08-explicitly-regenerated-comparison');
    report.checks.push('Requiring missing functional evidence refuses every route without physical mutation; removing the requirement still needs explicit regeneration');
   }
   if(process.env.WORKFLOW_LEDGER==='1'){
    await stage('Compare');await js(`(()=>{const d=document.querySelector('.workflow-checkpoint-burdens');if(!d.open)d.querySelector('summary').click();const ledger=[...d.querySelectorAll('details')].find(e=>e.querySelector('summary')?.textContent==='Complete-sequence breakdown');if(ledger&&!ledger.open)ledger.querySelector('summary').click()})()`);
    if(await js(`[...document.querySelectorAll('button')].some(e=>e.textContent.trim()==='Calculate sequence breakdown'&&e.checkVisibility()&&!e.disabled)`))await request('Calculate sequence breakdown','inspect');
    assert.equal(last.sequenceOutcome.stateId,last.view.stateId);if(process.env.WORKFLOW_EXPECT_WITHDRAWN==='1')assert.equal(last.sequenceOutcome.withdrawalComplete,true);assert.equal(last.sequenceOutcome.dimensions.nonTargetRemovedMm3.cumulative,last.view.metrics.nonTargetRemovedMm3);report.sequenceOutcome=last.sequenceOutcome;
    await capture('06-complete-sequence-ledger','.workflow-checkpoint-burdens');report.checks.push('Compare expands the single accepted-state sequence ledger, retaining access burden and unknown evidence');
   }
   await click('Save');await wait(`true`,'save dispatch');const savedAt=Date.now();while(!fs.existsSync(saved)){if(Date.now()-savedAt>30000)throw Error('Missing native save');await delay(80);}report.saved=saved;report.savedState=await state();
   assert.equal(sha(input),original);if(millingCheckpointPath)assert.equal(sha(millingCheckpointPath),report.millingCheckpointSha256);assert.deepEqual(backendSourceHashes(),report.backendSourceSha256,'Backend source must remain frozen throughout this run');assert(!window.isVisible()&&!report.windowEverVisible);report.backendSourceStable=true;report.sourcePreserved=true;report.passed=true;
  }catch(error){report.error=String(error);report.sourcePreserved=sha(input)===original;report.failureState=await state().catch(()=>null);report.body=await js('document.body.innerText').catch(()=>null);fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());console.error(error);}
  fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,error:report.error,saved:report.saved}));app.exit(report.passed?0:1);
 });
});
require('./main.cjs');
