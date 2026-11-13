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
const packagedApp=process.env.WORKFLOW_APP?fs.realpathSync(process.env.WORKFLOW_APP):null;
const appCode=packagedApp?path.join(packagedApp,'Contents/Resources/app.asar'):path.join(root,'desktop');
const electronCode=path.join(appCode,'electron');
const input=fs.realpathSync(process.env.WORKFLOW_CASE||path.join(base,'outputs/interactive-planning-20261031/bridge-v4-head-physics/interactive-planning.ressectionlab'));
const output=path.resolve(process.env.WORKFLOW_OUTPUT||path.join(base,'decision-system-v1/workflow-v1/hidden-v1'));
const externalSourceRoot='/Volumes/shafrir/old coding projects/RessectionLabData';
assert(input.startsWith(base+path.sep)||(process.env.WORKFLOW_RECORDED_ONLY==='1'&&input.startsWith(externalSourceRoot+path.sep)),'Use an ignored testcase or explicitly requested recorded external source');
assert(output.startsWith(base+path.sep),'Keep generated evidence in ignored testcases');
assert(!fs.existsSync(output),'Preserve previous evidence');fs.mkdirSync(output,{recursive:true});
const sha=p=>crypto.createHash('sha256').update((p.endsWith('.asar')?require('original-fs'):fs).readFileSync(p)).digest('hex'),original=sha(input),saved=path.join(output,'workflow-saved.ressectionlab');
const backendFiles=['surgical_staged_skills.py','surgical_sequence_search.py','surgical_staged_planning.py',
 'surgical_engine.py','surgical_scene.py','surgical_effects.py','surgical_tools.py','surgical_clock.py',
 'surgical_craniotomy.py','surgical_bone_transport.py','surgical_cranial_guards.py','surgical_craniotomy_skills.py',
 'surgical_cranial_stages.py','surgical_membrane_transport.py','surgical_joint_cranial_skills.py',
 'surgical_replay_cache.py','surgical_frame_replay.py','surgical_observed_cache.py','surgical_prefix_replay.py',
 'surgical_milling.py','surgical_transition_meter.py','surgical_drilling_v2.py','surgical_physics_v2.py','surgical_task_admission.py'].map(name=>path.join(root,'src/resectionlab',name));
const backendSourceHashes=()=>Object.fromEntries(backendFiles.map(file=>[path.basename(file),sha(file)]));
const report={input,output,sourceSha256:original,backendSourceSha256:backendSourceHashes(),rendererIndexSha256:sha(path.join(appCode,'dist/index.html')),
 launcher:{executable:process.execPath,isPackaged:app.isPackaged},appCode:{root:appCode,packagedApp,
 ...(packagedApp?{archiveSha256:sha(appCode)}:{}),
 files:Object.fromEntries(['main.cjs','preload.cjs','sidecar.cjs'].map(name=>[name,{path:path.join(electronCode,name),sha256:sha(path.join(electronCode,name))}]))},
 windowEverVisible:false,requests:[],checks:[],captures:[],passed:false};let last;
// Substitute only the real engine executable; all app IPC and UI actions remain production paths.
let workflowSidecar;
if(packagedApp&&process.env.WORKFLOW_ENGINE&&!app.isPackaged){
 // The development launcher preflights its Python path before constructing
 // Sidecar. Point that same executable check at the engine supplied below.
 const promises=require('node:fs/promises'),access=promises.access;
 const developmentPython=path.resolve(electronCode,'../..','.venv/bin/python');
 promises.access=(file,...args)=>access(file===developmentPython?fs.realpathSync(process.env.WORKFLOW_ENGINE):file,...args);
 report.launcher.engineOverrideIncludesStartupPreflight=true;
}
const sidecarModule=require(path.join(electronCode,'sidecar.cjs')),ProductionSidecar=sidecarModule.Sidecar;
sidecarModule.Sidecar=class extends ProductionSidecar{
 constructor(options){
  if(process.env.WORKFLOW_ENGINE){const executable=fs.realpathSync(process.env.WORKFLOW_ENGINE);report.engine={executable,sha256:sha(executable)};
   if(process.env.WORKFLOW_ACTUAL_APP==='1'){
    assert(app.isPackaged);assert.equal(process.execPath,path.join(packagedApp,'Contents/MacOS/RessectionLab'));
    assert.equal(options.executable,executable);assert.equal(options.python,undefined);assert.equal(options.sourcePath,undefined);assert.equal(options.observedSourceRoot,undefined);
    report.launcher.originalPackagedEngineOptions={executable:options.executable,cwd:options.cwd,runDir:options.runDir};
   }else options={...options,executable,cwd:output,python:undefined,sourcePath:undefined,observedSourceRoot:undefined,prefixWorker:undefined};
  }
  super(options);workflowSidecar=this;
 }
};
const canonical=value=>JSON.stringify(value,(_key,item)=>item&&typeof item==='object'&&!Array.isArray(item)?Object.fromEntries(Object.keys(item).sort().map(key=>[key,item[key]])):item);
const objectSha=value=>crypto.createHash('sha256').update(canonical(value)).digest('hex');
const acceptedSnapshot=value=>({caseHash:value.caseHash,branchId:value.branchId,position:value.position,branches:value.branches,
 viewSha256:objectSha(value.view),timelineSha256:objectSha(value.timeline),
 comparison:value.cranialPlanning,sequenceOutcome:value.sequenceOutcome});
const sourceInspection=process.env.WORKFLOW_SOURCE_INSPECTION==='1';
assert(!sourceInspection||!['WORKFLOW_CRANIAL_GENERATE','WORKFLOW_RECORDED_ONLY','WORKFLOW_STAGED_ONLY'].some(key=>process.env[key]==='1'),'Select one workflow scope');
const frozenWorkflow=sourceInspection||process.env.WORKFLOW_CRANIAL_GENERATE==='1';
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
let membraneCheckpoints,membraneCheckpointPath;
if(process.env.WORKFLOW_MEMBRANE==='1'){
 assert.equal(process.env.WORKFLOW_RECORDED_ONLY,'1','Membrane harness verifies recorded native controls only');
 assert.notEqual(process.env.WORKFLOW_MILLING,'1','Select one dynamic native checkpoint protocol');
 assert.notEqual(process.env.WORKFLOW_CRANIOTOMY,'1','Select one dynamic native checkpoint protocol');
 assert(process.env.WORKFLOW_MEMBRANE_CHECKPOINTS,'Use the sealed native stage checkpoint file');
 membraneCheckpointPath=path.resolve(process.env.WORKFLOW_MEMBRANE_CHECKPOINTS);
 assert(membraneCheckpointPath.startsWith(base+path.sep),'Keep checkpoint provenance in ignored testcases');
 membraneCheckpoints=JSON.parse(fs.readFileSync(membraneCheckpointPath,'utf8'));
 assert.equal(membraneCheckpoints.schema,'resectionlab.generated-membrane-checkpoints.v1');
 assert(membraneCheckpoints.checkpoints.length>=6,'Keep source, cuts, hold, extraction, work and exit checkpoints');
 assert.equal(process.env.WORKFLOW_MEMBRANE_ARCHIVE_SHA256||membraneCheckpoints.replayArchiveSha256,original,'Use the sealed native archive or the explicitly hashed real Save output');
 report.membraneNativeArchiveSha256=membraneCheckpoints.replayArchiveSha256;
 report.membraneCheckpointPath=membraneCheckpointPath;report.membraneCheckpointSha256=sha(membraneCheckpointPath);
}
app.setPath('userData',path.join(output,'hidden-profile'));process.argv.push('--case',input,'--automation-hidden');
dialog.showSaveDialog=async()=>({canceled:false,filePath:saved});
const handle=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(channel,handler)=>handle(channel,async(...args)=>{const started=Date.now();try{const value=await handler(...args);if(channel==='research:surgicalRehearsal'){last=value;report.requests.push({command:args[1]?.command,...(args[1]?.command==='plan'?{settings:{opponent:args[1].opponent,planningHorizon:args[1].planningHorizon,beamWidth:args[1].beamWidth,initialAccessRadiusMm:args[1].initialAccessRadiusMm,cumulativeReentry:args[1].cumulativeReentry,exploreExteriorFaces:args[1].exploreExteriorFaces}}:{}),...(process.env.WORKFLOW_CRANIAL_GENERATE==='1'&&args[1]?.command==='step'?{action:args[1].action,outcome:{status:value?.outcome?.status,reason:value?.outcome?.reason}}:{}),seconds:(Date.now()-started)/1000,stateId:value?.view?.stateId,branchId:value?.branchId,position:value?.position});}return value;}catch(error){if(channel==='research:surgicalRehearsal')report.requests.push({command:args[1]?.command,seconds:(Date.now()-started)/1000,error:String(error)});throw error;}});
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
   report.appCode.loadedUrl=window.webContents.getURL();
   report.appCode.preloadApiAvailable=await js("!!window.resectionApi&&typeof window.resectionApi.surgicalRehearsal==='function'");
   if(packagedApp){assert(report.appCode.loadedUrl.includes('/app.asar/dist/index.html'));assert(report.appCode.preloadApiAvailable);}
   await wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`,'source load');await click('Rehearse');await ready();
   if(await js(`[...document.querySelectorAll('button')].some(e=>e.textContent.trim()==='Verify saved replay'&&!e.disabled&&e.checkVisibility())`)){await request('Verify saved replay','inspect');}
   report.initial=await state();if(process.env.WORKFLOW_EXPECT_INITIAL_STATE)assert.equal(last.view.stateId,process.env.WORKFLOW_EXPECT_INITIAL_STATE);report.patient=last?.interactivePlanningAvailability?.admission?.sourceEvidence?.patientId;
   if(frozenWorkflow){
    const ping=await workflowSidecar.request('ping');report.backendBefore=ping.numericalBackend;report.runtime=ping.runtime;
    assert.equal(report.backendBefore.selected,process.env.WORKFLOW_EXPECT_BACKEND||'cpp');
    if(process.env.WORKFLOW_ENGINE){assert.equal(ping.runtime.frozen,true);assert(!ping.runtime.modulePath.startsWith(path.join(root,'src')+path.sep));assert(ping.runtime.cacheRoot&&!ping.runtime.cacheRoot.startsWith(path.dirname(report.engine.executable)+path.sep),'Frozen cache root must be outside the signed engine');}
   }
   await capture('01-case');assert(await js(`document.querySelector('[aria-label="Case anatomy and evidence"]').checkVisibility()`));
   const start=report.requests.length,baseline=await state();
   for(const name of ['Plan','Compare','Rehearse/edit','Case'])await stage(name);
   assert.equal(report.requests.length,start,'stage navigation must not request native simulation');assert.deepEqual(await state(),baseline);
   report.checks.push('Case → Plan → Compare → Rehearse/edit navigation preserves accepted state without native calls');
   if(process.env.WORKFLOW_CRANIAL_GENERATE==='1'){
    const inspectAccepted=async()=>{const value=await workflowSidecar.request('surgicalRehearsal',{caseHash:last.caseHash,command:'inspect',expectedStateId:last.view.stateId,includeSequenceOutcome:true},300000);last=value;return acceptedSnapshot(value);};
    const seek=async position=>{const start=report.requests.length;await set('Surgical timeline',position);const started=Date.now();while(!report.requests.slice(start).some(row=>row.command==='seek')){if(Date.now()-started>120000)throw Error('Missing cranial timeline seek');await delay(80);}await ready();assert.equal(last.position,position);};
    await stage('Plan');assert.equal(await js(`document.querySelector('.surgical-workflow').dataset.workflowPlanner`),'cranial');
    if(process.env.WORKFLOW_REOPEN==='1'){
     assert(process.env.WORKFLOW_EXPECT_REPORT,'Cold Open requires the exact warm report');
     const expected=JSON.parse(fs.readFileSync(process.env.WORKFLOW_EXPECT_REPORT,'utf8'));assert(expected.passed);
     assert(expected.engineTermination&&(expected.engineTermination.exitCode!==null||expected.engineTermination.signal!==null),'The warm engine must have terminated');
     report.cranialSnapshot=await inspectAccepted();assert.deepEqual(report.cranialSnapshot,expected.cranialSnapshot,'Cold Open preserves comparison, branch, action history, ledger, source anatomy and replay cursor');
     assert.equal(report.backendBefore.binarySha256,expected.backendBefore.binarySha256);assert.deepEqual(report.backendBefore.buildIdentity,expected.backendBefore.buildIdentity);
     assert.equal(report.requests.filter(row=>row.command==='plan').length,0,'Cold Open never plans automatically');
     await stage('Compare');await capture('02-cranial-cold-comparison');await stage('Rehearse/edit');
     report.checks.push('Cold Open restores the exact comparison, selected native branch, complete action history, ledger, anatomy and replay cursor with zero Plan commands');
    }else{
     assert.equal(last.position,0);assert(last.branches.every(row=>row.steps===0),'Generate begins at a fresh unacted generated source');
     assert.equal(last.cranialPlanning,undefined);assert.equal(last.cranialPlanningAvailability.available,true);
     const accepted=await inspectAccepted();await capture('02-cranial-fresh-plan');await click('Acknowledge cranial research context');
     const cancelStart=report.requests.length;await click('Generate strategies');await wait(`[...document.querySelectorAll('button')].some(e=>e.textContent.trim()==='Cancel computation'&&!e.disabled&&e.checkVisibility())`,'cranial owned cancellation');
     await click('Cancel computation');await wait(`![...document.querySelectorAll('button')].some(e=>e.textContent.trim()==='Cancel computation'&&!e.disabled&&e.checkVisibility())`,'cranial cancellation completion');
     const cancelledAt=Date.now();while(!report.requests.slice(cancelStart).some(row=>row.command==='plan')){if(Date.now()-cancelledAt>120000)throw Error('Missing cranial cancellation terminal result');await delay(80);}
     assert(report.requests.slice(cancelStart).some(row=>row.command==='plan'&&/cancelled/i.test(row.error??'')));await ready();
     assert.deepEqual(await inspectAccepted(),accepted,'Cancellation preserves authoritative accepted anatomy and branches');
     report.checks.push('Real cranial Generate cancellation terminates its owned request without accepted-state or branch mutation');
     report.backendBeforeGenerate=(await workflowSidecar.request('ping')).numericalBackend;
     await request('Generate strategies','plan');await wait(`document.querySelector('.surgical-workflow').dataset.workflowStage==='compare'`,'fresh cranial Compare');
     report.backendAfterGenerate=(await workflowSidecar.request('ping')).numericalBackend;
     if(report.backendAfterGenerate.selected==='cpp'){const multi=identity=>Object.entries(identity.counters.batchSizes).reduce((sum,[size,count])=>sum+(Number(size)>1?count:0),0);assert(multi(report.backendAfterGenerate)>multi(report.backendBeforeGenerate),'Fresh production Generate must submit a genuine multi-item native batch');}
     const comparison=last.cranialPlanning;assert(comparison);assert.equal(last.position,0);assert.equal(last.view.stateId,report.initial.stateId);
     assert.equal(comparison.schema,'resectionlab.cranial-strategy-comparison.v1');assert.equal(comparison.cost.physicalCalls,comparison.cost.previewCalls+comparison.cost.directStepCalls);
     const row=comparison.alternatives.find(row=>row.completedFrozenTask&&row.nativeActions>0);assert(row,'Declared generated engineering control must retain a complete task alternative');
     report.cranialGenerated={comparisonSeal:comparison.comparisonSeal,decision:comparison.decision,cost:comparison.cost,alternativeCount:comparison.alternatives.length};
     await capture('03-cranial-generated-comparison');await click('Inspect '+row.label);await ready();
     await wait(`document.querySelector('.surgical-workflow').dataset.workflowStage==='rehearse'`,'selected cranial Rehearse');
     assert.equal(last.branchId,row.branchId);assert.equal(last.position,row.nativeActions);assert.equal(last.view.stateId,row.finalStateId);
     assert(row.wholeExitComplete&&row.sequenceOutcome.withdrawalComplete&&row.ledger.withinLimits);assert(row.goalAssessment.originalTargetTreatmentFraction>=.5);
     report.cranialSelected={branchId:row.branchId,actions:row.nativeActions,stateId:row.finalStateId,ledger:row.ledger,goalAssessment:row.goalAssessment};
     await capture('04-cranial-selected-branch','.workflow-rehearsal');
     const replayStart=report.requests.length;await click('Play replay');const playStarted=Date.now();while(!report.requests.slice(replayStart).some(row=>row.command==='seekFrame')){if(Date.now()-playStarted>120000)throw Error('Missing actual cranial replay frame');await delay(80);}
     await click('Pause replay');await ready();await seek(row.nativeActions);assert.equal(last.branchId,row.branchId);assert.equal(last.view.stateId,row.finalStateId);
     const beforeRefusal=await inspectAccepted(),oldAction=await js(`document.querySelector('[aria-label="Instrument action"]').value`);
     await set('Instrument action','extract');await request('Apply','step');assert.equal(last.outcome.status,'rejected');assert.equal(last.outcome.reason,'UNSUPPORTED_EXTRACTION','Well-formed native extraction must reach the CUSA operation guard');const refusedRequest=report.requests.at(-1);assert.equal(refusedRequest.action.operation,'extract');assert.equal(refusedRequest.action.path,undefined);assert.equal(refusedRequest.action.parameters.duty,undefined);report.cranialRefusal=last.outcome;report.cranialRefusalAction=refusedRequest.action;
     await set('Instrument action',oldAction);assert.deepEqual(await inspectAccepted(),beforeRefusal,'Valid refusal never mutates accepted state');
     await js(`(()=>{const d=[...document.querySelectorAll('.workflow-rehearsal details')].find(d=>d.querySelector('summary')?.textContent==='Pose, stroke and tool settings');if(!d)throw Error('Missing actual tool settings');d.open=true;})()`);
     const draftBefore=report.requests.length,duration=Number(await js(`document.querySelector('[aria-label="Action duration"]').value`));await set('Action duration',duration+.01);
     await stage('Compare');await capture('05-cranial-ledger');await stage('Rehearse/edit');assert.equal(Number(await js(`document.querySelector('[aria-label="Action duration"]').value`)),duration+.01);assert.equal(report.requests.length,draftBefore);await set('Action duration',duration);
     report.cranialSnapshot=await inspectAccepted();
     report.checks.push('Fresh Generate → Compare → selected real branch → observed replay/pause/seek preserves complete task ledger; valid native refusal and draft navigation preserve accepted anatomy');
    }
    await capture('06-cranial-save-cursor','.workflow-rehearsal');
   }else if(sourceInspection){
    const inspectAccepted=async()=>{last=await workflowSidecar.request('surgicalRehearsal',{caseHash:last.caseHash,command:'inspect',expectedStateId:last.view.stateId},300000);return JSON.parse(JSON.stringify(acceptedSnapshot(last)));};
    const assertUnacted=()=>{
     const a=last.cranialPlanningAvailability,s=last.view.scene,adapter=s.source.adapter;
     assert.equal(last.branchId,'main');assert.equal(last.position,0);assert(last.branches.every(row=>row.steps===0));assert.deepEqual(last.view.history,[]);
     assert.equal(last.view.metrics.targetRemovedMm3,0);assert.equal(last.view.metrics.nonTargetRemovedMm3,0);assert.equal(last.view.metrics.elapsedSeconds,0);
     assert(!last.cranialPlanning&&!last.stagedPlanning&&!last.jointPlanning);assert.equal(last.view.protection.configured,false);
     assert.equal(adapter.type,'native-patient-source-inspection');assert.equal(adapter.sourceRole,'DEVELOPMENT_CANDIDATE');assert.equal(adapter.physicalProtectionIndexConstructed,false);
     assert(a.present&&!a.available);assert.deepEqual(a.roots,[]);assert(a.blockers.some(row=>row.code==='SOURCE_INSPECTION_ONLY'));
     const targets=s.regions.filter(row=>row.target);assert(targets.length>0);assert(targets.every(row=>adapter.sourceLayers[row.name]?.cells>0));
     if(process.env.WORKFLOW_EXPECT_TARGET_CELLS)assert.equal(targets.reduce((total,row)=>total+adapter.sourceLayers[row.name].cells,0),Number(process.env.WORKFLOW_EXPECT_TARGET_CELLS));
     if(process.env.WORKFLOW_EXPECT_CASE_HASH)assert.equal(last.caseHash,process.env.WORKFLOW_EXPECT_CASE_HASH);
    };
    report.inspectionSnapshot=await inspectAccepted();assertUnacted();
    report.sourceInspection={caseHash:last.caseHash,scene:last.view.scene,availability:last.cranialPlanningAvailability,patientTaskSuccess:false,acceptedNativeActions:0};
    if(process.env.WORKFLOW_REOPEN==='1'){
     assert(process.env.WORKFLOW_EXPECT_REPORT);const expected=JSON.parse(fs.readFileSync(process.env.WORKFLOW_EXPECT_REPORT,'utf8'));assert(expected.passed);
     assert(expected.engineTermination&&(expected.engineTermination.exitCode!==null||expected.engineTermination.signal!==null));
     assert.deepEqual(report.inspectionSnapshot,expected.inspectionSnapshot,'Cold inspection retains source identity, complete target declarations, blockers and unacted native state');
    }
    await stage('Plan');assert.equal(await js(`document.querySelector('.surgical-workflow').dataset.workflowPlanner`),'cranial');
    const blockers=await js(`document.querySelector('[aria-label="Cranial planning blockers"]').innerText`);
    for(const row of last.cranialPlanningAvailability.blockers){assert(blockers.includes(row.code));assert(blockers.includes(row.message));}
    await click('Acknowledge cranial research context');const before=report.requests.length;
    assert(await js(`(()=>{const e=[...document.querySelectorAll('.cranial-strategies button')].find(e=>e.textContent.trim()==='Generate strategies');return !!e&&e.checkVisibility()&&e.disabled})()`));
    await js(`(()=>{const e=[...document.querySelectorAll('.cranial-strategies button')].find(e=>e.textContent.trim()==='Generate strategies');e.click()})()`);await delay(250);assert.equal(report.requests.length,before);
    assert.deepEqual(await inspectAccepted(),report.inspectionSnapshot);assertUnacted();await capture('02-inspection-generate-blocked');
    await stage('Compare');assert(await js(`document.querySelector('.cranial-strategies').innerText.includes('No complete tumor-treatment sequence has been established')`));
    assert.equal(await js(`document.querySelectorAll('.cranial-strategies .workflow-alternative').length`),0);await capture('03-inspection-no-treatment');
    await stage('Rehearse/edit');await capture('04-inspection-unacted');
    assert(report.requests.every(row=>['inspect','inspectArchive','create'].includes(row.command)),'Inspection never dispatches physical actions or planning');
    report.checks.push('Patient source inspection retains full source/target declarations and blockers; acknowledged Generate stays disabled with zero physical/planning IPC and no treatment claim');
   }else if(process.env.WORKFLOW_RECORDED_ONLY==='1'){
    await stage('Rehearse/edit');report.recordedInitialBranch=last.branchId;await capture('02-recorded-endpoint','.workflow-rehearsal');
    const replayStart=report.requests.length;await click('Play replay');const playStarted=Date.now();while(!report.requests.slice(replayStart).some(row=>row.command==='seekFrame')){if(Date.now()-playStarted>120000)throw Error('Missing actual recorded replay frame');await delay(80);}
    await click('Pause replay');await ready();if(await js(`[...document.querySelectorAll('button')].some(e=>e.textContent.trim()==='Return to recorded boundary'&&e.checkVisibility())`))await request('Return to recorded boundary','seek');
    const endpoint=Number(process.env.WORKFLOW_EXPECT_END_POSITION??last.branches.find(row=>row.id===last.branchId).steps),seekStart=report.requests.length;await set('Surgical timeline',endpoint);const seekStarted=Date.now();while(!report.requests.slice(seekStart).some(row=>row.command==='seek')){if(Date.now()-seekStarted>120000)throw Error('Missing actual recorded endpoint seek');await delay(80);}await ready();assert.equal(last.position,endpoint);assert.equal(last.branchId,report.recordedInitialBranch);
    const byLabel=last.view.metrics.removedByLabelMm3;
    const removed=material=>last.view.scene.regions.filter(region=>region.material===material).reduce((total,region)=>total+Number(byLabel[region.name]??0),0);
    report.recordedOutcome={position:last.position,stateId:last.view.stateId,boneRemovedMm3:removed('bone'),brainRemovedMm3:removed('soft_tissue'),targetRemovedMm3:last.view.metrics.targetRemovedMm3};
    if(process.env.WORKFLOW_EXPECT_BONE_MM3!==undefined)assert(Math.abs(report.recordedOutcome.boneRemovedMm3-Number(process.env.WORKFLOW_EXPECT_BONE_MM3))<1e-5);
    if(process.env.WORKFLOW_EXPECT_NO_BRAIN_TARGET==='1'){assert.equal(report.recordedOutcome.brainRemovedMm3,0);assert.equal(report.recordedOutcome.targetRemovedMm3,0);}
    if(process.env.WORKFLOW_MEMBRANE==='1'){
     assert.equal(last.view.scene.sceneId,membraneCheckpoints.sourceSceneId);
     assert.equal(last.position,membraneCheckpoints.finalPosition);assert.equal(last.view.stateId,membraneCheckpoints.finalStateId);
     report.membraneCheckpoints=[];
     for(const item of membraneCheckpoints.checkpoints){
      const since=report.requests.length;await set('Surgical timeline',item.position);
      const started=Date.now();while(!report.requests.slice(since).some(row=>row.command==='seek')){if(Date.now()-started>120000)throw Error('Missing membrane checkpoint seek');await delay(80);}await ready();
      assert.equal(last.position,item.position);assert.equal(last.view.stateId,item.stateId);
      const native=last.view.cranialStages;
      assert.equal(native.schema,'resectionlab.cranial-stages.v1');assert.equal(native.stateId,last.view.stateId);assert.equal(native.sceneId,membraneCheckpoints.sourceSceneId);
      assert.equal(native.contractHash,membraneCheckpoints.contractHash);
      for(const [key,value]of Object.entries(item.stage))assert.deepEqual(native[key],value,'native stage '+key);
      assert.equal(last.view.metrics.targetRemovedMm3,item.targetRemovedMm3);assert.equal(last.view.metrics.nonTargetRemovedMm3,item.nonTargetRemovedMm3);
      assert.equal(last.view.moved.length,item.movedCells);
      const event=last.view.history?.at(-1);
      if(item.position===0&&Array.isArray(last.view.history))assert.equal(last.view.history.length,0);
      if(item.position>0&&event){assert.equal(event.status,item.lastNativeStatus);assert.equal(event.reason??null,item.lastNativeReason??null);}
      await capture('membrane-'+item.position+'-'+item.name,'.workflow-rehearsal');
      await stage('Compare');await js(`(()=>{const d=document.querySelector('.workflow-checkpoint-burdens');if(!d.open)d.querySelector('summary').click()})()`);
      assert(await js(`document.querySelector('[aria-label="Native membrane stage"]')?.checkVisibility()`));
      const displayed=await js(`document.querySelector('[aria-label="Native membrane stage"]').innerText`);
      assert(displayed.includes('prior exposure cost unknown'));assert(displayed.includes('tumor-task completion and clinical safety remain unassessed'));
      assert(displayed.includes('Work-region entry: '+(item.stage.assessmentCurrent?(item.stage.workEntrySupported?'Permitted by native stage checks':'Not permitted'):'Unassessed')));
      const membraneAmount=await js(`document.querySelector('[data-membrane-metric="removedMembraneMm3"]').textContent`);
      if(item.removedMembraneMm3===null)assert.equal(membraneAmount,'Unknown');else assert.equal(Number(membraneAmount),Number(Number(item.removedMembraneMm3).toFixed(2)));
      assert.equal(Number(await js(`document.querySelector('[data-craniotomy-metric="declaredBoneRemovedMm3"]').textContent`)),Number(Number(item.boneRemovedMm3).toFixed(2)));
      assert.equal(Number(await js(`document.querySelector('[data-surgical-metric="nonTargetRemovedMm3"]').textContent`)),Number(Number(item.nonTargetRemovedMm3).toFixed(2)));
      report.membraneCheckpoints.push({position:last.position,stateId:last.view.stateId,stage:native,displayed,movedCells:last.view.moved.length,lastNativeStatus:event?.status??null,lastNativeReason:event?.reason??null});
      await capture('membrane-summary-'+item.position,'.workflow-checkpoint-burdens');await stage('Rehearse/edit');
     }
     assert.equal(last.position,membraneCheckpoints.finalPosition);assert.equal(last.view.stateId,membraneCheckpoints.finalStateId);
     report.checks.push('Recorded native joint sequence retains actual cut membrane, whole-patch movement/extraction, bone burden, bounded work, true withdrawal and unknown prior scalp cost; no patient task is admitted');
    }
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
    assert.equal(last.sequenceOutcome.stateId,last.view.stateId);if(membraneCheckpoints){assert.equal(last.sequenceOutcome.target.sourceVolumeMm3,membraneCheckpoints.ownedTargetMm3);assert.deepEqual(last.sequenceOutcome.originalTarget,membraneCheckpoints.originalTarget);}
    if(process.env.WORKFLOW_EXPECT_WITHDRAWN==='1')assert.equal(last.sequenceOutcome.withdrawalComplete,true);assert.equal(last.sequenceOutcome.dimensions.nonTargetRemovedMm3.cumulative,last.view.metrics.nonTargetRemovedMm3);report.sequenceOutcome=last.sequenceOutcome;
    await capture('06-complete-sequence-ledger','.workflow-checkpoint-burdens');report.checks.push('Compare expands the single accepted-state sequence ledger, retaining access burden and unknown evidence');
   }
   await click('Save');await wait(`true`,'save dispatch');const savedAt=Date.now();while(!fs.existsSync(saved)){if(Date.now()-savedAt>30000)throw Error('Missing native save');await delay(80);}report.saved=saved;report.savedSha256=sha(saved);report.savedState=await state();
   if(frozenWorkflow){
    const ping=await workflowSidecar.request('ping');report.backendAfter=ping.numericalBackend;
    report.cacheDirectories=ping.cacheDirectories;
    if(process.env.WORKFLOW_ENGINE){if(!sourceInspection&&process.env.WORKFLOW_REOPEN!=='1')assert(report.cacheDirectories.observed,'Actual replay must report its observed-frame cache owner');for(const directory of Object.values(report.cacheDirectories))assert(!directory.startsWith(path.dirname(report.engine.executable)+path.sep),'A replay cache must not write inside the signed engine');}
    if(!sourceInspection&&report.backendAfter.selected==='cpp')assert(report.backendAfter.counters.bindingCalls>(process.env.WORKFLOW_REOPEN==='1'?0:report.backendBefore.counters.bindingCalls),'The actual workflow must execute native geometry');
    if(process.env.WORKFLOW_REOPEN==='1')assert.equal(report.requests.filter(row=>row.command==='plan').length,0,'No automatic cold planning');
   }
   assert.equal(sha(input),original);if(packagedApp)assert.equal(sha(appCode),report.appCode.archiveSha256);if(millingCheckpointPath)assert.equal(sha(millingCheckpointPath),report.millingCheckpointSha256);if(membraneCheckpointPath)assert.equal(sha(membraneCheckpointPath),report.membraneCheckpointSha256);assert.deepEqual(backendSourceHashes(),report.backendSourceSha256,'Backend source must remain frozen throughout this run');assert(!window.isVisible()&&!report.windowEverVisible);report.backendSourceStable=true;report.sourcePreserved=true;report.passed=true;
  }catch(error){report.error=String(error);report.sourcePreserved=sha(input)===original;report.failureState=await state().catch(()=>null);report.body=await js('document.body.innerText').catch(()=>null);fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());console.error(error);}
  if(frozenWorkflow&&workflowSidecar){
   try{await workflowSidecar.stop();report.engineTermination={exitCode:workflowSidecar.child.exitCode,signal:workflowSidecar.child.signalCode};assert(report.engineTermination.exitCode!==null||report.engineTermination.signal!==null,'Engine must terminate before cold Open');}
   catch(error){report.passed=false;report.error=String(error);}
  }
  fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,error:report.error,saved:report.saved}));app.exit(report.passed?0:1);
 });
});
require(path.join(electronCode,'main.cjs'));
app.on('will-quit',()=>{if(!fs.existsSync(path.join(output,'report.json'))){report.error??='Application exited before workflow verification';fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));app.exit(1);}});
