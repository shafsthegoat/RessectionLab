'use strict';
// Real Case → Plan → Compare → Rehearse, native Save and cold process reopen.
// No show(), focus(), visible capture, synthetic IPC result or planner stub.
const {app,ipcMain,dialog,BrowserWindow}=require('electron');
let fatal=false;
const failFatal=error=>{if(!fatal){fatal=true;console.error('HIDDEN_SEQUENTIAL_EPISODE_FATAL '+String(error));}app.exit(1);process.exit(1);};
process.prependListener('uncaughtException',failFatal);process.prependListener('unhandledRejection',failFatal);
dialog.showErrorBox=(title,body)=>failFatal(`${title}: ${body}`);
const {execFileSync}=require('node:child_process');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..'),base=path.join(root,'testcases');
const input=fs.realpathSync(process.env.SEQUENTIAL_EPISODE_CASE||''),output=path.resolve(process.env.SEQUENTIAL_EPISODE_OUTPUT||'');
assert(input.startsWith(base+path.sep)&&output.startsWith(base+path.sep));assert(!fs.existsSync(output),'Preserve earlier evidence');fs.mkdirSync(output,{recursive:true});
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const canonical=value=>JSON.stringify(value,(_key,v)=>v&&typeof v==='object'&&!Array.isArray(v)?Object.fromEntries(Object.keys(v).sort().map(k=>[k,v[k]])):v);
const digest=value=>crypto.createHash('sha256').update(canonical(value)).digest('hex');
const saved=path.join(output,'desktop-saved.ressectionlab'),sourceSha256=sha(input),expected=JSON.parse(fs.readFileSync(process.env.SEQUENTIAL_EPISODE_INDEX,'utf8'));assert.equal(expected.episodes.length,5);
const codeHashes=()=>Object.fromEntries([
 ...['surgical_recorded_sequential.py','surgical_engine.py','surgical_scene.py','surgical_procedure_policy.py','surgical_planning_admission.py','surgical_consequences.py','surgical_cranial_planning.py','surgical_cranial_geometry.py','surgical_sequential_task.py','surgical_rl_env.py','surgical_bridge.py','surgical_sequence_outcomes.py'].map(n=>path.join(root,'src/resectionlab',n)),
 ...['main.cjs','preload.cjs','sidecar.cjs','verify-sequential-episodes.cjs'].map(n=>path.join(__dirname,n)),
 path.join(root,'desktop/dist/index.html'),
 ...fs.readdirSync(path.join(root,'desktop/dist/assets')).filter(n=>/\.(js|css)$/.test(n)).map(n=>path.join(root,'desktop/dist/assets',n))
].map(p=>[path.relative(root,p),sha(p)]));
const initialCodeHashes=codeHashes();
const report={runStartedUtc:new Date().toISOString(),sourceCommit:execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim(),schema:'resectionlab.hidden-recorded-sequential-episodes.v1',input,output,sourceSha256,codeHashes:initialCodeHashes,requireFrozenCode:process.env.SEQUENTIAL_EPISODE_REQUIRE_FROZEN==='1',windowEverVisible:false,requests:[],captures:[],alternatives:[],checks:[],passed:false};
let last=null,inflight=0,sidecar;
const moduleSidecar=require('./sidecar.cjs'),ProductionSidecar=moduleSidecar.Sidecar;
moduleSidecar.Sidecar=class extends ProductionSidecar{constructor(options){super(options);sidecar=this;}};
app.setPath('userData',path.join(output,'hidden-profile'));process.argv.push('--case',input,'--automation-hidden');
dialog.showSaveDialog=async()=>({canceled:false,filePath:saved});
const handle=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(name,handler)=>handle(name,async(...args)=>{
 const start=Date.now();inflight++;
 try{const value=await handler(...args);if(name==='research:surgicalRehearsal'){
  last=value;report.requests.push({command:args[1]?.command,branchId:value?.branchId,position:value?.position,stateId:value?.view?.stateId,playback:value?.playback?{frameIndex:value.playback.frameIndex,actionIndex:value.playback.actionIndex,observedTimeSeconds:value.playback.observedTimeSeconds,isActionBoundary:value.playback.isActionBoundary}:null,seconds:(Date.now()-start)/1000});}return value;
 }catch(error){report.requests.push({command:args[1]?.command,error:String(error),seconds:(Date.now()-start)/1000});throw error;}finally{inflight--;}
});
const delay=ms=>new Promise(r=>setTimeout(r,ms));
app.on('browser-window-created',(_event,window)=>{
 window.on('show',()=>{report.windowEverVisible=true;window.hide();});
 window.webContents.once('did-finish-load',async()=>{
  window.webContents.setBackgroundThrottling(false);window.setContentSize(1600,1000);
  const js=code=>window.webContents.executeJavaScript(`(()=>{${code}})()`,true);
  const until=async(fn,label,seconds=300)=>{const start=Date.now();while(Date.now()-start<seconds*1000){if(await fn())return;await delay(80);}throw Error('Timeout '+label);};
  const ready=()=>until(async()=>inflight===0&&await js(`return !!document.querySelector('.surgical-workflow')&&document.querySelector('.rl-viewer')?.dataset.surfacesPending==='0'&&!document.querySelector('.viewer-loading')&&![...document.querySelectorAll('.surgical-panel [role="status"]')].some(e=>/^(Computing instrument action|Preparing an exact recorded state|Preparing accepted display|Verifying saved source-bound rehearsal)/.test(e.textContent.trim()))`),'accepted native display');
  const click=async label=>{await js(`const b=[...document.querySelectorAll('button,input[type=checkbox]')].find(b=>(b.textContent.trim()===${JSON.stringify(label)}||b.getAttribute('aria-label')===${JSON.stringify(label)})&&b.checkVisibility());if(!b||b.matches(':disabled'))throw Error('Missing visible enabled '+${JSON.stringify(label)});b.click();`);await delay(120);};
  const stage=async label=>{await click(label+' stage');await ready();};
  const request=async(label,command)=>{const start=report.requests.length;await click(label);await until(()=>report.requests.slice(start).some(r=>r.command===command),'IPC '+command);await ready();assert(!report.requests.slice(start).find(r=>r.command===command)?.error);};
  const seek=async position=>{const start=report.requests.length;await js(`const e=document.querySelector('[aria-label="Surgical timeline"]');Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(e,${JSON.stringify(String(position))});e.dispatchEvent(new Event('input',{bubbles:true}));`);await until(()=>report.requests.slice(start).some(r=>r.command==='seek'),'actual native seek');await ready();assert.equal(last.position,position);};
  const capture=async(name,selector='.workflow-rehearsal')=>{await ready();await js(`document.querySelector(${JSON.stringify(selector)})?.scrollIntoView({block:'start'});`);await delay(150);assert(!window.isVisible());fs.writeFileSync(path.join(output,name+'.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());report.captures.push({name:name+'.png',branchId:last?.branchId,position:last?.position,stateId:last?.view?.stateId,playback:last?.playback??null});};
  const inspect=async()=>{last=await sidecar.request('surgicalRehearsal',{caseHash:last.caseHash,command:'inspect',expectedStateId:last.view.stateId,includeSequenceOutcome:true},300000);return {caseHash:last.caseHash,branchId:last.branchId,position:last.position,stateId:last.view.stateId,sceneId:last.view.scene.sceneId,branches:last.branches,sequenceOutcome:last.sequenceOutcome,historyHash:digest(last.view.history),metrics:last.view.metrics,removedHash:digest(last.view.removed),cutsHash:digest(last.view.cutEdges)};};

  try{
   await until(()=>js(`return !!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`),'source volume');
   await click('Rehearse');await ready();
   if(await js(`return [...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify saved replay'&&b.checkVisibility()&&!b.disabled)`))await request('Verify saved replay','inspect');
   const initial=await inspect();report.initial=initial;await click('Fit 3D view');report.backend=(await sidecar.request('ping')).numericalBackend;
   assert.equal(last.caseHash,expected.caseHash);assert.equal(last.view.scene.sceneId,expected.sourceSceneId);
   if(process.env.SEQUENTIAL_EPISODE_EXPECT_REPORT){const previous=JSON.parse(fs.readFileSync(process.env.SEQUENTIAL_EPISODE_EXPECT_REPORT,'utf8'));assert(previous.passed&&previous.engineTermination);assert.deepEqual(initial,previous.savedSnapshot,'Cold process preserves exact native branch, action position, history and physical outcomes');report.checks.push('Cold process restores exact saved source/branch/history/outcomes/cursor');}
   await stage('Compare');await until(()=>js(`return document.querySelectorAll('[data-sequential-branch]').length===5`),'five actual episode branches');
   assert(await js(`return document.querySelector('.sequential-episodes').innerText.includes('Earlier opening/access costs unknown')`));
   await capture('01-actual-episode-comparison','.sequential-episodes');
   for(const [i,row] of expected.episodes.entries()){
    await stage('Compare');const startSelect=report.requests.length;
    await js(`document.querySelector('[data-sequential-branch="${row.branchId}"]').click()`);
    await until(()=>report.requests.slice(startSelect).some(r=>r.command==='selectBranch'),'actual episode selection');await ready();
    assert.equal(last.branchId,row.branchId);assert.equal(last.position,0);assert.equal(last.view.stateId,expected.sourceStateId);
    await click('Center recorded target');
    const firstWork=row.attempts.find(a=>a.status!=='rejected'&&a.action.operation==='apply');assert(firstWork);
    await seek(firstWork.nativePositionBefore);await capture(`episode-${i+1}-work-start`,'.sequential-episodes');
    const frameStart=report.requests.length;await click('Play replay');await until(()=>report.requests.slice(frameStart).some(r=>r.command==='seekFrame'&&r.playback&&!r.playback.isActionBoundary),'actual native interior frame');await click('Pause replay');await ready();
    await until(()=>js(`return document.querySelector('[data-recorded-selected-branch]')?.dataset.recordedSelectedBranch===${JSON.stringify(row.branchId)}`),'original generated memberships and accepted path overlay');
    const actionStart=last.playback.frames[0].observedTimeSeconds,actionEnd=actionStart+last.playback.actionDurationSeconds;
    const lateFrame=last.playback.frames.filter(f=>f.kind==='microstep'&&f.observedTimeSeconds<actionEnd-1e-7).at(-1);assert(lateFrame);
    const lateStart=report.requests.length;await js(`const e=document.querySelector('[aria-label="Observed action frames"]');Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(e,${JSON.stringify(String(lateFrame.frameIndex))});e.dispatchEvent(new Event('input',{bubbles:true}));`);
    await until(()=>report.requests.slice(lateStart).some(r=>r.command==='seekFrame'&&r.playback?.frameIndex===lateFrame.frameIndex),'late actual interior frame');await ready();assert(last.playback&&!last.playback.isActionBoundary&&last.playback.actionElapsedSeconds>last.playback.actionDurationSeconds*.5&&last.playback.actionElapsedSeconds<last.playback.actionDurationSeconds-1e-7);
    await capture(`episode-${i+1}-interior`,'.sequential-episodes');
    if(last.playback)await click('Return to recorded boundary');await ready();await seek(row.nativeActions);
    assert.equal(last.view.stateId,row.finalStateId);const endpoint=await inspect();
    assert.deepEqual(last.view.history.map(a=>a.action.operation),row.attempts.filter(a=>a.status!=='rejected').map(a=>a.action.operation));
    const ui=await js(`return {branch:document.querySelector('.sequential-episodes')?.dataset.selectedSequentialBranch,text:document.querySelector('.sequential-episodes')?.innerText,actualRequests:document.querySelectorAll('[data-sequential-attempt]').length,refusals:document.querySelectorAll('[data-request-status="rejected"]').length}`);
    assert.equal(ui.branch,row.branchId);assert.equal(ui.actualRequests,row.attempts.length);assert.equal(ui.refusals,row.attempts.filter(a=>a.status==='rejected').length);assert(ui.text.includes('final debris 0.00 mm³ and fluid 0.00 mm³'));
    await capture(`episode-${i+1}-endpoint`,'.sequential-episodes');
    if(ui.refusals){await js(`document.querySelector('.sequential-attempts').open=true`);await capture(`episode-${i+1}-actual-refusals`,'.sequential-attempts');}
    report.alternatives.push({branchId:row.branchId,method:row.method,seed:row.seed,checkpoint:row.checkpoint,endpoint,targetMm3:row.progress.originalTargetTreatmentMm3,residualCells:row.progress.residualOriginalTargetCells,actualRequests:ui.actualRequests,actualRefusals:ui.refusals,replayFrames:report.requests.slice(frameStart).filter(r=>r.command==='seekFrame')});
   }
   report.checks.push('Five original learned/baseline/control branches selected through real UI; accepted instrument motions and interior frames; actual refusals separate from native frames');
   await click('Save');await until(()=>fs.existsSync(saved),'native Save');await ready();report.saved=saved;report.savedSha256=sha(saved);report.savedSnapshot=await inspect();
   assert.equal(sha(input),sourceSha256);assert(!window.isVisible()&&!report.windowEverVisible);assert.equal(report.requests.filter(r=>r.command==='plan'||r.command==='step').length,0,'Recorded replay cannot trigger new planning or execution');
   report.codeStable=digest(codeHashes())===digest(initialCodeHashes);if(report.requireFrozenCode)assert(report.codeStable,'Source/build changed during frozen hidden verification');
   report.checks.push('Real Save preserves all5original episodes; source unchanged; no new policy inference/planning/physical editing; hidden throughout');report.passed=true;
  }catch(error){report.error=error.stack;report.body=await js('return document.body.innerText').catch(()=>null);fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());console.error(error);}
  try{await sidecar?.stop();report.engineTermination={exitCode:sidecar?.child.exitCode,signal:sidecar?.child.signalCode};assert(report.engineTermination.exitCode!==null||report.engineTermination.signal!==null);}catch(error){report.passed=false;report.terminationError=String(error);}
  report.runFinishedUtc=new Date().toISOString();fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,error:report.error,saved:report.saved}));app.exit(report.passed?0:1);
 });
});
require('./main.cjs');
app.on('will-quit',()=>{if(!fs.existsSync(path.join(output,'report.json'))){report.error??='Exited before hidden verification finished';fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));}});
