'use strict';
// Real Case → Plan → Compare → Rehearse, native Save and cold process reopen.
// No show(), focus(), visible capture, synthetic IPC result or planner stub.
const {app,ipcMain,dialog,BrowserWindow}=require('electron');
let fatal=false;
const failFatal=error=>{if(!fatal){fatal=true;console.error('HIDDEN_CRANIAL_FATAL '+String(error));}app.exit(1);process.exit(1);};
process.prependListener('uncaughtException',failFatal);process.prependListener('unhandledRejection',failFatal);
dialog.showErrorBox=(title,body)=>failFatal(`${title}: ${body}`);
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..'),base=path.join(root,'testcases');
const input=fs.realpathSync(process.env.CRANIAL_DEMO_CASE||''),output=path.resolve(process.env.CRANIAL_DEMO_OUTPUT||'');
assert(input.startsWith(base+path.sep)&&output.startsWith(base+path.sep));assert(!fs.existsSync(output),'Preserve earlier evidence');fs.mkdirSync(output,{recursive:true});
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const canonical=value=>JSON.stringify(value,(_key,v)=>v&&typeof v==='object'&&!Array.isArray(v)?Object.fromEntries(Object.keys(v).sort().map(k=>[k,v[k]])):v);
const digest=value=>crypto.createHash('sha256').update(canonical(value)).digest('hex');
const saved=path.join(output,'desktop-saved.ressectionlab'),sourceSha256=sha(input),minimumAlternatives=Number(process.env.CRANIAL_DEMO_MIN_ALTERNATIVES||2);assert([1,2,3].includes(minimumAlternatives));
const codeHashes=()=>Object.fromEntries([
 ...['surgical_engine.py','surgical_scene.py','surgical_procedure_policy.py','surgical_planning_admission.py','surgical_consequences.py','surgical_cranial_planning.py','surgical_cranial_geometry.py','surgical_sequential_task.py','surgical_rl_env.py','surgical_bridge.py','surgical_sequence_outcomes.py'].map(n=>path.join(root,'src/resectionlab',n)),
 ...['main.cjs','preload.cjs','sidecar.cjs','verify-sequential-cranial.cjs'].map(n=>path.join(__dirname,n)),
 path.join(root,'desktop/dist/index.html'),
 ...fs.readdirSync(path.join(root,'desktop/dist/assets')).filter(n=>/\.(js|css)$/.test(n)).map(n=>path.join(root,'desktop/dist/assets',n))
].map(p=>[path.relative(root,p),sha(p)]));
const initialCodeHashes=codeHashes();
const report={schema:'resectionlab.hidden-sequential-cranial.v1',input,output,sourceSha256,codeHashes:initialCodeHashes,requireFrozenCode:process.env.CRANIAL_DEMO_REQUIRE_FROZEN==='1',windowEverVisible:false,requests:[],captures:[],alternatives:[],checks:[],passed:false};
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
  const inspect=async()=>{last=await sidecar.request('surgicalRehearsal',{caseHash:last.caseHash,command:'inspect',expectedStateId:last.view.stateId,includeSequenceOutcome:true},300000);return {caseHash:last.caseHash,branchId:last.branchId,position:last.position,stateId:last.view.stateId,sceneId:last.view.scene.sceneId,branches:last.branches,comparison:last.cranialPlanning,sequenceOutcome:last.sequenceOutcome,historyHash:digest(last.view.history),metrics:last.view.metrics,removedHash:digest(last.view.removed),cutsHash:digest(last.view.cutEdges)};};
  const label=row=>{const p=row.recordedGeometry.paths.filter(p=>p.instrument==='cusa'&&p.operation==='apply').at(-1);return `${row.label} · target path ${p.pointsRasMm.at(-1).map(n=>n.toFixed(1)).join(' / ')} mm`;};
  try{
   await until(()=>js(`return !!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`),'source volume');
   await click('Rehearse');await ready();
   if(await js(`return [...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify saved replay'&&b.checkVisibility()&&!b.disabled)`))await request('Verify saved replay','inspect');
   const initial=await inspect();report.initial=initial;await click('Fit 3D view');report.backend=(await sidecar.request('ping')).numericalBackend;
   assert(last.cranialPlanning);assert.equal(last.cranialPlanning.schema,'resectionlab.cranial-strategy-comparison.v2');
   const comparison=last.cranialPlanning;assert(comparison.alternatives.length>=minimumAlternatives,'Required checked alternatives missing');
   report.comparisonSeal=comparison.comparisonSeal;report.policyHash=comparison.procedurePolicy.policyHash;
   if(process.env.CRANIAL_DEMO_EXPECT_REPORT){const previous=JSON.parse(fs.readFileSync(process.env.CRANIAL_DEMO_EXPECT_REPORT,'utf8'));assert(previous.passed&&previous.engineTermination);assert.deepEqual(initial,previous.savedSnapshot,'Cold process must preserve the exact saved source/policy/history/outcomes/cursor');report.checks.push('Cold process restores exact saved comparison, policy, native histories, residual, costs and STOP reason');}
   await stage('Plan');assert(await js(`return document.querySelector('[aria-label="Active procedure policy"]')?.checkVisibility()`));await capture('01-active-policy','.cranial-strategies');
   await stage('Compare');assert.equal(await js(`return document.querySelectorAll('.cranial-strategies .workflow-alternative').length`),comparison.alternatives.length);assert(await js(`return document.querySelector('[aria-label="Recorded route and source anatomy legend"]')?.checkVisibility()`));await click('Center generated motor anatomy');await capture('02-checked-alternatives','.cranial-strategies');
   const endpoints=new Set();
   for(const [i,row] of comparison.alternatives.entries()){
    const geometry=row.recordedGeometry;assert(geometry&&geometry.caseHash===last.caseHash&&geometry.policyHash===report.policyHash);
    const motor=geometry.structures.find(s=>s.id==='motor-cortex'),cst=geometry.structures.find(s=>s.id==='corticospinal-tract');assert(motor&&cst&&motor.sourceCells.some(c=>cst.sourceCells.includes(c)),'Independent memberships preserve overlap');
    assert.equal(geometry.cleanup.debrisPresentMm3,0);assert.equal(geometry.cleanup.fluidPresentMm3,0);assert(Math.abs(geometry.cleanup.debrisRemovedMm3-236)<1e-8);
    endpoints.add(JSON.stringify(geometry.paths.filter(p=>p.instrument==='cusa'&&p.operation==='apply').at(-1).pointsRasMm.at(-1)));
    await stage('Compare');await request('Inspect path '+(i+1),'selectBranch');assert.equal(last.branchId,row.branchId);assert.equal(last.view.stateId,row.finalStateId);assert.equal(last.position,row.nativeActions);
    await until(()=>js(`return document.querySelector('.rl-viewer-anatomy')?.dataset.recordedSelectedBranch===${JSON.stringify(row.branchId)}||[...document.querySelectorAll('[data-recorded-selected-branch]')].some(e=>e.dataset.recordedSelectedBranch===${JSON.stringify(row.branchId)})`),'source-bound route overlay');
    await capture(`plan-${i+1}-endpoint`);
    const firstWork=geometry.paths.find(p=>p.instrument==='cusa'&&p.operation==='apply').actionIndex;
    await seek(firstWork);await capture(`plan-${i+1}-work-start`);
    const start=report.requests.length;await click('Play replay');await until(()=>report.requests.slice(start).some(r=>r.command==='seekFrame'&&r.playback&&!r.playback.isActionBoundary),'actual interior replay frame');await click('Pause replay');await ready();await capture(`plan-${i+1}-physical-replay-frame`);
    const frames=report.requests.slice(start).filter(r=>r.command==='seekFrame');assert(frames.length);report.alternatives.push({branchId:row.branchId,geometrySeal:geometry.geometrySeal,endpoint:row.finalStateId,targetMm3:row.goalAssessment.originalTargetTreatmentMm3,residualMm3:row.goalAssessment.residualOriginalTargetMm3,ledger:row.ledger,stopReason:row.stopReason,cleanup:geometry.cleanup,replayFrames:frames});
    await seek(row.nativeActions);assert.equal(last.view.stateId,row.finalStateId);
   }
   assert(endpoints.size>=minimumAlternatives,'Paths must differ in RAS geometry');report.checks.push('Both checked alternatives are selected through real cards and replay their physical native actions and interior frames; distinct RAS routes and overlapped generated motor memberships visible');
   await stage('Compare');await capture('03-final-comparison','.cranial-strategies');if(comparison.failures.length){await capture('04-checked-refusals','[aria-label="Strategy refusals"]');report.refusalsDisplayed=await js(`return document.querySelector('[aria-label="Strategy refusals"]').innerText`);}await stage('Rehearse/edit');
   await click('Save');await until(()=>fs.existsSync(saved),'native Save');await ready();report.saved=saved;report.savedSha256=sha(saved);report.savedSnapshot=await inspect();
   assert.equal(sha(input),sourceSha256);assert(!window.isVisible()&&!report.windowEverVisible);assert.equal(report.requests.filter(r=>r.command==='plan').length,0,'Recorded replay cannot trigger new planning');
   report.codeStable=digest(codeHashes())===digest(initialCodeHashes);if(report.requireFrozenCode)assert(report.codeStable,'Source/build changed during frozen hidden verification');
   report.checks.push('Real Save preserves both native alternatives without new planning; source archive unchanged; automation window never shown');report.passed=true;
  }catch(error){report.error=error.stack;report.body=await js('return document.body.innerText').catch(()=>null);fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());console.error(error);}
  try{await sidecar?.stop();report.engineTermination={exitCode:sidecar?.child.exitCode,signal:sidecar?.child.signalCode};assert(report.engineTermination.exitCode!==null||report.engineTermination.signal!==null);}catch(error){report.passed=false;report.terminationError=String(error);}
  fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,error:report.error,saved:report.saved}));app.exit(report.passed?0:1);
 });
});
require('./main.cjs');
app.on('will-quit',()=>{if(!fs.existsSync(path.join(output,'report.json'))){report.error??='Exited before hidden verification finished';fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));}});
