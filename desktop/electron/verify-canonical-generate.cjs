'use strict';
// Real Generate, branch selection, Save and cold reopen; never show/focus.
const {app,ipcMain,dialog}=require('electron');
// Bootstrap failures must also stay hidden, including invalid input paths.
let bootstrapFatal=false;
const failBootstrap=error=>{if(!bootstrapFatal){bootstrapFatal=true;console.error('HIDDEN_CANONICAL_FATAL '+String(error));}app.exit(1);process.exit(1);};
process.prependListener('uncaughtException',failBootstrap);process.prependListener('unhandledRejection',failBootstrap);
dialog.showErrorBox=(title,body)=>failBootstrap(title+': '+body);
process.argv.push('--automation-hidden');
if(process.platform==='darwin')app.setActivationPolicy('accessory');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..'),base=path.join(root,'testcases');
const input=fs.realpathSync(process.env.CANONICAL_CASE||''),output=path.resolve(process.env.CANONICAL_OUTPUT||'');
assert(input.startsWith(base+path.sep)&&output.startsWith(base+path.sep)&&!fs.existsSync(output));fs.mkdirSync(output,{recursive:true});
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const hashes=()=>Object.fromEntries([
 ...['surgical_cranial_planning','surgical_joint_cranial_search','surgical_joint_cranial_forest','surgical_consequences','surgical_sequential_task','surgical_bridge','surgical_scene'].map(n=>path.join(root,'src/resectionlab',n+'.py')),
 __filename,path.join(root,'desktop/dist/index.html'),...fs.readdirSync(path.join(root,'desktop/dist/assets')).filter(n=>/\.(js|css)$/.test(n)).map(n=>path.join(root,'desktop/dist/assets',n))
].map(p=>[path.relative(root,p),sha(p)]));
const initialHashes=hashes(),saved=path.join(output,'desktop-saved.ressectionlab');
const report={input,sourceSha256:sha(input),sourceHashes:initialHashes,cold:process.env.CANONICAL_COLD==='1',passed:false,windowEverVisible:false,requests:[],captures:[],consoleMessages:[]};
const prior=report.cold?JSON.parse(fs.readFileSync(process.env.CANONICAL_EXPECT_REPORT,'utf8')):null;
if(prior){assert(prior.passed);assert.equal(input,fs.realpathSync(prior.saved));assert.equal(report.sourceSha256,prior.savedSha256);}
let last=null,inflight=0,sidecar;
const fatal=error=>{report.passed=false;report.error=String(error);fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));Promise.resolve(sidecar?.stop()).finally(()=>app.exit(1));};
process.removeListener('uncaughtException',failBootstrap);process.removeListener('unhandledRejection',failBootstrap);
process.prependListener('uncaughtException',fatal);process.prependListener('unhandledRejection',fatal);
dialog.showErrorBox=(title,body)=>fatal(title+': '+body);
const moduleSidecar=require('./sidecar.cjs'),Original=moduleSidecar.Sidecar;
moduleSidecar.Sidecar=class extends Original{constructor(options){super(options);sidecar=this;}};
app.setPath('userData',path.join(output,'hidden-profile'));process.argv.push('--case',input);
dialog.showSaveDialog=async()=>({canceled:false,filePath:saved});
const handle=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(name,handler)=>handle(name,async(...args)=>{inflight++;try{const value=await handler(...args);if(name==='research:surgicalRehearsal'){last=value;report.requests.push({command:args[1]?.command,opponent:args[1]?.opponent,branchId:value?.branchId,position:value?.position,stateId:value?.view?.stateId});}return value;}finally{inflight--;}});
const delay=ms=>new Promise(r=>setTimeout(r,ms));
app.on('browser-window-created',(_event,window)=>{
 window.on('show',()=>{report.windowEverVisible=true;window.hide();});
 window.webContents.on('console-message',(_event,level,message,line,sourceId)=>{report.consoleMessages.push({level,message,line,sourceId});});
 window.webContents.once('did-finish-load',async()=>{
  window.webContents.setBackgroundThrottling(false);window.setContentSize(1600,1000);
  const js=code=>window.webContents.executeJavaScript(`(()=>{${code}})()`,true);
  const until=async(fn,label,seconds=320)=>{const end=Date.now()+seconds*1000;while(Date.now()<end){if(await fn())return;await delay(100);}throw Error('Timeout '+label);};
  const ready=()=>until(async()=>inflight===0&&last&&await js(`const p=document.querySelector('.surgical-panel');return !!document.querySelector('.surgical-workflow')&&p?.dataset.stateId===${JSON.stringify(last.view?.stateId)}&&p?.dataset.recordedPosition===${JSON.stringify(String(last.position))}&&!document.querySelector('.viewer-loading')&&document.querySelector('.rl-viewer')?.dataset.surfacesPending==='0'&&![...document.querySelectorAll('.surgical-panel [role="status"]')].some(e=>/^(Computing instrument action|Preparing an exact recorded state|Preparing accepted display|Verifying saved source-bound rehearsal)/.test(e.textContent.trim()))`),'native display');
  const click=async label=>{await js(`const b=[...document.querySelectorAll('button,input')].find(e=>(e.textContent.trim()===${JSON.stringify(label)}||e.getAttribute('aria-label')===${JSON.stringify(label)})&&e.checkVisibility());if(!b||b.disabled)throw Error('Missing enabled '+${JSON.stringify(label)});b.click();`);await delay(150);};
  const capture=async name=>{await ready();assert(!window.isVisible());fs.writeFileSync(path.join(output,name+'.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());report.captures.push(name+'.png');};
  const inspect=async()=>{last=await sidecar.request('surgicalRehearsal',{caseHash:last.caseHash,command:'inspect',expectedStateId:last.view.stateId,includeSequenceOutcome:true},300000);return {caseHash:last.caseHash,sceneId:last.view.scene.sceneId,branchId:last.branchId,position:last.position,stateId:last.view.stateId};};
  // Verification body follows; process cleanup is shared by success/failure.
  try{
   await until(()=>js(`return !!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`),'source');await click('Rehearse');await ready();
   if(await js(`return [...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify saved replay'&&b.checkVisibility()&&!b.disabled)`)){await click('Verify saved replay');await ready();}
   report.initialCursor=await inspect();if(prior)assert.deepEqual(report.initialCursor,prior.savedCursor,'Cold process must preserve the actual saved branch, position and state');
   await click('Plan stage');await ready();
   if(!report.cold){await click('Acknowledge cranial research context');await click('Generate strategies');await until(()=>last?.cranialPlanning?.schema==='resectionlab.cranial-strategy-comparison.v3'&&inflight===0,'real Generate');await ready();}
   const c=last?.cranialPlanning;assert(c&&c.schema==='resectionlab.cranial-strategy-comparison.v3');
   assert(c.endpointContract&&c.selectionEvidence&&c.alternatives.length);assert(c.cost.wallSeconds<c.cost.maxWallSeconds);assert(c.cost.physicalCalls<=c.cost.maxPhysicalCalls);
   for(const row of c.alternatives){assert.equal(row.completedFrozenTask,row.endpointAssessment.completedTask);if(row.recommendationEligible)assert(row.validStop&&!row.endpointAssessment.truncated&&row.endpointAssessment.cleanupComplete);}
   if(prior){assert.equal(c.comparisonSeal,prior.comparisonSeal);assert.deepEqual(c.selectionEvidence,prior.selectionEvidence);}
   await click('Compare stage');await ready();assert.equal(await js(`return document.querySelectorAll('.cranial-strategies .workflow-alternative').length`),c.alternatives.length);assert(!await js(`return [...document.querySelectorAll('[role="alert"]')].some(e=>e.textContent.includes('could not be bound'))`));await capture('01-generated-comparison');
   const index=c.alternatives.findIndex(r=>r.goalAssessment.creditedOriginalTargetCells>0);assert(index>=0);const selected=c.alternatives[index];await click('Inspect path '+(index+1));await until(()=>last?.branchId===selected.branchId&&inflight===0,'selected native branch');await ready();assert.equal(last.view.stateId,selected.finalStateId);assert.equal(last.position,selected.nativeActions);await capture('02-selected-endpoint');
   report.comparisonSeal=c.comparisonSeal;report.selectionEvidence=c.selectionEvidence;report.decision=c.decision;report.alternatives=c.alternatives.map(r=>({branchId:r.branchId,endpoint:r.endpointAssessment,recommendationEligible:r.recommendationEligible}));
   await click('Save');await until(()=>fs.existsSync(saved),'native Save');report.saved=saved;report.savedSha256=sha(saved);report.savedCursor=await inspect();assert.equal(report.savedCursor.stateId,selected.finalStateId);assert.equal(report.savedCursor.position,selected.nativeActions);
   assert.equal(sha(input),report.sourceSha256);report.codeStable=JSON.stringify(hashes())===JSON.stringify(initialHashes);assert(report.codeStable);assert(!report.windowEverVisible&&!window.isVisible());
   assert.equal(report.requests.filter(r=>r.command==='plan').length,report.cold?0:1);report.passed=true;
  }catch(error){report.error=error.stack;report.body=await js('return document.body.innerText').catch(()=>null);await capture('failure').catch(()=>{});}
  try{await sidecar?.stop();report.engineTermination={exitCode:sidecar?.child.exitCode,signal:sidecar?.child.signalCode};assert(report.engineTermination.exitCode!==null||report.engineTermination.signal!==null);}catch(error){report.passed=false;report.terminationError=String(error);}
  fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,error:report.error,saved:report.saved}));app.exit(report.passed?0:1);
 });
});
require('./main.cjs');
