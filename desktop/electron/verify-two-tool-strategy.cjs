'use strict';
// Actual hidden UI execution; no primary-display show/focus calls.
const {app,ipcMain,dialog}=require('electron');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..'),base=path.join(root,'testcases');
const input=fs.realpathSync(process.env.STRATEGY_CASE||''),output=path.resolve(process.env.STRATEGY_OUTPUT||'');
for(const p of [input,output])assert(p.startsWith(base+path.sep));assert(!fs.existsSync(output),'Preserve previous evidence');fs.mkdirSync(output,{recursive:true});
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex'),original=sha(input);
const report={input,sourceSha256:original,output,expected:process.env.STRATEGY_EXPECT||'paired',requests:[],interactions:[],timingScope:'IPC backend separately from click-to-accepted-display; latter includes UI dispatch/polling and surface preparation',checks:[],windowEverVisible:false};
const saved=path.join(output,'strategies-saved.ressectionlab');let last=null;
app.setPath('userData',path.join(output,'hidden-profile'));process.argv.push('--case',input,'--automation-hidden');
dialog.showSaveDialog=async()=>({canceled:false,filePath:saved});
const handle=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(name,handler)=>handle(name,async(...args)=>{const start=Date.now(),value=await handler(...args);if(name==='research:surgicalRehearsal'){last=value;report.requests.push({command:args[1]?.command,seconds:(Date.now()-start)/1000,stateId:value?.view?.stateId,position:value?.position,branchId:value?.branchId,decision:value?.strategyPlanning?.publicPlan?.decision,preview:value?.approachPreview?.candidateId,phase:value?.approachPreview?.phaseId});}return value;});
const delay=ms=>new Promise(r=>setTimeout(r,ms));
app.on('browser-window-created',(_event,window)=>{
 window.on('show',()=>{report.windowEverVisible=true;window.hide();});
 window.webContents.once('did-finish-load',async()=>{
  window.webContents.setBackgroundThrottling(false);window.setContentSize(1440,900);
  const js=async code=>{const x=await window.webContents.executeJavaScript(`try{${code}}catch(error){({automationError:String(error)})}`,true);if(x?.automationError)throw Error(x.automationError);return x;};
  const until=async(fn,label,seconds=120)=>{const t=Date.now();while(Date.now()-t<seconds*1000){if(await fn())return;const error=await js(`document.querySelector('.surgical-panel .surgical-error')?.textContent`);if(error)throw Error(error);await delay(80);}throw Error('Timeout '+label);};
  const ready=()=>until(()=>js(`!!document.querySelector('[data-surgical-state-id]')&&document.querySelector('.rl-viewer')?.dataset.surfacesPending==='0'&&!document.querySelector('.viewer-loading')&&![...document.querySelectorAll('.surgical-panel [role="status"]')].some(n=>/^(Computing instrument action|Preparing an exact recorded state|Preparing accepted display|Verifying saved source-bound rehearsal)/.test(n.textContent.trim()))`),'accepted display');
  const click=async(label,selector='button')=>{await js(`(()=>{const b=[...document.querySelectorAll(${JSON.stringify(selector)})].find(b=>b.textContent.trim()===${JSON.stringify(label)}||b.getAttribute('aria-label')===${JSON.stringify(label)});if(!b||b.matches(':disabled'))throw Error('Missing enabled '+${JSON.stringify(label)});for(let p=b.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS'&&!(b.tagName==='SUMMARY'&&p===b.parentElement))p.open=true;b.click()})()`);await delay(100);};
  const set=async(label,value)=>{await js(`(()=>{const e=document.querySelector('[aria-label='+${JSON.stringify(JSON.stringify(label))}+']');if(!e)throw Error('Missing input');Object.getOwnPropertyDescriptor(e.tagName==='SELECT'?HTMLSelectElement.prototype:HTMLInputElement.prototype,'value').set.call(e,${JSON.stringify(String(value))});e.dispatchEvent(new Event(e.tagName==='SELECT'?'change':'input',{bubbles:true}));})()`);await delay(150);};
  const capture=async name=>{await ready();await js(`(()=>{const e=document.querySelector('.two-tool-planning');if(e){e.open=true;e.scrollIntoView({block:'start'})}})()`);await delay(150);assert(!window.isVisible());fs.writeFileSync(path.join(output,name+'.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());};
  const timing=(name,command,start,count)=>{const request=report.requests.slice(count).find(x=>x.command===command);assert(request);report.interactions.push({name,command,backendSeconds:request.seconds,clickToAcceptedDisplaySeconds:(Date.now()-start)/1000});};
  const generate=async name=>{const count=report.requests.length,start=Date.now();await click('Compare executable strategies');await until(()=>report.requests.slice(count).some(x=>x.command==='generateStrategies'),'strategy result');await ready();timing(name,'generateStrategies',start,count);const study=last.strategyPlanning;assert(study?.publicPlan);assert.equal(last.view.stateId,report.baselineStateId);fs.writeFileSync(path.join(output,name+'.json'),JSON.stringify(study,null,2));return study;};
  const preview=async(candidate,label,phase='operate')=>{await click('Select '+label);await set('Strategy preview checkpoint',phase);const count=report.requests.length,start=Date.now();await click('Preview strategy checkpoint');await until(()=>report.requests.slice(count).some(x=>x.command==='previewStrategy'),'strategy preview');await ready();timing(candidate+'-'+phase,'previewStrategy',start,count);assert.equal(last.approachPreview?.candidateId,candidate);assert.equal(last.approachPreview?.phaseId,phase);assert(last.approachPreview.receipt.nativeMatchesPublicPrediction&&last.approachPreview.receipt.withdrawalComplete);assert.equal(last.view.stateId,report.baselineStateId);return last.approachPreview;};
  try{
   await until(()=>js(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`),'case load');await click('Rehearse');await ready();
   if(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify saved replay')`)){await click('Verify saved replay');await ready();}
   report.baselineStateId=last.view.stateId;report.caseHash=last.caseHash;report.sourceSceneId=last.view.scene.sceneId;
   if(process.env.STRATEGY_REOPEN_ONLY==='1'){
    assert.equal(last.view.stateId,process.env.STRATEGY_EXPECTED_STATE);assert(last.strategyPlanning?.publicPlan.publicPlanSeal);report.checks.push('Fresh hidden process verifies saved comparison and exact native state');
   }else if(report.expected==='missing-target'){
    assert.equal(last.strategyPlanningAvailability.available,false);assert.match(last.strategyPlanningAvailability.reason,/tumor target is required/);
    assert.equal(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Compare executable strategies')`),false);await capture('missing-target');
   }else{
    assert(last.strategyPlanningAvailability.available);await click('Compare two-tool strategies','summary');await click('Acknowledge two-tool research scope','input[type=checkbox]');
    let study=await generate('default-comparison');report.defaultPublicPlanSeal=study.publicPlan.publicPlanSeal;
    if(report.expected==='paired'){
     assert.deepEqual(study.publicPlan.alternatives.map(x=>x.id),['short','long','stop']);assert.equal(study.publicPlan.decision.selectedId,'long');
     const [short,long]=study.publicPlan.alternatives;assert.equal(short.summary.additionalTargetMm3,0);assert.equal(short.summary.cusaPendingMm3,.5);assert.equal(long.summary.additionalTargetMm3,2);
     assert.equal(short.reassessment.probeStatus,'rejected');assert.equal(long.reassessment.probeStatus,'accepted');
     let p=await preview('short','Short contact');assert(p.view.slots.right.active);await capture('short-contact-preview');
     p=await preview('long','Longer contact');assert(p.view.slots.right.active);await capture('long-contact-preview');
     await preview('long','Longer contact','reassess');await capture('long-reassessment-preview');
     const count=report.requests.length;await click('Install strategy branch');await until(()=>report.requests.slice(count).some(x=>x.command==='applyStrategy'),'installed branch');await ready();
     assert.notEqual(last.view.stateId,report.baselineStateId);assert(Object.values(last.view.slots).every(s=>!s.active&&!s.inserted));report.longBranch=last.branchId;report.longStateId=last.view.stateId;await capture('installed-long-branch');
     await click('Open strategy checkpoint');await ready();assert.equal(last.view.stateId,report.baselineStateId);
     if(!await js(`document.querySelector('[aria-label="Acknowledge two-tool research scope"]').checked`))await click('Acknowledge two-tool research scope','input[type=checkbox]');
     await set('Short contact seconds',.75);study=await generate('modified-duration-comparison');assert.equal(study.publicPlan.decision.selectedId,'short');assert.equal(study.publicPlan.alternatives[0].summary.additionalTargetMm3,1);
     await preview('short','Short contact');await click('Install strategy branch');await ready();report.shortBranch=last.branchId;report.shortStateId=last.view.stateId;assert.notEqual(report.shortStateId,report.longStateId);
     await click('Open strategy checkpoint');await ready();assert.equal(last.view.stateId,report.baselineStateId);
     if(!await js(`document.querySelector('[aria-label="Acknowledge two-tool research scope"]').checked`))await click('Acknowledge two-tool research scope','input[type=checkbox]');
     await click('Target and shared entry patch','summary');await set('Shared entry patch radius',.5);
     const restricted=await generate('restricted-shared-entry');assert.equal(restricted.publicPlan.decision.operation,'STOP');assert.equal(restricted.publicPlan.chosenPair,null);await capture('restricted-entry-stop');
     await set('Shared entry patch radius',study.publicPlan.constraints.geometry.entry.radiusMm);
     report.checks.push('Actual controls change contact duration, removed target and preferred complete sequence; two distinct branches installed without new non-target removal');
    }else{
     assert.equal(study.publicPlan.decision.operation,'STOP');assert.equal(study.publicPlan.alternatives.length,1);assert.equal(study.publicPlan.decision.reason,'NO_COMPLETE_TOOL_ACCESS');await capture('no-access-stop');
    }
    await click('Require functional evidence for strategies','input[type=checkbox]');study=await generate('functional-evidence-stop');assert.equal(study.publicPlan.decision.operation,'STOP');assert.equal(study.publicPlan.decision.reason,'REQUIRED_EVIDENCE_UNAVAILABLE');assert.equal(study.publicPlan.alternatives.length,1);await capture('functional-stop');
    await click('Save');await until(()=>fs.existsSync(saved),'native save');report.saved=saved;report.savedStateId=last.view.stateId;
   }
   assert.equal(sha(input),original);assert(!report.windowEverVisible&&!window.isVisible());report.sourcePreserved=true;report.passed=true;
  }catch(error){report.error=String(error);report.body=await js('document.body.innerText').catch(()=>null);fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());console.error(error);}
  fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,error:report.error,saved:report.saved}));app.exit(report.passed?0:1);
 });
});
require('./main.cjs');
