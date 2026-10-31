'use strict';
// Real renderer/backend workflow. No primary-display show/focus calls.
const {app,ipcMain,dialog}=require('electron');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..'),base=path.join(root,'testcases');
const input=fs.realpathSync(process.env.PREPARATION_CASE||''),output=path.resolve(process.env.PREPARATION_OUTPUT||'');
for(const p of [input,output])assert(p.startsWith(base+path.sep));assert(!fs.existsSync(output));fs.mkdirSync(output,{recursive:true});
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex'),original=sha(input);
const report={input,sourceSha256:original,output,requests:[],interactions:[],checks:[],windowEverVisible:false};let last=null;
const saved=path.join(output,'preparation-studies-saved.ressectionlab');
app.setPath('userData',path.join(output,'hidden-profile'));process.argv.push('--case',input,'--automation-hidden');
dialog.showSaveDialog=async()=>({canceled:false,filePath:saved});
const handle=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(name,handler)=>handle(name,async(...args)=>{const start=Date.now(),value=await handler(...args);if(name==='research:surgicalRehearsal'){
 last=value;report.requests.push({command:args[1]?.command,seconds:(Date.now()-start)/1000,stateId:value?.view?.stateId,
 branchId:value?.branchId,position:value?.position,preparationRevision:value?.preparationPlanning?.revision,
 preparationSeal:value?.preparationPlanning?.reportSeal,admission:value?.interactivePlanningAvailability?.admission?.admissionSeal});}return value;});
const delay=ms=>new Promise(r=>setTimeout(r,ms));
app.on('browser-window-created',(_event,window)=>{
 window.on('show',()=>{report.windowEverVisible=true;window.hide();});
 window.webContents.once('did-finish-load',async()=>{
  window.webContents.setBackgroundThrottling(false);window.setContentSize(1440,900);
  const js=async code=>{const x=await window.webContents.executeJavaScript(`try{${code}}catch(error){({automationError:String(error)})}`,true);if(x?.automationError)throw Error(x.automationError);return x;};
  const until=async(fn,label,seconds=240)=>{const start=Date.now();while(Date.now()-start<seconds*1000){if(await fn())return;const error=await js(`document.querySelector('.surgical-panel .surgical-error')?.textContent`);if(error)throw Error(error);await delay(80);}throw Error('Observation timeout '+label);};
  const ready=()=>until(()=>js(`!!document.querySelector('[data-surgical-state-id]')&&document.querySelector('.rl-viewer')?.dataset.surfacesPending==='0'&&!document.querySelector('.viewer-loading')&&![...document.querySelectorAll('.surgical-panel [role="status"]')].some(n=>/^(Computing instrument action|Preparing an exact recorded state|Preparing accepted display|Verifying saved source-bound rehearsal)/.test(n.textContent.trim()))`),'accepted native display');
  const click=async(label,selector='button')=>{await js(`(()=>{const b=[...document.querySelectorAll(${JSON.stringify(selector)})].find(b=>b.textContent.trim()===${JSON.stringify(label)}||b.getAttribute('aria-label')===${JSON.stringify(label)});if(!b||b.matches(':disabled'))throw Error('Missing enabled '+${JSON.stringify(label)});for(let p=b.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS'&&!(b.tagName==='SUMMARY'&&p===b.parentElement))p.open=true;b.click()})()`);await delay(100);};
  const set=async(label,value)=>{await js(`(()=>{const e=document.querySelector('[aria-label='+${JSON.stringify(JSON.stringify(label))}+']');if(!e)throw Error('Missing input '+${JSON.stringify(label)});Object.getOwnPropertyDescriptor(e.tagName==='SELECT'?HTMLSelectElement.prototype:HTMLInputElement.prototype,'value').set.call(e,${JSON.stringify(String(value))});e.dispatchEvent(new Event(e.tagName==='SELECT'?'change':'input',{bubbles:true}));})()`);await delay(150);};
  const checked=label=>js(`document.querySelector('[aria-label='+${JSON.stringify(JSON.stringify(label))}+']')?.checked`);
  const acknowledge=async()=>{if(!await checked('Acknowledge preparation research scope'))await click('Acknowledge preparation research scope','input[type=checkbox]');};
  const request=async(label,command)=>{const count=report.requests.length,start=Date.now();await click(label);await until(()=>report.requests.slice(count).some(r=>r.command===command),command);await ready();const item=report.requests.slice(count).find(r=>r.command===command);report.interactions.push({command,backendSeconds:item.seconds,clickToAcceptedDisplaySeconds:(Date.now()-start)/1000});return last;};
  const capture=async(name,selector='.preparation-planning')=>{await ready();await js(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});for(let p=e;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;e?.scrollIntoView({block:'start'})})()`);await delay(500);assert(!window.isVisible());fs.writeFileSync(path.join(output,name+'.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());};
  const generate=async name=>{await request('Compare preparations and complete plans','generatePreparations');await until(()=>js(`!!document.querySelector('[data-preparation-seal]')?.getAttribute('data-preparation-seal')`),'current preparation table');const item=last.preparationPlanning;fs.writeFileSync(path.join(output,name+'.json'),JSON.stringify(item,null,2));assert.equal(last.view.stateId,report.baselineStateId);return item;};
  const preview=async width=>{await click('Select preparation width '+width+' mm');await set('Preparation checkpoint phase','prepared');await request('Preview preparation checkpoint','previewPreparation');await until(()=>js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Install preparation branch'&&!b.disabled)`),'admitted preparation preview');assert.equal(last.approachPreview.kind,'preparation');assert(last.approachPreview.receipt.nativeMatchesPublicPrediction&&last.approachPreview.receipt.withdrawalComplete);assert.equal(last.view.stateId,report.baselineStateId);return last.approachPreview;};
  try{
   await until(()=>js(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`),'case load');await click('Rehearse');await ready();
   if(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify saved replay')`)){await click('Verify saved replay');await ready();}
   report.baselineStateId=last.view.stateId;report.baselineBranchId=last.branchId;report.sourceSceneId=last.view.scene.sceneId;report.caseHash=last.caseHash;
   if(process.env.PREPARATION_LAYOUT_ONLY==='1'){
    if(!last.preparationPlanning.current){await click('Open preparation comparison checkpoint');await ready();}
    await click('Compare preparation and complete plans','summary');await capture('numeric-comparison-layout','[aria-label="Preparation comparison table"]');
    await set('Surgical branch',report.baselineBranchId);await ready();assert.equal(last.view.stateId,report.baselineStateId);
   }else if(process.env.PREPARATION_REOPEN_ONLY==='1'){
    assert.equal(last.view.stateId,process.env.PREPARATION_EXPECTED_STATE);assert.equal(last.preparationPlanning.revision,3);
    assert.equal(last.preparationPlanning.applications.length,2);assert(last.preparationPlanning.rows.every(r=>r.reason==='REQUIRED_EVIDENCE_UNAVAILABLE'));
    assert(last.strategyPlanning);assert.equal(last.strategyPlanning.startStateId,'sha256:a660418aea35538e66312100e4a88b606b94d78ee001a05e659f9dc13da8c571');
    await click('Inspect installed 0.3 mm preparation');await ready();assert.equal(last.view.stateId,'sha256:a660418aea35538e66312100e4a88b606b94d78ee001a05e659f9dc13da8c571');assert(last.interactivePlanningAvailability.available);
    await click('Inspect installed 0 mm preparation');await ready();assert.equal(last.view.stateId,'sha256:609321c5668efb2911b507143d24da691e6fd70b4d6be09c31de2dbaba60df02');assert(last.interactivePlanningAvailability.available);
    await click('Open preparation comparison checkpoint');await ready();assert.equal(last.view.stateId,report.baselineStateId);await capture('reopened-preparation-studies');
    await set('Surgical branch',report.baselineBranchId);await ready();assert.equal(last.branchId,report.baselineBranchId);assert.equal(last.view.stateId,report.baselineStateId);
    report.checks.push('Fresh hidden process verifies three preparation reports, two owned admissions, prior strategy report, source return and both installed branches');
   }else{
    assert(last.preparationPlanningAvailability.available);assert(!last.interactivePlanningAvailability.available);assert.equal(last.view.metrics.nonTargetRemovedMm3,0);
    await click('Compare preparation and complete plans','summary');await acknowledge();
    await request('Suggest source ray from public anatomy','suggestPreparation');await until(()=>js(`document.querySelector('[aria-label="Source surface cell A coordinate"]')?.value==='85'`),'suggested source cells');
    assert.equal(last.view.stateId,report.baselineStateId);assert.equal(last.position,0);await capture('suggested-source-ray');
    await set('Source surface cell A coordinate',83);
    await set('Source surface cell S coordinate',123);
    const context=last.preparationPlanningAvailability.contextOptions.find(x=>x.labelValue===2);assert(context);
    await set('Preparation context layer',context.layerId);await set('Preparation context near volume mm3',1000);
    let study=await generate('matched-preparations');assert.equal(study.rows.length,3);assert.deepEqual(study.rows.map(r=>r.widthMm),[0,.3,1]);
    assert.equal(study.rows[1].preparation.publicPlan.attempts[0].summary.additionalNonTargetMm3,66);assert.equal(study.rows[1].comparison.publicPlan.decision.selectedId,'stop');assert.equal(study.rows[2].preparedAdmission,null);
    assert.equal(study.policies.corridorFirst.selectedPreparationId,'width-1');assert.equal(study.policies.completePlanAware.selectedPreparationId,null);
    await capture('matched-preparation-table','[aria-label="Preparation comparison table"]');
    await set('Preparation non-target limit · mm³',199);assert.equal(await js(`document.querySelector('[aria-label="Preparation comparison draft"]')?.getAttribute('data-preparation-seal')`),null);
    await set('Preparation non-target limit · mm³',200);assert.equal(await js(`document.querySelector('[aria-label="Preparation comparison draft"]')?.getAttribute('data-preparation-seal')`),null);
    study=await generate('regenerated-after-edit');assert.equal(study.revision,2);
    let p=await preview(.3);assert.equal(p.view.metrics.nonTargetRemovedMm3,66);assert.equal(await js(`document.querySelector('[aria-label="Preparation checkpoint phase"] option[value="withdraw"]').disabled`),true);
    await capture('costful-preparation-preview','[aria-label="Selected preparation"]');
    await request('Install preparation branch','applyPreparation');assert.equal(last.view.stateId,'sha256:a660418aea35538e66312100e4a88b606b94d78ee001a05e659f9dc13da8c571');assert(last.interactivePlanningAvailability.available);report.firstBranch=last.branchId;report.firstStateId=last.view.stateId;
    await click('Compare two-tool strategies','summary');await click('Acknowledge two-tool research scope','input[type=checkbox]');await request('Compare executable strategies','generateStrategies');assert.equal(last.strategyPlanning.publicPlan.decision.selectedId,'stop');
    await click('Open preparation comparison checkpoint');await ready();assert.equal(last.view.stateId,report.baselineStateId);await acknowledge();await preview(0);
    await request('Install preparation branch','applyPreparation');assert.equal(last.view.stateId,'sha256:609321c5668efb2911b507143d24da691e6fd70b4d6be09c31de2dbaba60df02');assert(last.interactivePlanningAvailability.available);report.secondBranch=last.branchId;report.secondStateId=last.view.stateId;
    assert.equal(last.strategyPlanning.startStateId,report.firstStateId);assert.equal(last.preparationPlanning.applications.length,2);await capture('independent-installed-preparations');
    await click('Open preparation comparison checkpoint');await ready();await acknowledge();await click('Require functional evidence for preparation comparison','input[type=checkbox]');study=await generate('required-functional-stop');assert.equal(study.revision,3);assert(study.rows.every(r=>r.preparation===null&&r.reason==='REQUIRED_EVIDENCE_UNAVAILABLE'));assert.equal(last.view.metrics.nonTargetRemovedMm3,0);assert(!last.interactivePlanningAvailability.available);await capture('functional-evidence-stop');
    report.checks.push('Public source suggestion changes no action; actual controls compare three widths, preserve costful preparation and full-plan STOP, reject stale edited drafts, and enforce functional evidence');
    report.checks.push('Two preparations install with distinct source-bound admissions; prior two-tool study and original unadmitted source remain inspectable');
   }
   await click('Save');await until(()=>fs.existsSync(saved),'native save');report.saved=saved;report.savedStateId=last.view.stateId;
   assert.equal(sha(input),original);assert(!report.windowEverVisible&&!window.isVisible());report.sourcePreserved=true;report.passed=true;
  }catch(error){report.error=String(error);report.body=await js('document.body.innerText').catch(()=>null);fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());console.error(error);}
  fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,error:report.error,saved:report.saved}));app.exit(report.passed?0:1);
 });
});
require('./main.cjs');
