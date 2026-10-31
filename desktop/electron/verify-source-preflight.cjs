'use strict';
// Actual renderer/backend source readiness, always hidden and off the primary display.
const {app,ipcMain,dialog}=require('electron');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..'),base=path.join(root,'testcases');
const input=fs.realpathSync(process.env.PREFLIGHT_CASE||''),output=path.resolve(process.env.PREFLIGHT_OUTPUT||'');
for(const p of [input,output])assert(p.startsWith(base+path.sep));assert(!fs.existsSync(output),'Preserve prior evidence');fs.mkdirSync(output,{recursive:true});
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex'),original=sha(input);
const saved=path.join(output,'source-readiness-saved.ressectionlab');
const report={input,sourceSha256:original,expected:process.env.PREFLIGHT_EXPECT||'intact',windowEverVisible:false,requests:[]};let last=null;
app.setPath('userData',path.join(output,'hidden-profile'));process.argv.push('--case',input,'--automation-hidden');
dialog.showSaveDialog=async()=>({canceled:false,filePath:saved});
const handle=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(name,handler)=>handle(name,async(...args)=>{const value=await handler(...args);if(name==='research:surgicalRehearsal'){last=value;report.requests.push({command:args[1]?.command,stateId:value?.view?.stateId,position:value?.position});}return value;});
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
app.on('browser-window-created',(_event,window)=>{
 window.on('show',()=>{report.windowEverVisible=true;window.hide();});
 window.webContents.once('did-finish-load',async()=>{
  window.webContents.setBackgroundThrottling(false);window.setContentSize(1440,900);
  const js=async code=>{const value=await window.webContents.executeJavaScript(`try{${code}}catch(error){({automationError:String(error)})}`,true);if(value?.automationError)throw Error(value.automationError);return value;};
  const until=async(fn,label)=>{const start=Date.now();while(Date.now()-start<120000){if(await fn())return;const error=await js(`document.querySelector('.surgical-panel .surgical-error')?.textContent`);if(error)throw Error(error);await delay(80);}throw Error('Timeout '+label);};
  const ready=()=>until(()=>js(`!!document.querySelector('[data-surgical-state-id]')&&document.querySelector('.rl-viewer')?.dataset.surfacesPending==='0'&&!document.querySelector('.viewer-loading')`),'accepted native display');
  const click=async(label)=>{await js(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(label)});if(!b||b.disabled)throw Error('Missing enabled '+${JSON.stringify(label)});b.click()})()`);await delay(100);};
  const capture=async name=>{await ready();await js(`document.querySelector('[aria-label="Planning readiness"]')?.scrollIntoView({block:'start'})`);await delay(150);assert(!window.isVisible());fs.writeFileSync(path.join(output,name+'.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());};
  try{
   await until(()=>js(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`),'case load');
   await click('Rehearse');await ready();await until(()=>!!last?.interactivePlanningAvailability,'planning availability');
   if(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify saved replay')`)){await click('Verify saved replay');await ready();}
   const value=last.interactivePlanningAvailability;report.stateId=last.view.stateId;report.position=last.position;
   if(report.expected==='admitted'){
    assert(value.admission&&value.available);assert(!value.sourcePreflight);assert.equal(await js(`!!document.querySelector('[aria-label="Planning readiness"]')`),false);
    report.admission=value.admission.admissionSeal;
   }else{
    assert.equal(value.available,false);const preflight=value.sourcePreflight;assert(preflight);assert.equal(preflight.planningAdmitted,false);
    assert.equal(preflight.sourceSceneId,last.view.scene.sceneId);assert.equal(preflight.startStateId,last.view.stateId);assert.equal(preflight.prefixActions,last.position);
    const statuses=Object.fromEntries(preflight.checks.map(r=>[r.id,r.status]));report.statuses=statuses;report.preflight=preflight;
    if(report.expected==='intact'){
     assert.equal(statuses['source-role'],'passed');for(const key of ['model-versions','executed-exposure','declared-task'])assert.equal(statuses[key],'required');
     assert.equal(statuses['source-bytes'],'not-checked');assert.equal(statuses['evidence-review'],'requires-research-opt-in');
     assert.equal(last.position,0);assert.equal(last.view.metrics.targetRemovedMm3,0);assert.equal(last.view.metrics.nonTargetRemovedMm3,0);
    }else{assert.equal(statuses['source-role'],'refused');if(report.expected==='missing-target')assert.match(value.reason,/tumor target is required/);}
    assert.equal(await js(`[...document.querySelectorAll('button')].some(b=>['Generate approaches','Compare executable strategies','Assess declared execution offsets'].includes(b.textContent.trim()))`),false);
    assert(await js(`!!document.querySelector('[aria-label="Planning readiness"]')`));
    await capture('source-readiness');
   }
   await click('Save');await until(()=>fs.existsSync(saved),'native save');report.saved=saved;
   assert.equal(last.view.stateId,report.stateId);assert.equal(sha(input),original);assert(!report.windowEverVisible&&!window.isVisible());
   report.sourcePreserved=true;report.passed=true;
  }catch(error){report.error=String(error);report.body=await js('document.body.innerText').catch(()=>null);console.error(error);fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());}
  fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,error:report.error,saved:report.saved}));app.exit(report.passed?0:1);
 });
});
require('./main.cjs');
