'use strict';
// Actual UI/IPC/native persistence; never show or focus an automation window.
const {app,ipcMain,dialog}=require('electron');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..'),base=path.join(root,'testcases');
const input=fs.realpathSync(process.env.LABEL_CASE||''),output=path.resolve(process.env.LABEL_OUTPUT||'');
for(const p of [input,output])assert(p.startsWith(base+path.sep));assert(!fs.existsSync(output),'Preserve evidence');fs.mkdirSync(output,{recursive:true});
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex'),sourceSha256=sha(input);
const saved=path.join(output,'label-studies-saved.ressectionlab');
const report={input,sourceSha256,output,requests:[],checks:[],windowEverVisible:false};
let last=null,inflight=0,lastSettled=Date.now();
app.setPath('userData',path.join(output,'hidden-profile'));process.argv.push('--case',input,'--automation-hidden');
dialog.showSaveDialog=async()=>({canceled:false,filePath:saved});
const handle=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(name,handler)=>handle(name,async(...args)=>{
 const started=Date.now();inflight++;
 try{const value=await handler(...args);if(name==='research:surgicalRehearsal'){
  last=value;report.requests.push({command:args[1]?.command,seconds:(Date.now()-started)/1000,stateId:value?.view?.stateId,
   sensitivitySeal:value?.strategyPlanning?.anatomicalSensitivity?.study.studySeal});}return value;
 }finally{inflight--;lastSettled=Date.now();}
});
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
app.on('browser-window-created',(_event,window)=>{
 window.on('show',()=>{report.windowEverVisible=true;window.hide();});
 window.webContents.once('did-finish-load',async()=>{
  window.webContents.setBackgroundThrottling(false);window.setContentSize(1440,900);
  const js=code=>window.webContents.executeJavaScript(`(()=>{${code}})()`,true);
  const until=async(fn,label)=>{const start=Date.now();while(Date.now()-start<240000){if(await fn())return;
   const error=await js(`return document.querySelector('.surgical-panel .surgical-error')?.textContent`);if(error)throw Error(error);await delay(80);}throw Error('Timeout '+label);};
  const ready=()=>until(()=>js(`return !!document.querySelector('[data-surgical-state-id]')&&document.querySelector('.rl-viewer')?.dataset.surfacesPending==='0'&&!document.querySelector('.viewer-loading')&&![...document.querySelectorAll('.surgical-panel [role="status"]')].some(n=>/^(Computing instrument action|Preparing an exact recorded state|Preparing accepted display|Verifying saved source-bound rehearsal)/.test(n.textContent.trim()))`),'GPU-qualified accepted display');
  const click=async(label,selector='button')=>{await js(`const b=[...document.querySelectorAll(${JSON.stringify(selector)})].find(n=>n.textContent.trim()===${JSON.stringify(label)}||n.getAttribute('aria-label')===${JSON.stringify(label)});if(!b||b.matches(':disabled'))throw Error('Missing enabled '+${JSON.stringify(label)});for(let p=b.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS'&&!(b.tagName==='SUMMARY'&&p===b.parentElement))p.open=true;b.click();`);await delay(80);};
  const set=async(label,value)=>{await js(`const e=document.querySelector('[aria-label='+${JSON.stringify(JSON.stringify(label))}+']');if(!e)throw Error('Missing input');Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(e,${JSON.stringify(String(value))});e.dispatchEvent(new Event('input',{bubbles:true}));`);await delay(120);};
  const capture=async name=>{await ready();await js(`const e=document.querySelector('.anatomical-sensitivity');if(e){e.open=true;e.scrollIntoView({block:'start'});}`);await delay(120);assert(!window.isVisible());fs.writeFileSync(path.join(output,name+'.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());};
  const assess=async(step,name)=>{await set('Context label voxel shift',step);const count=report.requests.length;
   await click('Assess declared context shifts');await until(()=>report.requests.slice(count).some(x=>x.command==='analyzeStrategySensitivity'),'source-label study');await ready();
   const item=last.strategyPlanning.anatomicalSensitivity;assert(item?.study.nativeReplayVerified);const a=item.study.assessment;
   assert.equal(a.controls.contextVoxelShift,step);assert.equal(a.scenarioCount,step?7:1);assert.equal(a.calibratedUncertainty.supported,false);
   assert.equal(a.fixedMarginDefinition.isGuarantee,false);assert.equal(last.view.stateId,report.baselineStateId);
   fs.writeFileSync(path.join(output,name+'.json'),JSON.stringify(item,null,2));return a;
  };
  try{
   await until(()=>js(`return !!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`),'case load');
   await click('Rehearse');await ready();
   if(await js(`return [...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify saved replay')`)){await click('Verify saved replay');await ready();}
   report.baselineStateId=last.view.stateId;report.caseHash=last.caseHash;
   if(process.env.LABEL_NEGATIVE==='1'){
    assert.equal(last.strategyPlanningAvailability.available,false);
    assert(!await js(`return [...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Assess declared context shifts')`));
    await capture('missing-target-refusal');report.checks.push('Missing target exposes no anatomy sensitivity action');
   }else{
    assert(last.strategyPlanning?.contextBinding);await click('Compare two-tool strategies','summary');
    if(!await js(`return document.querySelector('[aria-label="Acknowledge two-tool research scope"]').checked`))await click('Acknowledge two-tool research scope','input[type=checkbox]');
    await click('Context-label sensitivity','summary');
    if(process.env.LABEL_REOPEN==='1'){
     const study=last.strategyPlanning.anatomicalSensitivity?.study;assert(study?.nativeReplayVerified);
     assert(await js(`return !!document.querySelector('[aria-label="Context sensitivity decision"]')`));
     report.restoredStudySeal=study.studySeal;report.policies=study.assessment.policies;
     report.checks.push('Cold native proof restores exact saved label-study controls and policy');
    }else{
     const zero=await assess(0,'zero-shift');assert(Object.values(zero.policies).every(id=>id===last.strategyPlanning.publicPlan.decision.selectedId));
     await set('Context label voxel shift',1);assert(!await js(`return !!document.querySelector('[aria-label="Context sensitivity decision"]')`));
     await set('Context label voxel shift',0);assert(await js(`return !!document.querySelector('[aria-label="Context sensitivity decision"]')`));
     const shifted=await assess(1,'one-voxel-shift');report.policies=shifted.policies;
     assert(last.strategyPlanning.executionSensitivity?.study,'Old execution-offset attachment was preserved');
     const point=shifted.candidates.flatMap(c=>c.outcomes.map(o=>({id:c.id,...o}))).find(o=>o.encounter.nearPositionsMm.length);
     if(point){await click('Show hypothetical context cell '+point.id+' '+point.scenarioId);report.cursorRasMm=point.encounter.nearPositionsMm[0];
      await until(()=>js(`return document.querySelector('.rl-viewer-statusbar')?.textContent.includes(${JSON.stringify(report.cursorRasMm.map(x=>x.toFixed(1)).join(' / '))})`),'linked hypothetical context point');}
     report.checks.push('Actual controls reproduce nominal physics, hide stale results, preserve old command stress and drive linked RAS views');
    }
    await click('Select result for all tested label shifts');
    assert(await js(`return document.querySelector('[data-strategy-candidate='+${JSON.stringify(JSON.stringify(report.policies.allDeclaredScenarios))}+']')?.getAttribute('aria-selected')==='true'`));
    const unknown=last.strategyPlanning.anatomicalSensitivity.study.assessment.candidates.flatMap(c=>c.outcomes.map(o=>({id:c.id,...o}))).find(o=>o.encounter.unknownEncounteredCells.length);
    if(unknown){
     const cell=unknown.encounter.unknownEncounteredCells[0],scene=last.view.scene,s=scene.shape;
     const ijk=[Math.floor(cell/(s[1]*s[2])),Math.floor(cell/s[2])%s[1],cell%s[2]];
     report.unknownCursorRasMm=scene.affine.slice(0,3).map(row=>row[0]*ijk[0]+row[1]*ijk[1]+row[2]*ijk[2]+row[3]);
     await click('Show unknown context cell '+unknown.id+' '+unknown.scenarioId);
     await until(()=>js(`return document.querySelector('.rl-viewer-statusbar')?.textContent.includes(${JSON.stringify(report.unknownCursorRasMm.map(x=>x.toFixed(1)).join(' / '))})`),'linked unknown-coverage cell');
     report.checks.push('The coverage refusal drives its exact destination cell into linked RAS source views');
    }
    assert.equal(last.view.stateId,report.baselineStateId);await capture('context-decision');
    await click('Save');await until(()=>fs.existsSync(saved)&&inflight===0&&Date.now()-lastSettled>600,'save and native library refresh');await ready();
    report.saved=saved;report.savedStateId=last.view.stateId;
   }
   assert.equal(sha(input),sourceSha256);assert(!report.windowEverVisible&&!window.isVisible());report.sourcePreserved=true;report.passed=true;
  }catch(error){report.error=String(error);report.body=await js('return document.body.innerText').catch(()=>null);fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());console.error(error);}
  fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,error:report.error,saved:report.saved}));app.exit(report.passed?0:1);
 });
});
require('./main.cjs');
