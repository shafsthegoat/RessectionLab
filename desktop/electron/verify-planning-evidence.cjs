'use strict';
// Hidden verification of the native and generated planning panels.
const {app,dialog,ipcMain,Menu}=require('electron');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {createHash}=require('node:crypto');

const root=fs.realpathSync(path.join(__dirname,'../../testcases'));
const input=fs.realpathSync(process.env.PLANNING_INPUT||'');
const output=path.resolve(process.env.PLANNING_OUTPUT||'');
assert(input.startsWith(root+path.sep)&&output.startsWith(root+path.sep));
assert(fs.realpathSync(path.dirname(output)).startsWith(root+path.sep));
fs.mkdirSync(output);
const prior=process.env.PLANNING_PRIOR?JSON.parse(fs.readFileSync(process.env.PLANNING_PRIOR,'utf8')):null;
const expected=process.env.PLANNING_EXPECT||'';
const native=input.endsWith('.rsworkspace')||path.basename(input)==='patient-anatomy.json';
const sha=file=>createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const report={input,inputSha256:sha(input),expected,pid:process.pid,windowEverVisible:false,captures:[],passed:false};
const saveTo=path.join(output,'saved.ressectionlab');
app.setPath('userData',path.join(output,'profile'));
if(process.platform==='darwin')app.setActivationPolicy('accessory');
process.argv.push('--automation-hidden','--case',input);
let lastSurgical,lastPlan,saveResult;
dialog.showSaveDialog=async()=>({canceled:false,filePath:saveTo});
dialog.showErrorBox=(_title,message)=>{report.error=message;finish();};
const handle=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(channel,handler)=>handle(channel,async(...args)=>{
  let value;
  try{value=await handler(...args);}
  catch(error){report.requestError={channel,message:error.message};throw error;}
  if(channel==='research:surgicalRehearsal')lastSurgical=value;
  if(channel==='research:anatomyPlan')lastPlan=value;
  if(channel==='research:saveCase')saveResult=value;
  return value;
});
const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
let finished=false;
function finish(){
  if(finished)return;
  finished=true;
  fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n');
  clearTimeout(watchdog);
  if(report.passed)app.quit();else app.exit(1);
}
const watchdog=setTimeout(()=>{report.error='Hidden verification timed out';finish();},native?150000:85000);
app.once('browser-window-created',(_event,window)=>{
  window.on('show',()=>{report.windowEverVisible=true;window.hide();});
  window.webContents.once('did-finish-load',async()=>{
    const js=code=>window.webContents.executeJavaScript(`(()=>{${code}})()`,true);
    const button=label=>`[...document.querySelectorAll('button')].find(b=>b.checkVisibility()&&!b.disabled&&
      (b.textContent.trim()===${JSON.stringify(label)}||b.getAttribute('aria-label')===${JSON.stringify(label)}))`;
    const wait=async(check,label)=>{
      const deadline=Date.now()+(native?110000:35000);
      while(Date.now()<deadline){if(await check())return;await pause(80);}
      throw Error('Timed out waiting for '+label);
    };
    const click=async label=>{
      await wait(()=>js(`return !!${button(label)}`),label);
      await js(`${button(label)}.click()`);
    };
    const capture=async(name,selector)=>{
      assert(!window.isVisible()&&!report.windowEverVisible);
      await js(`document.querySelector(${JSON.stringify(selector)})?.scrollIntoView({block:'start'})`);
      await pause(200);
      const image=await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true});
      assert(!image.isEmpty());
      fs.writeFileSync(path.join(output,name+'.png'),image.toPNG());
      report.captures.push(name+'.png');
    };
    try{
      window.setContentSize(1550,990);
      window.webContents.setBackgroundThrottling(false);
      if(native){
        await click('Plan & compare');
        const selector=`.approach-measurements [data-access-assessment="${expected}"]`;
        await wait(()=>js(`return document.querySelector(${JSON.stringify(selector)})?.checkVisibility() && !document.querySelector('.viewer-loading')`),'access readout');
        const selected=lastPlan.approaches.find(row=>row.id===lastPlan.selectedId)||lastPlan.approaches[0];
        assert(selected.measurements.simulationExecutionAuthorized===false);
        report.binding={target:selected.targetLayerId,snapshot:lastPlan.snapshotId,access:selected.measurements.accessAssessment,
          approach:selected.parameterHash,labelVolume:selected.measurements.targets[0].totalVolumeMm3};
        if(prior)assert.deepEqual(report.binding,prior.binding);
        await capture('source-bound-access',selector);
      }else{
        await wait(()=>js("return document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')&&document.body.innerText.includes('generated-sequential-parent-')"),'loaded source workspace');
        await click('Rehearse');
        await wait(()=>js(`return document.querySelector('.surgical-panel')?.dataset.stateId`),'native action history');
        await click('Compare stage');
        await wait(()=>js(`return document.querySelector('.planning-evidence')?.checkVisibility()`),'planning evidence panel');
        const packet=lastSurgical.planningEvidence,aware=packet.alternatives.find(row=>row.branchId==='harm-aware');
        report.binding={source:packet.sourceSceneId,baseline:packet.baselineStateId,seal:packet.seal,
          endpoint:aware.stateId,decisions:aware.intentDecisions.map(row=>row.decision)};
        if(prior){assert.equal(prior.savedSha256,report.inputSha256);assert.deepEqual(prior.binding,report.binding);assert.notEqual(report.pid,prior.pid);}
        if(expected==='overlap'){
          const older=packet.alternatives.find(row=>row.branchId==='conditional');
          assert.equal(older.endpoint.progress.originalTargetTreatmentMm3,81);
          assert.equal(aware.endpoint.progress.originalTargetTreatmentMm3,81);
          assert.equal(older.consequences.dimensions['structure:corticospinal:removedMm3'].cumulative,54);
          assert.equal(aware.consequences.dimensions['structure:corticospinal:removedMm3'].cumulative,0);
          assert(aware.intentDecisions[0].decision.controllingHarmMetrics.includes('structure:corticospinal:removedMm3'));
        }else{
          assert.equal(aware.intentExecution.executionStatus,'justified_residual_stop');
          assert.equal(aware.consequences.residualTargetMm3,81);
          assert(aware.endpoint.validWithdrawal);
        }
        await capture('matched-alternatives','.planning-evidence');
        await capture(expected==='overlap'?'preservation-choice':'finish-with-residual',
          expected==='overlap'?'[data-intent="continue"]':'[data-intent="stop_with_residual"]');
        if(!prior){
          const menu=Menu.getApplicationMenu().items.find(item=>item.label==='File');
          menu.submenu.items.find(item=>item.label==='Save Workspace…').click();
          await wait(()=>saveResult?.saved&&fs.existsSync(saveTo),'portable Save Workspace');
          report.saved=saveTo;report.savedSha256=sha(saveTo);
        }
      }
      assert.equal(sha(input),report.inputSha256);
      report.passed=true;
    }catch(error){
      report.error=error.stack;
      report.body=await js('return document.body.innerText').catch(()=>null);
    }finally{finish();}
  });
});
require('./main.cjs');
