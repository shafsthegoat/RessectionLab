'use strict';
// Hidden native replay of both generated approach controls.
const {app,ipcMain}=require('electron');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../../testcases');
const input=path.resolve(process.env.PROCEDURE_DEMO_REPORT||'');
const output=path.resolve(process.env.PROCEDURE_DEMO_OUTPUT||'');
assert(input.startsWith(root+path.sep)&&output.startsWith(root+path.sep));
const fixture=JSON.parse(fs.readFileSync(input,'utf8'));
fs.mkdirSync(output);
app.setPath('userData',path.join(output,'profile'));
process.argv.push('--automation-hidden','--case',fixture.package);
const report={passed:false,windowEverVisible:false,captures:[]}, start=Date.now();
let last, lastArgs;
const handle=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(name,handler)=>handle(name,async(...args)=>{
  const value=await handler(...args);
  if(name==='research:surgicalRehearsal'){last=value;lastArgs=args[1];}
  return value;
});
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function until(check,label){
  const deadline=Date.now()+45000;
  while(Date.now()<deadline){if(await check())return;await delay(80);}
  throw Error('Timed out waiting for '+label);
}
app.once('browser-window-created',(_event,window)=>{
  window.on('show',()=>{report.windowEverVisible=true;window.hide();});
  window.webContents.once('did-finish-load',async()=>{
    const js=code=>window.webContents.executeJavaScript(`(()=>{${code}})()`,true);
    const button=label=>`[...document.querySelectorAll('button')].find(b=>b.checkVisibility()&&!b.disabled&&
      (b.textContent.trim()===${JSON.stringify(label)}||b.getAttribute('aria-label')===${JSON.stringify(label)}))`;
    const click=async label=>{
      await until(()=>js(`return !!${button(label)}`),label);
      await js(`${button(label)}.click()`);
    };
    const capture=async name=>{
      assert(!window.isVisible()&&!report.windowEverVisible);
      await delay(160);
      const image=await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true});
      fs.writeFileSync(path.join(output,name+'.png'),image.toPNG());
      report.captures.push(name+'.png');
    };
    const state=()=>js("return document.querySelector('.surgical-panel')?.dataset.stateId");
    try{
      window.setContentSize(1540,950);
      window.webContents.setBackgroundThrottling(false);
      await until(()=>js(`return document.body.innerText.includes('Generated approach controls')
        &&!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`),'source workspace');
      await click('Rehearse');
      await until(state,'saved rehearsal');
      const verify=await js(`return [...document.querySelectorAll('button')].find(b=>b.checkVisibility()
        &&['Verify saved replay','Verify replay for planning'].includes(b.textContent.trim()))?.textContent.trim()`);
      if(verify)await click(verify);
      await click('Rehearse/edit stage');
      await until(()=>js(`return document.querySelector('.workflow-rehearsal')?.checkVisibility()`),'replay controls');
      const positive=fixture.coldChecks.find(row=>row.branch===fixture.positiveBranch);
      await until(async()=>await state()===positive.stateId,'superior endpoint');
      assert.equal(last.view.metrics.targetRemovedMm3,fixture.accepted.endpoint.progress.originalTargetTreatmentMm3);
      assert(Object.values(last.view.slots).every(slot=>!slot.inserted&&!slot.active));
      report.positiveState=positive.stateId;
      await capture('superior-checked-withdrawal');
      await click('Previous recorded state');
      await until(()=>last?.position===positive.nativeHistoryReplayRequests-1,'native timeline seek');
      await click('Next recorded state');
      await until(async()=>await state()===positive.stateId,'replayed endpoint');
      const negative=fixture.coldChecks.find(row=>row.branch===fixture.negativeBranch);
      await js(`const e=document.querySelector('[aria-label="Surgical branch"]');
        Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(e,${JSON.stringify(fixture.negativeBranch)});
        e.dispatchEvent(new Event('change',{bubbles:true}));`);
      await until(async()=>await state()===negative.stateId,'inferior/posterior branch');
      const wrong=fixture.refused.attempts.find(attempt=>attempt.outcome.status==='rejected');
      await js(`const right=[...document.querySelectorAll('.surgical-slots button')].find(b=>b.querySelector('strong')?.textContent==='right hand');
        right.click();document.querySelector('.manual-actions details').open=true;`);
      for(const [index,axis] of ['R','A','S'].entries()) {
        await js(`const e=document.querySelector('[aria-label="Tool position ${axis}"]');
          Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(e,${JSON.stringify(String(wrong.action.pose.position[index]))});
          e.dispatchEvent(new Event('input',{bubbles:true}));`);
        await delay(90);
      }
      await click('Preview');
      await until(()=>lastArgs?.command==='preview','native refusal response');
      const preview=last;
      const displayedPose=lastArgs.action.path.at(-1);
      assert.deepEqual(displayedPose.position,wrong.action.pose.position);
      assert(displayedPose.quaternion.every((value,index)=>Math.abs(value-wrong.action.pose.quaternion[index])<1e-9));
      assert.equal(preview.reasonCode,wrong.outcome.reasonCode);
      assert.equal(preview.view.stateId,negative.stateId);
      assert.equal(await state(),negative.stateId);
      report.negative={stateId:negative.stateId,reason:preview.reason};
      await until(()=>js(`const outcome=[...document.querySelectorAll('.surgical-preview')].find(
        row=>row.innerText.includes('PROTECTED STRUCTURE'));if(!outcome)return false;
        outcome.scrollIntoView({block:'center'});return true;`),'displayed guard refusal');
      await capture('inferior-posterior-refused');
      report.passed=true;
    }catch(error){
      report.error=error.stack;
      report.body=await js('return document.body.innerText').catch(()=>null);
    }finally{
      report.wallSeconds=(Date.now()-start)/1000;
      fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n');
      console.log(JSON.stringify({passed:report.passed,error:report.error,output}));
      process.exitCode=report.passed?0:1;
      app.quit();
    }
  });
});
require('./main.cjs');
