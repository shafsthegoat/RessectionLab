'use strict';
// Exercise actual UI controls and native persistence without showing a window.
const {app,ipcMain,dialog}=require('electron');
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..'),base=path.join(root,'testcases');
const input=fs.realpathSync(process.env.CLOCK_CASE||path.join(base,'shared-clock-v1/engineering.ressectionlab'));
const output=path.resolve(process.env.CLOCK_OUTPUT||path.join(base,'shared-clock-v1/desktop-01'));
for(const p of [input,output])assert(p.startsWith(base+path.sep),'Use ignored testcases paths');
assert(!fs.existsSync(output),'Preserve previous evidence; choose a fresh output');fs.mkdirSync(output,{recursive:true});
app.setPath('userData',path.join(output,'hidden-profile'));
process.argv.push('--case',input,'--automation-hidden');
const saved=path.join(output,'clock-saved.ressectionlab');dialog.showSaveDialog=async()=>({canceled:false,filePath:saved});
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex'),original=sha(input);
const report={input,output,inputSha256:original,windowEverVisible:false,checks:[],receipts:[]};
let lastResponse=null;
const handle=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(name,handler)=>handle(name,async(...args)=>{
  const result=await handler(...args);
  if(name==='research:surgicalRehearsal'){
    lastResponse=result;
    report.receipts.push({request:args[1],stateId:result?.view?.stateId,clock:result?.view?.physicalClock,
      position:result?.position,branchId:result?.branchId,outcome:result?.outcome,metrics:result?.view?.metrics});
  }
  return result;
});
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
app.on('browser-window-created',(_event,window)=>{
  window.on('show',()=>{report.windowEverVisible=true;window.hide();});
  window.webContents.once('did-finish-load',async()=>{
    window.webContents.setBackgroundThrottling(false);window.setContentSize(1440,900);
    const js=source=>window.webContents.executeJavaScript(source,true);
    const wait=async(expression,seconds=90)=>{const start=Date.now();while(Date.now()-start<seconds*1000){if(await js(expression))return;await delay(80);}throw Error('Timeout '+expression);};
    const ready=()=>wait(`!!document.querySelector('[data-surgical-state-id]')&&document.querySelector('.rl-viewer')?.dataset.surfacesPending==='0'&&!document.querySelector('.viewer-loading')&&![...document.querySelectorAll('.surgical-panel [role="status"]')].some(n=>/^(Computing instrument action|Preparing an exact recorded state|Preparing accepted display|Verifying saved source-bound rehearsal)/.test(n.textContent.trim()))`);
    const click=async(text,selector='button')=>{await js(`(()=>{const b=[...document.querySelectorAll(${JSON.stringify(selector)})].find(b=>b.textContent.trim()===${JSON.stringify(text)});if(!b||b.matches(':disabled'))throw Error('Missing enabled '+${JSON.stringify(text)});for(let p=b.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;b.click()})()`);await delay(100);};
    const value=async(label,v)=>{await js(`(()=>{const e=document.querySelector(${JSON.stringify('[aria-label="'+label+'"]')});if(!e||e.matches(':disabled'))throw Error('Missing enabled '+${JSON.stringify(label)});for(let p=e.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;const proto=e.tagName==='SELECT'?HTMLSelectElement.prototype:HTMLInputElement.prototype;Object.getOwnPropertyDescriptor(proto,'value').set.call(e,${JSON.stringify(String(v))});e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}))})()`);await delay(100);};
    const hand=async(name)=>{await js(`(()=>{const b=[...document.querySelectorAll('.surgical-slots button')].find(b=>b.querySelector('strong')?.textContent===${JSON.stringify(name+' hand')});if(!b||b.disabled)throw Error('Missing hand');b.click()})()`);await delay(80);};
    const action=async(operation,duration)=>{await value('Instrument action',operation);await value('Action duration',duration);const count=report.receipts.length;await click('Apply','.manual-actions button');while(report.receipts.length===count)await delay(80);await ready();assert.equal(lastResponse.outcome.status,'accepted');return lastResponse;};
    try{
      await wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`);
      await click('Rehearse');await ready();
      if(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify saved replay')`)){await click('Verify saved replay');await ready();}
      if(process.env.CLOCK_REOPEN_ONLY==='1'){
        assert.equal(lastResponse.view.physicalClock.revision,1);
        assert.equal(lastResponse.view.stateId,process.env.CLOCK_EXPECTED_STATE);
        report.endpointStateId=lastResponse.view.stateId;
        report.checks.push('New Electron process cold-reopens saved branches with exact endpoint and shared clock revision');
      }else{
      await click('Enable shared physical clock');await ready();assert.equal(lastResponse.view.physicalClock.revision,1);assert.equal(lastResponse.position,1);
      await click('Enable revised instrument mechanics');await ready();assert.equal(lastResponse.view.toolMechanicsRevision,2);
      // The UI writes each selected tool's displayed parameter defaults. Bind
      // both parameter records before the contrast, so the two branches differ
      // only in inspection ownership (including identical sparse metadata).
      await hand('left');await action('inspect',.1);
      await hand('right');await action('inspect',.1);await action('activate',.1);
      const start=lastResponse.position,initial=lastResponse.view.metrics.cusaInputCapacityMm3;
      const own=await action('inspect',.2);
      assert(Math.abs(own.view.metrics.cusaInputCapacityMm3-initial-.4)<1e-12);
      await value('Surgical timeline',start);await ready();assert.equal(lastResponse.position,start);
      await hand('left');const other=await action('inspect',.2);
      assert.equal(other.view.stateId,own.view.stateId);assert.deepEqual(other.view.metrics,own.view.metrics);
      report.checks.push('Actual clock control records revision without elapsed time; both inspection hands give identical state and +0.4 mm3 supplied capacity over 0.2 seconds');
      const position=other.position;
      await value('Surgical timeline',0);await ready();
      assert.equal(lastResponse.view.physicalClock,undefined);
      assert(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Enable shared physical clock'&&!b.matches(':disabled'))`));
      await value('Surgical timeline',position);await ready();
      assert.equal(lastResponse.view.stateId,other.view.stateId);assert.equal(lastResponse.view.physicalClock.revision,1);
      await click('Save');await wait(`document.body.innerText.includes('Workspace saved locally')`);assert(fs.existsSync(saved));
      report.saved=saved;report.endpointStateId=other.view.stateId;report.checks.push('Timeline before/after version boundary and native workspace save preserve scheduling version');
      }
      await js(`(()=>{const e=document.querySelector('[aria-label="Physical clock"]');for(let p=e.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;e.scrollIntoView({block:'center'})})()`);await delay(150);
      assert(await js(`document.querySelector('[aria-label="Physical clock"]').checkVisibility()`));
      fs.writeFileSync(path.join(output,'shared-clock.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());
      assert.equal(sha(input),original);assert(!report.windowEverVisible);assert.equal(window.isVisible(),false);report.passed=true;
    }catch(error){report.error=String(error);report.body=await js('document.body.innerText').catch(()=>null);fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());console.error(error);}
    fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,error:report.error,saved:report.saved}));app.exit(report.passed?0:1);
  });
});
require('./main.cjs');
