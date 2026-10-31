'use strict';
// Real desktop controls and rendering, hidden throughout. Never show/focus.
const {app,dialog,ipcMain}=require('electron');
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..');
const input=fs.realpathSync(path.resolve(process.env.INTERACTIVE_PLANNING_CASE||path.join(root,'testcases/outputs/interactive-planning-20261031/bridge-v4-head-physics/interactive-planning.ressectionlab')));
const output=path.resolve(process.env.INTERACTIVE_PLANNING_OUTPUT||path.join(root,'testcases/outputs/interactive-planning-20261031/desktop-v1'));
for(const value of [input,output])assert(value.startsWith(path.join(root,'testcases')+path.sep),'Use ignored testcases paths');
assert(!fs.existsSync(output),'Preserve prior evidence; choose a new output directory');fs.mkdirSync(output,{recursive:true});
const sha=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const original=sha(input),saved=path.join(output,'desktop-planning.ressectionlab');
const report={schema:'resectionlab.interactive-planning-hidden-native.v1',input,output,windowEverVisible:false,ipc:[],checks:[],states:[],passed:false};
let lastResponse=null;
const originalHandle=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(channel,handler)=>originalHandle(channel,async(...args)=>{
  const tick=performance.now();
  try{const result=await handler(...args);if(channel==='research:surgicalRehearsal'){
    lastResponse=result;report.ipc.push({command:args[1]?.command,ms:performance.now()-tick,
      stateId:result?.view?.stateId,branchId:result?.branchId,position:result?.position,
      routes:result?.interactivePlanning?.routes?.length,reportSeal:result?.interactivePlanning?.reportSeal,
      previewStateId:result?.approachPreview?.view?.stateId});}return result;
  }catch(error){report.ipc.push({channel,command:args[1]?.command,ms:performance.now()-tick,error:String(error)});throw error;}
});
app.setPath('userData',path.join(output,'user-data'));
process.argv.push('--case',input,'--automation-hidden');
dialog.showSaveDialog=async()=>({canceled:false,filePath:saved});
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
app.on('browser-window-created',(_event,window)=>{
  window.on('show',()=>{report.windowEverVisible=true;window.hide();});
  window.webContents.once('did-finish-load',async()=>{
    window.webContents.setBackgroundThrottling(false);window.setContentSize(1440,900);
    const js=code=>window.webContents.executeJavaScript(code,true);
    const wait=async(expression,seconds=120)=>{const tick=Date.now();while(Date.now()-tick<seconds*1000){if(await js(expression))return;await delay(80);}throw Error('Timeout '+expression+'\n'+await js('document.body.innerText'));};
    const click=async(label,scope='document')=>{await js(`(()=>{const root=${scope},b=[...root.querySelectorAll('button,input[type=checkbox]')].find(b=>b.textContent.trim().replace(/^●\\s*/,'')===${JSON.stringify(label)}||b.getAttribute('aria-label')===${JSON.stringify(label)});if(!b||b.disabled)throw Error('Missing enabled control '+${JSON.stringify(label)});for(let e=b.parentElement;e;e=e.parentElement)if(e.tagName==='DETAILS')e.open=true;b.click();})()`);await delay(50);};
    const inputValue=async(label,value)=>{await js(`(()=>{const e=[...document.querySelectorAll('input')].find(e=>e.getAttribute('aria-label')===${JSON.stringify(label)});if(!e||e.disabled)throw Error('Missing enabled input '+${JSON.stringify(label)});for(let p=e.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(e,${JSON.stringify(String(value))});e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));})()`);await delay(80);};
    const ready=()=>wait(`!!document.querySelector('.surgical-panel')&&!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')&&document.querySelector('.rl-viewer').dataset.surfacesPending==='0'&&![...document.querySelectorAll('.surgical-panel [role="status"]')].some(n=>/^(Computing instrument action|Preparing an exact recorded state|Preparing accepted display|Verifying saved source-bound rehearsal)/.test(n.textContent.trim()))`);
    const state=()=>js(`(()=>{const panel=document.querySelector('.interactive-planning'),draft=panel?.querySelector('[data-planning-status]');return {stateId:document.querySelector('.surgical-panel')?.dataset.stateId,branch:document.querySelector('[aria-label="Surgical branch"]')?.value,position:Number(document.querySelector('[aria-label="Surgical timeline"]')?.value),status:draft?.dataset.planningStatus,rows:[...document.querySelectorAll('[data-planning-candidate]')].map(n=>({id:n.dataset.planningCandidate,text:n.textContent})),text:panel?.textContent,overlays:[...document.querySelectorAll('[data-planning-routes]')].map(n=>n.dataset.planningRoutes),alerts:[...document.querySelectorAll('.surgical-panel [role=alert]')].map(n=>n.textContent)}})()`);
    const capture=async name=>{await ready();await js(`document.querySelector('.interactive-planning')?.scrollIntoView({block:'start'})`);await delay(100);
      assert.equal(window.isVisible(),false);const before=await state();
      fs.writeFileSync(path.join(output,name+'.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());
      assert.deepEqual(await state(),before);report.states.push({name,...before});return before;};
    const generate=async(name,count)=>{
      const index=report.ipc.length,tick=performance.now();await click('Generate approaches');
      while(!report.ipc.slice(index).some(row=>row.command==='generateApproaches')){if(performance.now()-tick>120000)throw Error('Generation did not complete');await delay(80);}
      await ready();await wait(`document.querySelector('[data-planning-status]')?.dataset.planningStatus==='current'`);
      const elapsed=performance.now()-tick,current=await capture(name);assert.equal(current.rows.length,count);
      if(count===0)assert(current.alerts.every(text=>/Approaches refused/.test(text)),current.alerts.join('\n'));
      else assert.equal(current.alerts.length,0,current.alerts.join('\n'));
      (report.endToEnd??=[]).push({name,ms:elapsed,candidates:count});return current;
    };
    try{
      await wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`);
      await click('Rehearse');await ready();
      if(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify saved replay'&&!b.disabled)`)){await click('Verify saved replay');await ready();}
      if(process.env.INTERACTIVE_PLANNING_REOPEN==='1'){
        await capture('01-cold-reopened-installed-branch');
        assert(lastResponse?.interactivePlanning&&!lastResponse.interactivePlanning.current);
        await click('Open planning checkpoint');await ready();
        const current=await capture('02-cold-reopened-current-comparison');assert.equal(current.rows.length,3);assert.equal(current.status,'current');
        report.checks.push('Fresh desktop reopen canonically replays installed branch and restores saved comparison at its exact source checkpoint');
      }else{
        const initial=await capture('01-saved-current-comparison');assert.equal(initial.status,'current');assert.equal(initial.rows.length,3);
        const baseline=initial.stateId;
        // Use the real linked-cursor selection controls for the target region.
        const regionScope=`[...document.querySelectorAll('.approach-region')].find(e=>e.querySelector('legend')?.textContent==='Target region')`;
        await click('Show in linked views',regionScope);await click('Use linked MRI cursor',regionScope);
        await wait(`document.querySelector('[data-planning-status]')?.dataset.planningStatus==='draft'`);
        await click('Acknowledge interactive planning unknowns');
        await inputValue('Allowable entry region radius',.5);
        const changed=await generate('02-restricted-entry',1);assert.equal(changed.stateId,baseline);assert.match(changed.text,/Only|No distinct/);
        await inputValue('Allowable entry region radius',7);await generate('03-restored-entry',3);
        await inputValue('Mapped non-target clearance mm',.5);await generate('03b-mapped-protection-blocks',0);
        assert.match((await state()).text,/NON TARGET CLEARANCE|NON_TARGET_CLEARANCE|non target clearance/i);
        await inputValue('Mapped non-target clearance mm',0);await generate('03c-mapped-protection-restored',3);
        await click('Require functional evidence');await generate('04-required-unknown-evidence',0);
        assert.match((await state()).text,/functional|evidence/i);
        await click('Require functional evidence');await generate('05-restored-current',3);
        await click('Source approach 1');await ready();
        const tick=performance.now();await click('Preview checked approach');await ready();
        await wait(`document.querySelector('.approach-selection')?.textContent.includes('Preview shown')`);
        report.previewEndToEndMs=performance.now()-tick;
        const preview=await capture('06-exact-native-preview');assert.equal(preview.stateId,lastResponse.approachPreview.view.stateId);
        assert.notEqual(preview.stateId,baseline);assert.equal(lastResponse.view.stateId,baseline);
        await click('Install checked branch');await ready();
        const installed=await capture('07-installed-normal-branch');assert.notEqual(installed.branch,initial.branch);assert.equal(lastResponse.interactivePlanning.current,false);
        assert.equal(installed.stateId,preview.stateId);assert.equal(installed.alerts.length,0);
        await click('Save');await wait(`document.body.innerText.includes('Workspace saved locally')`);assert(fs.existsSync(saved));report.saved=saved;
        await click('Open planning checkpoint');await ready();
        const returned=await capture('08-current-at-exact-checkpoint');assert.equal(returned.stateId,baseline);assert.equal(returned.rows.length,3);
        window.setContentSize(1180,760);await delay(100);await capture('09-small-window');
        const fit=await js(`(()=>{const e=document.querySelector('.approach-table');return {client:e.clientWidth,scroll:e.scrollWidth}})()`);assert(fit.scroll<=fit.client+1,'Comparison fits the panel');
        report.checks.push('Actual target cursor selection and entry edits invalidate stale guides; generation changes eligibility from three routes to one',
          'Mapped non-target clearance0.5 mm refuses all default approaches with source geometry conflicts',
          'Required unavailable functional evidence yields zero routes with an explanation',
          'Candidate preview passes native insertion/withdrawal without commit; install creates a normal branch; source return restores comparison',
          '1440×900 and 1180×760 hidden captures preserve responsive comparison layout');
      }
      assert.equal(sha(input),original);assert.equal(report.windowEverVisible,false);report.passed=true;
    }catch(error){report.error=String(error);try{report.body=await js('document.body.innerText');fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());}catch{}process.exitCode=1;}
    finally{fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({passed:report.passed,error:report.error,output,saved:report.saved}));setTimeout(()=>app.exit(report.passed?0:1),100);}
  });
});
setTimeout(()=>{if(!report.passed){report.error??='Global native verifier timeout';fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n');app.exit(1);}},600000).unref();
require('./main.cjs');
