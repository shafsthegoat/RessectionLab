'use strict';
// Hidden native verification only; never shows/focuses the primary monitor.
const {app,dialog}=require('electron');
const fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..');
const input=path.resolve(process.env.VASCULAR_CASE||''),output=path.resolve(process.env.VASCULAR_OUTPUT||'');
for(const p of [input,output])if(!p.startsWith(path.join(root,'testcases')+path.sep))throw Error('Use ignored testcases paths');
if(fs.existsSync(output))throw Error('Preserve prior evidence');fs.mkdirSync(output,{recursive:true});
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex'),original=hash(input);
const prepared=JSON.parse(fs.readFileSync(process.env.VASCULAR_REPORT||path.join(path.dirname(input),'report.json'),'utf8'));
assert.equal(prepared.schema,'resectionlab.vascular-qualification.v1');
const saved=path.join(output,'vascular-saved.ressectionlab');
const profile=fs.mkdtempSync(path.join(os.tmpdir(),'rslab-vascular-'));
app.setPath('userData',profile);process.argv.push('--case',input,'--automation-hidden');
dialog.showSaveDialog=async()=>({canceled:false,filePath:saved});
const report={input,output,temporaryProfile:profile,windowEverVisible:false,checks:[],states:[],errors:[]};
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
app.on('browser-window-created',(_event,window)=>{
  window.on('show',()=>{report.windowEverVisible=true;window.hide();});
  window.webContents.once('did-finish-load',async()=>{
    window.webContents.setBackgroundThrottling(false);window.setSize(1600,1050);
    const js=code=>window.webContents.executeJavaScript(code,true);
    const wait=async expression=>{const start=Date.now();while(Date.now()-start<180000){if(await js(expression))return;await delay(150);}throw Error('Timeout '+expression+'\n'+await js('document.body.innerText'));};
    const click=async label=>{await js(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(label)}||b.getAttribute('aria-label')===${JSON.stringify(label)});if(!b||b.disabled)throw Error('Missing enabled '+${JSON.stringify(label)});b.click();})()`);await delay(100);};
    const ready=()=>wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')&&document.querySelector('.rl-viewer').dataset.surfacesPending==='0'&&![...document.querySelectorAll('.surgical-panel [role="status"]')].some(n=>/^(Computing instrument action|Preparing an exact recorded state|Preparing accepted display|Verifying saved source-bound rehearsal)/.test(n.textContent.trim()))`);
    const state=()=>js(`(()=>{const p=document.querySelector('[data-operative-view="surgeon"][data-surgical-state-id]');return p?{id:p.dataset.surgicalStateId,removed:Number(p.dataset.removedCells),camera:JSON.parse(p.dataset.opticalCamera),readout:document.querySelector('[data-operative-readout-state]')?.dataset.operativeReadoutState,target:Number(document.querySelector('[data-surgical-metric="targetRemovedMm3"]').textContent),position:Number(document.querySelector('[aria-label="Surgical timeline"]').value)}:null})()`);
    const select=async name=>{await js(`(()=>{const s=document.querySelector('[aria-label="Surgical branch"]');s.value=${JSON.stringify(name)};s.dispatchEvent(new Event('change',{bubbles:true}));})()`);await ready();};
    const seek=async position=>{await js(`(()=>{const s=document.querySelector('[aria-label="Surgical timeline"]');if(s.disabled)throw Error('Timeline disabled');Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(s,${position});s.dispatchEvent(new Event('input',{bubbles:true}));s.dispatchEvent(new Event('change',{bubbles:true}));})()`);await wait(`Number(document.querySelector('[aria-label="Surgical timeline"]').value)===${position}`);await ready();};
    const capture=async name=>{
      await ready();await wait(`(()=>{const p=document.querySelector('[data-operative-view="surgeon"][data-surgical-state-id]'),f=window.vascularFrames?.at(-1);return !!p&&!!f&&!f.interactive&&f.state===p.dataset.surgicalStateId&&window.vascularGpu?.includes(f.serial)&&document.querySelector('[data-operative-readout-state]')?.dataset.operativeReadoutState===p.dataset.surgicalStateId})()`);
      const before=await state();const png=(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG();
      assert.deepEqual(await state(),before);assert.equal(before.id,before.readout);
      fs.writeFileSync(path.join(output,name+'.png'),png);report.states.push({name,...before});return before;
    };
    try{
      await js(`window.vascularFrames=[];window.vascularGpu=[];document.addEventListener('anatomy-frame',e=>{window.vascularFrames.push({serial:e.detail.serial,state:e.detail.surgicalStateId,interactive:e.detail.interactive});if(window.vascularFrames.length>512)window.vascularFrames.shift();});document.addEventListener('anatomy-gpu',e=>{if(e.detail.completed)window.vascularGpu.push(e.detail.serial);if(window.vascularGpu.length>512)window.vascularGpu.shift();});`);
      await wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`);await click('Rehearse');await ready();
      if(process.env.VASCULAR_REOPEN_EXPECT){
        const expected=JSON.parse(fs.readFileSync(process.env.VASCULAR_REOPEN_EXPECT,'utf8'));
        const current=await capture('01-cold-reopened');
        for(const key of ['id','removed','target','position','camera'])assert.deepEqual(current[key],expected.savedView[key],key+' survives native cold reopen');
        report.checks.push('Exact saved graph state, source counts, cursor and microscope optics survive cold reopen');
      }else{
        if(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify replay for planning')`)){await click('Verify replay for planning');await ready();}
        await click('Surgeon / Microscope');await ready();
        await js(`(()=>{const s=[...document.querySelectorAll('summary')].find(s=>s.textContent.trim()==='Source provenance and research assumptions');if(!s)throw Error('Missing provenance inspector');if(!s.parentElement.open)s.click();})()`);
        await wait(`document.body.innerText.includes('declared vascular graph separates compression')`);
        report.checks.push('Native inspector identifies the opted-in partial vascular model');
        let camera;
        for(const branch of prepared.branches){
          await select(branch.branch);await seek(branch.actions);const current=await capture(branch.branch);
          assert.equal(current.id,branch.stateId);assert.equal(current.removed,branch.removed.sourceCellCount);assert.equal(current.target,branch.removed.targetRemovedMm3);
          if(camera)assert.deepEqual(current.camera,camera);else camera=current.camera;
        }
        report.checks.push('All four exact branch endpoints and source/target counts agree with independent headless report; fixed microscope optics and synchronized GPU/HUD');
        await js(`(()=>{const s=[...document.querySelectorAll('summary')].find(s=>s.textContent.trim()==='Recorded V2 mechanics and fluid budget');if(!s)throw Error('Missing model budget');if(!s.parentElement.open)s.click();})()`);
        await wait(`document.body.innerText.includes('Pressure-supporting vascular storage')&&document.body.innerText.includes('no injury calibration')`);
        report.checks.push('Graph storage, supply/drainage and no-injury-calibration disclosures are available');
        report.savedView=await state();await click('Save');await wait(`document.body.innerText.includes('Workspace saved locally')`);
        assert(fs.existsSync(saved));report.saved=saved;report.checks.push('Native graph workspace save succeeds');
      }
      report.sourcePreserved=hash(input)===original;assert(report.sourcePreserved);report.passed=!report.windowEverVisible;
      fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,checks:report.checks,temporaryProfile:profile}));app.exit(report.passed?0:1);
    }catch(error){report.errors.push(String(error));report.body=await js('document.body.innerText').catch(()=>null);fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.error(error);app.exit(1);}
  });
});
require('./main.cjs');
