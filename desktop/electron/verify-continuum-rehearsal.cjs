'use strict';
// Actual native application, always hidden. No show()/focus() or recording.
const {app,dialog}=require('electron');
const fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..');
const input=path.resolve(process.env.CONTINUUM_CASE||''),output=path.resolve(process.env.CONTINUUM_OUTPUT||'');
for(const p of [input,output])if(!p.startsWith(path.join(root,'testcases')+path.sep))throw Error('Use ignored testcases paths');
if(fs.existsSync(output))throw Error('Preserve prior evidence');fs.mkdirSync(output,{recursive:true});
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex'),original=sha(input);
const prepared=JSON.parse(fs.readFileSync(process.env.CONTINUUM_REPORT||path.join(path.dirname(input),'report.json'),'utf8'));
const saved=path.join(output,'continuum-saved.ressectionlab');
app.setPath('userData',fs.mkdtempSync(path.join(os.tmpdir(),'rslab-continuum-')));
process.argv.push('--case',input,'--automation-hidden');
dialog.showSaveDialog=async()=>({canceled:false,filePath:saved});
const report={input,output,windowEverVisible:false,states:[],checks:[],errors:[]};
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
app.on('browser-window-created',(_event,window)=>{
  window.on('show',()=>{report.windowEverVisible=true;window.hide();});
  window.webContents.once('did-finish-load',async()=>{
    window.webContents.setBackgroundThrottling(false);window.setSize(1600,1050);
    const js=code=>window.webContents.executeJavaScript(code,true);
    const wait=async(expression,seconds=300)=>{const t=Date.now();while(Date.now()-t<seconds*1000){if(await js(expression))return;await delay(150);}throw Error('Timeout '+expression+'\n'+await js('document.body.innerText'));};
    const click=async label=>{await js(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(label)}||b.getAttribute('aria-label')===${JSON.stringify(label)});if(!b||b.disabled)throw Error('Missing enabled '+${JSON.stringify(label)});b.click();})()`);await delay(100);};
    const ready=()=>wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')&&document.querySelector('.rl-viewer').dataset.surfacesPending==='0'&&![...document.querySelectorAll('.surgical-panel [role="status"]')].some(n=>/^(Computing instrument action|Preparing an exact recorded state|Preparing accepted display|Verifying saved source-bound rehearsal)/.test(n.textContent.trim()))`);
    const state=()=>js(`(()=>{const p=document.querySelector('[data-operative-view="surgeon"][data-surgical-state-id]');return p?{id:p.dataset.surgicalStateId,removed:Number(p.dataset.removedCells),camera:JSON.parse(p.dataset.opticalCamera),readout:document.querySelector('[data-operative-readout-state]')?.dataset.operativeReadoutState,target:Number(document.querySelector('[data-surgical-metric="targetRemovedMm3"]').textContent),position:Number(document.querySelector('[aria-label="Surgical timeline"]').value)}:null})()`);
    const seek=async position=>{await js(`(()=>{const s=document.querySelector('[aria-label="Surgical timeline"]');if(s.disabled)throw Error('Timeline remains disabled');Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(s,${position});s.dispatchEvent(new Event('input',{bubbles:true}));s.dispatchEvent(new Event('change',{bubbles:true}));})()`);await wait(`Number(document.querySelector('[aria-label="Surgical timeline"]').value)===${position}`);await ready();};
    const capture=async name=>{
      await ready();
      await wait(`(()=>{const p=document.querySelector('[data-operative-view="surgeon"][data-surgical-state-id]'),f=window.continuumFrames?.at(-1);return !!p&&!!f&&!f.interactive&&f.state===p.dataset.surgicalStateId&&window.continuumGpu?.includes(f.serial)&&document.querySelector('[data-operative-readout-state]')?.dataset.operativeReadoutState===p.dataset.surgicalStateId})()`);
      const before=await state();const png=(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG();
      assert.deepEqual(await state(),before);assert.equal(before.id,before.readout);
      fs.writeFileSync(path.join(output,name+'.png'),png);report.states.push({name,...before});return before;
    };
    try{
      await js(`window.continuumFrames=[];window.continuumGpu=[];document.addEventListener('anatomy-frame',e=>{window.continuumFrames.push({serial:e.detail.serial,state:e.detail.surgicalStateId,interactive:e.detail.interactive});if(window.continuumFrames.length>512)window.continuumFrames.shift();});document.addEventListener('anatomy-gpu',e=>{if(e.detail.completed)window.continuumGpu.push(e.detail.serial);if(window.continuumGpu.length>512)window.continuumGpu.shift();});`);
      await wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`);await click('Rehearse');await ready();
      if(process.env.CONTINUUM_REOPEN_EXPECT){
        const expected=JSON.parse(fs.readFileSync(process.env.CONTINUUM_REOPEN_EXPECT,'utf8'));
        const current=await capture('01-cold-reopened');
        for(const key of ['id','removed','target','position','camera'])assert.deepEqual(current[key],expected.savedView[key],key+' survives cold native reopen');
        report.sourcePreserved=sha(input)===original;assert(report.sourcePreserved);
        report.checks.push('Cold native reopen preserves exact continuum state, source counts, cursor and microscope optics');
        report.passed=!report.windowEverVisible;fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));
        console.log(JSON.stringify({passed:report.passed,checks:report.checks}));app.exit(report.passed?0:1);return;
      }
      if(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify replay for planning')`)){await click('Verify replay for planning');await ready();}
      await click('Surgeon / Microscope');await ready();
      await js(`(()=>{const s=[...document.querySelectorAll('summary')].find(s=>s.textContent.trim()==='Source provenance and research assumptions');if(!s)throw Error('Missing provenance inspector');if(!s.parentElement.open)s.click();})()`);
      await wait(`document.body.innerText.includes('Opted-in voxel continuum')`);
      report.checks.push('Native application declares the actual opted-in continuum model');
      const select=async name=>{await js(`(()=>{const s=document.querySelector('[aria-label="Surgical branch"]');s.value=${JSON.stringify(name)};s.dispatchEvent(new Event('change',{bubbles:true}));})()`);await ready();};
      await select('short-hold');await seek(prepared.commonActions);const initial=await capture('01-common-held-state');
      for(const expected of prepared.branches){
        await select(expected.branch);await seek(expected.nativeActions);
        const current=await capture(expected.branch==='short-hold'?'02-short-hold-end':'03-sustained-hold-end');
        assert.equal(current.id,expected.finalStateId);assert.equal(current.target,expected.fullEpisode.targetRemovedMm3);
        assert.equal(current.removed,expected.fullEpisode.sourceCellCount);assert.deepEqual(current.camera,initial.camera);
      }
      report.checks.push('Both exact native continuum endpoints, target/source-cell counts, synchronized HUD and fixed microscope camera');
      await select('stop-at-model-limit');await seek(prepared.commonActions);const stoppedPrefix=await state();
      assert.equal(stoppedPrefix.id,initial.id);
      await seek(prepared.stop.nativeActions);const stopped=await capture('04-stop-cleanup');
      assert.equal(stopped.id,prepared.stop.finalStateId);assert.equal(stopped.removed,initial.removed);
      report.checks.push('STOP preserves the refused-action prefix, then releases and withdraws through checked replay');
      await select('short-hold');await seek(prepared.branches.find(b=>b.branch==='short-hold').nativeActions);
      report.savedView=await state();await click('Save');await wait(`document.body.innerText.includes('Workspace saved locally')`);
      assert(fs.existsSync(saved));report.saved=saved;report.sourcePreserved=sha(input)===original;assert(report.sourcePreserved);
      report.passed=!report.windowEverVisible;report.checks.push('Native save succeeds and source archive is unchanged');
      fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,saved,checks:report.checks}));app.exit(report.passed?0:1);
    }catch(error){report.errors.push(String(error));report.body=await js('document.body.innerText').catch(()=>null);fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.error(error);app.exit(1);}
  });
});
require('./main.cjs');
