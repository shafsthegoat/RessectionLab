'use strict';
// Real Electron, native engine and hidden GPU captures. Never show()/focus().
const {app,dialog}=require('electron');
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..');
const input=path.resolve(process.env.SURGEON_CASE||''),output=path.resolve(process.env.SURGEON_OUTPUT||'');
for(const value of [input,output])if(!value.startsWith(path.join(root,'testcases')+path.sep))throw Error('Use testcases input/output');
if(fs.existsSync(output))throw Error('Preserve prior evidence; choose a new output');fs.mkdirSync(output,{recursive:true});
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex'),original=sha(input);
app.setPath('userData',fs.mkdtempSync(path.join(os.tmpdir(),'rslab-surgeon-')));
process.argv.push('--case',input,'--automation-hidden');
const saved=path.join(output,'surgeon-rehearsal.ressectionlab');
dialog.showSaveDialog=async()=>({canceled:false,filePath:saved});
dialog.showOpenDialog=async()=>({canceled:false,filePaths:[saved]});
const report={input,output,windowMode:'hidden',windowEverVisible:false,checks:[],captures:[],errors:[]};
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
app.on('browser-window-created',(_event,window)=>{
  window.on('show',()=>{report.windowEverVisible=true;window.hide();});
  window.webContents.once('did-finish-load',async()=>{
    window.webContents.setBackgroundThrottling(false);window.setSize(1600,1050);
    const js=code=>window.webContents.executeJavaScript(code,true);
    const wait=async(expression,seconds=180)=>{const start=Date.now();while(Date.now()-start<seconds*1000){if(await js(expression))return;await delay(150);}throw Error('Timeout '+expression+'\n'+await js('document.body.innerText'));};
    const click=async label=>{await js(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(label)}||b.getAttribute('aria-label')===${JSON.stringify(label)});if(!b||b.disabled)throw Error('Missing enabled button '+${JSON.stringify(label)});b.click();})()`);await delay(100);};
    const ready=()=>wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')&&document.querySelector('.rl-viewer').dataset.surfacesPending==='0'&&![...document.querySelectorAll('.surgical-panel [role="status"]')].some(n=>/^(Computing instrument action|Preparing an exact recorded state|Preparing accepted display|Verifying saved source-bound rehearsal)/.test(n.textContent.trim()))`);
    const capture=async name=>{await ready();await delay(250);fs.writeFileSync(path.join(output,name+'.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());report.captures.push(name+'.png');};
    const optical=()=>js(`(()=>{const pane=document.querySelector('[data-operative-view="surgeon"][data-optical-camera]');return pane?{camera:JSON.parse(pane.dataset.opticalCamera),state:pane.dataset.surgicalStateId,removed:Number(pane.dataset.removedCells),tools:JSON.parse(pane.dataset.acceptedTools)}:null})()`);
    const seek=async position=>{await js(`(()=>{const s=document.querySelector('[aria-label="Surgical timeline"]');Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(s,${position});s.dispatchEvent(new Event('input',{bubbles:true}));s.dispatchEvent(new Event('change',{bubbles:true}));})()`);await wait(`Number(document.querySelector('[aria-label="Surgical timeline"]').value)===${position}`);await ready();};
    try{
      await js(`window.surgeonRenderFrames=[];document.addEventListener('anatomy-frame',e=>{const d=e.detail;if(window.surgeonRenderFrames.length<4096)window.surgeonRenderFrames.push({at:d.at,cpuMs:d.cpuMs,displayWork:d.displayWork,drawCalls:d.drawCalls,resources:d.resources,state:d.surgicalStateId});});`);
      await wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`);await click('Rehearse');await ready();
      if(process.env.SURGEON_REOPEN==='1'){
        report.reopened=await optical();assert(report.reopened,'Microscope selection must survive a cold application restart');
        report.position=await js(`Number(document.querySelector('[aria-label="Surgical timeline"]').value)`);
        report.target=await js(`Number(document.querySelector('[data-surgical-metric="targetRemovedMm3"]').textContent)`);
        if(process.env.SURGEON_EXPECT){const expected=JSON.parse(fs.readFileSync(process.env.SURGEON_EXPECT,'utf8'));assert.deepEqual(report.reopened.camera,expected.savedView.camera);assert.equal(report.reopened.state,expected.savedView.state);assert.equal(report.reopened.removed,expected.savedView.removed);}
        await capture('01-cold-reopened-surgeon');report.sourcePreserved=sha(input)===original;assert(report.sourcePreserved);report.checks.push('Cold Electron restart restores optical settings and exact native cursor');
        report.passed=!report.windowEverVisible;fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));app.exit(report.passed?0:1);return;
      }
      if(process.env.SURGEON_PREPARED==='1'){
        if(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify replay for planning')`)){await click('Verify replay for planning');await ready();}
      }else{await wait(`!!document.querySelector('.patient-cusa-learning')`);await click('Inspect PPO CUSA execution');await ready();}
      report.start=await js(`Number(document.querySelector('[aria-label="Surgical timeline"]').value)`);
      await click('Surgeon / Microscope');await ready();
      // Straight down the actual checked access corridor; oblique angles remain user adjustable.
      await js(`(()=>{const s=document.querySelector('[aria-label="Viewing angle"]');Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(s,${JSON.stringify(process.env.SURGEON_PREPARED==='1'?'12':'0')});s.dispatchEvent(new Event('input',{bubbles:true}));s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
      await ready();report.before=await optical();assert(report.before);assert.equal(Object.keys(report.before.tools).length,2);if(process.env.SURGEON_BIMANUAL==='1')assert(Object.values(report.before.tools).every(s=>s.inserted),'Both hands must complete native checked insertion');
      await js(`document.querySelector('.surgeon-view').scrollIntoView({block:'start'})`);await capture('01-fixed-surgeon-before');
      const metric=()=>js(`Number(document.querySelector('[data-surgical-metric="targetRemovedMm3"]').textContent)`);report.targetBefore=await metric();
      const images=new Map();report.observedFrames=[];const recordingStart=Date.now();
      await click('Play replay');
      while(await js(`Number(document.querySelector('[aria-label="Surgical timeline"]').value)<${report.start+3}`)){
        if(Date.now()-recordingStart>120000)throw Error('Observed replay recording timed out');
        if(process.env.SURGEON_RECORD==='1'&&await js(`document.querySelector('.rl-viewer').dataset.surfacesPending==='0'`)){
          const observed=await optical();
          if(observed&&observed.state!==report.observedFrames.at(-1)?.state){
            const png=(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG();
            const digest=crypto.createHash('sha256').update(png).digest('hex');let file=images.get(digest);
            if(!file){file='observed-'+String(images.size).padStart(4,'0')+'.png';fs.writeFileSync(path.join(output,file),png);images.set(digest,file);}
            report.observedFrames.push({state:observed.state,removed:observed.removed,wallMs:Date.now()-recordingStart,file});
          }
        }
        await delay(150);
      }
      await click('Pause replay');await ready();
      report.after=await optical();report.targetAfter=await metric();assert.deepEqual(report.after.camera,report.before.camera,'Camera must remain fixed during tool motion');assert(report.targetAfter>report.targetBefore);assert(report.after.removed>report.before.removed);assert.notEqual(report.after.state,report.before.state);
      report.pausedAt=await js(`Number(document.querySelector('[aria-label="Surgical timeline"]').value)`);await delay(500);assert.equal(await js(`Number(document.querySelector('[aria-label="Surgical timeline"]').value)`),report.pausedAt);
      report.render=await js(`(()=>{const f=window.surgeonRenderFrames,cpu=f.map(x=>x.cpuMs).sort((a,b)=>a-b);return {frames:f.length,cpuSubmissionMedianMs:cpu[Math.floor(cpu.length*.5)],cpuSubmissionP95Ms:cpu[Math.floor(cpu.length*.95)],first:f[0],last:f.at(-1)};})()`);
      // Observed microsteps can already be inside the next action when the
      // boundary counter advances. Compare scrub against the exact finalized
      // action boundary, separately from the paused observed state above.
      await seek(report.start);await seek(report.pausedAt);report.afterBoundary=await optical();
      assert(report.afterBoundary.removed>report.before.removed);
      // A real exchange requested while inserted must be refused in the native engine.
      if(process.env.SURGEON_PREPARED==='1'){
        await wait(`[...document.querySelectorAll('.surgical-toolbox button')].some(b=>b.textContent.includes('Cranial drill')&&!b.disabled)`);
        const beforeBlocked=await optical();await js(`(()=>{const b=[...document.querySelectorAll('.surgical-toolbox button')].find(b=>b.textContent.includes('Cranial drill'));if(!b||b.disabled)throw Error('Missing exchange control');b.click();})()`);await ready();
        await wait(`document.body.innerText.includes('WITHDRAW_BEFORE_EXCHANGE')`);assert.equal((await optical()).state,beforeBlocked.state);report.checks.push('Inserted-tool exchange is blocked without changing source state');
      }
      await capture('02-growing-native-cavity');report.checks.push('Fixed surgeon camera, both recorded tool poses, native removal and growing cavity during observed replay');
      await seek(report.start);const rewound=await optical();assert.equal(rewound.removed,report.before.removed);assert.deepEqual(rewound.camera,report.before.camera);await seek(report.pausedAt);assert.equal((await optical()).removed,report.afterBoundary.removed);report.checks.push('Scrub restores exact before/after native anatomy without changing optical view');
      await click('External anatomy view');await ready();await capture('03-external-residual');await click('Surgeon / Microscope');await ready();
      await js(`document.querySelector('.patient-position').open=true`);
      if(process.env.SURGEON_PREPARED==='1'){
        const selectBranch=async name=>{await js(`(()=>{const s=document.querySelector('[aria-label="Surgical branch"]');s.value=${JSON.stringify(name)};s.dispatchEvent(new Event('change',{bubbles:true}));})()`);await wait(`document.querySelector('[aria-label="Surgical branch"]').value===${JSON.stringify(name)}`);await ready();const end=await js(`Number(document.querySelector('[aria-label="Surgical timeline"]').max)`);await seek(end);};
        const effects=()=>js(`(()=>{const p=document.querySelector('[data-position-effects]');return {state:p.dataset.surgicalStateId,effects:JSON.parse(p.dataset.positionEffects),shift:Number(p.dataset.maxShiftMm)}})()`);
        await selectBranch('position-supine');report.supine=await effects();await capture('04-supine-gravity');
        await selectBranch('position-semisitting');report.semisitting=await effects();await capture('05-semisitting-gravity');
        assert.notDeepEqual(report.supine.effects.gravityRasMPerS2,report.semisitting.effects.gravityRasMPerS2);assert.notEqual(report.supine.effects.hydrostaticSiteMinusHeartPa,report.semisitting.effects.hydrostaticSiteMinusHeartPa);assert(report.supine.shift>0&&report.semisitting.shift>0);
        const positioned=await effects();const currentAngle=await js(`document.querySelector('[aria-label="Viewing angle"]').value`);
        await js(`(()=>{const s=document.querySelector('[aria-label="Viewing angle"]');Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(s,'25');s.dispatchEvent(new Event('input',{bubbles:true}));s.dispatchEvent(new Event('change',{bubbles:true}));})()`);await ready();assert.deepEqual(await effects(),positioned,'Optical adjustment cannot change physiology');
        await click('Undo');await ready();await click('Redo');await ready();assert.deepEqual(await effects(),positioned);
        const branchBefore=await js(`document.querySelector('[aria-label="Surgical branch"]').value`);await click('Branch');await ready();assert.notEqual(await js(`document.querySelector('[aria-label="Surgical branch"]').value`),branchBefore);assert.equal((await effects()).state,positioned.state);
        // Apply a new posture through the actual UI from this branch.
        await js(`(()=>{const s=document.querySelector('[aria-label="Patient position preset"]');s.value='prone';s.dispatchEvent(new Event('change',{bubbles:true}));})()`);await click('Apply position and settle 1 s');await ready();report.appliedPosition=await effects();assert.equal(report.appliedPosition.effects.preset,'prone');
        await js(`document.querySelector('.patient-position').scrollIntoView({block:'start'})`);await capture('06-actual-position-action');
        report.checks.push('Same-source posture branches, distinct gravity/hydrostatic/shift states, optical invariance, Undo/Redo, branching and actual UI position action');
        await selectBranch('surgeon-cusa');await seek(report.pausedAt);
      }
      report.savedView=await optical();
      await click('Save');await wait(`document.body.innerText.includes('Workspace saved locally')`);assert(fs.existsSync(saved));
      report.saved=saved;report.sourcePreserved=sha(input)===original;assert(report.sourcePreserved);report.checks.push('Native save retains inspected episode and microscope settings');
      report.passed=!report.windowEverVisible;fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));app.exit(report.passed?0:1);
    }catch(error){report.errors.push(String(error));report.body=await js('document.body.innerText');await capture('failure').catch(()=>{});fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.error(error);app.exit(1);}
  });
});
require('./main.cjs');
