'use strict';
// Actual GPU frames from native recorded states, always hidden.
const {app,dialog}=require('electron');
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..'),input=path.resolve(process.env.OPERATIVE_CASE||''),output=path.resolve(process.env.OPERATIVE_OUTPUT||'');
for(const p of [input,output])if(!p.startsWith(path.join(root,'testcases')+path.sep))throw Error('Use testcases paths');
if(fs.existsSync(output))throw Error('Preserve previous evidence');fs.mkdirSync(output,{recursive:true});
const preparation=JSON.parse(fs.readFileSync(process.env.OPERATIVE_PREPARATION||path.join(path.dirname(input),'preparation.json'),'utf8'));
const sourceHash=crypto.createHash('sha256').update(fs.readFileSync(input)).digest('hex');
const userData=fs.mkdtempSync(path.join(os.tmpdir(),'rslab-operative-'));
app.setPath('userData',userData);process.argv.push('--case',input,'--automation-hidden');
const saved=path.join(output,'operative-saved.ressectionlab');dialog.showSaveDialog=async()=>({canceled:false,filePath:saved});
const report={input,output,userData,windowEverVisible:false,checks:[],captures:[],states:[],errors:[],visualUpdates:[],
 timingScope:'Hidden native seek through synchronized source state, HUD, final CPU frame and completed GPU query; includes cached/no-op seeks and polling, not physical screen presentation.'};const delay=ms=>new Promise(r=>setTimeout(r,ms));
app.on('browser-window-created',(_event,window)=>{
 window.on('show',()=>{report.windowEverVisible=true;window.hide();});
 window.webContents.once('did-finish-load',async()=>{
  window.webContents.setBackgroundThrottling(false);window.setSize(1600,1050);
  const js=s=>window.webContents.executeJavaScript(s,true);
  const wait=async(s,seconds=180)=>{const t=Date.now();while(Date.now()-t<seconds*1000){if(await js(s))return;await delay(100);}throw Error('Timeout '+s+'\n'+await js('document.body.innerText'));};
  const click=async name=>{await js(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(name)}||b.getAttribute('aria-label')===${JSON.stringify(name)});if(!b||b.disabled)throw Error('Missing enabled '+${JSON.stringify(name)});b.click();})()`);await delay(100);};
  const ready=()=>wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')&&document.querySelector('.rl-viewer').dataset.surfacesPending==='0'&&![...document.querySelectorAll('.surgical-panel [role="status"]')].some(n=>/^(Computing instrument action|Preparing an exact recorded state|Preparing accepted display|Verifying saved source-bound rehearsal)/.test(n.textContent.trim()))`);
  const setRange=async(name,value)=>{await js(`(()=>{const s=document.querySelector('[aria-label=${JSON.stringify(name)}]');Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(s,${JSON.stringify(String(value))});s.dispatchEvent(new Event('input',{bubbles:true}));s.dispatchEvent(new Event('change',{bubbles:true}));})()`);await delay(100);};
  const state=()=>js(`(()=>{const p=document.querySelector('[data-surgical-state-id]');return {id:p.dataset.surgicalStateId,removed:Number(p.dataset.removedCells),camera:JSON.parse(p.dataset.opticalCamera||'null'),readoutState:document.querySelector('[data-operative-readout-state]')?.dataset.operativeReadoutState,readout:document.querySelector('.operative-readout')?.innerText,activation:[...document.querySelectorAll('.surgical-slots button')].map(b=>b.innerText),fluid:Number(document.querySelector('[data-surgical-metric="fluidPresentMm3"]').textContent),tools:JSON.parse(p.dataset.acceptedTools),target:Number(document.querySelector('[data-surgical-metric="targetRemovedMm3"]').textContent),position:Number(document.querySelector('[aria-label="Surgical timeline"]').value),frame:window.operativeFrames?.at(-1)}})()`);
  const synchronized=()=>wait(`(()=>{const p=document.querySelector('[data-surgical-state-id]'),f=window.operativeFrames?.at(-1);return !!p&&!!f&&!f.interactive&&f.state===p.dataset.surgicalStateId&&window.operativeGpu?.some(g=>g.completed&&g.serial===f.serial)&&document.querySelector('[data-operative-readout-state]')?.dataset.operativeReadoutState===p.dataset.surgicalStateId})()`);
  const seek=async position=>{const started=performance.now();await setRange('Surgical timeline',position);await wait(`!document.querySelector('[aria-label="Observed microstep replay"]')&&Number(document.querySelector('[aria-label="Surgical timeline"]').value)===${position}`);await ready();await synchronized();report.visualUpdates.push({position,latencyMs:performance.now()-started,state:(await state()).id});};
  const capture=async name=>{await ready();await synchronized();await delay(250);const s=await state();assert.equal(s.readoutState,s.id,'HUD and accepted surface must publish the same native state');report.states.push({name,...s});
    fs.writeFileSync(path.join(output,name+'.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());
    const rect=await js(`(()=>{const r=document.querySelector('.rl-viewer-anatomy-image').getBoundingClientRect();return{x:Math.ceil(r.x),y:Math.ceil(r.y),width:Math.floor(r.width),height:Math.floor(r.height)}})()`);
    fs.writeFileSync(path.join(output,name+'-field.png'),(await window.webContents.capturePage(rect,{stayHidden:true,stayAwake:true})).toPNG());report.captures.push(name+'.png');return s;
  };
  try{
    report.gpu=app.getGPUFeatureStatus();assert(!String(report.gpu.webgl).includes('disabled'),'GPU WebGL must be enabled');
    await js(`window.operativeFrames=[];window.operativeGpu=[];window.operativeWork=[];document.addEventListener('anatomy-gpu',e=>window.operativeGpu.push(e.detail));document.addEventListener('anatomy-work',e=>window.operativeWork.push(e.detail));document.addEventListener('anatomy-frame',e=>{const d=e.detail;window.operativeFrames.push({at:d.at,serial:d.serial,interactive:d.interactive,cpuMs:d.cpuMs,state:d.surgicalStateId,frame:d.surgicalFrame,displayWork:d.displayWork,drawCalls:d.drawCalls,gpuMs:d.gpuMs,sourceCaseHash:d.sourceCaseHash});if(window.operativeFrames.length>5000)window.operativeFrames.shift();});`);
    await wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`);await click('Rehearse');await ready();
    if(process.env.OPERATIVE_REOPEN==='1'){
      const expected=JSON.parse(fs.readFileSync(process.env.OPERATIVE_EXPECT,'utf8'));
      report.reopened=await capture('01-cold-reopened');
      for(const key of ['id','removed','target','position','camera','tools'])assert.deepEqual(report.reopened[key],expected.savedView[key],key+' survives cold restart');
      assert(await js(`document.querySelector('[aria-label="Operative surface material"]').value==='operative'`));
      const beforeBlocked=(await state()).id;
      await js(`(()=>{const b=[...document.querySelectorAll('.surgical-toolbox button')].find(b=>b.textContent.includes('Cranial drill'));if(!b||b.disabled)throw Error('Missing exchange');b.click();})()`);await ready();await wait(`document.body.innerText.includes('Withdraw this instrument before exchanging it.')`);assert.equal((await state()).id,beforeBlocked);await capture('02-readable-blocked-action');
      report.checks.push('Fresh Electron process restores exact native cursor, anatomy, tools and independent optical settings');
      report.sourcePreserved=crypto.createHash('sha256').update(fs.readFileSync(input)).digest('hex')===sourceHash;assert(report.sourcePreserved);report.passed=!report.windowEverVisible;
      fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,reopened:report.reopened.id}));app.exit(report.passed?0:1);return;
    }
    await click('Surgeon / Microscope');
    if(process.env.OPERATIVE_RAW==='1'){await js(`(()=>{const s=[...document.querySelectorAll('label')].find(l=>l.textContent.trim()==='Smooth 3D surfaces')?.querySelector('input');if(!s)throw Error('Missing surface control');if(s.checked)s.click();})()`);await ready();}
    await setRange('Viewing angle',16);await setRange('Field / magnification',26);await setRange('Focus plane offset',-2.5);await setRange('Operative light',1.8);await ready();
    await js(`document.querySelector('.surgeon-view').scrollIntoView({block:'start'})`);
    await seek(preparation.workStart);report.initial=await capture('01-before');
    if(process.env.OPERATIVE_RECORD==='1'){
      await js(`(()=>{const s=document.querySelector('[aria-label="Replay speed"]');s.value='4';s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
      report.observed=[];const started=Date.now();let lastProgress=0,paused=false;
      await click('Play replay');
      while(true){
        if(Date.now()-started>1800000)throw Error('Replay exceeded 30 minutes');
        const s=await state();
        if(s.frame?.state===s.id&&s.readoutState===s.id&&s.id!==report.observed.at(-1)?.id&&await js(`document.querySelector('.rl-viewer').dataset.surfacesPending==='0'`)){
          const rect=await js(`(()=>{const r=document.querySelector('.rl-viewer-anatomy-image').getBoundingClientRect();return{x:Math.ceil(r.x),y:Math.ceil(r.y),width:Math.floor(r.width),height:Math.floor(r.height)}})()`);
          const png=(await window.webContents.capturePage(rect,{stayHidden:true,stayAwake:true})).toPNG(),confirmed=await state();
          if(confirmed.id===s.id&&confirmed.frame?.state===s.id){const file='observed-'+String(report.observed.length).padStart(5,'0')+'.png';fs.writeFileSync(path.join(output,file),png);report.observed.push({...s,file,wallMs:Date.now()-started});}
        }
        if(!paused&&s.position>=preparation.workStart+4){
          await click('Pause replay');await ready();const stopped=await state();await delay(500);assert.equal((await state()).id,stopped.id);report.paused=stopped;paused=true;await click('Play replay');
        }
        if(s.position>=preparation.workEnd+3)break;
        if(Date.now()-lastProgress>10000){lastProgress=Date.now();fs.writeFileSync(path.join(output,'playback-progress.json'),JSON.stringify({wallSeconds:(Date.now()-started)/1000,captured:report.observed.length,...s},null,2));}
        if(await js(`![...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Pause replay')`))throw Error('Replay stopped before the end');
        await delay(75);
      }
      await click('Pause replay');await ready();report.playbackWallSeconds=(Date.now()-started)/1000;report.checks.push('Real automatic native microstep play/pause/resume; sampled GPU captures have matching surface, frame and volume identities');
      assert(report.observed.length>20);assert(report.observed.some(s=>s.fluid>1)&&report.observed.some(s=>s.fluid===0));assert(report.observed.at(-1).target>report.observed[0].target);
      fs.writeFileSync(path.join(output,'observed.json'),JSON.stringify(report.observed,null,2));
    }
    const progress=preparation.rows.filter(r=>r.index>preparation.workStart&&r.index<=preparation.workEnd&&r.targetDeltaMm3>0);
    const selected=progress.filter((r,i)=>i===0||i===Math.floor(progress.length/3)||i===Math.floor(progress.length*2/3)||i===progress.length-1);
    for(let i=0;i<selected.length;i++){await seek(selected[i].index);await capture('02-progress-'+String(i).padStart(2,'0'));}
    await seek(preparation.workEnd);report.after=await capture('03-after');
    await seek(preparation.workEnd+3);report.afterClear=await capture('03b-after-tools-clear');assert.equal(report.afterClear.target,report.after.target);
    assert.deepEqual(report.after.camera,report.initial.camera);assert(report.after.target-report.initial.target>=100);report.checks.push('Fixed camera and native positive target removal across source-bound intermediate states');
    await seek(preparation.workStart);const restored=await state();assert.equal(restored.id,report.initial.id);assert.equal(restored.removed,report.initial.removed);await capture('04-scrub-restored');
    await seek(preparation.workEnd);assert.equal((await state()).id,report.after.id);
    const box=await js(`(()=>{const l=[...document.querySelectorAll('.surgeon-view label')].find(l=>l.textContent.includes('Inspection-only transparency'));return !!l?.querySelector('input')})()`);assert(box);
    await js(`(()=>{const l=[...document.querySelectorAll('.surgeon-view label')].find(l=>l.textContent.includes('Inspection-only transparency'));l.querySelector('input').click();})()`);await capture('05-inspection-only');assert.equal((await state()).id,report.after.id);
    await js(`(()=>{const l=[...document.querySelectorAll('.surgeon-view label')].find(l=>l.textContent.includes('Inspection-only transparency'));l.querySelector('input').click();})()`);await ready();
    if(process.env.OPERATIVE_RECORD==='1'){
      // Actual native refusal with inserted tools, followed by editable actions.
      await js(`(()=>{const b=[...document.querySelectorAll('.surgical-toolbox button')].find(b=>b.textContent.includes('Cranial drill'));if(!b||b.disabled)throw Error('Missing exchange');b.click();})()`);await ready();await wait(`document.body.innerText.includes('WITHDRAW_BEFORE_EXCHANGE')`);assert.equal((await state()).id,report.after.id);await capture('06-blocked-exchange');
      await click('External anatomy view');await ready();await click('Surgeon / Microscope');await ready();assert.equal((await state()).id,report.after.id);
      await js(`document.querySelector('.patient-position').open=true`);
      const effects=()=>js(`(()=>{const p=document.querySelector('[data-position-effects]');return {state:p.dataset.surgicalStateId,effects:JSON.parse(p.dataset.positionEffects),shift:Number(p.dataset.maxShiftMm)}})()`);
      const branch=async name=>{await js(`(()=>{const s=document.querySelector('[aria-label="Surgical branch"]');s.value=${JSON.stringify(name)};s.dispatchEvent(new Event('change',{bubbles:true}));})()`);await ready();await seek(await js(`Number(document.querySelector('[aria-label="Surgical timeline"]').max)`));};
      report.postures=[];
      for(const preset of ['supine','semisitting','left_lateral']){await branch('position-'+preset);report.postures.push(await effects());await capture('07-position-'+preset);}
      assert(new Set(report.postures.map(p=>JSON.stringify(p.effects.gravityRasMPerS2))).size===3);assert(report.postures.every(p=>p.shift>0));
      const positioned=await effects();await setRange('Viewing angle',25);await ready();assert.deepEqual(await effects(),positioned);await setRange('Viewing angle',16);
      await click('Undo');await ready();assert.notEqual((await effects()).state,positioned.state);await click('Redo');await ready();assert.deepEqual(await effects(),positioned);
      const old=await js(`document.querySelector('[aria-label="Surgical branch"]').value`);await click('Branch');await ready();assert.notEqual(await js(`document.querySelector('[aria-label="Surgical branch"]').value`),old);assert.equal((await effects()).state,positioned.state);
      await js(`(()=>{const s=document.querySelector('[aria-label="Patient position preset"]');s.value='prone';s.dispatchEvent(new Event('change',{bubbles:true}));})()`);await click('Apply position and settle 1 s');await ready();assert.equal((await effects()).effects.preset,'prone');await capture('08-position-UI-action');
      report.checks.push('Blocked exchange, view switching, three matched posture branches, zero camera physiology effect, Undo/Redo, Branch and new position action');
      await branch('operative-cusa');await seek(preparation.workStart);
    }
    const materialState=(await state()).id;
    for(const value of ['labels','operative']){await js(`(()=>{const s=document.querySelector('[aria-label="Operative surface material"]');s.value=${JSON.stringify(value)};s.dispatchEvent(new Event('change',{bubbles:true}));})()`);await ready();assert.equal((await state()).id,materialState);}
    report.checks.push('Illustrative/source material switching changes no native anatomy');
    report.savedView=await state();
    await click('Save');await wait(`document.body.innerText.includes('Workspace saved locally')`);report.saved=saved;
    report.frames=await js('window.operativeFrames');report.gpuCompletion=await js('window.operativeGpu');report.surfaceWork=await js('window.operativeWork');report.sourcePreserved=crypto.createHash('sha256').update(fs.readFileSync(input)).digest('hex')===sourceHash;assert(report.sourcePreserved);
    report.passed=!report.windowEverVisible;fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,captures:report.captures,deltaTarget:report.after.target-report.initial.target}));app.exit(report.passed?0:1);
  }catch(error){report.errors.push(String(error));report.body=await js('document.body.innerText');await capture('failure').catch(()=>{});fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.error(error);app.exit(1);}
 });
});require('./main.cjs');
