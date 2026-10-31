'use strict';
// Actual native app; automated verification must never show or focus a window.
const {app,dialog}=require('electron');
const fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..');
const input=path.resolve(process.env.PLANAR_CASE||''),output=path.resolve(process.env.PLANAR_OUTPUT||'');
for(const p of [input,output])if(!p.startsWith(path.join(root,'testcases')+path.sep))throw Error('Use ignored testcases paths');
if(!fs.existsSync(input)||fs.existsSync(output))throw Error('Input required; preserve existing output');
fs.mkdirSync(output,{recursive:true});
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex'),original=sha(input);
const expected=process.env.PLANAR_EXPECT?JSON.parse(fs.readFileSync(process.env.PLANAR_EXPECT,'utf8')):null;
const saved=path.join(output,'planar-blade-native-saved.ressectionlab');
app.setPath('userData',fs.mkdtempSync(path.join(os.tmpdir(),'rslab-planar-')));
process.argv.push('--case',input,'--automation-hidden');
dialog.showSaveDialog=async()=>({canceled:false,filePath:saved});
const report={schema:'resectionlab.planar-blade-native-verification.v1',input,output,sourceSha256:original,
 windowEverVisible:false,states:[],checks:[],errors:[],visualUpdates:[],
 timingScope:'Hidden native button dispatch or timeline seek through synchronized HUD, final CPU frame and completed GPU query; includes renderer, transfer, surface preparation, polling, and per-phase PNG capture/encoding/write overhead. It is separate from Python BridgeSession timings and is not physical screen presentation time.'};
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
app.on('browser-window-created',(_event,window)=>{
 window.on('show',()=>{report.windowEverVisible=true;window.hide();});
 window.webContents.once('did-finish-load',async()=>{
  window.webContents.setBackgroundThrottling(false);window.setContentSize(1440,900);
  const js=code=>window.webContents.executeJavaScript(code,true);
  const wait=async(expression,seconds=300)=>{const started=Date.now();while(Date.now()-started<seconds*1000){if(await js(expression))return;await delay(100);}throw Error('Timeout '+expression+'\n'+await js('document.body.innerText'));};
  const click=async label=>js(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(label)}||b.getAttribute('aria-label')===${JSON.stringify(label)});if(!b||b.disabled)throw Error('Missing enabled '+${JSON.stringify(label)});b.click();})()`);
  const ready=()=>wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')&&document.querySelector('.rl-viewer').dataset.surfacesPending==='0'&&!document.querySelector('[aria-label="Planar blade maneuver"][aria-busy="true"]')&&![...document.querySelectorAll('.surgical-panel [role="status"]')].some(n=>/^(Computing instrument action|Preparing an exact recorded state|Preparing accepted display|Verifying saved source-bound rehearsal)/.test(n.textContent.trim()))`);
  const state=()=>js(`(()=>{const p=document.querySelector('[data-planar-blade-state-id]'),v=document.querySelector('[data-operative-view="surgeon"][data-surgical-state-id]');if(!p||!v)return null;const values={};for(const key of ['offsetMm','normalReactionN','contactAreaMm2','maxDisplacementMm']){const n=p.querySelector('[data-planar-blade-metric="'+key+'"]');values[key]=n?Number(n.dataset.value??n.textContent):null;}return {viewportCss:{width:innerWidth,height:innerHeight},id:p.dataset.planarBladeStateId,acceptedStateId:p.dataset.planarBladeAcceptedStateId,phase:p.dataset.planarBladePhase,viewStateId:v.dataset.surgicalStateId,readoutStateId:document.querySelector('[data-operative-readout-state]')?.dataset.operativeReadoutState,position:Number(document.querySelector('[aria-label="Surgical timeline"]').value),camera:JSON.parse(v.dataset.opticalCamera),values,frame:window.planarFrames?.at(-1)}})()`);
  const synchronized=()=>wait(`(()=>{const p=document.querySelector('[data-planar-blade-state-id]'),v=document.querySelector('[data-operative-view="surgeon"][data-surgical-state-id]'),f=window.planarFrames?.at(-1);return !!p&&!!v&&!!f&&!f.interactive&&p.dataset.planarBladeStateId===v.dataset.surgicalStateId&&f.state===v.dataset.surgicalStateId&&window.planarGpu?.includes(f.serial)&&document.querySelector('[data-operative-readout-state]')?.dataset.operativeReadoutState===v.dataset.surgicalStateId})()`);
  const capture=async name=>{await ready();await synchronized();const before=await state();assert.equal(before.id,before.viewStateId);assert.equal(before.id,before.readoutStateId);fs.writeFileSync(path.join(output,name+'.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());const after=await state();for(const key of ['id','phase','position','camera','values'])assert.deepEqual(after[key],before[key]);report.states.push({name,...before});return before;};
  const seek=async position=>{const started=performance.now();await js(`(()=>{const s=document.querySelector('[aria-label="Surgical timeline"]');if(!s||s.disabled)throw Error('Timeline unavailable');Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(s,${position});s.dispatchEvent(new Event('input',{bubbles:true}));s.dispatchEvent(new Event('change',{bubbles:true}));})()`);await wait(`Number(document.querySelector('[aria-label="Surgical timeline"]').value)===${position}`);await ready();await synchronized();report.visualUpdates.push({kind:'seek',requestedPosition:position,latencyMs:performance.now()-started,...await state()});};
  try{
   await js(`window.planarFrames=[];window.planarGpu=[];document.addEventListener('anatomy-frame',e=>{window.planarFrames.push({serial:e.detail.serial,state:e.detail.surgicalStateId,interactive:e.detail.interactive,cpuMs:e.detail.cpuMs});if(window.planarFrames.length>512)window.planarFrames.shift();});document.addEventListener('anatomy-gpu',e=>{if(e.detail.completed)window.planarGpu.push(e.detail.serial);if(window.planarGpu.length>512)window.planarGpu.shift();});`);
   await wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`);await click('Rehearse');await ready();
   if(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify replay for planning')`)){await click('Verify replay for planning');await ready();}
   await click('Surgeon / Microscope');await ready();await wait(`!!document.querySelector('[data-planar-blade-state-id]')`);
   await js(`document.querySelector('[data-planar-blade-state-id]').scrollIntoView({block:'nearest'})`);
   if(process.env.PLANAR_REOPEN==='1'){
    assert(expected?.savedView,'Reopen expected report required');report.reopened=await capture('01-cold-reopened');
    for(const key of ['id','phase','position','camera','values'])assert.deepEqual(report.reopened[key],expected.savedView[key],key+' survives cold native reopen');
    report.checks.push('Cold native reopen preserves canonical maneuver state, phase, quantities, timeline cursor and microscope optics');
   }else{
    for(const [label,value] of [['Viewing angle',30],['Field / magnification',12],['Operative light',1.8]]){
     await js(`(()=>{const n=document.querySelector('[aria-label="'+${JSON.stringify(label)}+'"]');if(!n)throw Error('Missing optical control');Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(n,${value});n.dispatchEvent(new Event('input',{bubbles:true}));n.dispatchEvent(new Event('change',{bubbles:true}));})()`);
    }
    await ready();await synchronized();
    report.initial=await capture('01-before-approach');
    const optical=expected?.steps?.[0]?.view?.planarBladeManeuver?.opticalFrame;
    assert(optical,'Expected declared optical frame required');assert.deepEqual(report.initial.camera.target,optical.targetMm,'Optics center actual declared reference face');
    const direction=report.initial.camera.position.map((value,i)=>value-report.initial.camera.target[i]);
    assert(direction.reduce((sum,value,i)=>sum+value*optical.outward[i],0)/Math.hypot(...direction)>.8,'Optics view exterior side of declared contact face');
    const phases=['approach','indent','hold','withdraw'];
    for(const phase of phases){
     const before=await state();
     await js(`(()=>{const b=document.querySelector('[data-planar-blade-command="preview"][data-planar-blade-phase="${phase}"]');if(!b||b.disabled)throw Error('Missing enabled preview ${phase}');b.scrollIntoView({block:'nearest'});b.click();})()`);
     await ready();await wait(`document.querySelector('[data-planar-blade-state-id] h3')?.textContent.includes('preview')`);assert.equal((await state()).acceptedStateId,before.id,'Preview must preserve accepted state identity');await click('Cancel preview');await ready();await synchronized();assert.equal((await state()).id,before.id,'Cancel preview restores accepted state');
     const started=performance.now();
     await js(`(()=>{const b=document.querySelector('[data-planar-blade-command="step"][data-planar-blade-phase="${phase}"]');if(!b||b.disabled)throw Error('Missing enabled step ${phase}');b.scrollIntoView({block:'nearest'});b.click();})()`);
     await wait(`document.querySelector('[data-planar-blade-state-id]')?.dataset.planarBladeStateId!==${JSON.stringify(before.id)}`);await ready();await synchronized();
     const current=await capture(String(phases.indexOf(phase)+2).padStart(2,'0')+'-'+phase);
     report.visualUpdates.push({kind:'phase',phase,latencyMs:performance.now()-started,stateId:current.id});
     assert.equal(current.phase,{approach:'approached',indent:'indented',hold:'held',withdraw:'withdrawn'}[phase]);assert.deepEqual(current.camera,report.initial.camera,'Camera remains fixed across maneuver');
     if(expected?.steps){const wanted=expected.steps.find(r=>r.phase===phase);assert(wanted,'Expected phase '+phase);assert.equal(current.id,wanted.stateId);for(const key of Object.keys(current.values)){assert.equal(current.values[key],wanted.view.planarBladeManeuver[key],'Native '+phase+' raw '+key+' agrees with independently checked BridgeSession');}}
     if(phase==='indent'){
      const resizeSerial=(await state()).frame.serial;window.setContentSize(1180,760);await wait(`window.planarFrames?.at(-1)?.serial>${resizeSerial}`);await ready();await synchronized();const narrow=await capture('03b-indent-default-1180x760');assert.deepEqual(narrow.viewportCss,{width:1180,height:760});
      const narrowSerial=(await state()).frame.serial;window.setContentSize(1440,900);await wait(`window.planarFrames?.at(-1)?.serial>${narrowSerial}`);await ready();await synchronized();const wide=await capture('03c-indent-default-1440x900');assert.deepEqual(wide.viewportCss,{width:1440,height:900});
      await js(`(()=>{const d=document.querySelector('[data-planar-blade-state-id] details');if(d&&!d.open)d.querySelector('summary').click();})()`);
      await capture('03d-indent-model-details');
      await js(`document.querySelector('[data-planar-blade-state-id] details .surgical-warning')?.scrollIntoView({block:'center'})`);
      await capture('03e-prior-cycle-work-failure');
      await js(`(()=>{const d=document.querySelector('[data-planar-blade-state-id] details');if(d?.open)d.querySelector('summary').click();document.querySelector('[data-planar-blade-state-id]').scrollIntoView({block:'nearest'});})()`);
     }
    }
    assert(report.states.find(s=>s.phase==='held').values.normalReactionN<report.states.find(s=>s.phase==='indented').values.normalReactionN,'Hold relaxes modeled reaction');
    assert.equal(report.states.at(-1).values.normalReactionN,0,'Final withdrawal exerts no attraction');
    assert.equal(report.states.at(-1).values.contactAreaMm2,0,'Final withdrawal releases contact');
    report.checks.push('Actual previews preserve accepted state; native approach, indentation, hold and withdrawal buttons publish synchronized canonical state/quantities with a fixed microscope');
    const final=await state();await seek(0);assert.equal((await state()).id,report.initial.id);await seek(final.position);assert.equal((await state()).id,final.id);report.checks.push('Timeline replay returns exact initial and final maneuver identities');
    report.savedView=await state();await click('Save');await wait(`document.body.innerText.includes('Workspace saved locally')`);assert(fs.existsSync(saved));report.saved=saved;
   }
   report.sourcePreserved=sha(input)===original;assert(report.sourcePreserved);assert.equal(report.windowEverVisible,false);
   report.passed=true;fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:true,saved:report.saved,states:report.states.length,checks:report.checks}));app.exit(0);
  }catch(error){report.errors.push(error.stack||String(error));report.body=await js('document.body.innerText').catch(()=>null);try{fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());}catch{}fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.error(error);app.exit(1);}
 });
});
require('./main.cjs');
