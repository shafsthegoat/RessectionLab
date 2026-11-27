'use strict';
// Real UI verification. Saving preferences is permitted only in a disposable copy.
const {countVesselPixels}=require('./vessel-pixels.cjs');
const {app,ipcMain}=require('electron'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),crypto=require('node:crypto');
// Reopening a study replaces the engine's transfers; a view requested before it finishes is discarded.
const opens={started:0,finished:0},handleIpc=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(channel,handler)=>handleIpc(channel,channel!=='research:openListedWorkspace'?handler:async(...args)=>{opens.started++;try{return await handler(...args)}finally{opens.finished++}});
process.argv.push('--automation-hidden');
const filename=process.argv[process.argv.indexOf('--anatomy')+1];
if(!filename?.includes('/ressection-vessel-display-'))throw Error('Use a disposable ressection-vessel-display- patient copy');
const output=path.resolve(process.env.VESSEL_VERIFY_OUTPUT||'../outputs/vessel-display-verification/render');
fs.mkdirSync(output,{recursive:true});app.setPath('userData',fs.mkdtempSync(path.join(os.tmpdir(),'rslab-vessel-pixels-')));
const initial=fs.readFileSync(filename),hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const report={errors:[],views:[],sourceManifestSha256:hash(initial)},pause=ms=>new Promise(r=>setTimeout(r,ms));
app.on('browser-window-created',(_event,window)=>window.webContents.once('did-finish-load',async()=>{
 const hiddenCapture=box=>{if(window.isVisible())throw Error('Vessel verification window became visible');return window.webContents.capturePage(box,{stayHidden:true,stayAwake:true})};
 const js=code=>window.webContents.executeJavaScript(code,true);
 window.webContents.on('console-message',(_event,d)=>{if(d.level==='error')report.errors.push(d.message)});
 const wait=async expr=>{for(let i=0;i<1200;i++){if(await js(expr))return;await pause(250)}throw Error('Timed out: '+expr)};
 const ready=async()=>{await pause(250);await wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')&&!document.querySelector('select[aria-label="Source scan"]')?.disabled`);await pause(750)};
 const click=text=>js(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(text)});if(!b||b.disabled)throw Error('Missing/disabled '+${JSON.stringify(text)});b.click()})()`);
 const select=async(label,value)=>{const start=performance.now();await js(`(()=>{const s=document.querySelector('select[aria-label="${label}"]');if(!s||s.disabled)throw Error('Unavailable ${label}');s.value=${JSON.stringify(value)};s.dispatchEvent(new Event('change',{bubbles:true}))})()`);await ready();return performance.now()-start};
 const scope=()=>js(`document.querySelector('select[aria-label="Vessel display"]').value`);
 const review=async on=>{const start=performance.now();await js(`(()=>{const c=document.querySelector('input[aria-label="Show hidden vessel parts"]');if(!c||c.disabled)throw Error('Unavailable review toggle');if(c.checked!==${on})c.click()})()`);await ready();return performance.now()-start};
 // Detached components, wall-thickening pieces and thin end-branch voxels hidden, summed over every vessel layer's receipt.
 const hiddenParts=()=>js(`(()=>{const texts=[...document.querySelectorAll('[data-vessel-receipt]')].map(s=>s.textContent);const n=(t,re)=>Number((t.match(re)?.[1]??'0').replaceAll(',',''));return texts.reduce((sum,t)=>sum+n(t,/([\\d,]+) detached components/)+n(t,/([\\d,]+) wall-thickening pieces/)+n(t,/([\\d,]+) thin end-branch/),0)})()`);
 const capture=async(name,loadMs)=>{
  const img=await hiddenCapture();fs.writeFileSync(path.join(output,name+'.png'),img.toPNG());
  const box=await js(`(()=>{const r=document.querySelector('.rl-viewer-anatomy-image').getBoundingClientRect();return {x:Math.ceil(r.x),y:Math.ceil(r.y),width:Math.floor(r.width),height:Math.floor(r.height)}})()`);
  const view=await hiddenCapture(box),{vessel,amber}=countVesselPixels(view.toBitmap());
  const row={name,scope:await scope(),loadMs,vesselPixels:vessel,amberPixels:amber,slices:await js(`document.querySelectorAll('.rl-viewer-slice').length`)};report.views.push(row);console.log(JSON.stringify(row));
 };
 try{
  window.webContents.setBackgroundThrottling(false);window.setSize(1440,960);await ready();
  await click('Anatomy');await wait(`!!document.querySelector('select[aria-label="Vessel display"]')`);
  report.defaultScope=await scope();if(report.defaultScope!=='rooted')throw Error('Legacy filter did not migrate to the connected tree');
  // The October fixtures switch to their CT scan; single-scan studies skip this step.
  const switchScan=process.argv.includes('--switch-scan')?process.argv[process.argv.indexOf('--switch-scan')+1]:null;
  if(switchScan){await select('Source scan',switchScan);if(await scope()!=='rooted')throw Error('Scan switch changed scope');}
  await select('Skull view','transparent');
  for(const mode of ['Structures','Volume']){
   await click(mode);await ready();
   if(await scope()!=='rooted')await select('Vessel display','rooted');
   await capture(mode+'-rooted',null);
   report[mode+'HiddenParts']=await hiddenParts();
   await capture(mode+'-rooted-review',await review(true));
   if(await scope()!=='rooted')throw Error('The review toggle changed the vessel display');
   await review(false);
   await capture(mode+'-all',await select('Vessel display','all'));
   const elapsed=await select('Vessel display','intracranial');await capture(mode+'-strict',elapsed);
   const start=performance.now();await click('Show full source');await ready();await capture(mode+'-restored',performance.now()-start);
   if(await scope()!=='all')throw Error('Show full source failed');
  }
  await js(`window.vesselFrames=[];window.vesselRefined=[];document.addEventListener('anatomy-frame',e=>window.vesselFrames.push(e.detail));document.addEventListener('anatomy-refined',e=>window.vesselRefined.push(e.detail));`);
  const r=await js(`(()=>{const r=document.querySelector('.rl-viewer-anatomy-image').getBoundingClientRect();return {x:Math.round(r.x+r.width*.5),y:Math.round(r.y+r.height*.5)}})()`);
  const presented=[];window.webContents.beginFrameSubscription(true,()=>presented.push(performance.now()));
  window.webContents.sendInputEvent({type:'mouseDown',...r,button:'left',clickCount:1});
  for(let i=1;i<=60;i++){window.webContents.sendInputEvent({type:'mouseMove',x:r.x+i*2,y:r.y,modifiers:['leftButtonDown']});await pause(16)}
  window.webContents.sendInputEvent({type:'mouseUp',x:r.x+120,y:r.y,button:'left',clickCount:1});window.webContents.endFrameSubscription();await pause(1000);
  const gaps=presented.slice(1).map((t,i)=>t-presented[i]).sort((a,b)=>a-b);
  report.rotation={presentedFrames:presented.length,medianFps:gaps.length?1000/gaps[Math.floor(gaps.length/2)]:null,refinements:await js('window.vesselRefined')};
  await select('Vessel display','intracranial');await click('Save');await wait(`document.querySelector('footer').textContent.includes('Workspace saved')`);
  const saved=JSON.parse(fs.readFileSync(path.join(path.dirname(filename),'workspace-view.json'),'utf8'));
  if(saved.vesselDisplayVersion!==3||saved.vesselScope!=='intracranial')throw Error('Versioned preference did not save');
  await js(`document.querySelector('button[aria-label="Cases"]').click()`);await pause(500);
  const patient=JSON.parse(initial).patient_id,opened=opens.finished;
  await js(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.querySelector('strong')?.textContent===${JSON.stringify(patient)});if(!b)throw Error('Saved collection missing');b.click()})()`);
  for(let i=0;opens.finished===opened||opens.started!==opens.finished;i++){if(i>=240)throw Error('The study did not reopen');await pause(250)}
  await ready();
  // Reopening may restore either sidebar tab.
  if(!await js(`!!document.querySelector('select[aria-label="Vessel display"]')`))await click('Anatomy');
  report.reopenedScope=await scope();if(report.reopenedScope!=='intracranial')throw Error('Explicit saved scope lost');
  await click('Show full source');await ready();await click('Save');await wait(`document.querySelector('footer').textContent.includes('Workspace saved')`);
  report.manifestUnchanged=hash(fs.readFileSync(filename))===hash(initial);
  if(!report.manifestUnchanged)throw Error('Patient manifest changed');
  for(const mode of ['Structures','Volume']){
   const a=report.views.find(v=>v.name===mode+'-all'),b=report.views.find(v=>v.name===mode+'-strict');
   if(a.vesselPixels<=b.vesselPixels||a.slices!==3)throw Error('Expected restored vessel coverage and three linked slices');
   const rooted=report.views.find(v=>v.name===mode+'-rooted'),shown=report.views.find(v=>v.name===mode+'-rooted-review');
   // The connected tree hides parts of the full source; it never adds vessel area beyond bridges.
   if(rooted.vesselPixels>a.vesselPixels*1.02+50)throw Error('Connected tree shows more vessel than the full source');
   if(report[mode+'HiddenParts']>0&&shown.amberPixels<=rooted.amberPixels)throw Error('Hidden vessel parts did not appear in amber');
  }
  report.passed=report.errors.length===0;
 }catch(e){report.passed=false;report.failure=e.stack;if(!window.isVisible())fs.writeFileSync(path.join(output,'failure.png'),(await hiddenCapture()).toPNG())}
 fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));app.exit(report.passed?0:1);
}));
require('./main.cjs');
