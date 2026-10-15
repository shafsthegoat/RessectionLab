'use strict';
// Real UI verification. Saving preferences is permitted only in a disposable copy.
const {app}=require('electron'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),crypto=require('node:crypto');
const filename=process.argv[process.argv.indexOf('--anatomy')+1];
if(!filename?.includes('/ressection-vessel-display-'))throw Error('Use a disposable ressection-vessel-display- patient copy');
const output=path.resolve(process.env.VESSEL_VERIFY_OUTPUT||'../outputs/vessel-display-verification/render');
fs.mkdirSync(output,{recursive:true});app.setPath('userData',fs.mkdtempSync(path.join(os.tmpdir(),'rslab-vessel-pixels-')));
const initial=fs.readFileSync(filename),hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const report={errors:[],views:[],sourceManifestSha256:hash(initial)},pause=ms=>new Promise(r=>setTimeout(r,ms));
app.on('browser-window-created',(_event,window)=>window.webContents.once('did-finish-load',async()=>{
 const js=code=>window.webContents.executeJavaScript(code,true);
 window.webContents.on('console-message',(_event,d)=>{if(d.level==='error')report.errors.push(d.message)});
 const wait=async expr=>{for(let i=0;i<1200;i++){if(await js(expr))return;await pause(250)}throw Error('Timed out: '+expr)};
 const ready=async()=>{await pause(250);await wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')&&!document.querySelector('select[aria-label="Source scan"]')?.disabled`);await pause(750)};
 const click=text=>js(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(text)});if(!b||b.disabled)throw Error('Missing/disabled '+${JSON.stringify(text)});b.click()})()`);
 const select=async(label,value)=>{const start=performance.now();await js(`(()=>{const s=document.querySelector('select[aria-label="${label}"]');if(!s||s.disabled)throw Error('Unavailable ${label}');s.value=${JSON.stringify(value)};s.dispatchEvent(new Event('change',{bubbles:true}))})()`);await ready();return performance.now()-start};
 const scope=()=>js(`document.querySelector('select[aria-label="Vessel display"]').value`);
 const capture=async(name,loadMs)=>{
  const img=await window.webContents.capturePage();fs.writeFileSync(path.join(output,name+'.png'),img.toPNG());
  const box=await js(`(()=>{const r=document.querySelector('.rl-viewer-anatomy-image').getBoundingClientRect();return {x:Math.ceil(r.x),y:Math.ceil(r.y),width:Math.floor(r.width),height:Math.floor(r.height)}})()`);
  const view=await window.webContents.capturePage(box),pixels=view.toBitmap();let pink=0;
  for(let i=0;i<pixels.length;i+=4){const b=pixels[i],g=pixels[i+1],r=pixels[i+2];if(r>70&&r>g*1.3&&r>b*1.1&&b>g*1.15)pink++}
  const row={name,scope:await scope(),loadMs,pinkPixels:pink,slices:await js(`document.querySelectorAll('.rl-viewer-slice').length`)};report.views.push(row);console.log(JSON.stringify(row));
 };
 try{
  window.webContents.setBackgroundThrottling(false);window.setSize(1440,960);window.show();await ready();
  await click('Anatomy');await wait(`!!document.querySelector('select[aria-label="Vessel display"]')`);
  report.defaultScope=await scope();if(report.defaultScope!=='all')throw Error('Legacy filter did not migrate');
  await select('Source scan','ct-brain');if(await scope()!=='all')throw Error('Scan switch changed scope');
  await select('Skull view','transparent');
  for(const mode of ['Structures','Volume']){
   await click(mode);await ready();
   await capture(mode+'-all',null);
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
  if(saved.vesselDisplayVersion!==2||saved.vesselScope!=='intracranial')throw Error('Versioned preference did not save');
  await js(`document.querySelector('button[aria-label="Cases"]').click()`);await pause(500);
  const patient=JSON.parse(initial).patient_id;
  await js(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.querySelector('strong')?.textContent===${JSON.stringify(patient)});if(!b)throw Error('Saved collection missing');b.click()})()`);await ready();
  // Reopening may restore either sidebar tab.
  if(!await js(`!!document.querySelector('select[aria-label="Vessel display"]')`))await click('Anatomy');
  report.reopenedScope=await scope();if(report.reopenedScope!=='intracranial')throw Error('Explicit saved scope lost');
  await click('Show full source');await ready();await click('Save');await wait(`document.querySelector('footer').textContent.includes('Workspace saved')`);
  report.manifestUnchanged=hash(fs.readFileSync(filename))===hash(initial);
  if(!report.manifestUnchanged)throw Error('Patient manifest changed');
  for(const mode of ['Structures','Volume']){
   const a=report.views.find(v=>v.name===mode+'-all'),b=report.views.find(v=>v.name===mode+'-strict');
   if(a.pinkPixels<=b.pinkPixels||a.slices!==3)throw Error('Expected restored vessel coverage and three linked slices');
  }
  report.passed=report.errors.length===0;
 }catch(e){report.passed=false;report.failure=e.stack;fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage()).toPNG())}
 fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));app.exit(report.passed?0:1);
}));
require('./main.cjs');
