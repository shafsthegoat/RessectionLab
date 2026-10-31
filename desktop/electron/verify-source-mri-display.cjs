'use strict';
// Actual acquired-source pixels and GPU-qualified display admission; hidden only.
const {app,ipcMain,dialog}=require('electron');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..'),base=path.join(root,'testcases');
const input=fs.realpathSync(process.env.MRI_CASE||''),output=path.resolve(process.env.MRI_OUTPUT||'');
assert(input.startsWith(base+path.sep)&&output.startsWith(base+path.sep)&&!fs.existsSync(output));fs.mkdirSync(output,{recursive:true});
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex'),inputSha256=sha(input);
const report={schema:'resectionlab.source-mri-display-verification.v1',input,inputSha256,windowEverVisible:false,pixelMetric:{minChannel:60,maxChannel:245,channelSpread:16,minimumPixels:1000,scope:'Near-neutral source signal allowing anatomy-overlay tint; not an imaging-quality score'},states:[],requests:[],checks:[]};let last=null,activeIpc=0,libraryRefreshes=0;
const saved=path.join(output,'mri-display-saved.ressectionlab');
app.setPath('userData',path.join(output,'hidden-profile'));process.argv.push('--case',input,'--automation-hidden');
dialog.showSaveDialog=async()=>({canceled:false,filePath:saved});
const handle=ipcMain.handle.bind(ipcMain);ipcMain.handle=(name,handler)=>handle(name,async(...args)=>{
 activeIpc++;try{const value=await handler(...args);if(name==='research:listCases')libraryRefreshes++;if(name==='research:surgicalRehearsal'){last=value;report.requests.push({command:args[1]?.command,branchId:value?.branchId,stateId:value?.view?.stateId});}return value;}finally{activeIpc--;}});
const delay=ms=>new Promise(r=>setTimeout(r,ms));
app.on('browser-window-created',(_event,window)=>{
 window.on('show',()=>{report.windowEverVisible=true;window.hide();});
 window.webContents.once('did-finish-load',async()=>{
  window.webContents.setBackgroundThrottling(false);window.setContentSize(1440,900);
  const js=code=>window.webContents.executeJavaScript(code,true);
  const until=async(fn,label)=>{const start=Date.now();while(Date.now()-start<450000){if(await fn())return;await delay(60);}throw Error('Observation timeout '+label);};
  const click=async label=>{await js(`(()=>{const e=[...document.querySelectorAll('button')].find(e=>e.textContent.trim()===${JSON.stringify(label)}||e.getAttribute('aria-label')===${JSON.stringify(label)});if(!e||e.disabled)throw Error('Missing enabled '+${JSON.stringify(label)});e.click()})()`);await delay(100);};
  const set=async(selector,value)=>{await js(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});if(!e)throw Error('Missing input');Object.getOwnPropertyDescriptor(e.tagName==='SELECT'?HTMLSelectElement.prototype:HTMLInputElement.prototype,'value').set.call(e,${JSON.stringify(String(value))});e.dispatchEvent(new Event(e.tagName==='SELECT'?'change':'input',{bubbles:true}))})()`);await delay(100);};
  const ready=()=>until(()=>js(`(()=>{const p=window.__mriDisplayProbe,f=p?.frames.at(-1),g=f&&p.gpu.findLast(g=>g.serial===f.serial&&g.at>=f.at&&g.completed);return document.querySelector('.rl-viewer')?.dataset.rendererReady==='true'&&!document.querySelector('.viewer-loading,.rl-viewer-error,.surgical-panel .surgical-error')&&!!g&&(!document.querySelector('.surgical-panel')||document.querySelector('.surgical-panel')?.dataset.stateId===f.surgicalStateId)&&![...document.querySelectorAll('.surgical-panel [role="status"]')].some(n=>/^(Computing instrument action|Preparing an exact recorded state|Preparing accepted display|Verifying saved source-bound rehearsal)/.test(n.textContent.trim()))})()`),'GPU-complete accepted display');
  const capture=async name=>{
   await ready();const state=await js(`(()=>{const p=window.__mriDisplayProbe,f=p.frames.at(-1),g=p.gpu.findLast(g=>g.serial===f.serial&&g.at>=f.at&&g.completed);return {width:innerWidth,frame:f,gpu:g,panes:[...document.querySelectorAll('.rl-viewer-slice')].map(e=>{const r=e.getBoundingClientRect();return {name:['axial','coronal','sagittal'].find(n=>e.classList.contains('rl-viewer-'+n)),x:r.x,y:r.y,width:r.width,height:r.height}})}})()`);
   assert(!window.isVisible());const image=await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true}),scale=image.getSize().width/state.width;
   fs.writeFileSync(path.join(output,name+'.png'),image.toPNG());const pixels={};
   for(const pane of state.panes){const rect={x:Math.ceil((pane.x+12)*scale),y:Math.ceil((pane.y+35)*scale),width:Math.floor((pane.width-24)*scale),height:Math.floor((pane.height-55)*scale)};
    const data=image.crop(rect).toBitmap();let gray=0;for(let i=0;i<data.length;i+=4){const low=Math.min(data[i],data[i+1],data[i+2]),high=Math.max(data[i],data[i+1],data[i+2]);if(low>60&&high<245&&high-low<=16)gray++;}
    assert(gray>1000,`${name} ${pane.name}: insufficient near-neutral source signal under current overlays`);pixels[pane.name]={sha256:crypto.createHash('sha256').update(data).digest('hex'),grayPixels:gray,rect};
   }
   assert.equal(Object.keys(pixels).length,3);report.states.push({name,stateId:last?.view?.stateId,frame:state.frame,gpu:state.gpu,pixels});return pixels;
  };
  try{
   await js(`window.__mriDisplayProbe={frames:[],gpu:[],publications:[],admissions:[]};document.addEventListener('anatomy-frame',e=>{const p=window.__mriDisplayProbe;p.frames.push({serial:e.detail.serial,at:e.detail.at,surgicalStateId:e.detail.surgicalStateId,sourceCaseHash:e.detail.sourceCaseHash,sliceDraws:e.detail.sliceDraws});if(p.frames.length>32)p.frames.shift()});document.addEventListener('anatomy-gpu',e=>{const p=window.__mriDisplayProbe;p.gpu.push({...e.detail,at:performance.now()});if(p.gpu.length>48)p.gpu.shift()});document.addEventListener('anatomy-surgical-display',e=>{const p=window.__mriDisplayProbe;p.publications.push({...e.detail,at:performance.now(),gpu:p.gpu.findLast(g=>g.serial===e.detail.frameSerial&&g.surgicalStateId===e.detail.surgicalStateId&&g.sourceCaseHash===e.detail.sourceCaseHash&&g.completed)??null})});new MutationObserver(records=>{const p=window.__mriDisplayProbe;for(const r of records){if(r.attributeName!=='data-state-id'||!r.target.matches('.surgical-panel'))continue;const stateId=r.target.getAttribute('data-state-id');if(!stateId)continue;const publication=p.publications.findLast(v=>v.surgicalStateId===stateId),g=publication&&p.gpu.findLast(g=>g.serial===publication.frameSerial&&g.surgicalStateId===stateId&&g.completed&&g.at<=publication.at);p.admissions.push({stateId,at:performance.now(),serial:publication?.frameSerial,gpuCompleted:!!g})}}).observe(document,{attributes:true,subtree:true,attributeFilter:['data-state-id']})`);
   await ready();await click('Rehearse');await until(()=>!!last?.view,'native source response');await ready();
   if(await js(`[...document.querySelectorAll('button')].some(e=>e.textContent.trim()==='Verify saved replay')`)){await click('Verify saved replay');await ready();}
   const baseline={stateId:last.view.stateId,branchId:last.branchId,position:last.position,caseHash:last.caseHash};report.baseline=baseline;
   const originalPixels=await capture('01-source');
   const applications=last.preparationPlanning?.applications??last.strategyPlanning?.applications??[];
   const branch=applications.find(a=>a.branchId!==baseline.branchId)?.branchId;assert(branch,'saved installed branch required');
   await set('[aria-label="Surgical branch"]',branch);await ready();assert.notEqual(last.branchId,baseline.branchId);await capture('02-installed');
   await set('[aria-label="Surgical branch"]',baseline.branchId);await ready();assert.equal(last.view.stateId,baseline.stateId);assert.equal(last.position,baseline.position);
   const returned=await capture('03-source-return');assert.deepEqual(returned,originalPixels,'all three source pane pixels must return exactly');
   await js(`(()=>{const e=document.querySelector('[aria-label="Preparation comparison table"]');for(let p=e;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;e?.scrollIntoView({block:'start'})})()`);await delay(100);
   assert.deepEqual(await capture('04-scrolled'),originalPixels,'sidebar scrolling must preserve source pixels');
   const contrast=Number(await js(`document.querySelector('.rl-viewer-statusbar input[type=range]').value`));
   await set('.rl-viewer-statusbar input[type=range]',contrast===2?1.95:contrast+.05);const altered=await capture('05-contrast');
   for(const plane of Object.keys(originalPixels))assert.notEqual(altered[plane].sha256,originalPixels[plane].sha256,'contrast must affect the acquired signal');
   await set('.rl-viewer-statusbar input[type=range]',contrast);assert.deepEqual(await capture('06-contrast-restored'),originalPixels);
   if(process.env.MRI_LOSE_CONTEXT==='1'){
    await js(`(()=>{const gl=document.querySelector('.rl-viewer canvas').getContext('webgl2');const extension=gl.getExtension('WEBGL_lose_context');if(!extension)throw Error('Missing context-loss control');extension.loseContext()})()`);
    await until(()=>js(`[...document.querySelectorAll('button')].some(e=>e.textContent.trim()==='Retry graphics')`),'declared graphics loss');
    const count=report.requests.length;await js(`document.querySelector('.manual-actions button.primary-button')?.click()`);await delay(250);
    assert.equal(report.requests.length,count,'lost graphics must refuse instrument execution before IPC');
    await click('Retry graphics');await ready();assert.equal(last.view.stateId,baseline.stateId);
    assert.deepEqual(await capture('07-graphics-restored'),originalPixels);report.checks.push('Actual GPU context loss blocks instrument IPC; Retry graphics restores exact source pixels/state');
   }
   const admissions=await js('window.__mriDisplayProbe.admissions');report.admissions=admissions;report.publications=await js('window.__mriDisplayProbe.publications');assert(admissions.length>=3);assert(admissions.every(a=>a.gpuCompleted),'UI must accept each state after its matching GPU frame');
   const refreshBeforeSave=libraryRefreshes;await click('Save');await until(()=>fs.existsSync(saved)&&libraryRefreshes>refreshBeforeSave&&activeIpc===0,'native save and library refresh');await ready();assert.equal(sha(input),inputSha256);assert(!report.windowEverVisible);
   report.checks.push('Installed branch/source return, scrolling and reversible contrast preserve three acquired source panes and exact native source cursor');report.saved=saved;report.sourceUnchanged=true;report.passed=true;
  }catch(error){report.admissions=await js('window.__mriDisplayProbe?.admissions??[]').catch(()=>[]);report.publications=await js('window.__mriDisplayProbe?.publications??[]').catch(()=>[]);report.error=String(error);report.body=await js('document.body.innerText').catch(()=>null);console.error(error);}
  fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,error:report.error,saved:report.saved}));app.exit(report.passed?0:1);
 });
});
require('./main.cjs');
