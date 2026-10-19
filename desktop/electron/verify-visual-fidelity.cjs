'use strict';
/** Actual Electron rendering checks. Writable operations target an APFS clone;
 * original manifest and every source array are checked with SHA-256. */
const {app,dialog}=require('electron');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path'),crypto=require('node:crypto');
const output=path.resolve(process.env.FIDELITY_OUTPUT||path.join(__dirname,'../../artifacts/desktop-renderer/medivis-fidelity-v1/final'));
fs.mkdirSync(output,{recursive:true});
const userData=fs.mkdtempSync(path.join(os.tmpdir(),'rslab-fidelity-user-'));
app.setPath('userData',userData);
const index=process.argv.indexOf('--anatomy');
if(index<0||!process.argv[index+1])throw Error('Pass --anatomy /absolute/patient-anatomy.json');
const originalSource=path.resolve(process.argv[index+1]),originalDirectory=path.dirname(originalSource);
const temporary=fs.mkdtempSync(path.join(os.tmpdir(),'rslab-fidelity-study-')),study=path.join(temporary,'study');
fs.mkdirSync(study);
for(const member of ['patient-anatomy.json','arrays','originals','meshes','revisions','workspace-view.json']){
  const from=path.join(originalDirectory,member);
  if(fs.existsSync(from))fs.cpSync(from,path.join(study,member),{recursive:true,mode:fs.constants.COPYFILE_FICLONE});
}
process.argv[index+1]=path.join(study,'patient-anatomy.json');
// Resolve only this verification process's native Open picker to its clone.
dialog.showOpenDialog=async()=>({canceled:false,filePaths:[process.argv[index+1]]});
function digest(filename){
  const hash=crypto.createHash('sha256'),file=fs.openSync(filename,'r'),buffer=Buffer.allocUnsafe(8*1024*1024);
  try{let size;while((size=fs.readSync(file,buffer,0,buffer.length,null)))hash.update(buffer.subarray(0,size));}finally{fs.closeSync(file)}
  return hash.digest('hex');
}
function inventory(){
  const files=['patient-anatomy.json'];
  function visit(relative){for(const name of fs.readdirSync(path.join(originalDirectory,relative)).sort()){
    const member=path.join(relative,name),stat=fs.statSync(path.join(originalDirectory,member));
    if(stat.isDirectory())visit(member);else if(stat.isFile())files.push(member);
  }}
  visit('arrays');return Object.fromEntries(files.map(name=>[name,{bytes:fs.statSync(path.join(originalDirectory,name)).size,sha256:digest(path.join(originalDirectory,name))}]));
}
const report={startedAt:Date.now(),originalSource,disposableStudy:true,errors:[],checks:[],views:[],surface:[],frames:[],gpu:[],refinement:[],motion:[]};
console.log('Hashing original manifest and every source array');const before=inventory();
fs.writeFileSync(path.join(output,'source-hashes-before.json'),JSON.stringify(before,null,2));
const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
function memoryReceipt(){
  const metrics=app.getAppMetrics();
  const rows=require('node:child_process').execFileSync('ps',['-axo','pid=,ppid=,rss=,comm='],{encoding:'utf8'}).trim().split('\n').map(line=>{const m=line.trim().match(/^(\d+)\s+(\d+)\s+(\d+)\s+(.+)$/);return m?{pid:+m[1],ppid:+m[2],rssKB:+m[3],command:m[4]}:null}).filter(Boolean);
  const ids=new Set([process.pid]);for(let i=0;i<8;i++)for(const row of rows)if(ids.has(row.ppid))ids.add(row.pid);
  const descendants=rows.filter(row=>ids.has(row.pid)),python=descendants.filter(row=>/python/i.test(row.command));
  return {at:Date.now(),units:'KiB as reported by Electron/ps',electron:metrics,totalElectronWorkingSetKB:metrics.reduce((sum,row)=>sum+(row.memory?.workingSetSize??0),0),peakAggregation:'sum of individual process peaks, not a simultaneous whole-app peak',totalElectronPeakWorkingSetKB:metrics.reduce((sum,row)=>sum+(row.memory?.peakWorkingSetSize??0),0),pythonSidecars:python,totalPythonRssKB:python.reduce((sum,row)=>sum+row.rssKB,0)};
}

const cleanup=()=>{fs.rmSync(temporary,{recursive:true,force:true});fs.rmSync(userData,{recursive:true,force:true})};
app.on('will-quit',cleanup);
app.on('browser-window-created',(_event,window)=>{
  window.webContents.on('console-message',(_event,d)=>{if(d.level==='error')report.errors.push(d.message)});
  window.webContents.once('did-finish-load',async()=>{
    window.webContents.setBackgroundThrottling(false);window.setSize(1440,960);window.show();
    const js=code=>window.webContents.executeJavaScript(code,true);
    const wait=async(expr,seconds=300)=>{for(let i=0;i<seconds*4;i++){if(await js(expr))return;await pause(250)}throw Error('Timed out '+expr)};
    const click=async(label,selector='button')=>{await js(`(()=>{const b=[...document.querySelectorAll(${JSON.stringify(selector)})].find(b=>b.textContent.trim()===${JSON.stringify(label)});if(!b||b.disabled)throw Error('Missing button '+${JSON.stringify(label)});b.click()})()`);await pause(200)};
    const select=async(label,value)=>{await js(`(()=>{const s=[...document.querySelectorAll('select')].find(s=>s.getAttribute('aria-label')===${JSON.stringify(label)});if(!s||s.disabled)throw Error('Missing select '+${JSON.stringify(label)});s.value=${JSON.stringify(value)};s.dispatchEvent(new Event('change',{bubbles:true}))})()`);await pause(300)};
    const ready=async()=>{await wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')&&document.querySelector('.rl-viewer').dataset.surfacesPending==='0'&&document.querySelector('select[aria-label="Source scan"]')?.disabled===false`);await pause(600)};
    const capture=async name=>{await ready();const filename=path.join(output,name+'.png');fs.writeFileSync(filename,(await window.webContents.capturePage()).toPNG());report.views.push({name,path:filename,at:Date.now()});console.log('Captured '+filename)};
    try{
      await js(`window.fidelityEvents={surface:[],frames:[],gpu:[],refinement:[]};for(const [event,key]of [['anatomy-surface','surface'],['anatomy-frame','frames'],['anatomy-gpu','gpu'],['anatomy-refined','refinement']])document.addEventListener(event,e=>window.fidelityEvents[key].push({...e.detail,observedAt:performance.now()}));`);
      await ready();
      report.scans=await js(`[...document.querySelector('select[aria-label="Source scan"]').options].map(o=>({value:o.value,label:o.textContent}))`);
      const t1=process.env.FIDELITY_MRA?report.scans.find(s=>/MRA|TOF/i.test(s.label+' '+s.value)):report.scans.find(s=>/T1CE|T1C|T1[-_ ]?POST/i.test(s.label+' '+s.value));if(!t1)throw Error('No requested acquired source option');
      await select('Source scan',t1.value);await ready();await click('Fit');await capture(process.env.FIDELITY_MRA?'mra-before-presentation':'mri-before-presentation');
      await click('Layout');await select('Camera preset','left');await js(`document.querySelector('button[aria-label="Close view controls"]').click()`);await capture(process.env.FIDELITY_MRA?'mra-before-left':'mri-before-left');
      await click(process.env.FIDELITY_MRA?'Vascular':'Segmentation');await ready();await click('Fit');await capture(process.env.FIDELITY_MRA?'mra-vascular':'segmentation');
      await click('Layout');await select('Camera preset','left');await js(`document.querySelector('button[aria-label="Close view controls"]').click()`);await capture(process.env.FIDELITY_MRA?'mra-vascular-left':'segmentation-left');
      if(process.env.FIDELITY_SUPERIOR){
        await click('Layout');await select('Camera preset','superior');await js(`document.querySelector('button[aria-label="Close view controls"]').click()`);await capture('mra-vascular-superior');
        window.setSize(1050,720);await pause(700);await capture('mra-minimum-1050x720');report.minimumWindow={requested:[1050,720],actual:window.getSize(),content:window.getContentSize()};
      }
      report.initialPresentation=await js(`({source:document.querySelector('select[aria-label="Source scan"]').value,scope:document.querySelector('select[aria-label="Vessel display"]')?.value,layers:[...document.querySelectorAll('.compact-layer input[type="checkbox"]')].map(c=>({label:c.getAttribute('aria-label'),checked:c.checked})),provenance:document.querySelector('.presentation-source')?.textContent,body:document.body.innerText})`);
      fs.writeFileSync(path.join(output,'initial-presentation.json'),JSON.stringify(report.initialPresentation,null,2));
      report.checks.push({name:process.env.FIDELITY_MRA?'initial independent MRA vascular captures':'initial T1CE and segmentation captures',passed:true});
      if(!process.env.FIDELITY_INITIAL_ONLY)await runFullVerification({window,js,wait,click,select,ready,capture,pause,report,output,study});
      Object.assign(report,await js('window.fidelityEvents'));
      report.memory=app.getAppMetrics();report.memoryReceipt=memoryReceipt();report.passed=report.errors.length===0&&report.checks.every(c=>c.passed);
    }catch(error){report.passed=false;report.failure=error.stack;console.error(error.stack);fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage()).toPNG());}
    console.log('Verifying original manifest and every source array');
    const after=inventory();fs.writeFileSync(path.join(output,'source-hashes-after.json'),JSON.stringify(after,null,2));
    report.sourceArrayCount=Object.keys(before).length-1;report.sourceUnchanged=JSON.stringify(before)===JSON.stringify(after);
    if(!report.sourceUnchanged){report.passed=false;report.failure='Original manifest or source arrays changed'}
    report.finishedAt=Date.now();fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));
    console.log(JSON.stringify({passed:report.passed,failure:report.failure,errors:report.errors,sourceUnchanged:report.sourceUnchanged,views:report.views.length}));cleanup();app.exit(report.passed?0:1);
  });
});
async function runFullVerification({window,js,wait,click,select,ready,capture,pause,report,output,study}){
  const assert=(condition,message)=>{if(!condition)throw Error(message)};
  const step=async(name,action)=>{if(process.env.FIDELITY_RECHECK&&!/presented orbit FPS|native save/.test(name))return;console.log('Checking '+name);try{const evidence=await action();report.checks.push({name,passed:true,evidence})}catch(error){report.checks.push({name,passed:false,error:error.stack});console.error(name+': '+error.message);fs.writeFileSync(path.join(output,'failure-'+name.replace(/[^a-z0-9]+/gi,'-')+'.png'),(await window.webContents.capturePage()).toPNG())}};
  const state=()=>js(`({source:document.querySelector('select[aria-label="Source scan"]').value,presentation:[...document.querySelectorAll('.anatomy-presets button')].filter(b=>b.getAttribute('aria-pressed')==='true').map(b=>b.textContent),scope:document.querySelector('select[aria-label="Vessel display"]')?.value,layers:[...document.querySelectorAll('.compact-layer input[type="checkbox"]')].map(c=>({label:c.getAttribute('aria-label'),checked:c.checked})),cursor:document.querySelector('.rl-viewer-statusbar>span')?.textContent,slices:[...document.querySelectorAll('.rl-viewer-slice-image')].map(e=>e.getAttribute('aria-label')),provenance:document.querySelector('.presentation-source')?.textContent,warning:document.querySelector('.alignment-banner')?.textContent??document.body.innerText.split('\\n').filter(s=>/alignment|align|source|estimated/i.test(s)).join(' | ')})`);
  const close=async()=>{await js(`document.querySelector('button[aria-label="Close view controls"]')?.click()`);await pause(150)};
  const setRange=async(label,value)=>{await js(`(()=>{const e=[...document.querySelectorAll('input')].find(e=>e.getAttribute('aria-label')===${JSON.stringify(label)});if(!e)throw Error('Missing range '+${JSON.stringify(label)});Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(e,${JSON.stringify(String(value))});e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}))})()`);await pause(200)};
  const center=selector=>js(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});if(!e)throw Error('Missing pane');const r=e.getBoundingClientRect();return {x:Math.round(r.x+r.width*.5),y:Math.round(r.y+r.height*.5),w:r.width,h:r.height}})()`);
  const mouseClick=async point=>{window.webContents.sendInputEvent({type:'mouseDown',x:point.x,y:point.y,button:'left',clickCount:1});await pause(30);window.webContents.sendInputEvent({type:'mouseUp',x:point.x,y:point.y,button:'left',clickCount:1});await pause(350)};
  const toggle=async(label)=>{await js(`(()=>{const e=[...document.querySelectorAll('input[type="checkbox"]')].find(e=>e.getAttribute('aria-label')===${JSON.stringify(label)});if(!e||e.disabled)throw Error('Missing enabled toggle '+${JSON.stringify(label)});e.click()})()`);await pause(350)};
  const pane='.rl-viewer-anatomy-image';
  const motionRoot=path.resolve(process.env.FIDELITY_MOTION_OUTPUT||path.join(__dirname,'../../outputs/medivis-fidelity-v1',process.env.FIDELITY_MRA?'mra':'sub06'));
  const record=async(kind)=>{const directory=path.join(motionRoot,kind);fs.mkdirSync(directory,{recursive:true});const frames=[];return async()=>{const index=frames.length+1,filename=path.join(directory,String(index).padStart(4,'0')+'.png'),at=performance.now();fs.writeFileSync(filename,(await window.webContents.capturePage()).toPNG());const item={index,at,capturedAt:performance.now(),path:filename,kind};frames.push(item);report.motion.push(item);return item}};
  const orbit=async({recordFrames=false,count=90,button='left',distance=120}={})=>{
    const point=await center(pane),captureFrame=recordFrames?await record('orbit'):null;
    const dragStarted=performance.now();window.webContents.sendInputEvent({type:'mouseDown',x:point.x,y:point.y,button,clickCount:1});
    for(let i=1;i<=count;i++){
      window.webContents.sendInputEvent({type:'mouseMove',x:point.x+Math.round(distance*i/count),y:point.y+Math.round(15*Math.sin(i/count*Math.PI)),modifiers:[button==='left'?'leftButtonDown':'rightButtonDown']});await pause(16);
      if(captureFrame&&i%3===0)await captureFrame();
    }
    window.webContents.sendInputEvent({type:'mouseUp',x:point.x+distance,y:point.y,button,clickCount:1});const dragEnded=performance.now();await pause(1000);return {dragStarted,dragEnded,dragMilliseconds:dragEnded-dragStarted,idleMilliseconds:performance.now()-dragEnded};
  };
  const initial=await state();
  await step('source scope and provenance',async()=>{assert(initial.scope==='all','Source vessel fragments were silently filtered');assert(initial.provenance?.length>0,'Missing source provenance');assert(initial.layers.filter(l=>['Show vessels','Show brain','Show tumor','Show bone'].includes(l.label)).every(l=>l.checked),'A primary anatomy layer was silently disabled');return initial});
  await step('vascular presentation preserves primary selections',async()=>{await click('Vascular');await ready();await click('Fit');await capture(process.env.FIDELITY_MRA?'mra-vascular-fit':'vascular-t1ce');const next=await state();assert(JSON.stringify(next.layers)===JSON.stringify(initial.layers),'Presentation changed selected structures');assert(next.scope==='all','Presentation filtered source fragments');assert(process.env.FIDELITY_MRA?/MR angiography/.test(next.provenance):/no acquired angiography/.test(next.provenance),'Modality provenance is misleading');return next});
  await step('structures and volume remain reversible',async()=>{await click('Structures');await capture('structures-only');assert(await js(`document.querySelector('button[aria-label="Show structures only"]').getAttribute('aria-pressed')==='true'`),'Structures button did not select mesh-only mode');await click('Volume');await capture('volume-and-structures');assert(await js(`document.querySelector('button[aria-label="Show scan volume with structures"]').getAttribute('aria-pressed')==='true'`),'Volume button did not restore scalar rendering');return await state()});
  await step('modality-safe preset controls and contour style',async()=>{await click('Appearance');const options=await js(`[...document.querySelector('select[aria-label="Volume rendering preset"]').options].map(o=>({value:o.value,disabled:o.disabled,label:o.textContent}))`);assert(options.filter(o=>o.value.startsWith('ct-')).every(o=>o.disabled),'CT presets enabled on MRI');const current=await js(`document.querySelector('select[aria-label="Slice annotation style"]').value`);assert(current==='contour','Presentation did not select contours');await select('Slice annotation style','fill');await close();await capture('slice-filled');await click('Appearance');await select('Slice annotation style','contour');await close();await capture('slice-contours');return options});
  await step('linked slice cursor click',async()=>{const before=await state(),point=await center('.rl-viewer-slice-image');await mouseClick({...point,x:point.x+Math.round(point.w*.12)});await capture('linked-slice-click');const after=await state();assert(after.cursor!==before.cursor,'Linked RAS cursor did not move');assert(after.slices.some((s,i)=>s!==before.slices[i]),'Other linked slice positions did not update');return {before:before.cursor,after:after.cursor,slices:after.slices}});
  const tumor=initial.layers.find(l=>/tumor/i.test(l.label));
  if(tumor)await step('tumor toggle and cross-scan shared structures',async()=>{await toggle(tumor.label);await capture('tumor-hidden');let next=await state();assert(next.layers.find(l=>l.label===tumor.label)?.checked===false,'Tumor did not hide');const ct=report.scans.find(s=>/ct-brain/i.test(s.value));if(ct){await select('Source scan',ct.value);await ready();assert((await state()).layers.find(l=>l.label===tumor.label)?.checked===false,'Scan switch restored a hidden tumor');await click('Fit');await capture('ct-shared-structures');await select('Source scan',initial.source);await ready();assert((await state()).layers.find(l=>l.label===tumor.label)?.checked===false,'Return scan restored hidden tumor')}await toggle(tumor.label);await click('Fit');await capture('tumor-restored');return await state()});
    await step('layer opacity slider',async()=>{const selected=await js(`(()=>{const b=[...document.querySelectorAll('.layer-name')].find(b=>${JSON.stringify(process.env.FIDELITY_MRA?'vessel':'brain')}==='brain'?b.textContent.trim()==='brain':/vessel/i.test(b.textContent));if(!b)throw Error('Missing opacity layer');b.click();return b.textContent.trim()})()`);await pause(200);const label=selected+' opacity',before=await js(`[...document.querySelectorAll('input')].find(e=>e.getAttribute('aria-label')===${JSON.stringify(label)}).value`);await setRange(label,.04);await capture('layer-opacity-low');const low=await js(`[...document.querySelectorAll('input')].find(e=>e.getAttribute('aria-label')===${JSON.stringify(label)}).value`);assert(Math.abs(+low-.04)<.001,'Layer opacity did not update');await setRange(label,+before);await js(`document.querySelector('button[aria-label="Close layer details"]').click()`);return {layer:selected,before,low}});
  await step('source fragment inspection highlights and locates without accepting a correction',async()=>{
    const before=await state();
    await js(`(()=>{const b=[...document.querySelectorAll('.layer-name')].find(b=>/vessel/i.test(b.textContent));if(!b)throw Error('Missing vessel layer');b.click()})()`);await pause(200);
    await click('Inspect vessel fragments');await ready();await wait(`!!document.querySelector('.component-list input[type="checkbox"]')`,60);await capture('fragment-review-source');
    const selected=await js(`(()=>{const e=document.querySelector('.component-list input[type="checkbox"]');e.click();return e.getAttribute('aria-label')})()`);await ready();
    await js(`document.querySelector('.component-list input[type="checkbox"]').parentElement.querySelector('button').click()`);await pause(300);await capture('fragment-selected-linked');
    const highlighted=await state();assert(highlighted.scope==='all','Fragment inspection restricted source scope');assert(highlighted.cursor!==before.cursor,'Fragment locate did not move linked cursor');
    await toggle(selected);await ready();await js(`document.querySelector('button[aria-label="Close layer details"]').click()`);await capture('fragment-review-restored-source');
    const after=await state();assert(after.scope==='all','Full source was not restored');assert(!await js(`!!document.querySelector('.unified-correction')`),'Fragment inspection unexpectedly entered a correction draft');return {selected,before:before.cursor,located:highlighted.cursor,sourceScope:after.scope,acceptedCorrection:false};
  });
  await step('presented orbit FPS without capture for intact default presentations',async()=>{
    const benchmarks=[];
    for(const presentation of (process.env.FIDELITY_MRA?['Vascular']:['Segmentation','Vascular'])){
      await close();await click(presentation);await ready();await click('Fit');await pause(1000);
      const baseline=await state(),eventStart=await js('window.fidelityEvents.refinement.length');
      assert(baseline.scope==='all','Performance presentation restricted source fragments');
      const timestamps=[];window.webContents.beginFrameSubscription(true,()=>timestamps.push(performance.now()));
      const started=performance.now(),drag=await orbit({count:120});window.webContents.endFrameSubscription();
      const elapsed=performance.now()-started,dragFrames=timestamps.filter(t=>t>=drag.dragStarted&&t<=drag.dragEnded);
      const intervals=dragFrames.slice(1).map((t,i)=>t-dragFrames[i]).sort((a,b)=>a-b);
      await pause(500);
      const refinement=await js(`window.fidelityEvents.refinement.slice(${eventStart}).filter(e=>e.completed)`),refinementMs=refinement.map(e=>e.milliseconds).sort((a,b)=>a-b);
      const result={presentation,source:baseline.source,sourceScope:baseline.scope,clipping:'disabled by presentation reset',...drag,deliveredFrames:dragFrames.length,totalSubscriptionFrames:timestamps.length,totalElapsedMs:elapsed,medianFps:intervals.length?1000/intervals[Math.floor(intervals.length*.5)]:null,p95FrameMs:intervals.length?intervals[Math.floor(intervals.length*.95)]:null,captureOverhead:false,cpuEnvironment:'run after desktop/Python suites completed',refinement:{durationField:'milliseconds',completedCount:refinement.length,p50Ms:refinementMs[Math.floor(refinementMs.length*.5)]??null,p95Ms:refinementMs[Math.min(refinementMs.length-1,Math.floor(refinementMs.length*.95))]??null,maxMs:Math.max(0,...refinementMs)}};
      assert(dragFrames.length>1,'No frames were presented during orbit');benchmarks.push(result);
      await capture('intact-'+presentation.toLowerCase()+'-benchmark');
    }
    report.performanceBenchmarks=benchmarks;return benchmarks;
  });
  await step('orbit motion and pan zoom',async()=>{await click('Fit');await orbit({recordFrames:true,count:72});await capture('orbit-settled');await orbit({button:'right',count:18,distance:45});const point=await center(pane);window.webContents.sendInputEvent({type:'mouseWheel',x:point.x,y:point.y,deltaX:0,deltaY:100,canScroll:true});await pause(700);await capture('pan-and-zoom');return {motionFrames:report.motion.filter(f=>f.kind==='orbit').length}});
  await step('cutaway controls and motion',async()=>{await click('Clip');await js(`(()=>{const e=[...document.querySelectorAll('.rl-control-popover input[type="checkbox"]')].find(e=>e.parentElement.textContent.trim()==='Cutaway');if(!e)throw Error('Missing cutaway checkbox');if(!e.checked)e.click()})()`);await pause(250);await select('Cutaway axis','x');const range=await js(`(()=>{const e=document.querySelector('input[aria-label="Cutaway position"]');return {min:+e.min,max:+e.max,value:+e.value}})()`),captureFrame=await record('cutaway');for(let i=0;i<15;i++){await setRange('Cutaway position',range.min+(range.max-range.min)*(.3+.4*i/14));await captureFrame()}await close();await capture('cutaway');return range});

  await step('native save and reopen preserves presentation cursor scope and layer choices',async()=>{await click('Save');await wait(`document.body.innerText.includes('Workspace saved')`,60);const filename=path.join(study,'workspace-view.json');assert(fs.existsSync(filename),'Native view preferences were not written');const saved=JSON.parse(fs.readFileSync(filename,'utf8'));fs.writeFileSync(path.join(output,'saved-disposable-view.json'),JSON.stringify(saved,null,2));const before=await state();await js(`(()=>{const b=[...document.querySelectorAll('button.header-button')].find(b=>b.querySelector('span')?.textContent.trim()==='Open');if(!b)throw Error('Missing Open');b.click()})()`);await wait(`!document.querySelector('.rl-viewer canvas')||document.querySelector('select[aria-label="Source scan"]')?.disabled===true`,60);await ready();const after=await state();assert(after.source===before.source,'Saved source changed');assert(JSON.stringify(after.presentation)===JSON.stringify(before.presentation),'Saved presentation mode changed');assert(after.cursor===before.cursor,'Saved cursor changed');assert(after.scope===before.scope,'Saved source fragment scope changed');assert(JSON.stringify(after.layers)===JSON.stringify(before.layers),'Saved structure selection changed');await click('Appearance');const appearance=await js(`({contour:document.querySelector('select[aria-label="Slice annotation style"]').value,preset:document.querySelector('select[aria-label="Volume rendering preset"]').value,density:+document.querySelector('input[aria-label="Scan volume density"]').value})`);await close();assert(appearance.contour==='contour','Saved contour presentation changed');await capture('reopened-disposable-study');await click('Save');await wait(`document.body.innerText.includes('Workspace saved')`,60);await pause(300);const roundTrip=JSON.parse(fs.readFileSync(filename,'utf8'));for(const id of Object.keys(saved.states)){for(const key of ['presentation','sliceOverlay','volumeRendering','layerOpacity','layerColor','clipping','clipBox','obliqueClip'])assert(JSON.stringify(roundTrip.states[id]?.settings[key])===JSON.stringify(saved.states[id]?.settings[key]),'Saved '+id+' '+key+' changed after reopen');const previous=saved.states[id]?.settings.camera,current=roundTrip.states[id]?.settings.camera;if(previous&&current)for(const key of ['position','target','up'])assert(previous[key].every((v,i)=>Math.abs(v-current[key][i])<1e-5),'Saved camera '+key+' changed')}return {source:after.source,cursor:after.cursor,scope:after.scope,appearance,settingsAndCameraRoundTrip:true,savedFile:filename}});
  fs.mkdirSync(path.join(output,'motion'),{recursive:true});fs.writeFileSync(path.join(output,'motion','frames.json'),JSON.stringify(report.motion,null,2));
  const telemetry=await js('window.fidelityEvents');
  const refinementMs=telemetry.refinement.map(e=>e.milliseconds).sort((a,b)=>a-b);report.refinementSummary={durationField:'milliseconds',count:telemetry.refinement.length,completed:telemetry.refinement.filter(e=>e.completed).length,p50Ms:refinementMs[Math.floor(refinementMs.length*.5)]??null,maxMs:Math.max(0,...refinementMs)};
  report.gpuSummary={count:telemetry.gpu.length,completed:telemetry.gpu.filter(e=>e.completed).length,maxMs:Math.max(0,...telemetry.gpu.map(e=>e.milliseconds))};
  report.resourceSummary={first:telemetry.frames[0]?.resources,last:telemetry.frames.at(-1)?.resources,maxGeometries:Math.max(0,...telemetry.frames.map(e=>e.resources.geometries)),maxTextures:Math.max(0,...telemetry.frames.map(e=>e.resources.textures))};
}
require('./main.cjs');
