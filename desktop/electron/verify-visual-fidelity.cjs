'use strict';
/** Actual Electron rendering checks. Writable operations target an APFS clone;
 * original manifest and every source array are checked with SHA-256. */
const {app,dialog}=require('electron');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path'),crypto=require('node:crypto');
const caseKind=process.env.FIDELITY_CTA?'cta':process.env.FIDELITY_MRA?'mra':'mri';
const profiles={
  mri:{source:/T1CE|T1C|T1[-_ ]?POST/i,presentation:'Segmentation',before:'mri',presented:'segmentation',provenance:/no acquired angiography/,ct:false},
  mra:{source:/MRA|TOF/i,presentation:'Vascular',before:'mra',presented:'mra-vascular',provenance:/MR angiography/,ct:false},
  cta:{source:/\bCTA\b|CT[-_ ]*ANGIO/i,presentation:'Vascular',before:'cta',presented:'cta-vascular',provenance:/CTA signal \+ sourced structures/,ct:true},
};
const profile=profiles[caseKind];
const showcaseOnly=Boolean(process.env.FIDELITY_SHOWCASE_ONLY);
const motionOnly=Boolean(process.env.FIDELITY_MOTION_ONLY);
const captureRecheck=Boolean(process.env.FIDELITY_RECHECK&&process.env.FIDELITY_CAPTURE_MOTION);
const fluidMotion=Boolean(process.env.FIDELITY_FLUID_MOTION);
const output=path.resolve(process.env.FIDELITY_OUTPUT||path.join(__dirname,'../../artifacts/desktop-renderer/medivis-fidelity-v2',caseKind));
fs.mkdirSync(output,{recursive:true});
const userData=fs.mkdtempSync(path.join(os.tmpdir(),'rslab-fidelity-user-'));
app.setPath('userData',userData);
const index=process.argv.indexOf('--anatomy');
if(index<0||!process.argv[index+1])throw Error('Pass --anatomy /absolute/patient-anatomy.json');
const originalSource=path.resolve(process.argv[index+1]),originalDirectory=path.dirname(originalSource);
const sourceDocument=JSON.parse(fs.readFileSync(originalSource,'utf8'));
const vesselLabels=sourceDocument.layers.filter(layer=>['vessel','vessels','artery','vein','sinus'].includes(layer.kind)).map(layer=>layer.label);
if(!vesselLabels.length)throw Error('Verification requires at least one supplied vessel layer');
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
const report={startedAt:Date.now(),caseKind,verificationMode:motionOnly?'motion-only':showcaseOnly?'showcase':process.env.FIDELITY_RECHECK?'recheck':'full',originalSource,sourceSeries:sourceDocument.series.map(({id,modality,intensity_units,provenance})=>({id,modality,intensity_units,provenance})),sourceVesselLabels:vesselLabels,disposableStudy:true,errors:[],checks:[],views:[],surface:[],frames:[],gpu:[],refinement:[],motion:[],motionRecordingReceipts:[]};
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
      const selectedSource=report.scans.find(s=>profile.source.test(s.label+' '+s.value));if(!selectedSource)throw Error('No requested acquired '+caseKind.toUpperCase()+' source option');
      const declaredSource=sourceDocument.series.find(s=>s.id===selectedSource.value);if(!declaredSource||!profile.source.test(declaredSource.modality))throw Error('Selected source modality does not match requested acquisition');report.selectedSource=declaredSource.id;
      await select('Source scan',selectedSource.value);await ready();await click('Fit 3D');await capture(profile.before+'-before-presentation');
      await click('Layout');await select('Camera preset','left');await js(`document.querySelector('button[aria-label="Close view controls"]').click()`);await capture(profile.before+'-before-left');
      await click(profile.presentation);await ready();await click('Fit 3D');await capture(profile.presented);
      await click('Layout');await select('Camera preset','left');await js(`document.querySelector('button[aria-label="Close view controls"]').click()`);await capture(profile.presented+'-left');
      if(process.env.FIDELITY_SUPERIOR){
        await click('Layout');await select('Camera preset','superior');await js(`document.querySelector('button[aria-label="Close view controls"]').click()`);await capture(profile.presented+'-superior');
        window.setSize(1050,720);await pause(700);await capture(caseKind+'-minimum-1050x720');report.minimumWindow={requested:[1050,720],actual:window.getSize(),content:window.getContentSize()};
      }
      report.initialPresentation=await js(`({source:document.querySelector('select[aria-label="Source scan"]').value,scope:document.querySelector('select[aria-label="Vessel display"]')?.value,framing:{fit:document.querySelector('select[aria-label="Source slice fit"]')?.value,zoom:document.querySelector('button[aria-label="Reset source slice zoom"]')?.textContent},layers:[...document.querySelectorAll('.compact-layer input[type="checkbox"]')].map(c=>({label:c.getAttribute('aria-label'),checked:c.checked})),provenance:document.querySelector('.presentation-source')?.textContent,body:document.body.innerText})`);
      fs.writeFileSync(path.join(output,'initial-presentation.json'),JSON.stringify(report.initialPresentation,null,2));
      report.checks.push({name:'initial '+caseKind.toUpperCase()+' '+profile.presentation.toLowerCase()+' captures',passed:true});
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
  const primaryLabels=sourceDocument.layers.filter(layer=>!['brain-region','review-region','head-envelope'].includes(layer.kind)).map(layer=>'Show '+layer.label);
  const step=async(name,action)=>{if(motionOnly&&!/source scope|orbit motion|cutaway controls|CTA axial showcase/.test(name))return;if(!motionOnly&&showcaseOnly&&!/source scope|presented orbit FPS|orbit motion|cutaway controls|CTA axial showcase|native save/.test(name))return;if(!motionOnly&&process.env.FIDELITY_RECHECK&&!showcaseOnly&&!(captureRecheck?/source scope|presented orbit FPS|orbit motion|cutaway controls|CTA axial showcase|native save/:/presented orbit FPS|native save/).test(name))return;console.log('Checking '+name);try{const evidence=await action();report.checks.push({name,passed:true,evidence})}catch(error){report.checks.push({name,passed:false,error:error.stack});console.error(name+': '+error.message);fs.writeFileSync(path.join(output,'failure-'+name.replace(/[^a-z0-9]+/gi,'-')+'.png'),(await window.webContents.capturePage()).toPNG())}};
  const state=()=>js(`({source:document.querySelector('select[aria-label="Source scan"]').value,presentation:[...document.querySelectorAll('.anatomy-presets button')].filter(b=>b.getAttribute('aria-pressed')==='true').map(b=>b.textContent),scope:document.querySelector('select[aria-label="Vessel display"]')?.value,framing:{fit:document.querySelector('select[aria-label="Source slice fit"]')?.value,zoom:document.querySelector('button[aria-label="Reset source slice zoom"]')?.textContent},layers:[...document.querySelectorAll('.compact-layer input[type="checkbox"]')].map(c=>({label:c.getAttribute('aria-label'),checked:c.checked})),cursor:document.querySelector('.rl-viewer-statusbar>span')?.textContent,slices:[...document.querySelectorAll('.rl-viewer-slice-image')].map(e=>e.getAttribute('aria-label')),provenance:document.querySelector('.presentation-source')?.textContent,warning:document.querySelector('.alignment-banner')?.textContent??document.body.innerText.split('\\n').filter(s=>/alignment|align|source|estimated/i.test(s)).join(' | ')})`);
  const close=async()=>{await js(`document.querySelector('button[aria-label="Close view controls"]')?.click()`);await pause(150)};
  const setRange=async(label,value)=>{await js(`(()=>{const e=[...document.querySelectorAll('input')].find(e=>e.getAttribute('aria-label')===${JSON.stringify(label)});if(!e)throw Error('Missing range '+${JSON.stringify(label)});Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(e,${JSON.stringify(String(value))});e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}))})()`);await pause(200)};
  const center=selector=>js(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});if(!e)throw Error('Missing pane');const r=e.getBoundingClientRect();return {x:Math.round(r.x+r.width*.5),y:Math.round(r.y+r.height*.5),w:r.width,h:r.height}})()`);
  const mouseClick=async point=>{window.webContents.sendInputEvent({type:'mouseDown',x:point.x,y:point.y,button:'left',clickCount:1});await pause(30);window.webContents.sendInputEvent({type:'mouseUp',x:point.x,y:point.y,button:'left',clickCount:1});await pause(350)};
  const toggle=async(label)=>{await js(`(()=>{const e=[...document.querySelectorAll('input[type="checkbox"]')].find(e=>e.getAttribute('aria-label')===${JSON.stringify(label)});if(!e||e.disabled)throw Error('Missing enabled toggle '+${JSON.stringify(label)});e.click()})()`);await pause(350)};
  const saveNative=async()=>{
    const filename=path.join(study,'workspace-view.json'),previous=fs.existsSync(filename)?fs.statSync(filename).mtimeMs:-1;
    await click('Save');
    for(let i=0;i<240;i++){
      if(fs.existsSync(filename)&&fs.statSync(filename).mtimeMs!==previous&&await js(`document.querySelector('select[aria-label="Source scan"]')?.disabled===false&&document.body.innerText.includes('Workspace saved')`))return {filename,saved:JSON.parse(fs.readFileSync(filename,'utf8'))};
      await pause(250);
    }
    throw Error('Native Save did not finish writing workspace-view.json');
  };
  const pane='.rl-viewer-anatomy-image';
  const motionRoot=path.resolve(process.env.FIDELITY_MOTION_OUTPUT||path.join(__dirname,'../../outputs/medivis-fidelity-v2',caseKind));
  const record=async(kind)=>{const directory=path.join(motionRoot,kind);fs.mkdirSync(directory,{recursive:true});const frames=[];return async()=>{const index=frames.length+1,filename=path.join(directory,String(index).padStart(4,'0')+'.png'),at=performance.now();fs.writeFileSync(filename,(await window.webContents.capturePage()).toPNG());const item={index,at,capturedAt:performance.now(),path:filename,kind,captureMethod:'capturePage'};frames.push(item);report.motion.push(item);return item}};
  const orbit=async({recordFrames=false,recordKind='orbit',count=90,button='left',distance=120}={})=>{
    const point=await center(pane),deliveredCapture=recordFrames&&fluidMotion;
    const captureFrame=recordFrames&&!deliveredCapture?await record(recordKind):null;
    const moves=deliveredCapture?240:count,travel=deliveredCapture?Math.max(distance,Math.min(340,Math.round(point.w*.45))):distance;
    const writes=[],captureErrors=[],firstFrame=report.motion.length;
    let subscribed=false,dragStarted=performance.now(),dragEnded=dragStarted;
    const directory=path.join(motionRoot,recordKind);
    if(deliveredCapture){
      fs.mkdirSync(directory,{recursive:true});let index=0;
      // onlyDirty=false is essential: each PNG must contain the complete
      // presented viewport, not a differently sized repainted rectangle.
      window.webContents.beginFrameSubscription(false,(image,dirtyRect)=>{
        const at=performance.now();
        try{
          const frameIndex=++index,filename=path.join(directory,String(frameIndex).padStart(4,'0')+'.png');
          // Reduce only this recording image before synchronous PNG encoding.
          // Retina delivery can be twice the logical viewport in each axis;
          // encoding those unused pixels delayed the actual pointer sequence.
          const nativePixelSize=image.getSize(),logicalViewport=window.getContentSize();
          const recording=image.resize({width:Math.min(nativePixelSize.width,1440,logicalViewport[0]),quality:'good'});
          const bytes=recording.toPNG({scaleFactor:1}),capturedAt=performance.now();
          const capturedPixelSize={width:bytes.readUInt32BE(16),height:bytes.readUInt32BE(20)};
          const item={index:frameIndex,at,capturedAt,path:filename,kind:recordKind,captureMethod:'delivered-frame-subscription',dirtyRect,nativePixelSize,capturedPixelSize,logicalViewport,encodingMs:capturedAt-at,bytes:bytes.length};
          report.motion.push(item);
          // Encoding is synchronous in Electron; disk writes do not stall
          // pointer delivery. Await all writes after ending the subscription.
          writes.push(fs.promises.writeFile(filename,bytes).then(()=>{item.writtenAt=performance.now()}).catch(error=>captureErrors.push(error.message)));
        }catch(error){captureErrors.push(error.message)}
      });subscribed=true;
    }
    try{
      dragStarted=performance.now();window.webContents.sendInputEvent({type:'mouseDown',x:point.x,y:point.y,button,clickCount:1});
      for(let i=1;i<=moves;i++){
        window.webContents.sendInputEvent({type:'mouseMove',x:point.x+Math.round(travel*i/moves),y:point.y+Math.round(15*Math.sin(i/moves*Math.PI)),modifiers:[button==='left'?'leftButtonDown':'rightButtonDown']});
        await pause(deliveredCapture?Math.max(1,dragStarted+4000*i/moves-performance.now()):16);
        if(captureFrame&&i%3===0)await captureFrame();
      }
      window.webContents.sendInputEvent({type:'mouseUp',x:point.x+travel,y:point.y,button,clickCount:1});dragEnded=performance.now();await pause(1000);
    }finally{
      // Never leak a recording subscription into the independent FPS check.
      if(subscribed)window.webContents.endFrameSubscription();
      window.webContents.sendInputEvent({type:'mouseUp',x:point.x+travel,y:point.y,button,clickCount:1});
      await Promise.all(writes);
    }
    if(recordFrames){
      const frames=report.motion.slice(firstFrame),duringDrag=frames.filter(frame=>frame.at>=dragStarted&&frame.at<=dragEnded);
      const receipt={kind:recordKind,captureMethod:deliveredCapture?'delivered-frame-subscription':'capturePage',frames:frames.length,duringDragFrames:duringDrag.length,requestedMoves:moves,requestedDragMs:deliveredCapture?4000:null,actualDragMs:dragEnded-dragStarted,travelPixels:travel,fullViewport:deliveredCapture,nativePixelSizes:[...new Set(frames.filter(frame=>frame.nativePixelSize).map(frame=>JSON.stringify(frame.nativePixelSize)))].map(value=>JSON.parse(value)),capturedPixelSizes:[...new Set(frames.filter(frame=>frame.capturedPixelSize).map(frame=>JSON.stringify(frame.capturedPixelSize)))].map(value=>JSON.parse(value)),recordingResize:deliveredCapture?'NativeImage.resize before PNG; logical viewport width, maximum1440 pixels; source scan arrays unchanged':null,encodingMs:frames.reduce((sum,frame)=>sum+(frame.encodingMs??0),0),captureOverheadIncluded:true,performanceUse:'Recording only. PNG encoding and writes are excluded from the separate no-capture orbit FPS benchmark.',errors:captureErrors};
      report.motionRecordingReceipts.push(receipt);
      if(captureErrors.length)throw Error('Delivered frame capture failed: '+captureErrors.join('; '));
      if(deliveredCapture&&duringDrag.length<2)throw Error('No delivered orbit frames were captured');
    }
    return {dragStarted,dragEnded,dragMilliseconds:dragEnded-dragStarted,idleMilliseconds:performance.now()-dragEnded};
  };
  const initial=await state();
  await step('source scope and provenance',async()=>{assert(initial.scope==='all','Source vessel fragments were silently filtered');assert(initial.provenance?.length>0,'Missing source provenance');assert(primaryLabels.every(label=>initial.layers.find(layer=>layer.label===label)?.checked),'A primary supplied anatomy layer was silently disabled');if(profile.ct){assert(sourceDocument.series.find(s=>s.id===initial.source)?.modality==='CTA','CT vascular evidence lacks a declared acquired CTA');assert(vesselLabels.some(label=>/arterial/i.test(label))&&vesselLabels.some(label=>/venous/i.test(label)),'CTA source labels are not separately represented')} return initial});
  await step('vascular presentation preserves primary selections',async()=>{await click('Vascular');await ready();await click('Fit 3D');await capture(caseKind+'-vascular-fit');const next=await state();assert(JSON.stringify(next.layers)===JSON.stringify(initial.layers),'Presentation changed selected structures');assert(next.scope==='all','Presentation filtered source fragments');assert(profile.provenance.test(next.provenance),'Modality provenance is misleading');return next});
  await step('structures and volume remain reversible',async()=>{await click('Structures');await capture('structures-only');assert(await js(`document.querySelector('button[aria-label="Show structures only"]').getAttribute('aria-pressed')==='true'`),'Structures button did not select mesh-only mode');await click('Volume');await capture('volume-and-structures');assert(await js(`document.querySelector('button[aria-label="Show scan volume with structures"]').getAttribute('aria-pressed')==='true'`),'Volume button did not restore scalar rendering');return await state()});
  await step('modality-safe preset controls and contour style',async()=>{await click('Appearance');const options=await js(`[...document.querySelector('select[aria-label="Volume rendering preset"]').options].map(o=>({value:o.value,disabled:o.disabled,label:o.textContent}))`);assert(options.filter(o=>o.value.startsWith('ct-')).every(o=>o.disabled===!profile.ct),profile.ct?'CT presets disabled on declared CTA':'CT presets enabled on MRI');if(profile.ct){await select('Volume rendering preset','ct-bone');await close();await capture('cta-bone-context');await click('Appearance');await select('Volume rendering preset','ct-angiography');await close();await capture('cta-angiography-source');await click('Appearance');await select('Volume rendering preset','auto')}const current=await js(`document.querySelector('select[aria-label="Slice annotation style"]').value`);assert(current==='contour','Presentation did not select contours');await select('Slice annotation style','fill');await close();await capture('slice-filled');await click('Appearance');await select('Slice annotation style','contour');await close();await capture('slice-contours');return options});
  await step('uniform and shell context retain source scope and camera',async()=>{
    await click('Fit 3D');const before=await state(),eventStart=await js('window.fidelityEvents.frames.length');
    await click('Appearance');await select('Context appearance','uniform');await close();await capture('context-uniform');
    const uniform=await state();await click('Appearance');await select('Context appearance','shell');await close();await capture('context-shell');
    const shell=await state();for(const candidate of [uniform,shell]){assert(candidate.source===before.source,'Context style changed the acquired scan');assert(candidate.scope==='all','Context style filtered source components');assert(candidate.cursor===before.cursor,'Context style changed linked RAS cursor');assert(JSON.stringify(candidate.layers)===JSON.stringify(before.layers),'Context style changed selected anatomy')}
    return {before,uniform,shell,interveningCameraCommands:0,frameEvents:await js(`window.fidelityEvents.frames.length-${eventStart}`),displayOnly:true};
  });
  await step('source slice framing zoom and RAS pointer agreement',async()=>{
    const evidence=[];
    const cursor=async()=>{const text=(await state()).cursor,match=text.match(/RAS\s+(-?[\d.]+)\s*\/\s*(-?[\d.]+)\s*\/\s*(-?[\d.]+)/);assert(match,'Missing numeric RAS coordinates');return match.slice(1).map(Number)};
    for(const fit of ['image','anatomy']){
      await select('Source slice fit',fit);
      await js(`document.querySelector('button[aria-label="Reset source slice zoom"]').click()`);await pause(250);
      for(const zoomed of [false,true]){
        if(zoomed){await js(`document.querySelector('button[aria-label="Zoom source slices in"]').click()`);await pause(250)}
        const metric=await js(`(()=>{const pane=document.querySelector('.rl-viewer-axial .rl-viewer-slice-image'),r=pane.getBoundingClientRect(),scale=pane.querySelector('.rl-viewer-scale');return {x:Math.round(r.x+r.width*.5),y:Math.round(r.y+r.height*.5),pixelsPerMm:scale.getBoundingClientRect().width/parseFloat(scale.textContent),label:pane.getAttribute('aria-label')}})()`);
        assert(metric.pixelsPerMm>0,'Missing physical source scale');
        await mouseClick(metric);const from=await cursor(),pixels=12;
        await mouseClick({...metric,x:metric.x+pixels});const to=await cursor(),expectedMm=pixels/metric.pixelsPerMm;
        const errorMm=Math.abs((to[0]-from[0])-expectedMm),toleranceMm=.25+1/metric.pixelsPerMm;
        assert(errorMm<toleranceMm,'Source slice pointer disagrees with rendered physical scale for '+fit+' zoom '+zoomed);
        assert(Math.abs(to[1]-from[1])<.21&&Math.abs(to[2]-from[2])<.21,'Horizontal pointer changed orthogonal RAS coordinates');
        const current=await state();assert(current.scope==='all','Slice framing restricted source fragments');
        const name='source-slices-'+fit+(zoomed?'-zoomed':'');await capture(name);
        evidence.push({fit,zoom:current.framing.zoom,from,to,pixels,expectedMm,errorMm,toleranceMm,pixelsPerMm:metric.pixelsPerMm,source:current.source,scope:current.scope,sourceLabel:metric.label});
      }
    }
    return evidence;
  });
  await step('linked slice cursor click',async()=>{const before=await state(),point=await center('.rl-viewer-slice-image');await mouseClick({...point,x:point.x+Math.round(point.w*.12)});await capture('linked-slice-click');const after=await state();assert(after.cursor!==before.cursor,'Linked RAS cursor did not move');assert(after.slices.some((s,i)=>s!==before.slices[i]),'Other linked slice positions did not update');return {before:before.cursor,after:after.cursor,slices:after.slices}});
  const tumor=initial.layers.find(l=>/tumor/i.test(l.label));
  if(tumor)await step('tumor toggle and cross-scan shared structures',async()=>{await toggle(tumor.label);await capture('tumor-hidden');let next=await state();assert(next.layers.find(l=>l.label===tumor.label)?.checked===false,'Tumor did not hide');const ct=report.scans.find(s=>/ct-brain/i.test(s.value));if(ct){await select('Source scan',ct.value);await ready();assert((await state()).layers.find(l=>l.label===tumor.label)?.checked===false,'Scan switch restored a hidden tumor');await click('Fit 3D');await capture('ct-shared-structures');await select('Source scan',initial.source);await ready();assert((await state()).layers.find(l=>l.label===tumor.label)?.checked===false,'Return scan restored hidden tumor')}await toggle(tumor.label);await click('Fit 3D');await capture('tumor-restored');return await state()});
    await step('layer opacity slider',async()=>{const selected=await js(`(()=>{const b=[...document.querySelectorAll('.layer-name')].find(b=>${JSON.stringify(caseKind==='mri'?['brain',...vesselLabels]:vesselLabels)}.includes(b.textContent.trim()));if(!b)throw Error('Missing opacity layer');b.click();return b.textContent.trim()})()`);await pause(200);const label=selected+' opacity',before=await js(`[...document.querySelectorAll('input')].find(e=>e.getAttribute('aria-label')===${JSON.stringify(label)}).value`);await setRange(label,.04);await capture('layer-opacity-low');const low=await js(`[...document.querySelectorAll('input')].find(e=>e.getAttribute('aria-label')===${JSON.stringify(label)}).value`);assert(Math.abs(+low-.04)<.001,'Layer opacity did not update');await setRange(label,+before);await js(`document.querySelector('button[aria-label="Close layer details"]').click()`);return {layer:selected,before,low}});
  await step('source fragment inspection highlights and locates without accepting a correction',async()=>{
    const before=await state();
    await js(`(()=>{const b=[...document.querySelectorAll('.layer-name')].find(b=>${JSON.stringify(vesselLabels)}.includes(b.textContent.trim()));if(!b)throw Error('Missing vessel layer');b.click()})()`);await pause(200);
    await click('Inspect vessel fragments');await ready();await wait(`!!document.querySelector('.component-list input[type="checkbox"]')`,60);await capture('fragment-review-source');
    const selected=await js(`(()=>{const e=document.querySelector('.component-list input[type="checkbox"]');e.click();return e.getAttribute('aria-label')})()`);await ready();
    await js(`document.querySelector('.component-list input[type="checkbox"]').parentElement.querySelector('button').click()`);await pause(300);await capture('fragment-selected-linked');
    const highlighted=await state();assert(highlighted.scope==='all','Fragment inspection restricted source scope');assert(highlighted.cursor!==before.cursor,'Fragment locate did not move linked cursor');
    await toggle(selected);await ready();await js(`document.querySelector('button[aria-label="Close layer details"]').click()`);await capture('fragment-review-restored-source');
    const after=await state();assert(after.scope==='all','Full source was not restored');assert(!await js(`!!document.querySelector('.unified-correction')`),'Fragment inspection unexpectedly entered a correction draft');return {selected,before:before.cursor,located:highlighted.cursor,sourceScope:after.scope,acceptedCorrection:false};
  });
  await step('presented orbit FPS without capture for intact default presentations',async()=>{
    const benchmarks=[];
    for(const presentation of (showcaseOnly||captureRecheck?[profile.presentation]:caseKind==='mri'?['Segmentation','Vascular']:['Vascular'])){
      await close();await click(presentation);await ready();await click('Fit 3D');await pause(1000);
      const baseline=await state(),eventStart=await js('window.fidelityEvents.refinement.length');
      assert(baseline.scope==='all','Performance presentation restricted source fragments');
      const timestamps=[];window.webContents.beginFrameSubscription(true,()=>timestamps.push(performance.now()));
      const started=performance.now();let drag;try{drag=await orbit({count:120})}finally{window.webContents.endFrameSubscription()}
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
  await step('orbit motion and pan zoom',async()=>{await click('Fit 3D');await orbit({recordFrames:true,count:72});await capture('orbit-settled');await orbit({button:'right',count:18,distance:45});const point=await center(pane);window.webContents.sendInputEvent({type:'mouseWheel',x:point.x,y:point.y,deltaX:0,deltaY:100,canScroll:true});await pause(700);await capture('pan-and-zoom');return {presentation:(await state()).presentation,sourceScope:(await state()).scope,motionFrames:report.motion.filter(f=>f.kind==='orbit').length}});
  await step('cutaway controls and motion',async()=>{await click('Clip');await js(`(()=>{const e=[...document.querySelectorAll('.rl-control-popover input[type="checkbox"]')].find(e=>e.parentElement.textContent.trim()==='Cutaway');if(!e)throw Error('Missing cutaway checkbox');if(!e.checked)e.click()})()`);await pause(250);await select('Cutaway axis','x');const range=await js(`(()=>{const e=document.querySelector('input[aria-label="Cutaway position"]');return {min:+e.min,max:+e.max,value:+e.value}})()`),captureFrame=await record('cutaway');for(let i=0;i<15;i++){await setRange('Cutaway position',range.min+(range.max-range.min)*(.3+.4*i/14));await captureFrame()}await close();await capture('cutaway');return range});


  if(profile.ct)await step('CTA axial showcase retains source artery vein labels',async()=>{
    await close();await click('Vascular');await ready();await click('Fit 3D');await capture('cta-polished-default');
    const before=await state();assert(before.scope==='all','CTA showcase restricted source vessel labels');
    await select('Skull view','exterior');await ready();await capture('cta-opaque-high-attenuation-context');
    // The explicit native cutaway control selects +RAS superior removal, keeps
    // the inferior source extent, and makes actual high-attenuation context
    // opaque. This is a display plane; it does not alter any source voxel.
    await select('Skull view','cutaway');await click('Clip');await select('Cutaway axis','z');
    const range=await js(`(()=>{const e=document.querySelector('input[aria-label="Cutaway position"]');return {min:+e.min,max:+e.max,value:+e.value}})()`);
    const position=range.min+(range.max-range.min)*.7;
    await setRange('Cutaway position',position);await close();
    await click('Layout');await select('Camera preset','superior');await close();await capture('cta-axial-bone-cutaway-superior');
    await orbit({recordFrames:true,recordKind:'cta-axial-cutaway-orbit',count:48,distance:55});await capture('cta-axial-bone-cutaway-orbit-settled');
    await click('Layout');await select('Camera preset','superior');await close();await click('Clip');
    const captureFrame=await record('cta-axial-cutaway');
    for(let i=0;i<15;i++){await setRange('Cutaway position',range.min+(range.max-range.min)*(.62+.16*i/14));await captureFrame()}
    await setRange('Cutaway position',position);await close();await capture('cta-axial-bone-cutaway');
    const after=await state();assert(after.scope==='all','CTA cutaway restricted source fragments');assert(after.source===before.source,'CTA cutaway changed source acquisition');assert(JSON.stringify(after.layers)===JSON.stringify(before.layers),'CTA cutaway disabled a supplied artery or vein label');
    return {source:after.source,scope:after.scope,layerSelection:after.layers,plane:{axis:'z',flipped:true,retainedSide:'RAS z <= cutaway position',fraction:.7,positionMm:position,sourceRangeMm:range},context:'Opaque source high-attenuation threshold context; not a verified bone class',camera:'superior',displayClippingOnly:true,motionFrames:report.motion.filter(f=>f.kind.startsWith('cta-axial')).length};
  });

  await step('native save and reopen preserves presentation cursor scope and layer choices',async()=>{await select('Source slice fit','image');await js(`document.querySelector('button[aria-label="Reset source slice zoom"]').click()`);await pause(200);await js(`document.querySelector('button[aria-label="Zoom source slices in"]').click()`);await pause(300);const {filename,saved}=await saveNative();fs.writeFileSync(path.join(output,'saved-disposable-view.json'),JSON.stringify(saved,null,2));const before=await state();await js(`window.fidelityBeforeReopenCanvas=document.querySelector('.rl-viewer canvas')`);await js(`(()=>{const b=[...document.querySelectorAll('button.header-button')].find(b=>b.querySelector('span')?.textContent.trim()==='Open');if(!b)throw Error('Missing Open');b.click()})()`);await wait(`document.querySelector('.rl-viewer canvas')!==window.fidelityBeforeReopenCanvas`,60);await ready();const after=await state();assert(after.source===before.source,'Saved source changed');assert(JSON.stringify(after.presentation)===JSON.stringify(before.presentation),'Saved presentation mode changed');assert(after.cursor===before.cursor,'Saved cursor changed');assert(after.scope===before.scope,'Saved source fragment scope changed');assert(JSON.stringify(after.layers)===JSON.stringify(before.layers),'Saved structure selection changed');await click('Appearance');const appearance=await js(`({contour:document.querySelector('select[aria-label="Slice annotation style"]').value,preset:document.querySelector('select[aria-label="Volume rendering preset"]').value,density:+document.querySelector('input[aria-label="Scan volume density"]').value,contextStyle:document.querySelector('select[aria-label="Context appearance"]').value,framing:{fit:document.querySelector('select[aria-label="Source slice fit"]').value,zoom:document.querySelector('button[aria-label="Reset source slice zoom"]').textContent}})`);await close();assert(appearance.contour==='contour','Saved contour presentation changed');assert(appearance.contextStyle==='shell','Saved shell context changed');assert(JSON.stringify(after.framing)===JSON.stringify(before.framing),'Saved source slice framing changed');await capture('reopened-disposable-study');const {saved:roundTrip}=await saveNative();for(const id of Object.keys(saved.states)){for(const key of ['presentation','sliceOverlay','sliceFraming','contextStyle','volumeRendering','layerOpacity','layerColor','clipping','clipBox','obliqueClip'])assert(JSON.stringify(roundTrip.states[id]?.settings[key])===JSON.stringify(saved.states[id]?.settings[key]),'Saved '+id+' '+key+' changed after reopen');const previous=saved.states[id]?.settings.camera,current=roundTrip.states[id]?.settings.camera;if(previous&&current)for(const key of ['position','target','up'])assert(previous[key].every((v,i)=>Math.abs(v-current[key][i])<1e-5),'Saved camera '+key+' changed')}return {source:after.source,cursor:after.cursor,scope:after.scope,appearance,settingsAndCameraRoundTrip:true,savedFile:filename}});
  fs.mkdirSync(path.join(output,'motion'),{recursive:true});fs.writeFileSync(path.join(output,'motion','frames.json'),JSON.stringify(report.motion,null,2));
  const telemetry=await js('window.fidelityEvents');
  const refinementMs=telemetry.refinement.map(e=>e.milliseconds).sort((a,b)=>a-b);report.refinementSummary={durationField:'milliseconds',count:telemetry.refinement.length,completed:telemetry.refinement.filter(e=>e.completed).length,p50Ms:refinementMs[Math.floor(refinementMs.length*.5)]??null,maxMs:Math.max(0,...refinementMs)};
  report.gpuSummary={count:telemetry.gpu.length,completed:telemetry.gpu.filter(e=>e.completed).length,maxMs:Math.max(0,...telemetry.gpu.map(e=>e.milliseconds))};
  report.resourceSummary={first:telemetry.frames[0]?.resources,last:telemetry.frames.at(-1)?.resources,maxGeometries:Math.max(0,...telemetry.frames.map(e=>e.resources.geometries)),maxTextures:Math.max(0,...telemetry.frames.map(e=>e.resources.textures))};
}
require('./main.cjs');
