'use strict';
// Rehearsal extension of the production interaction benchmark. All actions use
// the normal React controls, preload, authorized IPC, and Python sidecar.
const {app, ipcMain, dialog,screen} = require('electron');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const {execFileSync} = require('node:child_process');
const crypto = require('node:crypto');
function sourceTreeHash(directory,extension){const digest=crypto.createHash('sha256');const visit=folder=>{for(const item of fs.readdirSync(folder,{withFileTypes:true}).sort((a,b)=>a.name.localeCompare(b.name))){const filename=path.join(folder,item.name);if(item.isDirectory()&&item.name!=='__pycache__')visit(filename);else if(item.isFile()&&filename.endsWith(extension)){digest.update(path.relative(directory,filename));digest.update(fs.readFileSync(filename));}}};visit(directory);return digest.digest('hex');}
const root = path.resolve(__dirname, '../..');
const output = path.resolve(process.env.INTERACTION_OUTPUT || path.join(root, 'testcases/performance/rehearsal-2026-10-19/current'));
fs.mkdirSync(output, {recursive:true});
const userData = fs.mkdtempSync(path.join(os.tmpdir(), 'rslab-rehearsal-benchmark-'));
app.setPath('userData', userData);
// Optional frozen Python source is confined to this benchmark process. It
// permits balanced baseline/comparison runs without changing the checkout.
let enginePid=null;
{
  const sidecar = require('./sidecar.cjs'), Original = sidecar.Sidecar;
  sidecar.Sidecar = class extends Original {
    constructor(config) {
      super({...config,...(process.env.INTERACTION_PYTHON_SOURCE?{sourcePath:path.resolve(process.env.INTERACTION_PYTHON_SOURCE)}:{})});enginePid=this.child.pid;
      this.on('diagnostic',message=>report.engineEvents.push({at:performance.now(),event:'diagnostic',message}));
      this.on('event',message=>{if(message.event==='engineStopped')report.engineEvents.push({at:performance.now(),...message});});
      this.child.on('exit',(code,signal)=>report.engineEvents.push({at:performance.now(),event:'exit',code,signal}));
    }
  };
}
const launch = performance.now();
const secondaryOnly=process.env.INTERACTION_DISPLAY==='secondary';
if(secondaryOnly&&process.platform==='darwin')app.whenReady().then(()=>app.setActivationPolicy('accessory'));
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const sourcePath = process.argv[process.argv.indexOf('--case')+1];
const report = {
  schema:'resectionlab.rehearsal-performance.v1', label:process.env.INTERACTION_LABEL || 'current',
  commit:execFileSync('git', ['rev-parse','HEAD'], {cwd:root,encoding:'utf8'}).trim(),
  diffHash:crypto.createHash('sha256').update(execFileSync('git',['diff'],{cwd:root})).digest('hex'),
  recordedAt:new Date().toISOString(),launchAt:launch, hardware:{cpus:os.cpus().map(c=>c.model), memoryBytes:os.totalmem(),platform:os.platform(),arch:os.arch(),release:os.release()},
  harnessSha256:crypto.createHash('sha256').update(fs.readFileSync(__filename)).digest('hex'),
  versions:process.versions, window:[1600,1000], runs:[], ipc:[], errors:[], checks:{}, memory:[], windowEvents:[],engineEvents:[],lifecycle:[],
  dependencies:Object.fromEntries(['three','three-mesh-bvh','react','vite','typescript','electron'].map(name=>[name,JSON.parse(fs.readFileSync(path.join(root,'desktop/node_modules',name,'package.json'),'utf8')).version])),
  implementationIdentity:{pythonSourceRoot:process.env.INTERACTION_PYTHON_SOURCE||path.join(root,'src'),
    pythonTreeSha256:sourceTreeHash(process.env.INTERACTION_PYTHON_SOURCE||path.join(root,'src'),'.py'),
    rendererJavaScriptTreeSha256:sourceTreeHash(path.join(root,'desktop/dist'),'.js'),
    rendererIndexSha256:crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'desktop/dist/index.html'))).digest('hex')},
  source:sourcePath?{path:sourcePath,sha256:crypto.createHash('sha256').update(fs.readFileSync(sourcePath)).digest('hex')}:null,
  measurement:{primary:'No frame subscription or video capture during primary timing. anatomy-frame is CPU submission, not screen presentation.',
    visible:'consistentDisplayMs ends after pending state/meshes clear and the next renderer submission; Electron delivery measured only in the separate observer control.',
    gpu:'Legacy anatomy-gpu is submission-to-fence-poll upper bound. Only non-disjoint anatomy-gpu-time samples measure GPU elapsed time.',
    bytes:'IPC JSON payload bytes are estimated after the timed handler; readAsset byteLength is exact for that boundary. Neither is universal zero-copy.',
    cold:'Fresh Electron/Python processes; OS file cache is uncontrolled and explicitly not a cold disk claim.'}
};
let lastResponse = null, nextOpenPath = null, saveTarget=path.join(output,'saved-rehearsal.ressectionlab');
const originalHandle = ipcMain.handle.bind(ipcMain);
ipcMain.handle = (channel, handler) => originalHandle(channel, async (...args) => {
  const started = performance.now();
  try {
    const result = await handler(...args), ended = performance.now();
    const entry = {channel,started,ms:ended-started,command:args[1]?.command};
    if (channel === 'research:readAsset') entry.bytes = result?.byteLength ?? 0;
    if (channel === 'research:surgicalRehearsal') {
      lastResponse = result;
      entry.position=result?.position; entry.stateId=result?.view?.stateId ?? result?.stateId;
      entry.outcome=result?.outcome?.status ?? result?.status;
      entry.requestJsonBytes=Buffer.byteLength(JSON.stringify(args[1]));
      entry.responseJsonBytes=Buffer.byteLength(JSON.stringify(result));
    }
    report.ipc.push(entry); return result;
  } catch(error) { report.ipc.push({channel,started,ms:performance.now()-started,error:String(error)}); throw error; }
});
const originalOpen = dialog.showOpenDialog.bind(dialog);
dialog.showOpenDialog = (...args) => nextOpenPath ? Promise.resolve({canceled:false,filePaths:[nextOpenPath]}) : originalOpen(...args);
dialog.showSaveDialog = async () => ({canceled:false,filePath:saveTarget});
// Only this disposable process owns the mutations and saved testing bundle.
dialog.showMessageBox = async () => ({response:1});
const persist = () => fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));
app.on('browser-window-created', (_event, window) => {
  // This disposable foreground benchmark must not become an occluded app when
  // macOS changes document focus during loading or synthetic pointer gestures.
  let targetDisplay=null;
  if(secondaryOnly){
    targetDisplay=screen.getAllDisplays().find(display=>display.id!==screen.getPrimaryDisplay().id);
    if(!targetDisplay)throw Error('A secondary display is required for this benchmark');
    const area=targetDisplay.workArea,width=Math.min(1480,area.width),height=Math.min(900,area.height);
    report.window=[width,height];window.setBounds({x:area.x+Math.floor((area.width-width)/2),y:area.y+Math.floor((area.height-height)/2),width,height});
    window.show=window.showInactive.bind(window);
    report.monitor={selected:targetDisplay,primary:screen.getPrimaryDisplay(),all:screen.getAllDisplays()};
    report.measurement.foreground='Visible secondary display, background throttling disabled; native focus is informational while the user works on the primary display. Pointer input uses the target WebContents CDP transport, without OS mouse movement or app activation. These runs are not exclusive foreground measurements.';
  }
  window.setAlwaysOnTop(true);
  const visibleOnTarget=()=>{
    if(!window.isVisible()||window.isMinimized())return false;
    if(!targetDisplay)return true;
    const a=targetDisplay.workArea,b=window.getBounds();return b.x>=a.x&&b.y>=a.y&&b.x+b.width<=a.x+a.width&&b.y+b.height<=a.y+a.height;
  };
  window.on('focus',()=>report.windowEvents.push({event:'focus',at:performance.now()}));
  window.on('blur',()=>{
    report.windowEvents.push({event:'blur',at:performance.now()});
    if(!secondaryOnly)setTimeout(()=>{if(!window.isDestroyed()){app.focus({steal:true});window.focus();window.webContents.focus();}},0);
  });
  window.webContents.on('console-message', event => {if(event.level==='error')report.errors.push(event.message);});
  window.webContents.on('render-process-gone',(_event,details)=>report.lifecycle.push({at:performance.now(),event:'render-process-gone',...details}));
  window.on('closed',()=>report.lifecycle.push({at:performance.now(),event:'window-closed'}));
  window.webContents.once('did-finish-load', async () => {
    const js = code => window.webContents.executeJavaScript(code,true);
    const sendPointer=async event=>{
      if(!secondaryOnly){window.webContents.sendInputEvent(event);return;}
      const type={mouseDown:'mousePressed',mouseUp:'mouseReleased',mouseMove:'mouseMoved'}[event.type];
      return window.webContents.debugger.sendCommand('Input.dispatchMouseEvent',{type,x:event.x,y:event.y,button:event.type==='mouseMove'?'none':event.button??'left',buttons:event.type==='mouseUp'?0:event.type==='mouseDown'||event.modifiers?.includes('leftButtonDown')?1:0,clickCount:event.clickCount??0,pointerType:'mouse'});
    };
    const wait = async (expr, timeout=240000) => {
      const start=performance.now();
      while(performance.now()-start<timeout) {if(await js(expr))return;await pause(20);}
      throw Error('Timed out: '+expr);
    };
    const ready = () => wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')&&!document.querySelector('.rl-viewer-preparing')&&!document.querySelector('.surgical-panel [role="status"]')?.textContent.includes('Computing instrument')&&(!document.querySelector('.rl-viewer').dataset.surfacesPending||document.querySelector('.rl-viewer').dataset.surfacesPending==='0')&&!document.querySelector('.rl-viewer').dataset.layersPreparing`);
    const idle = async () => {
      await wait(`![...document.querySelectorAll('.surgical-panel [role="status"]')].some(e=>e.textContent.includes('Computing instrument')||e.textContent.includes('Preparing accepted display'))`);
      // React may clear the command status before its passive renderer effect
      // constructs the next worker. Flush those effects before checking meshes.
      await js('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))');
      await ready();
      await wait(`![...window.perfWorkers??[]].some(w=>w.perfSurfaceJobs>0)`);
    };
    const click = label => js(`(()=>{const b=[...document.querySelectorAll('button')].find(e=>e.textContent.trim()===${JSON.stringify(label)}||e.getAttribute('aria-label')===${JSON.stringify(label)}||e.querySelector('span')?.textContent.trim()===${JSON.stringify(label)});if(!b||b.disabled)throw Error('Missing enabled button '+${JSON.stringify(label)});b.click();})()`);
    const setValue = (selector,value,type='change') => js(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});if(!e||e.disabled)throw Error('Missing enabled control '+${JSON.stringify(selector)});window.perfInputs.push({at:performance.now(),selector:${JSON.stringify(selector)},value:${JSON.stringify(value)}});const proto=e.tagName==='SELECT'?HTMLSelectElement.prototype:HTMLInputElement.prototype;Object.getOwnPropertyDescriptor(proto,'value').set.call(e,${JSON.stringify(String(value))});e.dispatchEvent(new Event(${JSON.stringify(type)},{bubbles:true}));})()`);
    const collect = () => js(`({frames:window.perfFrames,gpu:window.perfGpu,gpuTimes:window.perfGpuTimes,picks:window.perfPicks,work:window.perfWork,inputs:window.perfInputs,pointerEvents:window.perfPointerEvents,longTasks:window.perfLong,states:window.perfStates,heap:performance.memory?.usedJSHeapSize,workers:window.perfWorkerCounts})`);
    const memory = () => {let backend=null;try{backend={pid:enginePid,rssKiB:Number(execFileSync('ps',['-o','rss=','-p',String(enginePid)],{encoding:'utf8'}).trim())};}catch{}report.memory.push({at:performance.now(),backend,processes:app.getAppMetrics().map(p=>({pid:p.pid,type:p.type,memory:p.memory,cpu:p.cpu}))});};
    const run = async (name, action, subscribed=false) => {
      window.show();
      if(!secondaryOnly){app.focus({steal:true});window.focus();const focusStarted=performance.now();while(!window.isFocused()){if(performance.now()-focusStarted>5000)throw Error('Foreground focus was not acquired');await pause(20);}}
      await idle(); await pause(180);
      await js(`window.perfFrames=[];window.perfGpu=[];window.perfGpuTimes=[];window.perfPicks=[];window.perfWork=[];window.perfInputs=[];window.perfLong=[];window.perfStates=[];window.perfPointerEvents={moves:0,downs:0,doubleClicks:0,trusted:0};`);
      // Loading/effect work above may outlast native app activation. Acquire
      // focus at the actual timing boundary as well as at setup.
      if(!secondaryOnly){app.focus({steal:true});window.focus();window.webContents.focus();const inputFocusAt=performance.now();while(!window.isFocused()){if(performance.now()-inputFocusAt>5000)throw Error('Foreground focus was lost before input');app.focus({steal:true});window.focus();await pause(20);}}
      if(!visibleOnTarget())throw Error('Benchmark window must stay visible within its selected display');
      const r={name,subscribed,started:performance.now(),windowFocused:window.isFocused(),windowAlwaysOnTop:window.isAlwaysOnTop(),ipcStart:report.ipc.length,delivered:[]};report.runs.push(r);
      r.windowVisibleOnTarget=visibleOnTarget();r.windowBounds=window.getBounds();
      r.inputAt=await js('performance.now()');
      if(subscribed)window.webContents.beginFrameSubscription(true,()=>r.delivered.push(performance.now()));
      await action();r.actionReturnedAt=performance.now();await idle();
      r.appliedAt=await js('performance.now()');
      await js('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))');
      r.consistentDisplayAt=await js('performance.now()');
      await pause(200);
      if(subscribed)window.webContents.endFrameSubscription();
      Object.assign(r,await collect());r.end=performance.now();r.ipc=report.ipc.slice(r.ipcStart);
      r.windowFocusedAtEnd=window.isFocused();
      r.windowVisibleOnTargetAtEnd=visibleOnTarget();
      r.focusEvents=report.windowEvents.filter(event=>event.at>=r.started&&event.at<=r.end);
      r.inputToAppliedStateMs=r.states.length?r.states.at(-1).at-r.inputAt:null;
      const afterState=r.frames.find(f=>f.at>=(r.states.at(-1)?.at??r.inputAt));
      r.inputToNextSubmissionMs=afterState?afterState.at-r.inputAt:null;
      r.state={stateId:lastResponse?.view?.stateId,position:lastResponse?.position,metrics:lastResponse?.view?.metrics,removedCells:lastResponse?.view?.removed?.length,
        displayed:await js(`({stateId:document.querySelector('.surgical-panel')?.dataset.stateId,position:document.querySelector('.surgical-panel')?.dataset.recordedPosition})`)};
      r.lastSubmittedStateId=await js('window.perfLatestFrame?.surgicalStateId??null');
      r.displayConsistent=r.state.displayed.stateId&&r.lastSubmittedStateId?r.state.displayed.stateId===r.lastSubmittedStateId:null;
      memory();persist();console.log(name,JSON.stringify({ms:r.consistentDisplayAt-r.inputAt,frames:r.frames.length,ipc:r.ipc.map(i=>[i.command,i.ms])}));
    };
    const gesture = async () => {
      const p=await js(`(()=>{const r=document.querySelector('.rl-viewer-anatomy-image').getBoundingClientRect();return {x:Math.round(r.x+r.width*.5),y:Math.round(r.y+r.height*.5)};})()`);
      await sendPointer({type:'mouseDown',...p,button:'left',clickCount:1});
      for(let i=1;i<=60;i++){await sendPointer({type:'mouseMove',x:p.x+Math.round(60*Math.sin(i/60*Math.PI*2)),y:p.y+Math.round(20*Math.sin(i/30*Math.PI*2)),modifiers:['leftButtonDown']});await pause(16);}
      await sendPointer({type:'mouseUp',...p,button:'left',clickCount:1});
    };
    try {
      if(secondaryOnly)window.webContents.debugger.attach('1.3');
      window.webContents.setBackgroundThrottling(false);window.setSize(...report.window);window.show();if(!secondaryOnly){app.focus({steal:true});window.focus();}
      await ready();report.startupMs=performance.now()-launch;
      report.display=await js(`({width:innerWidth,height:innerHeight,dpr:devicePixelRatio,visible:document.visibilityState})`);
      report.gpuInfo=await app.getGPUInfo('basic');memory();
      await js(`window.perfFrames=[];window.perfGpu=[];window.perfGpuTimes=[];window.perfPicks=[];window.perfWork=[];window.perfInputs=[];window.perfLong=[];window.perfStates=[];window.perfWorkerCounts={created:0,live:0,peak:0};window.perfWorkers=new Set();
        const OriginalWorker=window.Worker;window.Worker=class extends OriginalWorker{
          constructor(...args){super(...args);this.perfKind=String(args[0]);this.perfSurface=['surface','incrementalSurface','labelPacking'].some(name=>this.perfKind.includes('/'+name+'.worker'));this.perfSurfaceJobs=0;
            this.addEventListener('message',()=>{if(this.perfSurface)this.perfSurfaceJobs=Math.max(0,this.perfSurfaceJobs-1);});
            window.perfWorkers.add(this);window.perfWorkerCounts.created++;window.perfWorkerCounts.live++;window.perfWorkerCounts.peak=Math.max(window.perfWorkerCounts.peak,window.perfWorkerCounts.live);this.counted=true;
          }
          postMessage(...args){
            if(this.perfSurface)this.perfSurfaceJobs++;
            const inject=window.perfFailNextModel&&this.perfKind.includes('/incrementalSurface.worker')&&args[0]?.layerId;
            if(inject)window.perfFailNextModel=false;
            try{const result=super.postMessage(...args);
              if(inject)setTimeout(()=>{window.perfModeledFault={at:performance.now(),worker:this.perfKind,layer:args[0].layerId,pendingJobs:this.perfSurfaceJobs};this.dispatchEvent(new ErrorEvent('error',{message:'Injected pending modeled surface failure'}));},100);
              return result;
            }catch(error){if(this.perfSurface)this.perfSurfaceJobs--;throw error;}
          }
          terminate(){if(this.counted){window.perfWorkers.delete(this);window.perfWorkerCounts.live--;this.counted=false;}return super.terminate();}
        };
        document.addEventListener('anatomy-frame',e=>{window.perfLatestFrame=e.detail;window.perfFrames.push(e.detail);});
        document.addEventListener('anatomy-gpu',e=>window.perfGpu.push(e.detail));
        document.addEventListener('anatomy-gpu-time',e=>window.perfGpuTimes.push(e.detail));
        document.addEventListener('anatomy-pick',e=>window.perfPicks.push(e.detail));
        document.addEventListener('anatomy-work',e=>window.perfWork.push(e.detail));
        window.perfPointerEvents={moves:0,downs:0,doubleClicks:0,trusted:0};
        for(const [type,key]of [['pointermove','moves'],['pointerdown','downs'],['dblclick','doubleClicks']])document.addEventListener(type,e=>{if(e.target.closest?.('.rl-viewer')){window.perfPointerEvents[key]++;if(e.isTrusted)window.perfPointerEvents.trusted++;}},true);
        new PerformanceObserver(list=>window.perfLong.push(...list.getEntries().map(e=>({at:e.startTime,ms:e.duration})))).observe({type:'longtask',buffered:false});
        let previous='';new MutationObserver(()=>{const text=document.querySelector('.surgical-panel .surgical-actions span')?.textContent??[...document.querySelectorAll('.surgical-panel span')].find(e=>e.textContent.startsWith('Recorded state '))?.textContent;if(text&&text!==previous){previous=text;window.perfStates.push({at:performance.now(),text});}}).observe(document.body,{subtree:true,childList:true,characterData:true});`);
      await run('enter-rehearsal',async()=>{await click('Rehearse');await wait(`!!document.querySelector('.surgical-slots')`);});
      report.scene={caseHash:lastResponse?.caseHash,sceneId:lastResponse?.view?.scene?.sceneId,shape:lastResponse?.view?.scene?.shape,regions:lastResponse?.view?.scene?.regions,physics:lastResponse?.view?.physics?.profile,initialStateId:lastResponse?.view?.stateId};
      report.enabledLayers=await js(`[...document.querySelectorAll('.rehearsal-anatomy input:checked')].map(e=>e.getAttribute('aria-label'))`);
      await setValue('select[aria-label="Instrument action"]','inspect');
      await setValue('input[aria-label="Action duration"]',0.1,'input');
      const history=Number(process.env.INTERACTION_HISTORY||12),repeats=Number(process.env.INTERACTION_REPEATS||3);
      await run('seed-history',async()=>{for(let i=0;i<history;i++){await click('Apply');await idle();}});
      const steps=lastResponse.position;report.historySteps=steps;
      await run('camera-warmup',gesture);
      const warmPreparationAt=performance.now();
      await wait(`!window.perfLatestFrame?.picking||(window.perfLatestFrame.picking.activeJobs===0&&window.perfLatestFrame.picking.pendingGeometryRefs===0)`,120000);
      report.additionalWarmPreparationMs=performance.now()-warmPreparationAt;
      for(let i=0;i<(process.env.INTERACTION_TISSUE_ONLY?0:repeats);i++){
        await run('warm-camera-'+i,gesture);
        await run('tool-pose-'+i,async()=>{const start=await js(`Number(document.querySelector('input[aria-label="Tool position R"]').value)`);for(let j=1;j<=30;j++){await setValue('input[aria-label="Tool position R"]',start+Math.sin(j/30*Math.PI*2),'input');await pause(16);}});
        for(const position of [0,Math.floor(steps/2),steps])await run('seek-'+position+'-'+i,()=>setValue('input[aria-label="Surgical timeline"]',position,'input'));
      }
      if(!process.env.INTERACTION_TISSUE_ONLY)await run('picking',async()=>{const p=await js(`(()=>{const r=document.querySelector('.rl-viewer-anatomy-image').getBoundingClientRect();return {x:Math.round(r.x+r.width*.5),y:Math.round(r.y+r.height*.5)};})()`);for(let i=0;i<12;i++){for(const clickCount of [1,2]){await sendPointer({type:'mouseDown',x:p.x+i,y:p.y,button:'left',clickCount});await sendPointer({type:'mouseUp',x:p.x+i,y:p.y,button:'left',clickCount});}await pause(30);}});
      await run('preview',()=>click('Preview'));await click('Cancel');
      await run('complete-inspect-action',()=>click('Apply'));
      await run('branch',()=>click('Branch'));
      if(process.env.INTERACTION_TISSUE_RECIPE){
        const recipeReport=JSON.parse(fs.readFileSync(process.env.INTERACTION_TISSUE_RECIPE,'utf8'));
        const recipe=recipeReport.cases.find(c=>c.source?.sha256===report.source.sha256);
        if(!recipe)throw Error('Tissue recipe must match exact existing source bytes');
        const {Euler,Quaternion}=require('three');
        const parameterLabels={radiusMm:'Tip radius · mm',rpm:'Drill speed · rpm',forceN:'Contact force · N',duty:'Drill duty fraction'};
        const configure=async action=>{
          await setValue('select[aria-label="Instrument action"]',action.operation);
          if(action.pose){for(let axis=0;axis<3;axis++)await setValue('input[aria-label="Tool position '+['R','A','S'][axis]+'"]',action.pose.position[axis],'input');
            const angles=new Euler().setFromQuaternion(new Quaternion(...action.pose.quaternion));
            for(const [axis,value]of [['pitch',angles.x],['yaw',angles.y],['roll',angles.z]])await setValue('input[aria-label="Tool '+axis+'"]',value*180/Math.PI,'input');}
          for(const [key,value]of Object.entries(action.parameters??{})){if(!parameterLabels[key])throw Error('Unsupported benchmark parameter '+key);await setValue('input[aria-label="'+parameterLabels[key]+'"]',value,'input');}
          await setValue('input[aria-label="Action duration"]',action.duration,'input');
        };
        report.tissueRecipe={source:recipe.source,setup:recipe.setupActions,
          action:{...recipe.measuredAction,duration:1,parameters:{radiusMm:1,rpm:40000,forceN:5,duty:1}},
          kind:'Generated engineering action on acquired source anatomy; no treatment claim'};
        for(const action of recipe.setupActions){await configure(action);await run('tissue-place',()=>click('Apply'));}
        const before=lastResponse.view.stateId;
        await configure(report.tissueRecipe.action);
        if(process.env.INTERACTION_MODEL_WORKER_FAILURE){
          const maximum=lastResponse.position,authoritative=lastResponse.view.stateId;
          await js('window.perfFailNextModel=true');
          await run('verify-pending-model-worker-failure',()=>click('Preview'));
          report.modelWorkerFault=await js('window.perfModeledFault??null');
          report.checks.modelWorkerFailureRetainsAcceptedDisplay=report.modelWorkerFault?.pendingJobs>0&&await js(`document.querySelector('.surgical-panel').dataset.stateId===${JSON.stringify(authoritative)}&&window.perfLatestFrame.surgicalStateId===${JSON.stringify(authoritative)}&&!!document.querySelector('.rl-viewer-error')`);
          await js(`window.resectionApi.surgicalRehearsal({caseHash:${JSON.stringify(report.scene.caseHash)},command:'inspect'})`);
          report.checks.modelWorkerFailurePreservesAuthority=lastResponse.view.stateId===authoritative;
          await run('verify-model-worker-recovery-seek-zero',()=>setValue('input[aria-label="Surgical timeline"]',0,'input'));
          report.checks.modelWorkerSeekClearsError=await js(`!document.querySelector('.rl-viewer-error')&&!document.querySelector('.surgical-error')&&document.querySelector('.surgical-panel').dataset.stateId===window.perfLatestFrame.surgicalStateId&&[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Apply'&&!b.disabled)`);
          await run('verify-model-worker-recovery-seek-back',()=>setValue('input[aria-label="Surgical timeline"]',maximum,'input'));
          report.checks.modelWorkerRecoveryRestoresState=lastResponse.view.stateId===authoritative;
          await configure(report.tissueRecipe.action);
        }
        if(process.env.INTERACTION_CANCEL_MODEL_PREVIEW){
          const authoritative=lastResponse.view.stateId;
          await click('Preview');
          await wait(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Cancel preview'&&!b.disabled)&&[...window.perfWorkers].some(w=>w.perfKind.includes('/incrementalSurface.worker')&&w.perfSurfaceJobs>0)`);
          const started=performance.now();await click('Cancel preview');await idle();
          report.cancelPendingPreview={acknowledgedMs:performance.now()-started,retainedStateId:await js('document.querySelector(".surgical-panel").dataset.stateId')};
          await js(`window.resectionApi.surgicalRehearsal({caseHash:${JSON.stringify(report.scene.caseHash)},command:'inspect'})`);
          report.checks.cancelPendingPreviewPreservesAuthority=lastResponse.view.stateId===authoritative&&report.cancelPendingPreview.retainedStateId===authoritative;
          report.checks.cancelPendingPreviewReleasesJob=await js(`![...window.perfWorkers].some(w=>w.perfKind.includes('/incrementalSurface.worker')&&w.perfSurfaceJobs>0)&&window.perfLatestFrame.surgicalStateId===${JSON.stringify(authoritative)}`);
          await configure(report.tissueRecipe.action);
        }
        await run('tissue-preview',()=>click('Preview'));
        const previewState=lastResponse.view.stateId;
        await run('tissue-cancel-preview',()=>click('Cancel'));
        await configure(report.tissueRecipe.action);
        await run('tissue-apply',()=>click('Apply'));
        report.checks.tissueActuallyChanged=lastResponse.view.removed.length>0;
        report.checks.tissuePreviewMatchesCommit=lastResponse.view.stateId===previewState;
        const committed=lastResponse.view.stateId;
        for(let i=0;i<3;i++){await run('tissue-undo-'+i,()=>click('Undo'));report.checks['tissueUndo'+i]=lastResponse.view.stateId===before;
          await run('tissue-redo-'+i,()=>click('Redo'));report.checks['tissueRedo'+i]=lastResponse.view.stateId===committed;}
        await configure({...report.tissueRecipe.action,duration:.3});await run('tissue-next-local-action',()=>click('Apply'));
        report.tissueFinal={stateId:lastResponse.view.stateId,removedCells:lastResponse.view.removed.length,metrics:lastResponse.view.metrics};
        fs.writeFileSync(path.join(output,'tissue-settled.png'),(await window.webContents.capturePage()).toPNG());
      }
      await run('observer-control',gesture,true);
      const savedState=lastResponse?.view?.stateId;
      await run('save',async()=>{await click('Save');const started=performance.now();while(!fs.existsSync(path.join(output,'saved-rehearsal.ressectionlab'))){if(performance.now()-started>120000)throw Error('Save did not produce the disposable bundle');await pause(20);}});
      report.exportBytes=fs.existsSync(path.join(output,'saved-rehearsal.ressectionlab'))?fs.statSync(path.join(output,'saved-rehearsal.ressectionlab')).size:null;
      if(report.exportBytes){nextOpenPath=path.join(output,'saved-rehearsal.ressectionlab');await run('reopen',async()=>{const index=report.ipc.length;await click('Open');const started=performance.now();while(!report.ipc.slice(index).some(i=>i.channel==='research:openWorkspace')){if(performance.now()-started>120000)throw Error('Reopen did not complete');await pause(20);}await pause(300);await ready();await click('Rehearse');await wait(`!!document.querySelector('.surgical-slots')`);await js(`window.resectionApi.surgicalRehearsal({caseHash:${JSON.stringify(report.scene.caseHash)},command:'inspect'})`);});nextOpenPath=null;report.checks.saveReopenState=lastResponse?.view?.stateId===savedState;}
      if(process.env.INTERACTION_VERIFY){
        const max=lastResponse.position;
        await run('verify-rapid-seek',async()=>{for(const value of [0,max,1,Math.floor(max/2),max]){await setValue('input[aria-label="Surgical timeline"]',value,'input');await pause(2);}});
        report.checks.rapidSeekLatest=lastResponse.position===max;
        const authoritative=lastResponse.view.stateId;
        await click('Play replay');await wait(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Pause replay')`);
        const pauseAt=performance.now();await click('Pause replay');
        report.pauseAcknowledgedMs=performance.now()-pauseAt;report.checks.pauseAcknowledged=report.pauseAcknowledgedMs<100;
        await idle();await pause(250);
        report.checks.pauseRemainsPaused=await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Play replay'&&!b.getAttribute('aria-pressed')?.includes('true'))`);
        await setValue('input[aria-label="Surgical timeline"]',max,'input');await idle();
        report.checks.playbackDoesNotSimulate=lastResponse.view.stateId===authoritative;
        const activeBranch=lastResponse.branchId,activeState=lastResponse.view.stateId;
        const otherBranch=lastResponse.branches.find(b=>b.id!==activeBranch);
        if(otherBranch){
          await run('verify-branch-switch-away',()=>setValue('select[aria-label="Surgical branch"]',otherBranch.id));
          report.checks.branchSwitchSelectsRequested=lastResponse.branchId===otherBranch.id;
          await run('verify-branch-switch-back',()=>setValue('select[aria-label="Surgical branch"]',activeBranch));
          report.checks.branchSwitchRestoresCompleteState=lastResponse.branchId===activeBranch&&lastResponse.view.stateId===activeState;
        }
        if(process.env.INTERACTION_RESOURCE_CYCLES){
          report.resourceCycles=[];
          for(let cycle=0;cycle<Number(process.env.INTERACTION_RESOURCE_CYCLES);cycle++){
            await run('verify-resource-cycle-'+cycle,async()=>{for(const value of [0,Math.floor(max/2),max]){await setValue('input[aria-label="Surgical timeline"]',value,'input');await idle();}});
            report.resourceCycles.push(await js(`({resources:window.perfLatestFrame?.resources,allocation:window.perfLatestFrame?.estimatedAllocation,picking:window.perfLatestFrame?.picking,workers:window.perfWorkerCounts.live})`));
          }
          const stable=report.resourceCycles.slice(1),first=stable[0];
          report.checks.resourceCountsPlateau=stable.every(s=>s.resources.geometries===first.resources.geometries&&s.resources.textures===first.resources.textures&&s.workers===first.workers);
          report.checks.pickingWithinByteBudget=stable.every(s=>!s.picking||s.picking.retainedRootBytes<=s.picking.maxRetainedBytes&&s.picking.activeBuildInputBytes<=s.picking.maxInputBytes);
        }
        if(process.env.INTERACTION_PICK_WORKER_FAILURE){
          const before=lastResponse.view.stateId;
          report.pickWorkerFaultInjected=await js(`(()=>{const worker=[...window.perfWorkers].find(w=>w.perfKind.includes('sourcePickingBvh.worker'));if(!worker)return false;worker.dispatchEvent(new ErrorEvent('error',{message:'Injected picking worker failure'}));return true;})()`);
          await run('verify-picking-worker-failure',gesture);
          report.checks.pickingWorkerFailurePreservesState=report.pickWorkerFaultInjected&&lastResponse.view.stateId===before&&await js(`window.perfLatestFrame.picking.workerFailures>0&&document.querySelector('.surgical-panel').dataset.stateId===window.perfLatestFrame.surgicalStateId`);
        }
        if(process.env.INTERACTION_CANCEL_ACTION){
          const before=lastResponse.view.stateId,index=report.ipc.length;
          await setValue('select[aria-label="Instrument action"]','apply');
          await setValue('input[aria-label="Action duration"]',30,'input');
          await click('Apply');await wait(`!!document.querySelector('.cancel-operation')`,10000);
          const cancelAt=performance.now();await js(`document.querySelector('.cancel-operation').click()`);
          await idle();report.cancelAcknowledgedMs=performance.now()-cancelAt;
          await js(`window.resectionApi.surgicalRehearsal({caseHash:${JSON.stringify(report.scene.caseHash)},command:'inspect'})`);
          report.cancelledRequest=report.ipc.slice(index).find(i=>i.channel==='research:surgicalRehearsal'&&/Operation cancelled/.test(i.error??''));
          report.checks.cancelPreservesAuthoritativeState=lastResponse.view.stateId===before&&!!report.cancelledRequest;
          await setValue('input[aria-label="Action duration"]',.1,'input');
        }
        if(process.env.INTERACTION_CONTEXT_LOSS){
          const before=lastResponse.view.stateId;
          if(process.env.INTERACTION_MISSING_GPU_TIMER)await js(`(()=>{const original=WebGL2RenderingContext.prototype.getExtension;WebGL2RenderingContext.prototype.getExtension=function(name){return name==='EXT_disjoint_timer_query_webgl2'?null:original.call(this,name);};})()`);
          report.contextLossSupported=await js(`(()=>{const canvas=document.querySelector('.rl-viewer canvas'),gl=canvas.getContext('webgl2'),ext=gl?.getExtension('WEBGL_lose_context');if(!ext)return false;ext.loseContext();return true;})()`);
          if(report.contextLossSupported){await wait(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Retry graphics')`);await pause(150);await click('Retry graphics');await wait(`!document.querySelector('.rl-viewer-error')`);await ready();report.checks.graphicsRecoveryPreservesState=lastResponse.view.stateId===before;}
          if(process.env.INTERACTION_MISSING_GPU_TIMER){
            await run('verify-missing-gpu-timer',gesture);
            report.checks.missingGpuTimerFallbackPreservesDisplay=await js(`window.perfLatestFrame.gpuTimingSupported===false&&document.querySelector('.surgical-panel').dataset.stateId===window.perfLatestFrame.surgicalStateId`);
          }
        }
        if(process.env.INTERACTION_CASE_SWITCH_PATH){
          const previous=lastResponse.caseHash,other=path.resolve(process.env.INTERACTION_CASE_SWITCH_PATH);
          const hash=crypto.createHash('sha256').update(fs.readFileSync(other)).digest('hex');
          nextOpenPath=other;
          await run('verify-case-switch',async()=>{
            const index=report.ipc.length;await click('Open');
            const started=performance.now();while(!report.ipc.slice(index).some(i=>i.channel==='research:openWorkspace')){if(performance.now()-started>120000)throw Error('Case switch did not complete');await pause(20);}
            await pause(300);await ready();await click('Rehearse');await wait(`!!document.querySelector('.surgical-slots')`);await idle();
          });nextOpenPath=null;
          const identity=await js(`({panel:document.querySelector('.surgical-panel')?.dataset.stateId,frame:window.perfLatestFrame?.surgicalStateId,source:window.perfLatestFrame?.sourceCaseHash})`);
          report.caseSwitch={sourcePath:other,sha256:hash,caseHash:lastResponse.caseHash,identity};
          report.checks.caseSwitchReplacesAcceptedState=lastResponse.caseHash!==previous&&identity.panel===lastResponse.view.stateId&&identity.frame===identity.panel&&identity.source===lastResponse.caseHash;
          report.checks.caseSwitchSourcePreserved=crypto.createHash('sha256').update(fs.readFileSync(other)).digest('hex')===hash;
        }
      }
      report.checks.foreground=await js(`document.visibilityState==='visible'`);report.windowFocusedAtCompletion=window.isFocused();report.checks.historyCreated=steps>=history;
      // Reopen and the verify-* scenarios are functional validation after all
      // primary interactions, and retain their own raw focus/timing records.
      const primaryRuns=report.runs.filter(r=>!r.subscribed&&r.name!=='reopen'&&!r.name.startsWith('verify-'));
      report.primaryTimingForeground=primaryRuns.every(r=>r.windowFocused&&r.windowFocusedAtEnd!==false&&!r.focusEvents.some(event=>event.event==='blur'));
      report.primaryTimingVisible=primaryRuns.every(r=>r.windowVisibleOnTarget&&r.windowVisibleOnTargetAtEnd);
      report.validationTimingForeground=report.runs.filter(r=>r.name==='reopen'||r.name.startsWith('verify-')).every(r=>r.windowFocused&&r.windowFocusedAtEnd!==false);
      if(secondaryOnly)report.checks.primaryTimingVisible=report.primaryTimingVisible;
      else report.checks.primaryTimingForeground=report.primaryTimingForeground;
      report.checks.pointerInputDelivered=report.runs.every(r=>!r.name.includes('camera')&&r.name!=='observer-control'||r.pointerEvents.downs>0&&r.pointerEvents.moves>=30&&r.pointerEvents.trusted>0);
      if(!process.env.INTERACTION_TISSUE_ONLY)report.checks.doubleClickInputDelivered=report.runs.find(r=>r.name==='picking')?.pointerEvents.doubleClicks>=12;
      report.checks.sourcePreserved=crypto.createHash('sha256').update(fs.readFileSync(sourcePath)).digest('hex')===report.source.sha256;
      report.displayConsistencySupported=!process.env.INTERACTION_PYTHON_SOURCE;
      if(report.displayConsistencySupported)report.checks.submittedDisplayConsistent=report.runs.every(r=>r.displayConsistent===true);
      report.passed=report.errors.length===0&&Object.values(report.checks).every(Boolean);
      fs.writeFileSync(path.join(output,'settled.png'),(await window.webContents.capturePage()).toPNG());
    } catch(error) {
      report.passed=false;report.failure=error.stack;console.error(error.stack);
      if(!window.isDestroyed())try{fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage()).toPNG());}catch(captureError){report.failureCapture=String(captureError);}
    }
    finally {persist();console.log(JSON.stringify({output,passed:report.passed,failure:report.failure}));app.exit(report.passed?0:1);}
  });
});
app.on('before-quit',()=>{report.lifecycle.push({at:performance.now(),event:'before-quit'});persist();});
app.on('will-quit',()=>{report.lifecycle.push({at:performance.now(),event:'will-quit'});if(typeof report.passed!=='boolean'){report.passed=false;report.failure='Application quit before the benchmark completed';}persist();fs.rmSync(userData,{recursive:true,force:true});});
require('./main.cjs');
