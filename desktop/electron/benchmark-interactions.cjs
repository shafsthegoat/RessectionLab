'use strict';
// Production Electron/sidecar benchmark. No corrections, saves or source writes.
const {app, ipcMain} = require('electron');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const output = path.resolve(process.env.INTERACTION_OUTPUT || '../outputs/interaction-performance/current');
fs.mkdirSync(output, {recursive:true});
app.setPath('userData', fs.mkdtempSync(path.join(os.tmpdir(), 'rslab-performance-')));
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const report = {commit:process.env.INTERACTION_COMMIT||execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),
  recordedAt:new Date().toISOString(), platform:process.platform, arch:process.arch,
  versions:process.versions, window:[1600,1000], runs:[], errors:[], transfers:[], checks:{},
  measurement:'Frame subscription measures Electron frame delivery with observer overhead. Submission latency is not display latency. Raw gaps are retained; warm-up is separate.'};
const originalHandle = ipcMain.handle.bind(ipcMain);
let failNextView=false;
ipcMain.handle = (channel, handler) => originalHandle(channel, async (...args) => {
  if(channel==='research:anatomyView'&&failNextView){failNextView=false;throw Error('Injected incremental transfer failure');}
  const start=performance.now();
  const result=await handler(...args);
  if(channel==='research:readAsset')report.transfers.push({at:start,ms:performance.now()-start,bytes:result?.byteLength??0});
  return result;
});
let finished=false;
app.on('browser-window-created', (_event, window) => {
  window.webContents.on('console-message',event=>{if(event.level==='error')report.errors.push(event.message)});
  window.webContents.once('did-finish-load', async () => {
    const js = code => window.webContents.executeJavaScript(code,true);
    const waitFor = async (expr, timeout=240000) => {
      const start=performance.now();
      while(performance.now()-start<timeout){if(await js(expr))return;await pause(100)}
      throw Error('Timed out: '+expr);
    };
    const ready = () => waitFor(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')&&!document.querySelector('[data-layer-pending="true"]')&&!document.querySelector('.rl-viewer')?.dataset.layersPreparing`);
    const click = label => js(`(()=>{const e=[...document.querySelectorAll('button')].find(e=>e.textContent.trim()===${JSON.stringify(label)});if(!e)throw Error('Missing button '+${JSON.stringify(label)});e.click()})()`);
    const checkbox = label => js(`(()=>{const e=document.querySelector('input[aria-label='+${JSON.stringify(JSON.stringify(label))}+']');if(!e||e.disabled)throw Error('Missing/disabled checkbox '+${JSON.stringify(label)});e.click()})()`);
    const capture = async name => fs.writeFileSync(path.join(output,name+'.png'),(await window.webContents.capturePage()).toPNG());
    const point = selector => js(`(()=>{const r=document.querySelector(${JSON.stringify(selector)}).getBoundingClientRect();return {x:Math.round(r.x+r.width/2),y:Math.round(r.y+r.height/2)}})()`);
    const input = event => {report.runs.at(-1)?.inputs.push({at:performance.now(),...event});window.webContents.sendInputEvent(event)};
    const gesture = async kind => {
      const p=await point(kind==='slices'?'.rl-viewer-slice-image':kind==='sidebar'?'.unified-sidebar':'.rl-viewer-anatomy-image');
      if(kind==='rotate'||kind==='pan'){
        const button=kind==='pan'?'right':'left',modifiers=[kind==='pan'?'rightButtonDown':'leftButtonDown'];
        input({type:'mouseDown',...p,button,clickCount:1});
        for(let i=1;i<=60;i++){input({type:'mouseMove',x:p.x+Math.round(60*Math.sin(i/60*Math.PI*2)),y:p.y+Math.round(20*Math.sin(i/30*Math.PI*2)),modifiers});await pause(16)}
        input({type:'mouseUp',...p,button,clickCount:1});
      }else{
        input({type:'mouseMove',...p});await pause(100);
        for(let i=0;i<40;i++){input({type:'mouseWheel',...p,deltaY:i<20?-5:5,deltaX:0,canScroll:true});await pause(16)}
      }
    };
    const settle=()=>waitFor(`performance.now()-(window.perfFrames?.at(-1)?.at??0)>400`,30000);
    const reset=async()=>{await js(`window.perfFrames=[];window.perfGpu=[];window.perfRefined=[];window.perfInputs=[];window.perfLong=[];window.perfLifecycle=[];`)};
    const run=async(name, action, subscribed=true)=>{
      await reset();
      const r={name,subscribed,inputs:[],presented:[],transfersBefore:report.transfers.length,start:performance.now()};report.runs.push(r);
      if(subscribed)window.webContents.beginFrameSubscription(true,()=>r.presented.push(performance.now()));
      await action();r.inputEnded=performance.now();await pause(650);await settle();
      if(subscribed)window.webContents.endFrameSubscription();
      Object.assign(r,await js(`({frames:window.perfFrames,gpu:window.perfGpu,refined:window.perfRefined,domInputs:window.perfInputs,longTasks:window.perfLong,lifecycle:window.perfLifecycle,heap:performance.memory?.usedJSHeapSize})`));
      r.end=performance.now();r.transfers=report.transfers.slice(r.transfersBefore);
      console.log(name,JSON.stringify({frames:r.frames.length,presented:r.presented.length,transfers:r.transfers.length}));
      fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));
    };
    try{
      window.webContents.setBackgroundThrottling(false);
      window.setSize(...report.window);window.show();window.focus();await ready();await pause(1500);
      report.source=JSON.parse(fs.readFileSync(process.argv[process.argv.indexOf('--anatomy')+1],'utf8'));
      report.source={patient:report.source.patient_id,snapshot:report.source.snapshot_id,series:report.source.reference_series};
      await js(`window.perfFrames=[];window.perfGpu=[];window.perfRefined=[];window.perfInputs=[];window.perfLong=[];window.perfLifecycle=[];
        document.addEventListener('anatomy-frame',e=>window.perfFrames.push({...e.detail,inputAt:window.perfInputs.at(-1)?.at}));
        document.addEventListener('anatomy-refined',e=>window.perfRefined.push(e.detail));
        document.addEventListener('anatomy-gpu',e=>window.perfGpu.push({...e.detail,at:performance.now()}));
        for(const type of ['wheel','pointermove','pointerup','change'])document.addEventListener(type,e=>window.perfInputs.push({at:performance.now(),timeStamp:e.timeStamp,type,target:e.target.getAttribute?.('aria-label')}),true);
        new PerformanceObserver(list=>window.perfLong.push(...list.getEntries().map(e=>({at:e.startTime,ms:e.duration})))).observe({type:'longtask',buffered:false});
        let oldCanvas=document.querySelector('.rl-viewer canvas'),oldLoading=false;
        new MutationObserver(()=>{const canvas=document.querySelector('.rl-viewer canvas'),loading=!!document.querySelector('.viewer-loading');if(canvas!==oldCanvas||loading!==oldLoading)window.perfLifecycle.push({at:performance.now(),canvasChanged:canvas!==oldCanvas,loading});oldCanvas=canvas;oldLoading=loading;}).observe(document.body,{subtree:true,childList:true});`);
      await capture('initial');
      if(process.env.INTERACTION_VERIFY_ONLY){
        await js(`window.retainedCanvas=document.querySelector('.rl-viewer canvas')`);
        const region=await js(`document.querySelectorAll('.layer-group')[1].querySelector('.compact-layer input').getAttribute('aria-label')`);
        failNextView=true;await checkbox(region);await waitFor(`!!document.querySelector('[data-layer-error="true"]')`);
        report.checks.failureRetainsCanvas=await js(`window.retainedCanvas===document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`);
        await click('Retry');await ready();
        report.checks.retryRetainsCanvas=await js(`window.retainedCanvas===document.querySelector('.rl-viewer canvas')&&!document.querySelector('[data-layer-error="true"]')`);
        await run('rapid-loading-navigation',async()=>{
          await checkbox('Show all brain regions');await pause(50);await checkbox('Show all brain regions');await checkbox(region);
          await gesture('rotate');await ready();
        });
        report.checks.latestSelection=await js(`document.querySelectorAll('.layer-group')[1].querySelectorAll('.compact-layer input:checked').length===1`);
        report.checks.incrementalRetainsCanvas=await js(`window.retainedCanvas===document.querySelector('.rl-viewer canvas')`);
        await run('resident-resource-cycles',async()=>{for(let i=0;i<20;i++){await checkbox('Show vessels');await pause(50)}});
        const warm=report.runs.at(-1);report.checks.warmTransfers=warm.transfers.length===0;
        const resources=warm.frames.map(f=>JSON.stringify(f.resources));report.checks.stableResources=new Set(resources).size===1;
        await checkbox('Show all review layers');
        await js(`(()=>{const e=document.querySelector('select[aria-label="Source scan"]');e.value='ct-bone';e.dispatchEvent(new Event('change',{bubbles:true}))})()`);await ready();
        report.checks.pendingSourceSwitch=await js(`document.querySelector('.workspace-heading h2').textContent==='CT-BONE'&&!document.querySelector('[data-layer-error="true"]')`);
        await capture('switched-during-load');
        for(let i=0;i<2;i++)for(const series of ['t1ce','ct-bone']){
          await js(`(()=>{const e=document.querySelector('select[aria-label="Source scan"]');e.value=${JSON.stringify(series)};e.dispatchEvent(new Event('change',{bubbles:true}))})()`);await ready();await pause(200);
        }
        report.passed=Object.values(report.checks).every(Boolean)&&report.errors.length===0;return;
      }
      if(process.env.INTERACTION_RECORD_ONLY){
        const frames=[];let last=0;
        const directory=path.join(output,'recording');fs.mkdirSync(directory,{recursive:true});
        window.webContents.beginFrameSubscription(true,image=>{
          const at=performance.now();if(at-last<40)return;last=at;
          const file=`frame-${String(frames.length).padStart(5,'0')}.jpg`;
          fs.writeFileSync(path.join(directory,file),image.toJPEG(80));frames.push({file,at});
        });
        try{
          await pause(500);await gesture('rotate');await pause(700);await gesture('zoom');await pause(650);await settle();
          await gesture('slices');await pause(650);await settle();await checkbox('Show all brain regions');await ready();await pause(1000);
        }finally{window.webContents.endFrameSubscription();fs.writeFileSync(path.join(directory,'frames.json'),JSON.stringify(frames,null,2));}
        report.passed=true;report.recording='Separate visual demonstration; capture overhead excludes this run from timing comparisons';return;
      }
      await run('warmup',()=>gesture('rotate'));
      const repeats=Number(process.env.INTERACTION_REPEATS||5);
      for(let repeat=0;repeat<repeats;repeat++)for(const kind of ['rotate','pan','zoom','slices','sidebar'])await run(kind+'-'+repeat,()=>gesture(kind));
      await run('observer-control',()=>gesture('rotate'),false);
      for(let repeat=0;repeat<repeats;repeat++)await run('resident-toggle-'+repeat,async()=>{await checkbox('Show vessels');await pause(200);await checkbox('Show vessels')});
      await run('first-layer',async()=>{await checkbox('Show all brain regions');
        report.checks.navigationAvailableDuringLoad=await js(`!!document.querySelector('.rl-viewer-anatomy-image')&&!document.querySelector('.viewer-loading')`);
        if(report.checks.navigationAvailableDuringLoad)await gesture('rotate');
        await ready()},true);
      report.checks.firstLayerErrors=await js(`document.querySelectorAll('[data-layer-error="true"]').length`);
      await checkbox('Show all structures');await ready();
      await checkbox('Show all review layers');await ready();
      report.checks.denseLayers=await js(`document.querySelectorAll('.compact-layer input:checked').length`);
      await capture('all-regions');
      for(let repeat=0;repeat<repeats;repeat++)await run('dense-rotate-'+repeat,()=>gesture('rotate'));
      await checkbox('Show all brain regions');await ready();
      await checkbox('Show all review layers');await checkbox('Show bone');await ready();
      await click('vessels');
      await run('opacity',async()=>{for(const value of ['0.3','0.6','1']){await js(`(()=>{const e=document.querySelector('input[aria-label="vessels opacity"]');Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(e,${JSON.stringify(value)});e.dispatchEvent(new Event('input',{bubbles:true}))})()`);await pause(250)}});
      await click('Appearance');await checkbox('Scan volume');await js(`document.querySelector('button[aria-label="Close view controls"]').click()`);
      await run('volume-off',()=>gesture('rotate'));await capture('volume-off');
      await click('Appearance');await checkbox('Scan volume');await js(`document.querySelector('button[aria-label="Close view controls"]').click()`);
      await click('Layout');await click('Four views');await run('four-views',()=>gesture('slices'));await capture('four-views');
      await click('Layout');await click('3D + slices');
      await js(`(()=>{const e=document.querySelector('select[aria-label="Source scan"]');e.value='ct-bone';e.dispatchEvent(new Event('change',{bubbles:true}))})()`);await ready();await pause(500);
      await run('ct-zoom',()=>gesture('zoom'));await capture('ct');
      window.setSize(1050,720);await pause(300);await run('minimum-window',()=>gesture('rotate'));
      report.passed=report.errors.length===0&&!report.checks.firstLayerErrors;
    }catch(error){report.passed=false;report.failure=error.stack;await capture('failure').catch(()=>{})}
    finally{finished=true;fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,failure:report.failure,output}));app.quit()}
  });
});
process.on('uncaughtException',error=>{report.failure=error.stack;if(!finished)fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.error(error);app.exit(1)});
require('./main.cjs');
