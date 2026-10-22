'use strict';
// Bounded, read-only playback measurement of the actual Electron application.
// The Python monitoring guard aborts any public physics step before it executes.
const {app,ipcMain,screen}=require('electron');
const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');
const childProcess=require('node:child_process');
const {submittedPlayback}=require('./playback-submissions.cjs');
const root=path.resolve(__dirname,'../..');
const output=path.resolve(process.env.PLAYBACK_OUTPUT||path.join(root,'testcases/outputs/rl-readiness-20261022/playback'));
if(!output.startsWith(path.join(root,'testcases')+path.sep))throw Error('Keep benchmark evidence under testcases/');
fs.mkdirSync(output,{recursive:true});
const source=path.resolve(process.env.PLAYBACK_PYTHON_SOURCE||path.join(root,'src'));
const bridgeSource=process.env.PLAYBACK_BRIDGE_SOURCE?path.resolve(process.env.PLAYBACK_BRIDGE_SOURCE):null;
const sourcePath=process.argv[process.argv.indexOf('--case')+1];
if(!process.argv.includes('--case')||!sourcePath)throw Error('Supply an existing saved --case');
const monitor=path.join(root,'testcases/outputs/full-case-rehearsal-20261020/review');
const runner=path.join(output,'guarded-bridge.py');
const bridgeSetup=bridgeSource?`import importlib.util\nimport resectionlab.surgical_bridge as original_bridge\nspec=importlib.util.spec_from_file_location('resectionlab._measured_transport_bridge', ${JSON.stringify(bridgeSource)})\nmeasured_bridge=importlib.util.module_from_spec(spec)\nspec.loader.exec_module(measured_bridge)\noriginal_bridge.dispatch=measured_bridge.dispatch\n`:'';
fs.writeFileSync(runner,`import json, sys, importlib.util\nfrom pathlib import Path\nsys.path.insert(0, ${JSON.stringify(monitor)})\nfrom readonly_step_monitor import ReadOnlyStepMonitor\nfrom resectionlab.surgical_engine import SurgicalEngine\nfrom resectionlab import surgical_frame_replay, surgical_telemetry\nif importlib.util.find_spec('resectionlab.surgical_archive_view') is not None:\n    from resectionlab import surgical_archive_view\nfrom resectionlab.desktop_bridge import main\n${bridgeSetup}guard=ReadOnlyStepMonitor(SurgicalEngine)\ntry:\n    with guard:\n        main()\nfinally:\n    Path(${JSON.stringify(path.join(output,'physics-guard.json'))}).write_text(json.dumps({'publicStepAttempts': dict(guard.attempts), 'nonStepMonitoringCallbacks': guard.other_callbacks}))\n`);
const originalSpawn=childProcess.spawn;
childProcess.spawn=(command,args,options)=>args?.[1]==='-m'&&args?.[2]==='resectionlab.desktop_bridge'
  ?originalSpawn(command,['-u',runner,...args.slice(3)],options):originalSpawn(command,args,options);
const sidecar=require('./sidecar.cjs'),OriginalSidecar=sidecar.Sidecar;
let backend;
sidecar.Sidecar=class extends OriginalSidecar{constructor(config){super({...config,sourcePath:source});backend=this;this.on('diagnostic',message=>report.diagnostics.push(message));}};
const userData=path.join(output,'user-data');fs.mkdirSync(userData,{recursive:true});app.setPath('userData',userData);
const hash=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const report={schema:'resectionlab.bounded-playback-performance.v1',recordedAt:new Date().toISOString(),source:{path:sourcePath,sha256:hash(sourcePath)},pythonSource:source,
  versions:process.versions,ipc:[],diagnostics:[],errors:[],checks:{},measurement:'Actual Electron controls, authorized IPC and Python sidecar. anatomy-frame measures CPU submission, not screen presentation. Prefetched IPC is recorded separately from submitted states. No patient physics steps permitted. No training.',distSha256:hash(path.join(root,'desktop/dist/index.html')),harnessSha256:hash(__filename)};
if(process.env.PLAYBACK_IMPORT_ONLY)report.importOnly=true;
if(process.env.PLAYBACK_FIELD_SIZES)report.fieldSizeDiagnostics={enabled:true,measurement:'Extra per-field JSON serialization runs after handler timing and before IPC delivery. This pass is diagnostic and is excluded from primary timing comparisons.'};
if(bridgeSource)report.transportExperiment={source:bridgeSource,sha256:hash(bridgeSource),scope:'Current transport bridge dispatch over authenticated frozen numerical implementation; does not establish current numerical cache admission. No physics classes/functions/source/cache identities are replaced.'};
let lastResponse,windowRef;
const persist=()=>fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));
const originalHandle=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(channel,handler)=>originalHandle(channel,async(...args)=>{const started=performance.now();try{const result=await handler(...args);const entry={channel,command:args[1]?.command,started,ms:performance.now()-started};if(channel==='research:surgicalRehearsal'){lastResponse=result;Object.assign(entry,{branchId:result?.playback?.branchId,position:result?.playback?.position,frameIndex:result?.playback?.frameIndex,simulatedSeconds:result?.playback?.observedTimeSeconds,stateId:result?.view?.stateId,responseBytes:Buffer.byteLength(JSON.stringify(result))});
      if(process.env.PLAYBACK_FIELD_SIZES){const sizes=value=>Object.fromEntries(Object.entries(value??{}).map(([key,item])=>[key,Buffer.byteLength(JSON.stringify(item)??'null')]));entry.responseKeyBytes=sizes(result);entry.viewKeyBytes=sizes(result?.view);entry.viewFieldsOmitted=result?.viewFieldsOmitted;entry.viewFieldTokens=result?.viewFieldTokens;}
    }report.ipc.push(entry);persist();return result;}catch(error){report.ipc.push({channel,command:args[1]?.command,started,ms:performance.now()-started,error:String(error)});throw error;}});
const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
app.on('browser-window-created',(_,window)=>{
  windowRef=window;window.webContents.setBackgroundThrottling(false);
  const secondary=screen.getAllDisplays().find(display=>display.id!==screen.getPrimaryDisplay().id);
  if(secondary){const a=secondary.workArea;window.setBounds({x:a.x+20,y:a.y+20,width:Math.min(1480,a.width-40),height:Math.min(900,a.height-40)});window.showInactive();}
  else {window.setSize(1480,900);window.show();}
  report.display={secondary:secondary??null,all:screen.getAllDisplays()};
  window.webContents.on('console-message',event=>{if(event.level==='error')report.errors.push(event.message);});
  window.webContents.once('did-finish-load',async()=>{
    const js=code=>window.webContents.executeJavaScript(code,true);
    const wait=async(expr,timeout=120000)=>{const began=performance.now();while(performance.now()-began<timeout){if(await js(expr))return;await pause(30);}throw Error('Timed out: '+expr);};
    const click=label=>js(`(()=>{const button=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(label)});if(!button||button.disabled)throw Error('Missing enabled button '+${JSON.stringify(label)});button.click();})()`);
    const set=(selector,value,type='change')=>js(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});if(!e||e.disabled)throw Error('Missing enabled control');Object.getOwnPropertyDescriptor(e.tagName==='SELECT'?HTMLSelectElement.prototype:HTMLInputElement.prototype,'value').set.call(e,${JSON.stringify(String(value))});e.dispatchEvent(new Event(${JSON.stringify(type)},{bubbles:true}));})()`);
    const idle=()=>wait(`!!document.querySelector('.surgical-slots')&&!document.querySelector('.surgical-panel > p[role="status"]')&&!document.querySelector('.rl-viewer-preparing')`);
    try{
      await wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')&&!document.querySelector('.rl-viewer-preparing')`);
      await js(`window.playbackFrames=[];window.playbackWork=[];window.playbackLong=[];document.addEventListener('anatomy-frame',e=>window.playbackFrames.push(e.detail));document.addEventListener('anatomy-work',e=>window.playbackWork.push({...e.detail,at:performance.now()}));new PerformanceObserver(list=>window.playbackLong.push(...list.getEntries().map(e=>({at:e.startTime,ms:e.duration})))).observe({type:'longtask',buffered:false});`);
      const enterAt=performance.now();await click('Rehearse');await idle();report.enterMs=performance.now()-enterAt;
      report.initial={branch:lastResponse.branchId,stateId:lastResponse.view.stateId,shape:lastResponse.view.scene.shape};
      if(process.env.PLAYBACK_IMPORT_ONLY){
        report.imports=[];
        for(const branch of ['full-primary','full-vascular']){
          const started=performance.now();if(lastResponse.branchId!==branch){await set('select[aria-label="Surgical branch"]',branch);await idle();}
          const ui=await js(`({label:document.querySelector('[aria-label="Unverified imported checkpoint"]')?.textContent,blocked:[...document.querySelectorAll('button')].filter(b=>['Play replay','Apply','Preview','Branch','Undo','Redo'].includes(b.textContent.trim())).every(b=>b.disabled||b.closest('fieldset')?.disabled),displayedStateId:document.querySelector('.surgical-panel').dataset.stateId,lastSubmittedStateId:window.playbackFrames.at(-1)?.surgicalStateId})`);
          report.imports.push({branch,acceptedMs:performance.now()-started,position:lastResponse.position,stateId:lastResponse.view.stateId,verification:lastResponse.verification,ui});
          report.checks[branch+'UnverifiedReadOnly']=lastResponse.verification?.readOnly===true&&lastResponse.verification?.canonical===false&&lastResponse.verification?.physicsActionsExecuted===0&&ui.label?.includes('unverified')&&ui.blocked;
          report.checks[branch+'DisplayMatches']=ui.displayedStateId===lastResponse.view.stateId&&ui.lastSubmittedStateId===lastResponse.view.stateId;
          await click('Frame operative field');await pause(100);fs.writeFileSync(path.join(output,branch+'.png'),(await window.webContents.capturePage()).toPNG());
        }
        report.initialAcceptedFromLaunchMs=report.ipc.find(row=>row.command==='inspectArchive')?.started+report.enterMs;
        report.checks.noReplayCommands=report.ipc.filter(row=>row.channel==='research:surgicalRehearsal').every(row=>row.command==='inspectArchive');
        report.checks.sourcePreserved=hash(sourcePath)===report.source.sha256;
      }else{
      const branch=process.env.PLAYBACK_BRANCH||'full-primary';
      if(lastResponse.branchId!==branch){await set('select[aria-label="Surgical branch"]',branch);await idle();}
      const position=Number(process.env.PLAYBACK_POSITION||2);
      await set('input[aria-label="Surgical timeline"]',position,'input');await idle();
      await set('select[aria-label="Replay speed"]',4);
      const maxFrames=Number(process.env.PLAYBACK_FRAMES||12),maxMs=Number(process.env.PLAYBACK_MAX_MS||45000);
      await js(`(()=>{window.playbackStop={branchId:${JSON.stringify(branch)},position:${position},frameIndex:${maxFrames-1},triggered:false};document.addEventListener('anatomy-frame',e=>{const target=window.playbackStop,frame=e.detail.surgicalFrame;if(!target.triggered&&frame?.branchId===target.branchId&&frame.position===target.position&&frame.frameIndex===target.frameIndex){target.triggered=true;target.cpuSubmittedAt=performance.now();const button=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Pause replay');if(!button||button.disabled)throw Error('Target frame reached without enabled pause control');button.click();}});})()`);
      const index=report.ipc.length;report.playStarted=performance.now();report.playStartedRenderer=await js('performance.now()');await click('Play replay');
      const submissionOptions={position,mainToRendererOffset:report.playStarted-report.playStartedRenderer,startedRenderer:report.playStartedRenderer};
      while(performance.now()-report.playStarted<maxMs){await pause(40);const frames=await js('window.playbackFrames.map(f=>({at:f.at,serial:f.serial,surgicalStateId:f.surgicalStateId,surgicalFrame:f.surgicalFrame}))');if(submittedPlayback(report.ipc.slice(index),frames,submissionOptions).length>=maxFrames)break;if(report.ipc.slice(index).some(row=>row.error))throw Error('Playback request failed');}
      if(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Pause replay')`))await click('Pause replay');
      await idle();report.playEnded=performance.now();report.playEndedRenderer=await js('performance.now()');report.automaticPause=await js('window.playbackStop');
      report.playbackIpc=report.ipc.slice(index);
      Object.assign(report,await js(`({frames:window.playbackFrames,work:window.playbackWork,longTasks:window.playbackLong,visible:document.visibilityState,displayedStateId:document.querySelector('.surgical-panel').dataset.stateId})`));
      report.submittedPlayback=submittedPlayback(report.playbackIpc,report.frames,submissionOptions);
      const submitted=report.submittedPlayback.at(-1);
      report.final={branch,position:submitted?.position,frameIndex:submitted?.frameIndex,simulatedSeconds:submitted?.simulatedSeconds,stateId:submitted?.stateId};
      report.fetchedLookahead=report.playbackIpc.filter(row=>row.command==='seekFrame'&&!report.submittedPlayback.some(shown=>shown.branchId===row.branchId&&shown.position===row.position&&shown.frameIndex===row.frameIndex&&shown.stateId===row.stateId));
      report.checks.displayMatches=report.displayedStateId===report.final.stateId;
      report.checks.submissionMatches=report.frames.at(-1)?.surgicalStateId===report.final.stateId;
      report.checks.exactFrames=report.submittedPlayback.every((row,i)=>row.position===position&&row.frameIndex===i);
      report.checks.paused=await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Play replay')`);
      report.checks.sourcePreserved=hash(sourcePath)===report.source.sha256;
      report.checks.framesObserved=report.submittedPlayback.length>=maxFrames;
      report.hiddenTimeline=await js(`(()=>{const details=[...document.querySelectorAll('details')].find(d=>d.querySelector(':scope > summary')?.textContent==='Action timeline');return {open:details.open,buttons:details.querySelectorAll('button').length};})()`);
      await js(`(()=>{const details=[...document.querySelectorAll('details')].find(d=>d.querySelector(':scope > summary')?.textContent==='Action timeline');details.open=true;})()`);
      await wait(`[...document.querySelectorAll('details')].find(d=>d.querySelector(':scope > summary')?.textContent==='Action timeline')?.querySelectorAll('button').length===${lastResponse.branches.find(b=>b.id===branch).steps}`);
      report.checks.expandedTimelineComplete=true;
      await js(`(()=>{const details=[...document.querySelectorAll('details')].find(d=>d.querySelector(':scope > summary')?.textContent==='Action timeline');details.open=false;})()`);
      await pause(40);
      if(process.env.PLAYBACK_ORBIT){
        // Separate observer control: frame delivery includes subscription cost.
        // No captured pixels are retained, and primary playback has no observer.
        window.webContents.debugger.attach('1.3');
        const pointer=event=>window.webContents.debugger.sendCommand('Input.dispatchMouseEvent',event);
        const p=await js(`(()=>{const r=document.querySelector('.rl-viewer-anatomy-image').getBoundingClientRect();return {x:r.x+r.width*.5,y:r.y+r.height*.5};})()`);
        report.camera={measurement:'Electron frame delivery during trusted CDP orbit; includes frame-subscription overhead',delivered:[],startedRenderer:await js('performance.now()')};
        window.webContents.beginFrameSubscription(true,()=>report.camera.delivered.push(performance.now()));
        await pointer({type:'mousePressed',...p,button:'left',buttons:1,clickCount:1});
        for(let i=1;i<=60;i++){await pointer({type:'mouseMoved',x:p.x+45*Math.sin(i/60*Math.PI*2),y:p.y+20*Math.sin(i/30*Math.PI*2),button:'none',buttons:1});await pause(16);}
        await pointer({type:'mouseReleased',...p,button:'left',buttons:0,clickCount:1});
        await pause(150);window.webContents.endFrameSubscription();report.camera.endedRenderer=await js('performance.now()');
        report.camera.frames=await js(`window.playbackFrames.filter(f=>f.at>=${report.camera.startedRenderer}&&f.at<=${report.camera.endedRenderer})`);
        window.webContents.debugger.detach();
      }
      }
      fs.writeFileSync(path.join(output,'settled.png'),(await window.webContents.capturePage()).toPNG());
      report.passed=report.errors.length===0&&Object.values(report.checks).every(Boolean);
    }catch(error){report.passed=false;report.failure=error.stack;try{fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage()).toPNG());}catch{}}
    finally{await backend?.stop();report.physicsGuard=fs.existsSync(path.join(output,'physics-guard.json'))?JSON.parse(fs.readFileSync(path.join(output,'physics-guard.json'))):null;report.checks.zeroPhysics=!!report.physicsGuard&&!Object.keys(report.physicsGuard.publicStepAttempts).length;report.passed=report.passed&&report.checks.zeroPhysics;persist();console.log(JSON.stringify({output,passed:report.passed,failure:report.failure,final:report.final}));app.exit(report.passed?0:1);}
  });
});
setTimeout(async()=>{report.failure='Overall bounded benchmark timeout';report.passed=false;persist();await backend?.stop();if(windowRef&&!windowRef.isDestroyed())try{fs.writeFileSync(path.join(output,'timeout.png'),(await windowRef.webContents.capturePage()).toPNG());}catch{}app.exit(1);},240000).unref();
require('./main.cjs');
