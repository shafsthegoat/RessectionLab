'use strict';
// Bounded, read-only playback measurement of the actual Electron application.
// Default guard forbids steps; explicit prefix mode permits three ordered preparations.
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
const prefixMode=process.env.PLAYBACK_PREFIX==='1';
const closeToolMode=process.env.PLAYBACK_CLOSE_TOOL==='1';
if(closeToolMode&&!prefixMode)throw Error('Close-tool check requires the bounded current-production prefix lane');
if(prefixMode&&(source!==path.join(root,'src')||bridgeSource||process.env.PLAYBACK_IMPORT_ONLY||process.env.PLAYBACK_ORBIT||process.env.PLAYBACK_FIELD_SIZES))throw Error('Prefix qualification requires current production source and no bridge substitution, import-only or observers');
const sourcePath=process.argv[process.argv.indexOf('--case')+1];
if(!process.argv.includes('--case')||!sourcePath)throw Error('Supply an existing saved --case');
const monitor=path.join(root,'testcases/outputs/full-case-rehearsal-20261020/review');
const runner=path.join(output,'guarded-bridge.py');
const bridgeSetup=bridgeSource?`import importlib.util\nimport resectionlab.surgical_bridge as original_bridge\nspec=importlib.util.spec_from_file_location('resectionlab._measured_transport_bridge', ${JSON.stringify(bridgeSource)})\nmeasured_bridge=importlib.util.module_from_spec(spec)\nspec.loader.exec_module(measured_bridge)\noriginal_bridge.dispatch=measured_bridge.dispatch\n`:'';
const prefixSetup=prefixMode?`sys.path.insert(0, ${JSON.stringify(__dirname)})\nfrom prefix_playback_monitor import PrefixPlaybackMonitor\nfrom resectionlab import surgical_prefix_replay\nfrom resectionlab.desktop_bridge import BridgeSession\nguard=PrefixPlaybackMonitor(SurgicalEngine)\nguard.install(BridgeSession)\n`:'guard=ReadOnlyStepMonitor(SurgicalEngine)\n';
const receipt=prefixMode?'guard.receipt()':"{'publicStepAttempts': dict(guard.attempts), 'nonStepMonitoringCallbacks': guard.other_callbacks}";
fs.writeFileSync(runner,`import json, sys, importlib.util\nfrom pathlib import Path\nsys.path.insert(0, ${JSON.stringify(monitor)})\nfrom readonly_step_monitor import ReadOnlyStepMonitor\nfrom resectionlab.surgical_engine import SurgicalEngine\nfrom resectionlab import surgical_frame_replay, surgical_telemetry\nif importlib.util.find_spec('resectionlab.surgical_archive_view') is not None:\n    from resectionlab import surgical_archive_view\nfrom resectionlab.desktop_bridge import main\n${bridgeSetup}${prefixSetup}try:\n    with guard:\n        main()\nfinally:\n    Path(${JSON.stringify(path.join(output,'physics-guard.json'))}).write_text(json.dumps(${receipt}))\n`);
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
function sourceFiles(directory){const files={};for(const entry of fs.readdirSync(directory,{withFileTypes:true})){const file=path.join(directory,entry.name);if(entry.isDirectory()&&entry.name!=='__pycache__')Object.assign(files,sourceFiles(file));else if(entry.isFile()&&entry.name.endsWith('.py'))files[path.relative(source,file)]=hash(file);}return files;}
if(prefixMode){report.measurement='Actual Electron controls, current production IPC/Python. CPU anatomy submission is not screen presentation. Three ordered physical prefix preparations permitted; no warm physics or training.';report.prefix={scope:'Original complete saved archive, current production code, bounded reconstruction of its first three actions only. Whole archive remains unverified; no editable admission. Three physical preparations permitted, warm observations must be step-free.',originalActions:null};report.pythonSourceFiles=sourceFiles(source);report.pythonSourceSha256=crypto.createHash('sha256').update(JSON.stringify(report.pythonSourceFiles)).digest('hex');report.guardSha256=hash(path.join(__dirname,'prefix_playback_monitor.py'));}
let lastResponse,windowRef;
const persist=()=>fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));
const originalHandle=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(channel,handler)=>originalHandle(channel,async(...args)=>{const started=performance.now();try{const result=await handler(...args);const entry={channel,command:args[1]?.command,started,ms:performance.now()-started};if(channel==='research:surgicalRehearsal'){lastResponse=result;Object.assign(entry,{requestedPosition:args[1]?.position,requestedFrameIndex:args[1]?.frameIndex,branchId:result?.playback?.branchId??result?.branchId,position:result?.playback?.position??result?.position,frameIndex:result?.playback?.frameIndex,simulatedSeconds:result?.playback?.observedTimeSeconds,stateId:result?.view?.stateId,responseBytes:Buffer.byteLength(JSON.stringify(result))});
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
      if(prefixMode){
        if(position!==2||Number(process.env.PLAYBACK_FRAMES||12)!==12)throw Error('Prefix qualification fixes action2 and twelve observations');
        report.prefix.originalActions=lastResponse.branches.find(b=>b.id===branch).steps;
        if(lastResponse.verification?.status!=='unverified-import')throw Error('Prefix must start from unverified imported endpoint');
        await js(`(()=>{window.prefixStartPause={triggered:false};const pause=()=>{const stop=window.prefixStartPause;if(stop.triggered)return;const button=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Pause replay');if(button&&!button.disabled){stop.triggered=true;stop.stateId=document.querySelector('.surgical-panel')?.dataset.stateId;button.click();observer.disconnect();}};const observer=new MutationObserver(pause);observer.observe(document.querySelector('.surgical-panel'),{subtree:true,childList:true,attributes:true,characterData:true});document.addEventListener('anatomy-frame',pause);})();`);
        await click('Replay from start · read only');await wait(`!!document.querySelector('[aria-label="Reconstructed prefix"]')&&window.prefixStartPause.triggered`);await idle();
        report.prefix.initialPause=await js('window.prefixStartPause');
        if(!report.prefix.initialPause.triggered||lastResponse.verification?.status!=='reconstructed-prefix'||lastResponse.position!==0)throw Error('Initial prefix0 must pause before any automatic frame request');
      }
      await set('input[aria-label="Surgical timeline"]',position,'input');if(prefixMode)await wait(`document.querySelector('.surgical-panel')?.dataset.recordedPosition==='2'`);await idle();
      await set('select[aria-label="Replay speed"]',4);
      const maxFrames=Number(process.env.PLAYBACK_FRAMES||12),maxMs=Number(process.env.PLAYBACK_MAX_MS||45000);
      await js(`(()=>{window.playbackStop={branchId:${JSON.stringify(branch)},position:${position},frameIndex:${maxFrames-1},triggered:false};document.addEventListener('anatomy-frame',e=>{const target=window.playbackStop,frame=e.detail.surgicalFrame;if(!target.triggered&&frame?.branchId===target.branchId&&frame.position===target.position&&frame.frameIndex===target.frameIndex){target.triggered=true;target.cpuSubmittedAt=performance.now();const button=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Pause replay');if(!button||button.disabled)throw Error('Target frame reached without enabled pause control');button.click();}});})()`);
      const index=report.ipc.length;report.playStarted=performance.now();report.playStartedRenderer=await js('performance.now()');await click('Play replay');
      const submissionOptions={position,command:prefixMode?'seekPrefixFrame':'seekFrame',mainToRendererOffset:report.playStarted-report.playStartedRenderer,startedRenderer:report.playStartedRenderer};
      while(performance.now()-report.playStarted<maxMs){await pause(40);const frames=await js('window.playbackFrames.map(f=>({at:f.at,serial:f.serial,surgicalStateId:f.surgicalStateId,surgicalFrame:f.surgicalFrame}))');if(submittedPlayback(report.ipc.slice(index),frames,submissionOptions).length>=maxFrames)break;if(report.ipc.slice(index).some(row=>row.error))throw Error('Playback request failed');}
      if(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Pause replay')`))await click('Pause replay');
      await idle();report.playEnded=performance.now();report.playEndedRenderer=await js('performance.now()');report.automaticPause=await js('window.playbackStop');
      report.playbackIpc=report.ipc.slice(index);
      Object.assign(report,await js(`({frames:window.playbackFrames,work:window.playbackWork,longTasks:window.playbackLong,visible:document.visibilityState,displayedStateId:document.querySelector('.surgical-panel').dataset.stateId})`));
      report.submittedPlayback=submittedPlayback(report.playbackIpc,report.frames,submissionOptions);
      const submitted=report.submittedPlayback.at(-1);
      report.final={branch,position:submitted?.position,frameIndex:submitted?.frameIndex,simulatedSeconds:submitted?.simulatedSeconds,stateId:submitted?.stateId};
      report.fetchedLookahead=report.playbackIpc.filter(row=>row.command===(prefixMode?'seekPrefixFrame':'seekFrame')&&!report.submittedPlayback.some(shown=>shown.branchId===row.branchId&&shown.position===row.position&&shown.frameIndex===row.frameIndex&&shown.stateId===row.stateId));
      report.checks.displayMatches=report.displayedStateId===report.final.stateId;
      report.checks.submissionMatches=report.frames.at(-1)?.surgicalStateId===report.final.stateId;
      report.checks.exactFrames=report.submittedPlayback.every((row,i)=>row.position===position&&row.frameIndex===i);
      report.checks.paused=await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Play replay')`);
      report.checks.sourcePreserved=hash(sourcePath)===report.source.sha256;
      report.checks.framesObserved=report.submittedPlayback.length>=maxFrames;
      report.hiddenTimeline=await js(`(()=>{const details=[...document.querySelectorAll('details')].find(d=>d.querySelector(':scope > summary')?.textContent==='Action timeline');return {open:details.open,buttons:details.querySelectorAll('button').length};})()`);
      await js(`(()=>{const details=[...document.querySelectorAll('details')].find(d=>d.querySelector(':scope > summary')?.textContent==='Action timeline');details.open=true;})()`);
      await wait(`[...document.querySelectorAll('details')].find(d=>d.querySelector(':scope > summary')?.textContent==='Action timeline')?.querySelectorAll('button').length===${prefixMode?lastResponse.verification.verifiedThrough:lastResponse.branches.find(b=>b.id===branch).steps}`);
      report.checks.expandedTimelineComplete=true;
      if(prefixMode){
        const v=lastResponse.verification;
        const ui=await js(`({label:document.querySelector('[aria-label="Reconstructed prefix"]')?.textContent,blocked:[...document.querySelectorAll('button')].filter(b=>['Apply','Preview','Branch','Undo','Redo','Reset'].includes(b.textContent.trim())).every(b=>b.disabled||b.closest('fieldset')?.disabled),timelineButtons:[...document.querySelectorAll('details')].find(d=>d.querySelector(':scope > summary')?.textContent==='Action timeline')?.querySelectorAll('button').length})`);
        // Compact wire responses omit the resident timeline. Count its actual
        // hydrated UI entries against complete prefix progress instead.
        report.prefix.final={verification:v,timelineLength:ui.timelineButtons,wireTimelineOmitted:lastResponse.timelineOmitted===true,originalActions:lastResponse.branches.find(b=>b.id===branch).steps,fullCase:lastResponse.fullCase,ui};
        report.checks.prefixReadOnly=v?.status==='reconstructed-prefix'&&v.readOnly===true&&v.verifiedThrough===3&&v.totalActions===report.prefix.originalActions&&v.archiveEndpointVerified===false&&lastResponse.fullCase===null&&ui.label?.includes('read only')&&ui.blocked&&ui.timelineButtons===3;
        const baseline=JSON.parse(fs.readFileSync(path.join(root,'testcases/outputs/rl-readiness-20261022/desktop-baseline/playback-authenticated/report.json')));
        const expected=new Map(baseline.playbackIpc.filter(row=>row.command==='seekFrame').map(row=>[row.frameIndex,row.stateId]));
        report.checks.baselineStateIds=report.submittedPlayback.length===12&&report.submittedPlayback.every(row=>row.stateId===expected.get(row.frameIndex));
        report.checks.noCanonicalCommands=report.ipc.filter(row=>row.channel==='research:surgicalRehearsal').every(row=>['inspectArchive','seekPrefix','seekPrefixFrame'].includes(row.command));
      }
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
      if(closeToolMode){
        // Display-only checks run after all playback timers/submissions settle.
        // Appearance changes retain adaptive interaction for 150 ms. A quiet
        // worker/100 ms delay cannot establish settled display restoration.
        await js(`window.closeToolGpus=[];document.addEventListener('anatomy-gpu',e=>window.closeToolGpus.push(e.detail));`);
        const ui=()=>js(`(()=>{const section=document.querySelector('[aria-label="Close instrument view"]'),frame=window.playbackFrames.at(-1);return {stateId:document.querySelector('.surgical-panel')?.dataset.stateId,pinnedStateId:section?.dataset.pinnedStateId,pinnedSlot:section?.dataset.pinnedSlot,fieldMm:Number(section?.dataset.fieldMm),frame:frame?{serial:frame.serial,interactive:frame.interactive,surfaceQuality:frame.surfaceQuality,stateId:frame.surgicalStateId}:null,ranges:[...document.querySelectorAll('input[type="range"]')].map((e,i)=>({name:e.getAttribute('aria-label')||e.id||String(i),value:e.value})),slices:[...document.querySelectorAll('.rl-viewer-slice-image')].map(e=>e.getAttribute('aria-label'))};})()`);
        const settled=async(serial)=>{
          await idle();
          await wait(`window.playbackFrames.some(f=>f.serial>${serial}&&!f.interactive&&f.surfaceQuality?.adaptive===false&&f.surgicalStateId===${JSON.stringify(report.final.stateId)}&&window.closeToolGpus.some(g=>g.serial===f.serial&&g.completed===true))`);
        };
        await wait(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Close tool view'&&!b.disabled)`);
        await wait(`window.playbackFrames.at(-1)?.interactive===false&&window.playbackFrames.at(-1)?.surfaceQuality?.adaptive===false`);
        report.closeTool={scope:'Post-playback optional display controls; no new replay request or physics permitted',started:performance.now(),before:await ui()};
        fs.writeFileSync(path.join(output,'before-close-tool.png'),(await window.webContents.capturePage()).toPNG());
        await click('Close tool view');await wait(`!!document.querySelector('[aria-label="Close instrument view"]')?.dataset.pinnedStateId`);await settled(report.closeTool.before.frame.serial);
        report.closeTool.close=await ui();
        fs.writeFileSync(path.join(output,'close-tool.png'),(await window.webContents.capturePage()).toPNG());
        // Refocusing an unchanged accepted tip may be a no-op for geometry and
        // appearance; its already completed settled frame remains applicable.
        await click('Refocus accepted tool');await idle();report.closeTool.refocused=await ui();
        await click('Restore previous view');await wait(`!document.querySelector('[aria-label="Close instrument view"]')?.dataset.pinnedStateId`);await settled(report.closeTool.refocused.frame.serial);
        report.closeTool.restored=await ui();
        report.closeTool.gpuCompletions=await js('window.closeToolGpus');
        report.closeTool.ended=performance.now();
        report.closeTool.newReplayCommands=report.ipc.filter(row=>row.channel==='research:surgicalRehearsal'&&row.started>=report.closeTool.started);
        fs.writeFileSync(path.join(output,'restored-close-tool.png'),(await window.webContents.capturePage()).toPNG());
        const q=report.closeTool;
        report.checks.closeToolIdentity=[q.before,q.close,q.refocused,q.restored].every(item=>item.stateId===report.final.stateId)&&q.close.pinnedStateId===report.final.stateId&&q.refocused.pinnedStateId===report.final.stateId&&!q.restored.pinnedStateId;
        report.checks.closeToolFraming=!!q.close.pinnedSlot&&q.close.fieldMm>=18&&q.close.fieldMm<=42&&q.refocused.fieldMm===q.close.fieldMm;
        report.checks.closeToolRestore=JSON.stringify(q.before.ranges)===JSON.stringify(q.restored.ranges)&&JSON.stringify(q.before.slices)===JSON.stringify(q.restored.slices);
        report.checks.closeToolNoReplay=q.newReplayCommands.length===0;
        report.checks.closeToolSettled=[q.before,q.close,q.refocused,q.restored].every(item=>item.frame?.interactive===false&&item.frame.surfaceQuality?.adaptive===false&&item.frame.stateId===report.final.stateId);
        report.checks.closeToolCompleted=q.close.frame?.serial!==q.restored.frame?.serial&&[q.close,q.restored].every(item=>q.gpuCompletions.some(g=>g.serial===item.frame?.serial&&g.completed===true));
      }
      }
      report.checks.sourcePreserved=hash(sourcePath)===report.source.sha256;
      fs.writeFileSync(path.join(output,'settled.png'),(await window.webContents.capturePage()).toPNG());
      report.passed=report.errors.length===0&&Object.values(report.checks).every(Boolean);
    }catch(error){report.passed=false;report.failure=error.stack;try{fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage()).toPNG());}catch{}}
    finally{await backend?.stop();report.physicsGuard=fs.existsSync(path.join(output,'physics-guard.json'))?JSON.parse(fs.readFileSync(path.join(output,'physics-guard.json'))):null;if(prefixMode){const g=report.physicsGuard,owned=g?.ownership,stats=owned?.prefixStatistics;report.checks.boundedPreparation=!!g&&g.permittedSteps?.length===3&&!g.forbiddenSteps?.length&&!Object.keys(g.publicStepAttempts).length;report.checks.noCanonicalAdmission=owned?.beforeClose===true&&['canonicalSessionCreated','replayCacheCreated','frameCacheCreated','frameAdmissionsCreated'].every(key=>owned[key]===false);report.checks.privateRetentionBounded=!!stats&&stats.verifiedThrough===3&&stats.totalActions===report.prefix.originalActions&&stats.checkpointBytes<=stats.maxCheckpointBytes&&stats.frameBytes<=stats.maxFrameBytes&&stats.historyBytes<=stats.maxHistoryBytes;report.checks.pythonSourcePreserved=JSON.stringify(sourceFiles(source))===JSON.stringify(report.pythonSourceFiles);report.passed=report.passed&&Object.values(report.checks).every(Boolean);}else {report.checks.zeroPhysics=!!report.physicsGuard&&!Object.keys(report.physicsGuard.publicStepAttempts).length;report.passed=report.passed&&report.checks.zeroPhysics;}persist();console.log(JSON.stringify({output,passed:report.passed,failure:report.failure,final:report.final}));app.exit(report.passed?0:1);}
  });
});
setTimeout(async()=>{report.failure='Overall bounded benchmark timeout';report.passed=false;persist();await backend?.stop();if(windowRef&&!windowRef.isDestroyed())try{fs.writeFileSync(path.join(output,'timeout.png'),(await windowRef.webContents.capturePage()).toPNG());}catch{}app.exit(1);},240000).unref();
require('./main.cjs');
