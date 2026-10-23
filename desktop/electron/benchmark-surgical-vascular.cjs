'use strict';
// Separate full-vascular gate. Does not change the matched 12-frame benchmark.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const ROOT=path.resolve(__dirname,'../..');
const BRANCH='full-vascular';
const COUNTS=[3,152,152,117,2,2,2,15,2,2,2,48,2,2,2,11,33,36,35,3,3,3,3,48,3,3,3,3,3,52,50,3,3,3,3,3,52,31,3,3,3,13,3,3,13,3,3,7,3,12,27,3,3,3,7,17,3,27,3,3,3,7,12,3,3,3,3,7,12,3,3,27,3];
const END_STATE='sha256:af496ffbde3ca2ab71477c9226fe9f846f08171efd308362431414584cd3c87e';
const NATIVE=path.join(ROOT,'testcases/outputs/full-case-rehearsal-20261020/final/rehearsal.ressectionlab');
const NATIVE_SHA='095659e54f381355e8f5655b6c5907990b2ac4767dc9ce58297e3bc4d6f3268d';
const hash=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const tuple=row=>JSON.stringify([row.branchId,row.position,row.frameIndex,row.stateId]);
function expectedSequence(){return COUNTS.flatMap((count,position)=>Array.from({length:count},(_,frameIndex)=>({branchId:BRANCH,position,frameIndex})));}
function submittedVascularPlayback(requests,frames,{rendererReceipts=[],startedRenderer}){
  const delivered=new Map();
  for(const row of requests){
    if(row.command!=='seekPrefixFrame'||row.error||row.branchId!==BRANCH||!Number.isInteger(row.position)||!Number.isInteger(row.frameIndex)||typeof row.stateId!=='string'||!Number.isFinite(row.started)||!Number.isFinite(row.ms))continue;
    const key=tuple(row);if(!delivered.has(key))delivered.set(key,[]);delivered.get(key).push(row);
  }
  // The callback runs in the renderer before the invoke promise resolves to
  // the app. Its clock is the same one used by anatomy-frame. Main-process
  // timing remains separate; an asynchronous clock-offset estimate cannot
  // establish completion order for fast submissions.
  const receipts=new Map();let previousSerial=0,previousAt=-Infinity,valid=true;
  for(const row of rendererReceipts){
    if(row.command!=='seekPrefixFrame'||row.branchId!==BRANCH||!Number.isInteger(row.position)||!Number.isInteger(row.frameIndex)||typeof row.stateId!=='string'||!Number.isFinite(row.receivedAt)||!Number.isInteger(row.notificationSerial)||row.notificationSerial<=previousSerial||row.receivedAt<previousAt){valid=false;break;}
    previousSerial=row.notificationSerial;previousAt=row.receivedAt;
    if(row.receivedAt<startedRenderer)continue;
    const key=tuple(row);if(!receipts.has(key))receipts.set(key,[]);receipts.get(key).push(row);
  }
  const result=[];
  for(const frame of frames){
    if(!Number.isFinite(frame.at)||frame.at<startedRenderer||!frame.surgicalFrame)continue;
    const identity={...frame.surgicalFrame,stateId:frame.surgicalStateId};
    const prior=result.at(-1);if(prior&&tuple(prior)===tuple(identity))continue;
    const candidates=delivered.get(tuple(identity)),received=receipts.get(tuple(identity));
    const receipt=valid&&received?.length===1&&received[0].receivedAt<=frame.at?received[0]:null;
    const row=receipt&&candidates?.length===1?candidates[0]:null;
    result.push(row?{...row,rendererReceivedAt:receipt.receivedAt,receiptNotificationSerial:receipt.notificationSerial,firstSubmittedAt:frame.at,submissionSerial:frame.serial}:{...identity,firstSubmittedAt:frame.at,submissionSerial:frame.serial,unmatched:true});
  }
  return result;
}
function exactSequence(rows,expected=expectedSequence()){
  return rows.length===expected.length&&rows.every((row,i)=>!row.unmatched&&row.branchId===expected[i].branchId&&row.position===expected[i].position&&row.frameIndex===expected[i].frameIndex&&(!expected[i].stateId||row.stateId===expected[i].stateId));
}
function vascularPreparedOnly(value){
  if(value===undefined)return false;
  if(value==='1')return true;
  throw Error('VASCULAR_PREPARED_ONLY must be unset or exactly1');
}
function vascularStartButton(preparedOnly){return preparedOnly?'Replay prepared observations · read only':'Replay from start · read only';}
function vascularPreparationChecks(guard,preparedOnly){
  const cache=guard?.ownership?.prefixStatistics?.preparedCache;
  const clean=!!guard&&!guard.failure&&Array.isArray(guard.permittedSteps)&&Array.isArray(guard.forbiddenSteps)&&!guard.forbiddenSteps.length&&!!guard.publicStepAttempts&&!Object.keys(guard.publicStepAttempts).length;
  return {stepAuthority:clean&&guard.qualificationMode===(preparedOnly?'prepared-only':'fresh')&&guard.permittedSteps.length===(preparedOnly?0:73),
    preparedCache:!!cache&&cache.mode===(preparedOnly?'only':'prepare')&&cache.loadedActions===(preparedOnly?73:0)&&cache.publishedActions===(preparedOnly?0:73)&&cache.misses===0&&cache.lastSkip===null};
}
function checkedVascularAuthority(command,result,preparedOnly){
  if(result?.verification?.readOnly!==true||result.changed!==false)throw Error('Response granted unexpected authority');
  if(command==='inspectArchive'){
    if(result.verification.status!=='unverified-import'||result.verification.canonical!==false||result.verification.physicsActionsExecuted!==0)throw Error('Imported endpoint lost unverified read-only authority');
  }else if(['seekPrefix','seekPrefixFrame'].includes(command)){
    if(result.verification.status!=='reconstructed-prefix'||result.readOnly!==true||result.fullCase!==null||result.branchId!==BRANCH||result.verification.totalActions!==73)throw Error('Prefix response granted unexpected authority');
    if(preparedOnly!==undefined&&result.verification.preparedCache?.mode!==(preparedOnly?'only':'prepare'))throw Error('Prefix response changed prepared-only mode');
  }else throw Error('Unexpected surgical response command');
  return result;
}
function hardBudgetStop(writeReceipt,kill,exit){
  try{writeReceipt();}finally{try{kill();}finally{exit(1);}}
}
function isVascularBenchmarkEntry(runtime,filename,nodeMain=false){
  const entry=runtime.argv?.[1];
  if(typeof entry!=='string'||path.resolve(entry)!==path.resolve(filename))return false;
  return runtime.versions?.electron?runtime.type==='browser':nodeMain===true;
}
function qualifiedFlowCapture(before,after){
  if(!before?.frame||!after?.frame||!before.flow||!after.flow)return false;
  const f=before.frame,flow=before.flow;
  return f.branchId===BRANCH&&Number.isInteger(f.position)&&f.position>=0&&f.position<COUNTS.length&&Number.isInteger(f.frameIndex)&&f.frameIndex>=0&&f.frameIndex<COUNTS[f.position]&&
    ['branchId','position','frameIndex'].every(key=>f[key]===after.frame[key])&&/^sha256:[0-9a-f]{64}$/.test(before.stateId)&&before.stateId===after.stateId&&before.panelStateId===before.stateId&&after.panelStateId===after.stateId&&
    Number.isFinite(before.at)&&Number.isFinite(after.at)&&after.at>=before.at&&Number.isInteger(flow.operations)&&flow.operations>0&&flow.operations<=256&&Number.isInteger(flow.glyphInstances)&&flow.glyphInstances>0&&flow.glyphInstances<=512&&
    Number.isFinite(flow.startSeconds)&&Number.isFinite(flow.endSeconds)&&flow.startSeconds>=0&&flow.endSeconds>=flow.startSeconds&&typeof flow.truncated==='boolean'&&
    ['startSeconds','endSeconds','operations','truncated','glyphInstances'].every(key=>flow[key]===after.flow[key]);
}
module.exports={COUNTS,END_STATE,BRANCH,expectedSequence,submittedVascularPlayback,exactSequence,checkedVascularAuthority,hardBudgetStop,isVascularBenchmarkEntry,vascularPreparedOnly,vascularStartButton,vascularPreparationChecks,qualifiedFlowCapture};

if(isVascularBenchmarkEntry(process,__filename,require.main===module))run();
function run(){
  const {app,ipcMain,screen}=require('electron'),childProcess=require('node:child_process');
  if(process.env.RESSECTIONLAB_PLAYBACK_RECEIPT_OBSERVER!=='1')throw Error('Full vascular qualification requires explicit RESSECTIONLAB_PLAYBACK_RECEIPT_OBSERVER=1');
  const preparedOnly=vascularPreparedOnly(process.env.VASCULAR_PREPARED_ONLY);
  if(process.env.VASCULAR_CAPTURE_FLOW!==undefined&&process.env.VASCULAR_CAPTURE_FLOW!=='1')throw Error('VASCULAR_CAPTURE_FLOW must be absent or 1');
  const captureFlow=process.env.VASCULAR_CAPTURE_FLOW==='1';
  const configuredCache=process.env.RESSECTIONLAB_PREPARED_PREFIX_CACHE_DIRECTORY;
  const cacheDirectory=configuredCache?path.resolve(configuredCache):null;
  if((preparedOnly&&!cacheDirectory)||(configuredCache!==undefined&&(!cacheDirectory||!path.isAbsolute(configuredCache)||!cacheDirectory.startsWith(path.join(ROOT,'testcases/outputs/rl-readiness-20261022')+path.sep))))throw Error('Qualification requires an absolute isolated prepared-cache directory under ignored outputs');
  if(['PLAYBACK_PYTHON_SOURCE','PLAYBACK_BRIDGE_SOURCE','PLAYBACK_IMPORT_ONLY','PLAYBACK_ORBIT','PLAYBACK_FIELD_SIZES','PLAYBACK_PREFIX','PLAYBACK_CLOSE_TOOL'].some(name=>process.env[name]))throw Error('Full vascular gate requires current production source and its own dedicated controls; no substitutions/observer modes');
  const output=path.resolve(process.env.PLAYBACK_OUTPUT||path.join(ROOT,'testcases/outputs/rl-readiness-20261022/desktop-vascular-qualified'));
  if(!output.startsWith(path.join(ROOT,'testcases/outputs/rl-readiness-20261022')+path.sep)||fs.existsSync(output))throw Error('Supply a fresh ignored PLAYBACK_OUTPUT under rl-readiness-20261022');
  const caseIndex=process.argv.indexOf('--case'),sourcePath=caseIndex>=0?path.resolve(process.argv[caseIndex+1]||''):NATIVE;
  if(caseIndex<0||sourcePath!==NATIVE||hash(sourcePath)!==NATIVE_SHA)throw Error('Only the exact preserved original archive is permitted');
  fs.mkdirSync(output,{recursive:false});
  const source=path.join(ROOT,'src'),monitor=path.join(__dirname,'vascular_playback_monitor.py');
  const reference=path.join(path.dirname(NATIVE),'vascular-canonical.actions.jsonl');
  const historical=path.join(ROOT,'testcases/outputs/full-case-rehearsal-20261020/review/vascular-observability-28962e66-every-frame.json');
  const historicalCounts=JSON.parse(fs.readFileSync(historical)).actionsVerified.map(row=>row.frames);
  if(JSON.stringify(historicalCounts)!==JSON.stringify(COUNTS))throw Error('Historical observed counts differ');
  const originalFiles={native:{path:NATIVE,sha256:NATIVE_SHA},reference:{path:reference,sha256:hash(reference)},historicalFrameCountAudit:{path:historical,sha256:hash(historical)}};
  const codeFiles=()=>{const result={};for(const directory of [source,path.join(ROOT,'desktop/dist')]){const visit=dir=>{for(const entry of fs.readdirSync(dir,{withFileTypes:true}).sort((a,b)=>a.name.localeCompare(b.name))){if(entry.name==='__pycache__')continue;const file=path.join(dir,entry.name);if(entry.isDirectory())visit(file);else if(directory.endsWith('dist')||entry.name.endsWith('.py'))result[path.relative(ROOT,file)]=hash(file);}};visit(directory);}return result;};
  const frozenFiles=codeFiles();
  const report={schema:'resectionlab.full-vascular-prefix-gui.v1',recordedAt:new Date().toISOString(),
    scope:(preparedOnly?'Previously prepared original73 saved vascular actions via current-production read-only prefix; no fresh transitions.':'Original73 saved vascular actions freshly reconstructed via current-production read-only prefix; private prepared-cache publication tracked separately.')+' No editable/canonical admission, no RL. Every recorded tuple must be CPU-submitted in order. CPU submission is NOT compositor presentation; historical all-frame hashes are unavailable.',
    qualificationMode:preparedOnly?'prepared-only':'fresh',preparedCacheDirectory:cacheDirectory,
    limits:{wallMs:600000,wholePythonRssBytes:3*1024**3,binarySpoolBytes:1024**3,manifestBytes:16*1024**2,postGuiAuditWallMs:180000},
    originalFiles,codeFiles:frozenFiles,harnessSha256:hash(__filename),guardSha256:hash(monitor),versions:process.versions,
    ipc:[],errors:[],diagnostics:[],checks:{},passed:false,captureRecordedFlow:captureFlow,frameOperativeField:process.env.VASCULAR_FRAME_FIELD==='1'};
  const persist=()=>fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));
  const runner=path.join(output,'guarded-bridge.py');
  fs.writeFileSync(runner,`import sys\nsys.path.insert(0, ${JSON.stringify(__dirname)})\nfrom vascular_playback_monitor import VascularPlaybackMonitor,initialize_guard_inventory\nfrom resectionlab.surgical_engine import SurgicalEngine\nfrom resectionlab import surgical_prefix_replay,surgical_frame_replay,surgical_telemetry\nfrom resectionlab.desktop_bridge import BridgeSession,main\ninitialize_guard_inventory()\nguard=VascularPlaybackMonitor(SurgicalEngine, ${JSON.stringify(output)}, prepared_only=${preparedOnly?'True':'False'})\nguard.install(BridgeSession)\ntry:\n    with guard:\n        main()\nfinally:\n    guard.finish()\n`);
  const originalSpawn=childProcess.spawn;
  childProcess.spawn=(command,args,options)=>args?.[1]==='-m'&&args?.[2]==='resectionlab.desktop_bridge'?originalSpawn(command,['-u',runner,...args.slice(3)],options):originalSpawn(command,args,options);
  const sidecar=require('./sidecar.cjs'),OriginalSidecar=sidecar.Sidecar;let backend;
  sidecar.Sidecar=class extends OriginalSidecar{constructor(config){super({...config,sourcePath:source});backend=this;report.pythonExecutable=config.executable||config.python;this.on('diagnostic',message=>{if(report.diagnostics.length<100)report.diagnostics.push(message);});}};
  app.setPath('userData',path.join(output,'user-data'));
  let lastResponse,windowRef,finishing=false,timer;
  const originalHandle=ipcMain.handle.bind(ipcMain);
  ipcMain.handle=(channel,handler)=>originalHandle(channel,async(...args)=>{
    const began=performance.now(),request=args[1];
    const entry={channel,command:request?.command,started:began,requestedPosition:request?.position,requestedFrameIndex:request?.frameIndex,requestedCacheOnly:request?.cacheOnly};
    try{
      const result=await handler(...args);entry.ms=performance.now()-began;
      if(channel==='research:surgicalRehearsal'){
        lastResponse=result;
        Object.assign(entry,{branchId:result.playback?.branchId??result.branchId,position:result.playback?.position??result.position,frameIndex:result.playback?.frameIndex,
          stateId:result.view?.stateId,simulatedSeconds:result.playback?.observedTimeSeconds,frameCount:result.playback?.frameCount,
          verifiedThrough:result.verification?.verifiedThrough,readOnly:result.verification?.readOnly===true,verificationStatus:result.verification?.status,
          archiveEndpointVerified:result.verification?.archiveEndpointVerified,preparedCache:result.verification?.preparedCache,timelineOmitted:result.timelineOmitted===true,responseBytes:Buffer.byteLength(JSON.stringify(result))});
        checkedVascularAuthority(request.command,result,preparedOnly);
      }
      report.ipc.push(entry);return result;
    }catch(error){entry.ms=performance.now()-began;entry.error=String(error);report.ipc.push(entry);throw error;}
  });
  const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
  const finish=async()=>{
    if(finishing)return;finishing=true;clearTimeout(timer);
    await backend?.stop();
    report.ipcAtShutdown=report.ipc.length;
    report.allCompletedPlaybackIpc=report.ipc.slice(report.playbackStartIpcIndex??report.ipc.length).filter(row=>row.command==='seekPrefixFrame');
    const counts=new Map();for(const row of report.allCompletedPlaybackIpc)counts.set(tuple(row),(counts.get(tuple(row))??0)+1);
    report.fetchSummary={completed:report.allCompletedPlaybackIpc.length,duplicates:[...counts.values()].reduce((sum,n)=>sum+Math.max(0,n-1),0),lateAfterSnapshot:report.ipc.slice(report.ipcAtSnapshot??report.ipc.length).filter(row=>row.command==='seekPrefixFrame').length,unadmitted:report.allCompletedPlaybackIpc.filter(row=>!(report.submittedPlayback??[]).some(shown=>tuple(shown)===tuple(row))).length};
    if(fs.existsSync(path.join(output,'physics-guard.json')))report.physicsGuard=JSON.parse(fs.readFileSync(path.join(output,'physics-guard.json')));
    const g=report.physicsGuard,stats=g?.ownership?.prefixStatistics;
    const preparation=vascularPreparationChecks(g,preparedOnly);
    report.checks[preparedOnly?'zeroStepPreparedPlayback':'exact73Preparations']=preparation.stepAuthority;
    report.checks[preparedOnly?'exact73PreparedLoads':'exact73PreparedPublications']=preparation.preparedCache;
    report.checks.guardWallAndRss=!!g&&g.wallSeconds<600&&g.processPeakResidentBytes<=3*1024**3;
    report.checks.noCanonicalAdmission=!!g?.ownership&&['canonicalSessionCreated','replayCacheCreated','frameCacheCreated','frameAdmissionsCreated'].every(key=>g.ownership[key]===false);
    report.checks.retainedCaps=!!stats&&stats.verifiedThrough===73&&stats.totalActions===73&&stats.frameBytes<=stats.maxFrameBytes&&stats.checkpointBytes<=stats.maxCheckpointBytes&&stats.historyBytes<=stats.maxHistoryBytes;
    report.checks.sourcePreserved=Object.values(originalFiles).every(row=>hash(row.path)===row.sha256)&&JSON.stringify(codeFiles())===JSON.stringify(frozenFiles)&&hash(monitor)===report.guardSha256&&hash(__filename)===report.harnessSha256;
    persist();
    // Heavy independent decode/hash/conservation audit starts AFTER the GUI run
    // and sidecar shutdown. It cannot influence measured playback throughput.
    if(preparation.stepAuthority&&preparation.preparedCache&&report.checks.sourcePreserved){
      const auditStarted=performance.now();
      report.postGuiAudit=await new Promise(resolve=>{
        const child=originalSpawn(report.pythonExecutable,[monitor,'audit','--output',output],{cwd:ROOT,env:{...process.env,PYTHONPATH:source},stdio:['ignore','pipe','pipe']});
        let text='';const timeout=setTimeout(()=>{child.kill('SIGKILL');resolve({passed:false,error:'Post-GUI audit exceeded180seconds'});},180000);
        child.stdout.on('data',data=>{if(text.length<16000)text+=data;});child.stderr.on('data',data=>{if(text.length<16000)text+=data;});
        child.on('error',error=>{clearTimeout(timeout);resolve({passed:false,error:String(error)});});
        child.on('exit',(code,signal)=>{clearTimeout(timeout);resolve({passed:code===0,code,signal,log:text});});
      });
      report.postGuiAudit.wallMs=performance.now()-auditStarted;
      if(fs.existsSync(path.join(output,'spool-audit.json')))report.spoolAudit=JSON.parse(fs.readFileSync(path.join(output,'spool-audit.json')));
      if(report.spoolAudit?.passed&&fs.existsSync(path.join(output,'audited-observations.json'))){
        const expected=JSON.parse(fs.readFileSync(path.join(output,'audited-observations.json')));
        report.checks[preparedOnly?'everyPreparedObservationSubmitted':'everyFreshObservationSubmitted']=exactSequence(report.submittedPlayback??[],expected);
      }
    }
    report.checks.fullIndependentAudit=report.postGuiAudit?.passed===true&&report.spoolAudit?.passed===true&&report.spoolAudit.allFullCanonicalRecordsExact===true;
    report.checks.sourcePreservedAfterAudit=Object.values(originalFiles).every(row=>hash(row.path)===row.sha256)&&JSON.stringify(codeFiles())===JSON.stringify(frozenFiles);
    report.passed=!report.failure&&report.errors.length===0&&Object.values(report.checks).every(Boolean);persist();
    console.log(JSON.stringify({output,passed:report.passed,failure:report.failure,checks:report.checks,final:report.final}));app.exit(report.passed?0:1);
  };
  app.on('browser-window-created',(_,window)=>{
    windowRef=window;window.webContents.setBackgroundThrottling(false);
    const secondary=screen.getAllDisplays().find(d=>d.id!==screen.getPrimaryDisplay().id);
    if(secondary){const a=secondary.workArea;window.setBounds({x:a.x+20,y:a.y+20,width:Math.min(1480,a.width-40),height:Math.min(900,a.height-40)});window.showInactive();}
    else{window.setSize(1480,900);window.show();}
    report.display={secondary:secondary??null,all:screen.getAllDisplays()};
    window.webContents.on('console-message',event=>{if(event.level==='error'&&report.errors.length<100)report.errors.push(event.message);});
    window.webContents.once('did-finish-load',async()=>{
      const js=code=>window.webContents.executeJavaScript(code,true);
      const flowEvidence=()=>js(`(()=>{const frame=window.vascularFrames.at(-1),panel=document.querySelector('[aria-label="Recorded field transfers"]');return frame?.recordedFlow&&panel?{frame:frame.surgicalFrame,stateId:frame.surgicalStateId,panelStateId:document.querySelector('.surgical-panel')?.dataset.stateId,flow:frame.recordedFlow,text:panel.textContent,at:performance.now()}:null;})()`);
      const wait=async(expr,timeout=120000)=>{const began=performance.now();while(performance.now()-began<timeout){if(await js(expr))return;await pause(40);}throw Error('Timed out: '+expr);};
      const click=label=>js(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(label)});if(!b||b.disabled)throw Error('Missing enabled '+${JSON.stringify(label)});b.click();})()`);
      const set=(selector,value)=>js(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});if(!e||e.disabled)throw Error('Missing enabled control');Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(e,${JSON.stringify(value)});e.dispatchEvent(new Event('change',{bubbles:true}));})()`);
      const idle=()=>wait(`!!document.querySelector('.surgical-slots')&&!document.querySelector('.surgical-panel > p[role="status"]')&&!document.querySelector('.rl-viewer-preparing')`);
      try{
        await wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')&&!document.querySelector('.rl-viewer-preparing')`);
        await click('Rehearse');await idle();
        if(lastResponse.branchId!==BRANCH){await set('select[aria-label="Surgical branch"]',BRANCH);await idle();}
        if(lastResponse.verification?.status!=='unverified-import'||lastResponse.branches.find(b=>b.id===BRANCH)?.steps!==73)throw Error('Must start from original unverified vascular endpoint');
        await set('select[aria-label="Replay speed"]','4');
        await js(`(()=>{window.vascularFrames=[];window.vascularWork=[];window.vascularFailure=null;window.vascularField=null;window.vascularSubmitted=[];window.vascularReceipts=[];const expected=${JSON.stringify(expectedSequence())};
          if(typeof window.resectionApi.onSurgicalReceipt!=='function')throw Error('Explicit renderer receipt observer unavailable');
          window.vascularReceiptCleanup=window.resectionApi.onSurgicalReceipt(row=>{const receivedAt=performance.now();if(window.vascularReceipts.length>=20000){window.vascularFailure='Renderer receipt evidence limit exceeded';throw Error(window.vascularFailure);}window.vascularReceipts.push({...row,receivedAt});});
          const readOnly=()=>{const prefix=document.querySelector('[aria-label="Reconstructed prefix"]');if(!prefix)return;const buttons=[...document.querySelectorAll('button')].filter(b=>['Apply','Preview','Branch','Undo','Redo','Reset'].includes(b.textContent.trim()));if(!buttons.length||buttons.some(b=>!b.disabled&&!b.closest('fieldset')?.disabled)){window.vascularFailure='Editable controls enabled during prefix';}};
          new MutationObserver(readOnly).observe(document.querySelector('.surgical-panel'),{subtree:true,childList:true,attributes:true});
          document.addEventListener('anatomy-frame',e=>{const f=e.detail;if(window.vascularFrames.length>=20000){window.vascularFailure='CPU frame evidence limit exceeded';return;}const identity=f.surgicalFrame;if(identity){const row={...identity,stateId:f.surgicalStateId,at:f.at,serial:f.serial},prior=window.vascularSubmitted.at(-1);if(!prior||JSON.stringify([prior.branchId,prior.position,prior.frameIndex,prior.stateId])!==JSON.stringify([row.branchId,row.position,row.frameIndex,row.stateId])){const target=expected[window.vascularSubmitted.length];if(!target||row.branchId!==target.branchId||row.position!==target.position||row.frameIndex!==target.frameIndex)window.vascularFailure='CPU observation tuple order differs';if(window.vascularSubmitted.length<1150)window.vascularSubmitted.push(row);}}window.vascularFrames.push({at:f.at,serial:f.serial,surgicalStateId:f.surgicalStateId,surgicalFrame:f.surgicalFrame,cpuMs:f.cpuMs,interactive:f.interactive,surfaceQuality:f.surfaceQuality,estimatedAllocation:f.estimatedAllocation,resources:f.resources,recordedFlow:f.recordedFlow??null});readOnly();});
          document.addEventListener('anatomy-work',e=>{if(window.vascularWork.length>=20000){window.vascularFailure='Worker evidence limit exceeded';return;}window.vascularWork.push({...e.detail,at:performance.now()});});})();`);
        report.playStarted=performance.now();report.playStartedRenderer=await js('performance.now()');const index=report.ipc.length;report.playbackStartIpcIndex=index;
        await click(vascularStartButton(preparedOnly));
        const options={startedRenderer:report.playStartedRenderer};
        let count=0,lastProgressAt=-Infinity;
        while(performance.now()-report.playStarted<600000){
          await pause(100);
          const evidence=await js(`({count:window.vascularSubmitted.length,last:window.vascularSubmitted.at(-1),failure:window.vascularFailure,pending:!!document.querySelector('.rl-viewer-preparing'),playing:[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Pause replay'),inserted:[...document.querySelectorAll('.surgical-slots small')].some(e=>e.textContent.startsWith('Inserted')),stateId:document.querySelector('.surgical-panel')?.dataset.stateId})`);
          if(evidence.failure)throw Error(evidence.failure);
          if(report.ipc.slice(index).some(row=>row.error))throw Error('Playback request failed');
          count=evidence.count;
          if(performance.now()-lastProgressAt>=1000){lastProgressAt=performance.now();report.progress={at:lastProgressAt,submittedCount:count,last:evidence.last,pending:evidence.pending,playing:evidence.playing,ipcCount:report.ipc.length};fs.writeFileSync(path.join(output,'progress.json'),JSON.stringify(report.progress));}
          if(captureFlow&&!report.flowCapture&&(report.flowCaptureAttempts?.length??0)<3&&count&&!evidence.pending){
            const flow=await flowEvidence();
            if(flow?.flow.operations>0&&flow.flow.glyphInstances>0){
              const filename='recorded-flow-attempt-'+((report.flowCaptureAttempts?.length??0)+1)+'.png';
              fs.writeFileSync(path.join(output,filename),(await window.webContents.capturePage()).toPNG());
              const attempt={file:filename,before:flow,after:await flowEvidence(),meaning:'Intermediate screenshot; capture overhead INCLUDED in full-run timing. Static historical transfer glyphs; endpoint anatomy, no animated-fluid/compositor claim.'};
              attempt.qualified=qualifiedFlowCapture(attempt.before,attempt.after);
              if(attempt.qualified){fs.renameSync(path.join(output,filename),path.join(output,'recorded-flow-interval.png'));attempt.file='recorded-flow-interval.png';report.flowCapture=attempt;}
              (report.flowCaptureAttempts??=[]).push(attempt);
            }
          }
          if(report.frameOperativeField&&!report.displayChange&&evidence.inserted&&count){
            report.displayChange=await js(`(()=>{if(![...document.querySelectorAll('.surgical-slots small')].some(e=>e.textContent.startsWith('Inserted')))return null;const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Frame operative field');if(!b||b.disabled)throw Error('Missing field framing control');const change={at:performance.now(),acceptedTuple:window.vascularSubmitted.at(-1),stateId:document.querySelector('.surgical-panel')?.dataset.stateId,meaning:'One explicit operative-field framing at a source-contained instrument or same-source annotation; pinned thereafter, no automatic follow/interpolation'};b.click();return change;})()`);
          }
          if(count===1149&&!evidence.playing&&!evidence.pending)break;
        }
        report.playEnded=performance.now();report.playEndedRenderer=await js('performance.now()');
        Object.assign(report,await js(`({frames:window.vascularFrames,rendererReceipts:window.vascularReceipts,work:window.vascularWork,uiFailure:window.vascularFailure,displayedStateId:document.querySelector('.surgical-panel')?.dataset.stateId,visible:document.visibilityState,slicePositions:[...document.querySelectorAll('.rl-viewer-slice-position')].map(e=>e.textContent)})`));
        report.rendererReceiptObserver={enabled:true,scope:'Explicit benchmark observer; renderer callback timestamps before invoke result resolves to app. Same renderer clock as anatomy-frame; observer cost INCLUDED.',notifications:report.rendererReceipts.length};
        report.submittedPlayback=submittedVascularPlayback(report.ipc.slice(index),report.frames,{...options,rendererReceipts:report.rendererReceipts});
        report.checks.exactRendererReceipts=report.rendererReceipts.length===1149;
        if(captureFlow)report.checks.recordedFlowCaptured=qualifiedFlowCapture(report.flowCapture?.before,report.flowCapture?.after);
        await js('window.vascularReceiptCleanup()');
        report.playbackIpc=report.ipc.slice(index);report.ipcAtSnapshot=report.ipc.length;
        const last=report.submittedPlayback.at(-1);report.final=last;
        report.checks.exactTupleSequence=exactSequence(report.submittedPlayback);
        report.checks.finalEndpoint=last?.position===72&&last.frameIndex===2&&last.stateId===END_STATE&&report.displayedStateId===END_STATE;
        report.checks.automaticStop=await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Play replay')&&!document.querySelector('.rl-viewer-preparing')`);
        report.checks.readOnlyAlways=!report.uiFailure&&report.playbackIpc.every(row=>row.channel!=='research:surgicalRehearsal'||row.readOnly===true);
        report.finalVerification=lastResponse.verification;
        report.checks.fullReadOnlyPrefix=lastResponse.verification?.status==='reconstructed-prefix'&&lastResponse.verification.readOnly===true&&lastResponse.verification.verifiedThrough===73&&lastResponse.verification.totalActions===73&&lastResponse.verification.archiveEndpointVerified===true&&lastResponse.position===73&&lastResponse.fullCase===null;
        await js(`(()=>{const d=[...document.querySelectorAll('details')].find(d=>d.querySelector(':scope > summary')?.textContent==='Action timeline');d.open=true;})()`);
        await wait(`[...document.querySelectorAll('details')].find(d=>d.querySelector(':scope > summary')?.textContent==='Action timeline')?.querySelectorAll('button').length===73`);
        report.checks.all73TimelineEntries=true;
        await js(`(()=>{const d=[...document.querySelectorAll('details')].find(d=>d.querySelector(':scope > summary')?.textContent==='Action timeline');d.open=false;})()`);
        await wait(`window.vascularFrames.at(-1)?.interactive===false&&window.vascularFrames.at(-1)?.surfaceQuality?.adaptive===false`);
        fs.writeFileSync(path.join(output,'settled.png'),(await window.webContents.capturePage()).toPNG());
        report.checks.wallBound=report.playEnded-report.playStarted<600000;
        const first=report.submittedPlayback[0],lastWarm=report.submittedPlayback.at(-1);
        const warmPairs=report.submittedPlayback.slice(1).flatMap((row,i)=>{const previous=report.submittedPlayback[i];return row.position===previous.position?[{sim:row.simulatedSeconds-previous.simulatedSeconds,wall:(row.firstSubmittedAt-previous.firstSubmittedAt)/1000}]:[];});
        const warmSums=warmPairs.reduce((sum,row)=>({sim:sum.sim+row.sim,wall:sum.wall+row.wall}),{sim:0,wall:0});
        report.rates={fullRunSimulatedSeconds:last?.simulatedSeconds,fullRunWallSeconds:(report.playEnded-report.playStarted)/1000,
          fullRunIncludingColdRatio:last?.simulatedSeconds/((report.playEnded-report.playStarted)/1000),
          cpuSubmittedIntervalSeconds:(lastWarm?.firstSubmittedAt-first?.firstSubmittedAt)/1000,
          cpuSubmittedIntervalRatio:(lastWarm?.simulatedSeconds-first?.simulatedSeconds)/((lastWarm?.firstSubmittedAt-first?.firstSubmittedAt)/1000),
          warmSameActionSubmittedSimulatedSeconds:warmSums.sim,warmSameActionSubmittedWallSeconds:warmSums.wall,warmSameActionSubmittedRatio:warmSums.sim/warmSums.wall,
          [preparedOnly?'preparedActionLoadHandlerSeconds':'coldActionHandlerSeconds']:report.playbackIpc.filter(row=>row.command==='seekPrefixFrame'&&row.frameIndex===0).map(row=>({position:row.position,seconds:row.ms/1000})),
          meaning:(preparedOnly?'Full run includes prepared-action disk loads, validation and spool overhead; no fresh transitions. Submitted interval includes later action loads.':'Full run includes each cold action reconstruction and spool overhead. Submitted interval still contains later cold preparations;')+' Warm same-action IPC requests have frameIndex>=1. CPU timestamps are not compositor presentation.'};
      }catch(error){report.failure=error.stack;try{
        Object.assign(report,await js(`({frames:window.vascularFrames??[],rendererReceipts:window.vascularReceipts??[],work:window.vascularWork??[],uiFailure:window.vascularFailure,displayedStateId:document.querySelector('.surgical-panel')?.dataset.stateId})`));
        if(Number.isFinite(report.playStarted)&&Number.isFinite(report.playStartedRenderer)){report.playbackIpc=report.ipc.slice(report.playbackStartIpcIndex);report.submittedPlayback=submittedVascularPlayback(report.playbackIpc,report.frames,{rendererReceipts:report.rendererReceipts,startedRenderer:report.playStartedRenderer});report.final=report.submittedPlayback.at(-1);report.partialEvidence=true;}
        fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage()).toPNG());
      }catch(evidenceError){report.partialEvidenceError=String(evidenceError);}}
      await finish();
    });
  });
  timer=setTimeout(()=>{report.failure='600 second overall GUI watchdog expired';hardBudgetStop(persist,()=>backend?.child.kill('SIGKILL'),code=>app.exit(code));},600000);timer.unref();
  persist();require('./main.cjs');
}
