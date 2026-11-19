'use strict';
// Production React/preload/engine/render/save path. Never show or focus a window.
const {app,dialog,ipcMain}=require('electron');
process.argv.push('--automation-hidden');
if(process.platform==='darwin')app.setActivationPolicy('accessory');
process.on('uncaughtException',error=>{console.error(error.stack);app.exit(1);});
dialog.showErrorBox=(_title,message)=>{console.error(message);app.exit(1);};
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const {execFileSync}=require('node:child_process');
const {inputScript}=require('./verify-eden-planning-helpers.cjs');
const {studyLayerSelectionScript,studyScanEnabledScript,studyScanSelectionScript,studyVisibilityToggleScript}=require('./verify-study-ui-helpers.cjs');
const repo=path.resolve(__dirname,'../..'),root=process.env.RESSECTIONLAB_DATA_ROOT;
if(!root)throw Error('Set RESSECTIONLAB_DATA_ROOT');
try{execFileSync(path.join(repo,'.venv/bin/python'),['-c','from resectionlab.data_root import checked_root; checked_root(required_bytes=1024**3)'],{cwd:repo,env:process.env});}
catch(error){console.error('External storage preflight failed: '+error.message);app.exit(1);}
const input=fs.realpathSync(process.env.DATA_REBUILD_CASE),output=path.resolve(process.env.DATA_REBUILD_OUTPUT);
for(const file of [input,output])assert(file.startsWith(root+path.sep),'Keep case and evidence on the configured external root');
assert(!fs.existsSync(output),'Preserve prior evidence; use a fresh output directory');fs.mkdirSync(output,{recursive:true});
const digest=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const inputSha=digest(input),saved=path.join(output,'saved.ressectionlab');
const targetId=process.env.DATA_REBUILD_CTA_TARGET_ID;
const targetLabel=process.env.DATA_REBUILD_CTA_TARGET_LABEL;
const sourceSha=process.env.DATA_REBUILD_CTA_SOURCE_SHA256;
const collectionFile=process.env.DATA_REBUILD_CTA_COLLECTION;
const replayOnly=process.env.DATA_REBUILD_REPLAY_ONLY==='1';
const replayOpeningFile=process.env.DATA_REBUILD_REPLAY_OPENING_REPORT;
const replayOpening=replayOpeningFile?JSON.parse(fs.readFileSync(replayOpeningFile,'utf8')):null;
if(replayOpening){
  assert(replayOnly&&replayOpening.status==='target_effect_and_withdrawal');
  assert.equal(replayOpening.replay_sha256,inputSha);
  assert.equal(replayOpening.historical_prefix_actuated_from_zero,true);
  assert.equal(replayOpening.opening_verified,true);
}
const resectPairFile=process.env.DATA_REBUILD_RESECT_PAIR_REPORT;
const resectPair=resectPairFile?JSON.parse(fs.readFileSync(resectPairFile,'utf8')):null;
if(resectPair){
  assert.equal(resectPair.schema,'resectionlab-case3-observed-stage-benchmark-v1');
  assert.equal(resectPair.patient,'RESECT:Case3');
  assert.equal(resectPair.members[path.basename(input)],inputSha);
  assert.equal(resectPair.postoperative_observation_excluded_from_preoperative_case,true);
}
const islesPair=process.env.DATA_REBUILD_INSPECT_ISLES_CTA==='1';
const ixiStudy=process.env.DATA_REBUILD_INSPECT_IXI==='1';
const nativeStudy=islesPair||ixiStudy||process.env.DATA_REBUILD_INSPECT_STUDY==='1'?JSON.parse(fs.readFileSync(input,'utf8')):null;
const priorFile=process.env.DATA_REBUILD_CTA_PRIOR;
const prior=priorFile?JSON.parse(fs.readFileSync(priorFile,'utf8')):null;
if(targetId)assert(targetLabel&&/^[a-f0-9]{64}$/.test(sourceSha),'Provide the exact target label and baseline source digest');
if(nativeStudy)assert.equal(nativeStudy.schema,'patient-anatomy-v1');
if(prior){
  assert(targetId&&prior.passed&&prior.ctaPreview&&prior.pid!==process.pid);
  assert.equal(input,fs.realpathSync(prior.saved));
  assert.equal(inputSha,prior.savedSha256);
}
const report={schema:'ressectionlab-data-hidden-app-v1',input,inputSha256:inputSha,output,rendererIndexSha256:digest(path.join(repo,'desktop/dist/index.html')),
  sourceBindings:Object.fromEntries(['desktop/electron/main.cjs','src/resectionlab/desktop_bridge.py','src/resectionlab/workspace_bundle.py','src/resectionlab/surgical_engine.py'].map(file=>[file,digest(path.join(repo,file))])),
  pid:process.pid,windowEverVisible:false,checks:[],requests:[],startedUtc:new Date().toISOString(),passed:false};
// Chromium process locks require POSIX links, unsupported by ExFAT. Keep this
// small disposable runtime profile local; all imaging, transfers and evidence
// use the external drive. Remove it after the app exits.
const profile=fs.mkdtempSync(path.join(require('node:os').tmpdir(),'rslab-data-profile-'));
app.setPath('userData',profile);report.runtimeProfile=profile;
const sidecarModule=require('./sidecar.cjs'),OriginalSidecar=sidecarModule.Sidecar;
const sidecars=new Set();
sidecarModule.Sidecar=class extends OriginalSidecar {
  constructor(options){super(options);sidecars.add(this);}
};
let exiting=false;
async function finish(){
  if(exiting)return;exiting=true;
  try{
    await Promise.all([...sidecars].map(sidecar=>sidecar.stop()));
  }catch(error){
    report.passed=false;report.cleanupError=String(error);
  }
  app.quit();
}
app.once('will-quit',()=>{
  try{fs.rmSync(profile,{recursive:true,force:true});}
  catch(error){report.passed=false;report.cleanupError=String(error);}
  report.finishedUtc=new Date().toISOString();
  fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify({passed:report.passed,error:report.error??report.cleanupError,output}));
  if(!report.passed)app.exit(1);
});
process.argv.push('--case',input,'--automation-hidden');
dialog.showSaveDialog=async()=>({canceled:false,filePath:saved});dialog.showMessageBox=async()=>({response:1});
let selection=saved;dialog.showOpenDialog=async()=>({canceled:false,filePaths:[selection]});
const originalHandle=ipcMain.handle.bind(ipcMain);
let ctaPayload;
let resectPayload;
const resectAssets=new Map();
let nativeVolume;
const ixiT1Layers=new Map();
function inspectResectAsset(asset,kind,bytes){
  assert.equal(bytes.byteLength,asset.byteLength);
  const sha256=crypto.createHash('sha256').update(bytes).digest('hex');
  assert.equal(sha256,String(asset.sha256).replace(/^sha256:/,''));
  if(kind==='mask'){
    let positiveVoxels=0;
    for(let i=0;i<bytes.length;i++)positiveVoxels+=bytes[i];
    return {sha256,positiveVoxels};
  }
  const bits=Buffer.from(bytes.buffer,bytes.byteOffset,bytes.byteLength),length=bits.length/4,count=Math.min(4096,length);
  let hash=2166136261;
  for(let i=0;i<count;i++){
    const index=count>1?Math.floor(i*(length-1)/(count-1)):0;
    hash=Math.imul(hash^bits.readUInt32LE(index*4),16777619)>>>0;
  }
  return {sha256,sourceSignal:`sampled-f32:${length}:${hash.toString(16).padStart(8,'0')}`};
}
ipcMain.handle=(name,handler)=>originalHandle(name,async(...args)=>{
  const started=performance.now();
  try{const result=await handler(...args);
    if(targetId&&['research:startupWorkspace','research:openListedWorkspace','research:openWorkspace'].includes(name)&&result)ctaPayload=result;
    if(resectPair&&name==='research:startupWorkspace'&&result){
      resectPayload=result;
      for(const volume of [result,...result.workspaceSession.displaySeries.map(stage=>stage.volume)]){
        resectAssets.set(volume.mri.assetId,{asset:volume.mri,kind:'image'});
        for(const layer of volume.compartments)resectAssets.set(layer.array.assetId,{asset:layer.array,kind:'mask'});
      }
    }
    if(resectPair&&name==='research:readAsset'){
      const transfer=resectAssets.get(args[1]);
      if(transfer)transfer.receipt=inspectResectAsset(transfer.asset,transfer.kind,result);
    }
    if(nativeStudy&&name==='research:anatomyView'&&result)nativeVolume=result;
    if(ixiStudy&&name==='research:anatomyView'&&args[1]?.seriesId==='t1'){
      for(const layer of result.compartments)ixiT1Layers.set(layer.name,layer);
    }
    const receipt={operation:name,command:args[1]?.command,ms:performance.now()-started,
      stateId:result?.view?.stateId,caseHash:result?.caseHash,reason:result?.reason,status:result?.status};
    if(replayOnly&&name==='research:surgicalRehearsal'&&result?.view){
      const {view,playback}=result;
      receipt.position=result.position;
      receipt.slots=Object.fromEntries(Object.entries(view.slots).map(([id,{instrument,inserted,position,quaternion}])=>[id,{instrument,inserted,position,quaternion}]));
      receipt.effects={targetMm3:view.metrics.targetRemovedMm3,nonTargetMm3:view.metrics.nonTargetRemovedMm3,
        removedCells:view.removed?.length,elapsedSeconds:view.metrics.elapsedSeconds};
      if(replayOpening)Object.assign(receipt.effects,{byLabelMm3:view.metrics.removedByLabelMm3,
        drillMm3:view.metrics.drillRemovedVolumeMm3});
      if(playback){
        const {position,actionIndex,frameIndex,frameCount,frameStateId,observedTimeSeconds,actionElapsedSeconds,actionDurationSeconds,isActionBoundary,readOnly}=playback;
        const frames=playback.frames,start=frames[0].observedTimeSeconds,end=frames[frameCount-1].observedTimeSeconds;
        const timedInterior=frame=>frame.observedTimeSeconds>start&&frame.observedTimeSeconds<end;
        receipt.playback={position,actionIndex,frameIndex,frameCount,frameStateId,observedTimeSeconds,actionElapsedSeconds,actionDurationSeconds,
          isActionBoundary,readOnly,kind:frames[frameIndex].kind,
          firstTimedInteriorIndex:frames.findIndex(timedInterior),lastTimedInteriorIndex:frames.findLastIndex(timedInterior)};
      }
    }
    report.requests.push(receipt);return result;}
  catch(error){report.requests.push({operation:name,command:args[1]?.command,ms:performance.now()-started,error:String(error)});throw error;}
});
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
app.on('browser-window-created',(_event,window)=>{
  window.on('show',()=>{report.windowEverVisible=true;window.hide();});
  window.webContents.once('did-finish-load',async()=>{
    window.webContents.setBackgroundThrottling(false);window.setContentSize(1440,950);
    const js=code=>window.webContents.executeJavaScript(code,true);
    const wait=async expression=>{const start=Date.now();while(Date.now()-start<180000){if(await js(expression))return;await delay(100);}throw Error('Timeout '+expression+'\n'+await js('document.body.innerText'));};
    const click=async label=>{await js(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(label)}||b.getAttribute('aria-label')===${JSON.stringify(label)}||b.querySelector('span')?.textContent.trim()===${JSON.stringify(label)});if(!b||b.disabled)throw Error('Missing enabled '+${JSON.stringify(label)});for(let p=b.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;b.click()})()`);await delay(100);};
    const ready=async()=>{
      let stable=0;const started=Date.now();
      while(Date.now()-started<180000){
        const valid=await js(`(()=>{const v=document.querySelector('.rl-viewer');return !!v?.querySelector('canvas')&&v.dataset.sourceCaseHash?.startsWith('sha256:')&&v.dataset.rendererReady==='true'&&v.dataset.surfacesPending==='0'&&!document.querySelector('.viewer-loading,.rl-viewer-preparing')&&!document.body.innerText.includes('No case open')})()`);
        stable=valid?stable+1:0;if(stable>=10){await js(`new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))`);return;}
        await delay(100);
      }
      throw Error('Source rendering did not settle: '+await js('document.body.innerText'));
    };
    const screenshot=async()=>{
      assert.equal(window.isVisible(),false);
      const image=await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true});
      assert(!image.isEmpty(),'Hidden capture is empty');
      return image;
    };
    const capture=async name=>{
      await ready();
      const image=await screenshot();
      fs.writeFileSync(path.join(output,name+'.png'),image.toPNG());
      return image;
    };
    const completedFrame=async (stateId,since=0)=>{
      const state=JSON.stringify(stateId);
      await wait(`(()=>{const p=window.__dataRebuildFrames,s=${state},v=p?.publications.findLast(v=>v.surgicalStateId===s&&v.at>=${since});return document.querySelector('.surgical-panel')?.dataset.stateId===s&&!!v&&p.gpu.some(g=>g.serial===v.frameSerial&&g.surgicalStateId===s&&g.sourceCaseHash===v.sourceCaseHash&&g.completed&&g.at<=v.at)})()`);
      return js(`(()=>{const p=window.__dataRebuildFrames,v=p.publications.findLast(v=>v.surgicalStateId===${state});return{publication:v,gpu:p.gpu.findLast(g=>g.serial===v.frameSerial&&g.surgicalStateId===v.surgicalStateId&&g.completed)}})()`);
    };
    const inspectRecordedReplay=async()=>{
      const finalId=process.env.DATA_REBUILD_EXPECT_STATE_ID;
      const effectId=process.env.DATA_REBUILD_EXPECT_OPERATING_STATE_ID;
      const startId=process.env.DATA_REBUILD_REPLAY_START_STATE_ID;
      const total=Number(process.env.DATA_REBUILD_EXPECT_REPLAY_ACTIONS);
      const action=Number(process.env.DATA_REBUILD_REPLAY_ACTION);
      for(const id of [finalId,effectId,startId])assert.match(id??'',/^sha256:[a-f0-9]{64}$/);
      assert(Number.isSafeInteger(total)&&Number.isSafeInteger(action)&&action>=0&&action<total-1);
      const button=label=>`[...document.querySelectorAll('.surgical-panel button')].some(b=>b.textContent.trim()===${JSON.stringify(label)}&&!b.disabled)`;
      const enabledTimeline=`!!document.querySelector('[aria-label="Surgical timeline"]:enabled')`;
      const domState=`document.querySelector('.surgical-panel')?.dataset.stateId`;
      const observation=`document.querySelector('[aria-label="Observed action frames"]')`;
      const checkSource=(receipt,frame)=>{
        assert.equal(receipt.caseHash,frame.publication.sourceCaseHash);
        if(report.replaySource)assert.equal(frame.gpu.sourceSignalFingerprint,report.replaySource.sourceSignal);
      };
      const boundary=async(id,position,since)=>{
        const frame=await completedFrame(id,since);
        const receipt=report.requests.findLast(row=>row.stateId===id&&row.position===position&&!row.playback&&row.effects);
        assert(receipt,`Missing engine boundary ${position} receipt for ${id}`);
        checkSource(receipt,frame);
        return {receipt,frame};
      };
      const animateFrame=async({id,frameIndex},since)=>{
        const receipt=report.requests.findLast(row=>row.stateId===id&&row.playback?.frameIndex===frameIndex&&row.playback.actionIndex===action);
        assert(receipt&&['seekFrame','seekPrefixFrame'].includes(receipt.command));
        const p=receipt.playback;
        assert.equal(p.frameStateId,id);
        assert.equal(p.position,action);
        assert.equal(p.readOnly,true);
        assert.equal(p.isActionBoundary,false);
        assert(p.frameIndex>0&&p.frameIndex<p.frameCount-1);
        assert(p.actionElapsedSeconds>0&&p.actionElapsedSeconds<p.actionDurationSeconds);
        assert.equal(receipt.slots.right.instrument,'cusa');
        assert.equal(receipt.slots.right.inserted,true);
        const frame=await completedFrame(id,since);
        checkSource(receipt,frame);
        return {receipt,frame};
      };
      const timestamp=()=>js('performance.now()');
      let opening;
      if(replayOpening){
        const archived=JSON.parse(execFileSync('unzip',['-p',input,'manifest.json'],{encoding:'utf8',maxBuffer:64*1024*1024}));
        const record=archived.artifacts.surgical_toolbox,history=record.timeline[record.branchId];
        const openingPos=replayOpening.contract.opening_actions;
        assert(Number.isSafeInteger(openingPos)&&openingPos>0&&openingPos<action);
        assert.equal(archived.case_semantic_hash,replayOpening.contract.case_hash);
        assert.equal(history[0].stateBefore,replayOpening.initial_state_id);
        assert.equal(history[action].action.operation,'apply');
        assert.equal(history[total-1].action.operation,'withdraw');
        assert.equal(history[action].stateBefore,startId);
        assert.equal(history[action].stateAfter,effectId);
        assert.equal(history[total-1].stateAfter,finalId);
        assert.equal(history.length,total);
        const structural=archived.source_refs.find(ref=>ref.source_id==='structural');
        assert.equal(structural?.provenance,'observed');
        assert.match(structural.sha256,/^[a-f0-9]{64}$/);
        await click('Anatomy');
        await wait(`document.querySelectorAll('.anatomy-layer-identity').length===${Object.keys(archived.metadata.anatomy_layers).length}`);
        await gpuSourceReady();
        const source=await sourceState();
        assert.equal(source.caseHash,archived.case_semantic_hash);
        assert.match(source.modality,/CTA/i);
        assert.match(source.sourceSignal,/^sampled-f32:/);
        const layers=await js(`[...document.querySelectorAll('.anatomy-layer-identity')].map(e=>({
          label:e.querySelector('strong').textContent,provenance:e.querySelector('.anatomy-layer-origin').textContent
        }))`);
        for(const [id,layer] of Object.entries(archived.metadata.anatomy_layers)){
          assert(source.visible.includes(id),`Missing active native source layer ${id}`);
          if(layer.provenance==='simulated')assert.match(layers.find(row=>row.label===layer.label)?.provenance??'',/generated research geometry/i);
        }
        assert(layers.some(row=>/skull/i.test(row.label)&&/estimate/i.test(row.provenance)));
        assert(layers.some(row=>/arterial/i.test(row.label)&&/source-provided annotation/i.test(row.provenance)));
        assert(layers.some(row=>/venous/i.test(row.label)&&/source-provided annotation/i.test(row.provenance)));
        report.replaySource={...source,structural,layers};
        await capture('acquired-cta-source-overlays');
        opening={position:openingPos,stateId:history[openingPos-1].stateAfter};
      }
      let since=await timestamp();
      await click('Rehearse');
      await wait(`${enabledTimeline}||${button('Verify replay for planning')}||${button('Verify saved replay')}`);
      if(await js(button('Verify replay for planning')))await click('Verify replay for planning');
      else if(await js(button('Verify saved replay')))await click('Verify saved replay');
      await wait(enabledTimeline);
      const final=await boundary(finalId,total,since);
      assert.equal(final.receipt.slots.right.instrument,'cusa');
      assert.equal(final.receipt.slots.right.inserted,false);
      assert(final.receipt.effects.targetMm3>0);
      if(replayOpening){
        assert.equal(finalId,replayOpening.final_state_id);
        assert.equal(effectId,replayOpening.target_effect_state_id);
        assert.equal(final.receipt.effects.targetMm3,replayOpening.final_metrics.targetRemovedMm3);
        assert.equal(final.receipt.effects.nonTargetMm3,replayOpening.final_metrics.nonTargetRemovedMm3);
        assert.deepEqual(final.receipt.effects.byLabelMm3,replayOpening.final_metrics.removedByLabelMm3);
        assert.deepEqual(final.receipt.slots.right.position,replayOpening.final_slots.right.position);
        assert(Object.values(final.receipt.slots).every(slot=>!slot.inserted));
      }
      report.recordedReplay={final};
      await capture('withdrawn-endpoint');

      since=await timestamp();
      await click('Previous recorded state');
      const effect=await boundary(effectId,total-1,since);
      assert.equal(effect.receipt.slots.right.instrument,'cusa');
      assert.equal(effect.receipt.slots.right.inserted,true);
      assert.equal(effect.receipt.effects.targetMm3,final.receipt.effects.targetMm3);
      assert.equal(effect.receipt.effects.nonTargetMm3,final.receipt.effects.nonTargetMm3);
      if(replayOpening)assert.deepEqual(effect.receipt.effects.byLabelMm3,final.receipt.effects.byLabelMm3);
      report.recordedReplay.effect=effect;
      await capture('inserted-target-effect');

      if(opening){
        since=await timestamp();
        await js(inputScript('Surgical timeline',0));
        const pristine=await boundary(replayOpening.initial_state_id,0,since);
        assert.equal(pristine.receipt.effects.removedCells,0);
        assert.equal(pristine.receipt.effects.targetMm3,0);
        since=await timestamp();
        await js(inputScript('Surgical timeline',opening.position));
        const bone=await boundary(opening.stateId,opening.position,since);
        const effects=bone.receipt.effects;
        assert.equal(effects.removedCells,replayOpening.contract.limits.opening_bone_cells);
        assert.equal(effects.targetMm3,0);
        assert.deepEqual(Object.keys(effects.byLabelMm3),['estimated_skull']);
        assert(effects.drillMm3>0&&Math.abs(effects.drillMm3-effects.byLabelMm3.estimated_skull)<1e-7);
        report.recordedReplay.modeledOpening={before:pristine,after:bone};
        await capture('modeled-skull-opening');
        report.checks.push('Hidden GPU rendered the sealed bone-only opening from the intact starting state');
      }

      since=await timestamp();
      await js(inputScript('Surgical timeline',action));
      const start=await boundary(startId,action,since);
      report.recordedReplay.actionStart=start;
      assert(effect.receipt.effects.targetMm3>start.receipt.effects.targetMm3);
      await js(inputScript('Replay speed',.25));
      since=await timestamp();
      await click('Play replay');
      await wait(`(()=>{
        const el=${observation},id=${domState},i=Number(el?.value);
        const detail=el?.closest('[aria-label="Observed microstep replay"]')?.textContent;
        if(!i||i>=Number(el.max)||!detail?.includes('Action '+(${action}+1)+' · frame '+i+' /'))return false;
        const p=window.__dataRebuildFrames,v=p.publications.findLast(v=>v.surgicalStateId===id&&v.at>=${since});
        if(!v||!p.gpu.some(g=>g.serial===v.frameSerial&&g.surgicalStateId===id&&g.completed&&g.at<=v.at))return false;
        const pause=[...document.querySelectorAll('.surgical-panel button')].find(b=>b.textContent.trim()==='Pause replay'&&!b.disabled);
        if(!pause)return false;
        pause.click();window.__dataRebuildPausedPlayback={id,frameIndex:i};return true;
      })()`);
      const played=await animateFrame(await js('window.__dataRebuildPausedPlayback'),since);
      report.recordedReplay.played=played;
      await wait(`(${button('Play replay')})&&!!${observation}`);

      const {frameIndex,firstTimedInteriorIndex,lastTimedInteriorIndex}=played.receipt.playback;
      assert(firstTimedInteriorIndex>0&&lastTimedInteriorIndex>firstTimedInteriorIndex,'Need two distinct timed interiors to verify scrubbing');
      // A final microstep can share the exact time and state of the boundary.
      const scrubIndex=frameIndex===lastTimedInteriorIndex?firstTimedInteriorIndex:lastTimedInteriorIndex;
      since=await timestamp();
      await js(inputScript('Observed action frames',scrubIndex));
      await wait(`Number(${observation}?.value)===${scrubIndex}&&${domState}!==${JSON.stringify(played.receipt.stateId)}`);
      const scrubbed=await animateFrame({id:await js(domState),frameIndex:scrubIndex},since);
      report.recordedReplay.scrubbed=scrubbed;
      for(const sample of [played,scrubbed])assert(![startId,effectId,finalId].includes(sample.receipt.stateId));
      assert.notEqual(scrubbed.receipt.playback.observedTimeSeconds,played.receipt.playback.observedTimeSeconds);
      const interiors=[played,scrubbed].sort((a,b)=>a.receipt.playback.frameIndex-b.receipt.playback.frameIndex);
      assert(interiors[0].receipt.effects.targetMm3>=start.receipt.effects.targetMm3);
      assert(interiors[1].receipt.effects.targetMm3>=interiors[0].receipt.effects.targetMm3);
      assert(effect.receipt.effects.targetMm3>=interiors[1].receipt.effects.targetMm3);
      report.recordedReplay.interiorTargetIncreaseMm3=interiors[1].receipt.effects.targetMm3-start.receipt.effects.targetMm3;
      await capture('observed-intra-action');

      since=await timestamp();
      await js(inputScript('Surgical timeline',total));
      report.recordedReplay.restored=await boundary(finalId,total,since);
      assert.equal(digest(input),inputSha);
      assert.equal(window.isVisible(),false);
      report.checks.push('Hidden desktop rendered the sealed target-effect and withdrawn action boundaries with actual GPU completion',
        'Play, Pause and manual scrubbing displayed distinct, admitted read-only interiors of a recorded CUSA apply',
        'Returned to the sealed withdrawn endpoint without changing the input package');
      assert.equal(report.windowEverVisible,false);
      report.passed=true;
    };
    const inspectHostLayers=async()=>{
      await click('Anatomy');
      await wait(`document.querySelectorAll('.anatomy-layer').length>=4`);
      const layers=await js(`[...document.querySelectorAll('.anatomy-layer-identity')].map(layer=>({
        name:layer.querySelector('strong').textContent,
        provenance:layer.querySelector('.anatomy-layer-origin').textContent,
      }))`);
      for(const structure of [/brain/i,/skull|calvari/i]){
        assert(layers.some(layer=>structure.test(layer.name)&&/estimate/i.test(layer.provenance)),
          `Expected a visible estimated layer for ${structure}`);
      }
      for(const structure of [/arterial/i,/venous|sinus/i]){
        assert(layers.some(layer=>structure.test(layer.name)&&/^(Source-provided|Supplied) annotation\b/.test(layer.provenance)),
          `Expected a visible supplied annotation for ${structure}`);
      }
      assert(!layers.some(layer=>/synthetic|tumor|target/i.test(layer.name)), 'An anatomy host must not contain a tumor or target layer');
      return layers;
    };
    const set=async(label,value)=>{await js(`(()=>{const el=document.querySelector('[aria-label='+${JSON.stringify(JSON.stringify(label))}+']');if(!el)throw Error('Missing '+${JSON.stringify(label)});for(let p=el.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;Object.getOwnPropertyDescriptor(el.tagName==='SELECT'?HTMLSelectElement.prototype:HTMLInputElement.prototype,'value').set.call(el,${JSON.stringify(value)});el.dispatchEvent(new Event(el.tagName==='SELECT'?'change':'input',{bubbles:true}))})()`);await delay(100);};
    const sourceState=()=>js(`(()=>{
      const viewer=document.querySelector('.rl-viewer'),probe=window.__dataRebuildCtaFrames;
      const frame=probe.frames.findLast(f=>f.sourceCaseHash===viewer.dataset.sourceCaseHash
        &&(f.displayIdentity??'')===(viewer.dataset.sourceDisplayIdentity??'')&&probe.completed.has(probe.key(f)));
      return {caseHash:viewer.dataset.sourceCaseHash,modality:viewer.dataset.sourceModality,
        visible:JSON.parse(viewer.dataset.visibleLayerIds),sourceShape:frame?.sourceShape,
        sourceSignal:frame?.sourceSignalFingerprint,cursor:frame?.cursor,
        appearance:[...document.querySelectorAll('.anatomy-layer')].map(row=>({
          label:row.querySelector('strong').textContent,
          opacity:row.querySelector('input[type=range]').value,
          color:row.querySelector('input[type=color]').value,
        }))};
    })()`);
    const contourControl=`[...document.querySelectorAll('.anatomy-layer input[type=checkbox]')]
      .find(e=>e.getAttribute('aria-label')===${JSON.stringify('Show '+targetLabel)})`;
    const contourVisible=visible=>wait(`(()=>{const control=${contourControl};
      const layers=JSON.parse(document.querySelector('.rl-viewer')?.dataset.visibleLayerIds??'[]');
      return control?.checked===${visible}&&layers.includes(${JSON.stringify(targetId)})===${visible};
    })()`);
    const gpuSourceReady=()=>wait(`(()=>{
      const probe=window.__dataRebuildCtaFrames,viewer=document.querySelector('.rl-viewer');
      return probe.frames.some(f=>probe.completed.has(probe.key(f))&&f.sourceCaseHash===viewer?.dataset.sourceCaseHash
        &&(f.displayIdentity??'')===(viewer?.dataset.sourceDisplayIdentity??''));
    })()`);
    const inspectResectPair=async()=>{
      const readMember=(archive,member)=>JSON.parse(execFileSync('unzip',['-p',archive,member],{encoding:'utf8'}));
      const baseline=fs.realpathSync(path.join(root,resectPair.source_workspace.path));
      assert(baseline.startsWith(root+path.sep));
      assert.equal(digest(baseline),resectPair.source_workspace.sha256);
      const original=readMember(baseline,'manifest.json'),archive=readMember(input,'manifest.json'),workspace=readMember(input,'workspace.json');
      const {workspaceSession:session}=resectPayload;
      const {displaySeries}=session;
      assert.equal(resectPayload.caseId,'RESECT-Case3');
      assert.equal(resectPayload.caseHash,original.case_semantic_hash);
      assert.equal(archive.case_semantic_hash,original.case_semantic_hash);
      assert.equal(archive.array_sha256,original.array_sha256);
      assert.deepEqual(resectPayload.sourceRefs,original.source_refs);
      assert.deepEqual(resectPayload.compartments,[]);
      assert.deepEqual(archive.artifacts,{});
      assert.equal(session.episodeReplay,null);
      assert.equal(displaySeries.length,2);
      assert.equal(workspace.images.length,2);
      const captureGrid=async(volume,since=0)=>{
        const source=JSON.stringify(volume.caseHash);
        await wait(`(()=>{
          const p=window.__dataRebuildCtaFrames,hash=${source};
          return document.querySelector('.rl-viewer')?.dataset.sourceCaseHash===hash&&
            p.frames.some(f=>f.sourceCaseHash===hash&&f.at>=${since}&&p.completed.has(p.key(f)));
        })()`);
        await ready();
        const observed=await js(`(()=>{
          const p=window.__dataRebuildCtaFrames,f=p.frames.findLast(f=>f.sourceCaseHash===${source}&&f.at>=${since}&&p.completed.has(p.key(f)));
          const g=p.completed.get(p.key(f)),viewer=document.querySelector('.rl-viewer');
          return {serial:f.serial,completed:g.completed,caseHash:f.sourceCaseHash,sourceShape:f.sourceShape,sourceSignal:f.sourceSignalFingerprint,
            gpuSignal:g.sourceSignalFingerprint,cursor:f.cursor,visible:JSON.parse(viewer.dataset.visibleLayerIds),
            panes:viewer.querySelectorAll('.rl-viewer-slice-image').length};
        })()`);
        const transfer=resectAssets.get(volume.mri.assetId)?.receipt;
        assert(transfer,'Read and checksum the source bytes actually sent to the renderer');
        assert.deepEqual(observed.sourceShape,volume.shape);
        assert.equal(observed.sourceSignal,transfer.sourceSignal);
        assert.equal(observed.gpuSignal,transfer.sourceSignal);
        assert.equal(observed.panes,3);
        return {...observed,imageArraySha256:transfer.sha256,affine:volume.affine};
      };
      const scan='Inspect scan';
      const timestamp=()=>js('performance.now()');
      const visible=()=>js(`JSON.parse(document.querySelector('.rl-viewer').dataset.visibleLayerIds)`);
      const sliceCrops=()=>js(`[...document.querySelectorAll('.rl-viewer-slice-image')].map(e=>{
        const r=e.getBoundingClientRect();return{x:Math.ceil(r.x),y:Math.ceil(r.y),width:Math.floor(r.width)-2,height:Math.floor(r.height)-2};
      })`);
      const pixelHashes=(image,crops)=>crops.map(rect=>crypto.createHash('sha256').update(image.crop(rect).toBitmap()).digest('hex'));
      await click('Scans');
      const primary=await captureGrid(resectPayload);
      assert.deepEqual(primary.visible,[]);
      const primarySlices=await sliceCrops(),primaryPixels=pixelHashes(await screenshot(),primarySlices);
      report.resectPair={primary:{...primary,sourceSha256:original.source_refs[0].sha256,
        modality:original.metadata.selected_modality,caseRole:original.metadata.data_case.role}};
      for(const stage of ['during','after']){
        const sourceSha=resectPair.sources[stage].sha256,expected=resectPair.stage_visibility[stage];
        const series=displaySeries.find(s=>s.volume.sourceRefs.some(ref=>ref.source_id==='structural'&&ref.sha256===sourceSha));
        assert(series,`Missing transfer bound to the verified ${stage} image`);
        const volume=series.volume;
        const sealed=workspace.images.find(row=>row.source.seriesId===series.seriesId)?.source;
        assert.equal(sealed?.sourceCaseHash,volume.caseHash);
        assert.equal(sealed.sourceFrameHash,series.sourceFrameHash);
        assert.equal(series.referenceCaseHash,primary.caseHash);
        assert.equal(series.annotationKind,'source-provided');
        assert.equal(series.scope,'display-only-native-grid');
        assert.equal(series.planningEligible,false);
        assert.equal(series.evaluationEligible,false);
        assert.equal(series.registration.overlayPermitted,false);
        assert.equal(series.registration.sourceToReferenceRasMm,null);
        assert.equal(volume.metadata.acquisition_stage,`intraoperative-${stage}`);
        assert.equal(volume.metadata.annotationTimeRole,'intraoperative_observed_evaluator_only');
        assert.equal(volume.sourceRefs.find(ref=>ref.source_id==='supplied_target_annotation')?.sha256,expected.annotation_sha256);
        assert.deepEqual(volume.shape,expected.shape);
        assert.equal(volume.compartments.length,1);
        const layer=volume.compartments[0],label=`Observed visible cavity · ${stage} resection`;
        assert.equal(layer.name,label);
        const mask=resectAssets.get(layer.array.assetId)?.receipt;
        assert.equal(mask?.positiveVoxels,expected.positive_voxels);
        assert(Math.abs(layer.volumeMm3-expected.native_volume_mm3)<1e-4*expected.native_volume_mm3);

        let since=await timestamp();
        await js(inputScript(scan,series.seriesId));
        const selected=await captureGrid(volume,since);
        assert.deepEqual(selected.visible,[label]);
        const selectedLabel=await js(`document.querySelector('[aria-label="Inspect scan"]').selectedOptions[0].textContent`);
        assert.match(selectedLabel,/supplied visible cavity · source labels · separate frame$/);
        await click('Anatomy');
        const control=`document.querySelector(${JSON.stringify(`[aria-label="Show ${label}"]`)})`;
        await wait(`${control}?.checked===true`);
        const provenance=await js(`${control}.closest('.anatomy-layer').querySelector('.anatomy-layer-origin').textContent`);
        assert.match(provenance,/supplied annotation|source-provided annotation/i);
        const image=await capture(`resect-case3-${stage}-native-cavity`),crops=await sliceCrops();
        since=await timestamp();
        await js(`${control}.click()`);
        await wait(`${control}.checked===false`);
        const hidden=await captureGrid(volume,since);
        assert.deepEqual(hidden.visible,[]);
        assert.deepEqual(hidden.cursor,selected.cursor);
        const onPixels=pixelHashes(image,crops),offPixels=pixelHashes(await screenshot(),crops);
        const changedMprPanes=onPixels.filter((sha,i)=>sha!==offPixels[i]).length;
        assert(changedMprPanes>0,'The actual source-slice pixels should change when the supplied mask is hidden');
        await js(`${control}.click()`);
        await wait(`${control}.checked===true`);
        assert.deepEqual(await visible(),[label]);
        await click('Scans');
        report.resectPair[stage]={...selected,sourceSha256:sourceSha,sourceFrameHash:series.sourceFrameHash,seriesId:series.seriesId,
          selectedLabel,annotation:{label,provenance,publisherSha256:expected.annotation_sha256,mask,volumeMm3:layer.volumeMm3,changedMprPanes},
          registration:series.registration,annotationTimeRole:volume.metadata.annotationTimeRole};
      }
      const {during,after}=report.resectPair;
      for(const key of ['caseHash','sourceFrameHash','sourceShape','sourceSignal','affine'])assert.notDeepEqual(during[key],after[key]);
      const since=await timestamp();
      await js(inputScript(scan,''));
      const restored=await captureGrid(resectPayload,since);
      assert.deepEqual({...restored,serial:primary.serial},primary);
      assert.deepEqual(pixelHashes(await screenshot(),primarySlices),primaryPixels);
      report.resectPair.restoredPrimary={...restored,identicalMprPanes:primaryPixels.length};
      assert.equal(digest(input),inputSha);
      assert.equal(window.isVisible(),false);
      report.checks.push('Actual During and After controls render independent source grids and distinct checksum-bound ultrasound bytes',
        'Both evaluator-only supplied cavity masks change native source-slice pixels and stay outside the MRI reference',
        'Returning to the original MRI restores its source, cursor and three identical source-slice crops');
      report.passed=!report.windowEverVisible;
    };
    const ctaReadout=async()=>{
      await click('Scans');
      await wait(`!!document.querySelector('.poc-source-facts')`);
      const source=await js(`(()=>{
        const d=[...document.querySelectorAll('.poc-data-panel details')].find(e=>e.querySelector('summary')?.textContent==='Provenance and licenses');
        if(!d)throw Error('Missing source provenance');d.open=true;
        const e=[...d.querySelectorAll('.poc-original')].find(e=>e.querySelector('strong')?.textContent==='structural');
        return {sha256:e?.querySelector('code')?.textContent,provenance:e?.querySelector('span')?.textContent,uri:e?.querySelector('p')?.textContent};
      })()`);
      assert.equal(source.sha256,sourceSha);
      assert.match(source.provenance,/^observed\b/i);
      assert.equal(ctaPayload.sourceRefs.find(ref=>ref.source_id==='structural').sha256,source.sha256);
      assert.equal(ctaPayload.metadata.data_case.observations.primary_image,'baseline_acquired_background_geometry_only');
      const generated=Object.entries(ctaPayload.metadata.anatomy_layers)
        .filter(([,layer])=>layer.provenance==='simulated'||layer.provenance==='generated');
      assert.deepEqual(generated.map(([id,layer])=>[id,layer.kind]),[[targetId,'tumor']]);
      assert(!ctaPayload.artifacts?.surgical_toolbox,'A new contour must not inherit a recorded episode');
      await capture('cta-source-provenance');
      await click('Anatomy');
      await contourVisible(true);
      const layers=await js(`[...document.querySelectorAll('.anatomy-layer-identity')].map(e=>({
        label:e.querySelector('strong').textContent,provenance:e.querySelector('.anatomy-layer-origin').textContent
      }))`);
      assert.match(layers.find(row=>row.label===targetLabel)?.provenance??'',/^Generated research geometry\b/);
      for(const [structure,origin] of [[/brain/i,/estimate/i],[/skull|calvari/i,/estimate/i],
        [/arterial/i,/^Source-provided annotation\b/],[/venous|sinus/i,/^Source-provided annotation\b/]]){
        assert(layers.some(row=>structure.test(row.label)&&origin.test(row.provenance)),`Missing display provenance for ${structure}`);
      }
      await gpuSourceReady();
      return {source,layers,...await sourceState()};
    };
    const inspectCtaRehearsal=async()=>{
      await click('Rehearse');
      const prepare=`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Prepare instrument scene'&&!b.disabled)`;
      const action=`!!document.querySelector('[aria-label="Instrument action"]')`;
      await wait(`${action}||${prepare}`);
      if(await js(prepare))await click('Prepare instrument scene');
      await wait(action);
      const caseHash=JSON.stringify(report.ctaPreview.caseHash);
      const inspect=()=>js(`(async()=>{
        const result=await window.resectionApi.surgicalRehearsal({caseHash:${caseHash},command:'inspect'});
        return {caseHash:result.caseHash,stateId:result.view.stateId,position:result.position,
          targets:result.view.scene.regions.filter(row=>row.target).map(row=>row.name),
          targetRemovedMm3:result.view.metrics.targetRemovedMm3,planningAvailable:result.planningAdmission?.available};
      })()`);
      const before=await inspect();
      assert.equal(before.caseHash,report.ctaPreview.caseHash);
      assert.deepEqual(before.targets,[targetId]);
      assert.equal(before.position,0);
      assert.equal(before.targetRemovedMm3,0);
      assert.equal(before.planningAvailable,false);
      await set('Instrument action','inspect');
      await click('Preview');
      await wait(`document.querySelector('.surgical-panel')?.textContent.includes('Preview · uncommitted')`);
      await capture('cta-rehearsal-inspect');
      await click('Cancel preview');
      const after=await inspect();
      assert.deepEqual(after,before);
      report.ctaRehearsal=after;
      report.checks.push('The exact contour initializes a manual rehearsal, keeps procedure planning unavailable and cancels an inspection preview without changing state');
    };
    const inspectCtaPreview=async()=>{
      if(prior){
        report.ctaPreview=await ctaReadout();
        assert.deepEqual(report.ctaPreview,prior.ctaPreview,'Cold reopen changed the displayed source, target or view');
        await capture('cta-cold-reopened');
        report.checks.push('A fresh process restores the acquired CTA, target overlay and cursor from the saved workspace');
        if(process.env.DATA_REBUILD_CTA_REHEARSE==='1')await inspectCtaRehearsal();
        return;
      }
      const title=path.basename(input,'.ressectionlab');
      await click('Cases');
      const search='Search cases by name, source, anatomy or reported site';
      await wait(`!!document.querySelector('[aria-label="${search}"]')`);
      if(collectionFile){
        const collection=JSON.parse(fs.readFileSync(collectionFile,'utf8'));
        assert.equal(collection.cases.length,collection.generated_scenarios);
        const listed=await js('window.resectionApi.listCases()');
        report.ctaLibrary=collection.cases.map(entry=>{
          const rows=listed.filter(row=>row.title===entry.case_id);
          assert.equal(rows.length,1,`Expected one Cases card for ${entry.case_id}`);
          assert(!rows[0].reviewedExternal,'Development scenarios must retain local Cases status');
          return {caseId:entry.case_id,id:rows[0].id,sourcePerson:entry.person_id};
        });
        assert.equal(new Set(report.ctaLibrary.map(row=>row.id)).size,collection.generated_scenarios);
        report.checks.push('Actual Cases lists each prepared development scenario once with a distinct saved-case ID');
      }
      await js(inputScript(search,title));
      const openLabel=`Open ${title}`;
      await wait(`[...document.querySelectorAll('.poc-case-card')].some(e=>e.getAttribute('aria-label')===${JSON.stringify(openLabel)}&&!e.disabled)`);
      fs.writeFileSync(path.join(output,'cta-cases.png'),(await screenshot()).toPNG());
      const firstOpenRequest=report.requests.length;
      await click(openLabel);
      await wait(`!document.querySelector('.case-library-workspace')`);
      await ready();
      const opened=report.requests.slice(firstOpenRequest).find(row=>row.operation==='research:openListedWorkspace');
      assert(opened?.caseHash&&!opened.error,'The actual Cases card must open the contour package');
      const initial=await ctaReadout();
      assert.equal(initial.caseHash,opened.caseHash);
      assert.match(initial.modality,/CTA/i);
      await click('Show structures only');
      await click('Center on annotations');
      // Keep dense skull and brain visible while revealing the interior target.
      for(const [pattern,opacity] of [[/skull|calvari/i,.13],[/brain/i,.16]]){
        const layer=initial.layers.find(row=>pattern.test(row.label));
        await js(inputScript(`${layer.label} opacity`,opacity));
      }
      for(const [pattern,color] of [[/arterial/i,'#f06455'],[/venous|sinus/i,'#71b7ef']]){
        const layer=initial.layers.find(row=>pattern.test(row.label));
        await js(inputScript(`${layer.label} color`,color));
      }
      await js(`(()=>{const row=(${contourControl}).closest('.anatomy-layer');
        row.querySelector('details').open=true;row.scrollIntoView({block:'end'});
      })()`);
      const setTarget=async visible=>{
        await js(`(()=>{const e=${contourControl};if(e?.checked!==${visible})e.click()})()`);
        await contourVisible(visible);
      };
      const slices=await js(`[...document.querySelectorAll('.rl-viewer-slice-image')].map(e=>{
        const r=e.getBoundingClientRect();return {x:Math.ceil(r.x),y:Math.ceil(r.y),width:Math.floor(r.width)-2,height:Math.floor(r.height)-2};
      })`);
      assert.equal(slices.length,3,'All linked native slices are required');
      await setTarget(false);
      const without=await capture('cta-target-hidden');
      await setTarget(true);
      const withContour=await capture('cta-target-selected');
      const pixels=(image,rect)=>crypto.createHash('sha256').update(image.crop(rect).toBitmap()).digest('hex');
      report.ctaOverlayPixels=slices.map(rect=>({rect,hidden:pixels(without,rect),selected:pixels(withContour,rect)}));
      assert(report.ctaOverlayPixels.some(row=>row.hidden!==row.selected),'The contour must change actual pixels in a linked native slice');
      const selected=await sourceState();
      assert.equal(selected.sourceSignal,initial.sourceSignal);
      report.ctaPreview={...initial,...selected};
      report.checks.push('Cases opens the exact CTA package, distinguishes observed and generated provenance, and the selected contour changes linked slice pixels');
    };
    const nativeScan=async expectedId=>{
      await gpuSourceReady();
      const selected=await js(`(()=>{
        const shell=document.querySelector('.native-workspace'),viewer=document.querySelector('.rl-viewer');
        return {series:shell?.dataset.nativeSeries,display:JSON.parse(viewer.dataset.sourceDisplayIdentity),
          panes:document.querySelectorAll('.rl-viewer-slice-image').length};
      })()`);
      const series=nativeStudy.series.find(row=>row.id===selected.series);
      assert(series&&nativeVolume,'The app must load a declared display series');
      if(expectedId)assert.equal(series.id,expectedId);
      assert.equal(selected.panes,3);
      assert.equal(selected.display[1],series.id);
      assert.equal(selected.display[2],nativeVolume.mri.sha256);
      assert.equal(selected.display[3],nativeStudy.snapshot_id);
      const rendered=await sourceState();
      assert.deepEqual(rendered.sourceShape,nativeVolume.shape);
      assert(rendered.sourceSignal);
      const surfaces=nativeVolume.compartments.map(layer=>({
        id:layer.name,kind:layer.kind,sourceSeries:layer.sourceSeries,alignment:layer.alignment,
        volumeMm3:layer.volumeMm3,maskSha256:layer.mask?.sha256,
        sourceMeshSha256:layer.surface?.source?.sha256,displayMeshSha256:layer.surface?.positions?.sha256,
        diagnostics:layer.surfaceDiagnostics,
      }));
      return {patientId:nativeStudy.patient_id,snapshotId:nativeStudy.snapshot_id,id:series.id,
        modality:series.modality,displayNativeGrid:series.data.shape,publisherSource:series.publisher_source??null,
        gridSampling:series.display_sampling_ijk??null,rendered,surfaces,affine:nativeVolume.affine,
        originalSha256:series.original.sha256,sourceArraySha256:selected.display[2],registrationStatus:nativeVolume.registrationStatus,
        unavailableLayers:nativeVolume.unavailableLayers};
    };
    const inspectNativeStudy=async()=>{
      await click('Scans');
      report.nativeStudy=await nativeScan();
      await capture('native-study-scans');
      await click('Anatomy');
      const anatomy=await js(`[...document.querySelectorAll('.native-structure-row')].map(row=>({
        id:row.dataset.structureId,label:row.dataset.structureLabel,visible:!row.classList.contains('is-hidden')
      }))`);
      const declared=[...nativeStudy.layers,...(nativeStudy.tractography??[])];
      assert.deepEqual(anatomy.map(row=>row.id).sort(),declared.map(row=>row.id).sort(),
        'The app must show the study’s declared anatomy inventory');
      report.nativeStudy.anatomy=anatomy;
      if(process.env.DATA_REBUILD_STRUCTURES==='1')await click('Show structures only');
      await capture('native-study-anatomy');
      assert.equal(digest(input),inputSha);
      assert.equal(window.isVisible(),false);
      report.checks.push('The actual native viewer renders the manifest-bound scan and all three source panes');
      report.passed=!report.windowEverVisible;
    };
    const inspectIxiStudy=async()=>{
      assert.equal(nativeStudy.reference_series,'t1');
      assert.equal(nativeStudy.tractography?.length??0,0);
      const select=async id=>{
        await wait(studyScanEnabledScript(id));
        await js(studyScanSelectionScript(id));
        await wait(`document.querySelector('.native-workspace')?.dataset.nativeSeries===${JSON.stringify(id)}`);
        await ready();
      };
      const expectedLabels=['brain-stem','left cerebellum cortex','right cerebellum cortex',
        'left lateral ventricle','right lateral ventricle','3rd ventricle','4th ventricle',
        'left thalamus','right thalamus','left cerebral cortex','right cerebral cortex'];
      const regions=expectedLabels.map(label=>{
        const layer=nativeStudy.layers.find(row=>row.label===label&&row.source_series==='t1');
        assert(layer,`Missing T1-derived layer ${label}`);
        return layer;
      });
      await click('Anatomy');
      await select('t1');
      await js(`[...document.querySelectorAll('.anatomy-presets button')].find(b=>b.textContent.trim()==='Structures').click()`);
      await click('Show structures only');
      const opacity=async(label,value)=>{
        await js(studyLayerSelectionScript([label]));
        await js(inputScript(`${label} opacity`,value));
        await click('Close layer details');
      };
      await opacity(nativeStudy.layers.find(layer=>layer.id==='brain').label,.13);
      await ready();await gpuSourceReady();
      const before=await sourceState(),unmarked=await screenshot();
      for(const layer of regions){
        const selected=`JSON.parse(document.querySelector('.rl-viewer')?.dataset.visibleLayerIds??'[]').includes(${JSON.stringify(layer.id)})`;
        if(!await js(selected)){
          const button=`document.querySelector(${JSON.stringify('[data-structure-id="'+layer.id+'"] .native-structure-visibility')})`;
          if(await js(`${button}?.getAttribute('aria-pressed')==='true'`)){
            await js(studyVisibilityToggleScript('Hide '+layer.label));
            await wait(`${button}?.getAttribute('aria-pressed')==='false'`);
          }
          await js(studyVisibilityToggleScript('Show '+layer.label));
          await wait(selected);
        }
      }
      await wait(`!document.querySelector('[data-layer-pending="true"]')`);
      await ready();
      for(const region of regions.filter(layer=>layer.label.includes('cerebral cortex'))){
        await opacity(region.label,.22);
      }
      const marked=await capture('ixi-t1-structures');
      const t1=await nativeScan('t1');
      assert.equal(t1.rendered.sourceSignal,before.sourceSignal);
      assert.deepEqual(t1.rendered.cursor,before.cursor);
      const crops=await js(`[...document.querySelectorAll('.rl-viewer-slice-image,.rl-viewer-anatomy-image')].map(e=>{
        const r=e.getBoundingClientRect();return {x:Math.ceil(r.x),y:Math.ceil(r.y),
          width:Math.floor(r.width)-2,height:Math.floor(r.height)-2,anatomy:e.classList.contains('rl-viewer-anatomy-image')};
      })`);
      const pixels=(image,rect)=>crypto.createHash('sha256').update(image.crop(rect).toBitmap()).digest('hex');
      const changed=crops.map(({anatomy,...rect})=>({anatomy,changed:pixels(unmarked,rect)!==pixels(marked,rect)}));
      assert.equal(changed.length,4);
      assert(changed.some(row=>row.anatomy&&row.changed)&&changed.some(row=>!row.anatomy&&row.changed));
      report.ixi={t1,changed,regions:regions.map(layer=>{
        const observed=ixiT1Layers.get(layer.id);
        assert(t1.rendered.visible.includes(layer.id)&&observed?.sourceSeries==='t1');
        assert(observed.surfaceDiagnostics?.triangles>0&&observed.qa?.voxelCount>0);
        return {id:layer.id,label:layer.label,sourceSeries:observed.sourceSeries,
          voxels:observed.qa.voxelCount,surface:observed.surfaceDiagnostics};
      }),sources:[]};
      const sourceDisplay=JSON.parse(await js(`document.querySelector('.rl-viewer').dataset.sourceDisplayIdentity`));
      for(const id of ['mra','dwi-b0','dwi-fa']){
        const series=nativeStudy.series.find(row=>row.id===id);
        assert(series&&nativeStudy.transforms[id].status==='unreviewed');
        await select(id);
        const rendered=await nativeScan(id);
        assert.deepEqual(rendered.rendered.sourceShape,series.data.shape);
        assert(['independent source frame','alignment preview'].includes(rendered.registrationStatus));
        if(id==='mra')await capture('ixi-mra-source');
        if(id==='dwi-fa'){
          assert.equal(series.provenance.not_acquired_contrast,true);
          assert.match(series.label,/uncorrected tensor diagnostic/i);
          assert(await js(`document.body.innerText.includes('FA · uncorrected tensor diagnostic')`));
        }
        const raw=await js(`(async()=>{
          const v=await window.resectionApi.anatomyView(${JSON.stringify({
            collectionId:sourceDisplay[0],snapshotId:nativeStudy.snapshot_id,seriesId:id,layers:id==='mra'?['brain','vessels']:['brain'],
            original:true,patientModel:false,previewAlignment:false,vesselScope:'all'
          })});
          return {registration:v.registrationStatus,shape:v.shape,affine:v.affine,sha256:v.mri.sha256,
            layers:v.compartments.map(layer=>({id:layer.name,source:layer.sourceSeries}))};
        })()`);
        assert.equal(raw.registration,'independent source frame');
        assert.deepEqual(raw.shape,series.data.shape);
        assert(raw.affine.every((row,i)=>row.every((value,j)=>Math.abs(value-series.affine[i][j])<=1e-5)));
        assert.equal(raw.sha256,rendered.sourceArraySha256);
        assert(raw.layers.every(layer=>layer.source===id));
        report.ixi.sources.push({id,label:series.label??series.modality,rendered,raw});
      }
      assert.equal(digest(input),inputSha);
      assert.equal(window.isVisible(),false);
      report.checks.push('Actual T1 source panes and 3D view show positive model-derived regional anatomy');
      report.checks.push('Acquired MRA and diffusion b0 and the derived FA diagnostic retain exact independent source arrays; no reviewed registration or tractography is present');
      report.passed=!report.windowEverVisible;
    };
    const inspectIslesPair=async()=>{
      await click('Scans');
      const select=async id=>{
        await wait(studyScanEnabledScript(id));
        await js(studyScanSelectionScript(id));
        await wait(`document.querySelector('.native-workspace')?.dataset.nativeSeries===${JSON.stringify(id)}`);
        await ready();
        return nativeScan(id);
      };
      const structures=async()=>{
        await click('Anatomy');
        await js(`[...document.querySelectorAll('.anatomy-presets button')].find(b=>b.textContent.trim()==='Structures').click()`);
        await click('Show structures only');
        const context=nativeStudy.layers.filter(layer=>nativeVolume.compartments.some(displayed=>displayed.name===layer.id));
        if(context.some(layer=>layer.kind==='skull'))await set('Skull view','transparent');
        const brain=context.find(layer=>layer.kind==='brain');
        if(brain){
          await js(studyLayerSelectionScript([brain.label]));
          await js(inputScript(`${brain.label} opacity`,.16));
          await click('Close layer details');
        }
      };
      const cowId='publisher-estimated-cow';
      const cow=nativeStudy.layers.find(layer=>layer.id===cowId);
      assert(cow&&cow.source_series==='cta-ncct-space'&&cow.role==='context');
      assert.equal(cow.provenance.kind,'image_derived_estimate');
      assert.equal(cow.provenance.evidence_use,'context_only');
      assert.match(cow.label,/publisher automated/i);
      const ncct=await select('ncct');
      await structures();
      await capture('isles-ncct');
      const cta=await select('cta-ncct-space');
      assert.equal(ncct.modality,'CT');assert.equal(cta.modality,'CTA');
      assert.equal(ncct.registrationStatus,'patient frame');
      assert.equal(cta.registrationStatus,'patient frame');
      assert.notEqual(ncct.originalSha256,cta.originalSha256);
      assert.notEqual(ncct.sourceArraySha256,cta.sourceArraySha256);
      assert.notEqual(ncct.rendered.sourceSignal,cta.rendered.sourceSignal);
      assert.deepEqual(ncct.rendered.sourceShape,cta.rendered.sourceShape);
      assert(cta.affine.every((row,i)=>row.every((value,j)=>Math.abs(value-ncct.affine[i][j])<=1e-5)));
      const ncctCow=ncct.surfaces.find(layer=>layer.id===cowId);
      const ctaCow=cta.surfaces.find(layer=>layer.id===cowId);
      assert(ncctCow?.maskSha256&&ctaCow?.maskSha256,'Both linked scans must show the publisher CoW');
      assert.equal(ctaCow.maskSha256,ncctCow.maskSha256);
      assert.equal(ctaCow.sourceSeries,'cta-ncct-space');
      assert.equal(ctaCow.alignment,'source-provided');
      await structures();
      assert.equal(await js(studyLayerSelectionScript([cow.label])),cow.label);
      await click(`Focus ${cow.label}`);
      await wait(`document.querySelector('.rl-viewer')?.dataset.selectedStructure===${JSON.stringify(cowId)}`);
      const attribution=await js(`document.querySelector('.layer-inspector')?.textContent`);
      assert.match(attribution,/publisher automated/i);
      assert.match(attribution,/image derived estimate/i);
      await capture('isles-registered-cta-cow');
      report.islesPair={ncct,cta,cow:{...ctaCow,label:cow.label,publisherMaskSha256:cow.provenance.source.sha256}};
      const backToNcct=await select('ncct');
      assert.equal(backToNcct.sourceArraySha256,ncct.sourceArraySha256);
      assert.equal(backToNcct.rendered.sourceSignal,ncct.rendered.sourceSignal);
      report.checks.push('Actual scan selection renders distinct NCCT and registered CTA in the same frame with the same publisher-automated CoW mask');
      if(nativeStudy.series.some(series=>series.id==='native-cta')){
        const native=await select('native-cta');
        assert.equal(native.modality,'CTA');
        assert.equal(native.registrationStatus,'independent source frame');
        assert.notEqual(native.originalSha256,cta.originalSha256);
        assert.notEqual(native.sourceArraySha256,cta.sourceArraySha256);
        assert(!native.surfaces.some(layer=>layer.id===cowId));
        await wait(`!!document.querySelector('.patient-model-summary[data-frame="independent"]')`);
        await capture('isles-native-cta-independent');
        report.islesPair.native=native;
        report.checks.push('The optional original native CTA opens independently and carries no transferred CoW overlay');
      }
      assert.equal(digest(input),inputSha);
      assert.equal(window.isVisible(),false);
      report.passed=!report.windowEverVisible;
    };
    try{
      if(targetId||nativeStudy||resectPair||replayOpening)await js(`(()=>{
        const key=frame=>JSON.stringify([frame.sourceCaseHash,frame.displayIdentity,frame.serial]);
        const probe=window.__dataRebuildCtaFrames={frames:[],completed:new Map(),key};
        document.addEventListener('anatomy-frame',e=>{probe.frames.push(e.detail);if(probe.frames.length>32)probe.completed.delete(key(probe.frames.shift()))});
        document.addEventListener('anatomy-gpu',e=>{if(e.detail.completed)probe.completed.set(key(e.detail),e.detail)});
      })()`);
      if(replayOnly||process.env.DATA_REBUILD_REQUIRE_GPU_STATE==='1')await js(`(()=>{
        const p=window.__dataRebuildFrames={gpu:[],publications:[]};
        document.addEventListener('anatomy-gpu',e=>{p.gpu.push({...e.detail,at:performance.now()});if(p.gpu.length>64)p.gpu.shift()});
        document.addEventListener('anatomy-surgical-display',e=>{p.publications.push({...e.detail,at:performance.now()});if(p.publications.length>32)p.publications.shift()});
      })()`);
      if(process.env.DATA_REBUILD_EXPECT_FAILURE==='1'){
        await wait(`/Bundle must contain|INVALID_BUNDLE_MEMBERS|Reopened case differs|CASE_HASH_MISMATCH/.test(document.body.innerText)`);
        assert.equal(await js(`!!document.querySelector('.rl-viewer')?.dataset.sourceCaseHash`),false);
        report.errorBody=await js('document.body.innerText');assert.equal(window.isVisible(),false);
        fs.writeFileSync(path.join(output,'expected-safe-rejection.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());
        report.checks.push('Malformed native fixture visibly rejected without fallback source anatomy');report.passed=true;return;
      }
      const started=performance.now();await ready();report.coldReadyMs=performance.now()-started;
      if(islesPair){await inspectIslesPair();return;}
      if(ixiStudy){await inspectIxiStudy();return;}
      if(nativeStudy){await inspectNativeStudy();return;}
      if(replayOnly){await inspectRecordedReplay();return;}
      if(resectPair){await inspectResectPair();return;}
      if(process.env.DATA_REBUILD_STRUCTURES==='1'){
        await click('Show structures only');await ready();
        report.checks.push('Explicit structures-only 3D display with unchanged full native source slices');
      }
      await capture('01-native-source');report.checks.push('Production importer and renderer open actual external native package');
      if(targetId){
        await inspectCtaPreview();
        if(prior){
          assert.equal(digest(input),inputSha);
          assert.equal(window.isVisible(),false);
          report.passed=!report.windowEverVisible;
          return;
        }
      }
      if(process.env.DATA_REBUILD_EXPECT_ACQUIRED_HOST==='1'){
        report.acquiredHost={layers:await inspectHostLayers()};
        await capture('01a-host-anatomy-provenance');
        report.checks.push('Actual CTA host displays estimated brain and CT skull with separate supplied arterial and venous layers');
      }
      if(process.env.DATA_REBUILD_EXPECT_CONDITIONAL==='1'){
        assert((await js('document.body.innerText')).includes('Conditional research stage: scalp exposure is assumed; dural opening is not simulated.'));
        report.checks.push('Conditional superficial/dural assumptions and baseline CTA scope are visible');
        await click('Anatomy');await wait(`document.body.innerText.includes('Source-provided annotation. Derived union of supplied source classes')`);
        const body=await js('document.body.innerText');
        assert(body.includes('Supplied Arterial Annotation Union')&&body.includes('Supplied Venous Sinus Annotation Union'));
        await capture('01a-component-provenance');
        report.checks.push('Supplied arterial/sinus class unions retain source annotation provenance inside the synthetic case');
      }
      if(process.env.DATA_REBUILD_EXPECT_HEALTHY_EXPOSURE==='1'){
        const body=await js('document.body.innerText');
        assert(body.includes('Conditional exposed-brain geometry task: scalp exposure, a local skull opening and dural opening are assumed and not simulated.'));
        assert(body.includes('T1/MRA remain baseline references; skull CT is unavailable.'));
        assert(!body.includes('The acquired CTA remains a baseline reference.'));
        report.checks.push('Actual healthy task declares unmodeled exposure, baseline T1/MRA and unavailable skull CT');
      }
      if(process.env.DATA_REBUILD_FUSION==='1'){
        await click('Scans');await wait(`!!document.querySelector('[aria-label="Blend registered modality"]')`);
        const options=await js(`[...document.querySelector('[aria-label="Blend registered modality"]').options].filter(o=>o.value).map(o=>({value:o.value,label:o.textContent}))`);
        assert.equal(options.length,1,'Only independently reviewed derived MRA may fuse; original native MRA stays separate');
        await set('Blend registered modality',options[0].value);await ready();
        await wait(`!!document.querySelector('.rl-fusion-readout')`);await capture('01b-reviewed-registered-fusion');
        assert((await js('document.body.innerText')).includes('unresolved quantitative accuracy'));
        report.checks.push('Actual UI blends the reviewed registered MRA on primary T1 with alignment limits visible');
      }
      if(process.env.DATA_REBUILD_USE_CATALOG==='1'){
        const listedAt=performance.now();
        const listed=await js('window.resectionApi.listCases()');
        const reviewed=listed.filter(row=>row.reviewedExternal);
        report.catalog={listedRows:listed.length,reviewedExternalRows:reviewed.length,listMs:performance.now()-listedAt,
          releaseSha256:digest(path.join(root,'catalog/release.json'))};
        if(process.env.DATA_REBUILD_EXPECT_CATALOG_COUNT){
          const expected=Number(process.env.DATA_REBUILD_EXPECT_CATALOG_COUNT);
          assert(Number.isSafeInteger(expected)&&expected>0);assert.equal(reviewed.length,expected);
        }
        await click('Cases');await wait(`!!document.querySelector('.poc-case-card')`);
        report.catalog.renderedCards=await js(`document.querySelectorAll('.poc-case-card').length`);
        await capture('01a-reviewed-case-library');
        const firstOpenRequest=report.requests.length;
        await js(`(()=>{const b=[...document.querySelectorAll('.poc-case-card')].find(b=>b.querySelector('strong')?.textContent===${JSON.stringify(process.env.DATA_REBUILD_CATALOG_ID||'UCSF-PDGM-0009')});if(!b)throw Error('Reviewed external case missing from actual library');b.click()})()`);
        const openStarted=Date.now();
        let opened;
        while(!(opened=report.requests.slice(firstOpenRequest).find(row=>row.operation==='research:openListedWorkspace'))){
          if(Date.now()-openStarted>180000)throw Error('Timed out waiting for reviewed library case open');
          await delay(100);
        }
        assert(!opened.error,opened.error);
        if(opened.caseHash)await wait(`document.querySelector('.rl-viewer')?.dataset.sourceCaseHash===${JSON.stringify(opened.caseHash)}`);
        await ready();await capture('01b-opened-through-reviewed-library');
        report.catalog.openedCaseHash=opened.caseHash;
        report.checks.push('Actual Cases card uses opaque external ID and checksum-bound catalog open');
      }
      report.initialBody=await js('document.body.innerText');
      if(process.env.DATA_REBUILD_STAGES==='1'){
        await click('Scans');await wait(`!!document.querySelector('[aria-label="Inspect scan"]')`);
        const stages=await js(`[...document.querySelector('[aria-label="Inspect scan"]').options].filter(o=>/US before|US during|US after/.test(o.textContent)).map(o=>({value:o.value,label:o.textContent}))`);
        assert.equal(stages.length,3,'All observed US operative stages are distinctly named');
        for(let i=0;i<stages.length;i++){
          await set('Inspect scan',stages[i].value);await ready();await capture('stage-'+(i+1)+'-native-US');
          report.checks.push('Actual UI selects and renders '+stages[i].label);
        }
        await set('Inspect scan','');await ready();
      }
      if(process.env.DATA_REBUILD_REHEARSE==='1'){
        await click('Rehearse');await wait(`!!document.querySelector('.surgical-panel')`);
        await wait(`!!document.querySelector('[aria-label="Instrument action"]')||[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Prepare instrument scene'&&!b.disabled)`);
        if(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify saved replay'&&!b.disabled)`))await click('Verify saved replay');
        if(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Prepare instrument scene'&&!b.disabled)`))await click('Prepare instrument scene');
        await wait(`!!document.querySelector('[aria-label="Instrument action"]')`);await ready();await capture('02-rehearsal');
        report.checks.push('Actual production surgical scene initializes; missing evidence remains unavailable');
        const checkTaskDomain=process.env.DATA_REBUILD_EXPECT_TASK_DOMAIN==='1';
        if(checkTaskDomain||process.env.DATA_REBUILD_EXPECT_STATE_ID||process.env.DATA_REBUILD_REQUIRE_GPU_STATE==='1'){
          const caseHash=report.requests.findLast(row=>row.operation==='research:surgicalRehearsal'&&row.caseHash)?.caseHash;
          assert(caseHash,'Actual rehearsal source identity is required');
          const before=await js(`window.resectionApi.surgicalRehearsal({caseHash:${JSON.stringify(caseHash)},command:'inspect'})`);
          if(process.env.DATA_REBUILD_EXPECT_STATE_ID){
            assert.equal(before.view.stateId,process.env.DATA_REBUILD_EXPECT_STATE_ID);
            assert.equal(before.position,Number(process.env.DATA_REBUILD_EXPECT_REPLAY_ACTIONS));
            report.physicalReplay={stateId:before.view.stateId,position:before.position,metrics:before.view.metrics};
            report.checks.push('Actual desktop verifies the exact saved physical action prefix and metrics');
          }
          if(process.env.DATA_REBUILD_REQUIRE_GPU_STATE==='1'){
            report.physicalReplayGpu=await completedFrame(before.view.stateId);
            report.checks.push('Accepted physical state is bound to its actual completed GPU frame');
          }
          if(process.env.DATA_REBUILD_EXPECT_OPERATING_STATE_ID){
            const operatingState=process.env.DATA_REBUILD_EXPECT_OPERATING_STATE_ID;
            await click('Previous recorded state');
            const gpu=await completedFrame(operatingState);
            const operating=await js(`window.resectionApi.surgicalRehearsal({caseHash:${JSON.stringify(caseHash)},command:'inspect'})`);
            assert.equal(operating.view.stateId,operatingState);
            assert.equal(operating.position,before.position-1);
            assert.equal(operating.view.slots.right.instrument,'cusa');
            assert.equal(operating.view.slots.right.inserted,true);
            assert(operating.view.metrics.targetRemovedMm3>0);
            report.operatingReplay={stateId:operatingState,position:operating.position,metrics:operating.view.metrics,slots:operating.view.slots,gpu};
            await capture('02b-recorded-target-effect');
            await click('Next recorded state');
            await completedFrame(before.view.stateId);
            report.checks.push('Actual replay controls render the recorded target effect and return to the verified withdrawal');
          }
          if(checkTaskDomain){
            const denied=await js(`window.resectionApi.surgicalRehearsal({caseHash:${JSON.stringify(caseHash)},command:'preview',expectedStateId:${JSON.stringify(before.view.stateId)},action:{slot:'left',operation:'place',duration:.1,pose:{position:[10000,10000,10000],quaternion:[0,0,0,1]}}})`);
            const after=await js(`window.resectionApi.surgicalRehearsal({caseHash:${JSON.stringify(caseHash)},command:'inspect'})`);
            assert(JSON.stringify(denied).includes('REVIEWED_TASK_DOMAIN_OUTSIDE'));
            assert.equal(after.view.stateId,before.view.stateId);
            report.taskDomainProbe={sourceCaseHash:caseHash,stateBefore:before.view.stateId,stateAfter:after.view.stateId,
              refusal:'REVIEWED_TASK_DOMAIN_OUTSIDE',api:'Actual renderer preload -> production surgicalRehearsal preview',uncommitted:true};
            report.checks.push('Production desktop rejects unknown-space whole-tool placement with unchanged state');
          }
        }
        await set('Instrument action','inspect');await click('Preview');
        await wait(`document.querySelector('.surgical-panel').textContent.includes('Preview · uncommitted')${process.env.DATA_REBUILD_EXPECT_TASK_DOMAIN==='1'?"||document.querySelector('.surgical-panel').textContent.includes('REVIEWED_TASK_DOMAIN')":''}`);await ready();await capture('03-inspect-preview');
        if(await js(`document.querySelector('.surgical-panel').textContent.includes('Preview · uncommitted')`)){
          await click('Cancel preview');await ready();report.checks.push('Native instrument inspect preview and cancel preserve accepted state');
        }else report.checks.push('Unplaced instrument inspection is correctly restricted by the declared observed task box');
        if(process.env.DATA_REBUILD_ACTIONS==='1'){
          await js(`(()=>{const b=[...document.querySelectorAll('.surgical-toolbox button')].find(b=>b.textContent.includes('CUSA')&&!b.disabled);if(!b)throw Error('No available CUSA exchange');for(let p=b.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;b.click()})()`);
          await wait(`document.querySelector('.surgical-slots button')?.textContent.includes('CUSA')`);await ready();
          await capture('03b-instrument-exchange');report.checks.push('Actual UI exchanges withdrawn left instrument to CUSA without an invented access corridor');
        }
      }
      await click('Save');await wait(`document.body.innerText.includes('Workspace saved locally')`);assert(fs.existsSync(saved));
      report.saved=saved;report.savedSha256=digest(saved);report.checks.push('Actual UI saves native workspace externally');
      const count=report.requests.length;await click('Open');await wait(`document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`);
      while(!report.requests.slice(count).some(row=>row.operation==='research:openWorkspace'))await delay(100);
      if(process.env.DATA_REBUILD_REHEARSE==='1'){
        if(!await js(`!!document.querySelector('.surgical-panel')`))await click('Rehearse');
        await wait(`!!document.querySelector('.surgical-panel')`);
        await wait(`!!document.querySelector('[aria-label="Instrument action"]')||[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify saved replay'&&!b.disabled)`);
        if(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify saved replay'&&!b.disabled)`))await click('Verify saved replay');
        if(report.physicalReplay){
          await wait(`document.querySelector('.surgical-panel')?.dataset.stateId===${JSON.stringify(report.physicalReplay.stateId)}`);
          report.checks.push('Reopened desktop shows the same verified physical state');
        }
      }
      if(report.acquiredHost){
        report.acquiredHost.reopenedLayers=await inspectHostLayers();
        assert.deepEqual(report.acquiredHost.reopenedLayers,report.acquiredHost.layers);
        report.checks.push('Native save and reopen preserve all displayed host anatomy layers and their provenance');
      }
      await capture('04-native-reopened');report.checks.push('Actual UI reopens the saved native package');
      if(process.env.DATA_REBUILD_FUSION==='1'){
        await wait(`!!document.querySelector('.rl-fusion-readout')`);
        report.checks.push('Registered fusion choice survives actual UI save/reopen');
      }
      report.finalBody=await js('document.body.innerText');assert.equal(digest(input),inputSha);assert.equal(report.windowEverVisible,false);report.passed=true;
    }catch(error){report.error=String(error);report.body=await js('document.body.innerText').catch(()=>null);fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());}
    finally{await finish();}
  });
});
setTimeout(()=>{if(!report.passed){report.error='Global hidden verifier timeout';void finish();}},600000).unref();
require('./main.cjs');
