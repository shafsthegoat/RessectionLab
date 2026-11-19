'use strict';
// Electron SCRIPT writable-native-manifest episode output-dir target [--cold]
// Then repeat with --cold and --cold-midpoint in fresh Electron processes.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {execFileSync} = require('node:child_process');
const {app, dialog, ipcMain} = require('electron');
const {movieIntervals} = require('./study-engine-recording.cjs');
const {inputScript} = require('./verify-eden-planning-helpers.cjs');

const [, , nativeArg, episodeArg, outputArg, target] = process.argv;
assert(nativeArg && episodeArg && outputArg && target, 'Pass native manifest, episode, output directory and target');
const standalone = nativeArg==='--workspace';
const native = standalone?null:fs.realpathSync(nativeArg), episode = fs.realpathSync(episodeArg), output = path.resolve(outputArg);
const testRoot = path.resolve(__dirname, '../../testcases/outputs') + path.sep;
assert(standalone || native.startsWith(testRoot), 'Use a disposable native study under testcases/outputs');
const midpoint = process.argv.includes('--cold-midpoint');
const cold = midpoint || process.argv.includes('--cold');
const saved = path.join(output,'recorded-engine.ressectionlab');
const midSaved = path.join(output,'recorded-midpoint.ressectionlab');
const reportFile = path.join(output,midpoint?'cold-midpoint.json':cold?'cold-reopen.json':'recording.json');
assert(!fs.existsSync(reportFile) && (cold || !fs.existsSync(saved)), 'Choose a fresh output directory');
fs.mkdirSync(output,{recursive:true});
const scratch = fs.mkdtempSync(path.join(os.tmpdir(),'study-engine-replay-'));
const studyHash = native&&sha256File(native);
function sha256File(filename){return createHash('sha256').update(fs.readFileSync(filename)).digest('hex');}
const sourceHash = sha256File(episode);
const artifact = JSON.parse(execFileSync('unzip',['-p',episode,'manifest.json'],{maxBuffer:256*1024*1024})).artifacts.surgical_toolbox;
const history = artifact.timeline[artifact.branchId];
const endpoint = history.length;
const width = 1440, height = 900;
const started=performance.now();
const assets=path.join(__dirname,'../dist/assets'),rendererAssets=Object.fromEntries(fs.readdirSync(assets).filter(n=>n.startsWith('index-')&&/\.(js|css)$/.test(n)).map(n=>[n,sha256File(path.join(assets,n))]));
const movie=path.join(output,'engine-preparation.mp4');
const report = {rendererAssets,hardware:{cpu:os.cpus()[0].model,memoryGiB:os.totalmem()/2**30,os:os.release(),arch:os.arch()},observations:[],sourceHash,branch:artifact.branchId,target,windowEverVisible:false,states:[]};
const engineStates = new Map();
let lastObservation=null,captureImage=null,captureStarted=0,rejection=null;
const handle = ipcMain.handle.bind(ipcMain);
ipcMain.handle = (channel,handler) => handle(channel,async (...args)=>{
  const start=performance.now(),options=args[1]??{};
  if(channel==='research:openBoundSurgicalReplay') {report.requestedStudyReplay=true;console.log('Checking native study and recording');}
  const result = await handler(...args);
  if(channel==='research:openBoundSurgicalReplay') report.binding=result.verifiedStudyReplay;
  if(channel==='research:surgicalRehearsal' && result?.view) {
    const {stateId,slots}=result.view;
    engineStates.set(stateId,{debrisMm3:result.view.metrics.debrisMm3,debrisRemovedMm3:result.view.metrics.debrisRemovedMm3,ledger:result.sequenceOutcome??engineStates.get(stateId)?.ledger,tools:Object.fromEntries(Object.entries(slots).map(
      ([id,{instrument,position,quaternion,inserted}])=>[id,{instrument,position,quaternion,inserted}]))});
    if(result.outcome?.status==='rejected')rejection={reason:result.outcome.reasonCode??result.outcome.reason,stateId,position:result.position};
    const p=result.playback;
    if(p) {
      lastObservation={position:p.position,frameIndex:p.frameIndex,frameCount:p.frameCount,kind:p.frames[p.frameIndex].kind,
        time:p.observedTimeSeconds,stateId,engineMs:performance.now()-start,tools:engineStates.get(stateId).tools,
        targetRemoved:result.view.metrics.targetRemovedMm3,nonTargetRemoved:result.view.metrics.nonTargetRemovedMm3,
        contactCells:result.view.contact.length,removedCells:result.view.removed.length,residual:result.view.originalTargetResidualMm3,
        fluid:result.view.metrics.fluidPresentMm3,debris:result.view.metrics.debrisMm3};
      captureStarted=start;
    }
  }
  if(channel==='research:recordEngineMovie'&&options.command==='capture'){
    const o=lastObservation, h=history[o.position];
    assert.equal(o.stateId,options.stateId);
    if(o.frameIndex===0)assert.equal(o.stateId,h.stateBefore);
    if(o.frameIndex===o.frameCount-1)assert.equal(o.stateId,h.stateAfter);
    const pane=await args[0].sender.executeJavaScript(`(()=>{const p=document.querySelector('[data-accepted-tools]');return {
      id:p.dataset.surgicalStateId,tools:JSON.parse(p.dataset.acceptedTools),removed:Number(p.dataset.removedCells),
      readout:p.querySelector('[data-operative-readout-state]')?.textContent};})()`);
    assert.deepEqual(pane.tools,o.tools);assert.equal(pane.removed,o.removedCells);assert.equal(pane.id,o.stateId);
    assert(pane.readout.includes('Owned target removed '+o.targetRemoved.toFixed(1)+' mm³'));
    const sampleIndex=report.observations.length,previous=report.observations.at(-1);
    const demonstratesRemoval=previous && !report.firstInteriorRemoval && o.frameIndex>0 && o.frameIndex<o.frameCount-1 &&
      o.removedCells>previous.removedCells && o.tools[history[o.position].action.slot]?.inserted;
    if(demonstratesRemoval)report.firstInteriorRemoval={index:sampleIndex,previous:previous.stateId,observed:o.stateId,
      deltaCells:o.removedCells-previous.removedCells,withinAction:o.position};
    if(demonstratesRemoval || o.frameIndex===1 && [0,2,Math.floor(endpoint/2),endpoint-2].includes(o.position) ||
       o.frameIndex===0 && o.position===0 || o.position===endpoint-1 && o.frameIndex===o.frameCount-1 ||
       (o.frameIndex===Math.floor(o.frameCount/2)&&o.position===history.findIndex(x=>x.events.some(e=>e.kind==='removal')))){
      o.image='observation-'+sampleIndex+'.png';fs.writeFileSync(path.join(output,o.image),captureImage.toPNG());
    }
    report.observations.push({...o,captureMs:performance.now()-start,totalMs:performance.now()-captureStarted});
    if(o.frameIndex===0 && o.position%10===0)console.log('Captured action',o.position,'of',endpoint);
  }
  return result;
});
app.setPath('userData',path.join(scratch,'profile'));
if(process.platform==='darwin') app.setActivationPolicy('accessory');
process.argv.push('--automation-hidden',...(cold?['--case',midpoint?midSaved:saved]:standalone?['--case',episode]:['--anatomy',native]));
delete process.env.RESSECTIONLAB_DEV_URL;
dialog.showMessageBox=async(...args)=>{throw Error('Unexpected native dialog in hidden verification: '+args.at(-1)?.title);};
dialog.showOpenDialog=async (...args)=>{
  const options=args.at(-1);
  assert.match(options.title,/recorded engine episode/i);
  return {canceled:false,filePaths:[episode]};
};
dialog.showSaveDialog=async (...args)=>{
  if(args.at(-1).title==='Export observed engine replay')return {canceled:false,filePath:movie};
  const filename=cold?midSaved:saved;
  assert(!fs.existsSync(filename),'Save verifier output to a new file');
  return {canceled:false,filePath:filename};
};
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const expectedId=i=>i?history[i-1].stateAfter:history[0].stateBefore;
app.on('browser-window-created',(_,win)=>{
  const capture=win.webContents.capturePage.bind(win.webContents);
  win.webContents.capturePage=async(rect,opts)=>{
    const img=await capture(rect,opts);
    if(rect)captureImage=img.resize({width:rect.width,height:rect.height});
    return img;
  };
  win.on('show',()=>{report.windowEverVisible=true;win.hide();});
  win.webContents.once('did-finish-load',async()=>{
    const js=expr=>win.webContents.executeJavaScript(expr,true);
    const until=async(expr,limit=240000)=>{
      const start=Date.now();
      while(Date.now()-start<limit){
        if(await js(expr)) return;
        const error=await js("document.querySelector('.surgical-panel [role=alert]')?.textContent");
        if(error && !/^(partial|rejected):/.test(error)) throw Error(error);
        const exportError=await js("document.querySelector('[aria-label=\"Replay export status\"]')?.textContent");
        if(exportError?.includes('Error:')) throw Error(exportError);
        await delay(125);
      }
      throw Error('Timed out: '+expr+'\n'+await js('document.body.innerText.slice(-5000)'));
    };
    const click=async label=>{
      const button='[...document.querySelectorAll("button")].find(b=>b.textContent.trim()==='+JSON.stringify(label)+'||b.getAttribute("aria-label")==='+JSON.stringify(label)+')';
      await until('('+button+')?.disabled===false');
      await js('(()=>{const b='+button+';for(let p=b.parentElement;p;p=p.parentElement)if(p.tagName==="DETAILS")p.open=true;b.click()})()');
    };
    const ready=async i=>{
      await until(`document.querySelector('.surgical-panel')?.dataset.recordedPosition==='${i}' &&
        document.querySelector('[data-accepted-tools]')?.dataset.surgicalStateId===${JSON.stringify(expectedId(i))} &&
        document.querySelector('.rl-viewer')?.dataset.surfacesPending==='0' && !document.querySelector('.viewer-loading')`);
    };
    const seek=async i=>{await js(inputScript('Surgical timeline',i));await ready(i);};
    const state=async i=>{
      await ready(i);
      const data=await js(`(()=>{const pane=document.querySelector('[data-accepted-tools]'),checks=[...document.querySelectorAll('.surgeon-view input[type=checkbox]')];
        return {id:pane.dataset.surgicalStateId,camera:JSON.parse(pane.dataset.opticalCamera),tools:JSON.parse(pane.dataset.acceptedTools),
          cutawayOff:checks.length>=2&&checks.filter(x=>/inspection-only/i.test(x.parentElement.textContent)).every(x=>!x.checked),view:pane.dataset.operativeView,removedCells:Number(pane.dataset.removedCells)};})()`);
      assert.equal(data.view,'surgeon');assert(data.cutawayOff,'Microscope must be opaque with cutaway off');
      if(!engineStates.get(data.id)?.ledger) {
        await click('Calculate sequence breakdown');
        await until("!!document.querySelector('[aria-label=\"Complete sequence quantities\"]')");
      }
      const engine=engineStates.get(data.id);
      assert(engine,'Missing authoritative response for displayed state '+i);
      assert.deepEqual(data.tools,engine.tools);
      assert.equal(engine.ledger.stateId,data.id);
      const readout=await js(`document.querySelector('[data-operative-readout-state="${data.id}"]')?.textContent`);
      const residual=engine.ledger.originalTarget.residualMm3;
      assert(readout?.includes('Whole target remaining '+(residual===null?'unknown':residual.toFixed(1)+' mm³')),
        'Whole-target HUD must agree with the state ledger');
      assert(readout.includes('Owned target removed '+engine.ledger.dimensions.targetRemovedMm3.cumulative.toFixed(1)+' mm³'));
      data.readout=readout;
      assert.equal(await js("document.querySelector('[data-operative-locator-state]')?.dataset.operativeLocatorState"),data.id);
      assert.equal(await js("Number(document.querySelector('[data-surgical-metric=originalTargetResidualMm3]')?.textContent.replaceAll(',',''))"),residual);
      return {position:i,...data,ledger:engine.ledger};
    };
    try {
      win.webContents.setBackgroundThrottling(false);
      win.setContentSize(width,height);
      await until('innerWidth==='+width+'&&innerHeight==='+height);
      if(cold) {
        const expected=JSON.parse(fs.readFileSync(path.join(output,'recording.json')));
        await until("!!document.querySelector('.rl-viewer canvas')");
        await click('Rehearse');
        if(standalone){
          await until("!!document.querySelector('.surgical-panel [aria-label=\"Surgical timeline\"]')");
          const verify=await js(`[...document.querySelectorAll('button')].find(b=>b.checkVisibility()&&['Verify saved replay','Verify replay for planning'].includes(b.textContent.trim()))?.textContent.trim()`);
          if(verify)await click(verify);
          await click('Rehearse/edit stage');
        }
        const wanted=midpoint?expected.states[Math.floor(endpoint/2)]:expected.savedState;
        await until("!!document.querySelector('[data-operative-view=surgeon][data-optical-camera]')");
        report.restored=await state(wanted.position);
        assert.deepEqual(report.restored,wanted);
        const cleanupStart=history.findIndex((row,i)=>i>0&&history[i-1].action.operation==='withdraw'&&
          row.action.operation==='exchange'&&row.action.instrument==='frazier_suction');
        if(midpoint && standalone && cleanupStart>0){
          await seek(cleanupStart);const before=engineStates.get(expectedId(cleanupStart));
          await seek(endpoint);const after=engineStates.get(expectedId(endpoint));
          assert(before.debrisMm3>0);assert.equal(after.debrisMm3,0);
          assert(Math.abs(after.debrisRemovedMm3-before.debrisRemovedMm3-before.debrisMm3)<1e-6);
          report.cleanup={before:{position:cleanupStart,stateId:expectedId(cleanupStart),debrisMm3:before.debrisMm3},
            after:{position:endpoint,stateId:expectedId(endpoint),debrisMm3:after.debrisMm3,collectedMm3:after.debrisRemovedMm3-before.debrisRemovedMm3}};
          await seek(wanted.position);assert.deepEqual(await state(wanted.position),wanted);
        }
        if(!midpoint) {
          const obs=expected.observations;
          const firstWork=obs.find((o,i)=>i>0 && o.targetRemoved>obs[i-1].targetRemoved && o.frameIndex>0 && o.frameIndex<o.frameCount-1)??
            obs.find((o,i)=>i>0 && o.removedCells>obs[i-1].removedCells && o.frameIndex>0 && o.frameIndex<o.frameCount-1);
          assert(firstWork,'Need a physical tissue change inside a recorded action');
          await seek(firstWork.position);
          if(firstWork.frameIndex>3){
            await click('Play replay');await until("!!document.querySelector('[aria-label=\"Observed action frames\"]')");
            await click('Pause replay');
            const lead=firstWork.frameIndex-3,ref=obs.find(o=>o.position===firstWork.position&&o.frameIndex===lead);
            await js(inputScript('Observed action frames',lead));
            await until(`document.querySelector('[data-accepted-tools]')?.dataset.surgicalStateId===${JSON.stringify(ref.stateId)} &&
              document.querySelector('[aria-label="Observed action frames"]')?.value==='${lead}'`);
          }
          await js(`window.liveTrace=[];window.liveObserver=new MutationObserver(()=>{const p=document.querySelector('[aria-label="Observed microstep replay"]');
            if(!p)return;const text=p.querySelector('p').textContent;
            if(window.liveTrace.at(-1)?.text!==text)window.liveTrace.push({text,wallMs:performance.now(),id:document.querySelector('[data-accepted-tools]').dataset.surgicalStateId});});
            window.liveObserver.observe(document.querySelector('.rehearsal-replay'),{childList:true,characterData:true,subtree:true})`);
          await click('Play replay');
          await until("window.liveTrace.length>=8");
          await click('Pause replay');
          await delay(600);
          const observedState=()=>js(`(()=>{const panel=document.querySelector('[aria-label="Observed microstep replay"]'),pane=document.querySelector('[data-accepted-tools]');
            return {observation:panel?.querySelector('p')?.textContent,state:pane.dataset.surgicalStateId,tools:JSON.parse(pane.dataset.acceptedTools)};})()`);
          report.pausedMicrostep=await observedState();
          assert(report.pausedMicrostep.observation,'Expected an observed engine microstep');
          await delay(500);
          assert.deepEqual(await observedState(),report.pausedMicrostep);
          report.liveTrace=await js('window.liveObserver.disconnect();window.liveTrace');
          for(const f of report.liveTrace){
            const match=/Action (\d+) · frame (\d+) \/ (\d+)/.exec(f.text);
            assert(match,'Recorded frame label missing');
            const ref=expected.observations.find(o=>o.position===Number(match[1])-1&&o.frameIndex===Number(match[2]));
            assert.equal(ref?.stateId,f.id);
          }
          assert(report.liveTrace.some(f=>f.id===firstWork.stateId),'Live replay must display the first measured interior tissue change');
          report.liveTreatment={position:firstWork.position,frame:firstWork.frameIndex,stateId:firstWork.stateId,
            targetRemoved:firstWork.targetRemoved,otherRemoved:firstWork.nonTargetRemoved,time:firstWork.time};
          const at=await js(`Number(document.querySelector('[aria-label="Observed action frames"]').value)`);
          await click(at>0?'Previous microstep':'Next microstep');await until(`Number(document.querySelector('[aria-label="Observed action frames"]').value)!==${at}`);
          await click(at>0?'Next microstep':'Previous microstep');await until(`Number(document.querySelector('[aria-label="Observed action frames"]').value)===${at}`);
          assert.deepEqual(await observedState(),report.pausedMicrostep);
          const pausedImage=await win.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true});
          fs.writeFileSync(path.join(output,'paused-engine-microstep.png'),pausedImage.resize({width,height}).toPNG());
          await seek(endpoint-1);
          await click('Play replay');
          await until(`document.querySelector('.surgical-panel')?.dataset.recordedPosition==='${endpoint}' &&
            !document.querySelector('[aria-label="Observed microstep replay"]')`);
          const end=await state(endpoint);
          assert.deepEqual(end,wanted);report.liveEndpoint={stateId:end.id,withdrawalComplete:end.ledger.withdrawalComplete};
          const position=Math.floor(endpoint/2);
          await seek(position);assert.deepEqual(await state(position),expected.states[position]);
          await click('Save');await until("document.body.innerText.includes('Workspace saved locally')");
          assert(fs.existsSync(midSaved));
        }
      } else {
        await until("!!document.querySelector('.rl-viewer canvas')");
        if(standalone){
          await click('Rehearse');
          await until("!!document.querySelector('.surgical-panel [aria-label=\"Surgical timeline\"]')");
          const verify=await js(`[...document.querySelectorAll('button')].find(b=>b.checkVisibility()&&['Verify saved replay','Verify replay for planning'].includes(b.textContent.trim()))?.textContent.trim()`);
          if(verify)await click(verify);
          await click('Rehearse/edit stage');
          await ready(endpoint);
        }else{
        await click('Plan');
        await until(`[...document.querySelectorAll('[aria-label="Planning target"] option')].some(o=>o.value===${JSON.stringify(target)}&&!o.disabled)`);
        await js(inputScript('Planning target',target));
        await until(`document.querySelector('[aria-label="Planning target"]')?.value===${JSON.stringify(target)}&&!document.querySelector('.approach-computing')`);
        await click('Rehearse');console.log('Opening source-bound episode');
        // Native target selection can briefly disable this button across the
        // transition from Plan. Wait for an actual dialog dispatch.
        for(let attempt=0;attempt<3&&!report.requestedStudyReplay;attempt++){
          await click('Open recorded engine episode');await delay(750);
        }
        assert(report.requestedStudyReplay,'Study binding request was not dispatched by the visible control');
        await ready(endpoint);
        assert.equal(report.binding.endpointStateId,expectedId(endpoint));
        assert.equal(report.binding.targetLayerId,target);
        assert(report.binding.snapshotHash,'Expected the verified archival snapshot hash');
        }
        // Enter at the first inserted tool so the microscope looks down the
        // recorded working corridor, rather than centering on a parked tool.
        const firstInsertion=history.findIndex(action=>action.action.operation==='apply')+1;
        assert(firstInsertion>0,'Need an inserted working tool for the microscope');
        await seek(firstInsertion);
        await click('Surgeon / Microscope');
        await until("!!document.querySelector('[data-operative-view=surgeon][data-optical-camera]')");
        await js(inputScript('Viewing angle',15));
        const anchor=await state(firstInsertion);
        const inserted=Object.values(anchor.tools).find(tool=>tool.inserted);
        assert(inserted,'The working tool must be inserted at microscope placement');
        assert.deepEqual(anchor.camera.target,inserted.position);
        if(standalone){
          const hand=Object.keys(anchor.tools).find(k=>anchor.tools[k].inserted);
          await js(`[...document.querySelectorAll('.surgical-slots button')].find(b=>b.querySelector('strong').textContent===${JSON.stringify(hand+' hand')}).click()`);
          await js(`(()=>{const b=document.querySelector('.surgical-toolbox button[aria-pressed="false"]');if(!b||b.disabled)throw Error('No instrument exchange control');
            b.closest('details').open=true;b.click()})()`);
          await until("document.querySelector('.surgical-panel')?.innerText.includes('WITHDRAW_BEFORE_EXCHANGE')||document.querySelector('.surgical-panel')?.innerText.includes('Withdraw this instrument before exchanging it')");
          assert.equal(rejection?.stateId,anchor.id);assert.equal(rejection.position,firstInsertion);
          report.rejectedExchange={...rejection,unchangedTools:anchor.tools};
          assert.deepEqual(await state(firstInsertion),anchor);
          {
            const shot=await win.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true});
            fs.writeFileSync(path.join(output,'rejected-exchange.png'),shot.resize({width,height}).toPNG());
          }
        }
        report.cameraAnchor={position:firstInsertion,tool:inserted.instrument,target:anchor.camera.target};
        report.fieldCrop=await js(`(()=>{const r=document.querySelector('[data-accepted-tools]').getBoundingClientRect();
          return {x:Math.round(r.x+r.width*.22),y:Math.round(r.y+r.height*.18),
            width:Math.round(r.width*.56),height:Math.round(r.height*.54)};})()`);
        for(const i of [...new Set([0,2,3,firstInsertion,Math.floor(endpoint/2),endpoint])].sort((a,b)=>a-b)){
          await seek(i);
          const sample=await state(i);
          await js("document.querySelector('.rehearsal-replay').scrollIntoView({block:'start'});new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))");
          assert(!win.isVisible());
          const image=await win.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true});
          const frame=path.join(scratch,i+'.png');
          fs.writeFileSync(frame,image.resize({width,height}).toPNG());
          if([0,Math.floor(endpoint/2),endpoint].includes(i)) fs.copyFileSync(frame,path.join(output,'state-'+i+'.png'));
          report.states[i]=sample;
          if(i%10===0) console.log('Captured boundary',i);
        }
        assert(report.states.at(-1).ledger.withdrawalComplete);
        assert(report.states.some(s=>s&&Object.values(s.tools).some(t=>t.inserted)));
        await seek(3);assert.deepEqual(await state(3),report.states[3]);
        await click('Previous recorded state');assert.deepEqual(await state(2),report.states[2]);
        await seek(endpoint);assert.deepEqual(await state(endpoint),report.states.at(-1));
        report.savedState=await state(endpoint);
        await click('Save');
        await until("document.body.innerText.includes('Workspace saved locally')");
        assert(fs.existsSync(saved));
        await click('Export observed replay');
        await until("document.querySelector('[aria-label=\"Replay export status\"]')?.textContent.startsWith('Exported ')",60*60*1000);
        verifyMedia();
        await ready(endpoint);
        assert.deepEqual(await state(endpoint),report.savedState);
        const stop=artifact.cranialPlanning?.alternatives?.find(row=>row.validStop&&row.nativeActions===0);
        if(standalone && stop){
          await js(inputScript('Surgical branch',stop.branchId));
          await until(`document.querySelector('[data-accepted-tools]')?.dataset.surgicalStateId===${JSON.stringify(stop.finalStateId)} &&
            document.querySelector('.surgical-panel')?.dataset.recordedPosition==='0'`);
          await click('Calculate sequence breakdown');
          await until("!!document.querySelector('[aria-label=\"Complete sequence quantities\"]')");
          const ledger=engineStates.get(stop.finalStateId)?.ledger;
          assert(ledger?.withdrawalComplete);
          assert.equal(ledger.dimensions.targetRemovedMm3.cumulative,0);
          assert.equal(ledger.dimensions.nonTargetRemovedMm3.cumulative,0);
          assert.equal(ledger.dimensions.elapsedSeconds.cumulative,0);
          report.stopBeforeAccess={branch:stop.branchId,reason:stop.stopReason,stateId:stop.finalStateId,ledger};
          const shot=await win.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true});
          fs.writeFileSync(path.join(output,'stop-before-access.png'),shot.resize({width,height}).toPNG());
          await js(inputScript('Surgical branch',artifact.branchId));
          await ready(endpoint);
        }
      }
      assert.equal(sha256File(episode),sourceHash);
      if(native)assert.equal(sha256File(native),studyHash);
      assert(!report.windowEverVisible);
      report.passed=true;
    } catch(error) {
      report.passed=false;report.error=String(error.stack??error);
      report.body=await js('document.body.innerText').catch(()=>null);
      console.error(report.error);
    } finally {
      report.wallSeconds=(performance.now()-started)/1000;
      fs.writeFileSync(reportFile,JSON.stringify(report,null,2)+'\n');
      fs.rmSync(scratch,{recursive:true,force:true});app.exit(report.passed?0:1);
    }
  });
});
function verifyMedia(){
  const obs=report.observations,intervals=movieIntervals(obs),last=intervals.at(-1);
  const probe=JSON.parse(execFileSync('ffprobe',['-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,duration,nb_frames','-of','json',movie])).streams[0];
  assert.equal(probe.codec_name,'h264');assert(Math.abs(Number(probe.duration)-(last.milliseconds+last.duration)/1000)<.025);
  const middle=obs.filter(o=>o.frameIndex>0&&o.frameIndex<o.frameCount-1);
  assert(middle.some(o=>o.stateId!==history[o.position].stateAfter&&o.stateId!==history[o.position].stateBefore));
  const fieldEvidence=[];
  for(const i of intervals.filter(x=>obs[x.index].image && x.duration>=40)){
    const o=obs[i.index],filter='crop=iw*.5:ih*.45:iw*.25:ih*.23,scale=96:96,format=rgb24';
    const decode=(input,seek,video)=>execFileSync('ffmpeg',['-nostdin','-v','error',...(seek?['-ss',String(seek)]:[]),'-i',input,'-vf',(video?'crop=iw:ih-90:0:0,':'')+filter,'-frames:v','1','-f','rawvideo','pipe:1']);
    const a=decode(path.join(output,o.image),0,false),b=decode(movie,(i.milliseconds+i.duration/2)/1000,true);
    assert.equal(a.length,b.length);const mae=a.reduce((n,v,k)=>n+Math.abs(v-b[k]),0)/a.length;
    assert(mae<8,'Decoded field differs at '+o.image+' by '+mae);fieldEvidence.push({image:o.image,index:i.index,mae});
  }
  assert(fieldEvidence.length>=2);assert(fieldEvidence.some(e=>obs[e.index].frameIndex>0&&obs[e.index].frameIndex<obs[e.index].frameCount-1));
  const changes={pose:0,removal:0,firstContact:null,firstTargetTreatment:null};
  for(let i=1;i<obs.length;i++){
    const previous=obs[i-1],s=obs[i];
    if(s.contactCells>0 && changes.firstContact===null)changes.firstContact=i;
    if(s.targetRemoved>0 && changes.firstTargetTreatment===null)changes.firstTargetTreatment=i;
    if(s.frameIndex===0 || s.frameIndex===s.frameCount-1)continue;
    if(JSON.stringify(previous.tools)!==JSON.stringify(s.tools))changes.pose++;
    if(previous.removedCells!==s.removedCells)changes.removal++;
  }
  assert(changes.pose>0 && changes.removal>0,'Expected actual intermediate tool and tissue changes.');
  report.media={changes,...probe,sha256:sha256File(movie),timedImages:intervals.length,intermediateStates:middle.length,terminalHoldSeconds:1,fieldEvidence};
}
require('./main.cjs');
