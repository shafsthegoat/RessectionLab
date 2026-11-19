'use strict';
// Electron SCRIPT native-manifest episode output-dir target [--cold|--cold-midpoint|--cold-plan|--full-movie]
// Each cold invocation starts with an empty profile and opens only its saved file.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {execFileSync} = require('node:child_process');
const {app, dialog, ipcMain} = require('electron');
const {sourceTarget,tissueState,verifySection,encodeWalkthrough,continuousWalkthrough}=require('./verify-integrated-replay-helpers.cjs');
const {inputScript} = require('./verify-eden-planning-helpers.cjs');

const [, , nativeArg, episodeArg, outputArg, target] = process.argv;
assert(nativeArg && episodeArg && outputArg && target, 'Pass native manifest, episode, output directory and target');
const native = fs.realpathSync(nativeArg), episode = fs.realpathSync(episodeArg), output = path.resolve(outputArg);
const testRoot = path.resolve(__dirname, '../../testcases/outputs') + path.sep;
assert(native.startsWith(testRoot) && output.startsWith(testRoot), 'Use a disposable native study under testcases/outputs');
const planningCold=process.argv.includes('--cold-plan'),planFile=path.join(output,'selected-approach.rsworkspace');
const midpoint = process.argv.includes('--cold-midpoint');
const fullMovie=process.argv.includes('--full-movie');
const cold = fullMovie || planningCold || midpoint || process.argv.includes('--cold');
const saved = path.join(output,'recorded-engine.ressectionlab');
const midSaved = path.join(output,'recorded-midpoint.ressectionlab');
const reportFile = path.join(output,fullMovie?'full-movie.json':planningCold?'cold-plan.json':midpoint?'cold-midpoint.json':cold?'cold-reopen.json':'recording.json');
assert(!fs.existsSync(reportFile) && (cold || !fs.existsSync(saved)), 'Choose a fresh output directory');
fs.mkdirSync(output,{recursive:true});
const scratch = fs.mkdtempSync(path.join(os.tmpdir(),'study-engine-replay-'));
const studyHash=sha256File(native);
function sha256File(filename){return createHash('sha256').update(fs.readFileSync(filename)).digest('hex');}
const sourceHash = sha256File(episode);
const artifact = JSON.parse(execFileSync('unzip',['-p',episode,'manifest.json'],{maxBuffer:256*1024*1024})).artifacts.surgical_toolbox;
const history = artifact.timeline[artifact.branchId];
const source=sourceTarget(episode);
let plan=null,saveMid=false;
const samples=[];
const witnessFractions=[.12,.3,.5,.7,.88,.96],duration=history.reduce((sum,row)=>sum+row.elapsedSeconds,0);
const endpoint = history.length;
const width = 1440, height = 900;
const started=performance.now();
const assets=path.join(__dirname,'../dist/assets'),rendererAssets=Object.fromEntries(fs.readdirSync(assets).filter(n=>n.startsWith('index-')&&/\.(js|css)$/.test(n)).map(n=>[n,sha256File(path.join(assets,n))]));
const movie=path.join(output,'observed-engine-replay.mp4');
const report = {rendererAssets,hardware:{cpu:os.cpus()[0].model,memoryGiB:os.totalmem()/2**30,os:os.release(),arch:os.arch()},observations:[],sourceHash,branch:artifact.branchId,target,windowEverVisible:false,states:[]};
const engineStates = new Map();
let lastObservation=null;
const handle = ipcMain.handle.bind(ipcMain);
ipcMain.handle = (channel,handler) => handle(channel,async (...args)=>{
  const start=performance.now(),options=args[1]??{};
  if(channel==='research:openBoundSurgicalReplay') {report.requestedStudyReplay=true;console.log('Checking native study and recording');}
  const result = await handler(...args);
  if(channel==='research:recordEngineMovie'&&options.command==='capture'){
    assert.equal(lastObservation?.stateId,options.stateId);
    (report.exportFrames??=[]).push({id:options.stateId,rect:options.rect,time:lastObservation.time,position:lastObservation.position,frame:lastObservation.frameIndex});
    if(fullMovie&&witnessFractions.length&&lastObservation.time>=duration*witnessFractions[0]&&lastObservation.frameIndex>0&&lastObservation.frameIndex<lastObservation.frameCount-1){
      const rect=options.rect,shot=await args[0].sender.capturePage(rect,{stayHidden:true,stayAwake:true});
      const dir=path.join(output,'.movie-checks');fs.mkdirSync(dir,{recursive:true});
      fs.writeFileSync(path.join(dir,(report.exportFrames.length-1)+'.png'),shot.resize({width:rect.width,height:rect.height}).toPNG());
      witnessFractions.shift();
    }
    if(fullMovie&&report.exportFrames.length%100===0)console.log('Exported observations',report.exportFrames.length,lastObservation.time);
  }
  if(channel==='research:openBoundSurgicalReplay') report.binding=result.verifiedStudyReplay;
  if(channel==='research:anatomyPlan'&&result?.approaches)plan=result.approaches.find(r=>r.id===result.selectedId)??null;
  if(channel==='research:surgicalRehearsal' && result?.view) {
    const {stateId,slots}=result.view;
    engineStates.set(stateId,{tissue:tissueState(source,result.view),metrics:result.view.metrics,sceneId:result.view.scene?.sceneId??report.sceneId,caseHash:result.caseHash,ledger:result.sequenceOutcome??engineStates.get(stateId)?.ledger,tools:Object.fromEntries(Object.entries(slots).map(
      ([id,{instrument,position,quaternion,inserted}])=>[id,{instrument,position,quaternion,inserted}]))});
    report.sceneId=engineStates.get(stateId).sceneId;
    const p=result.playback;
    if(p) {
      lastObservation={position:p.position,frameIndex:p.frameIndex,frameCount:p.frameCount,kind:p.frames[p.frameIndex].kind,
        time:p.observedTimeSeconds,stateId,engineMs:performance.now()-start,tools:engineStates.get(stateId).tools,
        targetRemoved:result.view.metrics.targetRemovedMm3,nonTargetRemoved:result.view.metrics.nonTargetRemovedMm3,
        contactCells:result.view.contact.length,removedCells:result.view.removed.length,residual:result.view.originalTargetResidualMm3,
        fluid:result.view.metrics.fluidPresentMm3,debris:result.view.metrics.debrisMm3};
    }
  }
  return result;
});
app.setPath('userData',path.join(scratch,'profile'));
if(process.platform==='darwin') app.setActivationPolicy('accessory');
process.argv.push('--automation-hidden',...(cold?['--case',planningCold||fullMovie?planFile:midpoint?midSaved:saved]:['--anatomy',native]));
delete process.env.RESSECTIONLAB_DEV_URL;
dialog.showMessageBox=async(...args)=>{throw Error('Unexpected native dialog in hidden verification: '+args.at(-1)?.title);};
dialog.showOpenDialog=async (...args)=>{
  const options=args.at(-1);
  assert.match(options.title,/recorded engine episode/i);
  return {canceled:false,filePaths:[episode]};
};
dialog.showSaveDialog=async (...args)=>{
  if(args.at(-1).title==='Export observed engine replay')return {canceled:false,filePath:movie};
  const filename=args.at(-1).title==='Save portable patient study'?planFile:saveMid?midSaved:saved;
  assert(!fs.existsSync(filename),'Save verifier output to a new file');
  return {canceled:false,filePath:filename};
};
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const expectedId=i=>i?history[i-1].stateAfter:history[0].stateBefore;
app.on('browser-window-created',(_,win)=>{
  win.on('show',()=>{report.windowEverVisible=true;win.hide();});
  win.webContents.once('did-finish-load',async()=>{
    const js=async expr=>{try{return await win.webContents.executeJavaScript(expr,true);}catch(error){throw Error(expr+'\n'+error);}};
    const until=async(expr,limit=60000)=>{
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
      console.log('Selecting',label);await until('('+button+')?.disabled===false');
      await js('(()=>{const b='+button+';for(let p=b.parentElement;p;p=p.parentElement)if(p.tagName==="DETAILS")p.open=true;b.click()})()');
    };
    const ready=async i=>{
      await until(`document.querySelector('.surgical-panel')?.dataset.recordedPosition==='${i}' &&
        document.querySelector('[data-accepted-tools]')?.dataset.surgicalStateId===${JSON.stringify(expectedId(i))} &&
        document.querySelector('.rl-viewer')?.dataset.surfacesPending==='0' && !document.querySelector('.viewer-loading')`);
    };
    const seek=async i=>{await until("document.querySelector('[aria-label=\"Surgical timeline\"]')?.disabled===false");await js(inputScript('Surgical timeline',i));await ready(i);};
    const capture=async(name,selector)=>{
      assert(!win.isVisible());
      await delay(180);
      const rect=selector?await js(`(()=>{const b=document.querySelector(${JSON.stringify(selector)}).getBoundingClientRect();const x=Math.ceil(b.x),y=Math.ceil(b.y);return {x,y,width:Math.floor(b.right)-x,height:Math.floor(b.bottom)-y};})()`):undefined;
      const size=rect?{width:rect.width,height:rect.height}:await js('({width:innerWidth,height:innerHeight})');
      const shot=await win.webContents.capturePage(rect,{stayHidden:true,stayAwake:true});
      fs.writeFileSync(path.join(output,name+'.png'),shot.resize(size).toPNG());
    };
    const referenceReady=()=>until("!!document.querySelector('.rl-viewer canvas')&&[...document.querySelectorAll('.rl-viewer')].every(v=>v.dataset.surfacesPending==='0')&&!document.querySelector('.viewer-loading,.rl-viewer-preparing')");
    const selected=async()=>{
      await until("!!document.querySelector('.approach-card.selected')&&!document.querySelector('.approach-computing')");
      await referenceReady();
      const shown=await js("document.querySelector('.approach-card.selected').dataset.approachId");
      const archiveEntries=planningCold?execFileSync('unzip',['-Z1',planFile],{encoding:'utf8'}).split('\n'):null;
      const content=planningCold?execFileSync('unzip',['-p',planFile,archiveEntries.find(e=>e.endsWith('anatomy-approaches.json'))]):fs.readFileSync(path.join(path.dirname(native),'anatomy-approaches.json'));
      plan=JSON.parse(content).approaches.find(r=>r.id===shown)??plan;assert.equal(plan.id,shown);
      const {id,name,targetLayerId,entry,target,parameterHash,measurementHash}=plan;return {id,name,targetLayerId,entry,target,parameterHash,measurementHash};
    };
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
      const rendered=await js("JSON.parse(document.querySelector('[data-accepted-tools]').dataset.renderedToolSlots)");
      assert.deepEqual(rendered.sort(),Object.keys(engine.tools).filter(k=>engine.tools[k].inserted).sort());data.renderedToolSlots=rendered;
      assert.equal(engine.ledger.stateId,data.id);
      const readout=await js(`document.querySelector('[data-operative-readout-state="${data.id}"]')?.textContent`);
      const residual=engine.ledger.originalTarget.residualMm3;
      assert(readout?.includes('Whole target remaining '+(residual===null?'unknown':residual.toFixed(1)+' mm³')),
        'Whole-target HUD must agree with the state ledger');
      assert(readout.includes('Owned target removed '+engine.ledger.dimensions.targetRemovedMm3.cumulative.toFixed(1)+' mm³'));
      data.readout=readout;
      assert.equal(await js("document.querySelector('[data-operative-locator-state]')?.dataset.operativeLocatorState"),data.id);
      assert.equal(await js("Number(document.querySelector('[data-surgical-metric=originalTargetResidualMm3]')?.textContent.replaceAll(',',''))"),Number(residual.toFixed(2)));
      await until(`!!document.querySelector('canvas[data-engine-inspection-state="${data.id}"]')`);
      data.inspection=await verifySection(win.webContents,data.id,engine.tissue,source);
      assert.equal(data.inspection.sceneId,engine.sceneId);assert.equal(data.inspection.caseId,engine.caseHash);
      const reference=await js(`(()=>{const ref=document.querySelector('[data-engine-replay-layout] .engine-intact-reference');return ref?.querySelector('canvas')?{snapshot:ref.dataset.studySnapshot,approach:ref.dataset.recordedApproach,target:ref.dataset.targetLayer}:null;})()`);
      const receipt=report.binding??JSON.parse(fs.readFileSync(path.join(output,'recording.json'))).binding;
      assert.deepEqual(reference,{snapshot:receipt.sourceSnapshotId,approach:receipt.recordedApproach.id,target:receipt.targetLayerId});
      for(const metric of ['targetRemovedMm3','nonTargetRemovedMm3','fluidPresentMm3']){
        assert.equal(await js(`Number(document.querySelector('[data-surgical-metric="${metric}"]').textContent)`),Number(engine.metrics[metric].toFixed(2)));
      }
      return {position:i,...data,ledger:engine.ledger};
    };
    try {
      win.webContents.setBackgroundThrottling(false);
      win.setContentSize(width,height);
      await until('innerWidth==='+width+'&&innerHeight==='+height);
      await referenceReady();
      const expectedRecord=cold&&!fullMovie?JSON.parse(fs.readFileSync(path.join(output,'recording.json'))):null;
      if(fullMovie){
        assert(!fs.existsSync(movie));await click('Plan');await referenceReady();await click('Rehearse');
        for(let n=0;n<3&&!report.requestedStudyReplay;n++){await click('Open recorded engine episode');await delay(600);}
        await ready(endpoint);report.restored=await state(endpoint);
        await click('Export observed replay');
        await until("document.querySelector('[aria-label=\"Replay export status\"]')?.textContent.startsWith('Exported ')",60*60*1000);
        await ready(endpoint);assert.deepEqual(await state(endpoint),report.restored);
        const probe=JSON.parse(execFileSync('ffprobe',['-v','error','-select_streams','v:0','-show_entries','stream=width,height,duration,codec_name','-of','json',movie])).streams[0];
        assert.equal(probe.codec_name,'h264');assert(Math.abs(Number(probe.duration)-report.restored.ledger.dimensions.elapsedSeconds.cumulative-1)<0.03);
        const rect=report.exportFrames[0].rect;
        assert.equal(probe.width,Math.ceil(rect.width/2)*2);assert.equal(probe.height,Math.ceil(rect.height/2)*2+90);
        report.movie={...probe,sha256:sha256File(movie),observations:report.exportFrames.length};
        report.presentation=continuousWalkthrough(output,report,JSON.parse(fs.readFileSync(path.join(output,'recording.json'))));
      }else if(!cold||planningCold){
        if(!cold){await click('Anatomy');await referenceReady();await capture('patient-overview');}
        await click('Plan');report.planned=await selected();assert.equal(report.planned.targetLayerId,target);
        if(planningCold){
          assert.deepEqual(report.planned,expectedRecord.planned);assert.equal(await js("!!document.querySelector('[data-accepted-tools]')"),false);
          await capture('cold-planning');
        }else{
          await capture('selected-recorded-approach');
          await click('Scans');await click('Save patient study file…');
          await until("document.querySelector('footer')?.textContent.includes('Patient study file saved')",600000);
          assert(fs.statSync(planFile).size>0);
          await click('Rehearse');for(let n=0;n<3&&!report.requestedStudyReplay;n++){await click('Open recorded engine episode');await delay(600);}
          await until("!!document.querySelector('[data-engine-replay-layout]')");
          assert.equal(report.binding.endpointStateId,expectedId(endpoint));
          for(const key of ['id','parameterHash','measurementHash','entry','target'])assert.deepEqual(report.binding.recordedApproach[key],report.planned[key]);
          assert.equal(report.binding.actionGenerator.kind,'scripted');
        }
      }else{
        await click('Rehearse');
        const wanted=expectedRecord.states[midpoint?expectedRecord.workPosition:endpoint];
        report.restored=await state(wanted.position);assert.deepEqual(report.restored,wanted);
        await capture(midpoint?'cold-midpoint':'cold-procedure');
      }
      if(!cold){
        const first=history.findIndex(h=>h.action.operation==='apply')+1;assert(first);
        await seek(first);
        if(await js("document.querySelector('[data-accepted-tools]').dataset.operativeView")!=='surgeon')await click('Surgeon / Microscope');
        await until("!!document.querySelector('[data-operative-view=surgeon][data-optical-camera]')");
        if(await js("!!document.querySelector('[aria-label=\"Viewing angle\"]')"))await js(inputScript('Viewing angle',15));
        for(let i=0;i<=endpoint;i++){
          await seek(i);const data=await state(i);report.states.push(data);
          console.log('Verified boundary',i,data.inspection.gridRemoval.cells);
          const name='action-'+i;await capture(name,'[data-engine-replay-layout]');
          samples.push({image:name+'.png',position:i,operation:i?history[i-1].action.operation:'Ready',time:data.ledger.dimensions.elapsedSeconds.cumulative});
        }
        const end=report.states.at(-1);assert(end.inspection.cavityOnSlice.cells&&end.inspection.targetOnSlice.cells);
        assert(end.ledger.withdrawalComplete);await capture('procedure-final');
        await click('Save');await until("document.body.innerText.includes('Workspace saved locally')");assert(fs.existsSync(saved));
        win.setContentSize(1050,720);await until('innerWidth===1050&&innerHeight===720');await referenceReady();await capture('procedure-compact');
        assert.deepEqual(await state(endpoint),end);
        win.setContentSize(width,height);await until(`innerWidth===${width}&&innerHeight===${height}`);
        const work=report.states.findIndex(s=>s.inspection.gridRemoval.cells>0&&s.inspection.gridRemoval.cells<end.inspection.gridRemoval.cells);
        assert(work>0);report.workPosition=work;await seek(work);assert.deepEqual(await state(work),report.states[work]);
        saveMid=true;await click('Save');await until("document.body.innerText.includes('Workspace saved locally')");assert(fs.existsSync(midSaved));
        await seek(0);assert.deepEqual(await state(0),report.states[0]);await seek(endpoint);assert.deepEqual(await state(endpoint),end);
        report.rewindReplay=[work,0,endpoint];
        for(const [position,row] of history.entries()){
          if(row.action.operation!=='apply')continue;
          await seek(position);await click('Play replay');
          await until("!!document.querySelector('[aria-label=\"Observed action frames\"]')");
          await click('Pause replay');await delay(450);
          const max=await js("Number(document.querySelector('[aria-label=\"Observed action frames\"]').max)");
          for(const frame of [...new Set([.15,.33,.5,.67,.84,.96].map(f=>Math.floor(max*f)))].filter(f=>f>0&&f<max)){
            await js(inputScript('Observed action frames',frame));
            await until(`document.querySelector('[aria-label="Observed microstep replay"] p')?.textContent.startsWith('Action ${position+1} · frame ${frame} /')`);
            const id=await js("document.querySelector('[data-accepted-tools]').dataset.surgicalStateId");
            await until(`!!document.querySelector('canvas[data-engine-inspection-state="${id}"]')`);
            const state=engineStates.get(id),o=lastObservation;assert.equal(o.stateId,id);assert.equal(o.frameIndex,frame);assert.equal(o.position,position);
            assert.deepEqual(await js("JSON.parse(document.querySelector('[data-accepted-tools]').dataset.acceptedTools)"),state.tools);
            const rendered=await js("JSON.parse(document.querySelector('[data-accepted-tools]').dataset.renderedToolSlots)");
            assert.deepEqual(rendered.sort(),Object.keys(state.tools).filter(k=>state.tools[k].inserted).sort());
            const inspection=await verifySection(win.webContents,id,state.tissue,source);
            const name='observed-'+position+'-'+frame;await capture(name,'[data-engine-replay-layout]');
            report.observations.push({...o,inspection,renderedToolSlots:rendered,image:name+'.png'});samples.push({image:name+'.png',position:position+frame/max,operation:row.action.operation,time:o.time});
          }
        }
        assert(report.observations.some(o=>o.inspection.cavityOnSlice.cells>0));
        assert(report.observations.some(o=>o.targetRemoved>0),'Show interior target removal');
        await seek(endpoint);assert.deepEqual(await state(endpoint),end);
        await click('Export observed replay');
        await until("document.querySelector('[aria-label=\"Replay export status\"]')?.textContent.startsWith('Captured action')");
        await click('Cancel movie export');
        await ready(endpoint);assert.deepEqual(await state(endpoint),end);
        assert(report.exportFrames.length>0);report.exportCancelledAndRestored=true;

        report.walkthrough=encodeWalkthrough(output,samples,report.binding.patientId,report.planned.name,end.ledger);
        await click('Plan');assert.deepEqual(await selected(),report.planned);
        await click('Create manual approach');
        await until(`document.querySelector('.approach-card.selected')?.dataset.approachId!==${JSON.stringify(report.planned.id)} && !document.querySelector('.approach-computing')`);
        const alternate=await selected();assert.notEqual(alternate.id,report.planned.id);
        await click('Rehearse');await until("!!document.querySelector('.operative-inspector')");
        assert.equal(await js("!!document.querySelector('[data-engine-replay-layout],[data-accepted-tools]')"),false);
        report.otherPlan={id:alternate.id,recordingClosed:true};
      }

      assert.equal(sha256File(episode),sourceHash);
      assert.equal(sha256File(native),studyHash);
      assert(!report.windowEverVisible);
      report.passed=true;
    } catch(error) {
      await capture('failed-'+path.basename(reportFile,'.json')).catch(()=>{});report.passed=false;report.error=String(error.stack??error);
      report.body=await js('document.body.innerText').catch(()=>null);
      console.error(report.error);
    } finally {
      report.wallSeconds=(performance.now()-started)/1000;
      fs.writeFileSync(reportFile,JSON.stringify(report,null,2)+'\n');
      fs.rmSync(scratch,{recursive:true,force:true});app.exit(report.passed?0:1);
    }
  });
});
require('./main.cjs');
