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
const {sourceTarget,tissueState,verifySection,verifyResultBounds,encodeWalkthrough,continuousWalkthrough}=require('./verify-integrated-replay-helpers.cjs');
const {inputScript} = require('./verify-eden-planning-helpers.cjs');

const [, , nativeArg, episodeArg, outputArg, target] = process.argv;
assert(nativeArg && episodeArg && outputArg && target, 'Pass native manifest, episode, output directory and target');
const native = fs.realpathSync(nativeArg), episode = fs.realpathSync(episodeArg), output = path.resolve(outputArg);
const testRoot = fs.realpathSync(path.resolve(__dirname, '../../testcases/outputs')) + path.sep;
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
const exact=artifact.exactGridReplay;
assert(exact?.recordedApproach&&exact.actionGenerator&&exact.targetLayerId===target,'Episode must declare the approach, target and action generator');
const nativeStudy=JSON.parse(fs.readFileSync(native));
const targetInfo=nativeStudy.layers.find(l=>l.id===target);assert(targetInfo);
const source=sourceTarget(episode);
let plan=null,saveMid=false;
const samples=[],inspectionFrames=[];
const witnessFractions=[.12,.3,.5,.7,.88,.96],duration=history.reduce((sum,row)=>sum+row.elapsedSeconds,0);
const endpoint = history.length;
const width = 1440, height = 900;
const started=performance.now();
const assets=path.join(__dirname,'../dist/assets'),rendererAssets=Object.fromEntries(fs.readdirSync(assets).filter(n=>n.startsWith('index-')&&/\.(js|css)$/.test(n)).map(n=>[n,sha256File(path.join(assets,n))]));
const movie=path.join(output,'observed-engine-replay.mp4');
const report = {pid:process.pid,rendererAssets,hardware:{cpu:os.cpus()[0].model,memoryGiB:os.totalmem()/2**30,os:os.release(),arch:os.arch()},observations:[],sourceHash,studyHash,branch:artifact.branchId,target,windowEverVisible:false,states:[]};
const engineStates = new Map();
let lastObservation=null;
const handle = ipcMain.handle.bind(ipcMain);
ipcMain.handle = (channel,handler) => handle(channel,async (...args)=>{
  const start=performance.now(),options=args[1]??{};
  if(channel==='research:openBoundSurgicalReplay') {report.requestedStudyReplay=true;console.log('Checking native study and recording');}
  const result = await handler(...args);
  if(channel==='research:recordEngineMovie'&&options.command==='finish')report.media=result;
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
    if(engineStates.size>12)engineStates.delete(engineStates.keys().next().value);
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
    const waitForSave=async file=>{
      for(let i=0;i<600&&!fs.existsSync(file);i++)await delay(100);
      assert(fs.existsSync(file),'Missing completed save '+file);
      await until("document.body.innerText.includes('Workspace saved locally')");
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
    const routeContour=async()=>{
      const expr=`(()=>{const c=document.querySelector('.engine-route-plane canvas');if(!c)return 0;
        const rgba=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let n=0;
        for(let i=0;i<rgba.length;i+=4)if(rgba[i]===98&&rgba[i+1]===191&&rgba[i+2]===212)n++;return n;})()`;
      await until(expr);return js(expr);
    };
    const chooseRecordedApproach=async()=>{
      const route=exact.recordedApproach.id;
      await until("!document.querySelector('.approach-computing')&&!!document.querySelector('.approach-card')");
      if(await js("document.querySelector('[aria-label=\"Planning target\"]')?.value")!==target){
        await js(inputScript('Planning target',target));
      }
      const card=`document.querySelector('.approach-card[data-approach-id="'+${JSON.stringify(route)}+'"]')`;
      await until('('+card+')?.disabled===false');
      if(await js("document.querySelector('.approach-card.selected')?.dataset.approachId")!==route)await js('('+card+').click()');
      await until("document.querySelector('.approach-card.selected')?.dataset.approachId==="+JSON.stringify(route));
    };
    const selected=async()=>{
      await until("!!document.querySelector('.approach-card.selected')&&!document.querySelector('.approach-computing')");
      await referenceReady();
      const shown=await js("document.querySelector('.approach-card.selected').dataset.approachId");
      const archiveEntries=planningCold?execFileSync('unzip',['-Z1',planFile],{encoding:'utf8'}).split('\n'):null;
      const content=planningCold?execFileSync('unzip',['-p',planFile,archiveEntries.find(e=>e.endsWith('anatomy-approaches.json'))]):fs.readFileSync(path.join(path.dirname(native),'anatomy-approaches.json'));
      plan=JSON.parse(content).approaches.find(r=>r.id===shown)??plan;assert.equal(plan.id,shown);
      const {id,name,targetLayerId,entry,target,parameterHash,measurementHash}=plan;return {id,name,targetLayerId,entry,target,parameterHash,measurementHash};
    };
    const resultView=async(position,parts='both')=>{
      const id=expectedId(position),e=engineStates.get(id);
      assert(e,'Missing authoritative engine state for 3D result');
      const query="document.querySelector('.engine-result-orbit')";
      await until(`(${query})?.dataset.resultState===${JSON.stringify(id)}&&(${query}).dataset.resultParts===${JSON.stringify(parts)}`);
      const drawn=await js(`(()=>{const d=(${query}).dataset;return {id:d.resultState,parts:d.resultParts,
        cavityCells:+d.resultCavityCells,residualCells:+d.resultResidualCells,cavityHash:d.resultCavityHash,residualHash:d.resultResidualHash,
        cavityTriangles:+d.resultCavityTriangles,residualTriangles:+d.resultResidualTriangles,camera:JSON.parse(d.resultCamera)};})()`);
      const removed=new Set(e.tissue.removed),ownedRemaining=[...source.owned].filter(i=>!removed.has(i));
      const maskHash=indices=>indices.reduce((h,i)=>Math.imul(h^i,16777619)>>>0,2166136261).toString(16).padStart(8,'0');
      assert.equal(drawn.cavityCells,removed.size);assert.equal(drawn.cavityHash,maskHash([...removed]));
      assert.equal(drawn.residualCells,ownedRemaining.length);assert.equal(drawn.residualHash,maskHash(ownedRemaining));
      const bounds=await js(`(()=>{const d=(${query}).dataset;return d.resultCavityBounds?{cavity:JSON.parse(d.resultCavityBounds),residual:JSON.parse(d.resultResidualBounds)}:null;})()`);
      assert(bounds,'Missing submitted mesh bounds');
      const conditioning=await js(`(()=>{const d=(${query}).dataset;return d.resultCavitySourceBounds?{sourceBounds:JSON.parse(d.resultCavitySourceBounds),displacementMm:Number(d.resultCavityDisplacementMm)}:null;})()`);
      (report.meshBounds??={})[id]=verifyResultBounds(source,[...removed],ownedRemaining,bounds,conditioning);
      const overlap=[...source.source].filter(i=>!source.owned.has(i));
      const keyText=await js("document.querySelector('.result-key')?.textContent");
      assert(keyText?.includes('Owned residual target · '+(ownedRemaining.length*source.voxelMm3).toFixed(1)+' mm³'));
      if(overlap.length)assert(keyText.includes(overlap.filter(i=>!removed.has(i)).length.toLocaleString()+' of '+overlap.length.toLocaleString()+' overlapping target cells'));
      if(drawn.cavityCells>0)assert(drawn.cavityTriangles>0);
      if(drawn.residualCells>0)assert(drawn.residualTriangles>0);
      assert(drawn.camera.position.every(Number.isFinite)&&drawn.camera.target.every(Number.isFinite)&&drawn.camera.up.every(Number.isFinite));
      return drawn;
    };
    const sameInspection=(restored,saved)=>{
      const {camera:actual,...state}=restored,{camera:expected,...reference}=saved;
      assert.deepEqual(state,reference);
      let maximumCameraError=0;
      for(const key of ['position','target','up']){
        assert.equal(actual[key].length,3);
        for(let i=0;i<3;i++)maximumCameraError=Math.max(maximumCameraError,Math.abs(actual[key][i]-expected[key][i]));
      }
      assert(maximumCameraError<1e-8,'Restored 3D inspection camera differs by '+maximumCameraError);
      return maximumCameraError;
    };
    const orbitResult=async(position)=>{
      const initial=await resultView(position);
      const origin=await js("(()=>{const b=document.querySelector('.engine-result-orbit').getBoundingClientRect();return {x:Math.round(b.left+b.width*.45),y:Math.round(b.top+b.height*.45)};})()");
      const input=(type,x,y,down)=>win.webContents.sendInputEvent({type,x,y,button:'left',clickCount:1,buttons:down?['left']:[],modifiers:down?['leftButtonDown']:[]});
      input('mouseDown',origin.x,origin.y,true);
      for(let i=1;i<=8;i++){
        input('mouseMove',origin.x+5*i,origin.y+2*i,true);await delay(40);
        const name='result-orbit-'+i;await capture(name,'[data-engine-replay-layout]');
        inspectionFrames.push({image:name+'.png',seconds:.12,title:'Final result inspection',note:'Orbiting the cavity and residual target at the saved endpoint'});
      }
      input('mouseUp',origin.x+40,origin.y+16,false);
      await until(`document.querySelector('.engine-result-orbit').dataset.resultCamera!==${JSON.stringify(JSON.stringify(initial.camera))}`);
      return resultView(position);
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
      if(exact.observations){
        const witness=exact.observations[i];
        assert.equal(witness.stateId,data.id);
        const recordedPose=Object.fromEntries(Object.entries(witness.tools).map(([slot,{instrument,position,quaternion,inserted}])=>[slot,{instrument,position,quaternion,inserted}]));
        assert.deepEqual(data.tools,recordedPose);
        assert.equal('sha256:'+createHash('sha256').update(JSON.stringify(engine.tissue.removed)).digest('hex'),witness.removedHash);
        assert.equal(engine.ledger.originalTarget.residualMm3,witness.residualTargetMm3);
        assert.equal(engine.ledger.dimensions.elapsedSeconds.cumulative,witness.elapsedSeconds);
      }
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
      if(!report.intactRouteTargetPixels)report.intactRouteTargetPixels=await routeContour();
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
        assert(!fs.existsSync(movie));await click('Plan');await chooseRecordedApproach();await referenceReady();await click('Rehearse');
        for(let n=0;n<3&&!report.requestedStudyReplay;n++){await click('Open recorded engine episode');await delay(600);}
        await ready(endpoint);report.restored=await state(endpoint);
        await click('Export observed replay');
        await until("document.querySelector('[aria-label=\"Replay export status\"]')?.textContent.startsWith('Exported ')",60*60*1000);
        await ready(endpoint);assert.deepEqual(await state(endpoint),report.restored);
        const probe=JSON.parse(execFileSync('ffprobe',['-v','error','-select_streams','v:0','-show_entries','stream=width,height,duration,codec_name','-of','json',movie])).streams[0];
        assert.equal(probe.codec_name,'h264');assert(Math.abs(Number(probe.duration)-report.restored.ledger.dimensions.elapsedSeconds.cumulative-1)<0.03);
        const rect=report.exportFrames[0].rect;
        assert.equal(probe.width,Math.max(320,Math.ceil(rect.width/2)*2));assert.equal(probe.height,Math.ceil(rect.height/2)*2+report.media.captionBandPixels);
        report.movie={...probe,sha256:sha256File(movie),observations:report.exportFrames.length};
        report.presentation=continuousWalkthrough(output,report,JSON.parse(fs.readFileSync(path.join(output,'recording.json'))));
      }else if(!cold||planningCold){
        if(!cold){await click('Anatomy');await referenceReady();await capture('patient-overview');}
        await click('Plan');if(!planningCold)await chooseRecordedApproach();report.planned=await selected();assert.equal(report.planned.targetLayerId,target);
        if(planningCold){
          assert.deepEqual(report.planned,expectedRecord.planned);assert.equal(await js("!!document.querySelector('[data-accepted-tools]')"),false);
          await capture('cold-planning');
        }else{
          await js("document.querySelector('.approach-card.selected').scrollIntoView({block:'center'})");
          await capture('selected-recorded-approach');
          await click('Scans');await click('Save patient study file…');
          await until("document.querySelector('footer')?.textContent.includes('Patient study file saved')",600000);
          assert(fs.statSync(planFile).size>0);
          await click('Rehearse');for(let n=0;n<3&&!report.requestedStudyReplay;n++){await click('Open recorded engine episode');await delay(600);}
          await until("!!document.querySelector('[data-engine-replay-layout]')");
          assert.equal(report.binding.endpointStateId,expectedId(endpoint));
          for(const key of ['id','parameterHash','measurementHash','entry','target'])assert.deepEqual(report.binding.recordedApproach[key],report.planned[key]);
          assert.deepEqual(report.binding.actionGenerator,exact.actionGenerator);
          assert.equal(report.binding.branchId,artifact.branchId);assert.equal(report.binding.actions,history.length);
        }
      }else{
        await click('Rehearse');
        const wanted=expectedRecord.states[midpoint?expectedRecord.workPosition:endpoint];
        report.restored=await state(wanted.position);assert.deepEqual(report.restored,wanted);
        const savedResult=midpoint?expectedRecord.workInspection:expectedRecord.resultInspection;
        report.restoredInspection=await resultView(wanted.position);report.maximumCameraError=sameInspection(report.restoredInspection,savedResult);
        await capture(midpoint?'cold-midpoint':'cold-procedure');
      }
      if(!cold){
        const first=history.findIndex(h=>h.action.operation==='apply')+1;assert(first,'Demonstration needs an executed tissue action');
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
        await capture('operative-endpoint');
        win.setContentSize(1050,720);await until('innerWidth===1050&&innerHeight===720');await referenceReady();await capture('operative-endpoint-1050');
        win.setContentSize(width,height);await until(`innerWidth===${width}&&innerHeight===${height}`);await referenceReady();
        await click('Inspect result in 3D');const fitted=await resultView(endpoint);report.combinedFit=fitted;
        await capture('result-combined','[data-engine-replay-layout]');
        inspectionFrames.push({image:'result-combined.png',seconds:1.6,title:'Final result inspection',note:'Removed space and remaining owned target in patient coordinates'});
        await click('Show cavity only');report.cavityOnly=await resultView(endpoint,'cavity');await capture('cavity-only','[data-engine-replay-layout]');
        inspectionFrames.push({image:'cavity-only.png',seconds:1.6,title:'Final result inspection',note:'Removed space at the recorded endpoint'});
        await click('Show residual only');report.residualOnly=await resultView(endpoint,'residual');await capture('residual-only','[data-engine-replay-layout]');
        inspectionFrames.push({image:'residual-only.png',seconds:1.6,title:'Final result inspection',note:'Remaining owned target in the same patient frame'});
        await click('Show cavity and residual');await click('Fit result');
        report.resultInspection=await orbitResult(endpoint);assert.notDeepEqual(report.resultInspection.camera,fitted.camera);
        await capture('procedure-final');
        await click('Save');await waitForSave(saved);
        win.setContentSize(1050,720);await until('innerWidth===1050&&innerHeight===720');await referenceReady();await capture('procedure-compact');
        assert.deepEqual(await state(endpoint),end);
        const compactResult=await resultView(endpoint);sameInspection(compactResult,report.resultInspection);
        win.setContentSize(width,height);await until(`innerWidth===${width}&&innerHeight===${height}`);
        const work=report.states.findIndex(s=>s.inspection.gridRemoval.cells>0&&s.inspection.gridRemoval.cells<end.inspection.gridRemoval.cells);
        assert(work>0);report.workPosition=work;await seek(work);assert.deepEqual(await state(work),report.states[work]);
        report.workInspection=await resultView(work);
        saveMid=true;await click('Save');await waitForSave(midSaved);
        await seek(0);assert.deepEqual(await state(0),report.states[0]);await seek(endpoint);assert.deepEqual(await state(endpoint),end);
        sameInspection(await resultView(endpoint),report.resultInspection);
        report.rewindReplay=[work,0,endpoint];await click('Return to operative field');
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

        report.walkthrough=encodeWalkthrough(output,samples,{patient:report.binding.patientId,route:report.planned.name,
          acquisitions:[...new Set(nativeStudy.series.map(s=>s.modality))].join(' / '),
          target:((targetInfo.label??target)+' · '+(targetInfo.provenance?.kind??targetInfo.kind)).replaceAll('_',' '),
          generator:exact.actionGenerator.kind+' · '+exact.actionGenerator.name},end.ledger,inspectionFrames);
        if(await js("!!document.querySelector('.engine-approach-views')")){
          await seek(work);
          report.corridorViews=[];
          for(const [name,button,clipped] of [['corridor','Inspect recorded corridor',true],['approach','View down approach',false]]){
            await click(button);
            await until(`[...document.querySelectorAll('.surgeon-view input[type=checkbox]')].some(e=>/cutaway/i.test(e.parentElement.textContent)&&e.checked===${clipped})`);
            for(const [w,h] of [[1440,900],[1050,720]]){
              win.setContentSize(w,h);await until('innerWidth==='+w+'&&innerHeight==='+h);
              await ready(work);await capture('working-'+name+'-'+w);
              const frame=await js(`(()=>{const d=document.querySelector('[data-accepted-tools]').dataset;
                return {state:d.surgicalStateId,view:d.operativeView,optics:JSON.parse(d.opticalCamera),tools:JSON.parse(d.acceptedTools)};})()`);
              assert.equal(frame.state,report.states[work].id);
              assert.deepEqual(frame.tools,report.states[work].tools);
              report.corridorViews.push({name,width:w,height:h,clipped,...frame});
            }
          }
          win.setContentSize(width,height);await until('innerWidth==='+width+'&&innerHeight==='+height);
        }
        await click('Plan');assert.deepEqual(await selected(),report.planned);
        const another=await js(`[...document.querySelectorAll('.approach-card')].find(b=>b.dataset.approachId!==${JSON.stringify(report.planned.id)})?.dataset.approachId`);
        if(another)await js(`[...document.querySelectorAll('.approach-card')].find(b=>b.dataset.approachId===${JSON.stringify(another)}).click()`);
        else await click('Create manual approach');
        await until(`document.querySelector('.approach-card.selected')?.dataset.approachId!==${JSON.stringify(report.planned.id)} && !document.querySelector('.approach-computing')`);
        const alternate=await selected();assert.notEqual(alternate.id,report.planned.id);
        await click('Rehearse');await until("!!document.querySelector('.operative-playback[data-operative-phase]')");
        assert.equal(await js("!!document.querySelector('[data-engine-replay-layout],[data-operative-readout-state],canvas[data-engine-inspection-state],.engine-recording-info')"),false);
        assert.equal(await js("document.querySelector('.operative-inspector strong')?.textContent"),alternate.name);
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
