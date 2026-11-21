'use strict';
// Invoke with Electron, a disposable native manifest, a new evidence directory and a mode.
// preview exercises fused scan switching and operative playback. --video records the
// real hidden compositor. --export FILE saves a portable study. Reopen it in a
// second process with mode reopen and --open-archive FILE.
// --reuse-selection replays a saved route. --stage-detail captures intermediate poses.
// --inspect-shortest with --distinct selects the shortest returned window for the saved geometric example.
// --material-review with --visual-review --video captures layer isolation and matched final cutaways without encoding a film.
// --source-support checks the raw/supported MRI and SIAM receipts on a real tumor fixture.
// --field-review --acceptance-views records section and Follow poses at both desktop sizes.
// --match-operative-cameras and --match-movie-poses accept prior report and movie-state JSON paths.
// --geometry-ablations toggles exact cell faces and exterior at fixed camera and timeline.
// --capture-both-sweeps captures legacy controls then replays the default; --preserve-plan-view verifies the saved start.
// --acquired-review measures cold reopen and records the native camera and source layers.
// In baseline mode it stops after the anatomical captures. In preview mode
// --acquired-episode FILE compares the same plan, field, and two bound engine
// boundaries. --compare-bound-signal captures a fixed-state shader control.
// --capture-field-labels exports the checked local field for a numeric audit.
// baseline, prepare, stale and no-target exercise simpler study states.
const {app, dialog, ipcMain} = require('electron');
const {execFileSync} = require('node:child_process');
const {recordingSize,framePixels,recordOperativePlayback} = require('./verify-operative-recording.cjs');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {studyScanSelectionScript} = require('./verify-study-ui-helpers.cjs');
const {setFieldView,verifyOperativeCorridor}=require('./verify-operative-corridor.cjs');
const {inputScript} = require('./verify-eden-planning-helpers.cjs');
const root = path.resolve(__dirname, '../..');
const [, , manifestArg, outputArg, mode = 'preview'] = process.argv;
assert(['preview','baseline','prepare','no-target','stale','reopen','reopen-decision','source-switch','compare-anatomy'].includes(mode),'Unknown verification mode');
assert(manifestArg && outputArg, 'Pass a disposable native manifest, new evidence directory, and baseline or preview');
const manifest = fs.realpathSync(manifestArg);
const fixtureRoot = fs.realpathSync(path.join(root, 'testcases/outputs')) + path.sep;
assert(manifest.startsWith(fixtureRoot), 'Use a writable verification copy under testcases/outputs');
const output = path.resolve(outputArg);
assert(output.startsWith(fixtureRoot) && !fs.existsSync(output), 'Choose a new evidence directory under testcases/outputs');
fs.mkdirSync(output);
const hash = data => createHash('sha256').update(data).digest('hex');
const original = hash(fs.readFileSync(manifest));
const document = JSON.parse(fs.readFileSync(manifest));
const targetLayer = document.layers.find(l=>l.role==='target') ?? document.layers.find(l=>l.kind==='tumor');
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'rslab-operative-preview-'));
app.setPath('userData',profile);
if(process.platform==='darwin') app.setActivationPolicy('accessory');
const startup = path.join(profile,'empty');
execFileSync(path.join(root,'.venv/bin/python'),['-c','from resectionlab.patient_anatomy import PatientImagingCollection as C; import sys; C.create(sys.argv[1],"EMPTY-VERIFIER").save()', startup],{cwd:root,env:{...process.env,PYTHONPATH:path.join(root,'src')}});
process.argv.push('--automation-hidden','--anatomy',path.join(startup,'patient-anatomy.json'));
delete process.env.RESSECTIONLAB_DEV_URL;
const report = {mode, manifest, patient:document.patient_id, sourceSnapshot:document.snapshot_id, everVisible:false, captures:[], errors:[]};
const acquiredReview = process.argv.includes('--acquired-review');
const started = performance.now();
const flagValue = name => {
  const index = process.argv.indexOf(name);
  return index<0 ? null : path.resolve(process.argv[index+1]);
};
const archive = flagValue('--open-archive');
const boundEpisode = flagValue('--acquired-episode');
if (boundEpisode) assert(acquiredReview && archive && fs.statSync(boundEpisode).isFile());
if(archive) {
  assert(archive.startsWith(fixtureRoot) && fs.statSync(archive).isFile());
  report.openedArchive={filename:archive,sha256:hash(fs.readFileSync(archive))};
}
const saveTo = flagValue('--export');
if(saveTo) assert(saveTo.startsWith(fixtureRoot) && !fs.existsSync(saveTo));
dialog.showOpenDialog = async (...args) => {
  const recording = /recorded engine episode/i.test(args.at(-1)?.title ?? '');
  assert(!recording || boundEpisode, 'Expected an explicit procedural save');
  return {canceled:false, filePaths:[recording ? boundEpisode : archive ?? manifest]};
};
dialog.showSaveDialog = async () => {
  assert(saveTo && !fs.existsSync(saveTo),'Unexpected save dialog');
  return {canceled:false,filePath:saveTo};
};
let activeRequests = 0;
let labelPacket;
const handle = ipcMain.handle.bind(ipcMain);
ipcMain.handle = (channel, handler) => handle(channel, async (...args) => {
  activeRequests++;
  try {
    const result = await handler(...args);
    if(process.argv.includes('--capture-field-labels')){
      if(channel==='research:anatomyPlan' && args[1].action==='operativeField'){
        const {intensity, ...metadata}=result;
        labelPacket=metadata.labels;
        fs.writeFileSync(path.join(output,'field-metadata.json'),JSON.stringify(metadata,null,2)+'\n');
      }
      if(channel==='research:readAsset' && labelPacket && args[1]===labelPacket.assetId){
        const bytes=Buffer.from(result);
        assert.equal(bytes.length,labelPacket.byteLength);
        assert.equal(hash(bytes),labelPacket.sha256);
        fs.writeFileSync(path.join(output,'field-labels.bin'),bytes);
      }
    }
    if(boundEpisode && channel==='research:openBoundSurgicalReplay') report.binding=result.verifiedStudyReplay;
    if(boundEpisode && channel==='research:surgicalRehearsal' && result?.view) {
      const v=result.view;
      (report.boundaryResponses??={})[v.stateId]={caseHash:result.caseHash,position:result.position,sceneId:v.scene?.sceneId,
        metrics:v.metrics,removed:v.removed.length,contact:v.contact.length,
        tools:Object.fromEntries(Object.entries(v.slots).map(([slot,{instrument,inserted,position,quaternion}])=>
          [slot,{instrument,inserted,position,quaternion}]))};
    }
    if(channel==='research:prepareSurgicalEnvironment') {
      report.engineInspection={request:args[1],caseHash:result.caseHash};
    }
    if(channel==='research:surgicalRehearsal' && mode==='reopen-decision' && result?.selectedPlanDecision) report.coldDecision=result.selectedPlanDecision;
    if(channel==='research:surgicalRehearsal' && report.engineInspection && result?.view) {
      report.engineInspection.engine={position:result.position,state:result.view.stateId,
        admission:result.planningAdmission,metrics:result.view.metrics,decision:result.selectedPlanDecision};
    }
    if(channel==='research:anatomyView' && process.argv.includes('--source-support')) {
      (report.anatomyViews ??= []).push({series:args[1].seriesId,sourceHash:result.mri.sha256,
        layers:result.compartments.map(l=>({id:l.name,kind:l.kind,volumeMm3:l.volumeMm3,sourceSupport:l.sourceSupport}))});
    }
    if(channel==='research:anatomyView' && result.patientContext?.signal) {
      (report.patientViews ??= {})[args[1].seriesId] = {
        contextHash:result.patientContext.signal.sha256,sourceHash:result.mri.sha256,
        shape:result.patientContext.shape,affine:result.patientContext.affine,
        scans:result.patientContext.sources.map(s=>({series:s.seriesId,alignment:s.alignment,contributes:!!s.displayWindow}))};
    }
    if(channel === 'research:anatomyPlan') {
      const action = args[1].action;
      (report.plans ??= []).push({action,selectedId:result.selectedId,
        routes:result.approaches?.map(r=>({id:r.id,origin:r.origin,status:r.assessmentStatus,entry:r.entry,target:r.target,parameterHash:r.parameterHash,
          lengthMm:r.measurements?.lengthMm,coveredTargetMm3:r.measurements?.targets?.find(t=>t.layerId===r.targetLayerId)?.coveredVoxelCenterVolumeMm3,
          normalTraversalMm3:r.measurements?.normalTraversal?.supportedNonTargetVoxelCenterVolumeMm3})),
        spacingMm:result.spacingMm,shape:result.shape,origin:result.origin,basisRas:result.basisRas,targetRasMm:result.targetRasMm,entryDepthMm:result.entryDepthMm,parameterHash:result.parameterHash,stations:result.stations,fieldLabelSha256:result.labels?.sha256,
        intensity:result.intensity && {series:result.intensity.sourceSeries,modality:result.intensity.modality,
          sourceHash:result.intensity.sourceSha256,dataHash:result.intensity.dataSha256,signalHash:result.intensity.signal?.sha256,
          covered:result.intensity.coveredVoxels,window:result.intensity.window,shape:result.intensity.shape},
        layers:result.layers?.map(l=>({id:l.id,role:l.role,detail:l.detail,evidence:l.evidence,model:l.model,sampledDataSha256:l.sampledDataSha256,coveredVoxels:l.coveredVoxels,vascularContextOnly:l.vascularContextOnly,voxels:l.sampledVoxels,sourceSupport:l.sourceSupport,sourceSpacingMm:l.sourceSpacingMm,inferenceSpacingMm:l.inferenceSpacingMm})),
        target:result.targetLayerId,approach:result.approachId,excluded:result.excluded,issues:result.stationIssues,
        stationEvidence:result.stationEvidence,stationLayerIds:result.stationLayerIds,
        distinctEntryReport:result.distinctEntryReport?{status:result.distinctEntryReport.status,shortfall:result.distinctEntryReport.shortfall}:undefined});
      console.log('Plan operation',action,result.selectedId ?? result.approachId ?? '',result.approaches?.length ?? '');
    }
    return result;
  }
  catch(error) { report.errors.push({channel, message:String(error)}); throw error; }
  finally { activeRequests--; }
});
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const reportPath = path.join(output,'report.json');
function finish(error) {
  report.sourceUnchanged = hash(fs.readFileSync(manifest)) === original;
  const archiveUnchanged=!archive || hash(fs.readFileSync(archive))===report.openedArchive.sha256;
  if(archive) report.openedArchive.unchanged=archiveUnchanged;
  report.passed = !error && report.sourceUnchanged && archiveUnchanged && !report.everVisible && !report.errors.length;
  if(error) report.error = error.stack || String(error);
  fs.writeFileSync(reportPath, JSON.stringify(report,null,2)+'\n');
  console.log(report.passed ? reportPath : JSON.stringify(report));
  process.exitCode = report.passed ? 0 : 1;
  if(boundEpisode){
    // A verified recording can still carry a dirty workspace presentation.
    // This disposable harness has no file to save and no user to answer a dialog.
    fs.rmSync(profile,{recursive:true,force:true});
    app.exit(process.exitCode);return;
  }
  app.quit();
}
app.once('will-quit', () => {
  fs.rmSync(profile,{recursive:true,force:true});
  if(!report.passed) app.exit(1);
});
dialog.showErrorBox = (title,message) => finish(Error(title+' '+message));
process.on('uncaughtException',finish);
process.on('unhandledRejection',finish);
app.on('browser-window-created', (_, win) => {
  win.webContents.on('console-message',detail=>{
    if(detail.level==='error' || /shader error|VALIDATE_STATUS false/i.test(detail.message)) {
      report.errors.push({channel:'renderer console',message:detail.message});
    }
  });
  win.on('show', () => { report.everVisible = true; finish(Error('Window became visible')); });
  win.webContents.once('did-finish-load', async () => {
    win.webContents.setBackgroundThrottling(false);
    const js = expr => win.webContents.executeJavaScript(expr,true);
    const until = async (expr, timeout = 240000) => {
      const start = Date.now();
      while(Date.now()-start<timeout) {
        if(report.errors.length) throw Error('Application request failed: '+JSON.stringify(report.errors.at(-1)));
        if(await js(expr)) return;
        await pause(200);
      }
      throw Error('Timed out waiting for '+expr);
    };
    const button = name => `[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(name)}||b.getAttribute('aria-label')===${JSON.stringify(name)}||b.querySelector('span')?.textContent.trim()===${JSON.stringify(name)})`;
    const click = async name => {
      await until(`!!(${button(name)}) && !(${button(name)}).disabled`);
      await js(`(()=>{const b=${button(name)};
        for(let p=b.parentElement;p;p=p.parentElement) if(p.tagName==='DETAILS') p.open=true;
        b.scrollIntoView({block:'nearest'}); b.click();
      })()`);
    };
    const ready = async () => {
      const stable = `document.querySelector('.rl-viewer')?.dataset.rendererReady==='true' &&
          !!document.querySelector('.rl-viewer')?.dataset.sourceDisplayIdentity &&
          document.querySelector('.rl-viewer')?.dataset.surfacesPending==='0' &&
          !document.querySelector('.viewer-loading,.rl-viewer-preparing,[data-layer-pending="true"],[data-approach-pending="true"]')`;
      const start = Date.now();
      while(Date.now()-start<300000) {
        if(!activeRequests && await js(stable)) {
          await pause(300);
          if(!activeRequests && await js(stable)) return;
        }
        await pause(200);
      }
      throw Error('Timed out waiting for source renderer');
    };
    const capture = async name => {
      await pause(350);
      assert.equal(win.isVisible(),false);
      const shot = await win.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true});
      const size = await js('({width:innerWidth,height:innerHeight})');
      const image = shot.getSize().width === size.width ? shot : shot.resize(size);
      fs.writeFileSync(path.join(output,name+'.png'),image.toPNG());
      report.captures.push({name,...size});
      console.log('Captured '+name);
      return image;
    };
    const both = async name => {
      for(const [width,height] of [[1440,900],[1050,720]]) {
        win.setContentSize(width,height);
        await until(`innerWidth===${width}&&innerHeight===${height}`);
        await capture(name+'-'+width);
      }
      win.setContentSize(1440,900);
    };
    const saveStudy = async () => {
      await click('Scans');
      await click('Save patient study file…');
      await until("document.querySelector('footer')?.textContent.includes('Patient study file saved')",600000);
      report.archive={name:saveTo,bytes:fs.statSync(saveTo).size};
      assert(report.archive.bytes>0);
    };
    const presentation = () => js(`(()=>{
      const v=document.querySelector('.rl-viewer');
      return {frame:window.operativeVerificationFrame,layers:JSON.parse(v.dataset.visibleLayerIds).sort()};
    })()`);
    try {
      report.rendererAsset = await js("document.querySelector('script[type=module][src]')?.src.split('/').pop()");
      await until(`document.querySelector('footer')?.textContent.includes('Add a source scan')`);
      if(mode==='compare-anatomy' || acquiredReview || process.argv.includes('--preserve-plan-view') || process.argv.includes('--source-support') || process.argv.includes('--visual-review') || process.argv.includes('--expect-presentation')) await js(`document.addEventListener('anatomy-frame',event=>{
        const {cameraState,cursor,sourceSignalFingerprint}=event.detail;
        window.operativeVerificationFrame={cameraState,cursor,sourceSignalFingerprint};
      })`);
      if(process.argv.includes('--visual-review'))await js(`(()=>{
        // Read the actual matrices sent to the color pass on both renderer builds.
        const gl=WebGL2RenderingContext.prototype,locations=new WeakMap(),buffers=new WeakMap();
        const locate=gl.getUniformLocation,bind=gl.bindFramebuffer,upload=gl.uniformMatrix4fv;
        gl.getUniformLocation=function(program,name){const result=locate.call(this,program,name);if(result)locations.set(result,name);return result};
        gl.bindFramebuffer=function(target,buffer){if(target!==this.READ_FRAMEBUFFER)buffers.set(this,buffer);return bind.call(this,target,buffer)};
        gl.uniformMatrix4fv=function(location,...rest){
          const name=locations.get(location);
          if(!buffers.get(this)&&(name==='viewMatrix'||name==='projectionMatrix')&&this.canvas.closest('.operative-field-canvas')){
            (window.operativeCameraMatrices??={})[name]=Array.from(rest[1]);
          }
          return upload.call(this,location,...rest);
        };
      })()`);
      if(mode==='source-switch'||process.argv.includes('--cortical-switch'))await js('('+require('./verify-cortical-switch.cjs').observeCorticalDraws+')()');
      await click('Open');
      if(mode==='reopen-decision') {
        const prior=flagValue('--decision-report');
        assert(prior?.startsWith(fixtureRoot) && archive);
        const expected=JSON.parse(fs.readFileSync(prior,'utf8'));
        assert(expected.passed && expected.engineArchive.sha256===report.openedArchive.sha256);
        await until(`document.body.innerText.includes(${JSON.stringify(document.patient_id)})`);
        await click('Rehearse');
        await until(`!!document.querySelector('[aria-label="Selected route engine result"]')`);
        assert.deepEqual(report.coldDecision,expected.engineInspection.engine.decision);
        report.resultText=await js(`document.querySelector('[aria-label="Selected route engine result"]').innerText`);
        assert.equal(report.resultText,expected.resultText);
        await js(`document.querySelector('[aria-label="Selected route engine result"]').scrollIntoView({block:'center'})`);
        await both('cold-selected-decision');
        finish();
        return;
      }
      await until(`JSON.parse(document.querySelector('.rl-viewer')?.dataset.sourceDisplayIdentity || '[]')[3]===${JSON.stringify(document.snapshot_id)}`);
      await ready();
      if(acquiredReview){
        report.coldReadyMs=performance.now()-started;
        report.gpu=await app.getGPUInfo('basic');
        report.processWorkingSetsKiB=app.getAppMetrics().map(({type,memory})=>({type,workingSet:memory.workingSetSize}));
        report.nativePresentation=await presentation();
        report.sourceDisplayIdentity=await js("JSON.parse(document.querySelector('.rl-viewer').dataset.sourceDisplayIdentity)");
      }
      const priorPresentation=flagValue('--expect-presentation');
      if(priorPresentation){
        assert(archive && priorPresentation.startsWith(fixtureRoot));
        const prior=JSON.parse(fs.readFileSync(priorPresentation,'utf8'));
        report.restoredPresentation=await presentation();
        assert.deepEqual(report.restoredPresentation,prior.savedPresentation);
      }
      if(process.argv.includes('--preserve-plan-view')){
        assert(archive,'Plan verification requires a portable workspace');
        const saved=JSON.parse(execFileSync('unzip',['-p',archive,'workspace-view.json'],{encoding:'utf8'}));
        const savedState=saved.states[saved.seriesId];
        report.coldPlanPresentation=await presentation();
        const frame=report.coldPlanPresentation.frame;
        assert(frame?.cameraState,'Missing reopened anatomical camera');
        for(const key of ['position','target','up']){
          assert(frame.cameraState[key].every((x,i)=>Math.abs(x-savedState.settings.camera[key][i])<.001),'Reopened plan camera '+key);
        }
        assert.deepEqual(frame.cursor,savedState.cursor);
        assert.equal(savedState.cranialView,'planning');
        const view=await js("JSON.parse(document.querySelector('.native-workspace').dataset.nativeViewState)");
        assert.equal(view.selectedApproachId,saved.selectedApproachId);
        assert.deepEqual(view.states[view.seriesId].visible,savedState.visible);
        report.coldPlanPresentation.savedView={series:saved.seriesId,route:saved.selectedApproachId,visible:savedState.visible,
          cursor:savedState.cursor,camera:savedState.settings.camera,mode:savedState.cranialView};

      }
      await click('Anatomy');
      await ready();
      if(mode==='compare-anatomy'){
        await require('./verify-anatomy-comparison.cjs').compareAnatomy({win,js,until,ready,click,capture,report,document});
        finish();return;
      }
      if(mode==='source-switch'||process.argv.includes('--cortical-switch')){
        await require('./verify-cortical-switch.cjs').verifyCorticalSwitch({js,until,click,ready,capture,report,document});
        if(mode==='source-switch'){finish();return;}
      }
      await both('anatomy');
      if(acquiredReview && mode==='baseline'){finish();return;}
      if(process.argv.includes('--visual-review')){
        await js("document.querySelector('.anatomy-presets button:nth-child(2)').click()");
        await ready();
        await capture('structures');
        report.nativeFrame=await js("window.operativeVerificationFrame");
        await click('Show structures only');
        await ready();await capture('surfaces-only');
        const surfaceFrame=await js('window.operativeVerificationFrame');
        assert.deepEqual(surfaceFrame,report.nativeFrame);
        if(process.argv.includes('--material-review')){
          await require('./verify-operative-materials.cjs').verifyOperativeMaterials({js,ready,presentation,capture,report,surfaceFrame});
        }
        await click('Show scan volume with structures');await ready();
        const geometry=`document.querySelector('select[aria-label="Native surface representation"]')`;
        if(await js(`!!(${geometry})`)){
          await js(`(${geometry}).value='source';(${geometry}).dispatchEvent(new Event('change',{bubbles:true}));`);
          await ready();await capture('structures-source');
          report.nativeSourceFrame=await js("window.operativeVerificationFrame");
          assert.deepEqual(report.nativeFrame,report.nativeSourceFrame);
          await js(`(${geometry}).value='conditioned';(${geometry}).dispatchEvent(new Event('change',{bubbles:true}));`);
          await ready();
        }
      }
      if(process.argv.includes('--source-support')) {
        await js("document.querySelector('.anatomy-presets button:nth-child(2)').click()");
        await ready();
        const toggle = `[...document.querySelectorAll('input[type=checkbox]')].find(i=>
          /Unfiltered SIAM estimates/i.test(i.getAttribute('aria-label')||i.closest('label')?.textContent||''))`;
        await until(`!!(${toggle})`);
        await js(`(()=>{for(let p=(${toggle}).parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;})()`);
        const read = () => js(`({checked:(${toggle}).checked,
          source:JSON.parse(document.querySelector('.rl-viewer').dataset.sourceDisplayIdentity),
          layers:JSON.parse(document.querySelector('.rl-viewer').dataset.visibleLayerIds||'[]'),
          frame:window.operativeVerificationFrame,rect:document.querySelector('.rl-viewer-anatomy-image').getBoundingClientRect().toJSON()})`);
        const status = await read();
        assert.equal(status.checked,false,'BITE opens with source-supported estimates');
        const cleaned = await capture('source-supported-anatomy');
        await js(`(${toggle}).click()`);
        await ready();
        const raw = await read();
        report.sourceComparison = {supportedView:status, unfilteredView:raw};
        assert(raw.checked,'The source QC control must apply explicit unfiltered estimates');
        const unfiltered = await capture('unfiltered-anatomy');
        assert.deepEqual(status.source.slice(0,4),raw.source.slice(0,4),'Linked source identity changed');
        assert.equal(status.source[4],false);
        assert.equal(raw.source[4],true);
        assert.deepEqual(status.layers,raw.layers,'Visibility changed in addition to filtering');
        assert(status.frame?.cameraState && raw.frame?.cameraState,'Missing submitted camera state');
        for(const key of ['position','target','up']) {
          assert(status.frame.cameraState[key].every((value,i)=>Math.abs(value-raw.frame.cameraState[key][i])<1e-4),
            'Filtering changed the '+key+' of the source comparison camera');
        }
        assert.deepEqual(status.frame.cursor,raw.frame.cursor);
        assert.deepEqual(status.frame.sourceSignalFingerprint,raw.frame.sourceSignalFingerprint);
        const r=status.rect,region={x:Math.ceil(r.x+r.width*.05),y:Math.ceil(r.y+r.height*.07),
          width:Math.floor(r.width*.90),height:Math.floor(r.height*.82)};
        assert.deepEqual(raw.rect,status.rect);
        const left=cleaned.crop(region).toBitmap(),right=unfiltered.crop(region).toBitmap();
        assert.equal(left.length,right.length);
        let changed=0;
        for(let i=0;i<left.length;i+=4)
          if(Math.max(Math.abs(left[i]-right[i]),Math.abs(left[i+1]-right[i+1]),Math.abs(left[i+2]-right[i+2]))>=8)changed++;
        assert(changed>100,'Meaningful 3D geometry change must be visible under identical source, layers and viewport');
        Object.assign(report.sourceComparison,{changedPixels:changed,pixels:left.length/4});
        await js(`(${toggle}).click()`);
        await ready();
        assert.equal((await read()).checked,false);
      }

      if(mode==='preview' && document.series.length>1 && !acquiredReview && !process.argv.includes('--cortical-switch')) {
        const reference = document.reference_series;
        await js(studyScanSelectionScript(reference));
        await ready();
        await js("document.querySelector('.anatomy-presets button:nth-child(2)').click()");
        await ready();
        const readModel = () => js(`({source:document.querySelector('.native-workspace').dataset.nativeSeries,
          summary:document.querySelector('.patient-model-summary')?.textContent,
          frame:document.querySelector('.patient-model-summary')?.dataset.frame,
          layers:JSON.parse(document.querySelector('.rl-viewer').dataset.visibleLayerIds || '[]')})`);
        report.model = [await readModel()];
        assert.equal(report.model[0].source,reference);
        assert.equal(report.model[0].frame,'patient');
        const secondary = document.series.find(row=>row.id!==reference).id;
        await js(studyScanSelectionScript(secondary));
        await ready();
        report.model.push(await readModel());
        assert.equal(report.model[1].source,secondary);
        if(report.model[1].frame==='patient') {
          assert.equal(report.patientViews[reference].contextHash,report.patientViews[secondary].contextHash);
          assert.deepEqual(report.patientViews[reference].shape,report.patientViews[secondary].shape);
          assert.deepEqual(report.patientViews[reference].affine,report.patientViews[secondary].affine);
          if(document.series[0].original.sha256!==document.series[1].original.sha256)
            assert.notEqual(report.patientViews[reference].sourceHash,report.patientViews[secondary].sourceHash);
          assert.deepEqual([...report.model[0].layers].sort(),[...report.model[1].layers].sort());
        }
        await both('linked-second-scan');
        await js(studyScanSelectionScript(reference));
        await ready();
      }

      console.log('Opening Plan');
      await click('Plan');
      await until("!!document.querySelector('.native-approaches') && !document.querySelector('.approach-computing')");
      await both('plan');
      if(mode==='preview' && process.argv.includes('--reuse-selection')) {
        await until("!!document.querySelector('.approach-card.selected')");
        report.selected = await js("document.querySelector('.approach-card.selected').dataset.approachId");
      } else if(['prepare','preview'].includes(mode)) {
        const target = targetLayer;
        assert(target,'Study requires a target');
        await until(`[...document.querySelectorAll('select[aria-label="Planning target"] option')].some(o=>o.value===${JSON.stringify(target.id)}&&!o.disabled)`);
        await js(inputScript('Planning target',target.id));
        await until(`!!(${button('Suggest approaches')}) && !(${button('Suggest approaches')}).disabled`);
        const pending = (report.plans??[]).filter(r=>r.action==='suggest').length;
        await click('Suggest approaches');
        const start = Date.now();
        while((report.plans??[]).filter(r=>r.action==='suggest').length===pending) {
          if(Date.now()-start>720000 || report.errors.length) throw Error('Source suggestion did not finish');
          await pause(350);
        }
        await ready();
        let cards = await js("Array.from(document.querySelectorAll('.approach-card'),b=>b.dataset.approachId)");
        if(!cards.length) {
          report.suggestedCount = 0;
          await click('Create manual approach');
          await until("!!document.querySelector('.approach-card')");
          await ready();
          cards = await js("Array.from(document.querySelectorAll('.approach-card'),b=>b.dataset.approachId)");
        } else report.suggestedCount = report.plans.findLast(row=>row.action==='suggest').routes.filter(r=>r.origin==='bounded-native-entry-patch').length;
        if(process.argv.includes('--distinct') || process.argv.includes('--try-distinct')) {
          const previous = report.plans.findLast(result=>result.routes?.length).routes.filter(row=>row.origin!=='native-source-entry-windows');
          assert(previous.length,'Need pre-existing patch approaches for the preservation check');
          await click('Suggest distinct entries');
          await until("!document.querySelector('.approach-computing')",720000);
          await ready();
          const measured = report.plans.findLast(result=>result.action==='suggestDistinctEntries');
          assert(measured,'The distinct-entry action must reach the native engine');
          for(const before of previous) {
            const after = measured.routes.find(row=>row.id===before.id);
            assert.deepEqual(after,before,'Earlier patch approach changed');
          }
          const distinct = measured.routes.filter(row=>row.origin==='native-source-entry-windows');
          if(process.argv.includes('--distinct')) assert(distinct.length>1,'Study did not admit separated entries');
          const angle = (a,b) => {
            const u=a.entry.map((v,i)=>v-a.target[i]),v=b.entry.map((n,i)=>n-b.target[i]);
            const cosine=u.reduce((sum,n,i)=>sum+n*v[i],0)/(Math.hypot(...u)*Math.hypot(...v));
            return Math.acos(Math.max(-1,Math.min(1,cosine)))*180/Math.PI;
          };
          report.distinctPairs=[];
          for(let i=0;i<distinct.length;i++) for(let j=0;j<i;j++) {
            const gap=Math.hypot(...distinct[i].entry.map((n,k)=>n-distinct[j].entry[k]));
            const degrees=angle(distinct[i],distinct[j]);
            assert(gap>1 && degrees>1,'Distinct entries need separate opening positions and axes');
            report.distinctPairs.push({routes:[distinct[i].id,distinct[j].id],entryGapMm:gap,angleDegrees:degrees});
          }
          if(distinct.length) cards=distinct.map(row=>row.id);
        }
        const selected = await js("document.querySelector('.approach-card.selected')?.dataset.approachId");
        let route = cards.find(id=>id!==selected) ?? cards[0];
        if(process.argv.includes('--inspect-shortest')) {
          assert(process.argv.includes('--distinct'),'Choose separated entry windows for the length comparison');
          const windows=report.plans.findLast(row=>row.action==='suggestDistinctEntries').routes.filter(row=>cards.includes(row.id));
          assert(windows.length>1 && windows.every(row=>Number.isFinite(row.lengthMm)));
          route=windows.sort((a,b)=>a.lengthMm-b.lengthMm || a.id.localeCompare(b.id))[0].id;
          report.selectionCriterion='Shortest measured path among the distinct windows, for geometric inspection only';
        }
        await js(`document.querySelector('.approach-card[data-approach-id="${route}"]').click()`);
        await until(`document.querySelector('.approach-card.selected')?.dataset.approachId===${JSON.stringify(route)}`);
        report.selected = route;
        if(process.argv.includes('--distinct')) {
          const overlayIds = () => js("JSON.parse(document.querySelector('.rl-viewer').dataset.approachRoutes).map(r=>r.route_id)");
          await until(`JSON.parse(document.querySelector('.rl-viewer').dataset.approachRoutes).some(r=>r.selected&&r.route_id===${JSON.stringify(route)})`);
          report.windowOverlay=await overlayIds();
          assert.deepEqual([...report.windowOverlay].sort(),[...cards].sort());
          const patch=report.plans.findLast(row=>row.action==='suggest').routes.find(row=>row.origin==='bounded-native-entry-patch');
          if(patch) {
            console.log('Checking return to patch routes');
            await ready();
            await until(`!document.querySelector('.approach-card[data-approach-id="${patch.id}"]').disabled`);
            await js(`document.querySelector('.approach-card[data-approach-id="${patch.id}"]').click()`);
            await until(`JSON.parse(document.querySelector('.rl-viewer').dataset.approachRoutes).some(r=>r.selected&&r.route_id===${JSON.stringify(patch.id)})`);
            report.patchOverlay=await overlayIds();
            assert(report.patchOverlay.every(id=>!report.windowOverlay.includes(id)));
            await ready();
            await until(`!document.querySelector('.approach-card[data-approach-id="${route}"]').disabled`);
            await js(`document.querySelector('.approach-card[data-approach-id="${route}"]').click()`);
            await until(`JSON.parse(document.querySelector('.rl-viewer').dataset.approachRoutes).some(r=>r.selected&&r.route_id===${JSON.stringify(route)})`);
          }
        }
        await both('selected');
      }
      if(mode==='preview') {
        const steps = [['bone','Bone flap'],['dura','Dural opening'],['brain','Brain retraction'],['tumor','Tumor aspiration']];
        const previewReady = async () => {
          await until("!!document.querySelector('.operative-field-canvas canvas') || !!document.querySelector('.operative-placeholder[role=alert]')");
          assert(await js("!!document.querySelector('.operative-field-canvas canvas')"),await js("document.querySelector('.operative-placeholder')?.innerText"));
          assert.equal(await js("document.querySelector('.operative-field-canvas [role=alert]')?.innerText"),undefined);
        };
        let intro=null;
        if(process.argv.includes('--video')) {
          await recordingSize(win,js);
          await ready();
          if(process.argv.includes('--visual-review')){
            const start=await presentation();
            const center=await js(`(()=>{const r=document.querySelector('.rl-viewer-anatomy-image').getBoundingClientRect();
              return {x:Math.round(r.x+r.width/2),y:Math.round(r.y+r.height/2)};})()`);
            await js(`window.operativeOrbitFrames=[];document.addEventListener('anatomy-frame',e=>{
              if(e.detail.interactive)window.operativeOrbitFrames.push(performance.now());
            });`);
            win.webContents.sendInputEvent({type:'mouseDown',...center,button:'left',clickCount:1});
            for(let i=1;i<=60;i++){
              win.webContents.sendInputEvent({type:'mouseMove',x:center.x+i*2,y:center.y,modifiers:['leftButtonDown']});
              await pause(16);
            }
            win.webContents.sendInputEvent({type:'mouseUp',x:center.x+120,y:center.y,button:'left',clickCount:1});
            await ready();
            const end=await presentation();
            assert.notDeepEqual(end.frame.cameraState.position,start.frame.cameraState.position);
            assert.deepEqual(end.layers,start.layers);
            assert.equal(end.frame.sourceSignalFingerprint,start.frame.sourceSignalFingerprint);
            report.orbitFrameTimes=await js('window.operativeOrbitFrames');
            assert(report.orbitFrameTimes.length>1);
            await capture('rotated-structures');
          }
          if(process.argv.includes('--preserve-plan-view')){
            report.cranialView=await js(`document.querySelector('[aria-label="Cranial view"]')?.value`);
            assert.equal(report.cranialView,'planning');
            report.moviePlanPresentation=await presentation();
            assert.deepEqual(report.moviePlanPresentation.layers,report.coldPlanPresentation.layers);
          }else{
            await click('Layout');
            await js(`(()=>{const select=document.querySelector('select[aria-label="Camera preset"]');
              select.value='left';select.dispatchEvent(new Event('change',{bubbles:true}));})()`);
            await click('Close view controls');
          }
          await ready();
          await capture('selected-approach-1080p');
          intro=await framePixels(win);
        }
        if(saveTo&&process.argv.includes('--visual-review')){
          report.savedPresentation=await presentation();
          await saveStudy();
          await click('Plan');await ready();
        }
        if(process.argv.includes('--acceptance-views'))await js('('+require('./verify-operative-camera.cjs').observeField+')()');
        await click('Preview operative steps');
        await previewReady();
        const field = report.plans.findLast(p=>p.action==='operativeField');
        assert.equal(field.approach,report.selected);
        assert.equal(field.target,targetLayer.id);
        if(boundEpisode){
          await require('./verify-acquired-engine.cjs').captureBoundStudy({episode:boundEpisode,report,document,win,js,until,click,capture,field});
          finish();return;
        }
        if(process.argv.includes('--field-review')){
          const acceptance=process.argv.includes('--acceptance-views');
          const bothSweeps=process.argv.includes('--capture-both-sweeps');
          const sweepValue=()=>js("document.querySelector('.operative-sweep-choice select')?.value??'original spiral (single option)'");
          const chooseSweep=async value=>{
            await js(`(()=>{const choice=document.querySelector('.operative-sweep-choice select');
              choice.value=${JSON.stringify(value)};choice.dispatchEvent(new Event('change',{bubbles:true}))})()`);
            await until(`document.querySelector('.operative-sweep-choice select').value===${JSON.stringify(value)}`);
            await previewReady();
          };
          report.initialSweep=await sweepValue();
          if(bothSweeps)assert.equal(report.initialSweep,'overlapping');
          if((bothSweeps||process.argv.includes('--original-sweep'))&&report.initialSweep==='overlapping')await chooseSweep('spiral');
          report.comparisonSweep=report.sweep=await sweepValue();
          const reference=flagValue('--match-field-viewports');
          if(reference)assert(reference.startsWith(fixtureRoot));
          const match=reference?JSON.parse(fs.readFileSync(reference)).corridorViews:null;
          report.matchedFieldViewports=reference;
          const cameraReport=flagValue('--match-operative-cameras');
          if(cameraReport)assert(cameraReport.startsWith(fixtureRoot));
          const cameras=cameraReport?JSON.parse(fs.readFileSync(cameraReport)).corridorViews:null;
          const movieReference=flagValue('--match-movie-poses');
          if(movieReference)assert(movieReference.startsWith(fixtureRoot));
          const moviePoses=movieReference?JSON.parse(fs.readFileSync(movieReference)):null;
          report.corridorViews=await verifyOperativeCorridor({win,js,until,capture,match,cameras,moviePoses,acceptance,acquiredReview,ablations:process.argv.includes('--geometry-ablations')});
          if(process.argv.includes('--expect-vascular-estimate')){
            assert(field.layers.some(l=>l.role==='vessels'&&l.model&&l.vascularContextOnly&&l.voxels>0));
            assert.equal(report.corridorViews['endpoint-1440'].vesselLegend?.label,'Vascular estimate');
          }
          if(bothSweeps){
            await chooseSweep('overlapping');
            report.sweep=await sweepValue();
            report.defaultCorridorViews=await verifyOperativeCorridor({win,js,until,acceptance,expectEntry:true,expectOuterHead:process.argv.includes('--expect-outer-head'),capture:name=>capture('default-'+name)});
          }
          if(process.argv.includes('--review-video')){
            report.recording=await recordOperativePlayback(win,js,output,intro,true);
            report.video=report.recording.filename;
          }
          finish();return;
        }
        if(process.argv.includes('--visual-review')){
          const setting=label=>`[...document.querySelectorAll('.operative-options label')].find(l=>l.textContent.trim()===${JSON.stringify(label)})?.querySelector('input')`;
          const toggle=async(label,checked)=>{
            const expr=setting(label);
            const present=await js(`!!(${expr})`);
            if(present&&await js(`(${expr}).checked`)!==checked)await js(`(${expr}).click()`);
            return present;
          };
          const fieldState=()=>js(`(()=>{
            const el=document.querySelector('.operative-playback'),canvas=document.querySelector('.operative-field-canvas canvas');
            return {phase:el.dataset.operativePhase,progress:el.dataset.operativeProgress,
              removed:Number(el.querySelector('[data-removed-cells]')?.dataset.removedCells??0),
              submittedCamera:window.operativeCameraMatrices,
              camera:canvas.dataset.camera?JSON.parse(canvas.dataset.camera):null,
              surroundingLayers:canvas.parentElement.dataset.surroundingLayers?JSON.parse(canvas.parentElement.dataset.surroundingLayers):null,
              extent:canvas.getBoundingClientRect().toJSON()};
          })()`);
          report.layout=await js(`(()=>{
            const el=document.querySelector('.operative-legend'),style=getComputedStyle(el);
            return {position:style.position,rect:el.getBoundingClientRect().toJSON(),styles:Object.fromEntries(['top','right','bottom','left','width','height','minHeight','display','alignSelf'].map(k=>[k,style[k]])),
              matches:Array.from(document.styleSheets).flatMap(s=>Array.from(s.cssRules).filter(r=>r.selectorText&&el.matches(r.selectorText)).map(r=>r.cssText)).slice(-30)}
          })()`);
          await capture('opening-default');
          const fit=button('Fit source anatomy');
          if(await js(`!!(${fit})`)){
            await click('Fit source anatomy');await capture('whole-source');
            await click('Follow operative step');
          }else if(await js(`!!document.querySelector('[aria-label="Operative field view"] option[value="operative"]')`)){
            await setFieldView(js,'exterior');await capture('whole-source');
            await setFieldView(js,'operative');
          }
          await setFieldView(js,'section');
          await click('Tumor aspiration');
          await capture('teaching-end');
          report.teachingState=await fieldState();
          if(process.argv.includes('--material-review')){
            const size=win.getContentSize();
            win.setContentSize(1050,720);
            await until('innerWidth===1050&&innerHeight===720');
            await capture('cutaway-compact');
            report.compactCutaway=await fieldState();
            assert.equal(report.compactCutaway.removed,report.teachingState.removed);
            win.setContentSize(...size);
            await until(`innerWidth===${size[0]}&&innerHeight===${size[1]}`);
          }

          await setFieldView(js,'exterior');
          await capture('exterior-end');
          report.exteriorState=await fieldState();
          if(await toggle('Field voxel faces',true)){
            await capture('exterior-source-voxels');
            report.rawState=await fieldState();
            assert.deepEqual(report.rawState,report.exteriorState);
            await toggle('Field voxel faces',false);
          }
          await click('Reset operative steps');
          await capture('exterior-start');
          const canvas="document.querySelector('.operative-field-canvas canvas')";
          if(await js(`(${canvas}).getAttribute('aria-label').includes('Enter inspects')`)){
            await js(`(${canvas}).dispatchEvent(new KeyboardEvent('keydown',{key:'Enter'}))`);
            await until("!!document.querySelector('.operative-picked')");
            report.pickedTissue=await js("document.querySelector('.operative-picked').innerText");
            assert.match(report.pickedTissue,/Rest position RAS/);assert.match(report.pickedTissue,/voxel/);
            await capture('picked-source');
            await click('Close picked tissue');
          }
          report.framesMs=await js(`new Promise(resolve=>{
            const times=[];let previous;
            const tick=now=>{if(previous!==undefined)times.push(now-previous);previous=now;
              document.querySelector('.operative-field-canvas canvas').dispatchEvent(new KeyboardEvent('keydown',{key:times.length%2?'ArrowLeft':'ArrowRight'}));
              if(times.length<40)requestAnimationFrame(tick);else resolve(times)};
            requestAnimationFrame(tick);
          })`);
          if(process.argv.includes('--review-video')){
            const readReference=()=>js(`(()=>{
              const v=document.querySelector('[aria-label="Selected approach on intact anatomy"]');
              return v&&{identity:v.dataset.sourceDisplayIdentity,caseHash:v.dataset.sourceCaseHash,
                selected:JSON.parse(v.dataset.approachRoutes).find(r=>r.selected)?.route_id,
                viewport:v.querySelector('.rl-viewer-anatomy-image').getBoundingClientRect().toJSON()};
            })()`);
            report.reference=await readReference();
            if(report.reference){
              assert.equal(report.reference.selected,report.selected);
              assert.equal(JSON.parse(report.reference.identity)[3],document.snapshot_id);
            }
            const chapters=await js("Array.from(document.querySelectorAll('.operative-chapters button:not(:disabled)'),b=>b.getAttribute('aria-label'))");
            const seek=async(position)=>{await js(inputScript('Operative timeline',position));await pause(300);};
            report.stageViews={};
            for(const [name,position] of [['exposed',chapters.indexOf('Dural opening')+1],['instrument',chapters.indexOf('Tumor aspiration')+.15],['mid-resection',chapters.indexOf('Tumor aspiration')+.55],['final-cavity',chapters.length]]){
              if(position<=0)continue;
              await seek(position);await capture(name);report.stageViews[name]=await fieldState();
            }
            if(chapters.includes('Tumor aspiration')){
              await seek(chapters.indexOf('Tumor aspiration')+.55);
              assert.deepEqual(await fieldState(),report.stageViews['mid-resection']);
            }
            await seek(0);
            assert.equal((await fieldState()).removed,0);
            const size=win.getContentSize();
            win.setContentSize(1050,720);await until('innerWidth===1050&&innerHeight===720');
            await capture('operative-compact');
            report.compactReference=await readReference();
            win.setContentSize(...size);
            await setFieldView(js,'follow');
            report.recording=await recordOperativePlayback(win,js,output,intro);
            report.video=report.recording.filename;
          }
          finish();return;
        }
        if(process.argv.includes('--source-support')) {
          assert(field.spacingMm<=.35,'Expected the refined BITE field');
          const source=document.series.find(row=>row.id===field.intensity?.series);
          assert(source,'Expected an acquired intensity source');
          assert.equal(field.intensity.sourceHash,source.original.sha256);
          assert.equal(field.intensity.dataHash,source.data.sha256);
          assert(field.intensity.covered>0 && field.intensity.signalHash);
          const skull=document.layers.find(l=>l.kind==='skull'&&l.provenance?.model==='siam-v2');
          assert(skull,'Expected a native SIAM skull estimate');
          const readings=report.anatomyViews.flatMap(view=>view.layers).filter(l=>l.id===skull.id);
          const rawSkull=readings.find(l=>l.sourceSupport?.status==='unfiltered')?.sourceSupport;
          const supportedSkull=readings.find(l=>l.sourceSupport?.status==='supported')?.sourceSupport;
          const operativeSkull=field.layers.find(l=>l.id===skull.id)?.sourceSupport;
          for(const record of [rawSkull,supportedSkull,operativeSkull]) {
            assert(record,'Expected skull provenance in the raw, supported and operative views');
            assert.equal(record.sourceSha256,source.original.sha256);
            assert.equal(record.rawDataSha256,skull.data.sha256);
          }
          assert.equal(rawSkull.sampledDataSha256,skull.data.sha256);
          assert.equal(operativeSkull.sampledDataSha256,supportedSkull.sampledDataSha256);
          assert.notEqual(supportedSkull.sampledDataSha256,skull.data.sha256);
          assert(supportedSkull.hiddenVoxels>0 && supportedSkull.retainedVoxels>0);
          assert.equal(supportedSkull.sourceVoxels,supportedSkull.retainedVoxels+supportedSkull.hiddenVoxels);
          const tumor=field.layers.find(l=>l.id===targetLayer.id);
          assert(tumor?.voxels>0 && !tumor.sourceSupport && /supplied/i.test(tumor.evidence));
          report.sourceComparison.skullSupport={raw:rawSkull,supported:supportedSkull,operative:operativeSkull};

          await js("document.querySelector('.operative-sources').open=true");
          const evidenceText=await js("document.querySelector('.operative-sources').innerText");
          const superficial=evidenceText.match(/([\d,]+) superficial vessel samples are hidden/);
          assert(superficial && Number(superficial[1].replaceAll(',',''))>0,
            'BITE surface-vessel handling needs an explicit count in Field evidence');
          report.sourceComparison.superficialVesselSamples=Number(superficial[1].replaceAll(',',''));
          await capture('operative-source-provenance');
          await js("document.querySelector('.operative-sources').open=false");
        }
        const captureField = async name => {
          const im = await capture(name);
          const rect = await js(`(()=>{const r=document.querySelector('.operative-field-canvas canvas').getBoundingClientRect();
            return {x:Math.ceil(r.left+r.width*.04),y:Math.ceil(r.top+r.height*.04),width:Math.floor(r.width*.92),height:Math.floor(r.height*.86)};})()`);
          assert(rect.width>200 && rect.height>110);
          return im.crop(rect).toBitmap();
        };
        const delta = (a,b) => {
          assert.equal(a.length,b.length);
          let changed = 0;
          for(let i=0;i<a.length;i+=4) if(Math.max(Math.abs(a[i]-b[i]),Math.abs(a[i+1]-b[i+1]),Math.abs(a[i+2]-b[i+2]))>=10) changed++;
          return changed;
        };
        report.animation = [];
        for(const [width,height] of [[1440,900],[1050,720]]) {
          win.setContentSize(width,height);
          await until(`innerWidth===${width}&&innerHeight===${height}`);
          if(await js(`!(${button('Reset operative steps')}).disabled`)) await click('Reset operative steps');
          let prior = await captureField('intact-'+width);
          assert.equal(await js("Number(document.querySelector('[data-removed-cells]')?.dataset.removedCells ?? 0)"),0);
          const readout = {width,stages:[],evidence:await js("document.querySelector('.operative-sources').textContent")};
          for(const [role,name] of steps) {
            const enabled = await js(`!(${button(name)}).disabled`);
            if(!enabled && role==='tumor' && field.stations.tumor!==null) {
              assert.match(await js(`(${button(name)}).textContent`),/No eligible tip position/);
            } else assert.equal(enabled,field.stations[role]!==null);
            if(!enabled) continue;
            await click(name);
            await until(`document.querySelector('.operative-playback').dataset.operativePhase===${JSON.stringify(role)}`);
            if(process.argv.includes('--stage-detail')) {
              const endpoint=Number(await js(`document.querySelector('[aria-label="Operative timeline"]').value`));
              await js(inputScript('Operative timeline',endpoint-.55));
              await until(`document.querySelector('.operative-playback').dataset.operativeProgress==='0.450'`);
              await capture(role+'-mid-'+width);
              await click(name);
              await until(`document.querySelector('.operative-playback').dataset.operativeProgress==='1.000'`);
            }
            const frame = await captureField(role+'-'+width);
            const changed = delta(prior,frame);
            assert(changed>30,'Missing rendered transition at '+role+' ('+changed+' changed pixels)');
            const removed = await js("Number(document.querySelector('[data-removed-cells]')?.dataset.removedCells ?? 0)");
            if(role==='tumor') assert(removed>0,'Target endpoint must remove modeled label cells');
            else assert.equal(removed,0);
            readout.stages.push({role,changedPixels:changed,removed});
            prior = frame;
          }
          if(process.argv.includes('--inspect-access')){
            await setFieldView(js,'exterior');
            const exterior=await captureField('exterior-access-'+width);
            const identity=()=>js("({phase:document.querySelector('.operative-playback').dataset.operativePhase,progress:document.querySelector('.operative-playback').dataset.operativeProgress,removed:document.querySelector('[data-removed-cells]')?.dataset.removedCells})");
            const before=await identity();
            await setFieldView(js,'section');
            const teaching=await captureField('teaching-cutaway-'+width);
            assert.deepEqual(await identity(),before);
            assert(await js("!!document.querySelector('.operative-overview canvas')&&document.querySelector('.operative-scale').textContent.includes('mm')"));
            readout.teachingCutaway={...before,changedPixels:delta(exterior,teaching),scale:await js("document.querySelector('.operative-scale').textContent")};
            assert(readout.teachingCutaway.changedPixels>30);
            await setFieldView(js,'follow');
          }
          const timeline = 'input[aria-label="Operative timeline"]';
          await js(inputScript('Operative timeline',0.55));
          await until(`Number(document.querySelector(${JSON.stringify(timeline)}).value)===.55`);
          await click('Play steps');
          await until(`Number(document.querySelector(${JSON.stringify(timeline)}).value)>.64`,10000);
          await click('Pause steps');
          readout.playedTo = Number(await js(`document.querySelector(${JSON.stringify(timeline)}).value`));
          if(field.stations.brain!==null) {
            const toggle = () => js("[...document.querySelectorAll('.operative-options label')].find(l=>l.textContent.trim()==='Include retraction').querySelector('input').click()");
            await toggle();
            assert(await js(`(${button('Brain retraction')}).disabled`));
            await toggle();
          }
          readout.overflow=await js("document.documentElement.scrollWidth>innerWidth");
          assert.equal(readout.overflow,false);
          report.animation.push(readout);
        }
        if(process.argv.includes('--video')) {
          report.recording=await recordOperativePlayback(win,js,output,intro);
          report.video=report.recording.filename;
          const max=Number(await js(`document.querySelector('[aria-label="Operative timeline"]').max`));
          await js(inputScript('Operative timeline',max-1+.05));
          assert.equal(await js("Number(document.querySelector('[data-removed-cells]').dataset.removedCells)"),0);
          await click('Reset operative steps');
        }
        await click('Return to plan');
        await ready();
        assert.equal(await js("document.querySelector('.approach-card.selected')?.dataset.approachId"),report.selected);
        const identity=await js("JSON.parse(document.querySelector('.rl-viewer').dataset.sourceDisplayIdentity)");
        assert.equal(identity[3],document.snapshot_id);
        await click('Rehearse');
        await previewReady();
        await click('Return to plan');
        await ready();
        await click('Anatomy');
        await ready();
        await capture('restored-anatomy');
        if(saveTo) {
          await saveStudy();
        } else if(process.argv.includes('--save-view')) {
          await click('Save');
          await until("document.querySelector('footer')?.textContent.includes('Workspace saved')");
        }
      }
      if(mode==='reopen') {
        await until("!!document.querySelector('.approach-card.selected')");
        report.selected=await js("document.querySelector('.approach-card.selected').dataset.approachId");
        await click('Rehearse');
        await until("!!document.querySelector('.operative-field-canvas canvas')");
        await both('reopened');
        if(process.argv.includes('--engine-inspection')) {
          await click('Open instrument environment');
          const forecastSelector='[aria-label="Selected route forecast"]';
          await until(`!!document.querySelector(${JSON.stringify(forecastSelector)})`);
          await js(`document.querySelector(${JSON.stringify(forecastSelector)}).scrollIntoView({block:'center'})`);
          report.forecast=await js(`document.querySelector(${JSON.stringify(forecastSelector)}).innerText`);
          assert(report.forecast.includes('STOP before access'));
          await both('selected-route-forecast');
          await click('Inspect approach in engine');
          await until("!!document.querySelector('.surgical-panel[data-state-id]')",300000);
          const check=report.engineInspection;
          assert.equal(check.request.approach.approachId,report.selected);
          assert.equal(check.request.approach.snapshotId,document.snapshot_id);
          assert.equal(check.request.approach.targetLayerId,check.request.layers.tumor);
          assert.equal(check.engine.position,0);
          assert.equal(check.engine.metrics.targetRemovedMm3,0);
          assert.equal(check.engine.metrics.nonTargetRemovedMm3,0);
          assert.equal(check.engine.admission?.available,false);
          const decision=check.engine.decision;
          assert.equal(decision.routeId,report.selected);
          assert.equal(decision.sourceSnapshotId,document.snapshot_id);
          assert.equal(decision.measurementHash,check.request.approach.measurementHash);
          assert.equal(decision.caseHash,check.caseHash);
          assert.equal(decision.stateId,check.engine.state);
          assert.equal(decision.measured.actions,0);
          assert(decision.measured.originalTargetResidualMm3>0);
          await until(`!!document.querySelector('[aria-label="Selected route engine result"]')`);
          await js(`document.querySelector('[aria-label="Selected route engine result"]').scrollIntoView({block:'center'})`);
          report.resultText=await js(`document.querySelector('[aria-label="Selected route engine result"]').innerText`);
          assert(report.resultText.includes('target remains'));
          await both('planned-engine-inspection');
          if(saveTo) {
            await click('Save');
            await until("document.body.innerText.includes('Workspace saved locally')");
            assert(fs.statSync(saveTo).size>0);
            report.engineArchive={path:saveTo,sha256:hash(fs.readFileSync(saveTo))};
          }
        }
      }
      if(mode==='stale') {
        const preview = await js(`!!(${button('Preview operative steps')}) && !(${button('Preview operative steps')}).disabled`);
        assert.equal(preview,false);
        await click('Rehearse');
        await until("!!document.querySelector('.operative-inspector')");
        assert.equal(await js("!!document.querySelector('.operative-playback')"),false);
        report.unavailable = await js("document.querySelector('.operative-inspector').innerText");
        await both('stale-route');
      }
      if(mode==='no-target') {
        assert.equal(await js(`(${button('Suggest approaches')}).disabled`),true);
        await click('Rehearse');
        await until("!!document.querySelector('.operative-inspector')");
        assert.equal(await js("!!document.querySelector('.operative-playback')"),false);
        report.unavailable = await js("document.querySelector('.operative-inspector').innerText");
        await both('no-target');
      }
      if(mode==='baseline') {
        await click('Rehearse');
        await pause(700);
        await both('rehearse');
      }
      finish();
    } catch(error) {
      report.body = await js('document.body.innerText').catch(()=>null);
      await capture('failure').catch(()=>{});
      finish(error);
    }
  });
});
require('./main.cjs');
