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
const nativeStudy=process.env.DATA_REBUILD_INSPECT_STUDY==='1'?JSON.parse(fs.readFileSync(input,'utf8')):null;
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
let nativeVolume;
ipcMain.handle=(name,handler)=>originalHandle(name,async(...args)=>{
  const started=performance.now();
  try{const result=await handler(...args);
    if(targetId&&['research:startupWorkspace','research:openListedWorkspace','research:openWorkspace'].includes(name)&&result)ctaPayload=result;
    if(nativeStudy&&name==='research:anatomyView'&&result)nativeVolume=result;
    report.requests.push({operation:name,command:args[1]?.command,ms:performance.now()-started,
    stateId:result?.view?.stateId,caseHash:result?.caseHash,reason:result?.reason,status:result?.status});return result;}
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
    const completedFrame=async stateId=>{
      const state=JSON.stringify(stateId);
      await wait(`(()=>{const p=window.__dataRebuildFrames,s=${state},v=p?.publications.findLast(v=>v.surgicalStateId===s);return document.querySelector('.surgical-panel')?.dataset.stateId===s&&!!v&&p.gpu.some(g=>g.serial===v.frameSerial&&g.surgicalStateId===s&&g.sourceCaseHash===v.sourceCaseHash&&g.completed&&g.at<=v.at)})()`);
      return js(`(()=>{const p=window.__dataRebuildFrames,v=p.publications.findLast(v=>v.surgicalStateId===${state});return{publication:v,gpu:p.gpu.findLast(g=>g.serial===v.frameSerial&&g.surgicalStateId===v.surgicalStateId&&g.completed)}})()`);
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
      const frame=probe.frames.findLast(f=>f.sourceCaseHash===viewer.dataset.sourceCaseHash&&probe.completed.has(f.serial));
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
    const gpuSourceReady=()=>wait(`window.__dataRebuildCtaFrames.frames.some(f=>window.__dataRebuildCtaFrames.completed.has(f.serial)
      &&f.sourceCaseHash===document.querySelector('.rl-viewer')?.dataset.sourceCaseHash)`);
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
    const inspectNativeStudy=async()=>{
      await click('Scans');await gpuSourceReady();
      const selected=await js(`(()=>{
        const shell=document.querySelector('.native-workspace'),viewer=document.querySelector('.rl-viewer');
        return {series:shell?.dataset.nativeSeries,display:JSON.parse(viewer.dataset.sourceDisplayIdentity),
          panes:document.querySelectorAll('.rl-viewer-slice-image').length};
      })()`);
      const series=nativeStudy.series.find(row=>row.id===selected.series);
      assert(series&&nativeVolume,'The app must load a declared display series');
      assert.equal(selected.panes,3);
      assert.equal(selected.display[1],series.id);
      assert.equal(selected.display[2],nativeVolume.mri.sha256);
      assert.equal(selected.display[3],nativeStudy.snapshot_id);
      const rendered=await sourceState();
      assert.deepEqual(rendered.sourceShape,nativeVolume.shape);
      assert(rendered.sourceSignal);
      const surfaces=nativeVolume.compartments.map(layer=>({
        id:layer.name,kind:layer.kind,volumeMm3:layer.volumeMm3,maskSha256:layer.mask?.sha256,
        sourceMeshSha256:layer.surface?.source?.sha256,displayMeshSha256:layer.surface?.positions?.sha256,
        diagnostics:layer.surfaceDiagnostics,
      }));
      report.nativeStudy={patientId:nativeStudy.patient_id,snapshotId:nativeStudy.snapshot_id,id:series.id,
        modality:series.modality,displayNativeGrid:series.data.shape,publisherSource:series.publisher_source??null,
        gridSampling:series.display_sampling_ijk??null,rendered,surfaces};
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
    try{
      if(targetId||nativeStudy)await js(`(()=>{
        const probe=window.__dataRebuildCtaFrames={frames:[],completed:new Set()};
        document.addEventListener('anatomy-frame',e=>{probe.frames.push(e.detail);if(probe.frames.length>32)probe.completed.delete(probe.frames.shift().serial)});
        document.addEventListener('anatomy-gpu',e=>{if(e.detail.completed)probe.completed.add(e.detail.serial)});
      })()`);
      if(process.env.DATA_REBUILD_REQUIRE_GPU_STATE==='1')await js(`(()=>{
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
      if(nativeStudy){await inspectNativeStudy();return;}
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
