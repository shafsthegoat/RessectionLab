'use strict';

// Real renderer, dialogs, preload and engine. Only dialog destinations are supplied.
const {app, dialog, ipcMain}=require('electron');
const assert=require('node:assert/strict');
const {createHash}=require('node:crypto');
const {execFileSync}=require('node:child_process');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const {displayIdentity, geometry, inputScript, pixelSummary}=require('./verify-eden-planning-helpers.cjs');

if(process.platform==='darwin')app.setActivationPolicy('accessory');
process.argv.push('--automation-hidden');
delete process.env.RESSECTIONLAB_DEV_URL;

const root=path.resolve(__dirname,'../..');
const testcases=fs.realpathSync(path.join(root,'testcases'));
const read=filename=>JSON.parse(fs.readFileSync(filename,'utf8'));
function sha(filename){
  const descriptor=fs.openSync(filename,'r');
  const digest=createHash('sha256'),buffer=Buffer.allocUnsafe(1024*1024);
  try{
    for(let bytes=fs.readSync(descriptor,buffer);bytes;bytes=fs.readSync(descriptor,buffer))
      digest.update(buffer.subarray(0,bytes));
  }finally{fs.closeSync(descriptor);}
  return digest.digest('hex');
}
const write=(filename,value)=>fs.writeFileSync(filename,JSON.stringify(value,null,2)+'\n');
const serialized=value=>JSON.parse(JSON.stringify(value));
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const phase=process.env.ADAPTIVE_PHASE||'create';
const source=fs.realpathSync(process.env.ADAPTIVE_SOURCE||'');
const requestedOutput=path.resolve(process.env.ADAPTIVE_OUTPUT||'');
const output=path.join(fs.realpathSync(path.dirname(requestedOutput)),path.basename(requestedOutput));
const within=(parent,filename)=>filename.startsWith(parent+path.sep);
assert(['create','reopen','incremental','refresh-comparison','migrated'].includes(phase),'Unknown verification phase');
assert((phase==='migrated'||within(testcases,source))&&within(testcases,output),
  'Set ADAPTIVE_SOURCE and a new ADAPTIVE_OUTPUT under ignored testcases/');
assert(!fs.existsSync(output),'Verification output is already present');
const migratedCases={
  'EDEN2020-CTRL_0866':{patient:'EDEN2020:CTRL_0866',role:'DEVELOPMENT'},
  'TopCoW-001-CTA':{patient:'TopCoW-001-CTA',role:'DEVELOPMENT'},
};
function migratedSource(){
  const dataRoot=process.env.ADAPTIVE_MIGRATED_DATA_ROOT;
  assert(dataRoot,'Set ADAPTIVE_MIGRATED_DATA_ROOT to the recorded external data root');
  const guarded=execFileSync(path.join(root,'.venv/bin/python'),['-c',
    'from resectionlab.data_root import checked_root; import sys; print(checked_root(sys.argv[1]))',dataRoot],
    {cwd:root,encoding:'utf8',env:{...process.env,PYTHONPATH:path.join(root,'src')}}).trim();
  const key=path.basename(source), caseProfile=migratedCases[key];
  assert(caseProfile&&source===path.join(guarded,'prepared/adaptive-anatomy-migration-20261117-v1',key),
    'Migration verification needs a named study in the guarded migration output');
  const filename=path.join(source,'patient-anatomy.json'), document=read(filename);
  assert.equal(document.patient_id,caseProfile.patient);
  assert.equal(document.migration_source?.case_id,key);
  assert.equal(document.migration_source.role,caseProfile.role);
  assert.equal(document.migration_source.exposure,'private');
  const references=[...document.series.map(row=>row.original),
    ...document.layers.map(row=>row.provenance?.source).filter(ref=>ref?.path&&ref?.sha256)];
  const hashes={manifest:sha(filename)},paths={manifest:filename};
  for(const ref of references){
    if(path.isAbsolute(ref.path))continue;
    const member=path.join(source,ref.path);
    assert(within(source,member)&&fs.realpathSync(member)===member,'Unsafe migration source member');
    if(hashes[ref.path]){assert.equal(hashes[ref.path],ref.sha256);continue;}
    assert.equal(sha(member),ref.sha256,'Migration source changed: '+ref.path);
    hashes[ref.path]=ref.sha256;
    paths[ref.path]=member;
  }
  return {...caseProfile,key,paths,hashes,timepoint:document.study_timepoint,document,
    description:key.startsWith('EDEN')?'Acquired healthy MRI and TOF with separate synthetic targets':'Acquired CTA with supplied source labels',
    revision:document.snapshot_id};
}
assert(!fs.existsSync(output),'Verification output is already present');
function sourceProfile(){
  if(path.basename(source)==='candidate-sub-06'){
    const receipt=read(path.join(source,'provenance/range_receipt.json'));
    const file='CE-T1W.nii.gz';
    const row=receipt.find(item=>item.name==='meningioma/sub-06/'+file);
    assert.equal(row?.sha256,'1fe942294ad7e60247c78ae45619a1e4234e42a5293206d213e12e56d45b3dce');
    const filename=path.join(source,'scans',file);
    assert.equal(sha(filename),row.sha256);
    return {patient:'ZENODO-17486320-meningioma-sub-06',role:'DEVELOPMENT',revision:'Zenodo 17486320',
      description:'Native acquired T1 post-contrast alone; no supplied tumor annotation',
      paths:{T1CE:filename},hashes:{T1CE:row.sha256},initial:['T1CE'],indication:'meningioma',timepoint:'source-session-unknown'};
  }
  const patient=path.basename(source);
  assert(/^UCSF-PDGM-\d{4}$/.test(patient));
  const acquired=read(path.join(source,'source-receipt.json'));
  const declaration=read(path.join(source,'../development-declaration.json'));
  assert.equal(acquired.role,'DEVELOPMENT');
  assert.equal(declaration.role,'DEVELOPMENT');
  assert(declaration.cases.includes(patient));
  assert.equal(declaration.mirror_revision,acquired.mirror_revision);
  assert.equal(acquired.patient_id,patient);
  const paths=Object.fromEntries(Object.entries({T1CE:'T1c',T2:'T2',T1:'T1',FLAIR:'FLAIR',mask:'tumor_segmentation'})
    .map(([role,suffix])=>[role,path.join(source,patient+'_'+suffix+'.nii.gz')]));
  const hashes=Object.fromEntries(Object.entries(paths).map(([role,filename])=>{
    const pinned=acquired.files.find(row=>path.basename(row.remote_path)===path.basename(filename));
    assert(pinned,'Missing pinned source receipt: '+filename);
    assert.equal(fs.statSync(filename).size,pinned.size_bytes);
    assert.equal(sha(filename),pinned.sha256);
    return [role,pinned.sha256];
  }));
  return {patient,paths,hashes,role:acquired.role,revision:acquired.mirror_revision,
    description:'Publisher-preprocessed, skull-stripped UCSF-PDGM mirror NIfTIs and supplied multiclass lesion annotation',
    initial:['T1CE','T2'],timepoint:'preoperative'};
}
const study=phase==='migrated'?migratedSource():sourceProfile();
const {patient,paths:sourcePaths,hashes:sourceHashes}=study;
assert(phase!=='incremental'||sourcePaths.mask,'Incremental verification uses the multi-scan UCSF source profile');
const resumeFile=phase==='create'&&process.env.ADAPTIVE_RESUME
  ?fs.realpathSync(process.env.ADAPTIVE_RESUME):null;
const resume=resumeFile?read(resumeFile):null;
if(resume){
  assert(within(testcases,resumeFile)&&resume.phase==='create'&&resume.preparedSnapshot&&resume.importedSnapshot,
    'Resume requires a prior create receipt after completed preparation');
  assert.equal(resume.source,source);
  assert.deepEqual(resume.sourceHashes,sourceHashes);
  assert.notEqual(resume.pid,process.pid);
}
const previous=(phase==='create'||phase==='migrated')?null:read(fs.realpathSync(process.env.ADAPTIVE_EXPECTED||''));
if(previous){
  assert(previous.passed&&(previous.phase==='create'||phase==='reopen'&&previous.phase==='refresh-comparison'),'Expected finished create or refreshed comparison receipt');
  assert.equal(previous.source,source);
  assert.deepEqual(previous.sourceHashes,sourceHashes);
  assert.notEqual(previous.pid,process.pid,'Reopen must be a fresh process');
  assert(within(testcases,fs.realpathSync(previous.saved.path)));
  assert.equal(sha(previous.saved.path),previous.saved.sha256);
}
fs.mkdirSync(output);

const report={schema:'resectionlab.adaptive-patient-electron.v1',phase,pid:process.pid,source,sourceHashes,patient,
  sourceRole:study.role,sourceRevision:study.revision,sourceDescription:study.description,
  started:new Date().toISOString(),windowEverVisible:false,checks:[],screenshots:[],requests:[],errors:[],passed:false};
const reportFile=path.join(output,'report.json');
const profile=fs.mkdtempSync(path.join(os.tmpdir(),'rslab-adaptive-verifier-'));
app.setPath('userData',profile);
app.on('will-quit',()=>{
  fs.rmSync(profile,{recursive:true,force:true});
  if(!report.passed)app.exit(1);
});
let closed=false;
const finish=error=>{
  if(closed)return;
  closed=true;
  if(error){report.error=error.stack||String(error);console.error(report.error);}
  report.finished=new Date().toISOString();
  write(reportFile,report);
  console.log(JSON.stringify({phase,passed:report.passed,report:reportFile}));
  process.exitCode=report.passed?0:1;
  app.quit();
};
dialog.showErrorBox=(title,message)=>finish(new Error(title+': '+message));
process.on('uncaughtException',finish);
process.on('unhandledRejection',finish);
const deadline=setTimeout(()=>{
  finish(new Error('Adaptive verification exceeded its run deadline'));
  app.exit(1);
},35*60*1000);

const native=path.join(output,'native');
const manifestFile=path.join(native,'patient-anatomy.json');
const manifest=()=>read(manifestFile);
const saveTo=path.join(output,'patient.rsworkspace');
let nextOpen=null,nextSave=false;
dialog.showOpenDialog=async(_window,options)=>{
  const files=nextOpen;
  nextOpen=null;
  assert(files,'Unplanned file dialog: '+options.title);
  if(files.length>1)assert(options.properties.includes('multiSelections'),'Multiple acquisitions require a multiselect dialog');
  return {canceled:false,filePaths:files};
};
dialog.showSaveDialog=async()=>{
  assert(nextSave,'Unplanned save dialog');
  nextSave=false;
  return {canceled:false,filePath:saveTo};
};
dialog.showMessageBox=async(_window,options)=>{throw new Error('Unplanned dialog: '+options.message);};

const responseRows=[];
const handle=ipcMain.handle.bind(ipcMain);
let active=0;
ipcMain.handle=(channel,handler)=>handle(channel,async(...args)=>{
  const input=args[1],started=Date.now();
  const receipt={channel,action:input?.action,startedUtc:new Date(started).toISOString(),
    snapshotId:input?.snapshotId,seriesId:input?.seriesId,targetLayerId:input?.targetLayerId};
  report.requests.push(receipt);
  active++;
  try{
    const result=await handler(...args);
    if(phase==='migrated'&&channel==='research:anatomyView'){
      responseRows.push({receipt,input,result:{
        modality:result.modality,sha256:result.mri.sha256,shape:result.shape,registrationStatus:result.registrationStatus,
        alignmentPreviews:result.alignmentPreviews,preview:result.displayPreview,
        layers:result.compartments.map(row=>({id:row.name,kind:row.kind,volumeMm3:row.volumeMm3,
          sourceIdentity:row.sourceIdentity,alignmentPreviews:row.alignmentPreviews}))
      }});
    }
    if(channel==='research:anatomyPlan'||channel==='research:anatomySaveView'||
       channel==='research:openWorkspace'||channel.startsWith('research:anatomyPrep')){
      responseRows.push({receipt,input,result});
    }
    receipt.ok=true;
    return result;
  }catch(error){
    receipt.error=String(error);
    report.errors.push({channel,error:String(error)});
    throw error;
  }finally{
    receipt.elapsedMs=Date.now()-started;
    active--;
  }
});

// Start in an isolated empty collection so existing library favorites do not
// start unrelated registration or inference before the actual UI exercise.
const startup=path.join(output,'startup');
execFileSync(path.join(root,'.venv/bin/python'),['-c',
  'from resectionlab.patient_anatomy import PatientImagingCollection; import sys; PatientImagingCollection.create(sys.argv[1], "EMPTY-VERIFIER").save("Empty verification startup")',startup],
  {cwd:root,env:{...process.env,PYTHONPATH:path.join(root,'src')}});
process.argv.push('--anatomy',path.join(startup,'patient-anatomy.json'));

function verifySources(document,roles){
  assert.equal(document.patient_id,patient);
  assert.equal(document.study_timepoint,study.timepoint);
  assert.equal(document.series.length,roles.length);
  assert.equal(new Set(document.layers.map(row=>row.id)).size,document.layers.length);
  for(const role of roles){
    const row=document.series.find(series=>series.modality===role);
    assert(row,'Missing '+role);
    assert.equal(row.timepoint,study.timepoint);
    assert.equal(row.original.sha256,sourceHashes[role]);
    assert.equal(sha(path.join(native,row.original.path)),sourceHashes[role]);
  }
  if(!sourcePaths.mask)return null;
  const observed=document.layers.filter(row=>row.provenance?.source?.sha256===sourceHashes.mask);
  assert.deepEqual(observed.map(row=>row.label_value).sort(),[1,2,4]);
  assert.deepEqual(Object.fromEntries(observed.map(row=>[row.label_value,row.label])),{
    1:'Nonenhancing core or necrosis',2:'Edema-like abnormality',4:'Enhancing lesion'
  });
  const target=observed.find(row=>row.label_value===4);
  assert.equal(target.kind,'tumor');
  assert.equal(target.source_series,document.series.find(row=>row.modality==='T1CE').id);
  assert.match(target.review_status,/unreviewed/i);
  assert.equal(sha(path.join(native,target.provenance.source.path)),sourceHashes.mask);
  assert(observed.filter(row=>row.label_value!==4).every(row=>row.role==='context'));
  return target;
}

function verifyNativeGeometry(document){
  const comparison=execFileSync(path.join(root,'.venv/bin/python'),['-c',`
import json, sys
from pathlib import Path
import nibabel as nib
import numpy as np
from resectionlab.patient_anatomy import PatientImagingCollection
root = Path(sys.argv[1])
collection = PatientImagingCollection(root)
doc = collection.document
rows = doc["series"] + [row for row in doc["layers"]
                        if isinstance(source := row["provenance"].get("source"), dict)
                        and source.get("sha256") == sys.argv[2]]
results = []
for row in rows:
    original = row["original"] if "original" in row else row["provenance"]["source"]
    image = nib.load(root / original["path"])
    stored = np.load(root / row["data"]["path"], mmap_mode="r")
    assert np.array_equal(image.affine, row["affine"]), row["id"]
    assert image.shape == stored.shape, row["id"]
    values = image.get_fdata(dtype=np.float32, caching="unchanged")
    for start in range(0, stored.shape[2], 16):
        block = slice(start, start + 16)
        assert np.array_equal(values[..., block], stored[..., block]), row["id"]
    record = {"id": row["id"], "shape": image.shape, "affine": image.affine.tolist(),
              "spacingMm": list(map(float, image.header.get_zooms())), "units": image.header.get_xyzt_units()[0]}
    if "original" in row:
        transform = doc["transforms"].get(row["id"])
        assert transform is not None, row["id"] + " has no source-bound registration"
        assert transform["status"] in {"reference", "unreviewed", "source-provided"}, row["id"]
        record["registrationStatus"] = transform["status"]
        record["sourceToPatientRAS"] = collection.registration_matrix(row["id"]).tolist()
        if transform["status"] == "unreviewed":
            try:
                collection.registration_matrix(row["id"], require_review=True)
            except ValueError:
                pass
            else:
                raise AssertionError(row["id"] + " was admitted without alignment review")
        record["registrationQuality"] = transform.get("quality")
    results.append(record)
print(json.dumps(results))
`,native,sourceHashes.mask??''],{cwd:root,encoding:'utf8',env:{...process.env,PYTHONPATH:path.join(root,'src')}});
  const rows=JSON.parse(comparison);
  assert.equal(rows.length,document.series.length+(sourcePaths.mask?3:0));
  report.nativeGeometry=rows;
}

app.on('browser-window-created',(_event, win)=>{
  win.on('show',()=>{
    report.windowEverVisible=true;
    win.hide();
    finish(new Error('Automation window became visible'));
  });
  win.webContents.on('render-process-gone',(_ev, details)=>finish(new Error('Renderer stopped: '+details.reason)));
  win.webContents.once('did-finish-load',async()=>{
    win.setContentSize(1600,1000);
    win.webContents.setBackgroundThrottling(false);
    const js=expression=>win.webContents.executeJavaScript(expression,true);
    const until=async(predicate,label,seconds=300)=>{
      const expiry=Date.now()+seconds*1000;
      while(Date.now()<expiry){
        assert(!win.isVisible(),'Automation window must stay hidden');
        const failure=report.errors.find(row=>!(process.env.ADAPTIVE_CANCEL==='1'&&/cancel/i.test(row.error)));
        if(failure)throw new Error(failure.channel+': '+failure.error);
        if(await predicate())return;
        await delay(120);
      }
      throw new Error('Timed out waiting for '+label+'\n'+await js('document.body.innerText'));
    };
    const wait=(expression,label,seconds)=>until(()=>js(expression),label,seconds);
    const buttonScript=(label,action)=>`(()=>{
      const target=[...document.querySelectorAll('button')].find(button=>button.textContent.trim()===${JSON.stringify(label)}||button.getAttribute('aria-label')===${JSON.stringify(label)}||button.querySelector('span')?.textContent.trim()===${JSON.stringify(label)});
      if(!target||target.matches(':disabled'))return false;
      ${action}
      return true;
    })()`;
    const enabled=label=>buttonScript(label,'');
    const scrollAndClick=`for(let el=target.parentElement;el;el=el.parentElement){if(el.tagName==='DETAILS')el.open=true}
      target.scrollIntoView({block:'nearest'});target.click();`;
    const click=async label=>{
      await wait(enabled(label),'enabled '+label);
      assert(await js(buttonScript(label,scrollAndClick)),'Missing enabled '+label);
    };
    const openDetails=async label=>{
      assert(await js(`(()=>{
        const summary=[...document.querySelectorAll('summary')].find(summary=>summary.textContent.trim()===${JSON.stringify(label)});
        if(!summary)return false;
        if(!summary.parentElement.open)summary.click();
        return true;
      })()`),'Missing '+label);
    };
    const set=async(label,value)=>{
      assert.equal(await js(inputScript(label,value)),String(value));
    };
    const selectByText=async(label,substring)=>{
      const value=await js(`(()=>{
        const input=[...document.querySelectorAll('select[aria-label]')].find(e=>e.getAttribute('aria-label')===${JSON.stringify(label)});
        return input&&Array.from(input.options).find(option=>option.textContent.includes(${JSON.stringify(substring)})&&!option.disabled)?.value;
      })()`);
      assert(value,'Missing '+substring+' option in '+label);
      await set(label,value);
    };
    const checked=label=>js(`[...document.querySelectorAll('input[type=checkbox]')].find(input=>input.getAttribute('aria-label')===${JSON.stringify(label)})?.checked`);
    const toggle=async label=>{
      assert(await js(`(()=>{
        const target=[...document.querySelectorAll('input[type=checkbox]')].find(input=>input.getAttribute('aria-label')===${JSON.stringify(label)});
        if(!target||target.matches(':disabled'))return false;
        ${scrollAndClick}
        return true;
      })()`),'Missing enabled '+label);
    };
    const screen=()=>js(`(()=>{
      const viewer=document.querySelector('.rl-viewer'), workspace=document.querySelector('.native-workspace');
      const pane=document.querySelector('.rl-viewer-axial .rl-viewer-slice-image')?.getBoundingClientRect();
      return {ready:viewer?.dataset.rendererReady==='true',source:viewer?.dataset.sourceDisplayIdentity,
        view:JSON.parse(workspace?.dataset.nativeViewState||'null'),routes:JSON.parse(viewer?.dataset.approachRoutes||'[]'),
        visible:JSON.parse(viewer?.dataset.visibleLayerIds||'[]'),width:innerWidth,
        pane:pane&&{x:pane.x,y:pane.y,width:pane.width,height:pane.height},
        pending:!!document.querySelector('.viewer-loading,.rl-viewer-preparing,[data-layer-pending="true"],[data-approach-pending="true"]')};
    })()`);
    const ready=()=>until(async()=>{
      const state=await screen();
      return state.ready&&state.source&&!state.pending&&active===0;
    },'linked source renderer');
    const capture=async(name,{quiet=false}={})=>{
      if(!quiet){
        await ready();
        // Hidden captures can still contain the last composited busy overlay
        // immediately after the corresponding DOM node has been removed.
        await delay(240);
        await ready();
      }
      assert(!win.isVisible());
      const shot=await win.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true});
      fs.writeFileSync(path.join(output,name+'.png'),shot.toPNG());
      const state=await screen();
      let axial=null;
      if(!quiet){
        assert(state.pane?.width>80&&state.pane.height>80,'Axial source pane is clipped');
        const {x,y,width,height}=state.pane,scale=shot.getSize().width/state.width;
        const {signal,...pixels}=pixelSummary(shot.crop({
          x:Math.ceil((x+10)*scale),y:Math.ceil((y+10)*scale),
          width:Math.floor((width-20)*scale),height:Math.floor((height-20)*scale)
        }).toBitmap());
        assert(pixels.neutralPixels>500&&pixels.dynamicRange>20,'Hidden capture has no useful source-slice signal');
        axial=pixels;
      }
      report.screenshots.push({name:name+'.png',source:state.source,visible:state.visible,routes:state.routes,axial});
      return state;
    };
    const mark=message=>{report.checks.push(message);console.log(message);};
    const versions=channel=>report.requests.filter(row=>row.channel===channel);
    const responses=(channel,action)=>responseRows.filter(row=>row.receipt.channel===channel&&(!action||row.receipt.action===action));
    const plan=(action)=>responses('research:anatomyPlan',action).at(-1)?.result;
    const stages=()=>js(`[...document.querySelectorAll('[data-preparation-stage]')].map(row=>({id:row.dataset.preparationStage,status:row.dataset.stageStatus,text:row.innerText.trim()}))`);
    const prepare=async({cancel=false}={})=>{
      if(study.indication){
        await openDetails('Preparation options');
        await set('Tumor model indication',study.indication);
      }
      await click('Review preparation plan');
      await wait('!!document.querySelector("[data-preparation-stage]")','capability-driven job list');
      report.proposedStages=await stages();
      await ready();
      await capture('02-preparation-plan');
      await click('Prepare anatomy');
      if(cancel){
        await click('Cancel preparation');
        await wait(enabled('Resume preparation'),'resume after cancellation');
        report.cancelledStages=await stages();
        assert(await js('document.querySelector(".preparation-panel [role=status]")?.textContent.includes("Preparation cancelled")'),
          'Expected visible cancellation and resumability status');
        await capture('02b-cancelled-preparation');
        await click('Resume preparation');
      }
      await until(async()=>{
        const rows=await stages();
        return rows.length>0&&rows.every(row=>!/^(planned|queued|pending|running|ready)$/i.test(row.status))&&active===0;
      },'preparation stage outcomes',1200);
      report.preparationStages=await stages();
      await ready();
      mark('Prepared anatomy through the visible job plan');
    };
    const stem=filename=>path.basename(filename).replace(/\.nii(?:\.gz)?$/i,'');
    const stagedLabel=async(prefix,filename)=>{
      const wanted=prefix+' '+stem(filename);
      const lookup=`[...document.querySelectorAll('select[aria-label]')].map(select=>select.getAttribute('aria-label'))
        .find(label=>label.replace(/\\.nii(?:\\.gz)?$/i,'')===${JSON.stringify(wanted)})`;
      await wait(lookup,'staged '+wanted);
      return js(lookup);
    };
    const importBatch=async(roles,{annotation=false}={})=>{
      await click('Scans');
      nextOpen=roles.map(role=>sourcePaths[role]);
      await click('Add NIfTI files…');
      for(const role of roles){
        const label=await stagedLabel('Sequence for',sourcePaths[role]);
        const proposed=await js(`[...document.querySelectorAll('select')].find(e=>e.getAttribute('aria-label')===${JSON.stringify(label)})?.value`);
        report.detectedRoles??={};
        report.detectedRoles[role]=proposed;
        if(proposed!==role)await set(label,role);
      }
      if(annotation){
        nextOpen=[sourcePaths.mask];
        await click('Add segmentations…');
        const forMask=await stagedLabel('Source scan for',sourcePaths.mask);
        await selectByText(forMask,stem(sourcePaths.T1CE));
        await set('Segmentation role 1','context');
        await set('Segmentation name 1','Nonenhancing core or necrosis');
        await click('Add label');
        await set('Segmentation label 2',2);
        await set('Segmentation role 2','context');
        await set('Segmentation name 2','Edema-like abnormality');
        await click('Add label');
        await set('Segmentation label 3',4);
        await set('Segmentation role 3','tumor');
        await set('Segmentation name 3','Enhancing lesion');
      }
      if(!await checked('Confirm patient and timepoint'))await toggle('Confirm patient and timepoint');
      const imports=versions('research:anatomyImportBatch').length;
      await click('Import selected data');
      await until(()=>versions('research:anatomyImportBatch').length>imports&&versions('research:anatomyImportBatch').at(-1).ok,
        'committed reviewed MRI batch');
      await ready();
      assert.equal((await screen()).view.snapshotId,manifest().snapshot_id);
    };
    const openNormal=async filename=>{
      const before=responses('research:openWorkspace').length;
      nextOpen=[filename];
      await click('Open');
      await until(()=>responses('research:openWorkspace').length>before,'normal workspace open');
      const reopened=responses('research:openWorkspace').at(-1).result;
      assert.equal(reopened.document.patient_id,patient);
      await ready();
      return reopened;
    };
    const selectedCard=()=>js('document.querySelector(".approach-card.selected")?.dataset.approachId');
    const chooseCard=async id=>{
      await js(`(()=>{
        const button=[...document.querySelectorAll('.approach-card')].find(row=>row.dataset.approachId===${JSON.stringify(id)});
        if(!button||button.disabled)throw new Error('Missing enabled approach card');
        button.scrollIntoView({block:'nearest'});button.click();
      })()`);
      await until(async()=>await selectedCard()===id,'selected approach '+id);
      await ready();
    };
    const currentRoutes=(reply,document,target)=>{
      assert.equal(reply.activeTargetLayerId,target.id);
      const options=reply.targetOptions??plan('list')?.targetOptions;
      const binding=options?.find(candidate=>candidate.layerId===target.id)?.candidateBinding;
      assert(binding,'Expected explicit candidate binding from the native study');
      const approaches=reply.approaches.filter(row=>row.targetLayerId===target.id&&row.assessmentStatus==='current');
      assert(approaches.length>0,'The app did not produce a current native comparison');
      for(const row of approaches){
        assert.equal(row.sourceSnapshotId,document.snapshot_id);
        assert.equal(row.targetCandidateBinding,binding);
        assert(row.parameterHash&&row.measurementHash);
        assert(row.measurements.targets.some(entry=>entry.layerId===target.id&&entry.sourceSeries===target.source_series&&entry.targetPointInsideLabel));
        assert(row.measurements.unknowns.some(text=>/vascular anatomy unavailable/i.test(text)));
        assert.equal(row.measurements.simulationExecutionAuthorized,false);
        assert(!row.measurements.structures.some(entry=>entry.available&&/vessel|artery|vein|sinus/i.test(entry.kind)),'MRI-only source cannot establish vascular clearance');
      }
      return approaches.map(geometry);
    };
    const save=async()=>{
      await click('Scans');
      nextSave=true;
      await click('Save patient study file…');
      await until(()=>fs.existsSync(saveTo)&&active===0,'portable native study save',420);
      const view=responses('research:anatomySaveView').at(-1).input.viewState;
      assert.equal(view.selectedPlanningTargetId,report.target.id);
      report.saved={path:saveTo,sha256:sha(saveTo),view};
      mark('Saved portable native anatomy, explicit target and comparison');
    };
    const verifyMigrated=async()=>{
      const included=new Set(['originals','arrays','meshes','patient-anatomy.json','source-map.json','anatomy-preparation.json']);
      const preparedCopy=process.env.ADAPTIVE_MIGRATED_REUSE?fs.realpathSync(process.env.ADAPTIVE_MIGRATED_REUSE):source;
      if(preparedCopy!==source)assert(within(testcases,preparedCopy)&&
        sha(path.join(preparedCopy,'patient-anatomy.json'))===sourceHashes.manifest,
        'Reuse requires an exact ignored copy of the current migration manifest');
      fs.cpSync(preparedCopy,native,{recursive:true,mode:fs.constants.COPYFILE_FICLONE,filter:filename=>{
        if(filename===preparedCopy)return true;
        if(!included.has(path.relative(preparedCopy,filename).split(path.sep)[0]))return false;
        assert(!fs.lstatSync(filename).isSymbolicLink(),'No linked dependency in copied study');
        return !path.basename(filename).startsWith('._');
      }});
      report.native=native;
      report.migrationIdentity={...study.document.migration_source};
      assert.equal(sha(manifestFile),sourceHashes.manifest);
      const opened=await openNormal(manifestFile);
      const document=opened.document;
      assert.deepEqual(serialized(document),serialized(study.document));
      const provenanceLayer=row=>({id:row.id,kind:row.kind,role:row.role,sourceSeries:row.source_series,
        status:row.review_status,coverage:row.coverage,provenance:row.provenance});
      report.sourceSeries=document.series.map(row=>({id:row.id,modality:row.modality,shape:row.data.shape,
        originalSha256:row.original.sha256,affine:row.affine,spacingMm:row.grid_spacing_mm,
        transform:document.transforms[row.id]}));
      const captureSource=async(name,id)=>{
        const state=await capture(name),identity=displayIdentity(state.source);
        const view=responses('research:anatomyView').filter(item=>item.input.seriesId===id&&item.input.layers.length>0).at(-1);
        assert(view,'Missing source-bound anatomy response');
        assert.equal(identity.snapshotId,document.snapshot_id);
        assert.equal(identity.seriesId,id);
        assert.equal(identity.sha256,view.result.sha256);
        const scan=document.series.find(scan=>scan.id===id);
        assert.equal(view.result.modality,scan.modality);
        const preview=view.result.preview&&scan.preview_pyramid?.find(item=>item.factor===view.result.preview.factor);
        assert.deepEqual(view.result.shape,preview?.data.shape??scan.data.shape);
        return view.result;
      };
      await click('Anatomy');
      if(study.key.startsWith('EDEN')){
        const target=document.layers.find(row=>row.id==='synthetic-01');
        const candidates=document.layers.filter(row=>row.provenance?.planning_target_candidate===true);
        assert.equal(candidates.length,8);
        assert(candidates.every(row=>row.role==='context'&&row.provenance.kind==='simulated'&&row.provenance.healthy_anatomy_unchanged));
        assert.equal(document.tractography.length,7);
        assert.equal(document.transforms.tof.status,'unreviewed');
        report.target=provenanceLayer(target);
        report.t1View=await captureSource('01-eden-t1-anatomy','t1');
        await click('Scans');
        await set('Source scan','tof');
        await ready();
        await openDetails('Alignment & registration');
        report.tofView=await captureSource('02-eden-tof-alignment-preview','tof');
        assert.equal(report.tofView.registrationStatus,'alignment preview');
        assert(report.tofView.alignmentPreviews.includes('tof'));
        assert(report.tofView.layers.some(row=>row.id==='tof-vessel-candidates'&&row.volumeMm3>0));
        assert(await js('document.body.innerText.includes("Review landmarks before quantitative planning")'));
        await set('Source scan','t1');
        await ready();
        await click('Plan & compare');
        await until(()=>!!plan('list'),'available EDEN target options');
        assert.equal(plan('list').activeTargetLayerId??null,null);
        assert.equal(plan('list').targetOptions.filter(row=>row.target!==null).length,8);
        assert.equal(await js(`document.querySelector('select[aria-label="Planning target"]')?.value`),'');
        await js(inputScript('Planning target',target.id));
        await until(()=>plan('selectTarget')?.activeTargetLayerId===target.id,'explicit synthetic target selection');
        const option=plan('selectTarget').targetOptions.find(row=>row.layerId===target.id);
        assert.equal(option.origin,'Synthetic target');
        assert.equal(option.provenance,'simulated');
        assert.equal(option.sourceSeries,'t1');
        await click('Suggest approaches');
        await until(()=>!!plan('suggest'),'native geometric alternatives',420);
        let compared=plan('suggest');
        if(!compared.approaches.some(row=>row.assessmentStatus==='current')){
          report.suggestionAbstention=compared.proposalReport;
          await click('Create manual approach');
          await until(()=>!!plan('upsert'),'manual native assessment',420);
          compared=plan('upsert');
        }
        report.comparison=currentRoutes(compared,document,target);
        const vessels=report.comparison.flatMap(row=>row.measurements.structures.filter(item=>item.layerId==='tof-vessel-candidates'));
        assert(vessels.length>0&&vessels.every(item=>!item.available&&item.sourceSeries==='tof'&&
          !('clearanceLowerBoundMm' in item)),'Unreviewed TOF must remain unassessed');
        for(const route of report.comparison){
          const structures=route.measurements.structures;
          const tracts=structures.filter(row=>row.kind==='tractography');
          assert.equal(tracts.length,7);
          assert(tracts.every(row=>!row.available&&/registration.*needs review/i.test(row.reason)));
          assert(structures.some(row=>row.kind==='brainstem'&&row.available&&row.excluded));
          assert(structures.some(row=>row.kind.startsWith('cerebellar')&&row.available&&row.excluded));
        }
        await capture('03-eden-current-comparison');
        await click('Save study & comparison');
        await until(()=>fs.existsSync(path.join(native,'workspace-view.json'))&&active===0,'saved local EDEN comparison');
        assert.equal(read(path.join(native,'workspace-view.json')).selectedPlanningTargetId,target.id);
        mark('Inspected linked T1 and unreviewed TOF, explicitly selected a synthetic target and saved current geometry on the disposable copy');
      }else{
        assert.deepEqual(document.series.map(row=>row.modality),['CTA']);
        assert.equal(document.reference_series,'ct');
        assert(await js('document.querySelector(".study-badge")?.innerText.includes("Unspecified study")'),
          'An unknown CTA timepoint must remain unspecified');
        assert(!document.layers.some(row=>row.role==='target'||['tumor','motor','language','functional'].includes(row.kind)));
        report.ctaView=await captureSource('01-cta-source-supported-anatomy','ct');
        assert.equal(report.ctaView.registrationStatus,'patient frame');
        for(const kind of ['artery','vein','sinus','skull']){
          const measured=report.ctaView.layers.find(row=>row.kind===kind&&row.volumeMm3>0);
          assert(measured,'Missing nonempty CTA display layer: '+kind);
          const layer=document.layers.find(row=>row.id===measured.id);
          assert.equal(layer.source_series,'ct');
        }
        report.ctaLayers=document.layers.filter(row=>['artery','vein','sinus','skull'].includes(row.kind)).map(provenanceLayer);
        await click('Scans');
        await click('Review preparation plan');
        await wait('!!document.querySelector("[data-preparation-stage]")','CTA capability stages');
        report.proposedStages=await stages();
        for(const id of ['tumor','functional']){
          assert(report.proposedStages.some(stage=>(stage.id===id||stage.id.startsWith(id+':'))&&stage.status==='missing_input'),
            'CTA must disclose missing '+id);
        }
        await js(`document.querySelector('[data-preparation-stage="functional"]')?.scrollIntoView({block:'nearest'})`);
        await capture('02-cta-unavailable-capabilities');
        await click('Plan & compare');
        await until(()=>!!plan('list'),'CTA planning availability');
        assert.deepEqual(plan('list').targetOptions,[]);
        assert.equal(plan('list').activeTargetLayerId??null,null);
        assert(await js('document.body.innerText.includes("Geometric comparison needs a nonempty lesion label")'));
        await capture('03-cta-anatomy-only-planning');
        mark('Opened CTA without MRI, displayed source-bound bone and supplied vessels, and disclosed missing lesion and functional data');
      }
      assert.equal(sha(manifestFile),sourceHashes.manifest,'Display or planning unexpectedly revised anatomical evidence');
      assert(versions('research:anatomyRegister').every(row=>row.action==='ensure'));
      assert.equal(versions('research:anatomyReconstruct').length,0);
      assert(!report.requests.some(row=>row.channel.startsWith('research:anatomyPrep')&&row.action==='run'));
    };
    try{
      await wait('!!document.querySelector(".native-workspace")&&document.querySelector("footer")?.innerText.includes("Add a source scan")','empty startup');
      if(phase==='migrated'){
        await verifyMigrated();
      }else if(phase==='create'){
        report.native=native;
        let reference;
        if(resume){
          const baseline=fs.realpathSync(resume.native);
          assert(within(testcases,baseline),'Expected an ignored native study');
          fs.cpSync(baseline,native,{recursive:true,mode:fs.constants.COPYFILE_FICLONE});
          const opened=await openNormal(manifestFile);
          assert.equal(opened.document.snapshot_id,resume.preparedSnapshot);
          assert.equal((await screen()).view.snapshotId,resume.preparedSnapshot);
          report.resumedFrom=resumeFile;
          report.importedSnapshot=resume.importedSnapshot;
          report.proposedStages=resume.proposedStages;
          report.preparationStages=resume.preparationStages;
          reference=opened.document.reference_series;
          await capture('01-resumed-prepared-anatomy');
          mark('Opened the exact prepared study from the preceding hidden app run');
        }else{
          fs.mkdirSync(native);
          await click('Cases');
          await openDetails('New patient study');
          await set('New patient ID',patient);
          await set('New study timepoint',study.timepoint);
          nextOpen=[native];
          await click('Create study…');
          await until(()=>fs.existsSync(manifestFile),'GUI creates native patient');
          await importBatch(study.initial,{annotation:!!sourcePaths.mask});
          const imported=manifest();
          const supplied=verifySources(imported,study.initial);
          reference=imported.reference_series;
          assert.equal(reference,imported.series.find(row=>row.modality==='T1CE').id);
          if(supplied)assert.equal(reference,supplied.source_series);
          else assert(!imported.layers.some(row=>row.kind==='tumor'),'No lesion annotation was supplied');
          report.importedSnapshot=imported.snapshot_id;
          await capture('01-imported-linked-sources');
          mark(sourcePaths.mask?'Imported two real scans with documented lesion classes':'Imported native acquired T1 post-contrast without a supplied lesion');
          await prepare({cancel:process.env.ADAPTIVE_CANCEL==='1'});
        }
        const prepared=manifest();
        const target=verifySources(prepared,study.initial)??prepared.layers.find(row=>row.kind==='tumor'&&row.provenance?.model===study.indication);
        assert(target,'Preparation must produce a usable lesion candidate for this profile');
        if(study.indication){
          assert.equal(target.role,'context');
          assert.equal(target.source_series,prepared.reference_series);
          assert.match(target.review_status,/unreviewed/i);
        }
        assert.equal(prepared.reference_series,reference);
        const brain=prepared.layers.find(row=>row.kind==='brain'&&row.source_series===prepared.reference_series);
        assert(brain,'Prepared reference needs a brain envelope');
        assert.match(brain.review_status,/unreviewed/i);
        if(sourcePaths.mask)assert(!prepared.layers.some(row=>['vessels','artery','vein','sinus','skull','bone','dura','head-envelope'].includes(row.kind)),
          'Skull-stripped structural input cannot supply vascular or outer-head anatomy');
        report.target=target;
        report.brain=brain;
        report.preparedSnapshot=prepared.snapshot_id;
        verifyNativeGeometry(prepared);
        await click('Anatomy');
        await openDetails('Coverage & unavailable anatomy');
        await capture('03-prepared-anatomy');
        await click('Plan & compare');
        const targetSelector='select[aria-label="Planning target"]';
        await wait(`!!document.querySelector(${JSON.stringify(targetSelector)})`,'planning target selector');
        await ready();
        assert.equal((await screen()).view.selectedPlanningTargetId??null,null,'Target must be explicitly selected');
        assert.equal(await js(`document.querySelector(${JSON.stringify(targetSelector)})?.value`),'');
        await js(inputScript('Planning target',target.id));
        await until(()=>plan('selectTarget')?.activeTargetLayerId===target.id,'explicit source-bound selection');
        await wait(`document.querySelector(${JSON.stringify(targetSelector)})?.value===${JSON.stringify(target.id)}`,'rendered explicit target');
        await ready();
        mark(sourcePaths.mask?'Explicitly selected the nonempty supplied enhancing lesion':'Explicitly selected the nonempty unreviewed meningioma estimate');
        await click('Suggest approaches');
        await until(()=>!!plan('suggest'),'native approach suggestion',420);
        let reply=plan('suggest');
        if(!reply.approaches.some(row=>row.assessmentStatus==='current')){
          report.suggestionAbstention=reply.proposalReport;
          await click('Create manual approach');
          await until(()=>!!plan('upsert'),'native manual approach assessment',420);
          reply=plan('upsert');
        }
        await ready();
        report.comparison=currentRoutes(reply,prepared,target);
        report.selectedApproachId=await selectedCard();
        assert(report.comparison.some(row=>row.id===report.selectedApproachId));
        const displayed=await capture('04-native-comparison');
        assert(displayed.routes.length>0,'Comparison geometry must reach the linked viewer');
        assert.equal(displayIdentity(displayed.source).snapshotId,prepared.snapshot_id);
        // Suggested routes use compact sensitivity receipts. Compare two
        // explicit reassessments so their evidence representations match.
        await click('Reassess');
        await until(()=>!!plan('assess'),'assessment with baseline display settings',420);
        await ready();
        report.comparison=currentRoutes(plan('assess'),prepared,target);
        const initialAssessments=responses('research:anatomyPlan','assess').length;
        await click('Appearance');
        const smooth=await checked('Smooth 3D surfaces');
        assert.equal(typeof smooth,'boolean');
        await toggle('Smooth 3D surfaces');
        assert.equal(await checked('Smooth 3D surfaces'),!smooth);
        await click('Close view controls');
        await capture(smooth?'05-original-surfaces':'05-smoothed-surfaces');
        await click('Reassess');
        await until(()=>responses('research:anatomyPlan','assess').length>initialAssessments,
          'assessment with alternative display setting',420);
        await ready();
        assert.deepEqual(currentRoutes(plan('assess'),prepared,target),report.comparison);
        await click('Appearance');
        await toggle('Smooth 3D surfaces');
        assert.equal(await checked('Smooth 3D surfaces'),smooth);
        await click('Close view controls');
        mark('Display smoothing leaves target, route geometry and exact measurements unchanged');
        await save();
      }else if(phase==='reopen'){
        const opened=await openNormal(previous.saved.path);
        assert.equal(opened.document.snapshot_id,previous.preparedSnapshot);
        assert.equal(opened.viewState.selectedPlanningTargetId,previous.target.id);
        await click('Plan & compare');
        await ready();
        assert.equal(await selectedCard(),previous.selectedApproachId);
        report.target=opened.document.layers.find(row=>row.id===previous.target.id);
        assert.deepEqual(serialized(report.target),previous.target);
        assert.deepEqual(serialized(currentRoutes(plan('list'),opened.document,report.target)),previous.comparison);
        assert.deepEqual((await screen()).view,previous.saved.view);
        assert.equal(versions('research:anatomyReconstruct').length,0);
        assert(!report.requests.some(row=>row.channel.startsWith('research:anatomyPrep')&&/run|start|execute/i.test(row.channel+' '+row.action)),'Opening a study reran preparation');
        assert.equal(sha(previous.saved.path),previous.saved.sha256);
        await capture('01-cold-reopened-comparison');
        mark('Fresh process reopened the portable file with exact target, approach measurements and view');
      }else if(phase==='refresh-comparison'){
        const baseline=fs.realpathSync(previous.native);
        assert(within(testcases,baseline),'Expected an ignored native study');
        fs.cpSync(baseline,native,{recursive:true,mode:fs.constants.COPYFILE_FICLONE});
        report.native=native;
        const opened=await openNormal(manifestFile);
        assert.equal(opened.document.snapshot_id,previous.preparedSnapshot);
        report.preparedSnapshot=opened.document.snapshot_id;
        report.target=opened.document.layers.find(row=>row.id===previous.target.id);
        assert.deepEqual(serialized(report.target),previous.target);
        await click('Plan & compare');
        await ready();
        const loaded=plan('list');
        assert.equal(loaded.activeTargetLayerId,report.target.id);
        report.beforeRefresh=loaded.approaches.map(row=>({id:row.id,status:row.assessmentStatus}));
        for(const earlier of previous.comparison){
          await chooseCard(earlier.id);
          const count=responses('research:anatomyPlan','assess').length;
          await click('Reassess');
          await until(()=>responses('research:anatomyPlan','assess').length>count,'reassess '+earlier.id);
          await ready();
        }
        report.comparison=currentRoutes(plan('assess'),opened.document,report.target);
        assert.equal(report.comparison.length,previous.comparison.length);
        for(const refreshed of report.comparison){
          const earlier=previous.comparison.find(row=>row.id===refreshed.id);
          assert(earlier);
          for(const key of ['entry','target','radiusMm','entryPatch'])assert.deepEqual(serialized(refreshed[key]),earlier[key]);
        }
        await chooseCard(previous.selectedApproachId);
        report.selectedApproachId=previous.selectedApproachId;
        assert.equal(manifest().snapshot_id,opened.document.snapshot_id);
        assert(!report.requests.some(row=>row.channel==='research:anatomyReconstruct'||
          row.channel==='research:anatomyPrepare'&&row.action==='run'),'Assessment refresh invoked inference');
        await capture('01-refreshed-comparison');
        mark('Refreshed all saved approaches on their original geometry and anatomy snapshot without inference');
        await save();
      }else{
        const baseline=fs.realpathSync(previous.native);
        assert(within(testcases,baseline),'Expected an ignored native study');
        fs.cpSync(baseline,native,{recursive:true,mode:fs.constants.COPYFILE_FICLONE});
        report.native=native;
        await openNormal(manifestFile);
        const before=manifest();
        assert.equal(before.snapshot_id,previous.preparedSnapshot);
        await importBatch(['T1','FLAIR','T2']);
        const added=manifest();
        verifySources(added,['T1CE','T2','T1','FLAIR']);
        assert.equal(added.reference_series,before.reference_series);
        assert.deepEqual(added.series.filter(row=>before.series.some(old=>old.id===row.id)),before.series);
        assert.deepEqual(added.layers.filter(row=>before.layers.some(old=>old.id===row.id)),before.layers);
        await click('Plan & compare');
        await ready();
        const stale=plan('list');
        assert.equal(stale.activeTargetLayerId,previous.target.id);
        assert(stale.approaches.some(row=>row.id===previous.selectedApproachId&&row.assessmentStatus==='stale'),'Prior approach must be stale after anatomy changes');
        mark('Later additions preserve the reference and existing source layers, deduplicate T2 and invalidate the old assessment');
        await click('Scans');
        await prepare({cancel:process.env.ADAPTIVE_CANCEL==='1'});
        const updated=manifest();
        report.target=verifySources(updated,['T1CE','T2','T1','FLAIR']);
        assert.equal(updated.reference_series,before.reference_series);
        assert.deepEqual(serialized(updated.layers.find(row=>row.id===previous.brain.id)),previous.brain);
        verifyNativeGeometry(updated);
        await click('Plan & compare');
        await ready();
        await click('Reassess');
        await until(()=>!!plan('assess'),'reassessment of the stale approach',420);
        await ready();
        report.comparison=currentRoutes(plan('assess'),updated,report.target);
        await capture('03-reassessed-after-addition');
        await save();
        mark('Incremental preparation kept the existing brain estimate and restored a current comparison on the new snapshot');
      }
      for(const [role,filename] of Object.entries(sourcePaths))assert.equal(sha(filename),sourceHashes[role],'Source modified: '+filename);
      assert(!report.windowEverVisible&&!win.isVisible());
      assert(report.errors.every(row=>process.env.ADAPTIVE_CANCEL==='1'&&/cancel/i.test(row.error)),'Unexpected IPC failures: '+JSON.stringify(report.errors));
      report.passed=true;
      finish();
    }catch(error){
      report.body=await js('document.body.innerText').catch(()=>null);
      await capture('failure',{quiet:true}).catch(()=>null);
      finish(error);
    }finally{
      clearTimeout(deadline);
    }
  });
});

require('./main.cjs');
