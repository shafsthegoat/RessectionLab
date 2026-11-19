'use strict';
// Hidden Electron capture of the live viewer. Native studies are tested on disposable copies.
const {app,dialog,ipcMain}=require('electron');
const {execFile,execFileSync}=require('node:child_process');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const {inputScript}=require('./verify-eden-planning-helpers.cjs');
const {studyLayerSelectionScript,studyLayerVisibilityScript,studyScanEnabledScript,studyScanSelectionScript,studyVisibilityToggleScript}=require('./verify-study-ui-helpers.cjs');
const main=process.env.SURFACE_VERIFY_MAIN||path.join(__dirname,'main.cjs');
const output=path.resolve(process.env.SURFACE_VERIFY_OUTPUT);
fs.mkdirSync(output,{recursive:false});
const profile=fs.mkdtempSync(path.join(os.tmpdir(),'rslab-surface-quality-'));
app.setPath('userData',profile);
process.argv.push('--automation-hidden');
const nativeIndex=process.argv.indexOf('--anatomy'),caseIndex=process.argv.indexOf('--case');
assert((nativeIndex>=0)!==(caseIndex>=0),'Choose --anatomy or --case');
const inputIndex=(nativeIndex>=0?nativeIndex:caseIndex)+1,source=fs.realpathSync(process.argv[inputIndex]);
const sha=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const zipMember=(archive,member,maxBuffer=16*1024*1024)=>execFileSync('unzip',['-p',archive,member],{maxBuffer});
const sourceHash=sha(source);
const directFile=process.env.SURFACE_VERIFY_DIRECT_SOURCE;
const refinedLayer=process.env.SURFACE_VERIFY_REFINED_LAYER;
const refinedIds=refinedLayer?.split(',')??[];
const detailIds=process.env.SURFACE_VERIFY_DETAIL_FOCUS?.split(',')??[];
const selectedSeries=process.env.SURFACE_VERIFY_SERIES;
const detailCursor=process.env.SURFACE_VERIFY_DETAIL_CURSOR?.split(',').map(Number);
if(detailCursor)assert(detailIds.length===1&&detailCursor.length===3&&detailCursor.every(Number.isFinite),'Choose one detail layer and a numeric RAS cursor');
const branchFile=process.env.SURFACE_VERIFY_BRANCH_RECEIPT;
const branches=branchFile?JSON.parse(fs.readFileSync(branchFile,'utf8')):null;
const refinedPriorFile=process.env.SURFACE_VERIFY_REFINED_PRIOR;
const refinedPrior=refinedPriorFile?JSON.parse(fs.readFileSync(refinedPriorFile,'utf8')):null;
const direct=directFile?JSON.parse(fs.readFileSync(directFile,'utf8')):null;
const directConditionedFile=process.env.SURFACE_VERIFY_DIRECT_CONDITIONED;
const directConditioned=directConditionedFile?JSON.parse(fs.readFileSync(directConditionedFile,'utf8')):null;
if(directConditioned)assert(direct&&directConditioned.passed&&directConditioned.baselineMatched&&!directConditioned.windowEverVisible);
const priorFile=process.env.SURFACE_VERIFY_DIRECT_PRIOR;
const prior=priorFile?JSON.parse(fs.readFileSync(priorFile,'utf8')):null;
const pairedMra=process.env.SURFACE_VERIFY_DIRECT_MRA==='1';
const pairedMraAfterReopen=process.env.SURFACE_VERIFY_DIRECT_MRA_AFTER_REOPEN==='1';
const directoryOpen=nativeIndex>=0&&path.basename(source)==='patient-anatomy.json';
const document=directoryOpen?JSON.parse(fs.readFileSync(source,'utf8'))
  :(direct||refinedLayer||detailIds.length||selectedSeries)&&nativeIndex>=0&&source.endsWith('.rsworkspace')?JSON.parse(zipMember(source,'patient-anatomy.json')):null;
if(selectedSeries)assert(document?.series.some(row=>row.id===selectedSeries),'Unknown verification series '+selectedSeries);
if(branches)assert.equal(branches.manifest.sha256,sourceHash,'Branch receipt must bind this exact study');
for(const id of detailIds)assert(document?.layers.some(row=>row.id===id),'Unknown detail layer '+id);
const portable=direct&&!prior?path.join(output,'publisher-geometry.rsworkspace'):null;
const refinedPortable=refinedLayer&&directoryOpen&&process.env.SURFACE_VERIFY_SAVE_REFINED==='1'
  ?path.join(output,'conditioned-vault.rsworkspace'):null;
const savePath=portable??refinedPortable;
if(refinedPrior){
  assert(refinedLayer&&!directoryOpen&&refinedPrior.passed&&refinedPrior.pid!==process.pid);
  assert.equal(sourceHash,refinedPrior.refinedPortable.sha256);
  assert.equal(document.snapshot_id,refinedPrior.snapshotId);
}
if(savePath)dialog.showSaveDialog=async(_window,options)=>{
  assert.equal(options.title,'Save portable patient study');
  return {canceled:false,filePath:savePath};
};
if(pairedMra||pairedMraAfterReopen)assert(prior&&direct,'Verify paired MRA on the portable archive from a successful warm run');
if(prior){
  assert(direct&&prior.passed&&prior.pid!==process.pid&&prior.directSurface?.portable,'Use the successful warm receipt in a distinct Electron process');
  assert.equal(source,fs.realpathSync(prior.directSurface.portable.path));
  assert.equal(sourceHash,prior.directSurface.portable.sha256);
  assert.equal(sha(directFile),prior.directSurface.expectationSha256);
}
if(direct){
  assert(document?.schema==='patient-anatomy-v1'&&directoryOpen===!prior,'Warm input is a native manifest and cold input is the saved study archive');
  assert(direct.layerId&&direct.policy&&direct.transfer&&direct.points>0&&direct.triangles>0);
  const vtp=direct.sourceVtp;
  assert(/^[a-zA-Z0-9_./-]+\.vtp$/.test(vtp.path)&&!path.isAbsolute(vtp.path)&&!vtp.path.split('/').includes('..'));
  assert(/^[a-f0-9]{64}$/.test(vtp.sha256));
  const referenced=value=>value&&typeof value==='object'&&(
    value.path===vtp.path&&value.sha256===vtp.sha256||Object.values(value).some(referenced)
  );
  const layer=document.layers.find(row=>row.id===direct.layerId);
  assert(layer&&referenced(layer),'The surface layer must bind the published VTP member');
  if(directoryOpen)assert.equal(sha(path.join(path.dirname(source),vtp.path)),vtp.sha256);
}
const cacheSource=process.env.SURFACE_VERIFY_CACHE_FROM;
const retention=process.env.SURFACE_VERIFY_RETAIN_CACHE==='1';
let studyDirectory;
const engine=require(path.join(path.dirname(main),'sidecar.cjs')),Sidecar=engine.Sidecar;
let sidecarPid;
engine.Sidecar=class extends Sidecar{
  constructor(options){super(options);sidecarPid=this.child.pid;}
};
function clone(from,to){
  const target=fs.realpathSync(from);
  if(fs.statSync(target).isDirectory()){
    fs.mkdirSync(to); for(const name of fs.readdirSync(target))clone(path.join(target,name),path.join(to,name));
  }else fs.copyFileSync(target,to,fs.constants.COPYFILE_FICLONE);
}
if(directoryOpen){
  studyDirectory=path.join(profile,'study');fs.mkdirSync(studyDirectory);
  for(const member of ['patient-anatomy.json','arrays','originals','revisions']){
    const from=path.join(path.dirname(source),member),destination=path.join(studyDirectory,member);
    if(fs.existsSync(from))clone(from,destination);
  }
  const availableCache=cacheSource??(process.env.SURFACE_VERIFY_CACHE_MODE==='empty'?null:path.join(path.dirname(source),'meshes'));
  if(availableCache&&fs.existsSync(availableCache))clone(availableCache,path.join(studyDirectory,'meshes'));
  process.argv[inputIndex]=path.join(studyDirectory,'patient-anatomy.json');
}
const report={source,sourceHash,pid:process.pid,windowEverVisible:false,captures:{},requests:[],errors:[],memory:{peakElectronKiB:0,peakSidecarKiB:0,peakCombinedKiB:0},startedAt:new Date().toISOString()};
if(document){
  report.snapshotId=document.snapshot_id;
  report.grids=[...document.series,...document.layers].map(x=>({id:x.id,shape:x.data.shape,
    gridSpacingMm:x.grid_spacing_mm??[0,1,2].map(axis=>Math.hypot(...x.affine.slice(0,3).map(row=>row[axis]))),
    affine:x.affine,acquisitionResolution:x.acquisition_resolution}));
}
const dist=path.resolve(path.dirname(main),'../dist');
report.build=Object.fromEntries(fs.readdirSync(path.join(dist,'assets')).filter(n=>!n.endsWith('.map')).map(n=>[n,sha(path.join(dist,'assets',n))]));
if(prior)assert.deepEqual(report.build,prior.build,'Use the same renderer build for portable cold reopen');
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms)),started=performance.now();
let active=0;
const handle=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(name,handler)=>handle(name,async(event,args)=>{
  const at=performance.now()-started;active++;
  try{
    const result=await handler(event,args);
    if(['research:startupWorkspace','research:anatomyView','research:anatomyRegister'].includes(name)){
      const layers=result?.compartments?.map(({name,kind,sourceSeries,qa,surfaceDiagnostics,volumeMm3,mask,surface})=>({name,kind,sourceSeries,qa,surfaceDiagnostics,volumeMm3,maskSha256:mask?.sha256,triangles:surface?.indices?.shape?.reduce((a,b)=>a*b,1)/3,points:surface?.positions?.shape?.reduce((a,b)=>a*b,1)/3,indicesSha256:surface?.indices?.sha256,sourceMeshSha256:surface?.source?.sha256,displayMeshSha256:surface?.positions?.sha256}));
      report.requests.push({name,action:args?.action,seriesId:args?.seriesId,at,ms:performance.now()-started-at,
        sourceSignalSha256:result?.mri?.sha256,registrationStatus:result?.registrationStatus,layers});
    }
    return result;
  }catch(error){report.requests.push({name,error:String(error),at});throw error;}
  finally{active--;}
});
function sampleMemory(){
  const electron=app.getAppMetrics().reduce((sum,row)=>sum+(row.memory.workingSetSize||0),0);
  report.memory.peakElectronKiB=Math.max(report.memory.peakElectronKiB,electron);
  if(sidecarPid)execFile('/bin/ps',['-o','rss=','-p',String(sidecarPid)],{timeout:1500},(error,stdout)=>{
    if(error)return;
    const sidecar=Number(stdout.trim());
    report.memory.peakSidecarKiB=Math.max(report.memory.peakSidecarKiB,sidecar);
    report.memory.peakCombinedKiB=Math.max(report.memory.peakCombinedKiB,electron+sidecar);
  });
}
app.on('browser-window-created',(_event,window)=>{
  window.on('show',()=>{report.windowEverVisible=true;window.hide();});
  window.webContents.on('console-message',(_event,detail)=>{if(detail.level==='error')report.errors.push(detail.message);});
  window.webContents.once('did-finish-load',async()=>{
    window.webContents.setBackgroundThrottling(false);window.setContentSize(1440,960);
    const memory=setInterval(sampleMemory,300),js=code=>window.webContents.executeJavaScript(code,true);
    const wait=async(check,label,seconds=250)=>{
      const deadline=performance.now()+seconds*1000;
      while(performance.now()<deadline){const value=await check();if(value)return value;await delay(125);}
      throw Error('Timed out waiting for '+label);
    };
    const click=async label=>{
      await js(`(()=>{const buttons=[...document.querySelectorAll('button')];const b=buttons.find(e=>e.getAttribute('aria-label')===${JSON.stringify(label)})??buttons.find(e=>e.textContent.trim()===${JSON.stringify(label)});if(!b||b.disabled)throw Error('Missing enabled '+${JSON.stringify(label)});b.click()})()`);
    };
    const set=async(selector,value)=>{
      await js(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});if(!e||e.disabled)throw Error('Missing enabled control');Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(e,${JSON.stringify(value)});e.dispatchEvent(new Event('change',{bubbles:true}))})()`);
    };
    const ready=async(minimum=-1)=>{
      await delay(125);
      return wait(async()=>{
        if(report.requests.some(r=>r.error&&r.name.startsWith('research:anatomy')))throw Error(report.requests.find(r=>r.error).error);
        if(active)return null;
        const state=await js(`(()=>{const v=document.querySelector('.rl-viewer'),p=window.surfaceVerification;
          const f=p.frames.findLast(frame=>frame.sourceCaseHash===v?.dataset.sourceCaseHash
            &&(frame.displayIdentity??'')===(v?.dataset.sourceDisplayIdentity??''));
          const gpu=f&&p.gpu.findLast(g=>g.serial===f.serial&&g.completed&&g.sourceCaseHash===f.sourceCaseHash&&g.displayIdentity===f.displayIdentity);
          return {frame:f,gpu,visible:v?.dataset.rendererReady==='true',pending:!!document.querySelector('.viewer-loading,[data-layer-pending=true]'),layers:v?.dataset.visibleLayerIds,modality:v?.dataset.sourceModality};
        })()`);
        return state.gpu&&state.visible&&!state.pending&&state.frame.serial>minimum&&!state.frame.interactive?state:null;
      },'GPU-confirmed resting anatomy');
    };
    const capture=async(name,minimum=-1)=>{
      const view=await ready(minimum);
      assert(!window.isVisible());
      const image=await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true});
      assert(!image.isEmpty(),'No hidden-window screenshot');
      fs.writeFileSync(path.join(output,name+'.png'),image.toPNG());
      const paneBounds=await js(`[...document.querySelectorAll('.rl-viewer-anatomy-image,.rl-viewer-slice')].map(e=>{
        const r=e.getBoundingClientRect(); return {className:e.className,x:r.x,y:r.y,width:r.width,height:r.height,windowWidth:innerWidth};
      })`);
      const imageScale=image.getSize().width/paneBounds[0].windowWidth;
      const pixels=Object.fromEntries(paneBounds.map(pane=>{
        const id=['anatomy-image','axial','coronal','sagittal'].find(name=>pane.className.includes(name));
        const rect={x:Math.ceil((pane.x+8)*imageScale),y:Math.ceil((pane.y+40)*imageScale),
          width:Math.floor((pane.width-16)*imageScale),height:Math.floor((pane.height-58)*imageScale)};
        const data=image.crop(rect).toBitmap();
        return [id,{rect,sha256:crypto.createHash('sha256').update(data).digest('hex')}];
      }));
      report.captures[name]={...view,imageSize:image.getSize(),atMs:performance.now()-started,pixels};
      console.log('Captured '+name);
      return view.frame;
    };
    const serial=()=>js('window.surfaceVerification.frames.at(-1)?.serial??-1');
    const smooth=async(enabled)=>{
      await click('Appearance');
      await js(`(()=>{const c=document.querySelector('input[aria-label="Smooth 3D surfaces"]');if(c.checked!==${enabled})c.click()})()`);
      await click('Close view controls');
    };
    const camera=async(preset)=>{
      await click('Layout');await set('[aria-label="Camera preset"]',preset);await click('Close view controls');
    };
    const sidePanel=async label=>{await js(`(()=>{const x=[...document.querySelectorAll('summary')].find(e=>e.textContent.includes(${JSON.stringify(label)}));if(x)x.parentElement.open=true;})()`);};
    const compareSourceFrame=(before,after,label)=>{
      for(const point of ['target','position','up'])assert(before.frame.cameraState[point].every((value,i)=>Math.abs(value-after.frame.cameraState[point][i])<1e-5),label+' camera differs');
      assert.equal(before.frame.sourceSignalFingerprint,after.frame.sourceSignalFingerprint,label+' source signal differs');
      assert.deepEqual(before.frame.cursor,after.frame.cursor,label+' slice cursor differs');
    };
    const compareSlices=(before,after,label)=>{
      for(const plane of ['axial','coronal','sagittal'])assert.equal(before.pixels[plane].sha256,after.pixels[plane].sha256,label+' changed '+plane+' source pixels');
    };
    const verifyPairedMra=async()=>{
      const mra=document.series.find(series=>series.id==='mra');
      const artery=document.layers.find(layer=>layer.id==='mra-publisher-artery');
      assert(mra?.modality==='MRA'&&artery?.source_series===mra.id&&artery.data.dtype==='uint8');
      assert.deepEqual(artery.data.shape,mra.data.shape);
      const packed=zipMember(source,artery.data.path,64*1024*1024);
      const hash=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
      assert.equal(hash(packed),artery.data.sha256);
      assert.equal(packed.subarray(0,6).toString('hex'),'934e554d5059');
      const version=packed[6];
      assert([1,2,3].includes(version),'Unsupported NumPy archive layout');
      const headerStart=version===1?10:12;
      const start=headerStart+(version===1?packed.readUInt16LE(8):packed.readUInt32LE(8));
      const header=packed.toString('utf8',headerStart,start);
      assert(header.includes("'descr': '|u1'")&&header.includes("'fortran_order': False"));
      assert.equal(packed.length-start,artery.data.shape.reduce((a,b)=>a*b,1));
      const maskSha256=hash(packed.subarray(start));
      const select=async id=>{
        await wait(()=>js(studyScanEnabledScript(id)),id+' source control');
        await js(studyScanSelectionScript(id));
        await wait(()=>js(`document.querySelector('.native-workspace')?.dataset.nativeSeries===${JSON.stringify(id)}`),id+' active source');
        return ready();
      };
      const initial=await ready();
      const cta=document.reference_series;
      await select(mra.id);
      await click('Show structures only');
      const shown=await js(`document.querySelector(${JSON.stringify('[data-structure-id="'+artery.id+'"] .native-structure-visibility')})?.getAttribute('aria-pressed')==='true'`);
      if(!shown)await js(studyVisibilityToggleScript('Show '+artery.label));
      const displayed=await capture('paired-mra-independent');
      const mraRequests=report.requests.filter(request=>request.seriesId===mra.id&&request.layers);
      const own=mraRequests.flatMap(request=>request.layers).find(layer=>layer.name===artery.id);
      assert(own?.triangles>0&&own.sourceSeries===mra.id);
      assert.equal(own.maskSha256,maskSha256);
      assert(mraRequests.every(request=>request.registrationStatus==='independent source frame'
        &&request.layers.every(layer=>layer.sourceSeries===mra.id)),'MRA must not borrow CTA anatomy');
      assert.deepEqual(displayed.sourceShape,mra.data.shape);
      assert.notEqual(displayed.sourceSignalFingerprint,initial.frame.sourceSignalFingerprint);
      assert(await js(`!!document.querySelector('.patient-model-summary[data-frame="independent"]')`));
      const mraView=report.captures['paired-mra-independent'];
      assert(JSON.parse(mraView.layers).includes(artery.id));
      const restored=await select(cta);
      const ctaView=await capture('paired-cta-restored');
      assert.equal(ctaView.sourceSignalFingerprint,initial.frame.sourceSignalFingerprint);
      assert.deepEqual(ctaView.sourceShape,initial.frame.sourceShape);
      assert(JSON.parse(restored.layers).includes(direct.layerId));
      const publisher=report.requests.findLast(request=>request.seriesId===cta&&request.layers?.some(layer=>layer.name===direct.layerId))
        ?.layers.find(layer=>layer.name===direct.layerId);
      assert.equal(publisher?.sourceMeshSha256,direct.transfer.sourceMeshSha256);
      assert.equal(publisher?.indicesSha256,direct.transfer.indicesSha256);
      assert.equal(publisher?.surfaceDiagnostics?.publisherSourceSha256,direct.sourceVtp.sha256);
      report.pairedMra={id:mra.id,shape:mra.data.shape,originalSha256:mra.original.sha256,
        sourceSignal:displayed.sourceSignalFingerprint,arterialLayer:own,independent:true,ctaRestored:true};
    };
    const verifyDirect=async()=>{
      const layer=document.layers.find(row=>row.id===direct.layerId);
      const evidence=report.requests.flatMap(request=>request.layers??[]).find(row=>row.name===layer.id);
      assert(evidence,'Actual anatomy response omitted the direct surface');
      const transfer=Object.fromEntries(Object.keys(direct.transfer).map(key=>[key,evidence[key]]));
      for(const key of ['sourceMeshSha256','displayMeshSha256','indicesSha256','maskSha256']){
        assert(/^[a-f0-9]{64}$/.test(direct.transfer[key]),'Invalid expected transport hash for '+key);
      }
      assert.deepEqual(transfer,direct.transfer);
      assert.equal(evidence.sourceMeshSha256,evidence.displayMeshSha256);
      assert.equal(evidence.points,direct.points);
      assert.equal(evidence.triangles,direct.triangles);
      assert.equal(evidence.surfaceDiagnostics?.policy,direct.policy);
      assert.equal(evidence.surfaceDiagnostics?.publisherSourceSha256,direct.sourceVtp.sha256);
      assert.equal(evidence.surfaceDiagnostics?.maxDisplacementMm,0);
      const installed=await js(`window.surfaceVerification.surface.find(layer=>layer.name===${JSON.stringify(layer.id)}&&!layer.empty)`);
      assert.equal(installed?.diagnostics?.policy,direct.policy,'Renderer must install the publisher mesh');
      assert.equal(installed.diagnostics.publisherSourceSha256,direct.sourceVtp.sha256);
      const raw=report.captures['source-oblique'],smooth=report.captures['smooth-oblique'];
      assert([raw,smooth].every(view=>view.gpu.completed&&JSON.parse(view.layers).includes(layer.id)
        &&view.frame.drawStatistics.triangles>=direct.triangles),'Publisher layer must contribute to the completed GPU view');
      compareSourceFrame(raw,smooth,'Publisher selector');
      compareSlices(raw,smooth,'Publisher selector');
      await js(studyVisibilityToggleScript('Hide '+layer.label));
      await capture('publisher-mesh-hidden',smooth.frame.serial);
      const hidden=report.captures['publisher-mesh-hidden'];
      assert(!JSON.parse(hidden.layers).includes(layer.id));
      compareSourceFrame(smooth,hidden,'Hidden publisher layer');
      assert.notEqual(smooth.pixels['anatomy-image'].sha256,hidden.pixels['anatomy-image'].sha256,
        'Hiding the publisher layer must change the real 3D pixels');
      assert(smooth.frame.drawStatistics.triangles-hidden.frame.drawStatistics.triangles>=direct.triangles,
        'Hiding the publisher layer must remove its triangles from GPU draw statistics');
      await js(studyVisibilityToggleScript('Show '+layer.label));
      await capture('publisher-mesh-restored',hidden.frame.serial);
      const restored=report.captures['publisher-mesh-restored'];
      assert(JSON.parse(restored.layers).includes(layer.id));
      compareSourceFrame(smooth,restored,'Restored publisher layer');
      compareSlices(smooth,restored,'Restored publisher layer');
      report.directSurface={expectationSha256:sha(directFile),layerId:layer.id,
        sourceVtp:direct.sourceVtp,policy:direct.policy,points:evidence.points,triangles:evidence.triangles,transfer};
      if(directConditioned){
        const rows=directConditioned.refinedComparisons;
        assert.equal(rows.length,2);
        assert.deepEqual(rows.map(row=>row.layer).sort(),['brain','broader_skull_display']);
        report.directSurface.conditioned=rows.map(expected=>{
          const row=document.layers.find(layer=>layer.id===expected.layer);
          const attachment=row.conditioned_surface;
          assert.equal(attachment.proof.sha256,expected.probeSha256);
          assert.equal(attachment.stock_manifest_sha256,expected.stockManifest);
          const response=report.requests.flatMap(request=>request.layers??[]).find(layer=>layer.name===row.id);
          assert.equal(response?.displayMeshSha256,expected.displaySha256);
          assert.equal(response.surfaceDiagnostics?.policy,attachment.schema);
          assert(JSON.parse(smooth.layers).includes(row.id));
          return {layer:row.id,displaySha256:response.displayMeshSha256,probeSha256:attachment.proof.sha256};
        });
      }
      if(portable){
        await click('Scans');
        await click('Save patient study file…');
        await wait(async()=>fs.existsSync(portable)&&await js(`document.body.innerText.includes('Patient study file saved')`),'portable study save');
        report.directSurface.portable={path:portable,sha256:sha(portable)};
      }else{
        assert.equal(report.snapshotId,prior.snapshotId);
        assert.deepEqual(transfer,prior.directSurface.transfer);
        assert.deepEqual(report.directSurface.conditioned,prior.directSurface.conditioned);
        compareSourceFrame(prior.captures['smooth-oblique'],smooth,'Cold publisher layer');
        compareSlices(prior.captures['smooth-oblique'],smooth,'Cold publisher layer');
        report.directSurface.coldReopened=true;
      }
      const member=zipMember(portable??source,direct.sourceVtp.path);
      assert.equal(crypto.createHash('sha256').update(member).digest('hex'),direct.sourceVtp.sha256,
        'Portable study changed the publisher VTP bytes');
      if(directConditioned){
        for(const {layer} of report.directSurface.conditioned){
          const {positions,proof}=document.layers.find(row=>row.id===layer).conditioned_surface;
          for(const item of [positions,proof]){
            const digest=crypto.createHash('sha256').update(zipMember(portable??source,item.path)).digest('hex');
            assert.equal(digest,item.sha256,'Portable study changed a verified conditioned surface member');
          }
        }
      }
    };
    try{
      await js(`window.surfaceVerification={frames:[],gpu:[],surface:[],refinement:[]};for(const [name,key] of [['anatomy-frame','frames'],['anatomy-gpu','gpu'],['anatomy-surface','surface'],['anatomy-refined','refinement']])document.addEventListener(name,e=>{window.surfaceVerification[key].push(e.detail);});`);
      await ready();report.freshProcessToReadyMs=performance.now()-started;
      if(pairedMra){await verifyPairedMra();report.passed=true;return;}
      if(selectedSeries){
        await click('Scans');
        await wait(()=>js(studyScanEnabledScript(selectedSeries)),selectedSeries+' source control');
        await js(studyScanSelectionScript(selectedSeries));
        await wait(()=>js(`document.querySelector('.native-workspace')?.dataset.nativeSeries===${JSON.stringify(selectedSeries)}`),selectedSeries+' active source');
        await ready();
        report.selectedSeries=selectedSeries;
        await click('Anatomy');
      }
      if(refinedPortable){
        await click('Scans');await click('Save patient study file…');
        await wait(async()=>fs.existsSync(refinedPortable)&&await js(`document.body.innerText.includes('Patient study file saved')`),'refined portable study save');
        const members=refinedIds.flatMap(id=>{
          const attachment=document.layers.find(row=>row.id===id).conditioned_surface;
          return [attachment.positions,attachment.proof];
        });
        if(document.data_case.source_receipt)members.push(document.data_case.source_receipt);
        for(const member of members){
          assert.equal(crypto.createHash('sha256').update(zipMember(refinedPortable,member.path)).digest('hex'),member.sha256);
        }
        report.refinedPortable={path:refinedPortable,sha256:sha(refinedPortable)};
        await click('Anatomy');
      }
      if(process.env.SURFACE_VERIFY_NATIVE_PRESENTATION==='1'){
        await js(`(()=>{const button=[...document.querySelectorAll('.anatomy-presets button')].find(b=>b.textContent.trim()==='Structures');
          if(!button||button.disabled)throw Error('Native Structures presentation unavailable');
          button.click();
        })()`);
      }
      await click('Show structures only');
      const hasBone=report.requests.some(request=>request.layers?.some(layer=>['skull','bone'].includes(layer.kind)&&layer.points>0));
      const skull=hasBone&&await js(`!!document.querySelector('select[aria-label="Skull view"]')`);
      if(skull){await sidePanel('Display & fragments');await set('[aria-label="Skull view"]','transparent');}
      if(process.env.SURFACE_VERIFY_NATIVE_PRESENTATION==='1'){
        const brain=document.layers.find(layer=>layer.kind==='brain'&&
          (!selectedSeries||layer.source_series===selectedSeries||['reference','reviewed','source-provided'].includes(document.transforms[layer.source_series]?.status)&&
            ['reference','reviewed','source-provided'].includes(document.transforms[selectedSeries]?.status)));
        if(brain){
          await click(brain.label);
          await js(inputScript(`${brain.label} opacity`,.16));
          await click('Close layer details');
        }
      }
      const pair=async(name,frameView=async()=>{})=>{
        await smooth(false);await frameView();
        const raw=await capture('source-'+name);
        const warmStarted=performance.now();
        await smooth(true);await capture('smooth-'+name,raw.serial);
        report.captures['smooth-'+name].toggleToCaptureMs=performance.now()-warmStarted;
      };
      const visibility=()=>js(studyLayerVisibilityScript());
      const enabledControl=(className,label)=>js(`[...document.getElementsByClassName(${JSON.stringify(className)})].some(e=>e.getAttribute('aria-label')===${JSON.stringify(label)}&&!e.disabled)`);
      const inputValue=label=>js(`[...document.querySelectorAll('input')].find(e=>e.getAttribute('aria-label')===${JSON.stringify(label)}).value`);
      const showLayer=async(id,show)=>{
        const control=(await visibility()).find(row=>row.id===id);
        assert(control,'Missing layer '+id);
        if(control.checked!==show){
          const label=show?control.label:control.label.replace(/^Show /,'Hide ');
          await wait(()=>enabledControl('native-structure-visibility',label),label);
          await js(studyVisibilityToggleScript(label));
          await wait(async()=>(await visibility()).find(row=>row.id===id)?.checked===show,label+' state');
        }
      };
      const focusLayer=async label=>{
        await js(studyLayerSelectionScript([label]));
        await wait(()=>enabledControl('native-structure-focus','Focus '+label),'focus '+label);
        await click('Focus '+label);
        await ready();
      };
      const verifyOverlay=(before,after)=>{
        const a=report.captures[before],b=report.captures[after];
        compareSourceFrame(a,b,after);
        assert.notEqual(a.pixels['anatomy-image'].sha256,b.pixels['anatomy-image'].sha256,after+' did not change the 3D pixels');
      };
      await pair('oblique',()=>click('Fit 3D view'));
      if(document?.five_layer_inspection){
        const groups=document.five_layer_inspection.anatomy_layers;
        const visible=JSON.parse(report.captures['smooth-oblique'].layers);
        const surfaces=report.requests.flatMap(request=>request.layers??[]);
        for(const [group,ids] of Object.entries(groups)){
          assert(ids.some(id=>visible.includes(id)&&surfaces.some(row=>row.name===id&&row.triangles>0)),
            group+' must be visible and have native geometry at startup');
        }
        report.fiveLayerInspection={groups,visible};
      }
      if(direct){
        await verifyDirect();
        if(pairedMraAfterReopen)await verifyPairedMra();
        report.passed=true;
        return;
      }
      await pair('anterior',()=>camera('anterior'));
      if(skull){
        await set('[aria-label="Skull view"]','exterior');
        await pair('skull-anterior');
        await set('[aria-label="Skull view"]','transparent');
      }
      if(branches){
        const whole=document.layers.find(row=>row.id==='publisher_cow_arterial_mesh');
        const initial=await visibility();
        for(const {id,checked} of initial)if(checked)await showLayer(id,false);
        await showLayer(whole.id,true);
        await focusLayer(whole.label);
        const raw=await capture('whole-publisher-vessels');
        await showLayer(whole.id,false);
        const manualColors=[];
        for(const branch of branches.branches){
          const layer=document.layers.find(row=>row.id===branch.layer);
          await showLayer(layer.id,true);
          await wait(()=>js(`[...document.getElementsByClassName('native-structure-name')].some(e=>e.title===${JSON.stringify(layer.label)}&&!e.disabled)`),layer.label+' controls');
          await js(studyLayerSelectionScript([layer.label]));
          const label=layer.label+' color';
          if(await inputValue(label)!==layer.color){
            await js(inputScript(label,layer.color));
            assert.equal(await inputValue(label),layer.color);
            manualColors.push(layer.id);
          }
        }
        await capture('named-publisher-branches',raw.serial);
        verifyOverlay('whole-publisher-vessels','named-publisher-branches');
        const wholeView=report.captures['whole-publisher-vessels'];
        const branchView=report.captures['named-publisher-branches'];
        assert.deepEqual(JSON.parse(wholeView.layers),[whole.id]);
        assert.deepEqual(JSON.parse(branchView.layers).sort(),branches.branches.map(row=>row.layer).sort());
        assert.equal(branchView.frame.drawStatistics.triangles,wholeView.frame.drawStatistics.triangles);
        const rendered=Object.fromEntries(report.requests.flatMap(request=>request.layers??[])
          .filter(row=>row.triangles>0).map(row=>[row.name,row]));
        for(const branch of branches.branches){
          const surface=rendered[branch.layer];
          assert.equal(surface?.triangles,branch.triangles,branch.layer+' rendered triangle count');
          assert.equal(surface.points,branch.points,branch.layer+' rendered vertex count');
          assert.equal(surface.sourceMeshSha256,surface.displayMeshSha256);
          assert.equal(surface.surfaceDiagnostics?.policy,'publisher-vtp-ras-mm-v1');
          assert.equal(surface.surfaceDiagnostics?.maxDisplacementMm,0);
        }
        assert.equal(rendered[whole.id]?.triangles,branches.branches.reduce((total,row)=>total+row.triangles,0));
        report.publisherBranches={receiptSha256:sha(branchFile),manualColors,triangles:rendered[whole.id].triangles};
        for(const {id,checked} of initial)await showLayer(id,checked);
      }
      for(const id of detailIds){
        const layer=document.layers.find(row=>row.id===id);
        const wasVisible=(await visibility()).find(row=>row.id===id)?.checked;
        await showLayer(id,true);
        await focusLayer(layer.label);
        if(detailCursor){
          for(const [axis,plane] of ['Sagittal','Coronal','Axial'].entries()){
            await js(inputScript(plane+' slice position',detailCursor[axis]));
          }
          const positioned=await ready();
          assert(positioned.frame.cursor.every((v,i)=>Math.abs(v-detailCursor[i])<=.51),'Could not match the requested source location');
        }
        const opacity=await inputValue(layer.label+' opacity');
        if(opacity!=='1')await js(inputScript(layer.label+' opacity',1));
        const shown=await capture('with-'+id);
        await showLayer(id,false);
        await capture('without-'+id,shown.serial);
        verifyOverlay('without-'+id,'with-'+id);
        if(wasVisible)await showLayer(id,true);
        await js(inputScript(layer.label+' opacity',opacity));
      }
      if(process.env.SURFACE_VERIFY_FOCUS==='1'){
        const labels=await js(`[...document.querySelectorAll('.native-structure-name')].map(e=>e.textContent.trim())`);
        for(const [kind,re] of [['brain',/brain/i],['tumor',/tumor|lesion/i],['vessels',/vessel|arter|vein|branch/i]]){
          const label=labels.find(name=>re.test(name));if(!label)continue;
          await click(label);await click('Isolate');
          const selector='input[aria-label="'+label+' opacity"]';
          const opacity=await js(`document.querySelector(${JSON.stringify(selector)}).value`);
          const setOpacity=async value=>{await js(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(e,${JSON.stringify(String(value))});e.dispatchEvent(new Event('input',{bubbles:true}));})()`);};
          if(opacity!=='1')await setOpacity(1);
          await pair(kind,()=>click('Focus '+label));
          if(opacity!=='1')await setOpacity(opacity);
          await click('Restore others');
        }
      }
      await click('Fit 3D view');
      const resting=await ready();
      const center=await js(`(()=>{const r=document.querySelector('.rl-viewer-anatomy-image').getBoundingClientRect();return {x:Math.round(r.x+r.width*.5),y:Math.round(r.y+r.height*.5)}})()`);
      window.webContents.sendInputEvent({type:'mouseDown',...center,button:'left',clickCount:1});
      for(let k=1;k<=90;k++){
        window.webContents.sendInputEvent({type:'mouseMove',x:center.x+k,y:center.y+Math.round(10*Math.sin(k/9)),modifiers:['leftButtonDown']});
        await delay(16);
      }
      window.webContents.sendInputEvent({type:'mouseUp',x:center.x+90,y:center.y+Math.round(10*Math.sin(10)),button:'left',clickCount:1});
      await capture('post-orbit-refinement',resting.frame.serial);
      const data=await js('window.surfaceVerification');
      const percentile=(values,p)=>values.length?[...values].sort((a,b)=>a-b)[Math.floor((values.length-1)*p)]:null;
      const metrics=frames=>{
        const fences=frames.flatMap(f=>data.gpu.find(g=>g.serial===f.serial&&g.completed)?.milliseconds??[]);
        return {frames:frames.length,cpuP50Ms:percentile(frames.map(f=>f.cpuMs),.5),cpuP95Ms:percentile(frames.map(f=>f.cpuMs),.95),fenceP50Ms:percentile(fences,.5),fenceP95Ms:percentile(fences,.95),submissionGapP50Ms:percentile(frames.slice(1).map((f,i)=>f.at-frames[i].at),.5)};
      };
      report.resting=metrics(data.frames.filter(f=>!f.interactive));report.orbit=metrics(data.frames.filter(f=>f.interactive));
      assert(report.orbit.frames>0,'No actual interactive frame submitted');
      assert(!report.windowEverVisible,'Verification window appeared');
      for(const [name,view] of Object.entries(report.captures)){
        const raw=report.captures[name.replace(/^smooth-/,'source-')];if(!name.startsWith('smooth-')||!raw)continue;
        compareSourceFrame(raw,view,name);
        compareSlices(raw,view,name);
      }
      if(process.env.SURFACE_VERIFY_NATIVE_PRESENTATION==='1'){
        assert.notEqual(report.captures['source-oblique'].pixels['anatomy-image'].sha256,
          report.captures['smooth-oblique'].pixels['anatomy-image'].sha256,
          'Conditioned contours should change the actual 3D pixels at the same source and camera');
      }
      if(process.env.SURFACE_VERIFY_EXPECTED){
        const before=JSON.parse(fs.readFileSync(process.env.SURFACE_VERIFY_EXPECTED,'utf8'));
        const refined=refinedIds.map(id=>document.layers.find(row=>row.id===id)?.conditioned_surface);
        if(refinedLayer){
          assert(before.passed&&refined.every(row=>row?.stock_manifest_sha256===before.sourceHash),
            'Expected the checksum-bound original study for this refinement');
          // Both reports must compare the serialized grid record. JSON removes
          // optional undefined fields and preserves signed-zero geometry as 0.
          assert.deepEqual(JSON.parse(JSON.stringify(report.grids)),before.grids);
        }else assert(before.passed&&before.sourceHash===report.sourceHash,'Expected successful baseline with exact same source');
        for(const [name,view] of Object.entries(report.captures)){
          if(name==='post-orbit-refinement')continue;
          compareSourceFrame(before.captures[name],view,name+' baseline');
          if(refinedLayer)compareSlices(before.captures[name],view,name+' baseline');
        }
        const identities=log=>log.requests.flatMap(request=>request.layers??[]).map(layer=>[
          layer.name,layer.indicesSha256,layer.sourceMeshSha256,layer.maskSha256,layer.volumeMm3,layer.qa?.componentCount
        ]);
        assert.deepEqual(identities(before),identities(report),'Source masks, topology, geometry or source volume changed');
        if(refinedLayer){
          const layer=(log,name)=>log.requests.flatMap(request=>request.layers??[]).find(row=>row.name===name);
          const comparisons=refinedIds.map((id,index)=>{
            const stock=layer(before,id),current=layer(report,id),attachment=refined[index];
            assert.equal(current?.surfaceDiagnostics.policy,attachment.schema);
            assert.notEqual(current.displayMeshSha256,stock.displayMeshSha256);
            return {layer:id,stockManifest:before.sourceHash,stockDisplaySha256:stock.displayMeshSha256,
              displaySha256:current.displayMeshSha256,probeSha256:attachment.proof.sha256};
          });
          assert.equal(report.captures['source-skull-anterior'].pixels['anatomy-image'].sha256,
            before.captures['source-skull-anterior'].pixels['anatomy-image'].sha256);
          assert.notEqual(report.captures['smooth-skull-anterior'].pixels['anatomy-image'].sha256,
            before.captures['smooth-skull-anterior'].pixels['anatomy-image'].sha256);
          report.refinedComparisons=comparisons;
          if(comparisons.length===1)report.refinedComparison=comparisons[0];
          if(refinedPrior){
            assert.deepEqual(comparisons,refinedPrior.refinedComparisons??[refinedPrior.refinedComparison]);
            compareSourceFrame(refinedPrior.captures['smooth-skull-anterior'],report.captures['smooth-skull-anterior'],'Cold conditioned vault');
            compareSlices(refinedPrior.captures['smooth-skull-anterior'],report.captures['smooth-skull-anterior'],'Cold conditioned vault');
            assert.equal(report.captures['smooth-skull-anterior'].pixels['anatomy-image'].sha256,
              refinedPrior.captures['smooth-skull-anterior'].pixels['anatomy-image'].sha256);
            report.refinedColdReopened=true;
          }
        }
        report.baselineMatched=true;
      }
      if(retention&&studyDirectory&&fs.existsSync(path.join(studyDirectory,'meshes')))clone(path.join(studyDirectory,'meshes'),path.join(output,'meshes'));
      report.passed=true;
    }catch(error){
      report.failure=String(error.stack||error);
      report.body=await js('document.body.innerText').catch(()=>null);
      const image=await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true}).catch(()=>null);
      if(image&&!image.isEmpty())fs.writeFileSync(path.join(output,'failure.png'),image.toPNG());
    }finally{
      clearInterval(memory);sampleMemory();
      Object.assign(report,await js('window.surfaceVerification').catch(()=>({})));
      report.sourceUnchanged=sha(source)===sourceHash;
      report.passed=Boolean(report.passed&&report.sourceUnchanged&&!report.windowEverVisible&&!report.errors.length);
      fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));
      console.log(JSON.stringify({passed:report.passed,failure:report.failure,errors:report.errors,orbit:report.orbit}));
      if(!report.passed)process.exitCode=1;
      app.quit();
    }
  });
});
app.on('quit',()=>fs.rmSync(profile,{recursive:true,force:true}));
require(main);
