'use strict';

// Hidden native anatomy captures. Recipes describe exact viewer preferences so
// the same physical cameras, context opacity and source cursors can be reused.
const {app,dialog,ipcMain,nativeImage}=require('electron');
const assert=require('node:assert/strict');
const {createHash}=require('node:crypto');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const {enabledControlScript,inputScript}=require('./verify-eden-planning-helpers.cjs');
const {checkedInteraction,interactionScript,summarizeInteraction}=require('./vascular-interaction.cjs');
process.argv.push('--automation-hidden');
if(process.platform==='darwin')app.setActivationPolicy('accessory');

function argument(name){
  const index=process.argv.indexOf(name);
  assert(index>=0&&process.argv[index+1],`Pass ${name}`);
  return path.resolve(process.argv[index+1]);
}
function dimension(name,fallback){
  const index=process.argv.indexOf(name);
  if(index<0)return fallback;
  const value=Number(process.argv[index+1]);
  assert(Number.isInteger(value)&&value>=480&&value<=2560,`Invalid ${name}`);
  return value;
}
const source=fs.realpathSync(argument('--anatomy'));
const linked=/^vascular-review(?:\.[0-9a-f]{12})?\.anatomy\.json$/.test(path.basename(source));
assert(linked||path.basename(source)==='patient-anatomy.json');
const canonical=linked?path.join(path.dirname(source),'patient-anatomy.json'):source;
const output=argument('--output');
fs.mkdirSync(output);
const document=JSON.parse(fs.readFileSync(source,'utf8'));
const recipes=JSON.parse(fs.readFileSync(argument('--views'),'utf8'));
const expected=process.argv.includes('--expected')?JSON.parse(fs.readFileSync(argument('--expected'),'utf8')):null;
const windowSize={width:dimension('--width',1440),height:dimension('--height',960)};
const deviceScale=expected?.deviceScale??(expected?.views[0]?.size.width===windowSize.width*2?2:1);
assert([1,2].includes(deviceScale),'Unsupported reference display scale');
const exportName=process.argv.includes('--export-view')?process.argv[process.argv.indexOf('--export-view')+1]:undefined;
// Path-traced stills stop refining at this sample count, so they are captured
// at a fixed, reported count instead of whenever refinement ends.
const cinematicSamples=process.argv.includes('--cinematic-samples')?Number(process.argv[process.argv.indexOf('--cinematic-samples')+1]):256;
assert(Number.isInteger(cinematicSamples)&&cinematicSamples>=1&&cinematicSamples<=1024,'Invalid --cinematic-samples');
// The page takes its pixel ratio and refresh rate from the display its hidden
// window sits on, so comparable runs name that display's scale factor.
const displayScale=process.argv.includes('--display-scale')?Number(process.argv[process.argv.indexOf('--display-scale')+1]):undefined;
assert(displayScale===undefined||[1,2].includes(displayScale),'Invalid --display-scale');
assert(Array.isArray(recipes)&&recipes.length&&recipes.length<=32);
assert.equal(new Set(recipes.map(r=>r.name)).size,recipes.length);
assert(!process.argv.includes('--export-view')||recipes.some(recipe=>recipe.name===exportName),'Unknown export view');
const scans=new Map(document.series.map(scan=>[scan.id,scan]));
const layers=new Set([...document.layers,...document.tractography??[]].map(layer=>layer.id));
for(const recipe of recipes){
  assert(/^[A-Za-z0-9_-]{1,72}$/.test(recipe.name));
  assert(scans.has(recipe.seriesId),`Unknown scan ${recipe.seriesId}`);
  assert(recipe.layers.length&&recipe.layers.every(id=>layers.has(id)),`Unknown layer in ${recipe.name}`);
  if(recipe.useDefaultState)assert(recipes.length===1&&!recipe.settings&&!recipe.cursor&&!recipe.scope&&!recipe.compareLayerIds,
    'A default-state capture cannot supply view overrides');
  if(recipe.inspectCtExteriorToggle)assert(recipe.useDefaultState,'Check the detached CT signal control from the ordinary default');
  if(recipe.inspectCtaFullHead)assert(recipe.useDefaultState,'Compare the ordinary CTA opening with the Full head control');
  if(recipe.inspectCtaCoverage)assert(recipe.useDefaultState,'Compare arterial coverage from the ordinary CTA opening');
  assert(recipe.applyVesselsPreset===undefined||typeof recipe.applyVesselsPreset==='boolean');
  assert(recipe.applyScanPreset===undefined||typeof recipe.applyScanPreset==='boolean');
  assert(!recipe.applyScanPreset||!recipe.applyVesselsPreset,'Apply one ordinary preset per capture');
  if(recipe.inspectCtBoneOverlay)assert(recipe.applyScanPreset||recipe.settings?.volumeRendering?.showCtBoneOverlay===false,
    'Start with the acquired CT bone view before comparing the overlay');
  if(recipe.inspectPublisherSmoothing)assert(recipe.settings?.smoothSurfaces===true,
    'Start with conditioned publisher geometry before comparing source positions');
  if(recipe.inspectSourceVolumeToggle)assert(recipe.applyVesselsPreset,'Check the volume control after applying the ordinary Vessels preset');
  if(recipe.inspectBalancedTof)assert(recipe.applyVesselsPreset||recipe.settings?.volumeRendering?.balanceTofContext===true,
    'Enable balanced TOF context before comparing the control');
  if(recipe.expectLabelTextureBytesPerVoxel!==undefined){
    assert([1,2,4].includes(recipe.expectLabelTextureBytesPerVoxel),'Unsupported number of label texture channels');
    assert(typeof recipe.settings?.volumeRendering?.enabled==='boolean','Specify the acquired volume state in a live label check');
    assert(exportName!==recipe.name,'Measure live label storage in an export-free view');
  }
  assert(!recipe.compareLayerIds||recipe.compareLayerIds.length===2&&recipe.compareLayerIds.every(id=>recipe.layers.includes(id)));
  assert(recipe.depthSlab===undefined||typeof recipe.depthSlab==='boolean');
  assert(recipe.slabContext===undefined||typeof recipe.slabContext==='boolean'&&recipe.depthSlab);
  assert(!recipe.branchLayerId||recipe.layers.includes(recipe.branchLayerId));
  if(recipe.tofGapReview)assert(recipe.layers.includes(recipe.tofGapReview.layerId)&&
    Array.isArray(recipe.tofGapReview.findingIds)&&recipe.tofGapReview.findingIds.length>0&&
    recipe.tofGapReview.findingIds.length<=8&&recipe.tofGapReview.findingIds.every(id=>/^gap-[a-f0-9]{24}$/.test(id)),
    'Supply source-bound TOF gap witnesses for the selected layer');
  assert(!recipe.branchPick||recipe.branchLayerId&&Array.isArray(recipe.branchPick)&&recipe.branchPick.length===2&&
    recipe.branchPick.every(value=>Number.isFinite(value)&&value>0&&value<1));
  assert(!recipe.isolateBranch||recipe.branchLayerId);
  assert(!recipe.compareVesselShading||Array.isArray(recipe.compareVesselShading)&&recipe.compareVesselShading.length<=5&&
    new Set(recipe.compareVesselShading).size===recipe.compareVesselShading.length&&
    recipe.compareVesselShading.every(mode=>['source','balanced','large','fine','cta-contrast'].includes(mode)));
  assert(!recipe.preset||['superior','anterior','left','oblique'].includes(recipe.preset));
  assert(!recipe.scope||['all','brain','intracranial','excluded','exterior-air'].includes(recipe.scope));
  assert(!recipe.rotationFrames||Number.isInteger(recipe.rotationFrames)&&recipe.rotationFrames>=2&&recipe.rotationFrames<=120);
  assert(!recipe.rotationFrames||!recipe.compareLayerIds,'Use separate recipes for simultaneous camera comparisons and rotation');
  if(recipe.interaction){
    checkedInteraction(recipe.interaction);
    assert(!recipe.rotationFrames&&!recipe.compareLayerIds,'Measure continuous interaction in its own recipe');
  }
}

const profile=fs.mkdtempSync(path.join(os.tmpdir(),'rslab-vessel-capture-'));
app.setPath('userData',profile);
app.commandLine.appendSwitch('force-device-scale-factor',String(deviceScale));
dialog.showOpenDialog=async()=>({canceled:false,filePaths:[source]});
const hash=file=>createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const report={source,snapshotId:document.snapshot_id,manifestSha256:hash(source),canonicalManifestSha256:hash(canonical),
  windowSize,deviceScale,cinematicSamples,views:[],errors:[],windowEverVisible:false,passed:false};
const processStarted=performance.now();
let memoryPoll;
const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const persist=()=>fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));
let activeRecipe=recipes[0],openCount=0,lastPayload,lastGapReport;
let finished=false,exitCode=1,quitDeadline;
function exitCapture(event){
  event?.preventDefault();
  clearTimeout(quitDeadline);
  fs.rmSync(profile,{recursive:true,force:true});
  app.exit(exitCode);
}
function finish(error){
  if(finished)return;
  finished=true;
  clearInterval(memoryPoll);
  if(error){report.error=error.stack??String(error);console.error(report.error);}
  report.manifestUnchanged=hash(source)===report.manifestSha256;
  report.canonicalManifestUnchanged=hash(canonical)===report.canonicalManifestSha256;
  report.wallMs=performance.now()-processStarted;
  report.passed=!error&&!report.windowEverVisible&&!report.errors.length&&report.manifestUnchanged&&report.canonicalManifestUnchanged&&report.views.length===recipes.length;
  persist();
  exitCode=report.passed?0:1;
  // The app's before-quit handler stops the local engine and clears transfer
  // files before Electron publishes will-quit. Keep this disposable hidden
  // instance from blocking a batch if Chromium never completes its quit.
  quitDeadline=setTimeout(()=>{
    report.shutdown='hidden capture exit deadline';
    persist();
    exitCapture();
  },20000);
  app.quit();
}
app.once('will-quit',exitCapture);
dialog.showErrorBox=(title,message)=>finish(Error(`${title} ${message}`));
process.on('uncaughtException',finish);
process.on('unhandledRejection',finish);
process.on('SIGTERM',()=>finish(Error('Capture interrupted')));

const handle=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(name,handler)=>handle(name,async(event,...args)=>{
  const result=await handler(event,...args);
  if(name==='research:anatomyView')lastPayload=result;
  if(name==='research:anatomyVesselGapReview')lastGapReport=result;
  if(!['research:startupWorkspace','research:openWorkspace'].includes(name)||!result?.document)return result;
  assert.equal(result.document.snapshot_id,document.snapshot_id);
  if(linked)assert.equal(result.readOnlyStudy,true);
  const {seriesId,scope='all',cursor=null,settings={},layers:visibleLayers}=activeRecipe;
  openCount++;
  if(activeRecipe.useDefaultState)return result;
  return {...result,viewState:{schema:'patient-anatomy-view-v1',patientId:document.patient_id,snapshotId:document.snapshot_id,
    seriesId,vesselDisplayVersion:2,vesselScope:scope,patientCutaway:activeRecipe.patientCutaway??{},states:{[seriesId]:{
      cursor,visible:Object.fromEntries([...layers].map(id=>[id,visibleLayers.includes(id)])),settings:{presentation:'vascular',layout:'3d-focus',
      skullMode:'transparent',smoothSurfaces:document.vascular_review?.schema!=='eden2020-restricted-vascular-companion-v1',
      showOrientationGuide:false,sliceOverlay:'contour',volumeRendering:{enabled:false},...settings}}}}};
});

app.on('browser-window-created',(_event,window)=>{
  window.on('show',()=>{report.windowEverVisible=true;window.hide();});
  const {screen}=require('electron');
  if(displayScale!==undefined){
    const display=screen.getAllDisplays().find(candidate=>candidate.scaleFactor===displayScale);
    assert(display,`No connected display has scale factor ${displayScale}`);
    window.setPosition(display.workArea.x+40,display.workArea.y+40);
  }
  const display=screen.getDisplayMatching(window.getBounds());
  report.display={scaleFactor:display.scaleFactor,refreshHz:display.displayFrequency,internal:display.internal,requestedScale:displayScale??null};
  window.webContents.on('console-message',(_event,details)=>{if(details.level==='error')report.errors.push(details.message);});
  window.webContents.once('did-finish-load',async()=>{
    const js=code=>window.webContents.executeJavaScript(code,true);
    const until=async(check,description)=>{
      const deadline=Date.now()+300000;
      while(Date.now()<deadline){
        if(await js(check))return;
        const error=await js(`document.querySelector('.rl-viewer-error,[data-layer-error=true]')?.textContent`);
        if(error)throw Error(error);
        await wait(150);
      }
      throw Error(`Timed out waiting for ${description}`);
    };
    const ready=()=>until(`document.querySelector('.rl-viewer')?.dataset.surfacesPending==='0'&&!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`,'native anatomy surfaces');
    const button=label=>`[...document.querySelectorAll('button')].find(e=>e.getAttribute('aria-label')===${JSON.stringify(label)}||e.textContent.trim()===${JSON.stringify(label)}||e.querySelector('span')?.textContent.trim()===${JSON.stringify(label)})`;
    const click=async label=>{
      await until(`!!(${button(label)})&&!(${button(label)}).disabled`,`enabled ${label} control`);
      await js(`(${button(label)}).click()`);
    };
    const submittedFrame=()=>js('window.vascularFrame??null');
    const settledFrame=async(after=0)=>{
      await until(`window.vascularFrame?.serial>${after}&&window.vascularFrame.interactive===false&&
        window.vascularFrame.scanQuality?.opticsPending!==true&&
        (window.vascularFrame.scanQuality?.mode!=='cinematic'||(window.vascularFrame.scanQuality.shadowSampling==='path-traced'?
          window.vascularFrame.scanQuality.sampleLimit===${cinematicSamples}&&window.vascularFrame.scanQuality.samples===${cinematicSamples}:
          window.vascularFrame.scanQuality.settled===true))&&
        window.vascularGpu?.completed===true&&window.vascularGpu.serial===window.vascularFrame.serial&&
        window.vascularGpu.displayIdentity===window.vascularFrame.displayIdentity`,'settled GPU-completed anatomy frame');
      return submittedFrame();
    };
    const capture=async(name,interactive=false)=>{
      assert(!window.isVisible());
      const frame=interactive?await submittedFrame():await settledFrame();
      const filename=path.join(output,name+'.png');
      let image=await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true});
      assert(!image.isEmpty());
      const backingSize=image.getSize(),referenceSize=expected?.views[0]?.size;
      // macOS can return a Retina backing bitmap even for a hidden window
      // with the same logical size and Chromium device scale as the reference.
      if(referenceSize&&backingSize.width!==referenceSize.width){
        const scale=backingSize.width/referenceSize.width;
        assert([.5,2].includes(scale)&&backingSize.height===referenceSize.height*scale,'Matched capture viewport changed');
        image=image.resize({...referenceSize,quality:'best'});
      }
      fs.writeFileSync(filename,image.toPNG());
      const size=image.getSize(),captureScale=size.width/window.getContentSize()[0];
      const telemetry=await js(`({fence:window.vascularGpu?.serial===${frame?.serial??0}?window.vascularGpu:null,
        gpu:window.vascularTimes.get(${frame?.serial??0})??null,
        settledSamples:(()=>{
          const last=window.vascularSubmitted.get(${frame?.serial??0});
          if(!last?.scanQuality||last.interactive||last.scanQuality.opticsPending)return [];
          const found=[];
          for(let sample=last.scanQuality.samples,serial=last.serial;sample>0;sample--,serial--){
            const row=window.vascularSubmitted.get(serial);
            if(!row||row.interactive||row.scanQuality?.opticsPending||row.scanQuality?.samples!==sample||
              row.scanQuality?.shadowSampling!==last.scanQuality.shadowSampling||row.displayIdentity!==last.displayIdentity)break;
            found.unshift({sample,serial,cpuMs:row.cpuMs,gpu:window.vascularTimes.get(serial)??null});
          }
          return found;
        })(),pane:(()=>{const box=document.querySelector('.rl-viewer-anatomy-image').getBoundingClientRect();
          return {x:Math.ceil(box.x),y:Math.ceil(box.y),width:Math.floor(box.width)-1,height:Math.floor(box.height)-1};})()})`);
      const pane=Object.fromEntries(Object.entries(telemetry.pane).map(([key,value])=>[key,Math.round(value*captureScale)]));
      return {filename,size,backingSize,captureScale,frame,...telemetry,pane};
    };
    const exportView=async name=>{
      await until(`!!(${button('Export 3D PNG')})&&!(${button('Export 3D PNG')}).disabled`,'ready 3D PNG export');
      const started=performance.now();
      const result=await js(`new Promise((resolve,reject)=>{
        const create=URL.createObjectURL, click=HTMLAnchorElement.prototype.click;
        let exported,stats;
        const restore=()=>{clearTimeout(deadline);observer.disconnect();
          document.removeEventListener('anatomy-export',receive);
          URL.createObjectURL=create;HTMLAnchorElement.prototype.click=click;};
        const finish=()=>{if(!exported||!stats)return;
          restore();
          const reader=new FileReader();reader.onerror=()=>reject(reader.error);
          reader.onload=()=>resolve({data:reader.result,stats});reader.readAsDataURL(exported);};
        const receive=e=>{stats=e.detail;finish();};
        const deadline=setTimeout(()=>{restore();reject(Error('Timed out waiting for the 3D PNG'));},300000);
        const observer=new MutationObserver(()=>{
          const status=document.querySelector('[aria-label="3D PNG export status"]')?.textContent;
          if(status&&status!=='Rendering 3D PNG…'&&status!=='3D PNG ready.'){
            restore();reject(Error(status));
          }
        });
        const urls=new Map();
        URL.createObjectURL=function(blob){const url=create.call(this,blob);urls.set(url,blob);return url;};
        HTMLAnchorElement.prototype.click=function(){
          if(this.download==='resectionlab-3d.png'&&urls.has(this.href)){exported=urls.get(this.href);finish();}
          else click.call(this);
        };
        document.addEventListener('anatomy-export',receive);
        observer.observe(document.querySelector('.rl-viewer'),{childList:true,subtree:true,characterData:true});
        (${button('Export 3D PNG')}).click();
      })`);
      const filename=path.join(output,`${name}-export.png`);
      assert(result.data.startsWith('data:image/png;base64,'));
      const bytes=Buffer.from(result.data.split(',')[1],'base64');
      fs.writeFileSync(filename,bytes);
      const size=nativeImage.createFromBuffer(bytes).getSize();
      assert.equal(Math.max(size.width,size.height),2048);
      const frame=await settledFrame();
      return {filename,size,wallMs:performance.now()-started,stats:result.stats,frame,sha256:hash(filename)};
    };
    const setLayer=async(id,visible)=>{
      const element=`[...document.querySelectorAll('.native-structure-row')].find(row=>row.dataset.structureId===${JSON.stringify(id)})?.querySelector('.native-structure-visibility')`;
      await until(`!!(${element})&&!(${element}).disabled`,`visibility control for ${id}`);
      const previous=(await submittedFrame())?.serial??0;
      const changed=await js(`(()=>{const b=${element};if((b.getAttribute('aria-pressed')==='true')===${visible})return false;b.click();return true;})()`);
      await ready();if(changed)await settledFrame(previous);
    };
    const nativeView=()=>js(`JSON.parse(document.querySelector('.native-workspace').dataset.nativeViewState)`);
    const neutralSectionComparison=async(baseline,candidate=baseline)=>{
      const box=await js(`(()=>{const r=document.querySelector('.rl-viewer-anatomy-image').getBoundingClientRect();
        return {x:Math.ceil(r.x),y:Math.ceil(r.y),width:Math.floor(r.width)-1,height:Math.floor(r.height)-1};})()`);
      const crop=filename=>{
        const image=nativeImage.createFromPath(filename),scale=image.getSize().width/window.getContentSize()[0];
        return image.crop(Object.fromEntries(Object.entries(box).map(([key,value])=>[key,Math.round(value*scale)]))).toBitmap();
      };
      const a=crop(baseline);
      const b=candidate===baseline?a:crop(candidate);
      assert(a.length>0&&a.length===b.length);
      let neutral=0,matched=0,totalDelta=0;
      for(let i=0;i<a.length;i+=4){
        const ra=a[i+2],rb=b[i+2];
        if(ra<16||rb<16||ra>245||rb>245||
          Math.max(a[i],a[i+1],ra)-Math.min(a[i],a[i+1],ra)>2||
          Math.max(b[i],b[i+1],rb)-Math.min(b[i],b[i+1],rb)>2)continue;
        const delta=Math.abs(ra-rb);
        neutral++;totalDelta+=delta;if(delta<=2)matched++;
      }
      // A slice outside the acquired slab can leave a neutral pixel in a
      // crosshair or label. Require actual source signal in the image pane.
      const minimumSignalPixels=Math.ceil(box.width*box.height*.005);
      assert(neutral>=minimumSignalPixels,
        `Source section has insufficient visible signal (${neutral}/${minimumSignalPixels} neutral pixels)`);
      assert(matched/neutral>=.95,
        `Matched source sections differ outside colored overlays (${matched}/${neutral} neutral pixels within two sRGB levels)`);
      return {neutralPixels:neutral,matchedWithinTwoLevels:matched,meanAbsoluteDifference:totalDelta/neutral};
    };
    try{
      window.setSize(windowSize.width,windowSize.height);
      if(expected?.views[0]?.size){
        const {width,height}=expected.views[0].size;
        const referenceCaptureScale=expected.views[0].captureScale??deviceScale;
        const contentSize=expected.contentSize??[width/referenceCaptureScale,height/referenceCaptureScale];
        window.setContentSize(...contentSize);
      }
      report.contentSize=window.getContentSize();
      if(expected?.contentSize)assert.deepEqual(report.contentSize,expected.contentSize,'Matched logical viewport changed');
      window.webContents.setBackgroundThrottling(false);
      report.memory={peakSampledElectronWorkingSetBytes:0,pollMs:500};
      memoryPoll=setInterval(()=>{
        const bytes=app.getAppMetrics().reduce((sum,process)=>sum+process.memory.workingSetSize*1024,0);
        report.memory.peakSampledElectronWorkingSetBytes=Math.max(report.memory.peakSampledElectronWorkingSetBytes,bytes);
      },report.memory.pollMs);
      // The page reads the still sample count when an accumulation starts. A
      // view may already be refining past it, so restart refinement (a
      // document scroll relayouts and invalidates the anatomy view).
      await js(`window.resectionCinematicStillSamples=${cinematicSamples};window.dispatchEvent(new Event('scroll'));
        window.vascularTimes=new Map();window.vascularSubmitted=new Map();document.addEventListener('anatomy-gpu-time',e=>{
        window.vascularTimes.set(e.detail.serial,e.detail);
        if(window.vascularTimes.size>12)window.vascularTimes.delete(window.vascularTimes.keys().next().value);
      });document.addEventListener('anatomy-frame',e=>{
        const {serial,at,interactive,surfaceQuality,scanQuality,cameraState,cursor,displayIdentity,sourceSignalFingerprint,
          cpuMs,gpuTimingSupported,estimatedAllocation,volume,sliceDraws,vesselSlabContext}=e.detail;
        window.vascularFrame={serial,at,interactive,surfaceQuality,scanQuality,cameraState,cursor,displayIdentity,sourceSignalFingerprint,
          cpuMs,gpuTimingSupported,estimatedAllocation,volume,sliceDraws,vesselSlabContext};
        window.vascularSubmitted.set(serial,window.vascularFrame);
        if(window.vascularSubmitted.size>12)window.vascularSubmitted.delete(window.vascularSubmitted.keys().next().value);
        if(window.vascularGesture&&at>=window.vascularGesture.start&&!window.vascularGesture.frame)
          window.vascularGesture.frame=window.vascularFrame;
      });document.addEventListener('anatomy-gpu',e=>{
        window.vascularGpu=e.detail;
        if(window.vascularGesture?.frame?.serial===e.detail.serial)window.vascularGesture.fence=e.detail;
      });`);
      if(expected){
        assert(expected.passed&&!expected.windowEverVisible,'Reference capture did not pass in a hidden window');
        for(const key of ['snapshotId','manifestSha256','canonicalManifestSha256'])assert.equal(report[key],expected[key],`Matched ${key} changed`);
        if(expected.windowSize)assert.deepEqual(report.windowSize,expected.windowSize);
      }
      for(const [index,recipe] of recipes.entries()){
        const openStarted=performance.now();
        activeRecipe=recipe;
        if(index){
          lastPayload=undefined;
          const previous=await js(`document.querySelector('.rl-viewer')?.dataset.sourceDisplayIdentity`);
          await click('Open');
          await until(`!!document.querySelector('.rl-viewer')?.dataset.sourceDisplayIdentity&&document.querySelector('.rl-viewer').dataset.sourceDisplayIdentity!==${JSON.stringify(previous)}`,'reopened native study');
        }
        await ready();
        // The viewer can look idle before its first anatomy request resolves,
        // and it cancels requests that a newer one supersedes.
        const loaded=()=>{
          const names=new Set(lastPayload?.compartments?.map(layer=>layer.name)),missing=new Set(lastPayload?.unavailableLayers);
          return names.size>0&&recipe.layers.every(id=>names.has(id)||missing.has(id));
        };
        for(const deadline=Date.now()+300000;!loaded()&&Date.now()<deadline;)await wait(150);
        // The renderer still has to install a payload the main process has answered.
        await until(`document.querySelector('.rl-viewer')?.dataset.rendererReady==='true'`,'installed native anatomy');
        await ready();
        assert.equal(openCount,index+1);
        if(linked)assert(await js(`document.querySelector('.save-button')?.disabled`),'Linked review must be read-only');
        if(recipe.settings?.operativeView==='surgeon'){
          await until(`document.querySelector('.rl-viewer-anatomy-image')?.dataset.operativeView==='surgeon'`,'surgeon presentation');
        }
        if(recipe.preset){
          await click('Layout');
          await js(inputScript('Camera preset',recipe.preset));
          await click('Close view controls');
        }
        if(recipe.compareLayerIds)await setLayer(recipe.compareLayerIds[1],false);
        await wait(450);
        let beforePreset;
        const presentation=recipe.applyVesselsPreset?'Vessels':recipe.applyScanPreset?'Scan':null;
        if(presentation){
          const previous=(await nativeView()).states[recipe.seriesId];
          beforePreset={view:previous,...await capture(`${recipe.name}-before-${presentation.toLowerCase()}`)};
          const preset=`[...document.querySelectorAll('[aria-label="Anatomy presentation"] button')].find(e=>e.textContent.trim()===${JSON.stringify(presentation)})`;
          await until(`!!(${preset})&&!(${preset}).disabled`,`enabled ordinary ${presentation} presentation`);
          await js(`(${preset}).click()`);
          await until(`(${preset}).getAttribute('aria-pressed')==='true'`,`ordinary ${presentation} presentation`);
          await settledFrame(beforePreset.frame.serial);
          await wait(450);
        }
        const view=await nativeView();
        const frame=await settledFrame();
        assert.equal(view.seriesId,recipe.seriesId);
        assert.equal(view.vesselScope,recipe.scope??'all');
        if(recipe.useDefaultState)assert.deepEqual(
          Object.keys(view.states[recipe.seriesId].visible).filter(id=>view.states[recipe.seriesId].visible[id]).sort(),
          [...recipe.layers].sort(),'Default visible anatomy differs from the study declaration');
        assert(frame?.cameraState,'Missing submitted anatomy camera');
        if(recipe.settings?.camera&&!recipe.preset){
          for(const key of ['position','target','up'])assert(frame.cameraState[key].every((n,i)=>Math.abs(n-recipe.settings.camera[key][i])<1e-3),`Camera ${key} drifted`);
        }
        if(recipe.cursor)assert(frame.cursor.every((n,i)=>Math.abs(n-recipe.cursor[i])<1e-3),'Source cursor drifted');
        assert(lastPayload?.compartments?.length,'Missing native anatomy payload');
        assert.deepEqual(lastPayload.unavailableLayers.filter(id=>recipe.layers.includes(id)),[]);
        const surfaces=lastPayload.compartments.map(({name,sourceSeries,alignment,surfaceDiagnostics,surface,displayFilter})=>({name,sourceSeries,alignment,surfaceDiagnostics,sourceSurfaceSha256:surface?.source?.sha256,empty:surface?.empty??false,displayFilter}));
        const record={name:recipe.name,seriesId:view.seriesId,sourceSha256:lastPayload.mri.sha256,scope:view.vesselScope,view:view.states[view.seriesId],surfaces,
          readyMs:performance.now()-openStarted,...await capture(recipe.name)};
        const assertSameGeometry=previous=>{
          for(const key of ['position','target','up'])
            assert(record.frame.cameraState[key].every((n,i)=>Math.abs(n-previous.frame.cameraState[key][i])<1e-9),`Presentation changed camera ${key}`);
          for(const key of ['cursor','displayIdentity','sourceSignalFingerprint'])
            assert.deepEqual(record.frame[key],previous.frame[key],`Presentation changed ${key}`);
        };
        if(beforePreset){
          assertSameGeometry(beforePreset);
          assert.deepEqual(record.view.visible,beforePreset.view.visible,`${presentation} preset changed visible anatomy`);
          record[presentation==='Vessels'?'beforeVesselsPreset':'beforeScanPreset']=beforePreset;
        }
        if(recipe.expectLabelTextureBytesPerVoxel!==undefined){
          const voxels=lastPayload.shape.reduce((a,b)=>a*b,1);
          assert.equal(record.frame.estimatedAllocation?.labelTextureBytes,voxels*recipe.expectLabelTextureBytesPerVoxel,
            'Label texture storage does not match the expected bytes per voxel');
          assert.equal(record.frame.volume,recipe.settings.volumeRendering.enabled,'Unexpected acquired volume state in live label check');
        }
        if(lastPayload.ctExterior){
          const {schema,offset,mask,signalSha256}=lastPayload.ctExterior;
          assert.equal(signalSha256,record.sourceSha256);
          record.ctExterior={schema,offset,shape:mask.shape,maskSha256:mask.sha256};
        }
        if(expected){
          const reference=expected.views.find(view=>view.name===recipe.name);
          assert(reference,`Missing matched view ${recipe.name}`);
          for(const key of ['seriesId','sourceSha256','scope','size'])assert.deepEqual(record[key],reference[key],`Matched ${key} changed`);
          if(reference.captureScale!==undefined)assert.deepEqual(record.pane,reference.pane,'Matched anatomy pane changed');
          for(const key of ['sourceSignalFingerprint','cursor'])assert.deepEqual(record.frame[key],reference.frame[key],`Matched ${key} changed`);
          for(const key of ['position','target','up'])assert(record.frame.cameraState[key].every((value,i)=>
            Math.abs(value-reference.frame.cameraState[key][i])<1e-9),`Matched camera ${key} changed`);
          assert.deepEqual(record.view.visible,reference.view.visible,'Matched layer visibility changed');
          if(recipe.expectLabelTextureBytesPerVoxel!==undefined)
            assert.deepEqual(record.surfaces.map(row=>[row.name,row.sourceSeries,row.sourceSurfaceSha256]),
              reference.surfaces.map(row=>[row.name,row.sourceSeries,row.sourceSurfaceSha256]),'Compact label texture changed source surface identity');
          for(const key of ['clipping','clipBox','obliqueClip','activePlane','showPlane'])
            assert.deepEqual(record.view.settings[key],reference.view.settings[key],`Matched ${key} changed`);
        }
        if(exportName===recipe.name&&!recipe.slabContext){
          record.export=await exportView(recipe.name);
          for(const key of ['position','target','up'])
            assert(record.export.frame.cameraState[key].every((n,i)=>Math.abs(n-record.frame.cameraState[key][i])<1e-9),
              `3D export changed live camera ${key}`);
          assert.deepEqual(record.export.frame.sourceSignalFingerprint,record.frame.sourceSignalFingerprint,'3D export changed acquired signal');
        }
        if(recipe.inspectSourceVolumeToggle){
          assert.equal(record.view.settings.sourcedVesselContext,true,'Mixed MRI Vessels did not use the sourced cortex');
          assert.equal(record.frame.volume,false,'Mixed MRI Vessels unexpectedly enabled the source volume');
          const inspect=async enabled=>{
            const before=await submittedFrame();
            await click(enabled?'Show scan volume with structures':'Show structures only');
            await settledFrame(before.serial);
            const image=await capture(`${recipe.name}-${enabled?'with-volume':'restore-context'}`);
            assert.equal(image.frame.volume,enabled);
            assertSameGeometry(image);
            const state=(await nativeView()).states[view.seriesId];
            assert.equal(state.settings.sourcedVesselContext,true);
            assert.deepEqual(state.visible,record.view.visible,'Volume toggle changed visible anatomy');
            return image;
          };
          record.sourceVolumeToggle={enabled:await inspect(true),restored:await inspect(false)};
        }
        if(recipe.inspectCtExteriorToggle){
          assert(record.ctExterior,'Missing source-bound detached CT support');
          const toggle=`[...document.querySelectorAll('label')].find(label=>label.textContent.trim()==='Show detached CT signal')?.querySelector('input[type=checkbox]')`;
          const inspect=async(show)=>{
            const before=await submittedFrame();
            await click('Appearance');
            await until(`!!(${toggle})&&!(${toggle}).disabled`,'detached CT signal control');
            await js(`(${toggle}).click()`);
            await click('Close view controls');
            await settledFrame(before.serial);
            const checkedView=(await nativeView()).states[view.seriesId];
            assert.equal(checkedView.settings.volumeRendering.showCtExteriorSignal,show);
            const shown=await capture(`${recipe.name}-${show?'show-detached':'restore-default'}`);
            for(const key of ['cameraState','cursor','displayIdentity','sourceSignalFingerprint'])
              assert.deepEqual(shown.frame[key],record.frame[key],`Detached CT control changed ${key}`);
            return shown;
          };
          record.ctExteriorToggle={show:await inspect(true),restore:await inspect(false)};
        }
        if(recipe.inspectCtBoneOverlay){
          assert.equal(record.view.settings.presentation,'scan');
          assert.equal(record.view.settings.volumeRendering.showCtBoneOverlay,false);
          const toggle=`[...document.querySelectorAll('label')].find(label=>label.textContent.trim()==='Show 3D bone overlay')?.querySelector('input[type=checkbox]')`;
          const inspect=async show=>{
            const before=await submittedFrame();
            await click('Appearance');
            await until(`!!(${toggle})&&!(${toggle}).disabled`,'available 3D bone overlay control');
            assert.equal(await js(`(${toggle}).checked`),!show);
            await js(`(${toggle}).click()`);
            await click('Close view controls');
            await settledFrame(before.serial);
            const state=(await nativeView()).states[view.seriesId];
            assert.deepEqual(state.visible,record.view.visible,'Bone overlay changed linked slice layer visibility');
            assert.deepEqual(state.settings,{...record.view.settings,
              volumeRendering:{...record.view.settings.volumeRendering,showCtBoneOverlay:show}});
            const image=await capture(`${recipe.name}-${show?'show-bone-overlay':'restore-acquired-bone'}`);
            assertSameGeometry(image);
            assert.equal(image.frame.volume,true);
            return image;
          };
          record.ctBoneOverlay={enabled:await inspect(true),restored:await inspect(false)};
        }
        if(recipe.inspectPublisherSmoothing){
          const layers=record.surfaces.filter(layer=>recipe.layers.includes(layer.name)&&layer.surfaceDiagnostics?.publisherSourceSha256);
          assert(layers.length&&layers.every(layer=>layer.surfaceDiagnostics.policy==='source-bound-publisher-cta-fairing-v1'));
          assert(layers.some(layer=>layer.surfaceDiagnostics.maxDisplacementMm>0));
          const toggle=`document.querySelector('input[aria-label="Smooth 3D surfaces"]')`;
          const inspect=async enabled=>{
            const before=await submittedFrame();
            await click('Appearance');
            await until(`!!(${toggle})&&!(${toggle}).disabled`,'ordinary surface smoothing');
            assert.equal(await js(`(${toggle}).checked`),!enabled);
            await js(`(${toggle}).click()`);
            await click('Close view controls');
            await settledFrame(before.serial);
            const state=(await nativeView()).states[view.seriesId];
            assert.deepEqual(state.visible,record.view.visible);
            assert.deepEqual(state.settings,{...record.view.settings,smoothSurfaces:enabled});
            const image=await capture(`${recipe.name}-${enabled?'restore-conditioned':'publisher-source'}`);
            assertSameGeometry(image);
            return image;
          };
          record.publisherSmoothing={source:await inspect(false),restored:await inspect(true)};
        }
        if(recipe.inspectCtaCoverage){
          const declared=document.display.cta_arterial_coverage;
          const coverage=Array.isArray(declared)?declared.find(row=>row.series_id===view.seriesId&&!row.retired):declared;
          const initial=record.view.visible;
          assert(initial[coverage.layer_id]&&coverage.focus_layers.every(id=>!initial[id]));
          const inspect=async combined=>{
            const name=combined?'Combined arteries':'Focused branches';
            const before=await submittedFrame();
            await click(name);
            await until(`(${button(name)})?.getAttribute('aria-pressed')==='true'`,name);
            await ready();
            await settledFrame(before.serial);
            const state=await nativeView(),activeView=state.states[view.seriesId];
            const visible=combined?initial:{...initial,[coverage.layer_id]:false,
              ...Object.fromEntries(coverage.focus_layers.map(id=>[id,true]))};
            assert.deepEqual(activeView.visible,visible);
            assert.deepEqual(activeView.settings,record.view.settings);
            assert.deepEqual(state.patientCutaway,view.patientCutaway);
            const image=await capture(`${recipe.name}-${combined?'combined-restored':'focused'}`);
            assertSameGeometry(image);
            return image;
          };
          record.ctaCoverage={focused:await inspect(false),combined:await inspect(true)};
        }
        if(recipe.inspectCtaFullHead){
          const opening=view.patientCutaway?.clipping?.enabled?view.patientCutaway.clipping:record.view.settings.clipping;
          assert(opening?.enabled&&opening.axis==='z'&&opening.flipped,'Expected an ordinary source-linked CTA cut');
          let returned;
          let beforeFull=record.view;
          if(view.patientCutaway?.clipping?.enabled){
            const preset=async name=>{
              const before=await submittedFrame();
              const button=`[...document.querySelectorAll('[aria-label="Anatomy presentation"] button')].find(e=>e.textContent.trim()===${JSON.stringify(name)})`;
              await until(`!!(${button})&&!(${button}).disabled`,`CTA ${name} presentation`);
              await js(`(${button}).click()`);
              await until(`(${button}).getAttribute('aria-pressed')==='true'`,`CTA ${name} presentation active`);
              await settledFrame(before.serial);
              return nativeView();
            };
            const scan=await preset('Scan');
            assert.equal(scan.states[view.seriesId].settings.presentation,'scan');
            assert.deepEqual(scan.patientCutaway.clipping,opening,'Scan changed the shared 3D cutaway');
            const returnedView=await preset('Structures');
            assert.deepEqual(returnedView.patientCutaway.clipping,opening);
            assert.deepEqual(returnedView.states[view.seriesId].visible,record.view.visible);
            beforeFull=returnedView.states[view.seriesId];
            returned=await capture(`${recipe.name}-structures-return`);
            assertSameGeometry(returned);
          }
          const before=await submittedFrame();
          await click('Show full CTA head');
          await settledFrame(before.serial);
          const current=await nativeView(),state=current.states[view.seriesId];
          assert(!current.patientCutaway?.clipping?.enabled&&!state.settings.clipping?.enabled);
          assert.deepEqual(state.visible,record.view.visible);
          assert.deepEqual(state.settings.volumeRendering,beforeFull.settings.volumeRendering);
          assert.deepEqual(state.cursor,record.view.cursor);
          const full=await capture(`${recipe.name}-full-head`);
          assert.equal(full.frame.displayIdentity,record.frame.displayIdentity);
          assert.equal(full.frame.sourceSignalFingerprint,record.frame.sourceSignalFingerprint);
          record.ctaOpening={clipping:opening,returned,fullHead:full,fullHeadState:current};
        }
        if(recipe.inspectBalancedTof){
          const calibrated=record.frame.scanQuality?.tofContextDensity;
          assert(Number.isFinite(calibrated),'Missing calibrated TOF density');
          const toggle=`[...document.querySelectorAll('label')].find(label=>label.textContent.trim()==='Balance TOF context')?.querySelector('input[type=checkbox]')`;
          const inspect=async(enabled,label)=>{
            const before=await submittedFrame();
            await click('Appearance');
            await until(`!!(${toggle})&&!(${toggle}).disabled`,'available TOF balance control');
            await js(`(${toggle}).click()`);
            await click('Close view controls');
            await settledFrame(before.serial);
            const state=(await nativeView()).states[view.seriesId];
            assert.equal(state.settings.volumeRendering.balanceTofContext,enabled);
            assert.deepEqual(state.visible,record.view.visible);
            const image=await capture(`${recipe.name}-${label}`);
            assertSameGeometry(image);
            assert.equal(image.frame.scanQuality?.tofContextDensity,enabled?calibrated:undefined);
            return image;
          };
          record.tofContextToggle={manual:await inspect(false,'manual'),restored:await inspect(true,'restored')};
        }
        if(recipe.depthSlab){
          const captures=[];
          const captureSlab=async(suffix,previous)=>{
            const result=await capture(`${recipe.name}-${suffix}`);
            assert(result.frame.serial>previous.serial,'Depth slab frame did not advance');
            for(const key of ['position','target','up'])
              assert(result.frame.cameraState[key].every((n,i)=>Math.abs(n-frame.cameraState[key][i])<1e-9),'Depth slab moved the camera');
            for(const key of ['cursor','displayIdentity','sourceSignalFingerprint'])
              assert.deepEqual(result.frame[key],frame[key],`Depth slab changed ${key}`);
            return result;
          };
          let before=await submittedFrame();
          const slabButton=button('Vessel slab at cursor');
          const contextLabel='Faint vessel context outside slab';
          const toggleContext=async(enabled,suffix)=>{
            const prior=await submittedFrame();
            await click(contextLabel);
            await until(`(${button(contextLabel)})?.getAttribute('aria-pressed')==='${enabled}'`,'faint context control state');
            const updated=await settledFrame(prior.serial);
            assert.equal(updated.vesselSlabContext,enabled);
            assert.equal(updated.sliceDraws,0,'Context toggle invalidated a linked slice');
            return captureSlab(suffix,prior);
          };
          await until(`!!(${slabButton})?.closest('.rl-primary-tools')`,'vessel slab in the 3D toolbar');
          await click('Vessel slab at cursor');
          await until(`(${slabButton})?.getAttribute('aria-pressed')==='true'`,'active vessel slab');
          await click('Clip');await js(inputScript('Slab thickness',44));await click('Close view controls');
          await settledFrame(before.serial);
          captures.push({millimeters:44,...await captureSlab('slab-44',before)});
          if(recipe.slabContext){
            captures[0].context=await toggleContext(true,'slab-44-context');
            if(exportName===recipe.name)record.export=await exportView(`${recipe.name}-slab-44-context`);
            captures[0].contextOff=await toggleContext(false,'slab-44-context-off');
          }
          before=await submittedFrame();
          await click('Clip');await js(inputScript('Slab thickness',28));await click('Close view controls');
          await settledFrame(before.serial);
          captures.push({millimeters:28,...await captureSlab('slab-28',before)});
          if(recipe.slabContext){
            captures[1].context=await toggleContext(true,'slab-28-context');
            captures[1].contextOff=await toggleContext(false,'slab-28-context-off');
          }
          before=await submittedFrame();
          await click('Exit vessel slab in 3D');await settledFrame(before.serial);
          assert.equal(await js(`(${slabButton}).getAttribute('aria-pressed')`),'false');
          record.depthSlab={captures,restored:await captureSlab('slab-exit',before)};
          assert.deepEqual((await nativeView()).states[view.seriesId].settings,record.view.settings,'Depth slab changed the saved view');
        }
        if(recipe.compareVesselShading?.length){
          const startingMode=record.view.settings.vesselShading??(record.view.settings.presentation==='vascular'?'balanced':'source');
          const submitted=async mode=>{
            await until(enabledControlScript('Vessel shading'),'enabled vessel shading control');
            const prior=await js('window.vascularFrame.serial');
            await js(inputScript('Vessel shading',mode));
            await settledFrame(prior);
            const state=(await nativeView()).states[view.seriesId];
            assert.deepEqual(state.visible,record.view.visible);
            assert.deepEqual(state.settings,{...record.view.settings,vesselShading:mode});
            const image=await capture(`${recipe.name}-${mode}`);
            assertSameGeometry(image);
            return image;
          };
          record.shading=[];
          for(const mode of recipe.compareVesselShading){
            if(mode===startingMode)continue;
            record.shading.push({mode,...await submitted(mode)});
          }
          if(record.shading.length)record.shadingRestored={mode:startingMode,...await submitted(startingMode)};
        }
        const captureSection=async()=>{
          const prior=(await submittedFrame()).serial;
          await click('Clip');await click('Section at cursor');await click('Close view controls');
          await settledFrame(prior);
          const sectionWorkspace=await nativeView();
          const sectionView=sectionWorkspace.states[recipe.seriesId];
          assert(sectionView.settings.clipping?.enabled||sectionWorkspace.patientCutaway?.clipping?.enabled,'Source section did not apply a display clip');
          const image=await capture(`${recipe.name}-source-section`);
          const signal=await neutralSectionComparison(image.filename);
          return {view:sectionView,patientCutaway:sectionWorkspace.patientCutaway,...image,sourceSignalNeutralPixels:signal.neutralPixels};
        };
        if(recipe.sourceSection&&!recipe.compareLayerIds){
          record.section=await captureSection();
          if(recipe.inspectBalancedTof)assert.equal(record.section.frame.scanQuality?.tofContextDensity,undefined,
            'The physical source section must use its ordinary volume density');
        }
        if(recipe.contextVolume){
          await click('Show scan volume with structures');
          await wait(650);
          record.context=await capture(`${recipe.name}-tof-context`);
        }
        if(recipe.rotationFrames){
          const dir=path.join(output,`${recipe.name}-rotation`);fs.mkdirSync(dir);
          // OrbitControls requires pointer capture, which a synthetic pointer
          // cannot obtain. Contain the override to the hidden gesture target.
          const {x,y}=await js(`(()=>{const e=document.querySelector('.rl-viewer-anatomy-image'),r=e.getBoundingClientRect();e.setPointerCapture=()=>{};e.releasePointerCapture=()=>{};return {x:Math.round(r.x+r.width*.42),y:Math.round(r.y+r.height*.48)}})()`);
          const pointer=(type,x,y)=>js(`document.querySelector('.rl-viewer-anatomy-image').dispatchEvent(new PointerEvent(${JSON.stringify(type)},{bubbles:true,clientX:${x},clientY:${y},button:0,buttons:${type==='pointerup'?0:1},pointerId:91,pointerType:'mouse',isPrimary:true}))`);
          const angles=[],response=[];
          for(let i=0;i<recipe.rotationFrames;i++){
            await pointer('pointerdown',x,y);
            await js('window.vascularGesture={start:performance.now()};');
            await pointer('pointermove',x+5,y-2);
            await pointer('pointerup',x+5,y-2);
            await until('window.vascularGesture?.fence?.completed===true','GPU-completed orbit response');
            const gesture=await js('window.vascularGesture');
            const toFrameStart=gesture.frame.at-gesture.start;
            response.push({interactive:gesture.frame.interactive,toSubmissionMs:toFrameStart+gesture.frame.cpuMs,
              toFencePollMs:toFrameStart+gesture.fence.completionPollMs});
            await wait(130);
            await capture(path.join(`${recipe.name}-rotation`,String(i).padStart(4,'0')),true);
            angles.push(await js('window.vascularFrame.cameraState.position'));
          }
          await js(`(()=>{const e=document.querySelector('.rl-viewer-anatomy-image');delete e.setPointerCapture;delete e.releasePointerCapture;delete window.vascularGesture;})()`);
          assert(Math.hypot(...angles.at(-1).map((n,i)=>n-angles[0][i]))>1,'Orbit did not change the submitted camera path');
          record.rotationFrames=recipe.rotationFrames;
          record.rotationPositions=angles;
          record.rotationResponse={measurement:'Synthetic orbit input to matching GPU fence poll',samples:response};
        }
        if(recipe.interaction){
          const samples=await js(interactionScript(recipe.interaction));
          record.interaction={...summarizeInteraction(samples),samples};
          record.afterInteraction=await capture(`${recipe.name}-after-interaction`);
        }
        if(recipe.compareLayerIds){
          await setLayer(recipe.compareLayerIds[0],false);
          await setLayer(recipe.compareLayerIds[1],true);
          const candidateFrame=await js('window.vascularFrame');
          const comparisonCamera=await js(`JSON.parse(document.querySelector('.native-workspace').dataset.nativeViewState).states[${JSON.stringify(recipe.seriesId)}].settings.camera`);
          for(const key of ['position','target','up'])assert(candidateFrame.cameraState[key].every((n,i)=>Math.abs(n-frame.cameraState[key][i])<1e-4),'Comparison camera moved');
          record.comparison={layer:recipe.compareLayerIds[1],camera:comparisonCamera,...await capture(`${recipe.name}-comparison`)};
          if(recipe.sourceSection){
            record.section=await captureSection();
            await setLayer(recipe.compareLayerIds[1],false);
            await setLayer(recipe.compareLayerIds[0],true);
            const baselineWorkspace=await nativeView();
            const baselineView=baselineWorkspace.states[recipe.seriesId],candidateView=record.section.view;
            for(const key of ['activePlane','clipping','contrast','showPlane','sliceFraming','sliceOverlay','volumeRendering'])
              assert.deepEqual(baselineView.settings[key],candidateView.settings[key],`Matched source section setting changed: ${key}`);
            assert.deepEqual(baselineWorkspace.patientCutaway,record.section.patientCutaway,'Matched patient cutaway changed');
            assert.deepEqual(baselineView.cursor,candidateView.cursor,'Matched source section cursor changed');
            record.baselineSection={view:baselineView,...await capture(`${recipe.name}-source-section-baseline`)};
            assert.equal(record.baselineSection.frame.sourceSignalFingerprint,record.section.frame.sourceSignalFingerprint,'Matched source signal changed');
            for(const key of ['position','target','up'])
              assert(record.baselineSection.frame.cameraState[key].every((n,i)=>Math.abs(n-record.section.frame.cameraState[key][i])<1e-9),'Matched source section camera changed');
            record.section.sourceNeutralComparison=await neutralSectionComparison(record.baselineSection.filename,record.section.filename);
          }
        }
        if(recipe.conditionedReference){
          const previous=frame.displayIdentity;
          await js(inputScript('Native surface representation','conditioned'));
          await until(`!!document.querySelector('.rl-viewer')?.dataset.sourceDisplayIdentity&&document.querySelector('.rl-viewer').dataset.sourceDisplayIdentity!==${JSON.stringify(previous)}`,'conditioned native surface');
          await ready();await wait(350);
          assert(lastPayload.compartments.some(layer=>layer.surfaceDiagnostics?.maxDisplacementMm>0&&!layer.surfaceDiagnostics?.sourceSurfaceOnly),'No conditioned source perturbation was reported');
          record.conditioned=await capture(`${recipe.name}-conditioned`);
        }
        if(recipe.branchLayerId){
          const layerControl=`[...document.querySelectorAll('.native-structure-row')].find(row=>row.dataset.structureId===${JSON.stringify(recipe.branchLayerId)})?.querySelector('.native-structure-name')`;
          await until(`!!(${layerControl})&&!(${layerControl}).disabled`,'vascular layer inspector');
          if(recipe.branchPick){
            const [x,y]=recipe.branchPick;
            await js(`(()=>{const pane=document.querySelector('.rl-viewer-anatomy-image'),box=pane.getBoundingClientRect();
              pane.dispatchEvent(new MouseEvent('dblclick',{bubbles:true,clientX:box.left+box.width*${x},
                clientY:box.top+box.height*${y},button:0,detail:2}));})()`);
          }else{
            await js(`(${layerControl}).click()`);
            await until(`!!document.querySelector('[data-vessel-branches]')`,'source branch inspection panel');
            await js(`(()=>{const panel=document.querySelector('[data-vessel-branches]');panel.open=true;panel.scrollIntoView({block:'center'});})()`);
            await click('Inspect branches at cursor');
          }
          await until(`Number(document.querySelector('[data-vessel-branches]')?.dataset.vesselBranches)>0`,'source-bound branch query');
          if(recipe.branchPick)await until(`!!(${button('Hide branch trace')})`,'picked source branch trace');
          else await click('Show branch trace');
          await wait(400);
          const branchId=await js(`document.querySelector('[data-vessel-branches] code')?.title`);
          assert(branchId);
          record.branch={branchId,...await capture(`${recipe.name}-branch`)};
          if(recipe.isolateBranch){
            await click('Isolate branch area');
            await until(`document.querySelector('[data-vessel-branches]')?.dataset.vesselIsolation==='true'&&
              window.vascularFrame?.serial>${record.branch.frame.serial}`,'isolated branch display');
            await wait(250);
            const focused=await js('window.vascularFrame');
            assert.equal(focused.displayIdentity,record.branch.frame.displayIdentity,'Branch focus changed source evidence');
            for(const key of ['position','target','up'])
              assert(focused.cameraState[key].every((n,i)=>Math.abs(n-record.branch.frame.cameraState[key][i])<1e-9),'Branch focus moved the camera');
            record.branch.isolation=await capture(`${recipe.name}-branch-isolated`);
            await click('Show surrounding vessels');
            await until(`document.querySelector('[data-vessel-branches]')?.dataset.vesselIsolation==='false'&&
              window.vascularFrame?.serial>${focused.serial}`,'restored context');
            record.branch.restored=await capture(`${recipe.name}-branch-restored`);
          }
        }
        if(recipe.tofGapReview){
          await click('Anatomy');
          const layerId=JSON.stringify(recipe.tofGapReview.layerId);
          const control=`[...document.querySelectorAll('.study-inspector .native-structure-row')].find(row=>row.dataset.structureId===${layerId})?.querySelector('.native-structure-name')`;
          await until(`!!(${control})&&!(${control}).disabled`,'TOF layer inspector');
          await js(`(${control}).click()`);
          const gapPanel=`[...document.querySelectorAll('.layer-inspector details')].find(panel=>panel.querySelector('summary')?.textContent==='TOF gap review')`;
          await until(`!!(${gapPanel})`,'TOF gap review panel');
          await js(`(${gapPanel}).open=true`);
          const requested=performance.now();
          await click('Find nearby branch endpoints');
          const select=`document.querySelector('select[aria-label="TOF gap finding"]')`;
          await until(`!!(${select})&&!(${select}).disabled`,'native TOF gap findings');
          const options=await js(`Array.from((${select}).options,option=>({id:option.value,label:option.textContent}))`);
          const mounted=performance.now()-requested;
          const evidence=[];
          const canvasBoxes=()=>js(`Array.from(document.querySelectorAll('[data-tof-gap-review] canvas'),canvas=>{
            const box=canvas.getBoundingClientRect();return [box.x,box.y,box.width,box.height];})`);
          const referenceWindow=lastGapReport?.preview?.window;
          assert(Array.isArray(referenceWindow)&&referenceWindow.length===2);
          for(const [index,id] of recipe.tofGapReview.findingIds.entries()){
            assert(options.some(option=>option.id===id),`Missing TOF gap ${id}`);
            await js(inputScript('TOF gap finding',id));
            await until(`(${select}).value===${JSON.stringify(id)}&&!(${select}).disabled`,'selected TOF gap sections');
            const panel=await js(`(()=>{const node=document.querySelector('[data-tof-gap-review]');
              node.scrollIntoView({block:'start'});return {status:node.dataset.tofGapReview,text:node.innerText,
                captions:Array.from(node.querySelectorAll('figcaption'),el=>el.textContent)};})()`);
            assert.equal(panel.captions.length,3);
            assert.equal(lastGapReport.predictedVoxelsAdded,0);
            assert.equal(lastGapReport.signalSha256,scans.get(recipe.seriesId).data.sha256);
            assert.equal(lastGapReport.preview.findingId,id);
            assert.deepEqual(lastGapReport.preview.window,referenceWindow,'Findings must share a single TOF window');
            const boxes=await canvasBoxes();
            const captureId=`${recipe.name}-gap-${index+1}`;
            const annotated=await capture(captureId);
            await js(`Array.from(document.querySelectorAll('[data-tof-gap-review] label')).find(label=>label.textContent.trim()==='Show gap overlay').querySelector('input').click()`);
            const raw=await capture(`${captureId}-source`);
            assert.deepEqual(await canvasBoxes(),boxes,'The overlay toggle moved the TOF sections');
            assertSameGeometry(annotated);assertSameGeometry(raw);
            await js(`Array.from(document.querySelectorAll('[data-tof-gap-review] label')).find(label=>label.textContent.trim()==='Show gap overlay').querySelector('input').click()`);
            evidence.push({id,panel,boxes,annotated,raw});
          }
          const checked=(await nativeView()).states[view.seriesId];
          assert.deepEqual(checked.visible,record.view.visible);
          assert.deepEqual(checked.settings,record.view.settings);
          const previous=await submittedFrame();
          await click('Center linked slices');
          const focused=await settledFrame(previous.serial);
          assert.equal(focused.sourceSignalFingerprint,record.frame.sourceSignalFingerprint);
          for(const key of ['position','target','up'])assert(focused.cameraState[key].every((value,i)=>
            Math.abs(value-record.frame.cameraState[key][i])<1e-9),`Centering TOF slices moved the camera ${key}`);
          assert.notDeepEqual(focused.cursor,record.frame.cursor);
          record.tofGapReview={mountedMs:mounted,options,evidence,window:referenceWindow,focusedCursor:focused.cursor};
        }
        report.views.push(record);persist();console.log(`Captured ${record.filename}`);
      }
      finish();
    }catch(error){
      report.visibleText=await js('document.body.innerText').catch(()=>null);
      try{await capture('failure',true);}catch{}
      finish(error);
    }
  });
});
require('./main.cjs');
