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

function argument(name){
  const index=process.argv.indexOf(name);
  assert(index>=0&&process.argv[index+1],`Pass ${name}`);
  return path.resolve(process.argv[index+1]);
}
const source=fs.realpathSync(argument('--anatomy'));
const linked=/^vascular-review(?:\.[0-9a-f]{12})?\.anatomy\.json$/.test(path.basename(source));
assert(linked||path.basename(source)==='patient-anatomy.json');
const canonical=linked?path.join(path.dirname(source),'patient-anatomy.json'):source;
const output=argument('--output');
fs.mkdirSync(output);
const document=JSON.parse(fs.readFileSync(source,'utf8'));
const recipes=JSON.parse(fs.readFileSync(argument('--views'),'utf8'));
assert(Array.isArray(recipes)&&recipes.length&&recipes.length<=32);
assert.equal(new Set(recipes.map(r=>r.name)).size,recipes.length);
const scans=new Map(document.series.map(scan=>[scan.id,scan]));
const layers=new Set([...document.layers,...document.tractography??[]].map(layer=>layer.id));
for(const recipe of recipes){
  assert(/^[A-Za-z0-9_-]{1,72}$/.test(recipe.name));
  assert(scans.has(recipe.seriesId),`Unknown scan ${recipe.seriesId}`);
  assert(recipe.layers.length&&recipe.layers.every(id=>layers.has(id)),`Unknown layer in ${recipe.name}`);
  assert(!recipe.compareLayerIds||recipe.compareLayerIds.length===2&&recipe.compareLayerIds.every(id=>recipe.layers.includes(id)));
  assert(!recipe.branchLayerId||recipe.layers.includes(recipe.branchLayerId));
  assert(!recipe.branchPick||recipe.branchLayerId&&Array.isArray(recipe.branchPick)&&recipe.branchPick.length===2&&
    recipe.branchPick.every(value=>Number.isFinite(value)&&value>0&&value<1));
  assert(!recipe.isolateBranch||recipe.branchLayerId);
  assert(!recipe.compareVesselShading||Array.isArray(recipe.compareVesselShading)&&recipe.compareVesselShading.length<=4&&
    new Set(recipe.compareVesselShading).size===recipe.compareVesselShading.length&&
    recipe.compareVesselShading.every(mode=>['source','balanced','large','fine'].includes(mode)));
  assert(!recipe.preset||['superior','anterior','left','oblique'].includes(recipe.preset));
  assert(!recipe.scope||['all','brain','intracranial','excluded'].includes(recipe.scope));
  assert(!recipe.rotationFrames||Number.isInteger(recipe.rotationFrames)&&recipe.rotationFrames>=2&&recipe.rotationFrames<=120);
  assert(!recipe.rotationFrames||!recipe.compareLayerIds,'Use separate recipes for simultaneous camera comparisons and rotation');
}

const profile=fs.mkdtempSync(path.join(os.tmpdir(),'rslab-vessel-capture-'));
app.setPath('userData',profile);
process.argv.push('--automation-hidden');
if(process.platform==='darwin')app.setActivationPolicy('accessory');
dialog.showOpenDialog=async()=>({canceled:false,filePaths:[source]});
const hash=file=>createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const report={source,snapshotId:document.snapshot_id,manifestSha256:hash(source),canonicalManifestSha256:hash(canonical),
  views:[],errors:[],windowEverVisible:false,passed:false};
const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const persist=()=>fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));
let activeRecipe=recipes[0],openCount=0,lastPayload;
let finished=false,exitCode=1;
function finish(error){
  if(finished)return;
  finished=true;
  if(error){report.error=error.stack??String(error);console.error(report.error);}
  report.manifestUnchanged=hash(source)===report.manifestSha256;
  report.canonicalManifestUnchanged=hash(canonical)===report.canonicalManifestSha256;
  report.passed=!error&&!report.windowEverVisible&&!report.errors.length&&report.manifestUnchanged&&report.canonicalManifestUnchanged&&report.views.length===recipes.length;
  persist();
  exitCode=report.passed?0:1;
  // The app's before-quit handler stops the local engine and clears transfer
  // files before Electron publishes will-quit.
  app.quit();
}
app.once('will-quit',event=>{
  event.preventDefault();
  fs.rmSync(profile,{recursive:true,force:true});
  app.exit(exitCode);
});
dialog.showErrorBox=(title,message)=>finish(Error(`${title} ${message}`));
process.on('uncaughtException',finish);
process.on('unhandledRejection',finish);
process.on('SIGTERM',()=>finish(Error('Capture interrupted')));

const handle=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(name,handler)=>handle(name,async(event,...args)=>{
  const result=await handler(event,...args);
  if(name==='research:anatomyView')lastPayload=result;
  if(!['research:startupWorkspace','research:openWorkspace'].includes(name)||!result?.document)return result;
  assert.equal(result.document.snapshot_id,document.snapshot_id);
  if(linked)assert.equal(result.readOnlyStudy,true);
  const {seriesId,scope='all',cursor=null,settings={},layers:visibleLayers}=activeRecipe;
  openCount++;
  return {...result,viewState:{schema:'patient-anatomy-view-v1',patientId:document.patient_id,snapshotId:document.snapshot_id,
    seriesId,vesselDisplayVersion:2,vesselScope:scope,patientCutaway:activeRecipe.patientCutaway??{},states:{[seriesId]:{
      cursor,visible:Object.fromEntries([...layers].map(id=>[id,visibleLayers.includes(id)])),settings:{presentation:'vascular',layout:'3d-focus',
      skullMode:'transparent',smoothSurfaces:document.vascular_review?.schema!=='eden2020-restricted-vascular-companion-v1',
      showOrientationGuide:false,sliceOverlay:'contour',volumeRendering:{enabled:false},...settings}}}}};
});

app.on('browser-window-created',(_event,window)=>{
  window.on('show',()=>{report.windowEverVisible=true;window.hide();});
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
        window.vascularGpu?.completed===true&&window.vascularGpu.serial===window.vascularFrame.serial&&
        window.vascularGpu.displayIdentity===window.vascularFrame.displayIdentity`,'settled GPU-completed anatomy frame');
      return submittedFrame();
    };
    const capture=async(name,interactive=false)=>{
      assert(!window.isVisible());
      const frame=interactive?await submittedFrame():await settledFrame();
      const filename=path.join(output,name+'.png');
      const image=await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true});
      assert(!image.isEmpty());
      fs.writeFileSync(filename,image.toPNG());
      return {filename,size:image.getSize(),frame};
    };
    const setLayer=async(id,visible)=>{
      const element=`[...document.querySelectorAll('.native-structure-row')].find(row=>row.dataset.structureId===${JSON.stringify(id)})?.querySelector('.native-structure-visibility')`;
      await until(`!!(${element})&&!(${element}).disabled`,`visibility control for ${id}`);
      const previous=(await submittedFrame())?.serial??0;
      const changed=await js(`(()=>{const b=${element};if((b.getAttribute('aria-pressed')==='true')===${visible})return false;b.click();return true;})()`);
      await ready();if(changed)await settledFrame(previous);
    };
    const nativeView=()=>js(`JSON.parse(document.querySelector('.native-workspace').dataset.nativeViewState)`);
    const neutralSectionComparison=async(baseline,candidate)=>{
      const box=await js(`(()=>{const r=document.querySelector('.rl-viewer-anatomy-image').getBoundingClientRect();
        return {x:Math.ceil(r.x),y:Math.ceil(r.y),width:Math.floor(r.width)-1,height:Math.floor(r.height)-1};})()`);
      const a=nativeImage.createFromPath(baseline).crop(box).toBitmap();
      const b=nativeImage.createFromPath(candidate).crop(box).toBitmap();
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
      assert(neutral>0&&matched/neutral>=.95,
        `Matched source sections differ outside colored overlays (${matched}/${neutral} neutral pixels within two sRGB levels)`);
      return {neutralPixels:neutral,matchedWithinTwoLevels:matched,meanAbsoluteDifference:totalDelta/neutral};
    };
    try{
      window.setSize(1440,960);
      window.webContents.setBackgroundThrottling(false);
      await js(`document.addEventListener('anatomy-frame',e=>{
        const {serial,interactive,surfaceQuality,cameraState,cursor,displayIdentity,sourceSignalFingerprint}=e.detail;
        window.vascularFrame={serial,interactive,surfaceQuality,cameraState,cursor,displayIdentity,sourceSignalFingerprint};
      });document.addEventListener('anatomy-gpu',e=>{
        const {serial,completed,displayIdentity}=e.detail;window.vascularGpu={serial,completed,displayIdentity};
      });`);
      for(const [index,recipe] of recipes.entries()){
        activeRecipe=recipe;
        if(index){
          const previous=await js(`document.querySelector('.rl-viewer')?.dataset.sourceDisplayIdentity`);
          await click('Open');
          await until(`!!document.querySelector('.rl-viewer')?.dataset.sourceDisplayIdentity&&document.querySelector('.rl-viewer').dataset.sourceDisplayIdentity!==${JSON.stringify(previous)}`,'reopened native study');
        }
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
        const view=await nativeView();
        const frame=await settledFrame();
        assert.equal(view.seriesId,recipe.seriesId);
        assert.equal(view.vesselScope,recipe.scope??'all');
        assert(frame?.cameraState,'Missing submitted anatomy camera');
        if(recipe.settings?.camera&&!recipe.preset){
          for(const key of ['position','target','up'])assert(frame.cameraState[key].every((n,i)=>Math.abs(n-recipe.settings.camera[key][i])<1e-3),`Camera ${key} drifted`);
        }
        if(recipe.cursor)assert(frame.cursor.every((n,i)=>Math.abs(n-recipe.cursor[i])<1e-3),'Source cursor drifted');
        assert(lastPayload?.compartments?.length,'Missing native anatomy payload');
        assert.deepEqual(lastPayload.unavailableLayers.filter(id=>recipe.layers.includes(id)),[]);
        const surfaces=lastPayload.compartments.map(({name,sourceSeries,alignment,surfaceDiagnostics,surface,displayFilter})=>({name,sourceSeries,alignment,surfaceDiagnostics,sourceSurfaceSha256:surface?.source?.sha256,empty:surface?.empty??false,displayFilter}));
        const record={name:recipe.name,seriesId:view.seriesId,sourceSha256:lastPayload.mri.sha256,scope:view.vesselScope,view:view.states[view.seriesId],surfaces,
          ...await capture(recipe.name)};
        if(recipe.compareVesselShading?.length){
          const startingMode=view.states[view.seriesId].settings.vesselShading??'balanced';
          const submitted=async mode=>{
            await until(enabledControlScript('Vessel shading'),'enabled vessel shading control');
            const prior=await js('window.vascularFrame.serial');
            await js(inputScript('Vessel shading',mode));
            return settledFrame(prior);
          };
          record.shading=[];
          for(const mode of recipe.compareVesselShading){
            if(mode===startingMode)continue;
            const shadedFrame=await submitted(mode);
            for(const key of ['position','target','up'])
              assert(shadedFrame.cameraState[key].every((n,i)=>Math.abs(n-frame.cameraState[key][i])<1e-9),'Vessel shading moved the camera');
            assert.deepEqual(shadedFrame.cursor,frame.cursor,'Vessel shading moved the source cursor');
            assert.equal(shadedFrame.displayIdentity,frame.displayIdentity,'Vessel shading changed resident source geometry');
            record.shading.push({mode,...await capture(`${recipe.name}-${mode}`)});
          }
          if(record.shading.length)await submitted(startingMode);
        }
        const captureSection=async()=>{
          const prior=(await submittedFrame()).serial;
          await click('Clip');await click('Section at cursor');await click('Close view controls');
          await settledFrame(prior);
          const sectionWorkspace=await nativeView();
          const sectionView=sectionWorkspace.states[recipe.seriesId];
          assert(sectionView.settings.clipping?.enabled||sectionWorkspace.patientCutaway?.clipping?.enabled,'Source section did not apply a display clip');
          return {view:sectionView,patientCutaway:sectionWorkspace.patientCutaway,...await capture(`${recipe.name}-source-section`)};
        };
        if(recipe.sourceSection&&!recipe.compareLayerIds)record.section=await captureSection();
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
          const angles=[];
          for(let i=0;i<recipe.rotationFrames;i++){
            const previous=await js('window.vascularFrame.serial');
            await pointer('pointerdown',x,y);
            await pointer('pointermove',x+5,y-2);
            await pointer('pointerup',x+5,y-2);
            await until(`window.vascularFrame?.serial>${previous}`,'rendered orbit movement');
            await wait(130);
            await capture(path.join(`${recipe.name}-rotation`,String(i).padStart(4,'0')),true);
            angles.push(await js('window.vascularFrame.cameraState.position'));
          }
          await js(`(()=>{const e=document.querySelector('.rl-viewer-anatomy-image');delete e.setPointerCapture;delete e.releasePointerCapture;})()`);
          assert(Math.hypot(...angles.at(-1).map((n,i)=>n-angles[0][i]))>1,'Orbit did not change the submitted camera path');
          record.rotationFrames=recipe.rotationFrames;
          record.rotationPositions=angles;
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
