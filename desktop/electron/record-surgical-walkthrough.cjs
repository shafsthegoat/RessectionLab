'use strict';

// electron record-surgical-walkthrough.cjs CASE EMPTY_OUTPUT SHOT_PLAN.json
// The ordinary Export observed replay component makes every native seek.
// Only its media sink is replaced to capture a widescreen, chaptered film.
const assert=require('node:assert/strict');
const {createHash,randomUUID}=require('node:crypto');
const {execFileSync}=require('node:child_process');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const {app,dialog,ipcMain}=require('electron');
const {observeStudyFrame}=require('./study-engine-recording.cjs');
const {inputScript}=require('./verify-eden-planning-helpers.cjs');
const {nativeWallMotionReceipt,matchedNativeWallReceipt,nativeWallReviewReceipt}=require('./native-wall-recording.cjs');
const cameraMath=import('./walkthrough-camera.mjs');
const repo=path.resolve(__dirname,'../..');
const caseFile=fs.realpathSync(process.argv[2]);
const output=path.resolve(process.argv[3]);
const planFile=fs.realpathSync(process.argv[4]);
const plan=JSON.parse(fs.readFileSync(planFile,'utf8'));
const recordNativeWalls=!!(plan.matchedAnatomy||plan.reviewNativeWalls);
if(!Array.isArray(plan.chapters)||!plan.chapters.length||!plan.caseHash)throw Error('A source-bound shot plan is required');
const parent=fs.realpathSync(path.dirname(output));
const localEvidence=fs.realpathSync(path.join(repo,'testcases/outputs'));
if(parent!==localEvidence&&!parent.startsWith(localEvidence+path.sep)){
  let external=parent;
  while(!fs.existsSync(path.join(external,'.ressectionlab-data-root.json'))&&path.dirname(external)!==external)
    external=path.dirname(external);
  const qc=path.join(external,'qc');
  assert(parent===qc||parent.startsWith(qc+path.sep),'Choose an ignored test output or verified external QC directory.');
  execFileSync(path.join(repo,'.venv/bin/python'),['-c',
    'from resectionlab.data_root import checked_root; import sys; checked_root(sys.argv[1])',external],{cwd:repo});
}
assert(!fs.existsSync(output),'Choose a fresh output directory');
fs.mkdirSync(output);
const digest=data=>createHash('sha256').update(data).digest('hex');
const report={caseFile,sourceFileSha256:digest(fs.readFileSync(caseFile)),plan,chapters:plan.chapters,frames:[],
  rendererAsset:null,windowEverVisible:false,mode:'Offline capture of accepted native observations by the production replay exporter'};
if(plan.sourceFileSha256)assert.equal(report.sourceFileSha256,plan.sourceFileSha256,'Shot plan requires its recorded source file');
const profile=fs.mkdtempSync(path.join(os.tmpdir(),'rslab-walkthrough-'));
app.setPath('userData',profile);
if(process.platform==='darwin')app.setActivationPolicy('accessory');
process.argv.push('--case',caseFile,'--automation-hidden');
delete process.env.RESSECTIONLAB_DEV_URL;
dialog.showMessageBox=async(...args)=>{throw Error('Unexpected app dialog '+args.at(-1).title);};
const save=()=>fs.writeFileSync(path.join(output,'capture.json'),JSON.stringify(report,null,2)+'\n');
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
let lastNativeRequest,backendFailure,latest,authority,windowRef,ui,sinkDone=false,overview,position=-1,activeChapter=-1,previous=null;
let nativeWallScene;
const spool=fs.openSync(path.join(output,'frames.mjpg'),'wx');
let bytes=0;
function append(image,info){
  assert(!image.isEmpty(),'Empty compositor frame');
  const dimensions=image.getSize();
  const resized=image.resize({width:Math.min(dimensions.width,1920),quality:'best'});
  const data=resized.toJPEG(93),frame={...info,offset:bytes,bytes:data.length,width:resized.getSize().width,height:resized.getSize().height};
  fs.writeSync(spool,data);bytes+=data.length;report.frames.push(frame);return frame;
}
function chapterFor(n){
  const i=plan.chapters.findIndex(c=>n>=c.first&&n<=c.last);
  if(i<0)throw Error('No shot plan for accepted action '+n);
  return i;
}
async function applyShot(chapter,stateId,override){
  if(!override&&chapter===activeChapter)return;
  const shot=override??plan.chapters[chapter];
  activeChapter=override?-1:chapter;
  const accepted=await ui.pane(stateId);
  if(shot.focus==='opening')await ui.click('Frame operative opening');
  if(shot.azimuthDeg!=null)await ui.range('Angle direction',shot.azimuthDeg);
  if(shot.fieldMm!=null)await ui.range('Field / magnification',shot.fieldMm);
  if(shot.angleDeg!=null)await ui.range('Viewing angle',shot.angleDeg);
  if(shot.focusPoint||shot.focusHand){
    const hand=shot.focusHand,expected=shot.focusPoint??accepted.tools[hand]?.position;
    assert(expected&&expected.length===3,'The optical reference must be an accepted hand');
    let matching=false;
    for(let attempt=0;attempt<6&&!matching;attempt++){
      if(shot.focusPoint){
        await ui.focusPoint(expected);
      }else await ui.click('Center field on '+hand+' accepted tip');
      await ui.paint();
      const shown=await ui.pane(stateId);
      matching=expected.every((v,i)=>Math.abs(v-shown.camera.target[i])<1e-5);
      if(!matching)await delay(80);
    }
    assert(matching,'Camera did not center on its registered optical focus');
  }
  if(shot.offsetMm!=null)await ui.range('Focus plane offset',shot.offsetMm);
  if(shot.clearParked&&latest.view.operativeExposureV1?.exteriorNormalRas){
    const shown=await ui.pane(stateId),slots=latest.view.slots;
    const targets=[...Object.values(slots).filter(t=>t.inserted).map(t=>t.position),...(shot.workingPoints??[])];
    const raisedSheet=Object.entries(latest.view.operativeExposureV1.bodies)
      .filter(([name,b])=>name!=='bone'&&b.angleRad>.45)
      .flatMap(([,b])=>b.centersMm.filter((_,i)=>i%Math.max(1,Math.floor(b.centersMm.length/400))===0));
    const opticalSlots={...slots};
    for(const [side,pose] of Object.entries(shot.withdrawnPoses??{})){
      opticalSlots[side+' at withdrawal']={...slots[side],...pose,inserted:false};
    }
    const {clearWorkingView}=await cameraMath;
    const option=clearWorkingView({raisedSheet,supportPair:shot.supportPair,camera:shown.camera,fieldMm:shown.fieldMm,aspect:shown.rect.width/shown.rect.height,
      azimuthDeg:shot.azimuthDeg,outward:latest.view.operativeExposureV1.exteriorNormalRas,slots:opticalSlots,targets});
    await ui.range('Angle direction',option.azimuthDeg);
    (report.opticalChoices??=[]).push({chapter,stateId,targets,...option});
  }
  if(!report.presentationChecked){
    report.presentationChecked=true;
    if(await ui.exists('Expand operative field'))await ui.click('Expand operative field');
    else if(plan.requirePresentation&&!await ui.exists('Restore operative workspace'))
      throw Error('This shot plan requires the operative presentation control.');
  }
}

const originalHandle=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(name,handler)=>originalHandle(name,async(...args)=>{
  if(name==='research:recordEngineMovie'){
    const request=args[1];
    if(request.command==='begin'){
      assert.equal(request.caseHash,plan.caseHash);
      assert.equal(request.branchId,report.branchId);
      authority={caseHash:request.caseHash,branchId:request.branchId,id:randomUUID()};
      return authority.id;
    }
    assert.equal(request.id,authority?.id);
    if(request.command==='cancel') {if(!sinkDone)report.exportCancelled=true;return;}
    if(request.command==='finish'){
      assert(previous&&previous.position+1===authority.steps&&previous.frameIndex+1===previous.frameCount);
      sinkDone=true;return {captured:report.frames.length,simulatedSeconds:previous.time};
    }
    assert.equal(request.command,'capture');
    const receipt=authority.receipt;
    assert(receipt && receipt.stateId===request.stateId);
    const nextAction=!previous||previous.frameIndex+1===previous.frameCount;
    assert.equal(receipt.position,previous?previous.position+Number(nextAction):0);
    assert.equal(receipt.frameIndex,nextAction?0:previous.frameIndex+1);
    if(previous)assert(receipt.time>=previous.time-1e-8);
    previous=receipt;
    // An optional partial first cut stops at an actual action endpoint.
    if(sinkDone)return;
    const chapter=chapterFor(receipt.position);
    await applyShot(chapter,receipt.stateId);
    const paneStart=performance.now();
    const pane=await ui.pane(receipt.stateId);
    const paneWaitMs=performance.now()-paneStart;
    assert.equal(pane.view,'surgeon','Ordinary microscope required');
    assert(!pane.inspection,'Inspection-only rendering cannot enter an operative chapter');
    assert.deepEqual(pane.tools,receipt.tools);
    const exposure=latest.view.operativeExposureV1;
    if(exposure){
      assert(pane.exposure,'Accepted operative bodies must be drawn.');
      assert.deepEqual(pane.exposure.edges,Object.fromEntries(Object.entries(exposure.bodies).map(([name,b])=>[name,b.cutCount])));
      assert.deepEqual(pane.exposure.objects,exposure.objects.filter(o=>o.status!=='retrieved').map(o=>o.id));
    }
    const captureStart=performance.now();
    const image=await windowRef.webContents.capturePage(pane.rect,{stayHidden:true,stayAwake:true});
    const capturedAt=performance.now();
    assert.equal(authority.receipt,receipt);
    const row=append(image,{...receipt,chapter,tool:authority.actions[receipt.position].action.instrument??
      receipt.tools[authority.actions[receipt.position].action.slot]?.instrument,slot:authority.actions[receipt.position].action.slot,
      timing:{paneWaitMs,capturePageMs:capturedAt-captureStart,nativeRequestMs:lastNativeRequest?.elapsedMs??null,
        requestToCaptureMs:lastNativeRequest?capturedAt-lastNativeRequest.start:null},
      active:latest.playback.effectiveActivation,
      measures:Object.fromEntries(['brainMaxDisplacementMm','fluidAddedMm3','fluidPresentMm3','fluidEscapedMm3',
        'fluidRemovedMm3','debrisMm3','debrisRemovedMm3'].map(k=>[k,latest.view.metrics[k]??null])),
      sourcePorts:Object.fromEntries(Object.entries(latest.view.vascularGraphV1?.ports??{}).map(([id,v])=>
        [id,{sealFraction:v.sealFraction,compressionFraction:v.compressionFraction}])),
      aspiratedSolidMm3:latest.view.cusaV3?.aspiratedSolidMm3??null,
      jawMm:Object.fromEntries(Object.entries(latest.view.slots).map(([k,t])=>[k,t.parameters?.jawOpeningMm??null])),
      exposure:exposure?{bodies:Object.fromEntries(Object.entries(exposure.bodies).map(([name,b])=>[name,
          {angleRad:b.angleRad,heldBy:b.heldBy,status:b.status,cutCount:b.cutCount}])),
          objects:exposure.objects.map(o=>({id:o.id,kind:o.kind,status:o.status})),accessOpen:exposure.accessOpen,complete:exposure.complete}:null,
      camera:pane.camera,gpuSerial:pane.gpuSerial,fieldMm:pane.fieldMm,field:pane.field,exteriorSkin:pane.exposure?.exteriorSkin??null,
      ...(recordNativeWalls?nativeWallMotionReceipt(pane,latest.view,nativeWallScene):{})});
    if(position!==receipt.position){
      position=receipt.position;
      console.log('Exporting action',position+1,'of',authority.steps);
    }
    if(receipt.frameIndex+1===receipt.frameCount){
      if(plan.actions)assert.equal(plan.actions[receipt.position].stateAfter,receipt.stateId,'Shot plan action identity differs from the saved branch');
      if(!report.actions)report.actions=[];
      report.actions.push({position:receipt.position,stateId:receipt.stateId,lastCaptured:report.frames.length-1,time:row.time,
        native:authority.actions[receipt.position]});
      if(plan.stills?.includes(receipt.position)||receipt.position===plan.chapters[chapter].last)
        fs.writeFileSync(path.join(output,'action-'+String(receipt.position+1).padStart(3,'0')+'.png'),image.toPNG());
      const index=receipt.position+1,match=plan.matchBefore?.[index];
      if(match){
        assert.equal(receipt.stateId,match.stateId,'Comparison requires the exact earlier native state');
        assert.equal(plan.caseHash,match.caseHash);
        const previousImage=match.beforeFile?path.resolve(path.dirname(planFile),match.beforeFile):null;
        if(previousImage&&fs.existsSync(previousImage))
          assert.equal(digest(fs.readFileSync(previousImage)),match.beforeSha256);
        await applyShot(chapter,receipt.stateId,match.shot);
        const drawn=await ui.pane(receipt.stateId);
        assert.equal(drawn.fieldMm,match.fieldMm);
        for(const key of ['position','target','up'])
          assert(drawn.camera[key].every((v,j)=>Math.abs(v-match.camera[key][j])<1e-5),'Unmatched comparison camera '+key);
        const proof=await windowRef.webContents.capturePage(drawn.rect,{stayHidden:true,stayAwake:true});
        const afterFile=path.join(output,'same-camera-'+index+'.png'),pixels=proof.toPNG();
        fs.writeFileSync(afterFile,pixels);
        (report.matchedFrames??={})[index]={stateId:receipt.stateId,caseHash:plan.caseHash,
          camera:drawn.camera,fieldMm:drawn.fieldMm,gpuSerial:drawn.gpuSerial,beforeFile:previousImage,
          beforeSha256:match.beforeSha256,afterFile,afterSha256:digest(pixels),beforeRenderer:match.beforeRenderer,
          afterRenderer:report.rendererAsset,exteriorSkin:drawn.exposure?.exteriorSkin??null};
        fs.writeFileSync(path.join(output,'same-camera-'+index+'.json'),JSON.stringify(report.matchedFrames[index],null,2)+'\n');
      }
      if(plan.stills?.includes(receipt.position)||receipt.position===plan.chapters[chapter].last){
        fs.appendFileSync(path.join(output,'review-states.jsonl'),JSON.stringify({action:index,stateId:receipt.stateId,
          tools:receipt.tools,source:plan.caseHash,title:plan.chapters[chapter].title,camera:row.camera,fieldMm:row.fieldMm,
          exposure:row.exposure,measures:row.measures,field:row.field})+'\n');
      }
      if(receipt.position===plan.chapters.at(-1).last&&receipt.position+1<authority.steps){
        sinkDone=true;
        await ui.click('Cancel movie export');
      }
    }
    return;
  }
  const nativeStarted=performance.now();
  let result;
  try{
    result=await handler(...args);
  }catch(error){
    if(name==='research:surgicalRehearsal'){
      backendFailure=String(error.message??error);
      console.error('Native rehearsal refused the request:',backendFailure);
    }
    throw error;
  }
  if(name==='research:surgicalRehearsal'&&result?.view){
    latest=result;
    lastNativeRequest={start:nativeStarted,elapsedMs:performance.now()-nativeStarted};
    if(authority)observeStudyFrame(authority,result);
  }
  return result;
});
app.on('browser-window-created',(_,win)=>{
  windowRef=win;
  win.on('show',()=>{report.windowEverVisible=true;win.hide();});
  win.webContents.once('did-finish-load',async()=>{
    const wc=win.webContents,js=code=>wc.executeJavaScript(code,true);
    const label=text=>`[...document.querySelectorAll('button,summary')].find(e=>e.getAttribute('aria-label')===${JSON.stringify(text)}||e.textContent.trim()===${JSON.stringify(text)}||e.innerText.trim()===${JSON.stringify(text)}||e.querySelector('strong')?.textContent.trim()===${JSON.stringify(text)})`;
    async function wait(fn,name,ms=120000){
      const end=Date.now()+ms;
      let nextLog=Date.now()+15000;
      do{
        if(backendFailure)throw Error(backendFailure);
        if(await fn())return;
        if(Date.now()>=nextLog){
          console.log('Waiting for',name);
          nextLog=Date.now()+30000;
        }
        await delay(25);
      }while(Date.now()<end);
      throw Error('Timed out waiting for '+name);
    }
    async function click(name){
      await wait(()=>js(`!!(${label(name)}) && !(${label(name)}).disabled`),name);
      await js(`(()=>{const e=${label(name)};for(let p=e.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;e.click()})()`);
    }
    async function pane(stateId){
      const capture=()=>js(`(()=>{const p=document.querySelector('[data-accepted-tools]'),v=p?.closest('.rl-viewer'),w=window.walkthrough;
        if(!p||p.dataset.surgicalStateId!==${JSON.stringify(stateId)}||v.dataset.surfacesPending!=='0'||v.dataset.rendererReady!=='true'||
           w.id!==p.dataset.surgicalStateId||w.source!==${JSON.stringify(plan.caseHash)})return null;
        const optical=JSON.parse(p.dataset.opticalCamera||'null');
        if(!optical||!w.camera||!['target','position','up'].every(k=>optical[k].every((v,i)=>Math.abs(v-w.camera[k][i])<1e-5)))return null;
        const r=p.getBoundingClientRect();const x=Math.ceil(r.left),y=Math.ceil(r.top);
        return {rect:{x,y,width:Math.floor(r.right)-x,height:Math.floor(r.bottom)-y},
          id:p.dataset.surgicalStateId,camera:optical,gpuSerial:w.serial,tools:JSON.parse(p.dataset.acceptedTools),
          fieldMm:Number(document.querySelector('[aria-label="Field / magnification"]')?.value),view:p.dataset.operativeView,
          inspection:!!document.querySelector('.surgeon-view strong[role=status]')?.textContent?.includes('INSPECTION ONLY'),
          ...(${recordNativeWalls}?{nativeVessels:w.frame?.operativeVesselSurfaces??[]}:{}),
          field:w.frame?.fieldOptics??null,exposure:JSON.parse(p.dataset.exposureGeometry||'null')};
      })()`);
      let found;await wait(async()=>!!(found=await capture()),'source-bound completed GPU frame '+stateId,120000);
      assert(found.rect.width>=400&&found.rect.height>=300);return found;
    }
    const paint=()=>js('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))');
    const focusPoint=async(coords)=>{
      for(const [i,axis] of ['R','A','S'].entries()){
        await js(inputScript('Microscope focus '+axis,coords[i]));
        await paint();
      }
      await click('Apply microscope focus');
    };
    ui={click,paint,focusPoint,exists:name=>js(`!!(${label(name)})`),range:async(name,value)=>{
      const input=`document.querySelector('[aria-label=${JSON.stringify(name)}]')`;
      const current=await js(`Number(${input}?.value)`);
      if(current!==value)await js(inputScript(name,value));
      await wait(()=>js(`Number(${input}?.value)===${value}`),name);
      await js('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))');
    },pane};
    try{
      wc.setBackgroundThrottling(false);wc.debugger.attach('1.3');
      // Review stills may use an ordinary desktop or compact window; the film stays widescreen.
      const [width,height]=plan.windowSize??[1920,1080];
      assert(Number.isInteger(width)&&Number.isInteger(height)&&width>=1050&&height>=720&&width<=1920&&height<=1080,'Unsupported capture window');
      assert(plan.reviewBoundaries||(width===1920&&height===1080),'A film uses the 1920 × 1080 frame');
      report.windowSize=[width,height];
      await wc.debugger.sendCommand('Emulation.setDeviceMetricsOverride',{width,height,mobile:false,deviceScaleFactor:1});
      win.setContentSize(width,height);
      await js(`window.walkthrough={submitted:new Map()};
        document.addEventListener('anatomy-frame',({detail:d})=>{
          const w=window.walkthrough;w.submitted.set(d.serial,d);
          if(w.submitted.size>8)w.submitted.delete(w.submitted.keys().next().value);
        });
        document.addEventListener('anatomy-gpu',({detail:d})=>{
          const w=window.walkthrough,frame=w.submitted.get(d.serial);
          if(d.completed&&frame&&frame.sourceCaseHash===d.sourceCaseHash&&frame.surgicalStateId===d.surgicalStateId){
            w.id=d.surgicalStateId;w.source=d.sourceCaseHash;w.serial=d.serial;w.frame=frame;w.camera=frame.cameraState;
          }
        });`);
      report.rendererAsset=await js("document.querySelector('script[type=module][src]')?.src.split('/').pop()");
      report.rendererSha256=digest(fs.readFileSync(path.join(repo,'desktop/dist/assets',report.rendererAsset)));
      report.appVersion=app.getVersion();report.electronVersion=process.versions.electron;
      report.gpu=app.getGPUFeatureStatus();
      assert(report.gpu.webgl==='enabled'||report.gpu.webgl2==='enabled');
      if(recordNativeWalls)
        await wait(()=>js("document.querySelector('.rl-viewer')?.dataset.rendererReady==='true'"),'loaded native TOF case');
      await click('Rehearse');
      const coldLimit=Math.max(300000,(plan.actions?.length??0)*2500);
      await wait(()=>js("!!document.querySelector('[aria-label=\"Surgical timeline\"]')"),'rehearsal',coldLimit);
      for(const button of ['Verify saved replay','Verify replay for planning']){
        if(await js(`!!(${label(button)}) && !(${label(button)}).disabled`))await click(button);
      }
      if(await ui.exists('Rehearse/edit stage'))await click('Rehearse/edit stage');
      await wait(()=>latest?.view?.stateId&&latest?.caseHash===plan.caseHash&&!latest.verification?.readOnly,'verified episode',coldLimit);
      if(plan.branchId&&latest.branchId!==plan.branchId){
        await js(inputScript('Surgical branch',plan.branchId));
        await wait(()=>latest.branchId===plan.branchId,'selected branch');
      }
      report.branchId=latest.branchId;report.sceneId=latest.view.scene.sceneId;
      if(recordNativeWalls)nativeWallScene=latest.view.scene;
      report.instruments=Object.fromEntries(latest.view.catalog.map(tool=>[tool.id,tool.name]));
      report.sourceCaseHash=latest.caseHash;report.steps=latest.branches.find(b=>b.id===latest.branchId).steps;
      assert(plan.chapters[0].first===0 && plan.chapters.at(-1).last<report.steps);
      await js(inputScript('Surgical timeline',0));
      await wait(()=>latest.position===0&&!latest.playback,'origin',300000);
      const origin=latest.view.stateId;report.origin=origin;
      if(!await ui.exists('External anatomy view'))await click('Surgeon / Microscope');
      const external=await ui.exists('External scalp overview')?'External scalp overview':'External anatomy view';
      if(plan.requireOpaqueOverview)assert.equal(external,'External scalp overview');
      await click(external);
      await wait(()=>js("document.querySelector('[data-accepted-tools]')?.dataset.operativeView==='external'"),'exterior');
      if(plan.requireOpaqueOverview){
        if(await ui.exists('Expand operative field'))await click('Expand operative field');
        await wait(()=>js("document.querySelector('.field-presentation-readout')?.innerText.includes('External scalp overview')"),'opaque external scalp presentation');
      }
      overview=await pane(origin);
      if(plan.matchedAnatomy)report.anatomyWallProof={
        nativeVessels:overview.nativeVessels,collisionGeometryByLabel:latest.view.vesselMechanics?.geometryModeByLabel??{}};
      const img=await wc.capturePage(overview.rect,{stayHidden:true,stayAwake:true});
      append(img,{chapter:-1,stateId:origin,time:0,overview:true,camera:overview.camera,gpuSerial:overview.gpuSerial,tools:overview.tools});
      fs.writeFileSync(path.join(output,'overview.png'),img.toPNG());
      await click('Surgeon / Microscope');
      await wait(()=>js("document.querySelector('[data-accepted-tools]')?.dataset.operativeView==='surgeon'"),'microscope');
      if(plan.chapters[0].focus==='opening')await click('Frame operative opening');
      report.initialCamera=(await pane(origin)).camera;
      if(plan.matchedAnatomy){
        await applyShot(0,origin);
        const matched=await pane(origin);
        const photo=await wc.capturePage(matched.rect,{stayHidden:true,stayAwake:true});
        const anatomyPixels=img.toPNG(),operativePixels=photo.toPNG();
        report.matchedBifurcation=matchedNativeWallReceipt(plan,origin,overview,matched,anatomyPixels,operativePixels);
        fs.writeFileSync(path.join(output,'anatomy-close-up.png'),anatomyPixels);
        fs.writeFileSync(path.join(output,'operative-close-up.png'),operativePixels);
        report.initialCamera=matched.camera;
        activeChapter=-1;
      }
      if(plan.reviewBoundaries){
        report.review=[];
        for(const boundary of plan.reviewBoundaries){
          assert(Number.isInteger(boundary)&&boundary>0&&boundary<=report.steps);
          await js(inputScript('Surgical timeline',boundary));
          await wait(()=>latest.position===boundary&&!latest.playback,'accepted boundary '+boundary);
          const stateId=latest.view.stateId;
          assert.equal(latest.timeline[boundary-1].stateAfter,stateId);
          await applyShot(chapterFor(boundary-1),stateId);
          const drawn=await pane(stateId);
          const tools=Object.fromEntries(Object.entries(latest.view.slots).map(([slot,t])=>[slot,
            {instrument:t.instrument,position:t.position,quaternion:t.quaternion,inserted:t.inserted}]));
          assert.deepEqual(drawn.tools,tools);
          assert.equal(drawn.view,'surgeon');assert(!drawn.inspection);
          if(plan.reviewNativeWalls?.[boundary]){
            await ui.paint();
            await delay(120);
            Object.assign(drawn,await pane(stateId));
          }
          const fieldImage=await wc.capturePage(drawn.rect,{stayHidden:true,stayAwake:true});
          fs.writeFileSync(path.join(output,'action-'+String(boundary).padStart(3,'0')+'.png'),fieldImage.toPNG());
          report.review.push({boundary,stateId,tools,camera:drawn.camera,gpuSerial:drawn.gpuSerial,fieldMm:drawn.fieldMm,exposure:drawn.exposure,
            ...(recordNativeWalls?nativeWallReviewReceipt(plan,boundary,drawn,report.review[0]?.camera,latest.view,nativeWallScene,fieldImage.toPNG()):{})});
          console.log('Reviewed accepted boundary',boundary);
        }
        assert(!win.isVisible()&&!report.windowEverVisible);
        assert.equal(digest(fs.readFileSync(caseFile)),report.sourceFileSha256);
        report.mode='Review stills at verified accepted action boundaries';
        report.completeEpisode=false;report.passed=true;save();app.exit(0);return;
      }
      if(plan.approachZoom){
        const finalField=plan.chapters[0].fieldMm??38;
        const wide=Number(plan.approachZoom.fromMm);
        assert(wide>=finalField&&wide<=120);
        const count=16;
        for(let i=0;i<count;i++){
          const t=i/(count-1),ease=t*t*(3-2*t);
          await ui.range('Field / magnification',Math.round(wide+(finalField-wide)*ease));
          const shown=await pane(origin);
          const image=await wc.capturePage(shown.rect,{stayHidden:true,stayAwake:true});
          append(image,{chapter:-1,view:shown.view,fieldMm:shown.fieldMm,stateId:origin,time:0,
            opticalTransition:true,camera:shown.camera,gpuSerial:shown.gpuSerial,tools:shown.tools});
        }
      }
      await click('Export observed replay');
      await wait(async()=>{
        if(sinkDone)return true;
        if(report.exportCancelled)throw Error('Export cancelled: '+await js("document.querySelector('[aria-label=\"Replay export status\"]')?.textContent"));
        const msg=await js("document.querySelector('[aria-label=\"Replay export status\"]')?.textContent");
        if(authority&&msg&&!/Choosing|Captured|Encoding/.test(msg))throw Error(msg);
        return false;
      },'native movie export',Math.max(1800000,report.steps*30000));
      assert(!win.isVisible()&&!report.windowEverVisible);
      assert.equal(digest(fs.readFileSync(caseFile)),report.sourceFileSha256);
      assert.equal(report.actions.length,plan.chapters.at(-1).last+1);
      report.completeEpisode=report.actions.length===report.steps;
      report.spoolBytes=bytes;report.passed=true;save();app.exit(0);
    }catch(error){
      report.error=String(error.stack??error);report.body=await js('document.body.innerText.slice(0,16000)').catch(()=>null);
      fs.writeFileSync(path.join(output,'failure.png'),(await wc.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());
      save();console.error(report.error);app.exit(1);
    }
  });
});
app.on('quit',()=>{fs.closeSync(spool);fs.rmSync(profile,{recursive:true,force:true});});
require('./main.cjs');
