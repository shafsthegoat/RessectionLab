'use strict';
// Drive the real Cases, Anatomy and Plan controls without a visible window.
const {app,dialog,ipcMain}=require('electron');
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict');
const {createHash}=require('node:crypto'),{execFileSync}=require('node:child_process');
const {inputScript,enabledControlScript,displayIdentity}=require('./verify-eden-planning-helpers.cjs');
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const read=file=>JSON.parse(fs.readFileSync(file,'utf8'));
const sha=file=>createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const inside=(parent,child)=>child.startsWith(parent+path.sep);
const fatal=error=>{console.error(error?.stack||String(error));app.exit(1);};
assert.equal(process.platform,'darwin');
app.setActivationPolicy('accessory');
dialog.showErrorBox=(title,message)=>fatal(Error(title+' '+message));
process.on('uncaughtException',fatal);process.on('unhandledRejection',fatal);

const checkout=path.resolve(__dirname,'../..');
const root=execFileSync(path.join(checkout,'.venv/bin/python'),['-c',
  'from resectionlab.data_root import checked_root; print(checked_root(expected_uuid="9CC3623C-E38B-3EEE-B6BB-B9A06C193700"))'],
  {cwd:checkout,encoding:'utf8',timeout:8000}).trim();

const prepared=fs.realpathSync(path.join(root,'prepared/eden2020-approach-comparison-goal-v1'));
const source=fs.realpathSync(process.env.EDEN_STUDY||''),targetLayerId=process.env.EDEN_TARGET;
assert(inside(prepared,source)&&path.basename(source)==='patient-anatomy.json');
assert(['synthetic-target','synthetic-target-08'].includes(targetLayerId));
const requestedRadius=process.env.EDEN_CORRIDOR_RADIUS;
const corridorRadius=Number(requestedRadius??2);
assert(Number.isFinite(corridorRadius)&&corridorRadius>0&&corridorRadius<=50,
  'Use a finite corridor radius in (0, 50] mm');
const document=read(source);assert(document.display.planning_targets[targetLayerId]);
const localEvidence=process.env.EDEN_VERIFICATION_ROOT;
const verified=localEvidence?fs.realpathSync(localEvidence):path.join(prepared,'verification');
if(localEvidence){
  const testcases=fs.realpathSync(path.join(checkout,'testcases'));
  assert(inside(testcases,verified)&&fs.statSync(verified).dev===fs.statSync(testcases).dev,
    'Local verification root must be an existing directory on the checkout testcases volume');
}
fs.mkdirSync(verified,{recursive:true});assert.equal(fs.realpathSync(verified),verified);
const label=process.env.EDEN_RUN;
assert(/^[A-Za-z0-9_-]{1,120}$/.test(label??''));
const priorFile=process.env.EDEN_EXPECTED&&fs.realpathSync(process.env.EDEN_EXPECTED);
if(priorFile)assert(inside(verified,priorFile));
const prior=priorFile?read(priorFile):null;
if(prior){assert(prior.passed);assert.equal(prior.source,source);assert.equal(prior.targetLayerId,targetLayerId);assert.notEqual(prior.pid,process.pid);}
const resumeFile=process.env.EDEN_RESUME_SUGGEST&&fs.realpathSync(process.env.EDEN_RESUME_SUGGEST);
if(resumeFile)assert(inside(verified,resumeFile)&&!prior);
const resumed=resumeFile?read(resumeFile):null;
if(resumed){assert(!resumed.passed&&Number.isFinite(resumed.timingsMs.suggest));assert.equal(resumed.source,source);assert.equal(resumed.targetLayerId,targetLayerId);}
const output=path.join(verified,label);fs.mkdirSync(output);
const shared=process.env.EDEN_WORKSPACE&&fs.realpathSync(process.env.EDEN_WORKSPACE);
const study=prior?fs.realpathSync(prior.study):shared||path.join(output,'workspace','patient-anatomy.json');
assert(inside(verified,study)&&study!==source&&path.basename(study)==='patient-anatomy.json');
if(localEvidence)assert(prior||shared,'Stage a checked local workspace before launching Electron');
if(resumed)assert.equal(resumed.study,study);
const report={schema:'ressectionlab.eden-approach-comparison-ui.v1',source,study,targetLayerId,patientId:document.patient_id,
  snapshotId:document.snapshot_id,pid:process.pid,priorFile,started:new Date().toISOString(),windowEverVisible:false,
  concurrentWorkload:process.env.EDEN_CONCURRENT_WORKLOAD||null,
  operations:[],timingsMs:{},memory:{samples:0,electronPeakKiB:0,sidecarPeakKiB:0,combinedPeakKiB:0},passed:false};
const reportPath=path.join(output,'report.json');
const writeReport=()=>fs.writeFileSync(reportPath,JSON.stringify(report,null,2));
dialog.showErrorBox=(_title,message)=>{report.error=message;writeReport();app.exit(1);};

const profile=fs.mkdtempSync(path.join(os.tmpdir(),'ressection-eden-comparison-'));
app.setPath('userData',profile);
process.argv.push('--automation-hidden');
const library=require('./case-library-selection.cjs'),Library=library.CaseLibrarySelection;
library.CaseLibrarySelection=class extends Library {
  constructor(options){super({...options,defaultRoot:null,externalRoot:null,preparedRoot:path.join(profile,'prepared-cases'),nativePaths:[study]});}
};
const sidecar=require('./sidecar.cjs'),Sidecar=sidecar.Sidecar;
sidecar.Sidecar=class extends Sidecar {
  constructor(options){
    if(localEvidence)assert.equal(fs.statSync(options.transferDir).dev,fs.statSync(verified).dev,
      'Temporary renderer transfers must stay on the local verification volume');
    super(options);report.sidecarPid=this.child.pid;
  }
};
const setup=performance.now();
if(!prior&&!shared){
  const workspace=path.dirname(study),original=path.dirname(source);
  fs.mkdirSync(workspace);
  fs.copyFileSync(source,study,fs.constants.COPYFILE_EXCL);
  for(const member of ['arrays','originals','revisions']){
    fs.cpSync(path.join(original,member),path.join(workspace,member),{recursive:true,errorOnExist:true,force:false,filter:name=>!path.basename(name).startsWith('.')});
  }
}
assert.equal(sha(study),sha(source),'Working copy starts from exact prepared source manifest');
report.timingsMs.prepareWorkingCopy=Math.round(performance.now()-setup);
// The source root is verified above. Local runs use the isolated library and
// the app's normal local transfer directory, including for generated previews.
if(localEvidence)delete process.env.RESSECTIONLAB_DATA_ROOT;

let active=0,openCount=0,lastPlan=null,planVersion=0,lastPlanAction=null,savedView=null;
const originalHandle=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(name,handler)=>originalHandle(name,async(event,args)=>{
  const start=performance.now();active++;
  try{
    const result=await handler(event,args);
    if(name==='research:openListedWorkspace'){
      assert.equal(result.document.patient_id,document.patient_id);assert.equal(result.document.snapshot_id,document.snapshot_id);openCount++;
    }
    if(name==='research:anatomyPlan'){lastPlan=result;planVersion++;lastPlanAction=args.action;}
    if(name==='research:anatomySaveView')savedView=JSON.parse(JSON.stringify(args.viewState));
    report.operations.push({name,action:args?.action,ms:Math.round(performance.now()-start)});
    return result;
  }catch(error){report.requestError={name,action:args?.action,message:error.message};throw error;}finally{active--;}
});
function sampleMemory(){
  const electron=app.getAppMetrics().reduce((sum,row)=>sum+(row.memory.workingSetSize||0),0);
  const worker=report.sidecarPid?Number(execFileSync('/bin/ps',['-o','rss=','-p',String(report.sidecarPid)],{encoding:'utf8',timeout:1500}).trim()):0;
  const memory=report.memory;
  memory.samples++;memory.electronPeakKiB=Math.max(memory.electronPeakKiB,electron);
  memory.sidecarPeakKiB=Math.max(memory.sidecarPeakKiB,worker);memory.combinedPeakKiB=Math.max(memory.combinedPeakKiB,electron+worker);
}

app.on('browser-window-created',(_event,win)=>{
  win.on('show',()=>{report.windowEverVisible=true;win.hide();});
  win.webContents.once('did-finish-load',async()=>{
    const js=source=>win.webContents.executeJavaScript(source,true);
    async function until(check,description,seconds=300){
      const start=performance.now();
      while(performance.now()-start<seconds*1000){
        if(report.requestError)throw Error(JSON.stringify(report.requestError));
        const alert=await js('document.querySelector(".rl-viewer-error,.approach-error,[data-layer-error=true]")?.textContent');
        if(alert)throw Error(alert);
        if(await check())return;
        await delay(100);
      }
      throw Error('Timeout waiting for '+description);
    }
    const wait=(script,description)=>until(()=>js(script),description);
    async function click(label){
      await js(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(label)}||b.getAttribute('aria-label')===${JSON.stringify(label)});if(!b||b.matches(':disabled'))throw Error('Missing enabled control '+${JSON.stringify(label)});b.scrollIntoView({block:'nearest'});b.click()})()`);
    }
    const select=(label,value)=>js(inputScript(label,value));
    const state=()=>js(`(()=>{const v=document.querySelector('.rl-viewer'),w=document.querySelector('.app-shell.native-workspace'),panel=document.querySelector('.native-approaches');
      const frame=window.__edenComparison?.frame,gpu=window.__edenComparison?.gpu;
      return {identity:v?.dataset.sourceDisplayIdentity,ready:v?.dataset.rendererReady==='true',view:w?.dataset.nativeViewState?JSON.parse(w.dataset.nativeViewState):null,
        selected:panel?.querySelector('.approach-card.selected')?.dataset.approachId??null,target:panel?.querySelector('[aria-label="Planning target"]')?.value,
        routes:JSON.parse(v?.dataset.approachRoutes||'[]'),scenarioSet:panel?.querySelector('.approach-sensitivity')?.dataset.scenarioSet,
        pending:!!document.querySelector('.viewer-loading,.rl-viewer-preparing,[data-layer-pending=true],[data-approach-pending=true]'),frame,gpu}
    })()`);
    async function ready(){
      await until(async()=>{const s=await state();return s.ready&&!s.pending&&!active&&s.frame?.displayIdentity===s.identity&&s.gpu?.displayIdentity===s.identity;},'hidden source image GPU frame');
      return state();
    }
    async function capture(name,selector){
      if(selector)await js(`(()=>{const node=document.querySelector(${JSON.stringify(selector)});if(!node)return;if(node.tagName==='DETAILS')node.open=true;node.scrollIntoView({block:'start'});})()`);
      await ready();await delay(140);assert(!win.isVisible(),'Electron verification must remain hidden');
      const image=await win.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true});
      assert(!image.isEmpty(),'Hidden capture is empty');
      fs.writeFileSync(path.join(output,name+'.png'),image.toPNG());
    }
    async function measure(label,action,button){
      const version=planVersion,start=performance.now();
      await click(button);await until(()=>planVersion>version&&lastPlanAction===action,label);
      const scene=await ready();report.timingsMs[label]=Math.round(performance.now()-start);
      console.log(label+' '+report.timingsMs[label]+' ms');
      return {plan:lastPlan,scene};
    }
    let targetOptions=[];
    const comparison=plan=>({snapshotId:plan.snapshotId,approaches:plan.approaches,proposalReport:plan.proposalReport,inputComparison:plan.inputComparison,targetOptions});
    const selected=plan=>plan.approaches.find(route=>route.id===plan.selectedId);
    const planFile=path.join(path.dirname(study),'anatomy-approaches.json'),viewFile=path.join(path.dirname(study),'workspace-view.json');
    const pulse=setInterval(()=>{try{sampleMemory();}catch(error){report.memory.lastSampleError=error.message;}},750);
    const deadline=setTimeout(()=>{report.error='Global verifier timeout';writeReport();app.exit(1);},15*60*1000);
    try{
      win.webContents.setBackgroundThrottling(false);win.setContentSize(1560,1000);
      await js(`window.__edenComparison={};
        document.addEventListener('anatomy-frame',e=>{window.__edenComparison.frame=e.detail});
        document.addEventListener('anatomy-gpu',e=>{if(e.detail.completed)window.__edenComparison.gpu=e.detail});
        window.dispatchEvent(new Event('resize'));`);
      await ready();
      const openAt=performance.now();
      await wait(`[...document.querySelectorAll('button')].some(b=>b.getAttribute('aria-label')==='Cases')`,'Cases navigation');
      await click('Cases');
      await wait(`[...document.querySelectorAll('.poc-case-card')].some(card=>card.textContent.includes(${JSON.stringify(document.patient_id)})&&!card.disabled)`,'isolated source study in Cases');
      await js(`[...document.querySelectorAll('.poc-case-card')].find(card=>card.textContent.includes(${JSON.stringify(document.patient_id)})).click()`);
      await until(()=>openCount===1,'case selected in library');
      await wait(`[...document.querySelectorAll('.workspace-tabs button')].some(b=>b.textContent==='Anatomy'&&b.classList.contains('selected'))`,'Anatomy selected');
      const anatomy=await ready();assert.equal(displayIdentity(anatomy.identity).snapshotId,document.snapshot_id);
      await click('Plan & compare');await wait(enabledControlScript('Planning target'),'source-backed target options');
      await ready();targetOptions=lastPlan.targetOptions;
      report.timingsMs.casesToPlanning=Math.round(performance.now()-openAt);
      assert.deepEqual(targetOptions.filter(row=>row.target).map(row=>row.layerId),['synthetic-target','synthetic-target-08']);

      if(prior){
        assert.equal(sha(planFile),prior.persisted.planSha256);
        assert.equal(sha(viewFile),prior.persisted.viewSha256);
        assert.deepEqual(comparison(lastPlan),prior.persisted.comparison,'Cold restore includes exact candidate, sensitivity and paired receipts');
        const reopened=await ready();
        assert.deepEqual(reopened.view,prior.persisted.view);
        assert.equal(reopened.selected,prior.persisted.selected);assert.equal(reopened.target,targetLayerId);
        assert.equal(reopened.scenarioSet,prior.persisted.scenarioSet);assert.deepEqual(reopened.routes,prior.persisted.routes);
        assert.equal(displayIdentity(reopened.identity).sha256,prior.persisted.sourceSha256);
        report.restored={planSha256:sha(planFile),viewSha256:sha(viewFile),previousPid:prior.pid};
        await capture('reopened','.approach-cards');
        if(process.env.EDEN_OTHER_EXPECTED){
          const otherFile=fs.realpathSync(process.env.EDEN_OTHER_EXPECTED);assert(inside(verified,otherFile));
          const other=read(otherFile);assert(other.passed);assert.equal(other.study,study);assert.notEqual(other.targetLayerId,targetLayerId);
          const switchAt=performance.now();
          await select('Planning target',other.targetLayerId);
          await wait(`document.querySelector('[aria-label="Planning target"]').value===${JSON.stringify(other.targetLayerId)}`,'earlier saved target');
          const switched=await ready();
          report.timingsMs.switchSavedTarget=Math.round(performance.now()-switchAt);
          assert.deepEqual(comparison(lastPlan),other.persisted.comparison);
          assert.equal(switched.selected,other.persisted.selected);assert.equal(switched.scenarioSet,other.persisted.scenarioSet);
          assert.deepEqual(switched.routes,other.persisted.routes);
          report.restoredOtherTarget={targetLayerId:other.targetLayerId,report:otherFile};
          await capture('other-saved-target','.approach-cards');
        }
      }else{
        await capture('01-source-anatomy');
        if((await state()).target!==targetLayerId){
          const targetAt=performance.now();
          await select('Planning target',targetLayerId);
          await wait(`document.querySelector('[aria-label="Planning target"]').value===${JSON.stringify(targetLayerId)}`,'selected synthetic target');
          await ready();
          report.timingsMs.selectTarget=Math.round(performance.now()-targetAt);
        }
        await select('Entry patch radius',15);
        let suggestion;
        if(resumed){
          const scene=await ready();assert(lastPlan.proposalReport?.assessmentStatus==='current');
          assert.deepEqual(lastPlan.proposalReport.task.sensitivityConfig,{translationMm:1,boundaryMm:.5});
          report.resumedSuggestion={report:resumeFile,ms:resumed.timingsMs.suggest};
          suggestion={scene,plan:{...lastPlan,selectedId:scene.selected}};
        }else{
          if(requestedRadius!==undefined)await select('Corridor radius',corridorRadius);
          suggestion=await measure('suggest','suggest','Suggest approaches');
        }
        const proposals=suggestion.plan.proposalReport;
        assert.equal(proposals.task.targetLayerId,targetLayerId);
        assert.equal(proposals.task.radiusMm,corridorRadius);
        assert.deepEqual(proposals.task.target,targetOptions.find(row=>row.layerId===targetLayerId).target);
        assert.equal(proposals.task.entryPatch.radiusMm,15);
        assert.equal(proposals.candidates.length,proposals.enumerated);
        assert.equal(proposals.candidates.filter(row=>['rejected','scenario-vetoed'].includes(row.disposition)).length,proposals.rejected);
        const routes=suggestion.plan.approaches.filter(route=>proposals.selectedCandidateIds.includes(route.id));
        assert.equal(proposals.accepted,routes.length);
        if(proposals.accepted<3)assert(proposals.whyFewer);
        for(const [index,route] of routes.entries()){
          assert.equal(route.sourceSnapshotId,document.snapshot_id);
          assert.equal(route.measurements.sensitivity.scenarioSetId,proposals.scenarioSet.id);
          assert.strictEqual(route.measurements.simulatedRemovedTargetVolumeMm3,null);
          for(const previous of routes.slice(0,index)){
            assert(Math.hypot(...route.entry.map((value,i)=>value-previous.entry[i]))>=4-1e-6);
            assert(route.direction.reduce((sum,value,i)=>sum+value*previous.direction[i],0)<=Math.cos(3*Math.PI/180)+1e-8);
          }
          const drawn=suggestion.scene.routes.find(row=>row.route_id===route.id);
          assert.deepEqual(drawn.entry_mm,route.entry);assert.deepEqual(drawn.target_mm,route.target);assert.equal(drawn.tool.shaft_radius_mm,route.radiusMm);
        }
        await capture('02-alternatives',routes.length?'.approach-cards':'.approach-proposal-report');
        const paired=(await measure('compareInputs','compareInputs','Compare input sets')).plan.inputComparison;
        assert.deepEqual(paired.task,proposals.task,'Paired arms must use the exact displayed task');
        assert.equal(paired.candidateBudget,proposals.enumerated);
        assert.equal(paired.arms.rich.candidates.length,paired.arms.restricted.candidates.length);
        assert.equal(paired.arms.restricted.tractIds.length,0);
        for(const arm of Object.values(paired.arms))assert.equal(arm.abstentionCount+arm.nominated.length,3);
        await capture('03-rich-restricted','.approach-input-result');
        if(process.env.EDEN_EDIT==='1'){
          const editedPlan=suggestion.plan.selectedId?suggestion.plan:(await measure('manual','upsert','Create manual approach')).plan;
          const before=selected(editedPlan);
          let version=planVersion,start=performance.now();
          await select('Corridor radius',3);
          await until(()=>planVersion>version&&lastPlan.approaches.find(a=>a.id===before.id)?.radiusMm===3,'source-linked radius recompute');
          await ready();report.timingsMs.radiusEdit=Math.round(performance.now()-start);
          const radius=selected(lastPlan);
          assert.equal(radius.sourceSnapshotId,before.sourceSnapshotId);
          assert.notEqual(radius.measurementHash,before.measurementHash);assert.notDeepEqual(radius.measurements.structures,before.measurements.structures);
          version=planVersion;start=performance.now();
          await select('Boundary sensitivity',1);
          await until(()=>planVersion>version&&lastPlan.approaches.find(a=>a.id===before.id)?.sensitivityConfig.boundaryMm===1,'sensitivity recompute at fixed geometry');
          await ready();report.timingsMs.boundaryEdit=Math.round(performance.now()-start);
          const changed=selected(lastPlan);
          assert.deepEqual(changed.entry,radius.entry);assert.deepEqual(changed.target,radius.target);assert.equal(changed.radiusMm,radius.radiusMm);
          assert.notEqual(changed.measurements.sensitivity.scenarioSetId,radius.measurements.sensitivity.scenarioSetId);
          assert.notDeepEqual(changed.measurements.sensitivity.scenarios,radius.measurements.sensitivity.scenarios);
          report.edits={original:before.measurementHash,radius:radius.measurementHash,boundary:changed.measurementHash};
          await capture('04-manual-sensitivity','.approach-sensitivity');
        }
        const saveAt=performance.now();await click('Save study & comparison');
        await until(()=>savedView&&!active,'completed native Save');
        const final=await ready();report.timingsMs.save=Math.round(performance.now()-saveAt);
        assert.deepEqual(read(viewFile),savedView);assert.deepEqual(final.view,savedView);
        report.persisted={planSha256:sha(planFile),viewSha256:sha(viewFile),view:savedView,comparison:comparison(lastPlan),
          routes:final.routes,selected:final.selected,scenarioSet:final.scenarioSet,sourceSha256:displayIdentity(final.identity).sha256};
        await capture('05-saved','.approach-cards');
      }
      assert(!win.isVisible()&&!report.windowEverVisible);
      assert.equal(sha(study),sha(source),'The private prepared source manifest stays unchanged');
      report.passed=true;
    }catch(error){
      report.error=error.stack;report.body=await js('document.body.innerText').catch(()=>null);
      try{fs.writeFileSync(path.join(output,'failure.png'),(await win.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());}catch{}
      console.error(error.stack);
    }finally{
      clearInterval(pulse);clearTimeout(deadline);report.finished=new Date().toISOString();
      writeReport();console.log(reportPath);
      if(report.passed)app.quit();else app.exit(1);
    }
  });
});
require('./main.cjs');
