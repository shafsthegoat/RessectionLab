'use strict';
// One actual UI workflow for admitted, geometry-refused and missing-target cases.
const {app,ipcMain,dialog}=require('electron');
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..'),base=path.join(root,'testcases');
const input=fs.realpathSync(process.env.ADMISSION_CASE||''),output=path.resolve(process.env.ADMISSION_OUTPUT||'');
for(const p of [input,output])assert(p.startsWith(base+path.sep),'Use ignored testcases paths');
assert(!fs.existsSync(output),'Preserve prior evidence; choose a fresh output');fs.mkdirSync(output,{recursive:true});
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex'),original=sha(input);
const report={input,output,inputSha256:original,expected:process.env.ADMISSION_EXPECT||'geometric_candidates',windowEverVisible:false,checks:[],requests:[]};
const saved=path.join(output,'admitted-saved.ressectionlab');
const expectedSchema=process.env.ADMISSION_SCHEMA||'resectionlab.interactive-source-approaches.v3';
app.setPath('userData',path.join(output,'hidden-profile'));process.argv.push('--case',input,'--automation-hidden');
dialog.showSaveDialog=async()=>({canceled:false,filePath:saved});
let lastResponse=null;
const handle=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(name,handler)=>handle(name,async(...args)=>{
  const result=await handler(...args);
  if(name==='research:surgicalRehearsal'){
    lastResponse=result;report.requests.push({command:args[1]?.command,stateId:result?.view?.stateId,
      position:result?.position,branchId:result?.branchId,status:result?.interactivePlanning?.status,
      publicInputHash:result?.interactivePlanning?.publicObservation?.inputHash,decision:result?.interactivePlanning?.decision,
      publicPlanSeal:result?.interactivePlanning?.publicPlanSeal,routes:result?.interactivePlanning?.routes?.length,admission:result?.interactivePlanningAvailability?.admission?.admissionSeal,
      refusal:result?.interactivePlanningAvailability?.reason});
  }
  return result;
});
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
app.on('browser-window-created',(_event,window)=>{
  window.on('show',()=>{report.windowEverVisible=true;window.hide();});
  window.webContents.once('did-finish-load',async()=>{
    window.webContents.setBackgroundThrottling(false);window.setContentSize(1440,900);
    const js=async code=>{const value=await window.webContents.executeJavaScript(`try{${code}}catch(error){({automationError:String(error)})}`,true);if(value?.automationError)throw Error(value.automationError);return value;};
    const until=async(predicate,description,seconds=120)=>{const start=Date.now();while(Date.now()-start<seconds*1000){if(await predicate())return;await delay(80);}throw Error('Timeout '+description);};
    const ready=()=>until(()=>js(`!!document.querySelector('[data-surgical-state-id]')&&document.querySelector('.rl-viewer')?.dataset.surfacesPending==='0'&&!document.querySelector('.viewer-loading')&&![...document.querySelectorAll('.surgical-panel [role="status"]')].some(n=>/^(Computing instrument action|Preparing an exact recorded state|Preparing accepted display|Verifying saved source-bound rehearsal)/.test(n.textContent.trim()))`),'native display ready');
    const click=async(label,selector='button')=>{await js(`(()=>{const b=[...document.querySelectorAll(${JSON.stringify(selector)})].find(b=>b.textContent.trim()===${JSON.stringify(label)}||b.getAttribute('aria-label')===${JSON.stringify(label)});if(!b||b.matches(':disabled'))throw Error('Missing enabled '+${JSON.stringify(label)});for(let p=b.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS'&&!(b.tagName==='SUMMARY'&&p===b.parentElement))p.open=true;b.click()})()`);await delay(100);};
    const capture=async(name,selector='.interactive-planning')=>{
      await ready();await js(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});for(let p=e?.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;e?.scrollIntoView({block:'start'})})()`);await delay(150);
      assert(!window.isVisible());fs.writeFileSync(path.join(output,name+'.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());
    };
    const generate=async()=>{if(!await js(`document.querySelector('[aria-label="Acknowledge interactive planning unknowns"]').checked`))await click('Acknowledge interactive planning unknowns','input[type=checkbox]');const count=report.requests.length;await click('Generate approaches');await until(()=>report.requests.slice(count).some(r=>r.command==='generateApproaches'),'native approach result');await ready();return lastResponse.interactivePlanning;};
    try{
      await until(()=>js(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`),'case load');
      await click('Rehearse');await ready();
      if(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify saved replay')`)){await click('Verify saved replay');await ready();}
      const initial=lastResponse;report.caseHash=initial.caseHash;report.sceneId=initial.view.scene.sceneId;report.baselineStateId=initial.view.stateId;
      if(report.expected==='missing-target'){
        assert.equal(initial.interactivePlanningAvailability.available,false);
        assert.match(initial.interactivePlanningAvailability.reason,/tumor target is required/);
        assert(!initial.view.scene.regions.some(r=>r.target));
        assert.equal(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Generate approaches')`),false);
        report.checks.push('Exact missing-target case refuses the common planning interface and exposes no Generate control');
        await capture('missing-target-refusal');
      }else{
        const qualification=initial.interactivePlanningAvailability.admission;
        assert(qualification);assert.equal(initial.view.physicalClock.revision,1);assert.equal(initial.view.toolMechanicsRevision,2);
        assert(qualification.sourceEvidence.unknownEvidence.includes('functional'));report.patientId=qualification.sourceEvidence.patientId;
        await click('Patient and task evidence','summary');
        assert(await js(`document.querySelector('[aria-label="Patient task evidence"]').innerText.includes(${JSON.stringify(report.patientId)})`));
        await capture('qualified-source','[aria-label="Patient task evidence"]');
        await js(`document.querySelector('[aria-label="Patient task evidence"]').open=false`);
        if(process.env.ADMISSION_REOPEN_ONLY==='1'){
          assert.equal(initial.interactivePlanning?.schema,expectedSchema);
          assert.equal(initial.view.stateId,process.env.ADMISSION_EXPECTED_STATE);
          report.checks.push('Fresh hidden Electron process verifies the saved versioned report, exact native baseline and external source-byte admission');
        }else{
          assert(initial.interactivePlanningAvailability.available);
          await click('Acknowledge interactive planning unknowns','input[type=checkbox]');
          const comparison=await generate();assert.equal(comparison.schema,expectedSchema);
          assert(comparison.publicObservation?.inputHash);assert.equal(comparison.publicObservation.evaluationReferencesExposed,false);
          assert(comparison.publicPlanSeal);assert.equal(comparison.decision.operation,comparison.routes.length?'COMPARE':'STOP');
          assert(await js(`!!document.querySelector('[aria-label="Planning decision"]')`));
          for(const route of comparison.routes)assert(route.publicExecution?.accessAccepted&&route.publicExecution?.withdrawalAccepted);
          report.checks.push('V3 candidates, native suffix previews and explicit COMPARE/STOP share a sealed public observation');
          assert.equal(comparison.status,report.expected);assert.equal(lastResponse.view.stateId,initial.view.stateId);
          assert.equal(comparison.taskAdmissionSeal,qualification.admissionSeal);
          fs.writeFileSync(path.join(output,'desktop-comparison.json'),JSON.stringify(comparison,null,2));
          if(comparison.routes.length){
            await click('Source approach 1');await click('Preview checked approach');await ready();
            assert(lastResponse.approachPreview?.receipt?.publicAccessAccepted);assert(lastResponse.approachPreview.receipt.publicWithdrawalAccepted);
            assert(lastResponse.approachPreview.receipt.nativeMatchesPublicPrediction);
            assert.equal(lastResponse.approachPreview.receipt.publicInputHash,comparison.publicObservation.inputHash);
            assert.equal(lastResponse.view.stateId,initial.view.stateId);await capture('checked-native-preview');
            await click('Install checked branch');await ready();assert.notEqual(lastResponse.view.stateId,initial.view.stateId);
            assert.equal(lastResponse.interactivePlanningAvailability.available,false);await capture('installed-native-branch');
            await click('Open planning checkpoint');await ready();assert.equal(lastResponse.view.stateId,initial.view.stateId);
            report.checks.push('Actual generated route previews full native access/withdrawal, installs a distinct branch and returns to the exact admitted checkpoint');
          }else{
            assert.equal(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Preview checked approach')`),false);
            report.checks.push('Admitted source with insufficient geometric access returns no route or executable preview');
          }
          await capture('geometric-comparison');
          await click('Require functional evidence','input[type=checkbox]');
          const refusal=await generate();assert.equal(refusal.status,'required_evidence_unavailable');
          assert.equal(refusal.decision.operation,'STOP');assert.equal(refusal.publicObservation.inputHash,comparison.publicObservation.inputHash);
          fs.writeFileSync(path.join(output,'desktop-refusal.json'),JSON.stringify(refusal,null,2));
          assert.deepEqual(refusal.missingRequiredEvidence,['functional']);assert.deepEqual(refusal.routes,[]);
          assert.equal(lastResponse.view.stateId,initial.view.stateId);await capture('functional-evidence-refusal');
          report.checks.push('Missing functional evidence refuses every route without changing physical state');
          await click('Save');await until(()=>fs.existsSync(saved),'saved native workspace');
          report.saved=saved;report.savedStateId=lastResponse.view.stateId;
        }
      }
      assert.equal(sha(input),original);assert(!report.windowEverVisible);assert(!window.isVisible());report.sourcePreserved=true;report.passed=true;
    }catch(error){report.error=String(error);report.body=await js('document.body.innerText').catch(()=>null);fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());console.error(error);}
    fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,patientId:report.patientId,error:report.error,saved:report.saved}));app.exit(report.passed?0:1);
  });
});
require('./main.cjs');
