'use strict';
// Actual hidden UI execution; no primary-display show/focus calls.
const {app,ipcMain,dialog}=require('electron');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..'),base=path.join(root,'testcases');
const input=fs.realpathSync(process.env.STRATEGY_CASE||''),output=path.resolve(process.env.STRATEGY_OUTPUT||'');
for(const p of [input,output])assert(p.startsWith(base+path.sep));assert(!fs.existsSync(output),'Preserve previous evidence');fs.mkdirSync(output,{recursive:true});
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex'),original=sha(input);
const report={input,sourceSha256:original,output,expected:process.env.STRATEGY_EXPECT||'paired',requests:[],interactions:[],timingScope:'IPC backend separately from click-to-accepted-display; latter includes UI dispatch/polling and surface preparation',checks:[],windowEverVisible:false};
assert(['paired','no-route','no-sequence','missing-target'].includes(report.expected),'Declare the expected complete-plan result');
const stoppedReason=report.expected==='no-sequence'?'NO_COMPLETE_TWO_TOOL_SEQUENCE':'NO_COMPLETE_TOOL_ACCESS';
const saved=path.join(output,'strategies-saved.ressectionlab');let last=null;
app.setPath('userData',path.join(output,'hidden-profile'));process.argv.push('--case',input,'--automation-hidden');
dialog.showSaveDialog=async()=>({canceled:false,filePath:saved});
const handle=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(name,handler)=>handle(name,async(...args)=>{const start=Date.now(),value=await handler(...args);if(name==='research:surgicalRehearsal'){last=value;report.requests.push({command:args[1]?.command,seconds:(Date.now()-start)/1000,stateId:value?.view?.stateId,position:value?.position,branchId:value?.branchId,decision:value?.strategyPlanning?.publicPlan?.decision,preview:value?.approachPreview?.candidateId,phase:value?.approachPreview?.phaseId});}return value;});
const delay=ms=>new Promise(r=>setTimeout(r,ms));
app.on('browser-window-created',(_event,window)=>{
 window.on('show',()=>{report.windowEverVisible=true;window.hide();});
 window.webContents.once('did-finish-load',async()=>{
  window.webContents.setBackgroundThrottling(false);window.setContentSize(1440,900);
  const js=async code=>{const x=await window.webContents.executeJavaScript(`try{${code}}catch(error){({automationError:String(error)})}`,true);if(x?.automationError)throw Error(x.automationError);return x;};
  const until=async(fn,label,seconds=process.env.STRATEGY_REOPEN_SENSITIVITY==='1'?240:120)=>{const t=Date.now();while(Date.now()-t<seconds*1000){if(await fn())return;const error=await js(`document.querySelector('.surgical-panel .surgical-error')?.textContent`);if(error)throw Error(error);await delay(80);}throw Error('Timeout '+label);};
  const ready=()=>until(()=>js(`!!document.querySelector('[data-surgical-state-id]')&&document.querySelector('.rl-viewer')?.dataset.surfacesPending==='0'&&!document.querySelector('.viewer-loading')&&![...document.querySelectorAll('.surgical-panel [role="status"]')].some(n=>/^(Computing instrument action|Preparing an exact recorded state|Preparing accepted display|Verifying saved source-bound rehearsal)/.test(n.textContent.trim()))`),'accepted display');
  const click=async(label,selector='button')=>{await js(`(()=>{const b=[...document.querySelectorAll(${JSON.stringify(selector)})].find(b=>b.textContent.trim()===${JSON.stringify(label)}||b.getAttribute('aria-label')===${JSON.stringify(label)});if(!b||b.matches(':disabled'))throw Error('Missing enabled '+${JSON.stringify(label)});for(let p=b.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS'&&!(b.tagName==='SUMMARY'&&p===b.parentElement))p.open=true;b.click()})()`);await delay(100);};
  const set=async(label,value)=>{await js(`(()=>{const e=document.querySelector('[aria-label='+${JSON.stringify(JSON.stringify(label))}+']');if(!e)throw Error('Missing input');Object.getOwnPropertyDescriptor(e.tagName==='SELECT'?HTMLSelectElement.prototype:HTMLInputElement.prototype,'value').set.call(e,${JSON.stringify(String(value))});e.dispatchEvent(new Event(e.tagName==='SELECT'?'change':'input',{bubbles:true}));})()`);await delay(150);};
  const capture=async(name,selector='.two-tool-planning')=>{await ready();await js(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});if(e){e.open=true;e.scrollIntoView({block:'start'})}})()`);await delay(150);assert(!window.isVisible());fs.writeFileSync(path.join(output,name+'.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());};
  const timing=(name,command,start,count)=>{const request=report.requests.slice(count).find(x=>x.command===command);assert(request);report.interactions.push({name,command,backendSeconds:request.seconds,clickToAcceptedDisplaySeconds:(Date.now()-start)/1000});};
  const generate=async name=>{const count=report.requests.length,start=Date.now();await click('Compare executable strategies');await until(()=>report.requests.slice(count).some(x=>x.command==='generateStrategies'),'strategy result');await ready();timing(name,'generateStrategies',start,count);const study=last.strategyPlanning;assert(study?.publicPlan);assert.equal(last.view.stateId,report.baselineStateId);fs.writeFileSync(path.join(output,name+'.json'),JSON.stringify(study,null,2));return study;};
  const preview=async(candidate,label,phase='operate')=>{await click('Select '+label);await set('Strategy preview checkpoint',phase);const count=report.requests.length,start=Date.now();await click('Preview strategy checkpoint');await until(()=>report.requests.slice(count).some(x=>x.command==='previewStrategy'),'strategy preview');await ready();timing(candidate+'-'+phase,'previewStrategy',start,count);assert.equal(last.approachPreview?.candidateId,candidate);assert.equal(last.approachPreview?.phaseId,phase);assert(last.approachPreview.receipt.nativeMatchesPublicPrediction&&last.approachPreview.receipt.withdrawalComplete);assert.equal(last.view.stateId,report.baselineStateId);return last.approachPreview;};
  const assess=async(offset,name)=>{await set('Execution command offset mm',offset);const count=report.requests.length,start=Date.now();await click('Assess declared execution offsets');await until(()=>report.requests.slice(count).some(x=>x.command==='analyzeStrategySensitivity'),'execution-offset study');await ready();timing(name,'analyzeStrategySensitivity',start,count);const item=last.strategyPlanning.executionSensitivity;assert(item?.study.nativeReplayVerified);assert.equal(item.study.assessment.controls.offsetMm,offset);assert.equal(item.study.assessment.scenarioCount,offset?7:1);assert.equal(item.study.assessment.calibratedUncertainty.supported,false);assert.equal(last.view.stateId,report.baselineStateId);fs.writeFileSync(path.join(output,name+'.json'),JSON.stringify(item,null,2));return item.study.assessment;};
  try{
   await until(()=>js(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`),'case load');await click('Rehearse');await ready();
   if(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify saved replay')`)){await click('Verify saved replay');await ready();}
   report.baselineStateId=last.view.stateId;report.caseHash=last.caseHash;report.sourceSceneId=last.view.scene.sceneId;
   if(process.env.STRATEGY_REOPEN_ONLY==='1'){
    assert.equal(last.view.stateId,process.env.STRATEGY_EXPECTED_STATE);assert(last.strategyPlanning?.publicPlan.publicPlanSeal);report.checks.push('Fresh hidden process verifies saved comparison and exact native state');
    if(process.env.STRATEGY_REOPEN_SENSITIVITY==='1'){
     assert(last.strategyPlanning.executionSensitivity?.study.nativeReplayVerified);await click('Compare two-tool strategies','summary');await click('Execution-offset sensitivity','summary');
     const assessment=last.strategyPlanning.executionSensitivity.study.assessment;assert.equal(assessment.scenarioCount,7);assert.equal(assessment.policies.allDeclaredScenarios,'stop');assert(await js(`!!document.querySelector('[aria-label="Execution sensitivity decision"]')`));
     await click('Select result for all tested offsets');assert(await js(`document.querySelector('[data-strategy-candidate="stop"]').getAttribute('aria-selected')==='true'`));await capture('reopened-execution-offsets','.strategy-sensitivity');
     await click('Save');await until(()=>fs.existsSync(saved),'reopened sensitivity save');report.saved=saved;report.savedStateId=last.view.stateId;report.checks.push('Fresh process replays saved finite-offset studies and restores the scenario-based STOP selection');
    }
    if(process.env.STRATEGY_REOPEN_CONTEXT==='1'){
     await click('Compare two-tool strategies','summary');await click('Acknowledge two-tool research scope','input[type=checkbox]');
     if(await js(`document.querySelector('[aria-label="Require functional evidence for strategies"]').checked`))await click('Require functional evidence for strategies','input[type=checkbox]');
     const option=last.strategyPlanningAvailability.contextOptions.find(x=>x.labelValue===2);assert(option);await set('Anatomical context layer',option.layerId);await set('Context margin mm',3);await set('Context near volume limit mm3',0);
     const study=await generate('reopened-context-limit-stop');assert.equal(study.publicPlan.decision.reason,'ANATOMICAL_CONTEXT_LIMIT');const point=study.publicPlan.rejections.find(x=>x.anatomicalEncounter?.nearPositionsMm?.length).anatomicalEncounter.nearPositionsMm[0];
     await click('Show supplied context in linked views');await until(()=>js(`document.querySelector('.rl-viewer-statusbar')?.textContent.includes(${JSON.stringify(point.map(x=>x.toFixed(1)).join(' / '))})`),'linked source context coordinates');report.linkedContextRasMm=point;assert.equal(last.view.stateId,report.baselineStateId);await capture('linked-source-context','[aria-label="Source context refusals"]');
     await click('Save');await until(()=>fs.existsSync(saved),'native spatial save');report.saved=saved;report.savedStateId=last.view.stateId;report.checks.push('A source-bound representative context cell drives the linked RAS image cursor without altering native state');
    }
   }else if(report.expected==='missing-target'){
    assert.equal(last.strategyPlanningAvailability.available,false);assert.match(last.strategyPlanningAvailability.reason,/tumor target is required/);
    assert.equal(await js(`[...document.querySelectorAll('button')].some(b=>['Compare executable strategies','Assess declared execution offsets'].includes(b.textContent.trim()))`),false);await capture('missing-target');
   }else{
    assert(last.strategyPlanningAvailability.available);await click('Compare two-tool strategies','summary');await click('Acknowledge two-tool research scope','input[type=checkbox]');
    let study=await generate('default-comparison');report.defaultPublicPlanSeal=study.publicPlan.publicPlanSeal;
    if(process.env.STRATEGY_CONTEXT==='1'){
     const context=last.strategyPlanningAvailability.contextOptions.find(x=>x.labelValue===2);assert(context);report.contextLayerId=context.layerId;
     await set('Anatomical context layer',context.layerId);await set('Context margin mm',0);await set('Context near volume limit mm3',0);
     study=await generate('context-zero-margin');assert.equal(study.schema,'resectionlab.two-tool-comparison.v2');assert.equal(study.contextBinding.layerId,context.layerId);assert.equal(study.contextBinding.reviewStatus,context.reviewStatus);assert.equal(study.publicPlan.publicObservation.evaluationReferencesExposed,false);
     if(process.env.STRATEGY_CONTEXT_BINDING){assert.equal(study.contextBinding.schema,process.env.STRATEGY_CONTEXT_BINDING);if(study.contextBinding.schema==='resectionlab.source-context-binding.v2'){assert.equal(typeof study.contextBinding.eligibility.sourceSeries,'string');assert(study.contextBinding.eligibility.sourceSeries);assert.deepEqual(study.contextBinding.eligibility,{schema:'resectionlab.native-context-eligibility.v1',rule:'observed-lesion-context-sharing-verified-categorical-source-v1',nativeKind:'lesion-context',nativeRole:'context',provenanceKind:'observed_source_annotation',sourceSeries:study.contextBinding.eligibility.sourceSeries});}}
     if(report.expected==='paired'){
      assert.equal(study.publicPlan.decision.operation,'COMPARE');assert(study.publicPlan.alternatives.every(x=>x.anatomicalEncounter.nearMm3===0));
      await set('Context margin mm',3);const stopped=await generate('context-limit-stop');assert.equal(stopped.publicPlan.decision.reason,'ANATOMICAL_CONTEXT_LIMIT');assert.equal(stopped.publicPlan.chosenPair,null);assert(stopped.publicPlan.rejections.some(x=>x.anatomicalEncounter?.nearMm3>0));assert(await js(`!!document.querySelector('[aria-label="Source context refusals"]')`));await capture('context-stop-markers');
      await set('Context near volume limit mm3',2);const sequential=await generate('context-sequential-limit-stop');assert.equal(sequential.publicPlan.decision.reason,'ANATOMICAL_CONTEXT_LIMIT');assert.equal(sequential.publicPlan.chosenPair.route.id,'approach-3');assert(sequential.publicPlan.alternatives.slice(0,2).every(x=>!x.feasible&&x.anatomicalEncounter.nearMm3===3));
      await click('Select Longer contact');assert(await js(`[...document.querySelectorAll('button')].find(x=>x.textContent.trim()==='Preview strategy checkpoint').disabled`));report.contextSequentialRevision=sequential.revision;await capture('context-sequential-stop','[aria-label="Selected two-tool strategy"]');
      await set('Context near volume limit mm3',1000);study=await generate('context-allowed');assert.equal(study.publicPlan.decision.operation,'COMPARE');assert(study.publicPlan.alternatives.slice(0,2).every(x=>x.anatomicalEncounter.nearMm3===3&&x.anatomicalEncounter.newRemovedContextMm3===0&&x.anatomicalEncounter.baselineRemovedContextMm3===1036));
      report.contextRevision=study.revision;await capture('context-comparison','[aria-label="Anatomical encounter consequence"]');
     }else{
      assert.equal(study.publicPlan.decision.reason,stoppedReason);await capture('context-no-access');
      if(report.expected==='no-sequence'){
       await set('Context margin mm',3);const limited=await generate('context-primary-limit-stop');
       assert.equal(limited.publicPlan.decision.reason,'ANATOMICAL_CONTEXT_LIMIT');assert.equal(limited.publicPlan.chosenPair,null);
       assert(limited.publicPlan.rejections.some(x=>x.reason==='ANATOMICAL_CONTEXT_LIMIT:primary-access'&&x.anatomicalEncounter?.nearMm3>0));
       await capture('context-primary-limit-stop');await set('Context margin mm',0);study=await generate('context-restored-no-sequence');assert.equal(study.publicPlan.decision.reason,stoppedReason);
       report.checks.push('Changing the supplied context margin excludes primary access; restoring it retains the independent two-tool sequence refusal');
      }
     }
     report.checks.push('Exact supplied context selection propagates through public observation, source-linked limit and native comparison; no functional localization or injury probability');
    }
    if(report.expected==='paired'){
     assert.deepEqual(study.publicPlan.alternatives.map(x=>x.id),['short','long','stop']);assert.equal(study.publicPlan.decision.selectedId,'long');
     const [short,long]=study.publicPlan.alternatives;assert.equal(short.summary.additionalTargetMm3,0);assert.equal(short.summary.cusaPendingMm3,.5);assert.equal(long.summary.additionalTargetMm3,2);
     assert.equal(short.reassessment.probeStatus,'rejected');assert.equal(long.reassessment.probeStatus,'accepted');
     let p=await preview('short','Short contact');assert(p.view.slots.right.active);await capture('short-contact-preview');
     p=await preview('long','Longer contact');assert(p.view.slots.right.active);await capture('long-contact-preview');
     await preview('long','Longer contact','reassess');await capture('long-reassessment-preview');
     const count=report.requests.length;await click('Install strategy branch');await until(()=>report.requests.slice(count).some(x=>x.command==='applyStrategy'),'installed branch');await ready();
     assert.notEqual(last.view.stateId,report.baselineStateId);assert(Object.values(last.view.slots).every(s=>!s.active&&!s.inserted));report.longBranch=last.branchId;report.longStateId=last.view.stateId;await capture('installed-long-branch');
     await click('Open strategy checkpoint');await ready();assert.equal(last.view.stateId,report.baselineStateId);
     if(!await js(`document.querySelector('[aria-label="Acknowledge two-tool research scope"]').checked`))await click('Acknowledge two-tool research scope','input[type=checkbox]');
     await set('Short contact seconds',.75);study=await generate('modified-duration-comparison');assert.equal(study.publicPlan.decision.selectedId,'short');assert.equal(study.publicPlan.alternatives[0].summary.additionalTargetMm3,1);
     await preview('short','Short contact');await click('Install strategy branch');await ready();report.shortBranch=last.branchId;report.shortStateId=last.view.stateId;assert.notEqual(report.shortStateId,report.longStateId);
     await click('Open strategy checkpoint');await ready();assert.equal(last.view.stateId,report.baselineStateId);
     if(!await js(`document.querySelector('[aria-label="Acknowledge two-tool research scope"]').checked`))await click('Acknowledge two-tool research scope','input[type=checkbox]');
     await click('Target and shared entry patch','summary');await set('Shared entry patch radius',.5);
     const restricted=await generate('restricted-shared-entry');assert.equal(restricted.publicPlan.decision.operation,'STOP');assert.equal(restricted.publicPlan.chosenPair,null);await capture('restricted-entry-stop');
     await set('Shared entry patch radius',study.publicPlan.constraints.geometry.entry.radiusMm);
     report.checks.push('Actual controls change contact duration, removed target and preferred complete sequence; two distinct branches installed without new non-target removal');
    }else{
     assert.equal(study.publicPlan.decision.operation,'STOP');assert.equal(study.publicPlan.alternatives.length,1);assert.equal(study.publicPlan.decision.reason,stoppedReason);await capture('no-access-stop');
    }
    await click('Require functional evidence for strategies','input[type=checkbox]');study=await generate('functional-evidence-stop');assert.equal(study.publicPlan.decision.operation,'STOP');assert.equal(study.publicPlan.decision.reason,'REQUIRED_EVIDENCE_UNAVAILABLE');assert.equal(study.publicPlan.alternatives.length,1);await capture('functional-stop');
    if(process.env.STRATEGY_SENSITIVITY==='1'){
     await click('Require functional evidence for strategies','input[type=checkbox]');study=await generate('sensitivity-conditional-comparison');
     await click('Execution-offset sensitivity','summary');const zero=await assess(0,'execution-offset-zero');assert(Object.values(zero.policies).every(id=>id===study.publicPlan.decision.selectedId));
     const nonzero=await assess(1,'execution-offset-one');assert.equal(nonzero.policies.allDeclaredScenarios,'stop');
     await click('Select result for all tested offsets');assert(await js(`document.querySelector('[data-strategy-candidate="stop"]').getAttribute('aria-selected')==='true'`));await capture('execution-offset-decision','.strategy-sensitivity');
     if(report.expected==='paired'){
      const candidate=nonzero.candidates.find(c=>c.id==='long'),failed=candidate.outcomes.find(o=>o.scenarioId==='R+');assert(failed.failure);const point=failed.actionProgram[failed.failure.actionIndex].pose.position;
      await click('Show failed commanded pose long R+');await until(()=>js(`document.querySelector('.rl-viewer-statusbar')?.textContent.includes(${JSON.stringify(point.map(x=>x.toFixed(1)).join(' / '))})`),'failed command RAS cursor');assert.equal(last.view.stateId,report.baselineStateId);report.failedCommandRasMm=point;await capture('execution-failed-command','.strategy-sensitivity');
     }
     report.sensitivityPolicies=nonzero.policies;report.sensitivityScenarioCount=nonzero.scenarioCount;report.checks.push('Zero offset reproduces nominal preference; finite coherent offsets produce native-verified policy decisions without changing the baseline');
    }
    await click('Save');await until(()=>fs.existsSync(saved),'native save');report.saved=saved;report.savedStateId=last.view.stateId;
   }
   assert.equal(sha(input),original);assert(!report.windowEverVisible&&!window.isVisible());report.sourcePreserved=true;report.passed=true;
  }catch(error){report.error=String(error);report.body=await js('document.body.innerText').catch(()=>null);fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());console.error(error);}
  fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,error:report.error,saved:report.saved}));app.exit(report.passed?0:1);
 });
});
require('./main.cjs');
