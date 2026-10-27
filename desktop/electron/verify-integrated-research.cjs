'use strict';
// Actual Electron comparison/learned trajectory playback, with native replay.
const {app,dialog}=require('electron');
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..'),input=path.resolve(process.env.INTEGRATED_RESEARCH_CASE||''),output=path.resolve(process.env.INTEGRATED_RESEARCH_OUTPUT||''),mode=process.env.INTEGRATED_RESEARCH_MODE;
if(!['access','learned','cusa','cusa-execute','cusa-cancel','sector','sector-policy','sector-execute','joint','staged','frozen-assessment'].includes(mode))throw Error('Set a supported INTEGRATED_RESEARCH_MODE');
for(const p of [input,output])if(!p.startsWith(path.join(root,'testcases')+path.sep))throw Error('Input/output must be under testcases/');
if(fs.existsSync(output))throw Error('Preserve previous evidence; use a new output directory');
fs.mkdirSync(output,{recursive:true});
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex'),sourceHash=hash(input);
const userData=fs.mkdtempSync(path.join(os.tmpdir(),'rslab-integrated-test-'));app.setPath('userData',userData);
process.argv.push('--case',input,'--automation-hidden');
const saved=path.join(output,'desktop-saved.ressectionlab');
const checkpoint=process.env.INTEGRATED_RESEARCH_CHECKPOINT;
if((mode?.startsWith('cusa-')||mode==='sector-execute')&&(!checkpoint||!path.resolve(checkpoint).startsWith(path.join(root,'testcases')+path.sep)))throw Error('Choose an existing frozen checkpoint under testcases/');
const checkpointHash=checkpoint?hash(checkpoint):null;
const executionRoot=path.join(root,mode==='sector-execute'?'testcases/outputs/sector-desktop':'testcases/outputs/cusa-desktop');
const previousExecutions=new Set(fs.existsSync(executionRoot)?fs.readdirSync(executionRoot):[]);
const newExecutionForScene=scene=>fs.readdirSync(executionRoot).filter(name=>!previousExecutions.has(name)).filter(name=>{
  const receiptPath=path.join(executionRoot,name,'execution/receipt.json');
  if(!fs.existsSync(receiptPath))return false;
  const receipt=JSON.parse(fs.readFileSync(receiptPath,'utf8'));
  return (receipt.execution?.sourceSceneId??receipt.prefix?.sceneId)===scene;
});
dialog.showSaveDialog=async()=>({canceled:false,filePath:saved});
dialog.showOpenDialog=async(_window,options)=>({canceled:false,filePaths:[options?.title?.startsWith('Choose a compatible frozen ')?checkpoint:saved]});
const report={schema:'resectionlab.integrated-research-electron.v1',input,mode,windowMode:'hidden',windowEverVisible:false,checks:[],errors:[],screenshots:[],startedUtc:new Date().toISOString()};
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
app.on('browser-window-created',(_event,window)=>{
  window.on('show',()=>{report.windowEverVisible=true;window.hide();});
  window.webContents.on('console-message',event=>{if(event.level==='error')report.errors.push(event.message);});
  window.webContents.once('did-finish-load',async()=>{
    window.webContents.setBackgroundThrottling(false);window.setSize(1500,1000);
    const js=code=>window.webContents.executeJavaScript(code,true);
    const wait=async(expression,seconds=240)=>{const started=Date.now();let logged=0;while(Date.now()-started<seconds*1000){if(report.errors.length)throw Error('Renderer error: '+report.errors.join('; '));if(await js(expression))return;const error=await js(`document.querySelector('.surgical-error')?.textContent??''`);if(/^Error:|^.*SURGICAL_[A-Z_]+:/.test(error))throw Error('Desktop action failed: '+error);if(await js(`document.body.innerText.includes('Staged strategy evidence could not be bound')`))throw Error('Staged native result could not be bound in the desktop');if(Date.now()-logged>10000){logged=Date.now();fs.writeFileSync(path.join(output,'progress.json'),JSON.stringify({waiting:expression,elapsedSeconds:(Date.now()-started)/1000,body:await js('document.body.innerText')},null,2));}await delay(250);}throw Error('Timeout: '+expression);};
    const click=async label=>{await js(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(label)}||b.getAttribute('aria-label')===${JSON.stringify(label)}||b.querySelector('span')?.textContent.trim()===${JSON.stringify(label)});if(!b||b.matches(':disabled'))throw Error('Missing enabled button '+${JSON.stringify(label)});b.click();})()`);await delay(150);};
    const ready=()=>wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')&&document.querySelector('.rl-viewer').dataset.surfacesPending==='0'&&!document.querySelector('.surgical-panel').innerText.includes('Computing instrument action')`);
    const capture=async name=>{await delay(300);await ready();await delay(200);fs.writeFileSync(path.join(output,name+'.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());report.screenshots.push(name+'.png');};
    try{
      await wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`);await click('Rehearse');await wait(`!!document.querySelector('.surgical-panel')`);
      await wait(`!!document.querySelector('.research-planning')||!!document.querySelector('.learned-execution')`);
      if(mode==='access'&&await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify replay for planning'&&!b.matches(':disabled'))`)){
        await click('Verify replay for planning');
        await wait(`![...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify replay for planning')`);
      }
      await ready();report.checks.push(mode==='access'?'Actual native replay admitted the saved source and actions':mode==='learned'?'Imported learned endpoint retains read-only replay authority':'Fresh case retains source-bound comparison and learned branches');
      if(process.env.INTEGRATED_RESEARCH_SOURCE_BRANCH){
        const branch=process.env.INTEGRATED_RESEARCH_SOURCE_BRANCH;
        await js(`(()=>{const select=document.querySelector('[aria-label="Surgical branch"]');if(![...select.options].some(option=>option.value===${JSON.stringify(branch)}))throw Error('Source branch unavailable');select.value=${JSON.stringify(branch)};select.dispatchEvent(new Event('change',{bubbles:true}));})()`);
        await wait(`document.querySelector('[aria-label="Surgical branch"]').value===${JSON.stringify(branch)}&&Number(document.querySelector('[aria-label="Surgical timeline"]').value)===0`);
        await ready();report.checks.push('Returned through the actual desktop to the intact source branch');
      }
      if(mode==='cusa-cancel'){
        await wait(`!!document.querySelector('.frozen-cusa-execution')`);
        await click('Run frozen CUSA checkpoint');
        await wait(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Cancel frozen policy execution')`);
        await click('Cancel frozen policy execution');
        await wait(`document.body.innerText.includes('Operation cancelled')&&[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Run frozen CUSA checkpoint'&&!b.matches(':disabled'))`);
        assert.equal(hash(input),sourceHash);assert.equal(hash(checkpoint),checkpointHash);
        assert.equal(await js(`document.querySelectorAll('.patient-cusa-learning').length`),0);
        report.checks.push('User cancellation stops new inference and retains the source without admitting a learned branch');
        await capture('01-cancelled-source-preserved');report.sourcePreserved=true;report.passed=report.errors.length===0&&!report.windowEverVisible;
        report.finishedUtc=new Date().toISOString();fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));app.exit(report.passed?0:1);return;
      }else if(mode==='sector-policy'||mode==='sector-execute'){
        if(mode==='sector-execute'){
          await wait(`!!document.querySelector('.frozen-cusa-execution')`);
          const sourceScene=await js(`document.querySelector('.frozen-cusa-execution').dataset.sourceScene`);
          if(process.env.INTEGRATED_RESEARCH_ACCESS_METHOD){
            const method=process.env.INTEGRATED_RESEARCH_ACCESS_METHOD;
            await js(`(()=>{const select=document.querySelector('[aria-label="Frozen policy access approach"]');const option=[...select.options].find(option=>option.value.endsWith(${JSON.stringify(method)}));if(!option)throw Error('Requested source approach unavailable');select.value=option.value;select.dispatchEvent(new Event('change',{bubbles:true}));})()`);
            report.requestedAccessMethod=method;
          }
          await click('Run frozen sector checkpoint');
          await wait(`document.body.innerText.includes('New frozen policy episode')&&!!document.querySelector('.learned-sector-execution')`,180);
          const created=newExecutionForScene(sourceScene);assert.equal(created.length,1);
          report.executionReceipt=path.join(executionRoot,created[0],'execution/receipt.json');
          const receipt=JSON.parse(fs.readFileSync(report.executionReceipt,'utf8'));
          assert.equal(receipt.status,'evaluated');assert.equal(receipt.optimizerUpdates,0);
          assert.equal(receipt.exactReplay.exactNativeReplay,true);assert.equal(receipt.derived.reopenedExact,true);
          report.newExecution={status:receipt.execution.status,taskGateMet:receipt.execution.taskGateMet,
            cleanupComplete:receipt.execution.cleanupComplete,deltaTargetMm3:receipt.execution.deltaTargetMm3,
            deltaNonTargetMm3:receipt.execution.deltaNonTargetMm3,perSectorRemovedMm3:receipt.execution.perSectorRemovedMm3,
            decisions:receipt.execution.decisions.map(row=>row.optionId),budget:receipt.budget};
          assert.equal(hash(checkpoint),checkpointHash);report.checks.push('Actual desktop hierarchical frozen inference ran with zero optimizer updates');
        }
        await wait(`!!document.querySelector('.learned-sector-execution')`);
        report.learning=await js(`document.querySelector('.learned-sector-execution').innerText`);
        if(process.env.INTEGRATED_RESEARCH_EXPECT_EXIT_CHECKS==='1'){
          report.policyDecisionEvidence=await js(`[...document.querySelectorAll('.learned-sector-decision-evidence > li')].map(item=>item.textContent)`);
          assert(report.policyDecisionEvidence.some(text=>text.includes('exit check')),'Saved actor decisions retain native exit-check evidence');
          report.checks.push('Policy observation identities, exit-check refusals and discarded probe costs survive native export');
        }
        const target=await js(`(()=>{const b=document.querySelector('[data-learned-sector-branch]');return{branchId:b.dataset.learnedSectorBranch,start:Number(b.dataset.learnedSectorStart),end:Number(b.dataset.learnedSectorEnd)}})()`);
        assert(target.end>0);await js(`document.querySelector('.learned-sector-execution').scrollIntoView({block:'start'})`);
        await capture('01-learned-sector-evidence');await click('Inspect PPO sector policy');
        await wait(`document.querySelector('[aria-label="Surgical branch"]').value===${JSON.stringify(target.branchId)}&&Number(document.querySelector('[aria-label="Surgical timeline"]').value)===0`);
        await wait(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Inspect PPO sector policy'&&!b.matches(':disabled'))`);
        // Start the recorded sector interval after its actual corridor actions.
        await js(`(()=>{const s=document.querySelector('[aria-label="Surgical timeline"]');const set=Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set;set.call(s,${target.start});s.dispatchEvent(new Event('input',{bubbles:true}));s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
        await wait(`Number(document.querySelector('[aria-label="Surgical timeline"]').value)===${target.start}`);
      }else if(mode==='staged'){
        await wait(`!!document.querySelector('.staged-strategies')`);
        await js(`document.querySelector('.staged-strategies').open=true`);
        if(process.env.INTEGRATED_RESEARCH_GENERATE==='1'){
          await js(`document.querySelector('[aria-label="Acknowledge staged source assumptions"]').click()`);
          await click('Generate staged source strategies');
          await wait(`!!document.querySelector('[aria-label="Staged whole approach summaries"] > article')`,300);
          report.checks.push('Actual desktop source planning executed native staged alternatives');
        }
        await wait(`!!document.querySelector('[aria-label="Staged whole approach summaries"] > article')`);
        report.staged=await js(`(()=>{const panel=document.querySelector('.staged-strategies');const rows=[...panel.querySelectorAll('[aria-label="Staged whole approach summaries"] > article')];const count=row=>Number(row.textContent.match(/(\\d+) admitted work stages/)?.[1]??0);const selected=${process.env.INTEGRATED_RESEARCH_EXPECT_CLIPPED==='1'}?rows.find(row=>row.querySelector('[data-staged-execution="checked_collision_prefix"]')):[...rows].sort((a,b)=>count(b)-count(a))[0];if(!selected)throw Error('Required staged native branch unavailable');const index=rows.indexOf(selected);const stages=[...selected.querySelectorAll('ol > li')];const firstClip=stages.findIndex(row=>row.dataset.stagedExecution==='checked_collision_prefix');selected.querySelector('details').open=true;const positions=[...selected.querySelectorAll('ol [data-staged-position]')].map(button=>Number(button.dataset.stagedPosition));return{text:panel.innerText,rows:rows.length,index,admittedStages:count(selected),clippedStages:stages.filter(row=>row.dataset.stagedExecution==='checked_collision_prefix').length,selectedText:selected.innerText,stagePositions:${process.env.INTEGRATED_RESEARCH_EXPECT_CLIPPED==='1'}?positions.slice(firstClip):positions}})()`);
        assert.equal(report.staged.rows,3,'Three staged source directions remain inspectable');
        if(process.env.INTEGRATED_RESEARCH_EXPECT_CLIPPED==='1')assert(report.staged.clippedStages>0&&report.staged.selectedText.includes('shortened command passed fresh native checks'),'Collision shortening is distinct from reaching the original requested waypoint');
        assert(report.staged.text.includes('original 48-action')&&report.staged.text.includes('larger staged task'),'Larger task and old gate are displayed separately');
        await capture('01-staged-native-evidence');
        await js(`document.querySelectorAll('[aria-label="Staged whole approach summaries"] > article')[${report.staged.index}].querySelector('button').click()`);
        await wait(`document.querySelector('[aria-label="Surgical branch"]').value.startsWith('staged-')&&Number(document.querySelector('[aria-label="Surgical timeline"]').value)===0`);
        if(report.staged.stagePositions.length){
          const start=Math.max(0,report.staged.stagePositions[0]-2);
          await js(`(()=>{const s=document.querySelector('[aria-label="Surgical timeline"]');const set=Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set;set.call(s,${start});s.dispatchEvent(new Event('input',{bubbles:true}));s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
          await wait(`Number(document.querySelector('[aria-label="Surgical timeline"]').value)===${start}`);
          report.staged.playbackGoal=report.staged.stagePositions[1]??report.staged.stagePositions[0];
          await wait(`[...document.querySelectorAll('button')].some(button=>button.textContent.trim()==='Close tool view'&&!button.matches(':disabled'))`);
          report.staged.targetBefore=await js(`document.querySelector('[data-surgical-metric="targetRemovedMm3"]').textContent`);
          await click('Close tool view');
          await js(`(()=>{const label=[...document.querySelectorAll('[aria-label="Close instrument view"] label')].find(label=>label.textContent.includes('Follow accepted tool'));const box=label?.querySelector('input');if(box&&!box.checked)box.click()})()`);
          await capture('03-staged-close-start');
        }
        report.checks.push('Whole approach effects, retained stages, refusals and resume evidence are inspectable');
      }else if(mode==='frozen-assessment'){
        await wait(`!!document.querySelector('[aria-label="Frozen assessment sector outcomes"] [data-frozen-sector-branch]')`);
        report.learning=await js(`document.querySelector('.learned-sector-execution').innerText`);
        assert(report.learning.includes('saved native executions')&&report.learning.includes('does not run new inference'),'Assessment playback remains distinct from inference');
        const target=await js(`(()=>{const b=document.querySelector('[data-frozen-sector-branch]');return{branchId:b.dataset.frozenSectorBranch,start:Number(b.dataset.frozenSectorStart),end:Number(b.dataset.frozenSectorEnd)}})()`);
        report.frozenAssessment={...target,inspectionOnly:true};
        await js(`document.querySelector('.learned-sector-execution').scrollIntoView({block:'start'})`);
        await capture('01-frozen-assessment-evidence');
        await js(`document.querySelector('[data-frozen-sector-branch]').click()`);
        await wait(`document.querySelector('[aria-label="Surgical branch"]').value===${JSON.stringify(target.branchId)}&&Number(document.querySelector('[aria-label="Surgical timeline"]').value)===0`);
        report.checks.push('Frozen assessment source, checkpoint and native decisions are inspectable without new inference');
      }else if(mode==='joint'){
        await wait(`!!document.querySelector('.joint-strategies')`);
        await js(`document.querySelector('.joint-strategies').open=true`);
        if(process.env.INTEGRATED_RESEARCH_GENERATE==='1'){
          await wait(`!!document.querySelector('[aria-label="Acknowledge joint source assumptions"]')`);
          await js(`document.querySelector('[aria-label="Acknowledge joint source assumptions"]').click()`);
          await wait(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Generate joint source strategies'&&!b.matches(':disabled'))`);
          await click('Generate joint source strategies');
          await wait(`document.querySelectorAll('[aria-label="Joint sector executions"] [data-joint-branch]').length>0`,180);
          report.checks.push('Actual desktop joint planning generated native alternatives from the intact classic source');
        }
        await wait(`document.querySelectorAll('[aria-label="Joint sector executions"] [data-joint-branch]').length>0`);
        const wholeApproach=await js(`(()=>{const table=document.querySelector('[aria-label="Whole approach joint and classic reference outcomes"]');return{rows:table?.querySelectorAll('tbody tr').length??0,text:table?.textContent??'',summaryCards:document.querySelectorAll('[aria-label="Whole approach summaries"] > article').length}})()`);
        assert.equal(wholeApproach.rows,4,'All three joint endpoints and the strongest classic reference expose whole approach effects');
        assert.equal(wholeApproach.summaryCards,4,'Narrow inspector exposes readable whole approach summaries');
        assert(wholeApproach.text.includes('Residual')&&wholeApproach.text.toLowerCase().includes('non-target'),'Whole approach comparison includes residual and cumulative non-target effects');
        const selection=await js(`(()=>{const rows=[...document.querySelectorAll('[aria-label="Joint sector executions"] > div')];const row=rows.find(row=>row.textContent.includes('Task gate passed'))??rows[0];const buttons=[...row.querySelectorAll('[data-joint-branch]')];return{branchId:buttons[0].dataset.jointBranch,start:Number(buttons[0].dataset.jointPosition),end:Number(buttons[1].dataset.jointPosition),gate:row.textContent.includes('Task gate passed')}})()`);
        await js(`(()=>{const details=[...document.querySelectorAll('.joint-strategies details')].find(d=>d.querySelector('summary')?.textContent.startsWith('Original classic comparison'));details.open=true})()`);
        await wait(`document.querySelectorAll('[aria-label="Original access comparison"] [data-joint-branch]').length>0`);
        const baseline=await js(`({access:document.querySelectorAll('[aria-label="Original access comparison"] [data-joint-branch]').length/2,sectors:document.querySelectorAll('[aria-label="Original sector comparison"] [data-joint-branch]').length/2})`);
        assert.equal(baseline.access,3);assert.equal(baseline.sectors,12);
        report.joint={selection,baseline,wholeApproach,text:await js(`document.querySelector('.joint-strategies').innerText`)};
        await js(`document.querySelector('.joint-strategies').scrollIntoView({block:'start'})`);
        await capture('01-joint-and-classic-evidence');
        await js(`(()=>{const button=[...document.querySelectorAll('[aria-label="Joint sector executions"] [data-joint-branch]')].find(b=>b.dataset.jointBranch===${JSON.stringify(selection.branchId)}&&Number(b.dataset.jointPosition)===${selection.start});if(!button||button.disabled)throw Error('Joint native start unavailable');button.click()})()`);
        await wait(`document.querySelector('[aria-label="Surgical branch"]').value===${JSON.stringify(selection.branchId)}&&Number(document.querySelector('[aria-label="Surgical timeline"]').value)===${selection.start}`);
        report.checks.push('Joint source-ranked outcomes and all archived classic comparisons are inspectable');
      }else if(mode==='sector'){
        await wait(`!!document.querySelector('.sector-strategies')`);
        if(process.env.INTEGRATED_RESEARCH_GENERATE==='1'){
          await js(`document.querySelector('.sector-strategies input[type="checkbox"]').click()`);
          await click('Generate sector and hand strategies');
        }
        await wait(`document.querySelectorAll('[aria-label="Sector strategy outcomes"] tbody tr').length>0`,180);
        const selection=await js(`(()=>{const buttons=[...document.querySelectorAll('[data-sector-branch]')];const b=buttons.find(b=>b.dataset.sectorGate==='true'&&b.dataset.sectorHand==='right'&&b.dataset.sectorOrder==='far,near')??buttons.find(b=>b.dataset.sectorCleanup==='true'&&b.dataset.sectorOrder==='near,far')??buttons[0];return{branchId:b.dataset.sectorBranch,start:Number(b.dataset.sectorStart),end:Number(b.dataset.sectorEnd),hand:b.dataset.sectorHand,gate:b.dataset.sectorGate}})()`);
        report.sectors={comparison:await js(`document.querySelector('[aria-label="Sector strategy outcomes"]').innerText`),selection};
        if(process.env.INTEGRATED_RESEARCH_EXPECT_EXIT_CHECKS==='1'){
          report.sectors.exitChecks=await js(`[...document.querySelectorAll('.sector-exit-checks')].map(item=>item.textContent)`);
          assert(report.sectors.exitChecks.length>0,'Comparison retains prospective native exit checks');
          assert(report.sectors.exitChecks.some(text=>text.includes('discarded probe')),'Comparison reports discarded native work');
        }
        await js(`document.querySelector('.sector-strategies').scrollIntoView({block:'start'})`);
        await capture('01-sector-comparison');
        await js(`document.querySelector('[data-sector-branch='+CSS.escape(${JSON.stringify(selection.branchId)})+']').click()`);
        await wait(`document.querySelector('[aria-label="Surgical branch"]').value===${JSON.stringify(selection.branchId)}&&Number(document.querySelector('[aria-label="Surgical timeline"]').value)===${selection.start}`);
        report.checks.push('Real source-sector order and CUSA hand variants are visible and inspectable');
      }else if(mode==='access'){
        await js(`document.querySelector('.research-planning').open=true;document.querySelector('.research-planning').scrollIntoView({block:'start'})`);
        assert.equal(await js(`document.querySelectorAll('[aria-label="Access approach outcomes"] tbody tr').length`),3);
        assert(await js(`document.querySelector('.research-planning').innerText.includes('skull entry and craniotomy remain unsupported')`));
        report.comparison=await js(`document.querySelector('[aria-label="Access approach outcomes"]').innerText`);
        report.checks.push('Three distinct source routes and explicit unsupported skull gate are visible');await capture('01-comparison');
        await js(`(()=>{const b=document.querySelector('[aria-label="Access approach outcomes"] tbody button');if(!b||b.matches(':disabled'))throw Error('Route selection disabled');b.click();})()`);
        await wait(`document.querySelector('[aria-label="Surgical branch"]').value.includes('superior-access')`);
      }else if(mode==='learned'){
        const text=await js(`document.querySelector('.learned-execution').innerText`);
        assert(text.includes('64 actual PPO option transitions'));assert(text.includes('Playback shows recorded authoritative actions'));
        report.learning=text;report.checks.push('PPO training, checkpoint identity and observed decisions are visible');await capture('01-learned-provenance');
        await click('Inspect recorded PPO execution');
        await wait(`Number(document.querySelector('[aria-label="Surgical timeline"]').value)===51`);
      }else{
        if(mode==='cusa-execute'){
          await wait(`!!document.querySelector('.frozen-cusa-execution')`);
          const sourceScene=await js(`document.querySelector('.frozen-cusa-execution').dataset.sourceScene`);
          await js(`(()=>{const s=document.querySelector('[aria-label="Frozen policy access approach"]');s.value='access-superior-access';s.dispatchEvent(new Event('change',{bubbles:true}));})()`);
          await click('Run frozen CUSA checkpoint');
          await wait(`document.body.innerText.includes('New frozen policy episode')&&!!document.querySelector('.patient-cusa-learning')`,180);
          const created=newExecutionForScene(sourceScene);
          assert.equal(created.length,1);
          report.executionReceipt=path.join(executionRoot,created[0],'execution/receipt.json');
          const receipt=JSON.parse(fs.readFileSync(report.executionReceipt,'utf8'));
          assert.equal(receipt.status,'evaluated');assert.equal(receipt.optimizerUpdates,0);
          assert.equal(receipt.exactReplay.exactNativeReplay,true);assert.equal(receipt.derived.reopenedExact,true);
          report.newExecution={status:receipt.execution.status,taskGateMet:receipt.execution.taskGateMet,
            cleanupComplete:receipt.execution.cleanupComplete,deltaTargetMm3:receipt.execution.deltaTargetMm3,
            deltaNonTargetMm3:receipt.execution.deltaNonTargetMm3,decisions:receipt.execution.decisions.map(row=>row.actionId),
            budget:receipt.budget};
          assert.equal(hash(checkpoint),checkpointHash);
          report.checks.push('Desktop ran a frozen actor with new observations, zero optimizer updates and a verified native export');
        }
        await wait(`!!document.querySelector('.patient-cusa-learning')`);
        assert(await js(`document.querySelectorAll('[aria-label="Learned CUSA outcomes"] tbody tr').length>=1`));
        report.learning=await js(`document.querySelector('.patient-cusa-learning').innerText`);
        const target=await js(`(()=>{const b=[...document.querySelectorAll('.patient-cusa-learning button')].find(b=>b.textContent.trim()==='Inspect PPO CUSA execution');if(!b)throw Error('PPO boundary missing');return{branchId:b.dataset.policyBranch,position:Number(b.dataset.policyPosition)}})()`);
        await js(`document.querySelector('.patient-cusa-learning').scrollIntoView({block:'start'})`);
        await capture('01-fresh-learned-provenance');await click('Inspect PPO CUSA execution');
        await wait(`document.querySelector('[aria-label="Surgical branch"]').value===${JSON.stringify(target.branchId)}&&Number(document.querySelector('[aria-label="Surgical timeline"]').value)===${target.position}`);
      }
      await ready();
      const position=await js(`Number(document.querySelector('[aria-label="Surgical timeline"]').value)`);
      const end=await js(`Number(document.querySelector('[aria-label="Surgical timeline"]').max)`);
      const goal=Math.min(mode==='staged'&&report.staged?.playbackGoal?report.staged.playbackGoal:position+4,end);
      if(goal>position){
        await wait(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Play replay'&&!b.matches(':disabled'))`);
        await click('Play replay');
        await wait(`Number(document.querySelector('[aria-label="Surgical timeline"]').value)>=${goal}`,180);
        if(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Pause replay')`))await click('Pause replay');
        await ready();report.playback={startPosition:position,observedPosition:await js(`Number(document.querySelector('[aria-label="Surgical timeline"]').value)`)};
        if(mode==='staged'){
          report.staged.targetAfter=await js(`document.querySelector('[data-surgical-metric="targetRemovedMm3"]').textContent`);
          if(report.staged.stagePositions.length)assert(Number(report.staged.targetAfter)>Number(report.staged.targetBefore),'Recorded work playback changes measured native target removal');
        }
        report.checks.push('Play advances actual recorded instrument/physics frames and pause stops scheduling');await capture('02-authoritative-playback');
      }else{
        assert.equal(mode,'frozen-assessment','Selected policy must contain actual native actions');
        report.playback={startPosition:position,observedPosition:position,skipped:'Frozen actor recorded no native actions'};
        report.checks.push('Zero-action frozen abstention is retained without fabricating playback');
      }
      if(mode==='sector')assert(await js(`document.querySelectorAll('[aria-label="Sector strategy outcomes"] tbody tr').length>0`),'Comparison evidence remains visible during compact playback');
      if(mode==='joint')assert(await js(`!!document.querySelector('.joint-strategies')&&!document.body.innerText.includes('Saved joint strategy evidence could not be bound')`),'Joint evidence remains bound during compact playback');
      if(mode==='staged')assert(await js(`!!document.querySelector('.staged-strategies')&&!document.body.innerText.includes('Staged strategy evidence could not be bound')`),'Staged evidence remains bound during compact playback');
      await wait(`!![...document.querySelectorAll('header button')].find(b=>b.textContent.trim()==='Save'&&!b.matches(':disabled'))`);
      await click('Save');await wait(`![...document.querySelectorAll('header button')].find(b=>b.textContent.trim()==='Save')?.matches(':disabled')`);
      assert(fs.existsSync(saved));report.checks.push('Native desktop save writes the inspected trajectory');
      assert.equal(hash(input),sourceHash);report.sourcePreserved=true;report.savedCase=saved;report.passed=report.errors.length===0&&!report.windowEverVisible;
    }catch(error){report.passed=false;report.failure=error.stack;report.failureText=await js('document.body.innerText');fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());console.error(error.stack);}
    report.finishedUtc=new Date().toISOString();fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));app.exit(report.passed?0:1);
  });
});
app.on('will-quit',()=>fs.rmSync(userData,{recursive:true,force:true}));
require('./main.cjs');
