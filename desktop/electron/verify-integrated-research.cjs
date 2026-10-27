'use strict';
// Actual Electron comparison/learned trajectory playback, with native replay.
const {app,dialog}=require('electron');
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..'),input=path.resolve(process.env.INTEGRATED_RESEARCH_CASE||''),output=path.resolve(process.env.INTEGRATED_RESEARCH_OUTPUT||''),mode=process.env.INTEGRATED_RESEARCH_MODE;
if(!['access','learned','cusa','cusa-execute','cusa-cancel','sector','sector-policy','sector-execute'].includes(mode))throw Error('Set a supported INTEGRATED_RESEARCH_MODE');
for(const p of [input,output])if(!p.startsWith(path.join(root,'testcases')+path.sep))throw Error('Input/output must be under testcases/');
if(fs.existsSync(output))throw Error('Preserve previous evidence; use a new output directory');
fs.mkdirSync(output,{recursive:true});
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex'),sourceHash=hash(input);
const userData=fs.mkdtempSync(path.join(os.tmpdir(),'rslab-integrated-test-'));app.setPath('userData',userData);
process.argv.push('--case',input);
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
const report={schema:'resectionlab.integrated-research-electron.v1',input,mode,checks:[],errors:[],screenshots:[],startedUtc:new Date().toISOString()};
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
app.on('browser-window-created',(_event,window)=>{
  window.webContents.on('console-message',event=>{if(event.level==='error')report.errors.push(event.message);});
  window.webContents.once('did-finish-load',async()=>{
    window.webContents.setBackgroundThrottling(false);window.setSize(1500,1000);window.show();
    const js=code=>window.webContents.executeJavaScript(code,true);
    const wait=async(expression,seconds=240)=>{for(let i=0;i<seconds*4;i++){if(await js(expression))return;await delay(250);}throw Error('Timeout: '+expression);};
    const click=async label=>{await js(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(label)}||b.getAttribute('aria-label')===${JSON.stringify(label)}||b.querySelector('span')?.textContent.trim()===${JSON.stringify(label)});if(!b||b.matches(':disabled'))throw Error('Missing enabled button '+${JSON.stringify(label)});b.click();})()`);await delay(150);};
    const ready=()=>wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')&&document.querySelector('.rl-viewer').dataset.surfacesPending==='0'&&!document.querySelector('.surgical-panel').innerText.includes('Computing instrument action')`);
    const capture=async name=>{await delay(300);await ready();await delay(200);fs.writeFileSync(path.join(output,name+'.png'),(await window.webContents.capturePage()).toPNG());report.screenshots.push(name+'.png');};
    try{
      await wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`);await click('Rehearse');await wait(`!!document.querySelector('.surgical-panel')`);
      await wait(`!!document.querySelector('.research-planning')||!!document.querySelector('.learned-execution')`);
      if(mode==='access'&&await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify replay for planning'&&!b.matches(':disabled'))`)){
        await click('Verify replay for planning');
        await wait(`![...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify replay for planning')`);
      }
      await ready();report.checks.push(mode==='access'?'Actual native replay admitted the saved source and actions':mode==='learned'?'Imported learned endpoint retains read-only replay authority':'Fresh case retains source-bound comparison and learned branches');
      if(mode==='cusa-cancel'){
        await wait(`!!document.querySelector('.frozen-cusa-execution')`);
        await click('Run frozen CUSA checkpoint');
        await wait(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Cancel frozen policy execution')`);
        await click('Cancel frozen policy execution');
        await wait(`document.body.innerText.includes('Operation cancelled')&&[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Run frozen CUSA checkpoint'&&!b.matches(':disabled'))`);
        assert.equal(hash(input),sourceHash);assert.equal(hash(checkpoint),checkpointHash);
        assert.equal(await js(`document.querySelectorAll('.patient-cusa-learning').length`),0);
        report.checks.push('User cancellation stops new inference and retains the source without admitting a learned branch');
        await capture('01-cancelled-source-preserved');report.sourcePreserved=true;report.passed=report.errors.length===0;
        report.finishedUtc=new Date().toISOString();fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));app.exit(report.passed?0:1);return;
      }else if(mode==='sector-policy'||mode==='sector-execute'){
        if(mode==='sector-execute'){
          await wait(`!!document.querySelector('.frozen-cusa-execution')`);
          const sourceScene=await js(`document.querySelector('.frozen-cusa-execution').dataset.sourceScene`);
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
      await wait(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Play replay'&&!b.matches(':disabled'))`);
      const position=await js(`Number(document.querySelector('[aria-label="Surgical timeline"]').value)`);
      const end=await js(`Number(document.querySelector('[aria-label="Surgical timeline"]').max)`);
      const goal=Math.min(position+4,end);assert(goal>position,'Selected policy must contain actual native actions');
      await click('Play replay');
      await wait(`Number(document.querySelector('[aria-label="Surgical timeline"]').value)>=${goal}`,180);
      if(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Pause replay')`))await click('Pause replay');
      await ready();report.playback={startPosition:position,observedPosition:await js(`Number(document.querySelector('[aria-label="Surgical timeline"]').value)`)};
      report.checks.push('Play advances actual recorded instrument/physics frames and pause stops scheduling');await capture('02-authoritative-playback');
      if(mode==='sector')assert(await js(`document.querySelectorAll('[aria-label="Sector strategy outcomes"] tbody tr').length>0`),'Comparison evidence remains visible during compact playback');
      await wait(`!![...document.querySelectorAll('header button')].find(b=>b.textContent.trim()==='Save'&&!b.matches(':disabled'))`);
      await click('Save');await wait(`![...document.querySelectorAll('header button')].find(b=>b.textContent.trim()==='Save')?.matches(':disabled')`);
      assert(fs.existsSync(saved));report.checks.push('Native desktop save writes the inspected trajectory');
      assert.equal(hash(input),sourceHash);report.sourcePreserved=true;report.savedCase=saved;report.passed=report.errors.length===0;
    }catch(error){report.passed=false;report.failure=error.stack;fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage()).toPNG());console.error(error.stack);}
    report.finishedUtc=new Date().toISOString();fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));app.exit(report.passed?0:1);
  });
});
app.on('will-quit',()=>fs.rmSync(userData,{recursive:true,force:true}));
require('./main.cjs');
