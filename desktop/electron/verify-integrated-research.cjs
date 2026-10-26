'use strict';
// Actual Electron comparison/learned trajectory playback, with native replay.
const {app,dialog}=require('electron');
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..'),input=path.resolve(process.env.INTEGRATED_RESEARCH_CASE||''),output=path.resolve(process.env.INTEGRATED_RESEARCH_OUTPUT||''),mode=process.env.INTEGRATED_RESEARCH_MODE;
if(!['access','learned','cusa'].includes(mode))throw Error('Set INTEGRATED_RESEARCH_MODE to access, learned or cusa');
for(const p of [input,output])if(!p.startsWith(path.join(root,'testcases')+path.sep))throw Error('Input/output must be under testcases/');
if(fs.existsSync(output))throw Error('Preserve previous evidence; use a new output directory');
fs.mkdirSync(output,{recursive:true});
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex'),sourceHash=hash(input);
const userData=fs.mkdtempSync(path.join(os.tmpdir(),'rslab-integrated-test-'));app.setPath('userData',userData);
process.argv.push('--case',input);
const saved=path.join(output,'desktop-saved.ressectionlab');
dialog.showSaveDialog=async()=>({canceled:false,filePath:saved});
dialog.showOpenDialog=async()=>({canceled:false,filePaths:[saved]});
const report={schema:'resectionlab.integrated-research-electron.v1',input,mode,checks:[],errors:[],screenshots:[],startedUtc:new Date().toISOString()};
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
app.on('browser-window-created',(_event,window)=>{
  window.webContents.on('console-message',(_event,d)=>{if(d.level==='error')report.errors.push(d.message);});
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
      if(mode==='access'){
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
        await wait(`!!document.querySelector('.patient-cusa-learning')`);
        assert(await js(`document.querySelectorAll('[aria-label="Learned CUSA outcomes"] tbody tr').length>=1`));
        report.learning=await js(`document.querySelector('.patient-cusa-learning').innerText`);
        const target=await js(`(()=>{const b=[...document.querySelectorAll('.patient-cusa-learning button')].find(b=>b.textContent.trim()==='Inspect PPO CUSA execution');if(!b)throw Error('PPO boundary missing');return{branchId:b.dataset.policyBranch,position:Number(b.dataset.policyPosition)}})()`);
        await capture('01-fresh-learned-provenance');await click('Inspect PPO CUSA execution');
        await wait(`document.querySelector('[aria-label="Surgical branch"]').value===${JSON.stringify(target.branchId)}&&Number(document.querySelector('[aria-label="Surgical timeline"]').value)===${target.position}`);
      }
      await ready();const position=await js(`Number(document.querySelector('[aria-label="Surgical timeline"]').value)`);
      const end=await js(`Number(document.querySelector('[aria-label="Surgical timeline"]').max)`);
      const goal=Math.min(position+4,end);assert(goal>position,'Selected policy must contain actual native actions');
      await click('Play replay');
      await wait(`Number(document.querySelector('[aria-label="Surgical timeline"]').value)>=${goal}`,180);
      if(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Pause replay')`))await click('Pause replay');
      await ready();report.playback={startPosition:position,observedPosition:await js(`Number(document.querySelector('[aria-label="Surgical timeline"]').value)`)};
      report.checks.push('Play advances actual recorded instrument/physics frames and pause stops scheduling');await capture('02-authoritative-playback');
      await click('Save');await wait(`![...document.querySelectorAll('header button')].find(b=>b.textContent.trim()==='Save')?.matches(':disabled')`);
      assert(fs.existsSync(saved));report.checks.push('Native desktop save writes the inspected trajectory');
      assert.equal(hash(input),sourceHash);report.sourcePreserved=true;report.savedCase=saved;report.passed=report.errors.length===0;
    }catch(error){report.passed=false;report.failure=error.stack;fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage()).toPNG());console.error(error.stack);}
    report.finishedUtc=new Date().toISOString();fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));app.exit(report.passed?0:1);
  });
});
app.on('will-quit',()=>fs.rmSync(userData,{recursive:true,force:true}));
require('./main.cjs');
