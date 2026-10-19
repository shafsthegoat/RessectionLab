'use strict';
/** Actual React controls, preload IPC, source-bound planner, native replay/save.
 * Only disposable files under testcases/ are written. No fitting is launched. */
const {app,dialog}=require('electron');
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..');
const input=process.env.RESEARCH_PLANNING_CASE;
if(!input||!path.resolve(input).startsWith(path.join(root,'testcases')+path.sep))throw Error('Set RESEARCH_PLANNING_CASE to the disposable saved comparison under testcases/');
const output=path.resolve(process.env.RESEARCH_PLANNING_OUTPUT||path.join(path.dirname(path.resolve(input)),'desktop'));
if(!output.startsWith(path.join(root,'testcases')+path.sep))throw Error('Desktop output must stay under testcases/');
if(fs.existsSync(output))throw Error('Desktop result already exists; preserve it and use a new run');
fs.mkdirSync(output);
const userData=fs.mkdtempSync(path.join(os.tmpdir(),'rslab-research-plan-'));app.setPath('userData',userData);
const savePath=path.join(output,'desktop-saved.ressectionlab');
dialog.showSaveDialog=async()=>({canceled:false,filePath:savePath});
dialog.showOpenDialog=async()=>({canceled:false,filePaths:[savePath]});
const report={schema:'resectionlab.desktop-research-planning.v1',checks:[],errors:[],screenshots:[]};
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
app.on('browser-window-created',(_event,window)=>{
  window.webContents.on('console-message',(_event,d)=>{if(d.level==='error')report.errors.push(d.message)});
  window.webContents.once('did-finish-load',async()=>{
    window.webContents.setBackgroundThrottling(false);window.setSize(1500,1000);window.show();
    const js=code=>window.webContents.executeJavaScript(code,true);
    const wait=async(expression,seconds=120)=>{for(let i=0;i<seconds*4;i++){if(await js(expression))return;await delay(250)}throw Error('Timeout: '+expression)};
    const click=async label=>{await js(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(label)}||b.getAttribute('aria-label')===${JSON.stringify(label)}||b.querySelector('span')?.textContent.trim()===${JSON.stringify(label)});if(!b||b.disabled)throw Error('Missing enabled button: '+${JSON.stringify(label)});b.click()})()`);await delay(300)};
    const capture=async name=>{await wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')&&document.querySelector('.rl-viewer').dataset.surfacesPending==='0'`);await delay(400);fs.writeFileSync(path.join(output,name+'.png'),(await window.webContents.capturePage()).toPNG());report.screenshots.push(name+'.png')};
    try{
      await wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`);
      await click('Rehearse');await wait(`!!document.querySelector('.research-planning')`);
      await js(`document.querySelector('.research-planning').open=true;document.querySelector('.research-planning').scrollIntoView({block:'start'})`);
      assert(await js(`document.querySelector('.research-planning').innerText.includes('Seed 17')`));
      assert(await js(`[...document.querySelectorAll('.research-planning button')].find(b=>b.textContent.includes('Compare greedy / saved learner')).disabled`));
      report.checks.push('source-bound panel and explicit opt-in gate are visible');
      await capture('01-source-and-evidence');
      await js(`document.querySelector('[aria-label="Acknowledge research planning assumptions"]').click()`);
      await click('Compare greedy / saved learner');
      await wait(`!document.querySelector('.surgical-panel').innerText.includes('Computing instrument action')`);
      await wait(`document.querySelectorAll('.research-plan-alternatives article').length===2`);
      assert(!(await js(`!!document.querySelector('.surgical-error')`)));
      report.checks.push('actual UI invokes both greedy and frozen learner with no optimizer');
      await js(`document.querySelector('.research-planning').scrollIntoView({block:'start'})`);await capture('02-checked-alternatives');
      await click('Frozen learner · seed 17 · replay');
      await wait(`!!document.querySelector('.research-plan-alternatives button[aria-pressed="true"]')`);
      await js(`(()=>{const slider=document.querySelector('[aria-label="Research planning decision"]');Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(slider,'1');slider.dispatchEvent(new Event('input',{bubbles:true}));slider.dispatchEvent(new Event('change',{bubbles:true}));})()`);
      await delay(500);await wait(`!document.querySelector('.surgical-panel').innerText.includes('Computing instrument action')`);
      // Replay controls select a recorded source state; switching alternatives
      // at the same decision index must not run a new policy or train a model.
      await click('Observation-only greedy · replay');
      await wait(`!document.querySelector('.surgical-panel').innerText.includes('Computing instrument action')`);
      report.checks.push('both alternatives open through shared native 3D/MPR replay');
      await capture('03-aligned-greedy-replay');
      await click('Save');
      for(let i=0;i<240&&!fs.existsSync(savePath);i++)await delay(250);
      assert(fs.existsSync(savePath),'Native save dialog must create a real bundle');
      await wait(`![...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Open')?.disabled`);
      await click('Open');await wait(`!!document.querySelector('.surgical-panel')`);
      await wait(`document.querySelectorAll('.research-plan-alternatives article').length===2`);
      await js(`document.querySelector('.research-planning').open=true;document.querySelector('.research-planning').scrollIntoView({block:'start'})`);
      report.checks.push('native save/open preserves checked comparison and branches');
      await capture('04-native-reopen');
      report.savedPath=savePath;report.passed=report.errors.length===0;
    }catch(error){report.passed=false;report.failure=error.stack;fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage()).toPNG());console.error(error.stack)}
    fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
    app.exit(report.passed?0:1);
  });
});
app.on('will-quit',()=>fs.rmSync(userData,{recursive:true,force:true}));
require('./main.cjs');
