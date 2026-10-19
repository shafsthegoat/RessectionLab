'use strict';
/** Actual UI acceptance of a recovered, still-unadmitted anatomical candidate. */
const {app,dialog}=require('electron');
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict');
const repository=path.resolve(__dirname,'../..');
const root=path.resolve(process.env.PAT22_RECOVERY_ROOT||path.join(repository,'testcases/downloaded-datasets/BTC-PAT22-recovery'));
if(!root.startsWith(path.join(repository,'testcases')+path.sep))throw Error('Use the approved local testcases root');
const output=path.join(root,'desktop-qa');if(fs.existsSync(output))throw Error('Preserve prior QA output');fs.mkdirSync(output);
const qc=JSON.parse(fs.readFileSync(path.join(root,'PAT22-anatomy-review.qc.json'),'utf8'));
const userData=fs.mkdtempSync(path.join(os.tmpdir(),'rslab-btc-recovery-'));app.setPath('userData',userData);
const savePath=path.join(output,'PAT22-native-saved.ressectionlab');
dialog.showSaveDialog=async()=>({canceled:false,filePath:savePath});
dialog.showOpenDialog=async()=>({canceled:false,filePaths:[savePath]});
const report={schema:'resectionlab.btc-pat22-desktop-recovery.v1',checks:[],errors:[],screenshots:[]};
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
app.on('browser-window-created',(_event,window)=>window.webContents.once('did-finish-load',async()=>{
  window.webContents.setBackgroundThrottling(false);window.setSize(1440,1000);window.show();
  window.webContents.on('console-message',(_event,details)=>{if(details.level==='error')report.errors.push(details.message)});
  const js=code=>window.webContents.executeJavaScript(code,true);
  const wait=async(expression,seconds=120)=>{for(let i=0;i<seconds*4;i++){if(await js(expression))return;await delay(250)}throw Error('Timeout: '+expression)};
  const click=async label=>{await js(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(label)}||b.querySelector('span')?.textContent.trim()===${JSON.stringify(label)});if(!b||b.disabled)throw Error('Missing enabled button '+${JSON.stringify(label)});b.click()})()`);await delay(250)};
  const ready=()=>wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')&&document.querySelector('.rl-viewer').dataset.surfacesPending==='0'`);
  const capture=async name=>{await ready();await delay(350);fs.writeFileSync(path.join(output,name+'.png'),(await window.webContents.capturePage()).toPNG());report.screenshots.push(name+'.png')};
  const operation=async(name,args)=>js(`window.resectionApi[${JSON.stringify(name)}](${JSON.stringify(args)})`);
  try{
    await ready();assert(await js(`document.body.innerText.includes('PAT22')`));
    report.checks.push('recovered TRAIN source opens in actual acquired-image/3D workbench');await capture('01-recovered-source');
    await click('Rehearse');await wait(`!!document.querySelector('.surgical-panel')`);
    await click('Prepare instrument scene');await wait(`!!document.querySelector('.research-planning')`);
    await js(`document.querySelector('.research-planning').open=true;document.querySelector('.research-planning').scrollIntoView({block:'start'})`);
    const state=await operation('surgicalRehearsal',{caseHash:qc.workspaceCaseHash,command:'inspect'});
    assert.equal(state.position,0);assert.equal(state.planningAvailability.available,false);assert.equal(state.planningAvailability.checkpointAvailable,false);
    assert.equal(state.view.scene.environment.anatomy.bone,'unknown');assert.equal(state.view.scene.environment.anatomy.vessels,'unknown');
    report.checks.push('planning visibly refuses unadmitted source and missing critical evidence; no patient actions run');
    await capture('02-planning-refused');
    await click('Save');for(let i=0;i<240&&!fs.existsSync(savePath);i++)await delay(250);assert(fs.existsSync(savePath));
    await wait(`![...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Open')?.disabled`);
    await click('Open');await ready();await wait(`!!document.querySelector('.research-planning')`);
    const reopened=await operation('surgicalRehearsal',{caseHash:qc.workspaceCaseHash,command:'inspect'});
    assert.equal(reopened.view.stateId,state.view.stateId);assert.equal(reopened.position,0);assert.equal(reopened.planningAvailability.available,false);
    report.checks.push('native save/reopen retains source state, estimates and planning refusal');
    await js(`document.querySelector('.research-planning').open=true;document.querySelector('.research-planning').scrollIntoView({block:'start'})`);
    await capture('03-native-reopen');report.savedPath=savePath;report.passed=report.errors.length===0;
    await delay(500);
  }catch(error){report.passed=false;report.failure=error.stack;fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage()).toPNG());console.error(error.stack)}
  fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));app.exit(report.passed?0:1);
}));
app.on('will-quit',()=>fs.rmSync(userData,{recursive:true,force:true}));
require('./main.cjs');
