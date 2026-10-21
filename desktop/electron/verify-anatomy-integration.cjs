'use strict';
/** Actual preload/IPC, patient engine, frozen policy, and native save/reopen.
 * Inputs are existing source-bound exports. All writes target a new output. */
const {app,dialog}=require('electron');
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..');
const directory=path.resolve(process.env.ANATOMY_VERIFY_ROOT||path.join(root,'outputs/anatomy-integration-v3'));
const output=path.join(directory,'desktop-integration');fs.mkdirSync(output,{recursive:true});
const recipe=JSON.parse(fs.readFileSync(path.join(directory,'comparison-recipe.json'),'utf8'));
const afterPath=path.join(directory,'surgical-refined/patient.ressectionlab');
const savedPath=path.join(output,'rehearsal-reopened.ressectionlab');
const userData=fs.mkdtempSync(path.join(os.tmpdir(),'rslab-anatomy-integration-'));app.setPath('userData',userData);
let nextPath=afterPath;
dialog.showOpenDialog=async()=>({canceled:false,filePaths:[nextPath]});
dialog.showSaveDialog=async()=>({canceled:false,filePath:savedPath});
const report={schema:'resectionlab.desktop-anatomy-integration.v1',errors:[],checks:[],views:[]};
const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
app.on('browser-window-created',(_event,window)=>{
  window.webContents.on('console-message',(_event,d)=>{if(d.level==='error')report.errors.push(d.message)});
  window.webContents.once('did-finish-load',async()=>{
    window.webContents.setBackgroundThrottling(false);window.setSize(1440,960);window.show();
    const js=code=>window.webContents.executeJavaScript(code,true);
    const wait=async(expr,seconds=180)=>{for(let i=0;i<seconds*4;i++){if(await js(expr))return;await pause(250)}throw Error('Timed out '+expr)};
    const ready=()=>wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')&&document.querySelector('.rl-viewer').dataset.surfacesPending==='0'`);
    const click=async label=>{await js(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.getAttribute('aria-label')===${JSON.stringify(label)}||b.textContent.trim()===${JSON.stringify(label)}||b.querySelector('span')?.textContent.trim()===${JSON.stringify(label)});if(!b||b.disabled)throw Error('Missing enabled button '+${JSON.stringify(label)});b.click()})()`);await pause(250)};
    const capture=async name=>{await ready();await pause(400);fs.writeFileSync(path.join(output,name+'.png'),(await window.webContents.capturePage()).toPNG());report.views.push(name);console.log('Captured '+name)};
    const operation=async(name,args)=>js(`window.resectionApi[${JSON.stringify(name)}](${JSON.stringify(args)})`);
    try{
      await ready();await capture('baseline-source');
      const beforeInitial=await operation('surgicalRehearsal',{caseHash:recipe.beforeCaseHash,command:'inspect'});
      assert(beforeInitial?.view);report.checks.push('baseline installed through startup');
      await click('Open');await wait(`document.body.innerText.includes('Workspace')`);await ready();
      const afterInitial=await operation('surgicalRehearsal',{caseHash:recipe.afterCaseHash,command:'inspect'});
      assert(afterInitial?.view);await capture('refined-source');
      console.log('Running fixed anatomy comparison through Electron preload and shared Python backend');
      report.comparison=await operation('compareSurgicalAnatomy',recipe);
      assert.equal(report.comparison.controls.training,false);
      assert.equal(report.comparison.before.caseHash,recipe.beforeCaseHash);
      assert.equal(report.comparison.after.caseHash,recipe.afterCaseHash);
      for(const [key,initial]of [[recipe.beforeCaseHash,beforeInitial],[recipe.afterCaseHash,afterInitial]]){
        const current=await operation('surgicalRehearsal',{caseHash:key,command:'inspect'});
        assert.equal(current.view.stateId,initial.view.stateId,'comparison must leave desktop timelines unchanged');
      }
      report.checks.push('comparison uses IPC and leaves desktop timelines unchanged');
      let state=afterInitial;
      for(let i=0;i<recipe.prefix.length;i++){
        console.log('Desktop fixed prefix '+(i+1)+'/'+recipe.prefix.length);
        state=await operation('surgicalRehearsal',{caseHash:recipe.afterCaseHash,command:'step',expectedStateId:state.view.stateId,action:recipe.prefix[i]});
        assert.notEqual(state.outcome?.status,'rejected');
      }
      assert.equal(state.view.stateId,report.comparison.after.probe.stateId,'desktop and RL fixed-prefix state must match');
      const selected=report.comparison.after.policy[0];
      const preview=await operation('surgicalRehearsal',{caseHash:recipe.afterCaseHash,command:'preview',expectedStateId:state.view.stateId,action:selected.action});
      assert.equal(preview.status,selected.outcome.status);
      report.desktopPrefixState=state.view.stateId;report.desktopPreview={status:preview.status,stateId:preview.stateId};
      const committed=await operation('surgicalRehearsal',{caseHash:recipe.afterCaseHash,command:'step',expectedStateId:state.view.stateId,action:selected.action});
      assert.equal(committed.view.stateId,preview.stateId);
      assert.equal(committed.view.stateId,selected.stateId,'desktop and RL selected-policy state must match');
      report.checks.push('actual desktop prefix and policy preview match commit');
      // Opening this saved engine snapshot refreshes the normal React workspace.
      await operation('saveCase',{caseHash:recipe.afterCaseHash});
      nextPath=savedPath;await click('Open');await ready();await click('Rehearse');
      await wait(`!!document.querySelector('.surgical-panel .surgical-slots')`);await capture('refined-rehearsal-reopened');
      const reopened=await operation('surgicalRehearsal',{caseHash:recipe.afterCaseHash,command:'inspect'});
      assert.equal(reopened.view.stateId,committed.view.stateId);
      report.savedStateId=reopened.view.stateId;report.savedPath=savedPath;
      report.checks.push('native save and reopen preserve actual desktop rehearsal');
      report.passed=report.errors.length===0;
    }catch(error){report.passed=false;report.failure=error.stack;console.error(error.stack);fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage()).toPNG())}
    fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));
    console.log(JSON.stringify({passed:report.passed,checks:report.checks,failure:report.failure,delta:report.comparison?.delta}));
    app.exit(report.passed?0:1);
  });
});
app.on('will-quit',()=>fs.rmSync(userData,{recursive:true,force:true}));
require('./main.cjs');
