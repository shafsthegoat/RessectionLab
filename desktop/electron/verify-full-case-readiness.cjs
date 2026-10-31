'use strict';
// Native UI proof: all app windows and captures stay hidden.
const {app,dialog,ipcMain}=require('electron');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..'),base=path.join(root,'testcases');
const input=path.resolve(process.env.READINESS_CASE||path.join(base,'cases/meningioma-sub-06-standard-v2.ressectionlab'));
const output=path.resolve(process.env.READINESS_OUTPUT||'');
const cold=process.env.READINESS_COLD==='1';
for(const value of [input,output])if(!value.startsWith(base+path.sep))throw Error('Use inputs and a fresh output under testcases');
if(fs.existsSync(output))throw Error('Preserve previous evidence; choose a fresh output');
fs.mkdirSync(output,{recursive:true});
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex'),original=sha(input);
const saved=path.join(output,'bound-workspace.ressectionlab');
app.setPath('userData',path.join(output,'hidden-profile'));
process.argv.push('--case',input,'--automation-hidden');
const report={input,output,cold,windowEverVisible:false,checks:[],responses:[],captures:[]};
const manifests=['UCSF-PDGM-0004','meningioma-sub-06'].map(id=>path.join(base,'environments',id,'rl-protection.json'));
dialog.showOpenDialog=async()=>({canceled:false,filePaths:[manifests.shift()]});
dialog.showSaveDialog=async()=>({canceled:false,filePath:saved});
const originalHandle=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(name,handler)=>originalHandle(name,async(...args)=>{
  const result=await handler(...args);
  if(name==='research:surgicalRehearsal'&&result)report.responses.push({command:args[1]?.command,
    stateId:result.view?.stateId,branchId:result.branchId,readiness:result.fullCaseReadiness});
  return result;
});
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
app.on('browser-window-created',(_event,window)=>{
  window.on('show',()=>{report.windowEverVisible=true;window.hide();});
  window.webContents.once('did-finish-load',async()=>{
    window.webContents.setBackgroundThrottling(false);window.setSize(1500,1050);
    const js=code=>window.webContents.executeJavaScript(code,true);
    const wait=async(expression)=>{const start=Date.now();while(Date.now()-start<120000){if(await js(expression))return;await delay(100);}throw Error('Timeout: '+expression+'\n'+await js('document.body.innerText'));};
    const click=label=>js(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(label)});if(!b||b.disabled)throw Error('Missing enabled button '+${JSON.stringify(label)});b.click();})()`);
    const available=scenario=>js(`(()=>{const b=[...document.querySelectorAll('.full-case-rehearsal button')].find(b=>b.textContent.includes(${JSON.stringify(scenario)}));return !!b&&!b.disabled;})()`);
    const capture=async name=>{
      await js(`document.querySelector('.full-case-rehearsal').scrollIntoView({block:'center'})`);
      await delay(200);fs.writeFileSync(path.join(output,name+'.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());report.captures.push(name+'.png');
    };
    try{
      await wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`);
      await click('Rehearse');
      await wait(`!!document.querySelector('.full-case-rehearsal')&&document.querySelector('.rl-viewer')?.dataset.surfacesPending==='0'`);
      if(!cold){
        await wait(`document.querySelector('.full-case-rehearsal').innerText.includes('Full-case rehearsal blocked')`);
        assert.equal(await available('primary rehearsal'),false);assert.equal(await available('vascular injury'),false);
        const initial=report.responses.at(-1);await capture('01-missing-evidence');
        await click('Attach matching protection');
        await wait(`document.body.innerText.includes('selected protection manifest is unavailable or incompatible')`);
        assert.equal(report.responses.at(-1).stateId,initial.stateId);assert.equal(await available('primary rehearsal'),false);
        report.checks.push('Missing evidence disables start; explicit wrong-source attachment is refused without replacing accepted state');
        await click('Attach matching protection');
      }
      await wait(`document.querySelector('.full-case-rehearsal').innerText.includes('Ready for full-case rehearsal')`);
      await wait(`[...document.querySelectorAll('.full-case-rehearsal button')].some(b=>b.textContent.includes('primary rehearsal')&&!b.disabled)`);
      assert.equal(await available('primary rehearsal'),true);assert.equal(await available('vascular injury'),true);
      assert.equal(await js(`!![...document.querySelectorAll('.full-case-rehearsal details')].find(d=>d.querySelector('summary')?.textContent==='Readiness details')?.open`),false);
      const ready=report.responses.at(-1);assert.equal(ready.readiness.status,'ready');
      if(process.env.READINESS_EXPECT){const expected=JSON.parse(fs.readFileSync(process.env.READINESS_EXPECT,'utf8')).ready;assert.equal(ready.stateId,expected.stateId);assert.equal(ready.readiness.protectionContractId,expected.readiness.protectionContractId);}
      report.ready=ready;await capture(cold?'03-cold-reopened-ready':'02-current-case-ready');
      report.checks.push('Current source-bound readiness enables both scenarios and technical details remain collapsed');
      if(!cold){await click('Save');await wait(`document.body.innerText.includes('Workspace saved locally')`);assert(fs.existsSync(saved));report.saved=saved;report.checks.push('Native workspace saves attached self-contained protection');}
      assert.equal(sha(input),original);report.sourcePreserved=true;report.passed=!report.windowEverVisible;
      fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,checks:report.checks,saved:report.saved}));app.exit(report.passed?0:1);
    }catch(error){report.error=String(error);report.body=await js('document.body.innerText').catch(()=>null);fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.error(error);app.exit(1);}
  });
});
require('./main.cjs');
