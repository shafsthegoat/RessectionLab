'use strict';
// Production React/preload/engine/render/save path. Never show or focus a window.
const {app,dialog,ipcMain}=require('electron');
process.argv.push('--automation-hidden');
if(process.platform==='darwin')app.setActivationPolicy('accessory');
process.on('uncaughtException',error=>{console.error(error.stack);app.exit(1);});
dialog.showErrorBox=(_title,message)=>{console.error(message);app.exit(1);};
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const {execFileSync}=require('node:child_process');
const repo=path.resolve(__dirname,'../..'),root=process.env.RESSECTIONLAB_DATA_ROOT;
if(!root)throw Error('Set RESSECTIONLAB_DATA_ROOT');
try{execFileSync(path.join(repo,'.venv/bin/python'),['-c','from resectionlab.data_root import checked_root; checked_root(required_bytes=1024**3)'],{cwd:repo,env:process.env});}
catch(error){console.error('External storage preflight failed: '+error.message);app.exit(1);}
const input=fs.realpathSync(process.env.DATA_REBUILD_CASE),output=path.resolve(process.env.DATA_REBUILD_OUTPUT);
for(const file of [input,output])assert(file.startsWith(root+path.sep),'Keep case and evidence on the configured external root');
assert(!fs.existsSync(output),'Preserve prior evidence; use a fresh output directory');fs.mkdirSync(output,{recursive:true});
const digest=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const inputSha=digest(input),saved=path.join(output,'saved.ressectionlab');
const report={schema:'ressectionlab-data-hidden-app-v1',input,inputSha256:inputSha,output,rendererIndexSha256:digest(path.join(repo,'desktop/dist/index.html')),
  sourceBindings:Object.fromEntries(['desktop/electron/main.cjs','src/resectionlab/desktop_bridge.py','src/resectionlab/workspace_bundle.py','src/resectionlab/surgical_engine.py'].map(file=>[file,digest(path.join(repo,file))])),
  windowEverVisible:false,checks:[],requests:[],startedUtc:new Date().toISOString(),passed:false};
// Chromium process locks require POSIX links, unsupported by ExFAT. Keep this
// small disposable runtime profile local; all imaging, transfers and evidence
// use the external drive. Remove it after the app exits.
const profile=fs.mkdtempSync(path.join(require('node:os').tmpdir(),'rslab-data-profile-'));
app.setPath('userData',profile);report.runtimeProfile=profile;
process.argv.push('--case',input,'--automation-hidden');
dialog.showSaveDialog=async()=>({canceled:false,filePath:saved});dialog.showMessageBox=async()=>({response:1});
let selection=saved;dialog.showOpenDialog=async()=>({canceled:false,filePaths:[selection]});
const originalHandle=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(name,handler)=>originalHandle(name,async(...args)=>{
  const started=performance.now();
  try{const result=await handler(...args);report.requests.push({operation:name,command:args[1]?.command,ms:performance.now()-started,
    stateId:result?.view?.stateId,caseHash:result?.caseHash,reason:result?.reason,status:result?.status});return result;}
  catch(error){report.requests.push({operation:name,command:args[1]?.command,ms:performance.now()-started,error:String(error)});throw error;}
});
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
app.on('browser-window-created',(_event,window)=>{
  window.on('show',()=>{report.windowEverVisible=true;window.hide();});
  window.webContents.once('did-finish-load',async()=>{
    window.webContents.setBackgroundThrottling(false);window.setContentSize(1440,950);
    const js=code=>window.webContents.executeJavaScript(code,true);
    const wait=async expression=>{const start=Date.now();while(Date.now()-start<180000){if(await js(expression))return;await delay(100);}throw Error('Timeout '+expression+'\n'+await js('document.body.innerText'));};
    const click=async label=>{await js(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(label)}||b.getAttribute('aria-label')===${JSON.stringify(label)}||b.querySelector('span')?.textContent.trim()===${JSON.stringify(label)});if(!b||b.disabled)throw Error('Missing enabled '+${JSON.stringify(label)});for(let p=b.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;b.click()})()`);await delay(100);};
    const ready=async()=>{
      let stable=0;const started=Date.now();
      while(Date.now()-started<180000){
        const valid=await js(`(()=>{const v=document.querySelector('.rl-viewer');return !!v?.querySelector('canvas')&&v.dataset.sourceCaseHash?.startsWith('sha256:')&&v.dataset.rendererReady==='true'&&v.dataset.surfacesPending==='0'&&!document.querySelector('.viewer-loading,.rl-viewer-preparing')&&!document.body.innerText.includes('No case open')})()`);
        stable=valid?stable+1:0;if(stable>=10){await js(`new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))`);return;}
        await delay(100);
      }
      throw Error('Source rendering did not settle: '+await js('document.body.innerText'));
    };
    const capture=async name=>{await ready();assert.equal(window.isVisible(),false);fs.writeFileSync(path.join(output,name+'.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());};
    const set=async(label,value)=>{await js(`(()=>{const el=document.querySelector('[aria-label='+${JSON.stringify(JSON.stringify(label))}+']');if(!el)throw Error('Missing '+${JSON.stringify(label)});for(let p=el.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;Object.getOwnPropertyDescriptor(el.tagName==='SELECT'?HTMLSelectElement.prototype:HTMLInputElement.prototype,'value').set.call(el,${JSON.stringify(value)});el.dispatchEvent(new Event(el.tagName==='SELECT'?'change':'input',{bubbles:true}))})()`);await delay(100);};
    try{
      if(process.env.DATA_REBUILD_EXPECT_FAILURE==='1'){
        await wait(`/Bundle must contain|INVALID_BUNDLE_MEMBERS|Reopened case differs|CASE_HASH_MISMATCH/.test(document.body.innerText)`);
        assert.equal(await js(`!!document.querySelector('.rl-viewer')?.dataset.sourceCaseHash`),false);
        report.errorBody=await js('document.body.innerText');assert.equal(window.isVisible(),false);
        fs.writeFileSync(path.join(output,'expected-safe-rejection.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());
        report.checks.push('Malformed native fixture visibly rejected without fallback source anatomy');report.passed=true;return;
      }
      const started=performance.now();await ready();report.coldReadyMs=performance.now()-started;
      if(process.env.DATA_REBUILD_STRUCTURES==='1'){
        await click('Show structures only');await ready();
        report.checks.push('Explicit structures-only 3D display with unchanged full native source slices');
      }
      await capture('01-native-source');report.checks.push('Production importer and renderer open actual external native package');
      if(process.env.DATA_REBUILD_USE_CATALOG==='1'){
        const listedAt=performance.now();
        const listed=await js('window.resectionApi.listCases()');
        const reviewed=listed.filter(row=>row.reviewedExternal);
        report.catalog={listedRows:listed.length,reviewedExternalRows:reviewed.length,listMs:performance.now()-listedAt,
          releaseSha256:digest(path.join(root,'catalog/release.json'))};
        if(process.env.DATA_REBUILD_EXPECT_CATALOG_COUNT){
          const expected=Number(process.env.DATA_REBUILD_EXPECT_CATALOG_COUNT);
          assert(Number.isSafeInteger(expected)&&expected>0);assert.equal(reviewed.length,expected);
        }
        await click('Cases');await wait(`!!document.querySelector('.poc-case-card')`);
        report.catalog.renderedCards=await js(`document.querySelectorAll('.poc-case-card').length`);
        await capture('01a-reviewed-case-library');
        await js(`(()=>{const b=[...document.querySelectorAll('.poc-case-card')].find(b=>b.querySelector('strong')?.textContent===${JSON.stringify(process.env.DATA_REBUILD_CATALOG_ID||'UCSF-PDGM-0009')});if(!b)throw Error('Reviewed external case missing from actual library');b.click()})()`);
        await ready();await capture('01b-opened-through-reviewed-library');
        assert(report.requests.some(row=>row.operation==='research:openListedWorkspace'));
        report.checks.push('Actual Cases card uses opaque external ID and checksum-bound catalog open');
      }
      report.initialBody=await js('document.body.innerText');
      if(process.env.DATA_REBUILD_STAGES==='1'){
        await click('Data');await wait(`!!document.querySelector('[aria-label="Inspect scan"]')`);
        const stages=await js(`[...document.querySelector('[aria-label="Inspect scan"]').options].filter(o=>/US before|US during|US after/.test(o.textContent)).map(o=>({value:o.value,label:o.textContent}))`);
        assert.equal(stages.length,3,'All observed US operative stages are distinctly named');
        for(let i=0;i<stages.length;i++){
          await set('Inspect scan',stages[i].value);await ready();await capture('stage-'+(i+1)+'-native-US');
          report.checks.push('Actual UI selects and renders '+stages[i].label);
        }
        await set('Inspect scan','');await ready();
      }
      if(process.env.DATA_REBUILD_REHEARSE==='1'){
        await click('Rehearse');await wait(`!!document.querySelector('.surgical-panel')`);
        await wait(`!!document.querySelector('[aria-label="Instrument action"]')||[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Prepare instrument scene'&&!b.disabled)`);
        if(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify saved replay'&&!b.disabled)`))await click('Verify saved replay');
        if(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Prepare instrument scene'&&!b.disabled)`))await click('Prepare instrument scene');
        await wait(`!!document.querySelector('[aria-label="Instrument action"]')`);await ready();await capture('02-rehearsal');
        report.checks.push('Actual production surgical scene initializes; missing evidence remains unavailable');
        await set('Instrument action','inspect');await click('Preview');
        await wait(`document.querySelector('.surgical-panel').textContent.includes('Preview · uncommitted')`);await ready();await capture('03-inspect-preview');
        await click('Cancel preview');await ready();report.checks.push('Native instrument inspect preview and cancel preserve accepted state');
        if(process.env.DATA_REBUILD_ACTIONS==='1'){
          await js(`(()=>{const b=[...document.querySelectorAll('.surgical-toolbox button')].find(b=>b.textContent.includes('CUSA')&&!b.disabled);if(!b)throw Error('No available CUSA exchange');for(let p=b.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;b.click()})()`);
          await wait(`document.querySelector('.surgical-slots button')?.textContent.includes('CUSA')`);await ready();
          await capture('03b-instrument-exchange');report.checks.push('Actual UI exchanges withdrawn left instrument to CUSA without an invented access corridor');
        }
      }
      await click('Save');await wait(`document.body.innerText.includes('Workspace saved locally')`);assert(fs.existsSync(saved));
      report.saved=saved;report.savedSha256=digest(saved);report.checks.push('Actual UI saves native workspace externally');
      const count=report.requests.length;await click('Open');await wait(`document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`);
      while(!report.requests.slice(count).some(row=>row.operation==='research:openWorkspace'))await delay(100);
      if(process.env.DATA_REBUILD_REHEARSE==='1'){
        if(!await js(`!!document.querySelector('.surgical-panel')`))await click('Rehearse');
        await wait(`!!document.querySelector('.surgical-panel')`);
        await wait(`!!document.querySelector('[aria-label="Instrument action"]')||[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify saved replay'&&!b.disabled)`);
        if(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify saved replay'&&!b.disabled)`))await click('Verify saved replay');
      }
      await capture('04-native-reopened');report.checks.push('Actual UI reopens the saved native package');
      report.finalBody=await js('document.body.innerText');assert.equal(digest(input),inputSha);assert.equal(report.windowEverVisible,false);report.passed=true;
    }catch(error){report.error=String(error);report.body=await js('document.body.innerText').catch(()=>null);fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());}
    finally{report.finishedUtc=new Date().toISOString();fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({passed:report.passed,error:report.error,output}));app.exit(report.passed?0:1);}
  });
});
app.on('will-quit',()=>fs.rmSync(profile,{recursive:true,force:true}));
setTimeout(()=>{if(!report.passed){report.error='Global hidden verifier timeout';fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n');app.exit(1);}},600000).unref();
require('./main.cjs');
