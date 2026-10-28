'use strict';
// Real React controls -> preload -> sidecar -> native anatomy -> engine.
// Dialogs choose pinned local inputs; no anatomy or engine response is mocked.
const {app,dialog}=require('electron');
const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..');
const source=path.resolve(process.env.FRESH_PATIENT_SOURCE||'');
const output=path.resolve(process.env.FRESH_PATIENT_OUTPUT||'');
for(const p of [source,output])if(!p.startsWith(path.join(root,'testcases')+path.sep))throw Error('Set FRESH_PATIENT_SOURCE and FRESH_PATIENT_OUTPUT under ignored testcases/');
if(fs.existsSync(output))throw Error('Preserve the existing verification; select a new output directory');
const patient=path.basename(source);
if(!/^UCSF-PDGM-\d{4}$/.test(patient))throw Error('Source directory must identify the pinned UCSF case');
fs.mkdirSync(output,{recursive:true});
const native=path.join(output,'native');fs.mkdirSync(native);
const userData=fs.mkdtempSync(path.join(os.tmpdir(),'rslab-fresh-test-'));app.setPath('userData',userData);
// Start with an actual empty backend collection so historical library entries
// cannot auto-open a different patient while this verifier creates its study.
const startup=path.join(output,'empty-startup');
require('node:child_process').execFileSync(path.join(root,'.venv/bin/python'),['-c',
  'from resectionlab.patient_anatomy import PatientImagingCollection; import sys; PatientImagingCollection.create(sys.argv[1], "EMPTY-VERIFIER").save("empty verifier startup")',startup],
  {cwd:root,env:{...process.env,PYTHONPATH:path.join(root,'src')}});
process.argv.push('--anatomy',path.join(startup,'patient-anatomy.json'));
let selected=null;
dialog.showOpenDialog=async()=>{if(!selected)throw Error('Unplanned Open dialog');return{canceled:false,filePaths:[selected]};};
let savePath=path.join(output,'patient.rsworkspace');
dialog.showSaveDialog=async()=>({canceled:false,filePath:savePath});
dialog.showMessageBox=async()=>({response:1});
const report={schema:'resectionlab.fresh-patient-electron.v1',patient,source,checks:[],errors:[],screenshots:[],startedUtc:new Date().toISOString()};
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
app.on('browser-window-created',(_event,window)=>{
  window.webContents.on('console-message',(_event,d)=>{if(d.level==='error')report.errors.push(d.message);});
  window.webContents.once('did-finish-load',async()=>{
    window.webContents.setBackgroundThrottling(false);window.setSize(1500,1000);window.show();
    const js=code=>window.webContents.executeJavaScript(code,true);
    const wait=async(expression,seconds=240)=>{for(let i=0;i<seconds*4;i++){if(await js(expression))return;await delay(250);}throw Error('Timeout: '+expression);};
    const click=async label=>{await js(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(label)}||b.getAttribute('aria-label')===${JSON.stringify(label)}||b.querySelector('span')?.textContent.trim()===${JSON.stringify(label)});if(!b||b.matches(':disabled'))throw Error('Missing enabled button '+${JSON.stringify(label)});b.click();})()`);await delay(100);};
    const set=async(selector,value)=>{await js(`(()=>{const el=document.querySelector(${JSON.stringify(selector)});if(!el)throw Error('Missing control '+${JSON.stringify(selector)});Object.getOwnPropertyDescriptor(el instanceof HTMLSelectElement?HTMLSelectElement.prototype:HTMLInputElement.prototype,'value').set.call(el,${JSON.stringify(value)});el.dispatchEvent(new Event(el instanceof HTMLSelectElement?'change':'input',{bubbles:true}));})()`);await delay(100);};
    const ready=()=>wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')&&!![...document.querySelectorAll('header button')].find(b=>b.textContent.trim()==='Save'&&!b.matches(':disabled'))`);
    const capture=async name=>{await delay(300);if(name!=='failure'){
      await wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')&&document.querySelector('.rl-viewer').dataset.surfacesPending==='0'`);
      await delay(150);await ready();
    }fs.writeFileSync(path.join(output,name+'.png'),(await window.webContents.capturePage()).toPNG());report.screenshots.push(name+'.png');};
    const manifest=()=>JSON.parse(fs.readFileSync(path.join(native,'patient-anatomy.json'),'utf8'));
    try{
      await wait(`!!document.querySelector('.native-workspace')&&document.querySelector('footer').innerText.includes('Add a source scan')`);await click('Cases');
      await set('[aria-label="New patient ID"]',patient);selected=native;await click('Create study…');
      await wait(`!!document.querySelector('.import-data')`);
      for(const [sequence,suffix] of [['T1CE','T1c'],['T1','T1'],['T2','T2'],['FLAIR','FLAIR']]){
        await set('[aria-label="Import image sequence"]',sequence);selected=path.join(source,`${patient}_${suffix}.nii.gz`);
        await click('Import NIfTI…');await ready();
        assert(manifest().series.some(s=>s.modality===sequence));report.checks.push('UI imported '+sequence);
      }
      await set('[aria-label="Segmentation role 1"]','context');
      await click('Add label');await set('[aria-label="Segmentation label 2"]','2');await set('[aria-label="Segmentation role 2"]','context');
      await click('Add label');await set('[aria-label="Segmentation label 3"]','4');await set('[aria-label="Segmentation role 3"]','tumor');
      selected=path.join(source,`${patient}_tumor_segmentation.nii.gz`);await click('Import segmentation…');await ready();
      assert(manifest().layers.some(l=>l.kind==='tumor'&&l.label_value===4&&l.role==='target'));
      report.checks.push('UI source label mapping reaches operative target');
      await set('[aria-label="Reconstruction model"]','synthstrip');await click('Reconstruct selected scan');await ready();
      assert(manifest().layers.some(l=>l.kind==='brain'&&l.provenance?.model==='synthstrip'));
      report.checks.push('UI reconstructs real source with local SynthStrip');
      await click('3D workspace');await wait(`document.querySelectorAll('.compact-layer').length>=4`);await ready();
      assert(await js(`document.querySelector('[aria-label="Show all structures"]').checked`),'New operative layers must be visible for inspection');
      await click('Show structures only');await capture('01-anatomy');
      await click('Data');await click('Save patient study file…');await wait(`document.querySelector('footer').innerText.includes('Patient study file saved')`);
      assert(fs.existsSync(savePath));report.checks.push('Portable native study saved');
      selected=savePath;await click('Open');await ready();
      report.checks.push('Portable native study reopened in actual app');
      await click('Rehearse');await wait(`!!document.querySelector('[aria-label="Prepare standard rehearsal environment"]')`);
      await wait(`!!document.querySelector('[aria-label="Prepare standard rehearsal environment"] button')&&!document.querySelector('[aria-label="Prepare standard rehearsal environment"] fieldset:disabled')`);
      await capture('02-prepare');await click('Prepare standard environment');
      await wait(`!document.querySelector('.native-workspace')&&!!document.querySelector('.surgical-panel')&&!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`);
      await wait(`!!document.querySelector('.research-planning')`);
      report.checks.push('UI prepares source-bound surgical environment');await capture('03-environment');
      savePath=path.join(output,'prepared.ressectionlab');await click('Save');
      await wait(`![...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Save')?.disabled`);
      assert(fs.existsSync(savePath));selected=savePath;await click('Open');
      await delay(500);await ready();await click('Rehearse');
      await wait(`!!document.querySelector('.surgical-panel')&&!document.querySelector('.viewer-loading')`);
      report.checks.push('Prepared simulation saved and reopened');await capture('04-simulation-reopen');
      report.preparedCase=savePath;report.passed=report.errors.length===0;
    }catch(error){report.passed=false;report.failure=error.stack;await capture('failure');console.error(error.stack);}
    report.finishedUtc=new Date().toISOString();fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
    app.exit(report.passed?0:1);
  });
});
app.on('will-quit',()=>fs.rmSync(userData,{recursive:true,force:true}));
require('./main.cjs');
