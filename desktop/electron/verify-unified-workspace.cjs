'use strict';
const {app,dialog}=require('electron'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),assert=require('node:assert/strict');
const {verificationDirectory}=require('./verification-output.cjs');
const {studyScanSelectionScript,studyLayerVisibilityScript,studyLayerSelectionScript}=require('./verify-study-ui-helpers.cjs');
process.argv.push('--automation-hidden');
if(process.platform==='darwin')app.setActivationPolicy('accessory');
const fatalStartup=error=>{console.error(error?.stack||String(error));app.exit(1)};
dialog.showErrorBox=(title,message)=>fatalStartup(Error(title+': '+message));
process.on('uncaughtException',fatalStartup);process.on('unhandledRejection',fatalStartup);
const sourceIndex=process.argv.indexOf('--anatomy');assert(sourceIndex>=0&&process.argv[sourceIndex+1],'Pass --anatomy /absolute/patient-anatomy.json');
const sourceDocument=JSON.parse(fs.readFileSync(process.argv[sourceIndex+1],'utf8'));
const output=path.resolve(process.env.UNIFIED_VERIFY_OUTPUT||path.join(verificationDirectory(path.resolve(__dirname,'../..')),'unified-workspace'));fs.mkdirSync(output,{recursive:true});
app.setPath('userData',fs.mkdtempSync(path.join(os.tmpdir(),'rslab-unified-test-')));
let nextDialogPath=null;const originalOpen=dialog.showOpenDialog;dialog.showOpenDialog=async(...args)=>nextDialogPath?{canceled:false,filePaths:[nextDialogPath]}:originalOpen(...args);
const report={errors:[],checks:{},windowEverVisible:false},pause=ms=>new Promise(r=>setTimeout(r,ms));
app.on('browser-window-created',(_event,window)=>{
 window.on('show',()=>{report.windowEverVisible=true;window.hide()});
 window.webContents.on('console-message',(_event,d)=>{if(d.level==='error')report.errors.push(d.message)});
 window.webContents.once('did-finish-load',async()=>{
  window.webContents.setBackgroundThrottling(false);
  const js=code=>window.webContents.executeJavaScript(code,true);
  const click=text=>js(`(()=>{const b=[...document.querySelectorAll('button')].find(e=>e.textContent.trim()===${JSON.stringify(text)}||e.getAttribute('aria-label')===${JSON.stringify(text)}||e.querySelector('span')?.textContent.trim()===${JSON.stringify(text)});if(!b||b.disabled)throw Error('Missing enabled button '+${JSON.stringify(text)});for(let node=b.parentElement;node;node=node.parentElement)if(node.tagName==='DETAILS')node.open=true;b.scrollIntoView({block:'nearest'});b.click()})()`);
  const waitFor=async(expr,seconds=240)=>{await pause(200);for(let i=0;i<seconds*4;i++){if(await js(expr))return;await pause(250)}throw Error('Timed out: '+expr)};
  const capture=async name=>{assert.equal(window.isVisible(),false,'Verification must stay hidden');fs.writeFileSync(path.join(output,name+'.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());console.log('Captured '+name)};
  try{
   await waitFor(`!!document.querySelector('.native-workspace .rl-viewer canvas')&&!document.querySelector('.viewer-loading')`);await pause(1500);await click('Anatomy');await waitFor(`!!document.querySelector('.study-inspector-anatomy .native-structures')`);
   report.checks.initial=await js(`({native:!!document.querySelector('.native-workspace'),separateDestination:!![...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Native anatomy'),layout:document.querySelector('.rl-viewer').dataset.layout,panes:document.querySelectorAll('.rl-viewer-slice').length,controls:[...document.querySelectorAll('.rl-primary-tools button')].map(b=>b.textContent),status:document.querySelector('footer').textContent,layers:${studyLayerVisibilityScript()}})`);
   report.checks.inventoryComplete=JSON.stringify(report.checks.initial.layers.map(layer=>layer.id).sort())===JSON.stringify([...sourceDocument.layers.map(layer=>layer.id),...(sourceDocument.tractography??[]).map(tract=>tract.id)].sort());
   await capture('unified-default');await click('Appearance');report.checks.scanOn=await js(`document.querySelector('input[aria-label="Scan volume"]').checked`);await js(`document.querySelector('button[aria-label="Close view controls"]').click()`);
   await click('Layout');await click('Four views');await pause(500);report.checks.fourViews=await js(`document.querySelector('.rl-viewer').dataset.layout`);await capture('unified-four-views');await click('Layout');await click('3D + slices');await pause(500);
   await js(`window.unifiedFrames=[];window.unifiedRefinement=[];document.addEventListener('anatomy-frame',e=>window.unifiedFrames.push(e.detail));document.addEventListener('anatomy-refined',e=>window.unifiedRefinement.push(e.detail));`);
   const r=await js(`(()=>{const r=document.querySelector('.rl-viewer-anatomy-image').getBoundingClientRect();return {x:r.x+r.width*.5,y:r.y+r.height*.5}})()`);
   const presented=[];let collecting=true;window.webContents.beginFrameSubscription(true,()=>{if(collecting)presented.push(performance.now())});
   window.webContents.sendInputEvent({type:'mouseDown',x:Math.round(r.x),y:Math.round(r.y),button:'left',clickCount:1});
   for(let i=1;i<=60;i++){window.webContents.sendInputEvent({type:'mouseMove',x:Math.round(r.x+i*2),y:Math.round(r.y+10*Math.sin(i/10)),modifiers:['leftButtonDown']});await pause(16)}
   window.webContents.sendInputEvent({type:'mouseUp',x:Math.round(r.x+120),y:Math.round(r.y),button:'left',clickCount:1});collecting=false;window.webContents.endFrameSubscription();await pause(1500);report.presentedFrames=presented;
   report.frames=await js('window.unifiedFrames');report.refinement=await js('window.unifiedRefinement');
   if(process.env.UNIFIED_REVIEW_TEST){
    const filename=process.argv[process.argv.indexOf('--anatomy')+1];
    if(!filename.includes('ressection-unified-patient-review-'))throw Error('Review verification requires a disposable patient copy');
    const initial=JSON.parse(fs.readFileSync(filename,'utf8'));
    const ready=async()=>{await waitFor(`!!document.querySelector('.rl-viewer canvas')&&document.querySelector('.rl-viewer').dataset.rendererReady==='true'&&document.querySelector('.rl-viewer').dataset.surfacesPending==='0'&&!document.querySelector('.viewer-loading,.rl-viewer-preparing,[data-layer-pending="true"]')`);await pause(350)};
    const selectScan=async id=>{await js(studyScanSelectionScript(id));await ready()};
    await selectScan('ct-bone');report.checks.ctSwitch=await js(`({title:document.querySelector('.workspace-heading h2').textContent,layers:${studyLayerVisibilityScript()}.filter(layer=>layer.checked)})`);await capture('unified-ct');
    await selectScan('t1ce');
    await click('Show all brain regions');await ready();await pause(500);await capture('unified-regions');
    report.checks.regions=await js(`${studyLayerVisibilityScript()}.filter(layer=>layer.checked).length`);
    await click('Hide all brain regions');await ready();
    await js(studyLayerSelectionScript([sourceDocument.layers.find(layer=>layer.id==='vessels')?.label??'vessels']));await click('Edit segmentation');await click('Review fragments');
    await waitFor(`document.querySelectorAll('.component-list input').length>0`);report.checks.components=await js(`document.querySelectorAll('.component-list input').length`);
    await js(`document.querySelector('.component-list input').click()`);await ready();await capture('unified-fragment-selection');
    await click('Preview removal (1)');await ready();report.checks.preview=await js(`document.querySelector('.unified-correction').textContent`);await capture('unified-cleanup-preview');
    await click('Undo');await ready();report.checks.undo=await js(`document.querySelector('.unified-correction').textContent`);
    await click('Redo');await ready();await click('Show original');await ready();await click('Reject');await ready();await click('Edit segmentation');
    report.checks.rejectedUnchanged=JSON.parse(fs.readFileSync(filename,'utf8')).snapshot_id===initial.snapshot_id;
    await click('Review fragments');await waitFor(`document.querySelectorAll('.component-list input').length>0`);
    await js(`document.querySelector('.component-list input').click()`);await ready();await click('Preview removal (1)');await ready();
    await js(`(()=>{const e=document.querySelector('input[aria-label="Correction author"]');Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(e,'Electron verification');e.dispatchEvent(new Event('input',{bubbles:true}))})()`);
    await waitFor(`[...document.querySelectorAll('button')].some(b=>b.textContent==='Accept correction'&&!b.disabled)`);await click('Accept correction');await ready();
    const after=JSON.parse(fs.readFileSync(filename,'utf8'));report.checks.acceptedRevision=after.snapshot_id!==initial.snapshot_id;
    report.checks.modelOutputPreserved=after.layers.find(l=>l.id==='vessels').provenance!=null&&fs.existsSync(path.join(path.dirname(filename),initial.layers.find(l=>l.id==='vessels').data.path));
    await click('Save');await waitFor(`document.querySelector('footer').textContent.includes('Workspace saved')`);
    const saved=JSON.parse(fs.readFileSync(path.join(path.dirname(filename),'workspace-view.json'),'utf8'));report.checks.saved=saved.snapshotId===after.snapshot_id&&saved.seriesId==='t1ce'&&!!saved.states['ct-bone'];
    await js(`document.querySelector('button[aria-label="Cases"]').click()`);await pause(500);
    await js(`(()=>{const b=[...document.querySelectorAll('button')].find(e=>e.querySelector('strong')?.textContent===${JSON.stringify(initial.patient_id)});if(!b)throw Error('Saved collection missing');b.click()})()`);await ready();report.checks.reopened=await js(`document.querySelector('.native-workspace')?.dataset.nativeSeries==='t1ce'`);
   }
   if(process.env.UNIFIED_LEGACY_CASE){
    nextDialogPath=process.env.UNIFIED_LEGACY_CASE;await click('Open');
    await waitFor(`!document.querySelector('.native-workspace')&&!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`);nextDialogPath=null;await pause(1000);await capture('unified-legacy');
    report.checks.legacy=await js(`({separateDestination:!![...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Native anatomy'),panes:document.querySelectorAll('.rl-viewer-slice').length})`);
    nextDialogPath=process.argv[process.argv.indexOf('--anatomy')+1];await click('Open');await waitFor(`!!document.querySelector('.native-workspace .rl-viewer canvas')&&!document.querySelector('.viewer-loading')`);nextDialogPath=null;
   }
   window.setSize(1050,720);await pause(1000);await capture('unified-minimum');report.checks.minimum=await js(`({horizontalOverflow:document.documentElement.scrollWidth>innerWidth,viewer:document.querySelector('.rl-viewer').getBoundingClientRect().toJSON()})`);
   report.passed=(!process.env.UNIFIED_REVIEW_TEST||(report.checks.rejectedUnchanged&&report.checks.acceptedRevision&&report.checks.saved&&report.checks.reopened))&&!report.windowEverVisible&&report.errors.length===0&&report.checks.scanOn&&report.checks.inventoryComplete&&!report.checks.initial.separateDestination;
  }catch(e){report.passed=false;report.failure=e.stack;await capture('failure')}
  fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,errors:report.errors,failure:report.failure,output}));app.quit();
 });
});require('./main.cjs');
