'use strict';
// Actual Electron/GPU UI evidence: never show or focus the app window.
const {app,ipcMain}=require('electron');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..'),base=path.join(root,'testcases');
const input=path.resolve(process.env.DISCLOSURE_CASE||path.join(base,'cases/meningioma-sub-06-standard-v2.ressectionlab'));
const output=path.resolve(process.env.DISCLOSURE_OUTPUT||path.join(base,'outputs/rehearse-disclosure-20261030/proof-local'));
for(const p of [input,output])if(!p.startsWith(base+path.sep))throw Error('Use testcases paths');
if(fs.existsSync(output))throw Error('Preserve previous evidence; choose fresh output');
fs.mkdirSync(output,{recursive:true});app.commandLine.appendSwitch('force-device-scale-factor','1');app.setPath('userData',path.join(output,'hidden-profile'));
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex'),original=sha(input);
process.argv.push('--case',input,'--automation-hidden');
const report={input,output,inputSha256:original,rendererIndexSha256:sha(path.join(root,'desktop/dist/index.html')),windowEverVisible:false,checks:[],requests:[],responses:[],captures:[]};
const handle=ipcMain.handle.bind(ipcMain);ipcMain.handle=(name,handler)=>handle(name,async(...args)=>{if(name==='research:surgicalRehearsal')report.requests.push({at:Date.now(),...args[1]});const result=await handler(...args);if(name==='research:surgicalRehearsal')report.responses.push({command:args[1]?.command,stateId:result?.view?.stateId,branchId:result?.branchId,readiness:result?.fullCaseReadiness,status:result?.status,reason:result?.reason,outcome:result?.outcome,slots:result?.view?.slots,position:result?.position,playback:result?.playback});return result;});
const saveProgress=()=>fs.writeFileSync(path.join(output,'progress.json'),JSON.stringify({checks:report.checks,captures:report.captures,requests:report.requests.map(r=>({command:r.command,operation:r.action?.operation})),responses:report.responses.map(r=>({command:r.command,stateId:r.stateId,position:r.position})),at:new Date().toISOString()},null,2));
const delay=ms=>new Promise(r=>setTimeout(r,ms));
app.on('browser-window-created',(_event,window)=>{window.on('show',()=>{report.windowEverVisible=true;window.hide();});window.webContents.once('did-finish-load',async()=>{
window.webContents.setBackgroundThrottling(false);
const js=async s=>{const result=await window.webContents.executeJavaScript(`try{${s}}catch(error){({proofError:String(error),proofStack:error.stack})}`,true);if(result?.proofError)throw Error(result.proofError+'\n'+result.proofStack);return result;};
window.webContents.debugger.attach('1.3');
await window.webContents.debugger.sendCommand('Emulation.setFocusEmulationEnabled',{enabled:true});
const key=async(name,code,keyCode)=>{await window.webContents.debugger.sendCommand('Input.dispatchKeyEvent',{type:'keyDown',key:name,code,windowsVirtualKeyCode:keyCode,text:name==='Enter'?'\r':' ',unmodifiedText:name==='Enter'?'\r':' '});await window.webContents.debugger.sendCommand('Input.dispatchKeyEvent',{type:'keyUp',key:name,code,windowsVirtualKeyCode:keyCode});await delay(50);};
const wait=async(s,seconds=180)=>{const t=Date.now();while(Date.now()-t<seconds*1000){if(await js(s))return;await delay(100);}throw Error('Timeout '+s+'\n'+await js('document.body.innerText'));};
const click=name=>js(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(name)}||b.getAttribute('aria-label')===${JSON.stringify(name)});if(!b||b.disabled)throw Error('Missing enabled '+${JSON.stringify(name)});b.click()})()`);
const ready=()=>wait(`!!document.querySelector('.surgical-panel')&&!!document.querySelector('[data-surgical-state-id]')&&document.querySelector('.rl-viewer')?.dataset.surfacesPending==='0'&&!document.querySelector('.viewer-loading')&&![...document.querySelectorAll('.surgical-panel [role="status"]')].some(n=>/^(Computing instrument action|Preparing an exact recorded state|Preparing accepted display|Verifying saved source-bound rehearsal)/.test(n.textContent.trim()))`);
const capturePNG=async(name,width,height)=>{const native=await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true}),size=native.getSize();assert.equal(size.width/width,size.height/height,'Uniform native compositor backing scale');assert(size.width>=width&&size.width<=width*3,'Expected normal backing scale');if(size.width!==width)fs.writeFileSync(path.join(output,name+'-native-'+size.width+'x'+size.height+'.png'),native.toPNG());const normalized=size.width===width?native:native.resize({width,height,quality:'best'});assert.deepEqual(normalized.getSize(),{width,height});fs.writeFileSync(path.join(output,name+'.png'),normalized.toPNG());report.captures.push({name:name+'.png',width,height,nativeBackingWidth:size.width,nativeBackingHeight:size.height,normalization:size.width===width?'none':'Native Retina backing normalized to the asserted CSS viewport; original native capture preserved'});};
const capture=async(name,w,h)=>{window.setContentSize(w,h);await delay(250);await ready();await js(`(()=>{for(const e of document.querySelectorAll('*'))if(e.scrollTop)e.scrollTop=0;window.scrollTo(0,0)})()`);await delay(250);const size=await js(`({width:innerWidth,height:innerHeight})`);assert.deepEqual(size,{width:w,height:h});await capturePNG(name,w,h);report[name]=await js(`({body:document.body.innerText,panel:document.querySelector('.surgical-panel').innerText,state:document.querySelector('[data-surgical-state-id]')?.dataset.surgicalStateId,details:[...document.querySelectorAll('.surgical-panel details')].map(d=>({summary:d.querySelector('summary')?.textContent,open:d.open})),controls:[...document.querySelectorAll('.surgical-panel button,.surgical-panel summary')].map(e=>{const r=e.getBoundingClientRect();return {text:e.textContent.trim(),x:r.x,y:r.y,width:r.width,height:r.height,disabled:e.disabled||e.matches(':disabled'),visible:e.checkVisibility(),ancestors:[...function*(n){while(n=n.parentElement)if(n.tagName==='DETAILS')yield {summary:n.querySelector('summary')?.textContent,open:n.open}}(e)]}})})`);};

const state=()=>js(`(()=>{const v=document.querySelector('[data-surgical-state-id]');return {id:v?.dataset.surgicalStateId,tools:JSON.parse(v?.dataset.acceptedTools||'null'),position:document.querySelector('[aria-label="Surgical timeline"]')?.value}})()`);
const setValue=async(label,value)=>{await js(`(()=>{const e=document.querySelector(${JSON.stringify('[aria-label="'+label+'"]')});if(!e)throw Error('Missing field '+${JSON.stringify(label)});const prototype=e.tagName==='SELECT'?HTMLSelectElement.prototype:HTMLInputElement.prototype;Object.getOwnPropertyDescriptor(prototype,'value').set.call(e,${JSON.stringify(String(value))});e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));})()`);await delay(100);};
const reveal=selector=>js(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});if(!e)throw Error('Missing '+${JSON.stringify(selector)});for(let p=e.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;return true})()`);
const pair=async name=>{await capture(name+'-1440x900',1440,900);await capture(name+'-1050x720',1050,720);};
async function computation(){
  const started=report.requests.length;await click('Start primary rehearsal');
  await wait(`[...document.querySelectorAll('.full-case-rehearsal button')].some(b=>b.textContent.trim()==='Pause computation'&&!b.matches(':disabled'))`,30);
  report.computationPending=[];
  for(const [width,height] of [[1440,900],[1050,720]]){
    window.setContentSize(width,height);await delay(100);await js(`document.querySelectorAll('*').forEach(e=>{if(e.scrollTop)e.scrollTop=0})`);
    const controls=await js(`[...document.querySelectorAll('.surgical-panel button')].filter(b=>/^(Pause computation|Cancel)$/.test(b.textContent.trim())&&b.checkVisibility()).map(b=>({text:b.textContent.trim(),disabled:b.matches(':disabled'),y:b.getBoundingClientRect().y,height:b.getBoundingClientRect().height}))`);
    assert(controls.some(c=>c.text==='Pause computation'&&!c.disabled&&c.y>=48&&c.y+c.height<=height-26),'Pause fully visible during actual bounded computation');assert(controls.some(c=>c.text==='Cancel'&&!c.disabled&&c.y>=48&&c.y+c.height<=height-26),'Cancel fully visible while computation pending');
    const name=`08-computing-${width}x${height}`;await capturePNG(name,width,height);report.computationPending.push({width,height,controls});
  }
  await click('Pause computation');await ready();await wait(`![...document.querySelectorAll('.full-case-rehearsal button')].some(b=>b.textContent.trim()==='Pause computation')`,30);
  await wait(`document.querySelector('.full-case-rehearsal')?.innerText.includes('computation paused')`,30);report.pausedLabelVerified=true;
  const count=report.requests.filter(r=>r.command==='fullCase').length;await delay(500);assert.equal(report.requests.filter(r=>r.command==='fullCase').length,count);assert(count>0,'Real native bounded full-case request issued');report.fullCaseRequests=report.requests.slice(started);report.checks.push('Actual bounded primary computation exposes enabled Pause and Cancel at both sizes; Pause starts no later slices');await pair('09-paused-computation');
}
async function after(){
  const initial=await state();report.initial=initial;
  if(process.env.DISCLOSURE_COMPUTATION_ONLY==='1'){await computation();return;}
  const missing=report.responses.at(-1)?.readiness?.status!=='ready';
  report.primaryActionLocations=await js(`(()=>{return [...document.querySelectorAll('.surgical-panel button')].filter(e=>/Start primary rehearsal|Start vascular injury|Preview|Apply$|Cancel$|Play replay/.test(e.textContent.trim())).map(e=>({text:e.textContent.trim(),visible:e.checkVisibility(),disabled:e.matches(':disabled'),y:e.getBoundingClientRect().y}))})()`);
  for(const summary of ['Pose, stroke and tool settings','Advanced research workflows','Environment and source setup','Exchange instrument'])assert(await js(`(()=>{const d=[...document.querySelectorAll('.surgical-panel details')].find(d=>d.querySelector('summary')?.textContent===${JSON.stringify(summary)});return !!d&&!d.open})()`),'Default collapsed '+summary);
  // Native disclosures are keyboard operated and must preserve mounted editable drafts.
  await reveal('[aria-label="Search instruments"]');await setValue('Search instruments','cranial');
  await reveal('[aria-label="Import mesh resolution"]');await setValue('Import mesh resolution','1.7');
  await reveal('[aria-label="Tool position R"]');await setValue('Tool position R',initial.tools.left.position[0]+.25);
  await reveal('[aria-label="Action duration"]');await js(`document.querySelector('[aria-label="Action duration"]').focus()`);await setValue('Action duration','');
  report.mappingDraft=await js(`(()=>{const e=document.querySelector('.surgical-map-row select');return e?{label:e.getAttribute('aria-label'),value:e.value}:null})()`);
  if(report.mappingDraft)await setValue(report.mappingDraft.label,report.mappingDraft.value==='unknown'?'soft_tissue':'unknown');
  await js(`window.disclosureDraftNodes=[document.querySelector('[aria-label="Search instruments"]'),document.querySelector('[aria-label="Import mesh resolution"]'),document.querySelector('[aria-label="Tool position R"]'),document.querySelector('.surgical-map-row select'),document.querySelector('[aria-label="Action duration"]')];window.disclosureDraftValues=window.disclosureDraftNodes.map(n=>n?.value)`);
  const count=report.requests.length,beforeToggle=await state();
  const summaries=await js(`(()=>{const a=[...document.querySelectorAll('.surgical-panel details')];return [...new Set(a)].map((e,i)=>{e.dataset.disclosureProof=String(i);return {id:i,summary:e.querySelector('summary')?.textContent}})})()`);
  report.keyboardDisclosures=[];
  for(const item of summaries){
    await js(`(()=>{const d=document.querySelector('[data-disclosure-proof="${item.id}"]');for(let p=d.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;d.open=false;d.querySelector('summary').focus()})()`);
    report.keyboardFocus=await js(`({hasFocus:document.hasFocus(),active:document.activeElement?.outerHTML})`);
    await key('Enter','Enter',13);
    assert(await js(`document.querySelector('[data-disclosure-proof="${item.id}"]').open`),'Enter opens '+item.summary);
    await key(' ','Space',32);
    assert.equal(await js(`document.querySelector('[data-disclosure-proof="${item.id}"]').open`),false,'Space closes '+item.summary);
    report.keyboardDisclosures.push(item.summary);
  }
  assert.equal(report.requests.length,count,'Disclosure toggles must issue no engine request');assert.deepEqual(await state(),beforeToggle);
  assert(await js(`window.disclosureDraftNodes.every(n=>n?.isConnected)&&window.disclosureDraftValues.every((v,i)=>v===window.disclosureDraftNodes[i]?.value)`),'Draft controls remain mounted with same values');
  await setValue('Action duration',1);
  if(report.mappingDraft)await setValue(report.mappingDraft.label,report.mappingDraft.value);
  await setValue('Search instruments','');await js(`document.querySelectorAll('.surgical-panel details').forEach(d=>d.open=false)`);saveProgress();report.checks.push('Enter/Space disclosure toggles preserve mounted drafts including an empty Numeric draft and accepted state with no engine command');
  if(process.env.DISCLOSURE_TOGGLES_ONLY==='1')return;
  await reveal('[aria-label="Instrument action"]');await setValue('Instrument action','inspect');
  await click('Preview');
  report.pendingControls=await js(`[...document.querySelectorAll('.surgical-panel button')].filter(b=>/Cancel|Pause/.test(b.textContent)).map(b=>({text:b.textContent.trim(),disabled:b.matches(':disabled'),visible:b.checkVisibility()}))`);
  await wait(`document.querySelector('.surgical-panel')?.textContent.includes('Preview · uncommitted')`);await ready();assert.equal(report.responses.filter(r=>r.command!=='preview').at(-1)?.stateId,initial.id,'Preview retains accepted engine cursor');
  assert(await js(`[...document.querySelectorAll('.manual-actions button')].some(b=>b.textContent.trim()==='Apply'&&!b.matches(':disabled'))`),'Apply stays enabled in editable preview');
  await pair('03-preview');await click('Cancel preview');await ready();assert.equal((await state()).id,initial.id);report.checks.push('Real native inspect preview, Apply enabled, Cancel restores accepted anatomy');
  if(missing){report.checks.push('Missing evidence correctly keeps full-case start disabled');assert(await js(`[...document.querySelectorAll('.full-case-rehearsal button')].filter(b=>b.textContent.includes('Start')).every(b=>b.matches(':disabled'))`));await pair('04-missing-evidence');return;}
  await setValue('Instrument action','place');await setValue('Action duration',1);await click('Apply');await ready();
  await wait(`Number(document.querySelector('[aria-label="Surgical timeline"]').max)>=1`);
  await setValue('Instrument action','move');await setValue('Tool position S',initial.tools.left.position[2]-1);await click('Apply');await ready();
  await wait(`Number(document.querySelector('[aria-label="Surgical timeline"]').max)>=2`);const inserted=await state();assert.notEqual(inserted.id,initial.id);assert(Object.values(inserted.tools).some(t=>t.inserted),'Native checked free-space move marks tool inserted');
  await reveal('.surgical-toolbox');await js(`(()=>{const b=[...document.querySelectorAll('.surgical-toolbox button')].find(b=>!b.matches(':disabled')&&b.textContent.includes('CUSA'));if(!b)throw Error('Missing exchange target');b.click()})()`);await ready();await wait(`document.querySelector('.surgical-panel').textContent.includes('WITHDRAW_BEFORE_EXCHANGE')`);assert.equal((await state()).id,inserted.id);await js(`document.querySelectorAll('.surgical-panel details').forEach(d=>d.open=false)`);await pair('05-refused-exchange');saveProgress();report.checks.push('Actual inserted-tool exchange refusal is visible without changing accepted state');
  await setValue('Surgical timeline',0);await ready();await setValue('Replay speed',.25);await click('Play replay');await wait(`!!document.querySelector('[aria-label="Observed microstep replay"]')`,30);
  if(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Pause replay')`))await click('Pause replay');await ready();
  await setValue('Observed action frames',1);await wait(`document.querySelector('[aria-label="Observed action frames"]')?.value==='1'`);await ready();const frameState=await state();report.readOnlyReceipt=report.responses.findLast(r=>r.playback?.readOnly&&r.playback.frameIndex===1&&r.stateId===frameState.id);assert(report.readOnlyReceipt,'Displayed interior native microstep matches a real frame1 receipt despite background prefetch');
  assert(await js(`[...document.querySelectorAll('.manual-actions button')].filter(b=>b.textContent.trim()==='Preview'||b.textContent.trim()==='Apply').every(b=>b.matches(':disabled'))`),'Observed replay disables manual mutation');await pair('06-read-only-replay');report.checks.push('Actual native microstep replay labels read-only and disables manual Preview/Apply; Pause remains available');
  await click('Return to recorded boundary');await ready();assert.equal((await state()).id,initial.id,'Top boundary recovery restores initial accepted state');
  await setValue('Surgical timeline',2);await ready();await click('Surgeon / Microscope');await ready();
  await js(`(()=>{const l=[...document.querySelectorAll('.surgeon-view label')].find(l=>l.textContent.includes('Inspection-only transparency'));if(!l)throw Error('Missing inspection transparency');for(let p=l.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;l.querySelector('input').click();for(const d of document.querySelectorAll('.surgeon-view details'))d.open=false})()`);await ready();
  assert(await js(`[...document.querySelectorAll('.surgeon-view strong')].some(e=>e.textContent.includes('INSPECTION ONLY')&&e.checkVisibility())`),'Inspection indicator survives optical disclosure collapse');assert(await js(`(()=>{const e=document.querySelector('[aria-label="Current rehearsal state"]');return e.innerText.includes('Inspection only')&&e.getBoundingClientRect().bottom<innerHeight-26})()`),'Inspection notice remains above fold');await pair('07-inspection-collapsed');report.checks.push('Inspection-only transparency warning remains visible with optical settings collapsed');
  const opticalState=await state(),opticalRequests=report.requests.length;
  const opticalToggle=async text=>{await js(`(()=>{const l=[...document.querySelectorAll('.surgeon-view label')].find(l=>l.textContent.trim()===${JSON.stringify(text)});if(!l)throw Error('Missing '+${JSON.stringify(text)});for(let p=l.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;l.querySelector('input').click()})()`);await delay(100);};
  await opticalToggle('Inspection-only transparency / hidden anatomy');await opticalToggle('Inspection-only cutaway');
  await js(`document.querySelectorAll('.surgeon-view details').forEach(d=>d.open=false)`);await ready();
  assert(await js(`document.querySelector('[aria-label="Current rehearsal state"]').innerText.includes('Inspection only · cutaway')`),'Independent cutaway notice survives optical settings collapse');assert.equal(report.requests.length,opticalRequests,'Optical settings do not issue engine commands');assert.deepEqual(await state(),opticalState,'Optical settings do not change accepted state');await pair('07b-cutaway-collapsed');report.checks.push('Independent cutaway badge survives collapsed optical settings; optical toggles issue no engine commands and preserve accepted state');
}

try{window.setContentSize(1440,900);await wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`);await click('Rehearse');await ready();await capture('01-default-1440x900',1440,900);await capture('02-default-1050x720',1050,720);
if(process.env.DISCLOSURE_AFTER==='1')await after();
report.checks.push('Actual hidden default Rehearse captures at exact 1440×900 and minimum 1050×720 CSS pixels');assert.equal(sha(input),original);report.sourcePreserved=true;assert(!report.windowEverVisible);report.passed=true;
}catch(error){report.error=String(error);report.sourcePreserved=sha(input)===original;report.body=await js('document.body.innerText').catch(()=>null);fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());console.error(error);}
fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,captures:report.captures,error:report.error}));app.exit(report.passed?0:1);
});});require('./main.cjs');
