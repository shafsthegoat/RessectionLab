'use strict';
// Actual native application, hidden for its entire lifetime. Never show/focus.
const {app,dialog}=require('electron');
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..');
const input=fs.realpathSync(path.resolve(process.env.CONSEQUENCE_EDITOR_CASE||path.join(root,'testcases/cases/UCSF-PDGM-0007-consequence-comparison.ressectionlab')));
const output=path.resolve(process.env.CONSEQUENCE_EDITOR_OUTPUT||path.join(root,'testcases/outputs/consequence-editor-20261030/desktop'));
for(const value of [input,output])assert(value.startsWith(path.join(root,'testcases')+path.sep),'Use ignored testcases paths');
assert(!fs.existsSync(output),'Preserve prior evidence; use a new output directory');fs.mkdirSync(output,{recursive:true});
const sha=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const original=sha(input),saved=path.join(output,'editor-saved.ressectionlab');
app.setPath('userData',path.join(output,'user-data'));
process.argv.push('--case',input,'--automation-hidden');
dialog.showSaveDialog=async()=>({canceled:false,filePath:saved});
const report={schema:'resectionlab.consequence-editor-hidden-native.v1',input,output,
  windowEverVisible:false,width:1440,height:900,states:[],checks:[],errors:[],passed:false};
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
app.on('browser-window-created',(_event,window)=>{
  window.on('show',()=>{report.windowEverVisible=true;window.hide();});
  window.webContents.once('did-finish-load',async()=>{
    window.webContents.setBackgroundThrottling(false);window.setContentSize(1440,900);
    const js=code=>window.webContents.executeJavaScript(code,true);
    const wait=async(expression,seconds=600)=>{const tick=Date.now();while(Date.now()-tick<seconds*1000){if(await js(expression))return;await delay(150);}throw Error('Timeout '+expression+'\n'+await js('document.body.innerText'));};
    const click=async label=>{await js(`(()=>{const b=[...document.querySelectorAll('button,input[type=checkbox]')].find(b=>b.textContent.trim()===${JSON.stringify(label)}||b.getAttribute('aria-label')===${JSON.stringify(label)});if(!b||b.disabled)throw Error('Missing enabled button '+${JSON.stringify(label)});b.click();})()`);await delay(150);};
    const ready=()=>wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')&&document.querySelector('.rl-viewer').dataset.surfacesPending==='0'&&![...document.querySelectorAll('.surgical-panel [role="status"]')].some(n=>/^(Computing instrument action|Preparing an exact recorded state|Preparing accepted display|Verifying saved source-bound rehearsal)/.test(n.textContent.trim()))`);
    const open=async prefix=>{await js(`(()=>{const s=[...document.querySelectorAll('summary')].find(s=>s.textContent.trim().startsWith(${JSON.stringify(prefix)}));if(!s)throw Error('Missing details '+${JSON.stringify(prefix)});s.parentElement.open=true;})()`);};
    const inputValue=async(label,value)=>{await js(`(()=>{const e=document.querySelector('input[aria-label='+${JSON.stringify(JSON.stringify(label))}+']');if(!e||e.disabled)throw Error('Missing enabled input');Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(e,${JSON.stringify(String(value))});e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));})()`);await delay(150);};
    const editorState=()=>js(`(()=>{const e=document.querySelector('.consequence-editor'),s=e?.querySelector('[data-consequence-analysis-status]'),p=document.querySelector('[data-operative-view="surgeon"][data-surgical-state-id]');return {stateId:p?.dataset.surgicalStateId,position:Number(document.querySelector('[aria-label="Surgical timeline"]')?.value),branch:document.querySelector('[aria-label="Surgical branch"]')?.value,status:s?.dataset.consequenceAnalysisStatus,decision:s?.textContent,minimum:document.querySelector('[aria-label="Minimum additional target removal mm³"]')?.value,budget:document.querySelector('[aria-label="Cumulative non-target removal budget mm³"]')?.value,motor:document.querySelector('[aria-label="Require Motor encounter evidence"]')?.checked,rows:[...(e?.querySelectorAll('[data-consequence-analysis-branch]')??[])].map(r=>({branch:r.dataset.consequenceAnalysisBranch,text:r.textContent})),viewerState:document.querySelector('[data-operative-readout-state]')?.dataset.operativeReadoutState}})()`);
    const capture=async(name,selector='.consequence-editor')=>{
      await ready();await js(`document.querySelector(${JSON.stringify(selector)}).scrollIntoView({block:'start'})`);await delay(250);
      const before=await editorState();
      const png=(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG();
      fs.writeFileSync(path.join(output,name+'.png'),png);assert.deepEqual(await editorState(),before);
      assert.equal(window.isVisible(),false);report.states.push({name,...before});return before;
    };
    const captureCard=async name=>{
      const selector='[data-consequence-analysis-branch="short-hold"]';
      const current=await capture(name,selector);
      const layout=await js(`(()=>{const e=document.querySelector(${JSON.stringify(selector)}),r=e.getBoundingClientRect();return {width:innerWidth,height:innerHeight,cardWidth:r.width,cardHeight:r.height,clientWidth:e.clientWidth,scrollWidth:e.scrollWidth,left:r.left,right:r.right,top:r.top,bottom:r.bottom,text:e.textContent}})()`);
      assert(layout.scrollWidth<=layout.clientWidth+1,'Alternative card must not require horizontal scrolling');
      assert(layout.left>=0&&layout.right<=layout.width,'Alternative card fits viewport horizontally');
      assert(layout.top>=0&&layout.bottom<=layout.height,'Complete alternative card fits viewport vertically');
      for(const text of ['Additional target','Residual target','Cumulative non-target','Modeled duration','8.00','18800.00','4940.00','Inspect short-hold'])assert(layout.text.includes(text),'Visible card retains '+text);
      (report.cardLayouts??=[]).push({name,...layout});
      return current;
    };
    const assess=async(revision,feasible)=>{
      await click('Reassess checked alternatives');await ready();
      await wait(`document.querySelector('[data-consequence-analysis-status="current"]')?.textContent.includes('revision ${revision}')`);
      const state=await editorState();
      assert.match(state.decision,new RegExp('Eligible continuations: '+(feasible?'short-hold, sustained-hold':'none')));
      assert.match(state.decision,new RegExp('declared research task '+(feasible?'met':'not met')));
      assert.equal(state.branch,'retraction-start');assert.equal(state.position,15);
      return state;
    };
    try{
      await wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')`);
      await click('Rehearse');await ready();
      if(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify saved replay'&&!b.disabled)`)){await click('Verify saved replay');await ready();}
      else if(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Verify replay for planning'&&!b.disabled)`)){await click('Verify replay for planning');await ready();}
      await click('Surgeon / Microscope');await ready();
      await open('Advanced research workflows');await open('Recorded consequence comparison');
      await wait(`!!document.querySelector('.consequence-editor')`);
      if(process.env.CONSEQUENCE_EDITOR_REOPEN_EXPECT){
        const expected=JSON.parse(fs.readFileSync(process.env.CONSEQUENCE_EDITOR_REOPEN_EXPECT,'utf8'));
        const current=await capture('01-cold-reopened-policy');
        for(const key of ['stateId','position','branch','status','decision','minimum','budget','motor','rows'])assert.deepEqual(current[key],expected.savedView[key],key+' survives cold reopen');
        await capture('02-cold-reopened-decision','[data-consequence-analysis-status]');await captureCard('03-cold-reopened-alternative-card');
        report.checks.push('Fresh native reopen and explicit replay verification preserve the exact common checkpoint, editor inputs and current sealed decision');
      }else{
        if(await js(`[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Open common checkpoint'&&!b.disabled)`)){await click('Open common checkpoint');await ready();}
        const initial=await capture('01-common-checkpoint-editor');
        assert.equal(initial.branch,'retraction-start');assert.equal(initial.position,15);
        await inputValue('Minimum additional target removal mm³',8);
        await inputValue('Cumulative non-target removal budget mm³',4940);
        await assess(1,true);await capture('02-eight-target-editor');await capture('03-eight-target-decision','[data-consequence-analysis-status]');
        await inputValue('Minimum additional target removal mm³',9);
        assert.equal((await editorState()).status,'historical','Editing must immediately invalidate current decision');
        await capture('04-edited-draft-not-current','[data-consequence-analysis-status]');
        await assess(2,false);await capture('05-nine-target-stop','[data-consequence-analysis-status]');await captureCard('05b-nine-target-alternative-card');
        await inputValue('Minimum additional target removal mm³',8);
        await inputValue('Cumulative non-target removal budget mm³',4939);
        await assess(3,false);await capture('06-whole-access-budget-stop','[data-consequence-analysis-status]');
        await inputValue('Cumulative non-target removal budget mm³',4940);
        await click('Require Motor encounter evidence');
        await assess(4,false);await capture('07-required-motor-editor');
        const unknown=await capture('08-unknown-motor-stop','[data-consequence-analysis-status]');
        assert(unknown.rows.every(r=>r.text.includes('Motor encounter: Unknown')));
        await captureCard('08b-unknown-motor-alternative-card');
        assert.match(unknown.decision,/STOP with residual: stop-at-model-limit/);assert.match(unknown.decision,/STOP cleanup executable/);
        report.checks.push('Actual editor changes eligibility from two continuations to none for minimum9, cumulative budget4939 and required unknown motor evidence; draft edits invalidate current decision');
        report.savedView=await editorState();await click('Save');await wait(`document.body.innerText.includes('Workspace saved locally')`);
        assert(fs.existsSync(saved));report.saved=saved;
        await click('Inspect stop-at-model-limit');await ready();
        await wait(`document.querySelector('[aria-label="Surgical branch"]')?.value==='stop-at-model-limit'&&Number(document.querySelector('[aria-label="Surgical timeline"]').value)===17`);
        const stopped=await capture('09-checked-stop-replay-historical','[data-consequence-analysis-status]');
        assert.equal(stopped.status,'historical');assert.notEqual(stopped.stateId,initial.stateId);
        assert(!stopped.decision.includes('Current reassessment'));
        await click('Open common checkpoint');await ready();
        await wait(`document.querySelector('[aria-label="Surgical branch"]')?.value==='retraction-start'`);
        report.checks.push('Save retains current common-checkpoint policy; checked STOP opens canonical cleanup replay and labels the baseline assessment historical');
        const beforeResize=await editorState();window.setContentSize(1180,760);await delay(300);
        const compact=await captureCard('10-compact-1180-alternative-card');
        assert.equal(compact.stateId,beforeResize.stateId);assert.equal(compact.status,'current');
        report.checks.push('Complete alternative cards, source counts, required Unknown evidence and grouped exclusions fit without horizontal scrolling at both 1440×900 and 1180×760 logical pixels');
      }
      report.sourcePreserved=sha(input)===original;assert(report.sourcePreserved);
      report.passed=!report.windowEverVisible;
    }catch(error){report.errors.push(error.stack||String(error));report.body=await js('document.body.innerText').catch(()=>null);
      try{fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true})).toPNG());}catch{}
    }
    fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify({passed:report.passed,saved:report.saved,checks:report.checks,errors:report.errors}));app.exit(report.passed?0:1);
  });
});
require('./main.cjs');
