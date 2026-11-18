'use strict';
// Electron SCRIPT writable-native-manifest episode output-dir target [--cold]
// Then repeat with --cold and --cold-midpoint in fresh Electron processes.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {execFileSync} = require('node:child_process');
const {app, dialog, ipcMain} = require('electron');
const {encodeStudyReplay} = require('./study-engine-recording.cjs');
const {inputScript} = require('./verify-eden-planning-helpers.cjs');

const [, , nativeArg, episodeArg, outputArg, target] = process.argv;
assert(nativeArg && episodeArg && outputArg && target, 'Pass native manifest, episode, output directory and target');
const native = fs.realpathSync(nativeArg), episode = fs.realpathSync(episodeArg), output = path.resolve(outputArg);
const testRoot = path.resolve(__dirname, '../../testcases/outputs') + path.sep;
assert(native.startsWith(testRoot), 'Use a disposable native study under testcases/outputs');
const midpoint = process.argv.includes('--cold-midpoint');
const cold = midpoint || process.argv.includes('--cold');
const saved = path.join(output,'recorded-engine.ressectionlab');
const midSaved = path.join(output,'recorded-midpoint.ressectionlab');
const reportFile = path.join(output,midpoint?'cold-midpoint.json':cold?'cold-reopen.json':'recording.json');
assert(!fs.existsSync(reportFile) && (cold || !fs.existsSync(saved)), 'Choose a fresh output directory');
fs.mkdirSync(output,{recursive:true});
const scratch = fs.mkdtempSync(path.join(os.tmpdir(),'study-engine-replay-'));
const studyHash = sha256File(native);
function sha256File(filename){return createHash('sha256').update(fs.readFileSync(filename)).digest('hex');}
const sourceHash = sha256File(episode);
const artifact = JSON.parse(execFileSync('unzip',['-p',episode,'manifest.json'],{maxBuffer:256*1024*1024})).artifacts.surgical_toolbox;
const history = artifact.timeline[artifact.branchId];
const endpoint = history.length;
const width = 1440, height = 900;
const report = {sourceHash,branch:artifact.branchId,target,windowEverVisible:false,states:[]};
const engineStates = new Map();
const handle = ipcMain.handle.bind(ipcMain);
ipcMain.handle = (channel,handler) => handle(channel,async (...args)=>{
  const result = await handler(...args);
  if(channel==='research:openBoundSurgicalReplay') report.binding=result.verifiedStudyReplay;
  if(channel==='research:surgicalRehearsal' && result?.view) {
    const {stateId,slots}=result.view;
    engineStates.set(stateId,{ledger:result.sequenceOutcome,tools:Object.fromEntries(Object.entries(slots).map(
      ([id,{instrument,position,quaternion,inserted}])=>[id,{instrument,position,quaternion,inserted}]))});
  }
  return result;
});
app.setPath('userData',path.join(scratch,'profile'));
if(process.platform==='darwin') app.setActivationPolicy('accessory');
process.argv.push('--automation-hidden',...(cold?['--case',midpoint?midSaved:saved]:['--anatomy',native]));
delete process.env.RESSECTIONLAB_DEV_URL;
dialog.showOpenDialog=async (...args)=>{
  const options=args.at(-1);
  assert.match(options.title,/recorded engine episode/i);
  return {canceled:false,filePaths:[episode]};
};
dialog.showSaveDialog=async ()=>{
  const filename=cold?midSaved:saved;
  assert(!fs.existsSync(filename),'Save verifier output to a new file');
  return {canceled:false,filePath:filename};
};
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const expectedId=i=>i?history[i-1].stateAfter:history[0].stateBefore;
app.on('browser-window-created',(_,win)=>{
  win.on('show',()=>{report.windowEverVisible=true;win.hide();});
  win.webContents.once('did-finish-load',async()=>{
    const js=expr=>win.webContents.executeJavaScript(expr,true);
    const until=async(expr)=>{
      const start=Date.now();
      while(Date.now()-start<240000){
        if(await js(expr)) return;
        const error=await js("document.querySelector('.surgical-panel [role=alert]')?.textContent");
        if(error) throw Error(error);
        await delay(125);
      }
      throw Error('Timed out: '+expr+'\n'+await js('document.body.innerText.slice(-5000)'));
    };
    const click=async label=>{
      const button='[...document.querySelectorAll("button")].find(b=>b.textContent.trim()==='+JSON.stringify(label)+'||b.getAttribute("aria-label")==='+JSON.stringify(label)+')';
      await until('('+button+')?.disabled===false');
      await js('(()=>{const b='+button+';for(let p=b.parentElement;p;p=p.parentElement)if(p.tagName==="DETAILS")p.open=true;b.click()})()');
    };
    const ready=async i=>{
      await until(`document.querySelector('.surgical-panel')?.dataset.recordedPosition==='${i}' &&
        document.querySelector('[data-accepted-tools]')?.dataset.surgicalStateId===${JSON.stringify(expectedId(i))} &&
        document.querySelector('.rl-viewer')?.dataset.surfacesPending==='0' && !document.querySelector('.viewer-loading')`);
    };
    const seek=async i=>{await js(inputScript('Surgical timeline',i));await ready(i);};
    const state=async i=>{
      await ready(i);
      const data=await js(`(()=>{const pane=document.querySelector('[data-accepted-tools]'),checks=[...document.querySelectorAll('.surgeon-view input[type=checkbox]')];
        return {id:pane.dataset.surgicalStateId,camera:JSON.parse(pane.dataset.opticalCamera),tools:JSON.parse(pane.dataset.acceptedTools),
          cutawayOff:checks.length===2&&checks.every(x=>!x.checked),view:pane.dataset.operativeView,removedCells:Number(pane.dataset.removedCells)};})()`);
      assert.equal(data.view,'surgeon');assert(data.cutawayOff,'Microscope must be opaque with cutaway off');
      if(!engineStates.get(data.id)?.ledger) {
        await click('Calculate sequence breakdown');
        await until("!!document.querySelector('[aria-label=\"Complete sequence quantities\"]')");
      }
      const engine=engineStates.get(data.id);
      assert(engine,'Missing authoritative response for displayed state '+i);
      assert.deepEqual(data.tools,engine.tools);
      assert.equal(engine.ledger.stateId,data.id);
      const readout=await js(`document.querySelector('[data-operative-readout-state="${data.id}"]')?.textContent`);
      const residual=engine.ledger.originalTarget.residualMm3;
      assert(readout?.includes('Whole target remaining '+(residual===null?'unknown':residual.toFixed(1)+' mm³')),
        'Whole-target HUD must agree with the state ledger');
      assert(readout.includes('Owned target removed '+engine.ledger.dimensions.targetRemovedMm3.cumulative.toFixed(1)+' mm³'));
      data.readout=readout;
      assert.equal(await js("document.querySelector('[data-operative-locator-state]')?.dataset.operativeLocatorState"),data.id);
      assert.equal(await js("Number(document.querySelector('[data-surgical-metric=originalTargetResidualMm3]')?.textContent.replaceAll(',',''))"),residual);
      return {position:i,...data,ledger:engine.ledger};
    };
    try {
      win.webContents.setBackgroundThrottling(false);
      win.setContentSize(width,height);
      await until('innerWidth==='+width+'&&innerHeight==='+height);
      if(cold) {
        const expected=JSON.parse(fs.readFileSync(path.join(output,'recording.json')));
        await until("!!document.querySelector('.rl-viewer canvas')");
        await click('Rehearse');
        const wanted=midpoint?expected.states[Math.floor(endpoint/2)]:expected.savedState;
        await until("!!document.querySelector('[data-operative-view=surgeon][data-optical-camera]')");
        report.restored=await state(wanted.position);
        assert.deepEqual(report.restored,wanted);
        if(!midpoint) {
          await seek(2);
          await click('Play replay');
          await until("Number(document.querySelector('[aria-label=\"Observed action frames\"]')?.value)>0");
          await click('Pause replay');
          await delay(600);
          const observedState=()=>js(`(()=>{const panel=document.querySelector('[aria-label="Observed microstep replay"]'),pane=document.querySelector('[data-accepted-tools]');
            return {observation:panel?.querySelector('p')?.textContent,state:pane.dataset.surgicalStateId,tools:JSON.parse(pane.dataset.acceptedTools)};})()`);
          report.pausedMicrostep=await observedState();
          assert(report.pausedMicrostep.observation,'Expected an observed engine microstep');
          await delay(500);
          assert.deepEqual(await observedState(),report.pausedMicrostep);
          assert.notEqual(report.pausedMicrostep.state,expected.states[2].id);
          const pausedImage=await win.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true});
          fs.writeFileSync(path.join(output,'paused-engine-microstep.png'),pausedImage.resize({width,height}).toPNG());
          const position=Math.floor(endpoint/2);
          await seek(position);assert.deepEqual(await state(position),expected.states[position]);
          await click('Save');await until("document.body.innerText.includes('Workspace saved locally')");
          assert(fs.existsSync(midSaved));
        }
      } else {
        await until("!!document.querySelector('.rl-viewer canvas')");
        await click('Plan');
        await until(`[...document.querySelectorAll('[aria-label="Planning target"] option')].some(o=>o.value===${JSON.stringify(target)}&&!o.disabled)`);
        await js(inputScript('Planning target',target));
        await until(`document.querySelector('[aria-label="Planning target"]')?.value===${JSON.stringify(target)}&&!document.querySelector('.approach-computing')`);
        await click('Rehearse');console.log('Opening source-bound episode');
        await click('Open recorded engine episode');
        await ready(endpoint);
        assert.equal(report.binding.endpointStateId,expectedId(endpoint));
        assert.equal(report.binding.targetLayerId,target);
        assert(report.binding.snapshotHash,'Expected the verified archival snapshot hash');
        // Enter at the first inserted tool so the microscope looks down the
        // recorded working corridor, rather than centering on a parked tool.
        const firstInsertion=history.findIndex(action=>action.action.operation==='apply')+1;
        assert(firstInsertion>0,'Need an inserted working tool for the microscope');
        await seek(firstInsertion);
        await click('Surgeon / Microscope');
        await until("!!document.querySelector('[data-operative-view=surgeon][data-optical-camera]')");
        await js(inputScript('Viewing angle',15));
        const anchor=await state(firstInsertion);
        const inserted=Object.values(anchor.tools).find(tool=>tool.inserted);
        assert(inserted,'The working tool must be inserted at microscope placement');
        assert.deepEqual(anchor.camera.target,inserted.position);
        report.cameraAnchor={position:firstInsertion,tool:inserted.instrument,target:anchor.camera.target};
        report.fieldCrop=await js(`(()=>{const r=document.querySelector('[data-accepted-tools]').getBoundingClientRect();
          return {x:Math.round(r.x+r.width*.22),y:Math.round(r.y+r.height*.18),
            width:Math.round(r.width*.56),height:Math.round(r.height*.54)};})()`);
        await seek(0);
        for(let i=0;i<=endpoint;i++){
          if(i) {await click('Next recorded state');await ready(i);}
          const sample=await state(i);
          await js("document.querySelector('.rehearsal-replay').scrollIntoView({block:'start'});new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))");
          assert(!win.isVisible());
          const image=await win.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true});
          const frame=path.join(scratch,i+'.png');
          fs.writeFileSync(frame,image.resize({width,height}).toPNG());
          if([0,Math.floor(endpoint/2),endpoint].includes(i)) fs.copyFileSync(frame,path.join(output,'state-'+i+'.png'));
          report.states.push(sample);
          if(i%10===0) console.log('Captured boundary',i);
        }
        assert(report.states.at(-1).ledger.withdrawalComplete);
        assert(report.states.some(s=>Object.values(s.tools).some(t=>t.inserted)));
        await seek(3);assert.deepEqual(await state(3),report.states[3]);
        await click('Previous recorded state');assert.deepEqual(await state(2),report.states[2]);
        await seek(endpoint);assert.deepEqual(await state(endpoint),report.states.at(-1));
        report.savedState=await state(endpoint);
        await click('Save');
        await until("document.body.innerText.includes('Workspace saved locally')");
        assert(fs.existsSync(saved));
        report.media=encodeStudyReplay({framesDirectory:scratch,filename:path.join(output,'engine-preparation.mp4'),report,history});
      }
      assert.equal(sha256File(episode),sourceHash);
      assert.equal(sha256File(native),studyHash);
      assert(!report.windowEverVisible);
      report.passed=true;
    } catch(error) {
      report.passed=false;report.error=String(error.stack??error);
      report.body=await js('document.body.innerText').catch(()=>null);
      console.error(report.error);
    } finally {
      fs.writeFileSync(reportFile,JSON.stringify(report,null,2)+'\n');
      fs.rmSync(scratch,{recursive:true,force:true});app.exit(report.passed?0:1);
    }
  });
});
require('./main.cjs');
