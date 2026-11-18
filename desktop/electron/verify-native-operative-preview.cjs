'use strict';
// Invoke with Electron, a disposable native manifest, a new evidence directory and a mode.
// preview exercises fused scan switching and operative playback. --video records the
// real hidden compositor. --export FILE saves a portable study. Reopen it in a
// second process with mode reopen and --open-archive FILE.
// baseline, prepare, stale and no-target exercise simpler study states.
const {app, dialog, ipcMain} = require('electron');
const {execFileSync,spawn} = require('node:child_process');
const {once} = require('node:events');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {studyScanSelectionScript} = require('./verify-study-ui-helpers.cjs');
const {inputScript} = require('./verify-eden-planning-helpers.cjs');
const root = path.resolve(__dirname, '../..');
const [, , manifestArg, outputArg, mode = 'preview'] = process.argv;
assert(['preview','baseline','prepare','no-target','stale','reopen'].includes(mode),'Unknown verification mode');
assert(manifestArg && outputArg, 'Pass a disposable native manifest, new evidence directory, and baseline or preview');
const manifest = fs.realpathSync(manifestArg);
const fixtureRoot = fs.realpathSync(path.join(root, 'testcases/outputs')) + path.sep;
assert(manifest.startsWith(fixtureRoot), 'Use a writable verification copy under testcases/outputs');
const output = path.resolve(outputArg);
assert(output.startsWith(fixtureRoot) && !fs.existsSync(output), 'Choose a new evidence directory under testcases/outputs');
fs.mkdirSync(output);
const hash = data => createHash('sha256').update(data).digest('hex');
const original = hash(fs.readFileSync(manifest));
const document = JSON.parse(fs.readFileSync(manifest));
const targetLayer = document.layers.find(l=>l.role==='target') ?? document.layers.find(l=>l.kind==='tumor');
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'rslab-operative-preview-'));
app.setPath('userData',profile);
if(process.platform==='darwin') app.setActivationPolicy('accessory');
const startup = path.join(profile,'empty');
execFileSync(path.join(root,'.venv/bin/python'),['-c','from resectionlab.patient_anatomy import PatientImagingCollection as C; import sys; C.create(sys.argv[1],"EMPTY-VERIFIER").save()', startup],{cwd:root,env:{...process.env,PYTHONPATH:path.join(root,'src')}});
process.argv.push('--automation-hidden','--anatomy',path.join(startup,'patient-anatomy.json'));
delete process.env.RESSECTIONLAB_DEV_URL;
const report = {mode, manifest, patient:document.patient_id, sourceSnapshot:document.snapshot_id, everVisible:false, captures:[], errors:[]};
const flagValue = name => {
  const index = process.argv.indexOf(name);
  return index<0 ? null : path.resolve(process.argv[index+1]);
};
const archive = flagValue('--open-archive');
if(archive) assert(archive.startsWith(fixtureRoot) && fs.statSync(archive).isFile());
const saveTo = flagValue('--export');
if(saveTo) assert(saveTo.startsWith(fixtureRoot) && !fs.existsSync(saveTo));
dialog.showOpenDialog = async () => ({canceled:false, filePaths:[archive ?? manifest]});
dialog.showSaveDialog = async () => {
  assert(saveTo && !fs.existsSync(saveTo),'Unexpected save dialog');
  return {canceled:false,filePath:saveTo};
};
let activeRequests = 0;
const handle = ipcMain.handle.bind(ipcMain);
ipcMain.handle = (channel, handler) => handle(channel, async (...args) => {
  activeRequests++;
  try {
    const result = await handler(...args);
    if(channel==='research:anatomyView' && result.patientContext?.signal) {
      (report.patientViews ??= {})[args[1].seriesId] = {
        contextHash:result.patientContext.signal.sha256,sourceHash:result.mri.sha256,
        shape:result.patientContext.shape,affine:result.patientContext.affine,
        scans:result.patientContext.sources.map(s=>({series:s.seriesId,alignment:s.alignment,contributes:!!s.displayWindow}))};
    }
    if(channel === 'research:anatomyPlan') {
      const action = args[1].action;
      (report.plans ??= []).push({action,selectedId:result.selectedId,
        routes:result.approaches?.map(r=>({id:r.id,origin:r.origin,status:r.assessmentStatus,entry:r.entry,target:r.target,parameterHash:r.parameterHash})),
        stations:result.stations, layers:result.layers?.map(l=>({id:l.id,evidence:l.evidence,voxels:l.sampledVoxels})),
        target:result.targetLayerId,approach:result.approachId,excluded:result.excluded,issues:result.stationIssues,
        stationEvidence:result.stationEvidence,stationLayerIds:result.stationLayerIds,
        distinctEntryReport:result.distinctEntryReport?{status:result.distinctEntryReport.status,shortfall:result.distinctEntryReport.shortfall}:undefined});
      console.log('Plan operation',action,result.selectedId ?? result.approachId ?? '',result.approaches?.length ?? '');
    }
    return result;
  }
  catch(error) { report.errors.push({channel, message:String(error)}); throw error; }
  finally { activeRequests--; }
});
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const reportPath = path.join(output,'report.json');
function finish(error) {
  report.sourceUnchanged = hash(fs.readFileSync(manifest)) === original;
  report.passed = !error && report.sourceUnchanged && !report.everVisible && !report.errors.length;
  if(error) report.error = error.stack || String(error);
  fs.writeFileSync(reportPath, JSON.stringify(report,null,2)+'\n');
  console.log(report.passed ? reportPath : JSON.stringify(report));
  process.exitCode = report.passed ? 0 : 1;
  app.quit();
}
app.once('will-quit', () => {
  fs.rmSync(profile,{recursive:true,force:true});
  if(!report.passed) app.exit(1);
});
dialog.showErrorBox = (title,message) => finish(Error(title+' '+message));
process.on('uncaughtException',finish);
process.on('unhandledRejection',finish);
app.on('browser-window-created', (_, win) => {
  win.on('show', () => { report.everVisible = true; finish(Error('Window became visible')); });
  win.webContents.once('did-finish-load', async () => {
    win.webContents.setBackgroundThrottling(false);
    const js = expr => win.webContents.executeJavaScript(expr,true);
    const until = async (expr, timeout = 240000) => {
      const start = Date.now();
      while(Date.now()-start<timeout) {
        if(report.errors.length) throw Error('Application request failed: '+JSON.stringify(report.errors.at(-1)));
        if(await js(expr)) return;
        await pause(200);
      }
      throw Error('Timed out waiting for '+expr);
    };
    const button = name => `[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(name)}||b.getAttribute('aria-label')===${JSON.stringify(name)}||b.querySelector('span')?.textContent.trim()===${JSON.stringify(name)})`;
    const click = async name => {
      await until(`!!(${button(name)}) && !(${button(name)}).disabled`);
      await js(`(()=>{const b=${button(name)};
        for(let p=b.parentElement;p;p=p.parentElement) if(p.tagName==='DETAILS') p.open=true;
        b.scrollIntoView({block:'nearest'}); b.click();
      })()`);
    };
    const ready = async () => {
      const stable = `document.querySelector('.rl-viewer')?.dataset.rendererReady==='true' &&
          !!document.querySelector('.rl-viewer')?.dataset.sourceDisplayIdentity &&
          document.querySelector('.rl-viewer')?.dataset.surfacesPending==='0' &&
          !document.querySelector('.viewer-loading,.rl-viewer-preparing,[data-layer-pending="true"],[data-approach-pending="true"]')`;
      const start = Date.now();
      while(Date.now()-start<300000) {
        if(!activeRequests && await js(stable)) {
          await pause(300);
          if(!activeRequests && await js(stable)) return;
        }
        await pause(200);
      }
      throw Error('Timed out waiting for source renderer');
    };
    const capture = async name => {
      await pause(350);
      assert.equal(win.isVisible(),false);
      const shot = await win.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true});
      const size = await js('({width:innerWidth,height:innerHeight})');
      const image = shot.getSize().width === size.width ? shot : shot.resize(size);
      fs.writeFileSync(path.join(output,name+'.png'),image.toPNG());
      report.captures.push({name,...size});
      console.log('Captured '+name);
      return image;
    };
    const both = async name => {
      for(const [width,height] of [[1440,900],[1050,720]]) {
        win.setContentSize(width,height);
        await until(`innerWidth===${width}&&innerHeight===${height}`);
        await capture(name+'-'+width);
      }
      win.setContentSize(1440,900);
    };
    try {
      await until(`document.querySelector('footer')?.textContent.includes('Add a source scan')`);
      await click('Open');
      await until(`JSON.parse(document.querySelector('.rl-viewer')?.dataset.sourceDisplayIdentity || '[]')[3]===${JSON.stringify(document.snapshot_id)}`);
      await ready();
      await click('Anatomy');
      await ready();
      await both('anatomy');
      if(mode==='preview' && document.series.length>1) {
        const reference = document.reference_series;
        await js(studyScanSelectionScript(reference));
        await ready();
        await js("document.querySelector('.anatomy-presets button:nth-child(2)').click()");
        await ready();
        const readModel = () => js(`({source:document.querySelector('.native-workspace').dataset.nativeSeries,
          summary:document.querySelector('.patient-model-summary')?.textContent,
          frame:document.querySelector('.patient-model-summary')?.dataset.frame,
          layers:JSON.parse(document.querySelector('.rl-viewer').dataset.visibleLayerIds || '[]')})`);
        report.model = [await readModel()];
        assert.equal(report.model[0].source,reference);
        assert.equal(report.model[0].frame,'patient');
        const secondary = document.series.find(row=>row.id!==reference).id;
        await js(studyScanSelectionScript(secondary));
        await ready();
        report.model.push(await readModel());
        assert.equal(report.model[1].source,secondary);
        if(report.model[1].frame==='patient') {
          assert.equal(report.patientViews[reference].contextHash,report.patientViews[secondary].contextHash);
          assert.deepEqual(report.patientViews[reference].shape,report.patientViews[secondary].shape);
          assert.deepEqual(report.patientViews[reference].affine,report.patientViews[secondary].affine);
          if(document.series[0].original.sha256!==document.series[1].original.sha256)
            assert.notEqual(report.patientViews[reference].sourceHash,report.patientViews[secondary].sourceHash);
          assert.deepEqual([...report.model[0].layers].sort(),[...report.model[1].layers].sort());
        }
        await both('linked-second-scan');
        await js(studyScanSelectionScript(reference));
        await ready();
      }

      console.log('Opening Plan');
      await click('Plan');
      await until("!!document.querySelector('.native-approaches') && !document.querySelector('.approach-computing')");
      await both('plan');
      if(['prepare','preview'].includes(mode)) {
        const target = targetLayer;
        assert(target,'Study requires a target');
        await until(`[...document.querySelectorAll('select[aria-label="Planning target"] option')].some(o=>o.value===${JSON.stringify(target.id)}&&!o.disabled)`);
        await js(inputScript('Planning target',target.id));
        await until(`!!(${button('Suggest approaches')}) && !(${button('Suggest approaches')}).disabled`);
        const pending = (report.plans??[]).filter(r=>r.action==='suggest').length;
        await click('Suggest approaches');
        const start = Date.now();
        while((report.plans??[]).filter(r=>r.action==='suggest').length===pending) {
          if(Date.now()-start>720000 || report.errors.length) throw Error('Source suggestion did not finish');
          await pause(350);
        }
        await ready();
        let cards = await js("Array.from(document.querySelectorAll('.approach-card'),b=>b.dataset.approachId)");
        if(!cards.length) {
          report.suggestedCount = 0;
          await click('Create manual approach');
          await until("!!document.querySelector('.approach-card')");
          await ready();
          cards = await js("Array.from(document.querySelectorAll('.approach-card'),b=>b.dataset.approachId)");
        } else report.suggestedCount = report.plans.findLast(row=>row.action==='suggest').routes.filter(r=>r.origin==='bounded-native-entry-patch').length;
        if(process.argv.includes('--distinct') || process.argv.includes('--try-distinct')) {
          const previous = report.plans.findLast(result=>result.routes?.length).routes.filter(row=>row.origin!=='native-source-entry-windows');
          assert(previous.length,'Need pre-existing patch approaches for the preservation check');
          await click('Suggest distinct entries');
          await until("!document.querySelector('.approach-computing')",720000);
          await ready();
          const measured = report.plans.findLast(result=>result.action==='suggestDistinctEntries');
          assert(measured,'The distinct-entry action must reach the native engine');
          for(const before of previous) {
            const after = measured.routes.find(row=>row.id===before.id);
            assert.deepEqual(after,before,'Earlier patch approach changed');
          }
          const distinct = measured.routes.filter(row=>row.origin==='native-source-entry-windows');
          if(process.argv.includes('--distinct')) assert(distinct.length>1,'Study did not admit separated entries');
          const angle = (a,b) => {
            const u=a.entry.map((v,i)=>v-a.target[i]),v=b.entry.map((n,i)=>n-b.target[i]);
            const cosine=u.reduce((sum,n,i)=>sum+n*v[i],0)/(Math.hypot(...u)*Math.hypot(...v));
            return Math.acos(Math.max(-1,Math.min(1,cosine)))*180/Math.PI;
          };
          report.distinctPairs=[];
          for(let i=0;i<distinct.length;i++) for(let j=0;j<i;j++) {
            const gap=Math.hypot(...distinct[i].entry.map((n,k)=>n-distinct[j].entry[k]));
            const degrees=angle(distinct[i],distinct[j]);
            assert(gap>1 && degrees>1,'Distinct entries need separate opening positions and axes');
            report.distinctPairs.push({routes:[distinct[i].id,distinct[j].id],entryGapMm:gap,angleDegrees:degrees});
          }
          if(distinct.length) cards=distinct.map(row=>row.id);
        }
        const selected = await js("document.querySelector('.approach-card.selected')?.dataset.approachId");
        const route = cards.find(id=>id!==selected) ?? cards[0];
        await js(`document.querySelector('.approach-card[data-approach-id="${route}"]').click()`);
        await until(`document.querySelector('.approach-card.selected')?.dataset.approachId===${JSON.stringify(route)}`);
        report.selected = route;
        if(process.argv.includes('--distinct')) {
          const overlayIds = () => js("JSON.parse(document.querySelector('.rl-viewer').dataset.approachRoutes).map(r=>r.route_id)");
          await until(`JSON.parse(document.querySelector('.rl-viewer').dataset.approachRoutes).some(r=>r.selected&&r.route_id===${JSON.stringify(route)})`);
          report.windowOverlay=await overlayIds();
          assert.deepEqual([...report.windowOverlay].sort(),[...cards].sort());
          const patch=report.plans.findLast(row=>row.action==='suggest').routes.find(row=>row.origin==='bounded-native-entry-patch');
          if(patch) {
            console.log('Checking return to patch routes');
            await ready();
            await until(`!document.querySelector('.approach-card[data-approach-id="${patch.id}"]').disabled`);
            await js(`document.querySelector('.approach-card[data-approach-id="${patch.id}"]').click()`);
            await until(`JSON.parse(document.querySelector('.rl-viewer').dataset.approachRoutes).some(r=>r.selected&&r.route_id===${JSON.stringify(patch.id)})`);
            report.patchOverlay=await overlayIds();
            assert(report.patchOverlay.every(id=>!report.windowOverlay.includes(id)));
            await ready();
            await until(`!document.querySelector('.approach-card[data-approach-id="${route}"]').disabled`);
            await js(`document.querySelector('.approach-card[data-approach-id="${route}"]').click()`);
            await until(`JSON.parse(document.querySelector('.rl-viewer').dataset.approachRoutes).some(r=>r.selected&&r.route_id===${JSON.stringify(route)})`);
          }
        }
        await both('selected');
      }
      if(mode==='preview') {
        const steps = [['bone','Bone flap'],['dura','Dural opening'],['brain','Brain retraction'],['tumor','Tumor aspiration']];
        const previewReady = async () => {
          await until("!!document.querySelector('.operative-field-canvas canvas') || !!document.querySelector('.operative-placeholder[role=alert]')");
          assert(await js("!!document.querySelector('.operative-field-canvas canvas')"),await js("document.querySelector('.operative-placeholder')?.innerText"));
          assert.equal(await js("document.querySelector('.operative-field-canvas [role=alert]')?.innerText"),undefined);
        };
        await click('Preview operative steps');
        await previewReady();
        const field = report.plans.findLast(p=>p.action==='operativeField');
        assert.equal(field.approach,report.selected);
        assert.equal(field.target,targetLayer.id);
        const captureField = async name => {
          const im = await capture(name);
          const rect = await js(`(()=>{const r=document.querySelector('.operative-field-canvas canvas').getBoundingClientRect();
            return {x:Math.ceil(r.left+r.width*.04),y:Math.ceil(r.top+r.height*.04),width:Math.floor(r.width*.92),height:Math.floor(r.height*.86)};})()`);
          assert(rect.width>200 && rect.height>110);
          return im.crop(rect).toBitmap();
        };
        const delta = (a,b) => {
          assert.equal(a.length,b.length);
          let changed = 0;
          for(let i=0;i<a.length;i+=4) if(Math.max(Math.abs(a[i]-b[i]),Math.abs(a[i+1]-b[i+1]),Math.abs(a[i+2]-b[i+2]))>=10) changed++;
          return changed;
        };
        report.animation = [];
        for(const [width,height] of [[1440,900],[1050,720]]) {
          win.setContentSize(width,height);
          await until(`innerWidth===${width}&&innerHeight===${height}`);
          if(await js(`!(${button('Reset operative steps')}).disabled`)) await click('Reset operative steps');
          let prior = await captureField('intact-'+width);
          assert.equal(await js("Number(document.querySelector('[data-removed-cells]')?.dataset.removedCells ?? 0)"),0);
          const readout = {width,stages:[],evidence:await js("document.querySelector('.operative-sources').textContent")};
          for(const [role,name] of steps) {
            const enabled = await js(`!(${button(name)}).disabled`);
            if(!enabled && role==='tumor' && field.stations.tumor!==null) {
              assert.match(await js(`(${button(name)}).textContent`),/No eligible tip position/);
            } else assert.equal(enabled,field.stations[role]!==null);
            if(!enabled) continue;
            await click(name);
            await until(`document.querySelector('.operative-playback').dataset.operativePhase===${JSON.stringify(role)}`);
            const frame = await captureField(role+'-'+width);
            const changed = delta(prior,frame);
            assert(changed>30,'Missing rendered transition at '+role+' ('+changed+' changed pixels)');
            const removed = await js("Number(document.querySelector('[data-removed-cells]')?.dataset.removedCells ?? 0)");
            if(role==='tumor') assert(removed>0,'Target endpoint must remove modeled label cells');
            else assert.equal(removed,0);
            readout.stages.push({role,changedPixels:changed,removed});
            prior = frame;
          }
          const timeline = 'input[aria-label="Operative timeline"]';
          await js(inputScript('Operative timeline',0.55));
          await until(`Number(document.querySelector(${JSON.stringify(timeline)}).value)===.55`);
          await click('Play steps');
          await until(`Number(document.querySelector(${JSON.stringify(timeline)}).value)>.64`,10000);
          await click('Pause steps');
          readout.playedTo = Number(await js(`document.querySelector(${JSON.stringify(timeline)}).value`));
          if(field.stations.brain!==null) {
            const toggle = () => js("[...document.querySelectorAll('.operative-options label')].find(l=>l.textContent.trim()==='Include retraction').querySelector('input').click()");
            await toggle();
            assert(await js(`(${button('Brain retraction')}).disabled`));
            await toggle();
          }
          readout.overflow=await js("document.documentElement.scrollWidth>innerWidth");
          assert.equal(readout.overflow,false);
          report.animation.push(readout);
        }
        if(process.argv.includes('--video')) {
          win.setContentSize(1280,800);
          await until('innerWidth===1280&&innerHeight===800');
          const max = Number(await js(`document.querySelector('[aria-label="Operative timeline"]').max`));
          const dest = path.join(output,'operative-walkthrough.mp4');
          const encoder = spawn('ffmpeg',['-hide_banner','-loglevel','error','-n','-f','image2pipe','-framerate','10',
            '-c:v','png','-i','pipe:0','-c:v','libx264','-preset','veryfast','-crf','19','-pix_fmt','yuv420p',
            '-movflags','+faststart',dest],{stdio:['pipe','ignore','inherit']});
          const encoded = once(encoder,'close');
          try {
            for(let index=0;index<=max*34;index++) {
              const position = Number((index/34).toFixed(2));
              await js(inputScript('Operative timeline',position));
              await js('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))');
              assert.equal(win.isVisible(),false);
              const actual = await win.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true});
              const frame = actual.getSize().width===1280?actual:actual.resize({width:1280,height:800});
              const bytes = frame.toPNG();
              const repeats = index===0||index===max*34 ? 8 : 1;
              for(let hold=0;hold<repeats;hold++) if(!encoder.stdin.write(bytes)) await once(encoder.stdin,'drain');
            }
            encoder.stdin.end();
            const [code] = await encoded;
            assert.equal(code,0);
            report.video = dest;
          } finally { if(encoder.exitCode===null) encoder.kill(); }
          await js(inputScript('Operative timeline',max-1+.05));
          assert.equal(await js("Number(document.querySelector('[data-removed-cells]').dataset.removedCells)"),0);
          await click('Reset operative steps');
        }
        await click('Return to plan');
        await ready();
        assert.equal(await js("document.querySelector('.approach-card.selected')?.dataset.approachId"),report.selected);
        const identity=await js("JSON.parse(document.querySelector('.rl-viewer').dataset.sourceDisplayIdentity)");
        assert.equal(identity[3],document.snapshot_id);
        await click('Rehearse');
        await previewReady();
        await click('Return to plan');
        await ready();
        await click('Anatomy');
        await ready();
        await capture('restored-anatomy');
        if(saveTo) {
          await click('Scans');
          await click('Save patient study file…');
          await until("document.querySelector('footer')?.textContent.includes('Patient study file saved')",600000);
          assert(fs.statSync(saveTo).size>0);
          report.archive={name:saveTo,bytes:fs.statSync(saveTo).size};
        } else if(process.argv.includes('--save-view')) {
          await click('Save');
          await until("document.querySelector('footer')?.textContent.includes('Workspace saved')");
        }
      }
      if(mode==='reopen') {
        await until("!!document.querySelector('.approach-card.selected')");
        report.selected=await js("document.querySelector('.approach-card.selected').dataset.approachId");
        await click('Rehearse');
        await until("!!document.querySelector('.operative-field-canvas canvas')");
        await both('reopened');
      }
      if(mode==='stale') {
        const preview = await js(`!!(${button('Preview operative steps')}) && !(${button('Preview operative steps')}).disabled`);
        assert.equal(preview,false);
        await click('Rehearse');
        await until("!!document.querySelector('.operative-inspector')");
        assert.equal(await js("!!document.querySelector('.operative-playback')"),false);
        report.unavailable = await js("document.querySelector('.operative-inspector').innerText");
        await both('stale-route');
      }
      if(mode==='no-target') {
        assert.equal(await js(`(${button('Suggest approaches')}).disabled`),true);
        await click('Rehearse');
        await until("!!document.querySelector('.operative-inspector')");
        assert.equal(await js("!!document.querySelector('.operative-playback')"),false);
        report.unavailable = await js("document.querySelector('.operative-inspector').innerText");
        await both('no-target');
      }
      if(mode==='baseline') {
        await click('Rehearse');
        await pause(700);
        await both('rehearse');
      }
      finish();
    } catch(error) {
      report.body = await js('document.body.innerText').catch(()=>null);
      await capture('failure').catch(()=>{});
      finish(error);
    }
  });
});
require('./main.cjs');
