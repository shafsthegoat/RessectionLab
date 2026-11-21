'use strict';

// electron verify-manual-practice.cjs SOURCE EMPTY_OUTPUT [--benchmark|--cold|--cold-benchmark]
// Each cold invocation opens only the prior warm save in a new Electron profile.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {execFileSync} = require('node:child_process');
const {app, dialog, ipcMain} = require('electron');
const responseCheck = require('./response-practice-evidence.cjs');
const {inputScript} = require('./verify-eden-planning-helpers.cjs');
const {delay, summary, stateWitness, checkAction, startRecording} = require('./manual-practice-evidence.cjs');

const launch = performance.now();
const wall = () => performance.timeOrigin + performance.now();
const root = path.resolve(__dirname, '../..');
const source = fs.realpathSync(process.argv[2]);
const output = path.resolve(process.argv[3]);
const complete = process.argv.includes('--complete-cranial') ? require('./complete-cranial-practice.cjs') : null;
const pairedIndex = process.argv.indexOf('--paired-actions');
assert(!complete || pairedIndex === -1, 'Choose one source task');
const paired = pairedIndex >= 0 ? JSON.parse(fs.readFileSync(process.argv[pairedIndex+1], 'utf8')) : null;
const detail = process.argv.includes('--response-detail');
const enclosed = process.argv.includes('--enclosed-outlet');
const gestureOnly = process.argv.includes('--continuous-benchmark');
assert(!enclosed || paired, 'Provide enclosed control actions');
const cold = process.argv.includes('--cold') || process.argv.includes('--cold-benchmark');
const benchmark = process.argv.includes('--benchmark') || process.argv.includes('--cold-benchmark');
const sample = process.argv.find(arg=>arg.startsWith('--cold-sample='))?.split('=')[1];
assert(sample===undefined || (cold&&benchmark&&/^\d{1,2}$/.test(sample)), 'Use a numeric --cold-sample only for repeated cold benchmarks');
const take = (benchmark ? 'benchmark' : 'interaction') + (cold ? '-cold' : '') + (sample ? '-'+sample : '');
const saved = path.join(output, (benchmark ? 'benchmark' : 'practice') + '.ressectionlab');
const paper = path.join(output, take + '.json');
const original = cold ? saved : source;
const hashFile = f => createHash('sha256').update(fs.readFileSync(f)).digest('hex');
assert(path.resolve(output).startsWith(path.join(root, 'testcases', 'outputs') + path.sep), 'Use an ignored fixture output or a link there');
assert(!fs.existsSync(paper) && (cold || !fs.existsSync(saved)), 'Choose a fresh output');
fs.mkdirSync(output, {recursive: true});
const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'rslab-manual-'));
const captureScratch = benchmark ? null : fs.mkdtempSync(path.join(output, '.compositor-'));
const report = {pid: process.pid, windowMode:'hidden', windowShown:false, mode: take, source, originalSha256: hashFile(original), inputFile: original,
  gitHead:execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim(),
  sourceFiles:Object.fromEntries(['desktop/electron/main.cjs','desktop/electron/verify-manual-practice.cjs','desktop/electron/manual-practice-evidence.cjs','desktop/electron/response-practice-evidence.cjs','desktop/electron/complete-cranial-practice.cjs',
    'src/resectionlab/desktop_bridge.py','src/resectionlab/surgical_bridge.py','src/resectionlab/surgical_practice.py','src/resectionlab/surgical_engine.py',
    'src/resectionlab/surgical_deformation_v2.py','src/resectionlab/surgical_effects.py','src/resectionlab/surgical_bipolar_thermal.py','src/resectionlab/surgical_vascular_graph.py'].map(name=>[name,hashFile(path.join(root,name))])),
  rendererAssets: {}, machine: {model: os.cpus()[0].model, totalGiB: os.totalmem()/2**30, os: os.release()},
  actions: [], memory: [], timing: {}, captures: []};
if(complete) report.sourceFiles['desktop/electron/complete-cranial-practice.cjs']=hashFile(require.resolve('./complete-cranial-practice.cjs'));
for (const name of fs.readdirSync(path.join(root, 'desktop/dist/assets')).filter(n => /^index-.*\.(js|css)$/.test(n)))
  report.rendererAssets[name] = hashFile(path.join(root, 'desktop/dist/assets', name));
const persist = () => fs.writeFileSync(paper, JSON.stringify(report, null, 2) + '\n');
let latest = null, pending = null, observeHeldStep = null;
const handle = ipcMain.handle.bind(ipcMain);
ipcMain.handle = (channel, handler) => handle(channel, async (...args) => {
  const started = wall();
  const result = await handler(...args);
  const received = wall();
  if (channel === 'research:surgicalRehearsal' && result?.view && !result.playback) {
    const item = {command: args[1].command, witness: stateWitness(result), admission: result.practiceAdmission,
      episode: result.practiceEpisode, status: result.outcome?.status};
    if(complete) item.nativeView = result.view;
    if(observeHeldStep && item.command === 'step') {
      observeHeldStep(args[1].action, result.outcome, item.witness, received, received-started);
    }
    if(cold&&result.practiceEpisode){
      const rows=result.timeline??result.view.history;
      if(rows?.length)report.coldHistory=rows.slice(result.practiceEpisode.parent.position).map(row=>
        ({stateBefore:row.stateBefore,stateAfter:row.stateAfter,status:row.status,operation:row.action.operation}));
    }
    if (pending && item.command === 'step') {
      assert.equal(pending.result, undefined, 'A single input must dispatch exactly one engine action');
      const {status, completedFraction, reasonCode, reason, events} = result.outcome;
      pending.result = {requested:args[1].action, status, completedFraction, reasonCode, reason,
        eventKinds:[...new Set((events ?? []).map(e => e.kind))],
        cusaAllocatedMm3:(events??[]).reduce((sum,e)=>sum+Number(e.allocatedCapacityMm3??0),0),
        cusaRemovedCells:(events??[]).flatMap(e=>e.kind==='cusa_contact'&&Array.isArray(e.removedCells)?e.removedCells:[]),witness:item.witness,
        backendResponseEpochMs:received, ipcMs:received-started};
    }
    latest = item;
  }
  return result;
});
app.setPath('userData', path.join(scratch, 'profile'));
if (process.platform === 'darwin') app.setActivationPolicy('accessory');
process.argv.push('--case', original, '--automation-hidden');
delete process.env.RESSECTIONLAB_DEV_URL;
dialog.showSaveDialog = async () => { assert(!cold && !fs.existsSync(saved)); return {canceled:false,filePath:saved}; };
dialog.showMessageBox = async (...args) => { throw Error('Unexpected dialog ' + args.at(-1).title); };

app.on('browser-window-created', (_, win) => {
  win.on('show', () => { win.hide(); report.windowShown = true; });
  win.webContents.once('did-finish-load', async () => {
    const wc = win.webContents;
    const js = code => wc.executeJavaScript(code, true);
    const q = JSON.stringify;
    const width = 1440, height = 900;
    const wait = async (check, label, limit = 120000) => {
      const started = performance.now();
      while (true) {
        if (await check()) return;
        if (performance.now()-started > limit) throw Error('Timed out waiting for ' + label);
        await delay(30);
      }
    };
    const until = (expr, limit) => wait(() => js(expr), expr, limit);
    const byName = name => `[...document.querySelectorAll('button,summary,input,select,[tabindex]')].find(e=>
      e.getAttribute('aria-label')===${q(name)} || (e.matches('button,summary') && (e.textContent.trim()===${q(name)}
      || e.querySelector('strong')?.textContent.trim()===${q(name)})))`;
    const ready = id => until(`document.querySelector('.surgical-panel')?.dataset.stateId===${q(id)}
      && document.querySelector('[data-accepted-tools]')?.dataset.surgicalStateId===${q(id)}
      && document.querySelector('.rl-viewer')?.dataset.surfacesPending==='0' && !document.querySelector('.viewer-loading')
      && ![...document.querySelectorAll('.surgical-panel p[role=status]')].some(e=>/Preparing an exact|Computing instrument|Preparing accepted display|Verifying saved/.test(e.textContent))`);
    const point = async expr => {
      await until(`!!(${expr}) && !(${expr}).matches(':disabled')`);
      let summaryPoint;
      // Open real disclosure controls in their natural outer-to-inner order.
      do {
        summaryPoint = await js(`(()=>{const e=${expr};let s=null;for(let p=e.parentElement;p;p=p.parentElement)
          if(p.tagName==='DETAILS'&&!p.open&&p.querySelector(':scope > summary')!==e)s=p.querySelector(':scope > summary');
          const t=s??e;t.scrollIntoView({block:'center',inline:'nearest',behavior:'instant'});const r=t.getBoundingClientRect();
          const x=Math.round(r.x+r.width/2),y=Math.round(r.y+r.height/2);
          const points=[[x,y],[x,Math.floor(r.bottom-2)],[x,Math.ceil(r.top+2)]];
          const hit=points.find(([px,py])=>t.contains(document.elementFromPoint(px,py)));
          return {x:hit?.[0]??x,y:hit?.[1]??y,expand:!!s,visible:!!hit,hit:document.elementFromPoint(x,y)?.outerHTML.slice(0,160)};})()`);
        assert(summaryPoint.visible, 'Control is obscured or outside the viewport: ' + expr + ' ' + JSON.stringify(summaryPoint));
        if (summaryPoint.expand) {await mouse(summaryPoint);await delay(75);}
      } while(summaryPoint.expand);
      return summaryPoint;
    };
    const send = (method, payload) => wc.debugger.sendCommand('Input.' + method, payload);
    const mouse = async (p, mark=false) => {
      const coords={x:p.x,y:p.y};
      await send('dispatchMouseEvent', {type:'mouseMoved',...coords});
      await send('dispatchMouseEvent', {type:'mousePressed',button:'left',buttons:1,clickCount:1,...coords});
      if(mark) pending.inputEpochMs = wall();
      await send('dispatchMouseEvent', {type:'mouseReleased',button:'left',buttons:0,clickCount:1,...coords});
    };
    const click = async (label, mark=false) => {console.log(take,'click',label);await mouse(await point(byName(label)), mark);};
    const key = async (keyName, code, vk, extra={}) => {
      const info={key:keyName,code,windowsVirtualKeyCode:vk,nativeVirtualKeyCode:vk,...extra};
      await send('dispatchKeyEvent',{type:'keyDown',...info});
      await send('dispatchKeyEvent',{type:'keyUp',...info});
    };
    const number = async (label, value) => {
      const expr = byName(label);
      if (Number(await js(`(${expr})?.value`))===value) return;
      await mouse(await point(expr));
      // Select the input's existing draft; insertText sends trusted Chromium editing input.
      await js(`(${expr}).select()`);
      await send('insertText',{text:String(value)});
      await key('Tab','Tab',9);
      assert.equal(Number(await js(`(${expr}).value`)),value,label);
    };
    const instrument = async id => {
      const expr=byName('Practice instrument');
      const p=await point(expr);
      await send('dispatchMouseEvent',{type:'mouseMoved',x:p.x,y:p.y});
      // Chromium's native select popup is unsuitable for hidden-window capture.
      // Focus and use native keyboard typeahead, which commits the same change.
      const option=await js(`(()=>{const e=${expr};e.focus({preventScroll:true});return Array.from(e.options).find(o=>o.value===${q(id)})?.textContent.trim()})()`);
      assert(option,'Instrument missing from practice inventory: '+id);
      const {current, options}=await js(`(()=>{const e=${expr};return {current:e.selectedIndex,options:Array.from(e.options,o=>({id:o.value,label:o.textContent}))}})()`);
      const target=options.findIndex(o=>o.id===id),char=option[0].toLowerCase();
      const nextMatch=options.map((_,i)=>(i+current+1)%options.length).find(i=>options[i].label.toLowerCase().startsWith(char));
      assert.equal(nextMatch,target,'Native typeahead must select '+id);
      pending.inputEpochMs=wall();
      await key(char,'Key'+char.toUpperCase(),char.toUpperCase().charCodeAt(0),{text:char});
    };
    const installTelemetry = () => js(`(()=>{
      const data=window.manualEvidence={frames:[],fences:[],gpu:[],mesh:[],inputs:[],active:false};
      const now=()=>performance.timeOrigin+performance.now();
      document.addEventListener('anatomy-frame',({detail:d})=>{data.latestFrame={at:d.at,cpuMs:d.cpuMs,serial:d.serial,stateId:d.surgicalStateId,drawCalls:d.drawCalls,field:d.fieldOptics};if(data.active)data.frames.push(data.latestFrame)});
      document.addEventListener('anatomy-gpu',({detail:d})=>{if(d.completed){const row={at:now(),stateId:d.surgicalStateId,source:d.sourceCaseHash,ms:d.milliseconds};data.latestFence=row;if(data.active)data.fences.push(row)}});
      document.addEventListener('anatomy-gpu-time',({detail:d})=>{if(data.active&&d.completed)data.gpu.push(d.milliseconds)});
      document.addEventListener('anatomy-work',({detail:d})=>{if(data.active&&d.kind==='modeled-surface-preparation')data.mesh.push({name:d.name,incremental:d.incremental,ms:d.preparationMs,remeshedChunks:d.remeshedChunks,totalChunks:d.totalChunks})});
      const ring=document.createElement('div'), badge=document.createElement('output');
      if(!${benchmark}) {
        ring.style.cssText='position:fixed;pointer-events:none;z-index:2147483647;width:22px;height:22px;margin:-12px;border:2px solid #ffd271;border-radius:50%;box-shadow:0 0 2px black;display:none';
        badge.style.cssText='position:fixed;pointer-events:none;z-index:2147483647;bottom:6px;left:9px;font:13px system-ui;padding:4px 8px;background:#12232e;color:white;border-radius:4px;display:none';
        document.body.append(ring,badge);
      }
      const onInput=e=>{if(!e.isTrusted)return;const t=e.target, control=t.closest?.('button,summary,input,select'), label=control?.getAttribute('aria-label')??control?.textContent.trim()??t.closest?.('[aria-label]')?.getAttribute('aria-label')??t.tagName;
        if(e.type!=='pointermove'){
          if(data.active)data.inputs.push({type:e.type,at:now(),key:e.key,label});
          badge.style.display='block';badge.textContent=(e.type==='keydown'?e.key+' · ':e.type==='input'?'Edit · ':'Click · ')+label;
        }
        if(e.type.startsWith('pointer')){ring.style.display='block';ring.style.left=e.clientX+'px';ring.style.top=e.clientY+'px';ring.style.background=e.type==='pointerdown'?'#ffd27188':'transparent';}
      };
      for(const type of ['pointermove','pointerdown','keydown','input'])document.addEventListener(type,onInput,true);
    })()`);
    const snapshot = async witness => {
      await ready(witness.stateId);
      const drawn=await js(`(()=>{const p=document.querySelector('[data-accepted-tools]'),h=document.querySelector('[data-operative-readout-state]');
        return {id:p.dataset.surgicalStateId,tools:JSON.parse(p.dataset.acceptedTools),removed:Number(p.dataset.removedCells),interactionMode:p.dataset.practiceInteraction,
          view:p.dataset.operativeView,fieldMm:Number(document.querySelector('[aria-label=\"Field / magnification\"]')?.value),pane:p.getBoundingClientRect().toJSON(),camera:JSON.parse(p.dataset.opticalCamera??'null'),hud:h?.dataset.operativeReadoutState,
          readout:h?.textContent,quantities:Object.fromEntries([...h?.querySelectorAll('dl > div')??[]].map(row=>[row.querySelector('dt').textContent,row.querySelector('dd').textContent])),field:window.manualEvidence.latestFrame?.field,frame:window.manualEvidence.latestFence}})()`);
      assert.equal(drawn.view,'surgeon');assert.equal(drawn.hud,witness.stateId);
      assert.equal(drawn.id,witness.stateId);assert.deepEqual(drawn.tools,witness.tools);assert.equal(drawn.removed,witness.removedCells);
      assert.equal(drawn.frame?.stateId,witness.stateId);assert.equal(drawn.frame.source,witness.caseHash);
      if(drawn.quantities['Original target remaining']!==undefined){
        const expected={'Original target remaining':witness.remaining,...(drawn.quantities['Field fluid']!==undefined?{'Field fluid':witness.metrics.fluidPresentMm3}:{
          'Free-field fluid':witness.freeField.fluidMm3,'Outlet reservoir':witness.reservoirMm3,
          'Retained fluid total':witness.metrics.fluidPresentMm3}),'Field debris':witness.metrics.debrisMm3};
        for(const [key,value] of Object.entries(expected))assert.equal(drawn.quantities[key],(+value.toFixed(2))+' mm³');
      }
      if(witness.freeField.fluidCells||witness.freeField.debrisCells){
        assert.equal(drawn.field?.stateId,witness.stateId);
        for(const key of ['fluidCells','debrisCells','fluidMm3','debrisMm3'])
          assert(Math.abs(drawn.field[key]-witness.freeField[key])<1e-6,'Visible per-cell field inventory differs: '+key);
      }
      if(drawn.field?.joinedFluidCells!==undefined){
        assert(drawn.field.joinedFluidCells>=0 && drawn.field.joinedFluidCells<=witness.freeField.fluidCells);
      }
      if(witness.reservoirMm3>0 && !witness.freeField.fluidCells){
        assert.equal(drawn.field?.fluidCells??0,0);assert.equal(drawn.field?.wetTextureBytes??0,0);
        assert.equal(drawn.field?.joinedFluidCells??0,0);
      }
      return drawn;
    };
    const capture = async name => {
      if(benchmark && !name.endsWith('-failure') && !name.startsWith('matched-')) return;
      const pic=await wc.capturePage(undefined,{stayHidden:true,stayAwake:true});
      fs.writeFileSync(path.join(output,name+'.png'),pic.toPNG());
      report.captures.push(name+'.png');
    };
    const memory = () => {
      const processes=app.getAppMetrics().map(p=>({pid:p.pid,type:p.type,workingSetKiB:p.memory.workingSetSize}));
      const children=execFileSync('ps',['-axo','pid=,ppid=,rss=,command='],{encoding:'utf8'}).split('\n')
        .map(row=>row.trim().match(/^(\d+)\s+(\d+)\s+(\d+)\s+(.+)$/))
        .filter(row=>row&&+row[2]===process.pid&&row[4].includes('desktop_bridge'))
        .map(row=>({pid:+row[1],type:'Python engine',workingSetKiB:+row[3]}));
      report.memory.push([...processes,...children]);
    };
    const action = async (label, trigger, expected) => {
      await ready(latest.witness.stateId);
      const before=latest.witness;
      const item={label,before:before.stateId};
      pending=item;
      await trigger();
      await wait(()=>item.result,label+' engine response',12000);
      const {witness,status}=item.result;
      const drawn=await snapshot(witness);
      assert.equal(drawn.interactionMode,'navigate','Committed practice geometry must be unobscured after implicit jog/numeric placement');
      checkAction(before,witness,item.result,item.result.requested);
      const allowed = Array.isArray(expected) ? expected : [expected];
      assert(allowed.includes(status),label+' returned '+status+' ('+(item.result.reasonCode??item.result.reason)+')');
      if(status!=='rejected') {
        const first=await js(`window.manualEvidence.fences.find(f=>f.stateId===${q(witness.stateId)} && f.at>=${item.inputEpochMs})`);
        assert(first,'New accepted state must be submitted to and completed by the GPU');
        item.visibleMs=first.at-item.inputEpochMs;
      }
      item.acceptedMs=item.result.backendResponseEpochMs-item.inputEpochMs;
      item.display=drawn;
      if(!benchmark && status!=='rejected'){
        const image=await wc.capturePage(undefined,{stayHidden:true,stayAwake:true});
        assert(!image.isEmpty());item.inputToCapturedViewportUpperBoundMs=wall()-item.inputEpochMs;
        assert.equal(await js("document.querySelector('[data-operative-readout-state]').dataset.operativeReadoutState"),witness.stateId);
      }
      report.actions.push(item);pending=null;memory();persist();
      console.log(take,label,status,witness.position,witness.metrics.targetRemovedMm3);
      if(!benchmark)await delay(complete?230:paired?1300:2600);
      return witness;
    };
    const press = (label,expected='accepted') => action(label,()=>click(label,true),expected);
    const choose = async (id,expected='accepted') => {
      const options=await js(`Array.from((${byName('Practice instrument')}).options,o=>({id:o.value,label:o.textContent.trim().toLowerCase()}))`);
      const current=await js(`(${byName('Practice instrument')}).selectedIndex`), target=options.findIndex(o=>o.id===id);
      if(current===target) return latest.witness;
      // Each native select navigation is a real exchange. Find the shortest path,
      // then record and validate every intermediate exchange as its own engine action.
      const queue=[[current,[]]],seen=new Set([current]);
      while(queue.length){
        const [from,route]=queue.shift();
        const reachable=new Set();
        const letters=new Set();
        for(let i=1;i<=options.length;i++){const t=(from+i)%options.length,k=options[t].label[0];if(!letters.has(k)){reachable.add(t);letters.add(k);}}
        for(const next of reachable){if(next<0||next>=options.length||seen.has(next))continue;const steps=[...route,next];
          if(next===target){for(const at of steps){const wanted=options[at].id;
            const result=await action('Select '+wanted,()=>instrument(wanted),at===target?expected:'accepted');
            if(at!==target||expected!=='rejected')assert.equal(result.tools.left.instrument,wanted);}
            return latest.witness;}
          seen.add(next);queue.push([next,steps]);
        }
      }
      throw Error('Unreachable native select choice '+id);
    };
    const record = () => startRecording(win,captureScratch,width,height,()=>js(`(()=>{
      const hud=document.querySelector('[data-operative-readout-state]')?.dataset.operativeReadoutState,
        pane=document.querySelector('[data-accepted-tools]')?.dataset.surgicalStateId;
      return hud&&hud===pane&&pane===window.manualEvidence.latestFence?.stateId&&document.querySelector('.rl-viewer')?.dataset.surfacesPending==='0'?pane:null;
    })()`));
    let stopRecording;
    try {
      wc.setBackgroundThrottling(false);wc.debugger.attach('1.3');
      await wc.debugger.sendCommand('Emulation.setFocusEmulationEnabled',{enabled:true});
      win.setContentSize(width,height);await until(`innerWidth===${width} && innerHeight===${height}`);
      assert(!win.isVisible());await installTelemetry();
      await until("!!document.querySelector('.rl-viewer canvas') && !document.querySelector('.viewer-loading')",300000);
      report.coldCanvasMs=performance.now()-launch;
      report.gpu=app.getGPUFeatureStatus();assert(report.gpu.webgl==='enabled'||report.gpu.webgl2==='enabled','GPU WebGL is unavailable');
      if(complete&&!benchmark) stopRecording=record();
      await click('Rehearse');
      await until("!!document.querySelector('[aria-label=\"Surgical timeline\"]')",300000);
      const verify=await js(`[...document.querySelectorAll('button')].find(b=>['Verify saved replay','Verify replay for planning'].includes(b.textContent.trim())&&!b.disabled)?.textContent.trim()`);
      if(verify)await click(verify);
      if(await js(`!!(${byName('Rehearse/edit stage')})`))await click('Rehearse/edit stage');
      await wait(()=>latest?.witness?.caseHash===(enclosed?'sha256:33adb0e7a4405f6dc5231512a637fc519f3339bc3967b01dbfb2f8627962ed9e':paired?'sha256:c298887cb67b1034d782d682b01ee03cdaf86ee13237602eb3239ec114d5a710':'sha256:9818506d2f779b8fece10d048e0f21e2357f86851936ad0f5b86ee6f6cfaa1e0'),'frozen source');
      await ready(latest.witness.stateId);
      if(!await js("!!document.querySelector('[data-operative-view=surgeon][data-optical-camera]')"))await click('Surgeon / Microscope');
      await until("!!document.querySelector('[data-operative-view=surgeon][data-optical-camera]')");
      report.coldVerifiedBoundaryMs=performance.now()-launch;
      if(cold) {
        const previous=JSON.parse(fs.readFileSync(path.join(output, (benchmark?'benchmark':'interaction')+'.json')));
        assert.notEqual(report.pid,previous.pid);assert.equal(report.originalSha256,previous.savedSha256);
        assert.deepEqual(latest.witness,previous.finalState);
        const steps=previous.actions.filter(a=>a.result.status!=='rejected');
        assert.deepEqual(report.coldHistory,steps.map(a=>({stateBefore:a.before,stateAfter:a.result.witness.stateId,status:a.result.status,operation:a.result.requested.operation})));
        const restored=await snapshot(latest.witness);
        assert.deepEqual(restored.camera,previous.actions.at(-1).display.camera,'Cold reopen changed the optical field');
        report.restoredCamera=restored.camera;
        if(!benchmark&&!stopRecording)stopRecording=record();
        await capture('cold-reopened');
        await click('Previous recorded state');
        const prior=previous.actions.findLast(a=>a.result.witness.position===previous.finalState.position-1).result.witness;
        await ready(prior.stateId);assert.deepEqual(latest.witness,prior);await snapshot(prior);
        if(!benchmark)await delay(2300);
        await click('Next recorded state');await ready(previous.finalState.stateId);
        assert.deepEqual(latest.witness,previous.finalState);await snapshot(latest.witness);
        if(!benchmark)await delay(2000);
        report.replayed=[prior.stateId,previous.finalState.stateId];
      } else {
        // Prepare the verified post-membrane checkpoint through the existing timeline.
        if(paired&&!enclosed){await js(inputScript('Surgical branch','configured-start'));await wait(()=>latest?.witness.branchId==='configured-start','paired source branch');}
        await until(`(${byName('Surgical timeline')})?.disabled===false`);
        const startPosition=complete?complete.parentPosition:enclosed?0:paired?1:105;
        await js(inputScript('Surgical timeline',startPosition));
        await wait(()=>latest?.witness.position===startPosition,'verified source boundary',300000);
        await ready(latest.witness.stateId);report.parent=latest.witness;
        await until(`!!(${byName('Practice from this verified state')}) && !(${byName('Practice from this verified state')}).disabled`);
        if(!benchmark&&!stopRecording)stopRecording=record();
        if(complete) report.approach=await complete.inspectApproach({click,js,capture,recorded:!benchmark});
        const recordingStarted=performance.now();
        await click('Practice from this verified state');await wait(()=>latest?.episode,'child episode');
        assert.notEqual(latest.witness.branchId,report.parent.branchId);
        assert.equal(latest.witness.stateId,report.parent.stateId);
        report.episode=latest.episode;
        await ready(latest.witness.stateId);
        await click('left hand');
        await js('window.manualEvidence.active=true');
        const gestureUi={click,byName,point,send,wait,read:()=>latest,
          observe:fn=>{observeHeldStep=fn;},snapshot,report};
        if(gestureOnly){
          await responseCheck.benchmarkHold({...gestureUi,choose,number,press,js});
        } else if(complete) {
          await complete.run({number,press,choose,capture,click,report,read:()=>latest,recorded:!benchmark,
            byName,point,send,wait,snapshot,js,key,observe:fn=>{observeHeldStep=fn;}});
        } else if(paired) {
          const labels={hold:'Hold tissue',apply:'Apply tool',move:'Move tool',place:'Place tool',withdraw:'Withdraw tool',release:'Release tool',inspect:'Observe interval'};
          const parameterNames={radiusMm:'Tip radius · mm',forceN:'Contact force · N',power:'Power · model units',
            jawOpeningMm:'Jaw opening · mm',depthMm:'Working depth · mm',duty:'Drill duty fraction',flowMm3PerSecond:'Flow · mm³/s'};
          const sampled=[];
          const optics = async () => {
            const range=byName('Field / magnification');
            await mouse(await point(range));await key('Home','Home',36);
            assert.equal(Number(await js(`(${range}).value`)),12);
            await click('Microscope settings and inspection');
          };
          if(detail) {
            await optics();
            assert.equal((await snapshot(latest.witness)).fieldMm,12);
          }
          for(const [i,row] of paired.entries()){
            assert.equal(row.slot,'left');
            await number('Action duration',row.duration);
            if(row.operation==='exchange'){sampled.push(await choose(row.instrument));continue;}
            for(const [k,value] of Object.entries(row.parameters??{})){
              if(await js(`!!(${byName(parameterNames[k])})`))await number(parameterNames[k],value);
            }
            if(row.pose&&row.operation!=='withdraw'){
              for(const [axis,n] of ['R','A','S'].entries())await number('Tool position '+n,row.pose.position[axis]);
              assert(row.pose.quaternion[0]===0&&row.pose.quaternion[1]===0,'This paired fixture uses axial orientation');
              await number('Tool roll',row.pose.quaternion[2]===0?0:90);
            }
            const label=row.operation==='deactivate'?(await js(`!!(${byName('Deactivate tool')})`)?'Deactivate tool':'Release tool'):labels[row.operation];
            assert(label,'Unsupported fixture input '+row.operation);
            const partial=enclosed&&[3,8,14].includes(i);
            sampled.push(await press(label,partial?'partial':'accepted'));
            if(detail && !enclosed && i===9){
              await click('Center field on accepted tip once');
              await click('Microscope settings and inspection');
            }
            if(detail && (enclosed?[0,3,9,14,21]:[1,2,3,4,5,9,11,16,22,28]).includes(i)){
              await delay(300);await capture('detail-'+String(i).padStart(2,'0')+'-'+row.operation);
              (report.detailFrames??=[]).push({index:i,drawn:await snapshot(sampled[i]),stateId:sampled[i].stateId});
            }
            if([3,5,9,16,22,28].includes(i))await capture('paired-'+i+'-'+row.operation);
          }
          if(detail) responseCheck.checkCameras(report.detailFrames,enclosed);
          if(enclosed)report.outletEffects=responseCheck.checkOutlet(sampled);
          else {
          const retracted=sampled[3].metrics.brainMaxDisplacementMm, released=sampled[4].metrics.brainMaxDisplacementMm, relaxed=sampled[5].metrics.brainMaxDisplacementMm;
          assert(retracted>released && released>relaxed && relaxed>=0);
          assert(sampled[11].bleedingSources>0 && sampled[11].emittedMm3>0);
          assert(sampled[16].metrics.bipolarHeatInputJ>0 && Object.keys(sampled[16].woundPorts).length===2);
          assert(Object.values(sampled[16].woundPorts).every(p=>p.sealFraction===1));
          assert.equal(sampled[17].bleedingSources,0);
          const irrigated=sampled[22].metrics.fluidAddedMm3-sampled[21].metrics.fluidAddedMm3;
          const aspirated=sampled[28].metrics.fluidRemovedMm3-sampled[27].metrics.fluidRemovedMm3;
          assert(Math.abs(irrigated-5)<1e-6 && Math.abs(aspirated-5)<1e-6);
          report.pairedEffects={retracted,released,relaxed,earlyWoundEmittedMm3:sampled[11].emittedMm3,finalWoundEmittedMm3:latest.witness.emittedMm3,finalFieldPresentMm3:latest.witness.metrics.fluidPresentMm3,ports:sampled[16].woundPorts,
            bipolarHeatJ:sampled[16].metrics.bipolarHeatInputJ,irrigatedMm3:irrigated,aspiratedMm3:aspirated};
          }
        } else {
        await choose('cusa');
        await number('Tool position R',32);await number('Tool position A',32);await number('Tool position S',55);
        await press('Place tool');
        const pad=byName('Instrument jog pad');
        const padPoint=await point(pad);
        await mouse(padPoint);
        // Continuous proposed pose input exercises the interactive render loop;
        // the accepted engine state must stay fixed until a command is submitted.
        const jogStart=await js('performance.now()');
        await send('dispatchMouseEvent',{type:'mousePressed',x:padPoint.x,y:padPoint.y,button:'left',buttons:1,clickCount:1});
        for(let i=1;i<=75;i++){
          await send('dispatchMouseEvent',{type:'mouseMoved',x:padPoint.x+Math.round(Math.sin(i/14)*24),y:padPoint.y,buttons:1});
          await delay(16);
        }
        await send('dispatchMouseEvent',{type:'mouseReleased',x:padPoint.x,y:padPoint.y,button:'left',buttons:0,clickCount:1});
        const jogFrames=await js(`window.manualEvidence.frames.filter(f=>f.at>=${jogStart}).map(f=>f.at)`);
        report.timingJogCadenceMs=summary(jogFrames.slice(1).map((at,i)=>at-jogFrames[i]));
        assert.equal(latest.witness.stateId,report.actions.at(-1).result.witness.stateId);
        await key('PageDown','PageDown',34);await key('Escape','Escape',27);
        await press('Activate tool');
        await number('Action duration',2);
        await number('Tool position S',66.5);
        const partial=await press('Apply tool','partial');
        await capture('tool-contact');
        await choose('frazier_suction','rejected');
        assert.match(report.actions.at(-1).result.reason,/WITHDRAW_BEFORE_EXCHANGE/);
        await capture('exchange-refused');
        await number('Action duration',2);
        await press('Apply tool');
        assert(latest.witness.metrics.targetRemovedMm3>report.parent.metrics.targetRemovedMm3);
        assert(report.actions.some(a=>a.result.eventKinds.includes('cusa_contact_dwell')));
        assert(report.actions.at(-1).result.cusaAllocatedMm3>0);
        assert(latest.witness.remaining<report.parent.remaining);
        report.controlledAspiration={partial:partial.stateId,dwell:latest.witness.stateId};
        await capture('partial-cavity');
        await press('Release tool');await press('Withdraw tool');
        await choose('frazier_suction');
        await number('Tool position R',26);await number('Tool position A',32);await number('Tool position S',55);
        await number('Action duration',.2);await press('Place tool');
        const beforeSuction=latest.witness.metrics;
        await number('Tool position S',60);await press('Apply tool');await number('Action duration',.75);
        await press('Apply tool');
        assert(latest.witness.metrics.debrisRemovedMm3>beforeSuction.debrisRemovedMm3);
        assert(Math.abs((beforeSuction.debrisMm3-latest.witness.metrics.debrisMm3)-
          (latest.witness.metrics.debrisRemovedMm3-beforeSuction.debrisRemovedMm3))<1e-6);
        await capture('field-suction');await press('Release tool');await press('Withdraw tool');
        }
        report.finalState=latest.witness;
        assert(Object.values(report.finalState.tools).every(t=>!t.inserted));
        await capture('withdrawn');
        report.telemetry=await js('window.manualEvidence.active=false;window.manualEvidence');
        await click('Save');await until("document.body.innerText.includes('Workspace saved locally') && !document.querySelector('.viewer-loading')");
        assert(fs.existsSync(saved));report.savedSha256=hashFile(saved);report.saveWallEpochMs=wall();
        const eventLog=JSON.parse(execFileSync('python3',['-c',
          "import json,sys,zipfile; a=json.loads(zipfile.ZipFile(sys.argv[1]).read('manifest.json'))['artifacts']['surgical_toolbox']; b=a['branchId']; p=a['practiceEpisodes'][b]; s=a['branches'][b]; print(json.dumps(dict(episode=p,position=a['position'],stateId=s['checkpoint']['stateId'],actions=s['actions'][p['parent']['position']:])))",saved],{maxBuffer:16*1024*1024}));
        assert.deepEqual(eventLog.episode,report.episode);assert.equal(eventLog.position,report.finalState.position);
        const steps=report.actions.filter(a=>a.result.status!=='rejected');assert.equal(eventLog.stateId,report.finalState.stateId);
        assert.deepEqual(eventLog.actions,steps.map(a=>a.result.requested));
        report.savedActionsVerified=steps.length;
        if(process.argv.includes('--matched-stills')){
          assert(benchmark&&paired&&!enclosed);
          for(const after of [24,30]){
            const item=report.actions[after];
            const boundary=item.result.witness;
            await js(inputScript('Surgical timeline',boundary.position));
            await ready(boundary.stateId);
            assert.deepEqual(latest.witness,boundary);
            const drawn=await snapshot(boundary);
            assert.deepEqual(drawn.camera,item.display.camera);
            await capture('matched-'+(after===24?'irrigated':'suctioned'));
            (report.matchedStills??=[]).push({after,stateId:boundary.stateId,drawn});
          }
        }
        if(!benchmark)await delay(Math.max(1200,(detail?0:62000)-(performance.now()-recordingStarted)));
      }
      if(stopRecording){report.movie=await stopRecording(path.join(output,take+'.mp4'));stopRecording=undefined;
        if(!cold){
          if(!complete&&!detail) assert(report.movie.wallSeconds>=60 && report.movie.wallSeconds<=120,'Manual take must be 60–120 wall seconds');
          for(const item of report.actions.filter(a=>a.result.status!=='rejected')){
            const frame=report.movie.stateChanges.find(f=>f.stateId===item.result.witness.stateId);
            if(!frame && item.heldInput) continue; // A continuous tick can fall between real compositor samples.
            assert(frame,'Recording missed accepted state '+item.result.witness.stateId);
            item.firstRecordedFrame=frame.frame;
            item.inputToFirstRecordedFrameMs=report.movie.firstCaptureEpochMs+frame.offsetMs-item.inputEpochMs;
          }
        }}
        if(!benchmark&&!cold&&!complete&&!detail){
          win.setContentSize(1050,720);await until('innerWidth===1050 && innerHeight===720');
          await point(byName('Place tool'));await point(byName('Instrument jog pad'));await capture('compact-controls');
          assert((await js('document.documentElement.scrollWidth'))<=1050,'Compact workspace has horizontal overflow');
          win.setContentSize(width,height);await until(`innerWidth===${width} && innerHeight===${height}`);
        }
      if(report.telemetry){
        const t=report.telemetry;
        report.timing={interactiveFrameIntervalMs:report.timingJogCadenceMs,compositorCaptureUpperBoundMs:summary(report.actions.map(a=>a.inputToCapturedViewportUpperBoundMs)),acceptedMs:summary(report.actions.filter(a=>a.result.status!=='rejected').map(a=>a.acceptedMs)),
          visibleGpuFenceMs:summary(report.actions.map(a=>a.visibleMs)),ipcMs:summary(report.actions.map(a=>a.result.ipcMs)),
          responseToVisibleGpuFenceMs:summary(report.actions.filter(a=>a.visibleMs!==undefined).map(a=>a.visibleMs-a.acceptedMs)),
          cpuFrameSubmissionMs:summary(t.frames.map(f=>f.cpuMs)),gpuElapsedMs:summary(t.gpu),
          gpuFenceUpperBoundMs:summary(t.fences.map(f=>f.ms)),incrementalMeshWorkerMs:summary(t.mesh.filter(m=>m.incremental).map(m=>m.ms)),
          meshWorkerMs:summary(t.mesh.map(m=>m.ms)),electronAndEngineWorkingSetMiB:summary(report.memory.map(rows=>rows.reduce((sum,p)=>sum+p.workingSetKiB/1024,0)))};
      }
      assert(!win.isVisible()&&!report.windowShown);assert.equal(hashFile(original),report.originalSha256);
      report.passed=true;persist();console.log(paper);app.exit(0);
    } catch(error) {
      report.error=String(error.stack??error);report.latest=latest&&{...latest,nativeView:undefined};report.pending=pending;
      report.body=await js('document.body.innerText.slice(0,18000)').catch(()=>null);
      report.focus=await js(`({focused:document.hasFocus(),active:document.activeElement?.outerHTML.slice(0,250),inputs:window.manualEvidence?.inputs?.slice(-8)})`).catch(()=>null);
      console.error(error);
      await capture(take+'-failure').catch(()=>{});persist();app.exit(1);
    }
  });
});
app.on('quit',()=>{
  fs.rmSync(scratch,{recursive:true,force:true});
  if(captureScratch)fs.rmSync(captureScratch,{recursive:true,force:true});
});
require('./main.cjs');
