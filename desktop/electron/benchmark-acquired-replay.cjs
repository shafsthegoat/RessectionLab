'use strict';

// electron benchmark-acquired-replay.cjs APP_ROOT INPUTS_JSON NEW_OUTPUT MODE TRIALS
// Open the two frozen studies through the normal desktop. Sessions stay hidden.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {execFileSync} = require('node:child_process');
const {app, BrowserWindow, dialog, ipcMain} = require('electron');
const {inputScript, displayIdentity} = require('./verify-eden-planning-helpers.cjs');
const {delay, stateWitness} = require('./manual-practice-evidence.cjs');
const {alignRendererClock} = require('./practice-performance-evidence.cjs');
const {installFieldProbe} = require('./acquired-field-probe.cjs');

function exitHidden(error) {
  console.error(error);
  for (const window of BrowserWindow.getAllWindows()) window.hide();
  app.exit(1);
}
process.prependListener('uncaughtException', exitHidden);
process.prependListener('unhandledRejection', exitHidden);
dialog.showErrorBox = (title, content) => exitHidden(Error(title + ' ' + content));

const [, , appArg, inputsArg, outputArg, mode, trialsArg] = process.argv;
assert(['ucsf', 'bite'].includes(mode), 'Select ucsf or bite');
const trials = Number(trialsArg);
assert(Number.isSafeInteger(trials) && trials >= 1 && trials <= 10);
const appRoot = fs.realpathSync(appArg);
const workRoot = path.resolve(__dirname, '../..');
const fixtureRoot = fs.realpathSync(path.join(workRoot, 'testcases'));
const output = path.resolve(outputArg);
assert(output.startsWith(fixtureRoot + path.sep) && !fs.existsSync(output), 'Choose a fresh output in testcases');
const manifest = JSON.parse(fs.readFileSync(inputsArg, 'utf8'));
const hash = data => createHash('sha256').update(data).digest('hex');
const hashFile = filename => hash(fs.readFileSync(filename));
const sourceFiles = {};
function fixture(key) {
  const {filename, sha256} = manifest[key];
  const file = fs.realpathSync(filename);
  assert(file.startsWith(fixtureRoot + path.sep), 'Use a verified copy in testcases');
  assert.equal(hashFile(file), sha256, key + ' bytes changed');
  sourceFiles[key] = {filename: file, sha256, bytes: fs.statSync(file).size};
  return file;
}
const member = (file, name) => JSON.parse(execFileSync('/usr/bin/unzip', ['-p', file, name], {maxBuffer: 64 * 1024 * 1024}));
const archive = fixture(mode === 'bite' ? 'biteStudy' : 'ucsfStudy');
const study = member(archive, 'patient-anatomy.json');
const episode = mode === 'ucsf' ? fixture('ucsfEpisode') : null;
const sealed = episode ? member(episode, 'manifest.json').artifacts.surgical_toolbox : null;
const exact = sealed?.exactGridReplay;
const history = sealed?.timeline[sealed.branchId];
const savedView = episode ? member(episode, 'workspace.json').imagingState : null;
const savedInspection = savedView?.states?.[savedView.selectedSeriesId ?? 'primary']?.viewer?.resultInspection;
if (exact) {
  assert.equal(exact.sourceManifest.snapshot_id, study.snapshot_id);
  assert.equal(exact.branchId, sealed.branchId);
  assert.deepEqual(exact.observations.map(o => o.stateId), [history[0].stateBefore, ...history.map(h => h.stateAfter)]);
}
fs.mkdirSync(output, {recursive: true});
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'rslab-acquired-'));
app.setPath('userData', profile);
app.setAppLogsPath(path.join(profile, 'logs'));
process.argv.push('--automation-hidden', '--anatomy', archive);
delete process.env.RESSECTIONLAB_DEV_URL;
delete process.env.RESSECTIONLAB_DATA_ROOT;
delete process.env.RESSECTIONLAB_PREFIX_PIPELINE;
if (process.platform === 'darwin') app.setActivationPolicy('accessory');
dialog.showOpenDialog = async (...args) => {
  assert(episode && /recorded engine episode/i.test(args.at(-1).title), 'Unexpected file chooser');
  return {canceled: false, filePaths: [episode]};
};
dialog.showMessageBox = async (...args) => { throw Error('Unexpected dialog ' + args.at(-1).title); };

const launched = performance.now();
const epoch = () => performance.timeOrigin + performance.now();
const assets = path.join(appRoot, 'desktop/dist/assets');
const report = {
  schema: 'resectionlab.acquired-desktop-performance.v1',
  mode, appRoot, pid: process.pid, trials, sourceFiles, passed: false, windowEverVisible: false, captures: 0,
  workload: mode === 'ucsf' ? 'Source-bound recorded boundary scrubbing' : 'Geometric aspiration phase scrubbing',
  timingBoundary: 'Timeline input to GPU fence poll for the matching rendered boundary',
  hardware: {cpu: os.cpus()[0].model, memoryGiB: os.totalmem() / 2 ** 30, os: os.release(), arch: os.arch(), electron: process.versions.electron,
    systemVersion: process.platform === 'darwin' ? execFileSync('/usr/bin/sw_vers', ['-productVersion'], {encoding: 'utf8'}).trim() : os.version()},
  buildFiles: Object.fromEntries(['desktop/electron/main.cjs', 'desktop/electron/sidecar.cjs', 'desktop/dist/index.html', 'src/resectionlab/surgical_bridge.py'].map(name =>
    [name, hashFile(path.join(appRoot, name))])),
  rendererAssets: Object.fromEntries(fs.readdirSync(assets).filter(name => /^(index-|.*worker-).*\.(js|css)$/.test(name)).sort().map(name =>
    [name, hashFile(path.join(assets, name))])),
  study: {patientId: study.patient_id, snapshotId: study.snapshot_id, referenceSeries: study.reference_series,
    series: study.series.map(({id, modality, affine, grid_spacing_mm: spacingMm, data}) =>
      ({id, modality, affine, spacingMm, sha256: data.sha256, shape: data.shape}))},
  requests: [], stageTiming: [], samples: [], warmup: []
};
if (sealed) report.replay = {
  caseHash: exact.caseHash, sceneId: exact.sceneId, branchId: sealed.branchId, contractHash: exact.contractHash,
  sourceMode: exact.sourceMode, targetLayerId: exact.targetLayerId, approach: exact.recordedApproach,
  physicsProfile: sealed.physicsProfile, actions: history.map(row => row.action), boundaryIds: exact.observations.map(row => row.stateId)
};

const sidecar = require(path.join(appRoot, 'desktop/electron/sidecar.cjs'));
let engine, lastSeek, field;
sidecar.Sidecar = class ObservedSidecar extends sidecar.Sidecar {
  constructor(options) {
    super(options);
    engine = this;
    this.on('rehearsalTiming', value => report.stageTiming.push(value));
  }
};
const backend = async () => {
  const ping = await engine.request('ping', {}, 30000);
  return {runtime: ping.runtime, numericalBackend: ping.numericalBackend};
};

const register = ipcMain.handle.bind(ipcMain);
ipcMain.handle = (channel, handler) => register(channel, async (...args) => {
  const start = epoch();
  const result = await handler(...args);
  const end = epoch();
  const request = args[1];
  if (channel === 'research:anatomyPlan' && request.action === 'operativeField') {
    field = {
      shape: result.shape, spacingMm: result.spacingMm, radiusMm: result.radiusMm, approachId: result.approachId,
      targetLayerId: result.targetLayerId, labelsSha256: result.labels.sha256, scanSha256: result.intensity?.sourceSha256,
      stations: result.stations, parameterHash: result.parameterHash
    };
  }
  if (channel === 'research:openBoundSurgicalReplay') report.binding = result.verifiedStudyReplay;
  if (channel === 'research:surgicalRehearsal') {
    const row = {command: request.command, requestId: request.requestId, position: result.position,
      caseHash: result.caseHash, branchId: result.branchId, stateId: result.view?.stateId,
      start, end, ipcMs: end - start};
    report.requests.push(row);
    if (request.command === 'seek') lastSeek = {row, result};
  }
  return result;
});

const scope = '[data-engine-replay-layout] .engine-operated-view';
const pane = `${scope} [data-accepted-tools]`;
function installReplayProbe(scope) {
  const data = window.acquiredReplay = {pending: null};
  const collect = name => ({target, detail}) => {
    const pending = data.pending;
    if (!pending || !target.closest(scope)) return;
    const at = performance.now();
    if (name === 'work') pending.work.push({at, ...detail});
    else if (detail.surgicalStateId === pending.id && detail.sourceCaseHash === pending.caseHash) {
      if (name === 'frame' && detail.at >= pending.started) pending.frames.set(detail.serial, detail);
      if (name === 'gpu' && detail.completed && pending.frames.has(detail.serial))
        pending.fences.set(detail.serial, {at, fence: detail});
      if (name === 'publication' && pending.fences.has(detail.frameSerial) && !pending.completed) {
        pending.completed = {...pending.fences.get(detail.frameSerial), frame: pending.frames.get(detail.frameSerial), publication: {at, ...detail}};
        pending.frames.clear();
        pending.fences.clear();
      }
    }
  };
  document.addEventListener('anatomy-frame', collect('frame'));
  document.addEventListener('anatomy-gpu', collect('gpu'));
  document.addEventListener('anatomy-work', collect('work'));
  document.addEventListener('anatomy-surgical-display', collect('publication'));
}

app.on('browser-window-created', (_, win) => {
  win.on('show', () => { report.windowEverVisible = true; win.hide(); });
  win.webContents.once('did-finish-load', async () => {
    const wc = win.webContents;
    const js = expression => wc.executeJavaScript(expression, true);
    const wait = async (predicate, label, limit = 240000) => {
      const deadline = performance.now() + limit;
      while (performance.now() < deadline) {
        const value = await predicate();
        if (value) return value;
        await delay(50);
      }
      throw Error('Timed out waiting for ' + label);
    };
    const until = expression => wait(() => js(expression), expression);
    const button = label => `[...document.querySelectorAll('button')].find(b => !b.disabled && b.checkVisibility() &&
      (b.textContent.trim() === ${JSON.stringify(label)} || b.getAttribute('aria-label') === ${JSON.stringify(label)}))`;
    const click = async label => {
      const control = button(label);
      await until(`!!(${control})`);
      await js(`(${control}).click()`);
    };
    const ready = position => {
      const id = exact.observations[position].stateId;
      return until(`document.querySelector(${JSON.stringify(pane)})?.dataset.surgicalStateId === ${JSON.stringify(id)} &&
        document.querySelector(${JSON.stringify(scope)})?.dataset.surfacesPending === '0' &&
        document.querySelector('[aria-label="Surgical timeline"]')?.disabled === false &&
        !!document.querySelector('[data-operative-readout-state=' + CSS.escape(${JSON.stringify(id)}) + ']')`);
    };
    try {
      wc.setBackgroundThrottling(false);
      win.setContentSize(1440, 900);
      await until('innerWidth === 1440 && innerHeight === 900');
      report.viewport = await js('({width: innerWidth, height: innerHeight, devicePixelRatio})');
      report.loadedRenderer = await js("document.querySelector('script[type=module][src]')?.src.split('/').pop()");
      assert(report.rendererAssets[report.loadedRenderer]);
      report.gpu = await app.getGPUInfo('basic');
      report.backendBefore = await backend();
      const sourceReady = `document.querySelector('.rl-viewer')?.dataset.rendererReady === 'true' &&
        JSON.parse(document.querySelector('.rl-viewer')?.dataset.sourceDisplayIdentity || '[]')[3] === ${JSON.stringify(study.snapshot_id)} &&
        !document.querySelector('.viewer-loading,.rl-viewer-preparing')`;
      await until(sourceReady);
      report.nativeReadyMs = performance.now() - launched;
      console.log('Acquired source ready', report.nativeReadyMs.toFixed(0), 'ms');
      report.display = displayIdentity(await js("document.querySelector('.rl-viewer').dataset.sourceDisplayIdentity"));
      await click('Plan');
      await until("!!document.querySelector('.approach-card.selected') && !document.querySelector('.approach-computing')");
      const selected = await js("document.querySelector('.approach-card.selected').dataset.approachId");
      if (exact) assert.equal(selected, exact.recordedApproach.id);
      if (mode === 'bite') await js(`(${installFieldProbe})()`);
      await click('Preview operative steps');
      await until("!!document.querySelector('.operative-field-canvas canvas[data-camera]')");
      assert.equal(field.approachId, selected);
      report.field = field;
      if (mode === 'ucsf') {
        await js(`(${installReplayProbe})(${JSON.stringify(scope)})`);
        await click('Open recorded engine episode');
        await until(`!!document.querySelector(${JSON.stringify(pane)})`);
        const endpoint = history.length;
        assert.equal(report.binding.sourceSnapshotId, study.snapshot_id);
        assert.equal(report.binding.targetLayerId, exact.targetLayerId);
        assert.equal(report.binding.caseHash, exact.caseHash);
        assert.equal(report.binding.sceneId, exact.sceneId);
        assert.equal(report.binding.branchId, exact.branchId);
        assert.equal(report.binding.contractHash, exact.contractHash);
        assert.equal(report.binding.endpointStateId, exact.endpointStateId);
        if (savedInspection) {
          await until(`document.querySelector(${JSON.stringify(scope)})?.dataset.resultInspection === ${JSON.stringify(savedInspection)}`);
          report.restoredInspection = savedInspection;
        }
        if (await js(`!!(${button('Return to operative field')})`)) await click('Return to operative field');
        if (await js(`document.querySelector(${JSON.stringify(pane)}).dataset.operativeView`) !== 'surgeon')
          await click('Surgeon / Microscope');
        await ready(endpoint);
        report.bindingReadyMs = performance.now() - launched;
        console.log('Acquired recording bound', report.bindingReadyMs.toFixed(0), 'ms');
        report.practiceAdmission = await js(`(()=>{const entry=document.querySelector('.practice-entry');return {
          disabled:entry?.querySelector('button').disabled, text:entry?.innerText};})()`);
        assert.equal(report.practiceAdmission.disabled, true);
        assert.match(report.practiceAdmission.text, /recorded playback only/i);
        report.clock = await alignRendererClock(js);
        const seek = async (position, rows) => {
          const id = exact.observations[position].stateId;
          lastSeek = null;
          const input = await js(`(()=>{const data=window.acquiredReplay, at=performance.now();
            data.pending={id:${JSON.stringify(id)},caseHash:${JSON.stringify(exact.caseHash)},started:at,work:[],frames:new Map(),fences:new Map()};
            ${inputScript('Surgical timeline', position)};return at;})()`);
          const completed = await wait(() => js('window.acquiredReplay.pending.completed'), 'matching replay GPU fence');
          await ready(position);
          assert(lastSeek, 'A replay seek must use production IPC');
          const {row, result} = lastSeek;
          const work = await js('window.acquiredReplay.pending.work');
          await js('window.acquiredReplay.pending = null');
          assert.equal(row.position, position);
          assert.equal(row.stateId, id);
          assert.equal(row.caseHash, exact.caseHash);
          assert.equal(row.branchId, exact.branchId);
          assert(row.end >= row.start && completed.at >= input);
          for (const key of ['displayIdentity', 'modelGeneration', 'sourceSurfaceGeneration', 'sceneId', 'surgicalFrame'])
            assert.equal(completed.frame[key], completed.fence[key], 'GPU fence has a different ' + key);
          assert(!completed.frame.preview && !completed.fence.preview);
          const witness = stateWitness(result);
          assert.equal(witness.remaining, exact.observations[position].residualTargetMm3);
          const display = await js(`(()=>{const p=document.querySelector(${JSON.stringify(pane)});return {
            camera:JSON.parse(p.dataset.opticalCamera || 'null'), source:p.closest('.rl-viewer').dataset.sourceDisplayIdentity,
            hud:p.querySelector('[data-operative-readout-state]')?.textContent, tools:JSON.parse(p.dataset.acceptedTools),
            removed:Number(p.dataset.removedCells), view:p.dataset.operativeView, viewport:p.getBoundingClientRect().toJSON()};})()`);
          assert.equal(display.view, 'surgeon');
          assert(display.camera);
          assert.deepEqual(display.tools, witness.tools);
          assert.equal(display.removed, witness.removedCells);
          assert(display.hud?.includes('Owned target removed ' + witness.metrics.targetRemovedMm3.toFixed(1) + ' mm³'));
          const timing = report.clock.valid ? {
            responseToGpuMs: completed.at + report.clock.offset - row.end,
            queueAndDispatchMs: row.start - (input + report.clock.offset)
          } : {};
          rows.push({position, inputRendererMs: input, inputToGpuMs: completed.at - input, ipcMs: row.ipcMs, ...timing,
            requestId: row.requestId, witness, display, ...completed, work});
        };
        for (const position of [3, endpoint, 0]) await seek(position, report.warmup);
        const ascending = history.map((_, index) => index + 1);
        const descending = history.map((_, index) => endpoint - 1 - index);
        for (let trial = 0; trial < trials; trial++) {
          for (const position of [...ascending, ...descending]) await seek(position, report.samples);
          console.log('Completed acquired replay round', trial + 1, 'of', trials);
        }
        assert(report.requests.every(row => ['inspect', 'seek'].includes(row.command)),
          'The acquired benchmark may only inspect archived boundaries');
      } else {
        await until(`[...document.querySelectorAll('.rl-viewer')].filter(v=>v.checkVisibility()).every(v=>
          v.dataset.surfacesPending==='0'&&!v.querySelector('.viewer-loading,.rl-viewer-preparing'))`);
        report.geometricReadyMs = performance.now() - launched;
        const end = await js("Number(document.querySelector('[aria-label=\"Operative timeline\"]').max)");
        const expectedRemoved = {3.2: 380, 3.55: 1847, 4: 4741};
        assert.equal(end, 4, 'Use the four saved BITE geometric chapters');
        const seek = async (position, rows) => {
          await js(`window.acquiredField.start(${position}, () => ${inputScript('Operative timeline', position)})`);
          const sample = await wait(() => js('window.acquiredField.sample'), 'geometric phase GPU fence');
          assert.equal(sample.position, position);
          assert(sample.triangles > 0);
          const expected = expectedRemoved[position];
          assert(expected);
          await until(`Number(document.querySelector('.operative-cavity')?.dataset.removedCells) === ${expected}`);
          const removed = await js("Number(document.querySelector('.operative-cavity').dataset.removedCells)");
          rows.push({...sample, removed});
        };
        for (const position of [end - .8, end, end - .8]) await seek(position, report.warmup);
        for (let trial = 0; trial < trials; trial++) {
          for (const position of [end - .45, end, end - .8, end, end - .45, end - .8]) await seek(position, report.samples);
        }
        report.visibleMeshes = await js('window.acquiredField.visibleMeshes()');
        assert(report.visibleMeshes.some(mesh => mesh.triangles > 1_000_000), 'The acquired field must include a verified large source mesh');
        report.contextMeshes = await js("JSON.parse(document.querySelector('.operative-field-canvas').dataset.surroundingSources).map(({role,triangles,exposedVertices,sampledDataSha256}) => ({role,triangles,exposedVertices,sampledDataSha256}))");
      }
      report.backendAfter = await backend();
      report.memory = app.getAppMetrics().map(({type, memory}) => ({type, workingSetKiB: memory.workingSetSize}));
      for (const {filename, sha256} of Object.values(sourceFiles)) assert.equal(hashFile(filename), sha256);
      assert(!report.windowEverVisible && !win.isVisible());
      report.passed = true;
    } catch (error) {
      report.error = String(error.stack ?? error);
      report.body = await js('document.body.innerText.slice(-14000)').catch(() => null);
      console.error(error);
    } finally {
      fs.writeFileSync(path.join(output, 'report.json'), JSON.stringify(report, null, 2) + '\n');
      if (engine && !engine.closed) await engine.stop().catch(() => {});
      fs.rmSync(profile, {recursive: true, force: true});
      app.exit(report.passed ? 0 : 1);
    }
  });
});
require(path.join(appRoot, 'desktop/electron/main.cjs'));
