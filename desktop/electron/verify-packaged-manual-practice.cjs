'use strict';

// Loaded at the real packaged main's first statement by the hidden launcher.
// Reuse the input and state checks from the normal manual verifier.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const {createHash} = require('node:crypto');
const {fileURLToPath} = require('node:url');
const {app, BrowserWindow, dialog} = require('electron');

const {PACKAGED_PRACTICE_APP, PACKAGED_PRACTICE_CASE, PACKAGED_PRACTICE_ACTIONS, PACKAGED_PRACTICE_OUTPUT} = process.env;
const cold = process.env.PACKAGED_PRACTICE_COLD === '1';
const appDir = fs.realpathSync(PACKAGED_PRACTICE_APP);
const output = path.resolve(PACKAGED_PRACTICE_OUTPUT);
const phase = cold ? 'cold' : 'warm';
const engineDir = fs.realpathSync(path.join(appDir, 'Contents/Resources/research-engine'));
const executable = fs.realpathSync(path.join(engineDir, 'ressectionlab-engine'));
const appCode = path.join(appDir, 'Contents/Resources/app.asar');
const report = {schema: 'resectionlab.packaged-manual-practice.v1', phase, pid: process.pid, appCode,
  appExecutable: process.execPath, productionEngine: executable, isPackaged: app.isPackaged,
  passed: false, stageTiming: [], windowEverVisible: false};
const persist = () => fs.writeFileSync(path.join(output, 'packaged-' + phase + '.json'), JSON.stringify(report, null, 2) + '\n');
const sha = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const appExit = app.exit.bind(app);
function fatal(error) {
  report.error = String(error.stack ?? error);
  console.error(error);
  for (const window of BrowserWindow.getAllWindows()) window.hide();
  try { persist(); } finally { appExit(1); }
}
process.prependListener('uncaughtException', fatal);
process.prependListener('unhandledRejection', fatal);
dialog.showErrorBox = (title, content) => fatal(Error(title + ' ' + content));

assert(app.isPackaged);
assert.equal(app.getAppPath(), appCode);
assert.equal(fs.realpathSync(process.execPath), path.join(appDir, 'Contents/MacOS/RessectionLab'));
assert(process.argv.includes('--automation-hidden'));
assert(fs.existsSync(output));
app.setAppLogsPath(path.join(output, 'packaged-' + phase + '-logs'));
process.env.RESSECTIONLAB_REHEARSAL_TIMING = '1';
delete process.env.RESSECTIONLAB_NUMERICAL_BACKEND;
delete process.env.RESSECTIONLAB_DATA_ROOT;

const sidecar = require(path.join(appCode, 'electron/sidecar.cjs'));
let engine, opened, before, exiting = false;
sidecar.Sidecar = class ObservedProductionSidecar extends sidecar.Sidecar {
  constructor(options) {
    assert.equal(fs.realpathSync(options.executable), executable);
    assert.equal(options.python, undefined);
    assert.equal(options.sourcePath, undefined);
    assert.equal(options.observedSourceRoot, undefined);
    super(options);
    engine = this;
    this.on('rehearsalTiming', row => report.stageTiming.push(row));
  }
};

async function identity() {
  const {runtime, numericalBackend} = await engine.request('ping', {}, 30000);
  assert(runtime.frozen);
  assert.equal(runtime.architecture, 'arm64');
  assert.equal(fs.realpathSync(runtime.executable), executable);
  assert.equal(numericalBackend.selected, 'cpp');
  const binary = fs.realpathSync(numericalBackend.binaryPath);
  assert(binary.startsWith(engineDir + path.sep));
  assert.equal(sha(binary), numericalBackend.binarySha256);
  return {runtime, numericalBackend};
}
app.on('browser-window-created', (_, window) => {
  opened = window;
  window.on('show', () => { report.windowEverVisible = true; window.hide(); });
  window.webContents.once('did-finish-load', () => {
    before = identity();
    before.catch(() => {});
  });
});

app.exit = code => {
  if (exiting) return;
  exiting = true;
  void (async () => {
    try {
      const proof = JSON.parse(fs.readFileSync(path.join(output, cold ? 'benchmark-cold.json' : 'benchmark.json'), 'utf8'));
      assert.equal(code, 0, proof.error);
      assert(proof.passed);
      assert(opened && !opened.isVisible() && !report.windowEverVisible && !proof.windowShown);
      const renderer = fileURLToPath(opened.webContents.getURL());
      assert.equal(renderer, path.join(appCode, 'dist/index.html'));
      const assetUrl = await opened.webContents.executeJavaScript("document.querySelector('script[type=module][src]').src");
      const asset = fileURLToPath(assetUrl);
      assert(asset.startsWith(path.join(appCode, 'dist/assets') + path.sep));
      report.loadedRenderer = {index: renderer, indexSha256: sha(renderer), asset, assetSha256: sha(asset)};
      report.engineBefore = await before;
      report.engineAfter = await identity();
      report.finalState = cold ? proof.replayed.at(-1) : proof.finalState.stateId;
      report.reopened = cold ? proof.replayed : null;
      const warm = cold ? JSON.parse(fs.readFileSync(path.join(output, 'packaged-warm.json'), 'utf8')) : null;
      if (cold) {
        assert(warm.passed && warm.pid !== report.pid);
        assert.equal(warm.appExecutable, report.appExecutable);
        assert.deepEqual(warm.loadedRenderer, report.loadedRenderer);
        assert.equal(warm.engineAfter.numericalBackend.binarySha256, report.engineAfter.numericalBackend.binarySha256);
        assert.equal(warm.finalState, report.finalState);
        assert.equal(warm.savedSha256, proof.originalSha256);
        assert(proof.coldHistory.length > 0);
        report.reopenedHistory = proof.coldHistory.length;
      } else {
        assert(proof.pairedEffects?.retracted > proof.pairedEffects?.released);
        assert(proof.pairedEffects.bipolarHeatJ > 0);
        report.savedSha256 = proof.savedSha256;
        report.pairedEffects = proof.pairedEffects;
        const actions = new Set(proof.actions.map(row => row.result.witness.stateId));
        const rows = report.stageTiming.filter(row => row.process === 'python' && row.command === 'step' &&
          row.terminalEvent === 'result' && actions.has(row.responseStateId));
        assert.equal(rows.length, proof.actions.length, 'Collect native execution counters for every real packaged action');
        const total = name => rows.reduce((sum, row) => sum + (row.deformationKernels?.[name] ?? 0), 0);
        report.manualKernels = Object.fromEntries([
          'continuumTicks', 'latticeTicks', 'latticeSubsteps', 'legacySweepObstacleCalls',
          'sweepNativeCalls', 'sweepNativeSamplePoints', 'sweepVectorCalls', 'strainVectorCalls',
          'strainNativeCalls', 'strainNativeEdges', 'strainRefinedEdges'
        ].map(name => [name, total(name)]));
        for (const name of ['sweepFallbackReasons', 'strainFallbackReasons']) {
          const counts = {};
          for (const row of rows) {
            for (const [reason, count] of Object.entries(row.deformationKernels?.[name] ?? {}))
              counts[reason] = (counts[reason] ?? 0) + count;
          }
          report.manualKernels[name] = counts;
        }
        const {continuumTicks, latticeTicks, latticeSubsteps} = report.manualKernels;
        assert(latticeTicks > 0 || continuumTicks > 0, 'The signed app must actually run tissue deformation');
        if (process.env.PACKAGED_EXPECT_NATIVE_SWEEP === '1') {
          assert(latticeSubsteps > 0 && report.manualKernels.sweepNativeCalls > 0 && report.manualKernels.sweepNativeSamplePoints > 0,
            'The signed app must dispatch native swept contact on documented samples');
        }
      }
      report.passed = true;
    } catch (error) {
      report.error = String(error.stack ?? error);
      console.error(error);
    }
    try {
      if (engine && !engine.closed) await engine.stop();
    } catch (error) {
      report.passed = false;
      report.error ??= String(error);
    }
    persist();
    appExit(report.passed ? 0 : 1);
  })().catch(fatal);
};

const manual = path.join(__dirname, 'verify-manual-practice.cjs');
const devMain = path.join(__dirname, 'main.cjs');
process.argv.splice(1, Infinity, manual, PACKAGED_PRACTICE_CASE, output,
  cold ? '--cold-benchmark' : '--benchmark', '--paired-actions', PACKAGED_PRACTICE_ACTIONS);
// The verification script normally requires the development main after
// installing its hooks. Its packaged counterpart is already paused. Suppress
// that single development import and allow the real packed main to resume.
assert(!require.cache[devMain]);
const placeholder = new Module(devMain, module);
placeholder.loaded = true;
require.cache[devMain] = placeholder;
try { require(manual); } finally { delete require.cache[devMain]; }
