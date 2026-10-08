'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const vm = require('node:vm');
const { createRequire } = require('node:module');
const { execFileSync } = require('node:child_process');
const { EventEmitter } = require('node:events');
const { Sidecar, OPERATIONS } = require('./sidecar.cjs');
const { validateObservedResult, transportDigest, PINS } = require('./observed-landmark-contract.cjs');

const repo = path.resolve(__dirname, '../..');
const studyId = 'resect-case4-sparse-update-v1';

function actualFrames() {
  return JSON.parse(execFileSync(path.join(repo, '.venv/bin/python'), ['-c',
    'import json; from pathlib import Path; from resectionlab.observed_landmark_replay import ObservedLandmarkReplay,B_IDS,transport_digest; '
    + 'r=ObservedLandmarkReplay.load(Path.cwd()); '
    + 'frames=[r.advance(phase,row).frame() for phase in ("before","during") for row in B_IDS]; '
    + 'print(json.dumps([{"result":f,"digest":transport_digest(f)} for f in frames]))'],
  { cwd: repo, env: { ...process.env, PYTHONPATH: path.join(repo, 'src') }, encoding: 'utf8' }));
}

function mainHandlers(engine) {
  const handlers = new Map();
  const mainFile = path.join(__dirname, 'main.cjs');
  const localRequire = createRequire(mainFile);
  const frame = { url: 'file:///trusted-local-app.html' };
  const window = { isDestroyed: () => false, webContents: { mainFrame: frame } };
  const context = vm.createContext({
    __dirname, process, Buffer, console, __engine: engine, __window: window, __url: frame.url,
    require: name => name === 'electron' ? {
      app: { requestSingleInstanceLock: () => false, quit() {}, on() {} },
      ipcMain: { handle: (name, callback) => handlers.set(name, callback) },
    } : localRequire(name),
  });
  vm.runInContext(fs.readFileSync(mainFile, 'utf8'), context);
  vm.runInContext('engine = __engine; window = __window; allowedUrl = __url; bindOperations();', context);
  const event = { sender: window.webContents, senderFrame: frame };
  return { inspect: args => handlers.get('research:inspectObservedLandmarkUpdate')(event, args),
    untrusted: args => handlers.get('research:inspectObservedLandmarkUpdate')({ ...event, sender: {} }, args) };
}

test('actual six-B replay traverses Electron main, sidecar and Python without an anatomical case', async () => {
  const transferDir = fs.mkdtempSync(path.join(os.tmpdir(), 'observed-replay-'));
  const sidecar = new Sidecar({ python: path.join(repo, '.venv/bin/python'), cwd: repo,
    sourcePath: path.join(repo, 'src'), transferDir, observedSourceRoot: repo });
  try {
    assert.equal(OPERATIONS.has('inspectObservedLandmarkUpdate'), true);
    const main = mainHandlers(sidecar);
    const before = await main.inspect({ action: 'open', studyId });
    assert.equal(before.state.phase, 'before');
    assert.equal(before.prediction, null);
    assert.deepEqual(before.state.boundaryIds, [1, 14, 8, 17, 19, 7]);
    const during = await main.inspect({ action: 'advance', studyId,
      expectedStateHash: before.stateHash, phase: 'during', landmarkId: 14 });
    const actual = JSON.parse(fs.readFileSync(path.join(repo, 'artifacts', studyId,
      'fit-freeze/comparison/prediction-field.json'), 'utf8')).baseline.observations;
    assert.deepEqual(during.state.selectedInspectionPoint.positionRasMm, actual.observed_ras_mm[1]);
    assert.equal(during.state.acquisitionTimestamp, null);
    assert.equal(during.state.physicalClearanceMm, null);
    assert.equal(during.state.selectedAction, 'ABSTAIN_UNSUPPORTED_CLEARANCE');
    assert.equal(during.prediction.method, 'proper_rigid');
    await assert.rejects(main.inspect({ action: 'advance', studyId,
      expectedStateHash: before.stateHash, phase: 'during', landmarkId: 1 }), /OBSERVED_STATE_STALE/);
    const reopened = await main.inspect({ action: 'reopen', studyId,
      expectedStateHash: during.stateHash, snapshot: before.snapshot });
    assert.deepEqual(reopened, before);
    // A delayed historical response cannot impersonate the requested later point.
    const staleMain = mainHandlers({ request: async () => before });
    await assert.rejects(staleMain.inspect({ action: 'advance', studyId,
      expectedStateHash: before.stateHash, phase: 'during', landmarkId: 14 }), /does not match/);
    for (const [section, field, value] of [
      ['binding', 'fieldSha256', '0'.repeat(64)], ['state', 'frame', 'LPS+'],
      ['state', 'physicalClearanceMm', 0], ['state', 'acquisitionTimestamp', 0],
      ['snapshot', 'stateHash', during.stateHash],
    ]) {
      const changed = structuredClone(before);
      changed[section][field] = value;
      const wrongMain = mainHandlers({ request: async () => changed });
      await assert.rejects(wrongMain.inspect({ action: 'open', studyId }), /does not match/);
    }
  } finally {
    await sidecar.stop();
    fs.rmSync(transferDir, { recursive: true, force: true });
  }
});

test('main refuses renderer paths, geometry, unknown studies and invalid state selectors before dispatch', async () => {
  let calls = 0;
  const main = mainHandlers({ request: async () => { calls++; } });
  for (const key of ['path', 'observedSourceRoot', 'source_ras_mm', 'observed_ras_mm', 'frame', 'method', 'config', 'toolPose']) {
    await assert.rejects(main.inspect({ action: 'open', studyId, [key]: 'injected' }), /does not match/);
  }
  const invalid = [
    {}, { action: 'open', studyId: 'other' }, { action: 'open', studyId, phase: 'during' },
    { action: 'advance', studyId, expectedStateHash: null, phase: 'during', landmarkId: 1 },
    { action: 'advance', studyId, expectedStateHash: 'sha256:' + '0'.repeat(64), phase: 'after', landmarkId: 1 },
    { action: 'advance', studyId, expectedStateHash: 'sha256:' + '0'.repeat(64), phase: 'during', landmarkId: 2 },
    { action: 'advance', studyId, expectedStateHash: 'sha256:' + '0'.repeat(64), phase: 'during', landmarkId: true },
    { action: 'reopen', studyId, expectedStateHash: null, snapshot: { path: 'injected' } },
  ];
  for (const args of invalid) await assert.rejects(main.inspect(args));
  await assert.rejects(main.untrusted({ action: 'open', studyId }), /Untrusted desktop frame/);
  assert.equal(calls, 0);
});

test('preload exposes only the named operation and forwards its request unchanged', async () => {
  let api;
  const calls = [];
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, 'preload.cjs'), 'utf8'), {
    require: name => {
      assert.equal(name, 'electron');
      return { contextBridge: { exposeInMainWorld: (_name, value) => { api = value; } },
        ipcRenderer: { invoke: async (...values) => { calls.push(values); } } };
    },
  });
  assert.equal(Object.isFrozen(api), true);
  assert.equal(api.request, undefined);
  const args = { action: 'open', studyId };
  await api.inspectObservedLandmarkUpdate(args);
  assert.deepEqual(calls, [['research:inspectObservedLandmarkUpdate', args]]);
});

test('all 12 real phase/row states have the same Python and JavaScript transport digest', () => {
  const frames = actualFrames();
  assert.equal(frames.length, 12);
  for (const { result, digest } of frames) {
    const key = `${result.state.phase}:${result.state.selectedInspectionPoint.landmarkId}`;
    assert.equal(transportDigest(result), digest);
    assert.equal(digest, PINS[key]);
    assert.equal(validateObservedResult(result, { action: 'open', studyId }), result);
    assert.equal(Object.isFrozen(result.state.observations[0].sourceRasMm), true);
  }
});

test('sidecar rejects corrupted real responses before asset traversal, event emission or resolution', async () => {
  const frames = actualFrames();
  const mutations = [
    r => { r.validationLandmarks = 'FORBIDDEN_SENTINEL'; },
    r => { r.state.observations[0].sourceRasMm[0] += 1; },
    r => { r.state.observations[0].landmarkId = 2; },
    r => { r.state.observations[0].observedRasMm = r.state.observations[0].sourceRasMm; },
    r => { r.state.elapsedSeconds = 1; },
    r => { r.state.measurementAvailabilityTimestamp = 1; },
    r => { r.state.observations[0].sourceRasMm[0] = Infinity; },
    r => { r.state.toolPose = { path: 'FORBIDDEN_SENTINEL', dtype: 'float64', byteLength: 8 }; },
    r => { r.binding.sourceImageSha256 = 'sha256:' + '0'.repeat(64); },
    r => { r.snapshot.landmarkId = 14; },
    r => { r.stateHash = r.snapshot.stateHash = 'sha256:' + '0'.repeat(64); },
    r => { r.resultHash = r.snapshot.resultHash = 'sha256:' + '0'.repeat(64); },
  ];
  for (const mutate of mutations) {
    const response = structuredClone(frames[0].result); mutate(response);
    const sidecar = Object.create(Sidecar.prototype);
    const events = []; let traversals = 0, resolutions = 0;
    sidecar.pending = new Map([['test', { op: 'inspectObservedLandmarkUpdate',
      observedArgs: { action: 'open', studyId }, resolve() { resolutions++; }, reject() {} }]]);
    sidecar.emit = (...args) => events.push(args);
    sidecar.assets = { expose: async value => { traversals++; return value; } };
    await assert.rejects(sidecar.handle({ id: 'test', event: 'result', result: response }));
    assert.equal(traversals, 0); assert.equal(resolutions, 0); assert.deepEqual(events, []);
  }
  const during = structuredClone(frames[6].result);
  during.state.observations[0].observedRasMm[0] += 1;
  assert.throws(() => validateObservedResult(during, { action: 'open', studyId }));
  const prediction = structuredClone(frames[6].result);
  prediction.prediction.positionRasMm[0] += 1;
  assert.throws(() => validateObservedResult(prediction, { action: 'open', studyId }));
});

test('packaged main and packaged sidecar omit new arguments for the existing bundled engine', async () => {
  let config, spawnArgs;
  const mainFile = path.join(__dirname, 'main.cjs');
  const localRequire = createRequire(mainFile);
  const context = vm.createContext({ __dirname, Buffer, console,
    process: { ...process, resourcesPath: '/packaged/resources' },
    require: name => {
      if (name === 'electron') return { app: { isPackaged: true, getPath: () => '/user/data',
        requestSingleInstanceLock: () => false, quit() {}, on() {} } };
      if (name === 'node:fs/promises') return { mkdtemp: async () => '/temporary/transfers',
        chmod: async () => {}, mkdir: async () => {}, access: async () => {} };
      if (name === './sidecar.cjs') return { Sidecar: class { constructor(value) { config = value; } on() {} } };
      return localRequire(name);
    },
  });
  vm.runInContext(fs.readFileSync(mainFile, 'utf8'), context);
  await vm.runInContext('startEngine()', context);
  assert.equal(Object.hasOwn(config, 'observedSourceRoot'), false);
  assert.equal(config.executable, '/packaged/resources/research-engine/ressectionlab-engine');
  const child = new EventEmitter();
  child.stdin = new EventEmitter(); child.stdout = new EventEmitter(); child.stderr = new EventEmitter();
  child.stdout.setEncoding = () => {};
  const module = { exports: {} };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, 'sidecar.cjs'), 'utf8'), {
    module, process, Buffer, setTimeout, clearTimeout,
    require: name => name === 'node:child_process' ? {
      spawn: (_command, args) => { spawnArgs = args; return child; },
    } : localRequire(name),
  });
  new module.exports.Sidecar({ executable: '/trusted/existing-engine', cwd: repo,
    transferDir: '/temporary/transfers', observedSourceRoot: repo });
  assert.deepEqual(Array.from(spawnArgs), ['--transfer-dir', '/temporary/transfers']);
});
