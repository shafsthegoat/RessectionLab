'use strict';

// Use the generated bipolar thermal control. Reopen in a separate Electron process:
// electron verify-practice-stress.cjs SOURCE OUTPUT
// electron verify-practice-stress.cjs SOURCE OUTPUT --cold
// Add --smoke to both invocations for a five-second functional pilot.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { execFileSync } = require('node:child_process');
const { app, BrowserWindow, dialog, ipcMain } = require('electron');
const { delay, stateWitness, checkAction } = require('./manual-practice-evidence.cjs');

const [sourceArgument, outputArgument, ...options] = process.argv.slice(2);
const smoke = options.includes('--smoke');
const cold = options.includes('--cold');
assert(options.every(option => ['--cold', '--smoke'].includes(option)));
const root = path.resolve(__dirname, '../..');
const main = path.resolve(process.env.REHEARSAL_PERF_MAIN ?? path.join(__dirname, 'main.cjs'));
const buildRoot = path.resolve(path.dirname(main), '../..');
const source = fs.realpathSync(sourceArgument);
const output = path.resolve(outputArgument);
assert(output.startsWith(path.join(root, 'testcases', 'outputs') + path.sep), 'Use an ignored test output');
const savePath = path.join(output, 'stress.ressectionlab');
const receipt = path.join(output, cold ? 'cold.json' : 'stress.json');
assert(!fs.existsSync(receipt) && (cold || !fs.existsSync(savePath)), 'Use a fresh hidden run directory');
assert(smoke || process.env.RESSECTIONLAB_REHEARSAL_TIMING === '1', 'Enable the hidden rehearsal timing monitor');
const caseHash = 'sha256:c298887cb67b1034d782d682b01ee03cdaf86ee13237602eb3239ec114d5a710';
const parentBranch = 'configured-start';
const parentPosition = 1;
const hashFile = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const sourceSha256 = hashFile(source);
assert.equal(sourceSha256, 'a2ce46b66912199e891c761fff64114444ed925cf5b2a9ffbecbf93db2b8ce2a',
  'This workload requires the saved generated tissue and wound fixture');
const previous = cold ? JSON.parse(fs.readFileSync(path.join(output, 'stress.json'), 'utf8')) : null;
if (cold) {
  assert(previous.passed && previous.mode === (smoke ? 'smoke' : 'stress'));
  assert.equal(hashFile(savePath), previous.save.sha256);
}
const startedNs = process.hrtime.bigint();
const toMs = ns => Number(BigInt(ns) - startedNs) / 1e6;
const now = () => toMs(process.hrtime.bigint());
const outputMode = cold ? 'cold' : smoke ? 'smoke' : 'stress';
const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'ressectionlab-practice-stress-'));
const blockedSavePath = path.join(scratch, 'protected-parent-copy.ressectionlab');
const report = {
  schemaVersion: 1, passed: false, mode: outputMode, startedUtc: new Date().toISOString(),
  pid: process.pid, mainClockOriginNs: startedNs.toString(), actionReceipts: [], phase: 'startup',
  hidden: true, appeared: false, source, sourceSha256, caseHash, checks: {},
  machine: { chip: os.cpus()[0].model, architecture: process.arch, os: os.release(), totalMemory: os.totalmem() },
  build: { main, packaged: app.isPackaged, electron: process.versions.electron, executable: process.execPath },
};
const assetDir = path.join(buildRoot, 'desktop/dist/assets');
report.build.assets = Object.fromEntries(fs.readdirSync(assetDir).filter(name => /\.(js|css)$/.test(name))
  .map(name => [name, hashFile(path.join(assetDir, name))]));
report.build.sourceFiles = Object.fromEntries([
  'desktop/electron/main.cjs', 'desktop/electron/sidecar.cjs', 'src/resectionlab/desktop_bridge.py',
  'src/resectionlab/surgical_bridge.py', 'src/resectionlab/surgical_engine.py', 'src/resectionlab/surgical_deformation_v2.py',
].map(name => [name, hashFile(path.join(buildRoot, name))]));
report.build.verifierSha256 = hashFile(__filename);
argumentsForApp();

function argumentsForApp() {
  fs.mkdirSync(output, { recursive: true });
  app.setPath('userData', path.join(scratch, 'profile'));
  if (process.platform === 'darwin') app.setActivationPolicy('accessory');
  process.argv.push('--case', cold ? savePath : source, '--automation-hidden');
  delete process.env.RESSECTIONLAB_DEV_URL;
  for (const method of ['show', 'showInactive', 'focus']) {
    BrowserWindow.prototype[method] = function () {
      report.appeared = true;
      throw Error('Hidden stress verifier refused BrowserWindow.' + method);
    };
  }
}

const { Sidecar } = require(path.join(path.dirname(main), 'sidecar.cjs'));
const sidecars = new Set();
const timings = [];
const events = [];
const sidecarRequest = Sidecar.prototype.request;
const sidecarEmit = Sidecar.prototype.emit;
let maxSidecarPending = 0;
Sidecar.prototype.request = function (op, args, ...rest) {
  sidecars.add(this);
  const promise = sidecarRequest.call(this, op, args, ...rest);
  maxSidecarPending = Math.max(maxSidecarPending, this.pending.size);
  return promise;
};
Sidecar.prototype.emit = function (type, ...args) {
  const payload = args[0];
  if (type === 'rehearsalTiming') timings.push({ at: now(), ...payload });
  if (type === 'event' && payload.op === 'surgicalRehearsal') {
    events.push({ at: now(), id: payload.id, event: payload.event, op: payload.op });
  }
  if (type === 'event' && payload.op === 'ping' && payload.event === 'result') {
    report.build.engine = payload.result.runtime;
    report.build.native = payload.result.numericalBackend;
    const binary = payload.result.numericalBackend?.binaryPath;
    if (binary && fs.existsSync(binary)) report.build.nativeBinarySha256 = hashFile(binary);
  }
  return sidecarEmit.call(this, type, ...args);
};

const ipcRows = [];
const steps = [];
const outstandingSteps = new Set();
let maxOutstandingSteps = 0;
let latest = null;
const acceptIpc = ipcMain.handle.bind(ipcMain);
ipcMain.handle = (channel, handler) => acceptIpc(channel, async (...args) => {
  if (!['research:surgicalRehearsal', 'research:cancel', 'research:saveCase'].includes(channel)) return handler(...args);
  const request = args[1];
  const row = {
    channel, phase: report.phase, requestId: request?.requestId ?? (typeof request === 'string' ? request : undefined),
    command: request?.command, expectedStateId: request?.expectedStateId, expectedEpisodeId: request?.expectedEpisodeId,
    source: request?.caseHash, requestedBranchId: request?.branchId, requestedPosition: request?.position,
    at: now(), action: request?.action,
  };
  ipcRows.push(row);
  if (row.command === 'step') {
    steps.push(row);
    outstandingSteps.add(row);
    maxOutstandingSteps = Math.max(maxOutstandingSteps, outstandingSteps.size);
  }
  try {
    const result = await handler(...args);
    row.received = now();
    row.cancelled = result?.cancelled;
    row.saved = result?.saved;
    if (result?.view && !result.playback) {
      latest = result;
      row.response = identity(result);
      row.status = result.outcome?.status;
      row.reasonCode = result.outcome?.reasonCode;
      if (row.command === 'step') row.raw = result;
    }
    return result;
  } catch (error) {
    row.received = now();
    row.error = String(error);
    throw error;
  } finally {
    outstandingSteps.delete(row);
  }
});

function identity(result) {
  return {
    caseHash: result.caseHash, sceneId: result.view.scene.sceneId, stateId: result.view.stateId,
    branchId: result.branchId, position: result.position, readOnly: result.readOnly === true,
  };
}

function history(result) {
  return (result.timeline ?? result.view.history).slice(0, result.position).map(row => ({
    stateBefore: row.stateBefore, stateAfter: row.stateAfter, status: row.status, action: row.action,
  }));
}

let picker = null;
const dialogs = [];
dialog.showSaveDialog = async (_window, options) => {
  assert(!cold && picker, 'Unexpected save dialog');
  const mode = picker;
  picker = null;
  dialogs.push({ mode, title: options.title, at: now() });
  if (mode === 'cancel') return { canceled: true };
  return { canceled: false, filePath: mode === 'protected' ? blockedSavePath : savePath };
};
dialog.showMessageBox = async (_window, options) => {
  dialogs.push({ title: options.title, at: now() });
  assert.equal(options.title, 'Unsaved rehearsal');
  // Close only this run's disposable profile, including on a failed check.
  return { response: 1 };
};

function installTrace() {
  const trace = window.practiceStress = { inputs: [], frames: [], fences: [], publications: [], phase: 'startup' };
  trace.control = name => [...document.querySelectorAll('button,summary,input,select')].find(element =>
    !element.closest('[hidden]') &&
    (element.getAttribute('aria-label') === name ||
      element.matches('button,summary') &&
      (element.textContent.trim() === name || element.querySelector('strong')?.textContent.trim() === name)));
  const identity = data => ({
    serial: data.serial ?? data.frameSerial, source: data.sourceCaseHash, stateId: data.surgicalStateId,
    sceneId: data.sceneId, displayIdentity: data.displayIdentity, modelGeneration: data.modelGeneration,
    sourceSurfaceGeneration: data.sourceSurfaceGeneration, preview: data.preview,
  });
  document.addEventListener('anatomy-frame', ({ detail }) => trace.frames.push({
    ...identity(detail), at: detail.at, submittedAt: detail.submittedAt, camera: detail.cameraState,
    shape: detail.sourceShape, interactive: detail.interactive, draw: detail.drawStatistics,
    resources: detail.resources, allocation: detail.estimatedAllocation, cache: detail.surfaceCache,
    workers: detail.workers, cpuMs: detail.cpuMs, programs: detail.renderStages?.programCountAfter,
    displayWork: detail.displayWork, fieldWork: detail.fieldWork,
  }));
  document.addEventListener('anatomy-gpu', ({ detail }) => trace.fences.push({
    ...identity(detail), at: detail.completionAt, completed: detail.completed,
  }));
  document.addEventListener('anatomy-surgical-display', ({ detail }) => trace.publications.push({
    ...identity(detail), at: performance.now(),
  }));
  for (const type of ['click', 'keydown', 'keyup', 'pointerdown', 'pointerup', 'pointermove', 'wheel']) {
    document.addEventListener(type, event => {
      if (!event.isTrusted) return;
      if (type === 'pointermove' && !event.buttons) return;
      const target = event.target.closest?.('button,summary,input,select');
      const pane = event.target.closest?.('[data-accepted-tools]');
      const practice = document.querySelector('.practice-controls');
      trace.inputs.push({
        seq: trace.inputs.length + 1, phase: trace.phase, type, at: performance.now(), eventAt: event.timeStamp,
        label: target?.getAttribute('aria-label') ?? target?.textContent.trim(), key: event.key, repeat: event.repeat,
        disabled: target?.matches(':disabled'), pane: !!pane, x: event.clientX, y: event.clientY, buttons: event.buttons,
        position: Number(document.querySelector('.surgical-panel')?.dataset.recordedPosition),
        branchId: practice?.dataset.practiceEpisode, stateId: practice?.dataset.practiceStateId,
      });
    }, true);
  }
}

function matchFrame(trace, source, sceneId, stateId, after) {
  const publication = trace.publications.findLast(row => row.at >= after && row.stateId === stateId &&
    row.source === source && row.sceneId === sceneId && !row.preview);
  if (!publication) return null;
  const same = row => ['serial', 'source', 'stateId', 'sceneId', 'displayIdentity', 'modelGeneration',
    'sourceSurfaceGeneration', 'preview'].every(key => row[key] === publication[key]);
  const frame = trace.frames.findLast(row => row.at >= after && same(row));
  const fence = trace.fences.findLast(row => row.completed && row.at <= publication.at && same(row));
  return frame && fence && fence.at >= frame.submittedAt ? { frame, fence, publication } : null;
}

function distribution(values) {
  assert(values.every(Number.isFinite), 'Timing sample is absent');
  const sorted = [...values].sort((a, b) => a - b);
  const rank = q => sorted[Math.ceil(q * sorted.length) - 1] ?? null;
  return {
    count: values.length, p50: rank(0.5), p95: rank(0.95), max: sorted.at(-1) ?? null,
    ...(values.length >= 200 ? { p99: rank(0.99) } : {}),
  };
}

function subsetSteps(phase) {
  return steps.filter(row => row.phase === phase);
}

async function wait(check, description, budget = 30000, interval = 12) {
  const deadline = now() + budget;
  do {
    const result = await check();
    if (result) return result;
    await delay(interval);
  } while (now() < deadline);
  throw Error('Timed out waiting for ' + description);
}

function stageProof(requestId, stage) {
  const py = timings.find(row => row.requestId === requestId && row.process === 'python');
  const node = timings.find(row => row.requestId === requestId && row.process === 'electron');
  const bounds = py?.stageBoundsNs?.[stage];
  if (!bounds || !node?.requestSentMainMonotonicNs || !node?.responseLineReadyMainMonotonicNs) return null;
  assert(Number.isFinite(py.responseWriteStartSinceQueueNs));
  // Causality brackets the Python enqueue origin on the Electron monotonic clock.
  const originLow = toMs(node.requestSentMainMonotonicNs);
  const originHigh = toMs(node.responseLineReadyMainMonotonicNs) - py.responseWriteStartSinceQueueNs / 1e6;
  assert(originLow <= originHigh, 'Impossible cross-process clock bracket');
  const interval = (start, end) => ({ start: originHigh + start / 1e6, end: originLow + end / 1e6 });
  return {
    requestId, stage, originLow, originHigh, ticks: py.calls[stage], durationMs: py.durationsNs[stage] / 1e6,
    envelope: interval(bounds.firstStart, bounds.lastEnd), longest: interval(bounds.longestStart, bounds.longestEnd),
    terminalEvent: py.terminalEvent,
  };
}

async function run(win) {
  const wc = win.webContents;
  const q = JSON.stringify;
  const js = code => wc.executeJavaScript(code, true);
  const evaluate = (fn, ...args) => js('(' + fn.toString() + ')(...' + q(args) + ')');
  const until = (code, budget) => wait(() => js(code), code, budget);
  const control = name => 'window.practiceStress.control(' + q(name) + ')';
  const ready = result => until(
    'document.querySelector(".surgical-panel")?.dataset.stateId===' + q(result.view.stateId) +
    '&& Number(document.querySelector(".surgical-panel")?.dataset.recordedPosition)===' + result.position +
    '&& document.querySelector(\'[aria-label="Surgical branch"]\')?.value===' + q(result.branchId) +
    '&& document.querySelector("[data-accepted-tools]")?.dataset.surgicalStateId===' + q(result.view.stateId) +
    '&& document.querySelector(".rl-viewer")?.dataset.surfacesPending==="0" && !document.querySelector(".viewer-loading")' +
    '&& document.querySelector(".practice-controls")?.getAttribute("aria-busy")!=="true"' +
    '&& ![...document.querySelectorAll(".surgical-panel p[role=status]")].some(e=>/Preparing an exact|Computing instrument|Preparing accepted display|Verifying saved/.test(e.textContent))',
    300000);
  const send = (method, args) => wc.debugger.sendCommand('Input.' + method, args);
  const pressMouse = point => send('dispatchMouseEvent', { type: 'mousePressed', button: 'left', buttons: 1, clickCount: 1, ...point });
  const releaseMouse = point => send('dispatchMouseEvent', { type: 'mouseReleased', button: 'left', buttons: 0, clickCount: 1, ...point });
  const mouse = async point => {
    await send('dispatchMouseEvent', { type: 'mouseMoved', ...point });
    await pressMouse(point);
    await releaseMouse(point);
  };
  const key = async (key, code, vk, extra = {}) => {
    const event = { key, code, windowsVirtualKeyCode: vk, nativeVirtualKeyCode: vk, ...extra };
    await send('dispatchKeyEvent', { type: 'keyDown', ...event });
    await send('dispatchKeyEvent', { type: 'keyUp', ...event });
  };
  const locate = async (name, allowDisabled = false) => {
    await until('!!(' + control(name) + ')' + (allowDisabled ? '' : '&&!(' + control(name) + ').matches(":disabled")'));
    while (true) {
      const location = await evaluate(name => {
        const target = window.practiceStress.control(name);
        let disclosure;
        for (let host = target.parentElement; host; host = host.parentElement) {
          if (host.tagName === 'DETAILS' && !host.open && host.querySelector(':scope > summary') !== target) {
            disclosure = host.querySelector(':scope > summary');
          }
        }
        const element = disclosure ?? target;
        element.scrollIntoView({ block: 'center', inline: 'nearest', behavior: 'instant' });
        const rect = element.getBoundingClientRect();
        const x = Math.round(rect.x + rect.width / 2);
        const yOptions = [Math.round(rect.y + rect.height / 2), Math.floor(rect.bottom - 2), Math.ceil(rect.top + 2)];
        const y = yOptions.find(row => element.contains(document.elementFromPoint(x, row)));
        return { x, y, expanded: !!disclosure, disabled: target.matches(':disabled') };
      }, name);
      assert(Number.isFinite(location.y), 'Control is obscured: ' + name);
      const point = { x: location.x, y: location.y };
      if (!location.expanded) return { point, disabled: location.disabled };
      await mouse(point);
    }
  };
  const click = async name => mouse((await locate(name)).point);
  const number = async (name, value) => {
    if (Number(await js('(' + control(name) + ')?.value')) === value) return;
    await click(name);
    await js('(' + control(name) + ').select()');
    await send('insertText', { text: String(value) });
    await key('Tab', 'Tab', 9);
    assert.equal(Number(await js('(' + control(name) + ').value')), value, name);
  };
  const select = async (name, value) => {
    await locate(name);
    const char = await evaluate((name, value) => {
      const element = window.practiceStress.control(name);
      const options = [...element.options];
      const wanted = options.find(option => option.value === value);
      if (!wanted || wanted.selected) return null;
      const initial = wanted.textContent.trim()[0].toLowerCase();
      const next = options.slice(element.selectedIndex + 1).concat(options.slice(0, element.selectedIndex + 1))
        .find(option => !option.disabled && option.textContent.trim().toLowerCase().startsWith(initial));
      if (next !== wanted) throw Error('Native select typeahead is ambiguous: ' + value);
      element.focus({ preventScroll: true });
      return initial;
    }, name, value);
    assert(char, 'Select requires a different enabled option: ' + value);
    await key(char, 'Key' + char.toUpperCase(), char.toUpperCase().charCodeAt(0), { text: char });
  };
  const phase = async value => {
    report.phase = value;
    if (smoke) process.stdout.write('stress smoke: ' + value + '\n');
    await evaluate(value => { window.practiceStress.phase = value; }, value);
  };
  const calibrate = async () => {
    const samples = [];
    for (let i = 0; i < 9; i++) {
      const low = now();
      const at = await js('performance.now()');
      samples.push({ low: low - at, high: now() - at });
    }
    // Chromium rounds non-isolated renderer timestamps to 100 microseconds.
    const timerAllowanceMs = 0.2;
    const low = Math.max(...samples.map(row => row.low)) - timerAllowanceMs;
    const high = Math.min(...samples.map(row => row.high)) + timerAllowanceMs;
    assert(low <= high, 'Main and renderer monotonic bounds do not intersect');
    return { low, high, timerAllowanceMs, samples };
  };
  const trustedInput = (type, label, since) => evaluate((type, label, since) =>
    window.practiceStress.inputs.find(row => row.type === type && row.label === label && row.at >= since), type, label, since);
  const completion = async (result, after) => wait(() => js(
    '(' + matchFrame.toString() + ')(window.practiceStress,' +
    [result.caseHash, result.view.scene.sceneId, result.view.stateId, after].map(q).join(',') + ')'),
  'matching accepted GPU fence at position ' + result.position);
  const visible = async result => {
    await ready(result);
    const actual = await evaluate(() => {
      const pane = document.querySelector('[data-accepted-tools]');
      return {
        tools: JSON.parse(pane.dataset.acceptedTools), camera: JSON.parse(pane.dataset.opticalCamera ?? 'null'),
        view: pane.dataset.operativeView, hud: document.querySelector('[data-operative-readout-state]')?.dataset.operativeReadoutState,
        removed: Number(pane.dataset.removedCells),
      };
    });
    assert.deepEqual(actual.tools, stateWitness(result).tools);
    assert.equal(actual.removed, result.view.removed.length);
    if (actual.view === 'surgeon' && result.practiceEpisode) assert.equal(actual.hud, result.view.stateId);
    return actual;
  };
  const transition = async (command, trigger) => {
    const from = ipcRows.length;
    await trigger();
    const row = await wait(() => ipcRows.slice(from).find(row => row.command === command && row.received !== undefined), command, 360000);
    assert(!row.error && row.response, row.error);
    assert.deepEqual(identity(latest), row.response);
    await ready(latest);
    return latest;
  };
  const recordStep = async (row, before, input, expected, after = input?.at) => {
    assert(row.raw && !row.error, row.error ?? 'Missing instrument result');
    const result = row.raw;
    assert.equal(row.status, expected, 'Unexpected result for ' + row.action.operation);
    assert.equal(row.expectedStateId, before.stateId);
    assert.equal(row.expectedEpisodeId, before.branchId);
    assert.equal(row.source, before.caseHash);
    const draw = expected !== 'rejected' ? await completion(result, after) : null;
    const shown = await visible(result);
    const witness = stateWitness(result);
    checkAction(before, witness, result.outcome, row.action);
    const historyRow = (result.timeline ?? result.view.history)[result.position - 1];
    if (expected !== 'rejected') {
      assert.equal(historyRow.stateBefore, before.stateId);
      assert.equal(historyRow.stateAfter, witness.stateId);
      assert.deepEqual(historyRow.action, row.action);
    }
    const item = {
      requestId: row.requestId, phase: row.phase, action: row.action, before: before.stateId, response: row.response,
      status: row.status, reasonCode: row.reasonCode, input, draw, displacementMm: witness.metrics.brainMaxDisplacementMm,
      ipc: { at: row.at, received: row.received },
      commitLatencyMs: draw && input ? draw.fence.at - input.at : null,
    };
    assert(item.commitLatencyMs === null || item.commitLatencyMs >= 0);
    report.actionReceipts.push(item);
    delete row.raw;
    return { witness, item, shown };
  };
  const beginStep = async (label, mode, trigger) => {
    const from = steps.length;
    const since = await js('performance.now()');
    await trigger();
    const row = await wait(() => steps[from], 'dispatch from ' + label);
    const input = await trustedInput(mode, label, since);
    assert(input && input.branchId === row.expectedEpisodeId && input.stateId === row.expectedStateId &&
      !input.disabled, 'Instrument input lacks a matching trusted DOM event');
    return { from, row, input };
  };
  const perform = async (label, expected = 'accepted', trigger = () => click(label), mode = 'click') => {
    const before = stateWitness(latest);
    const start = await beginStep(label, mode, trigger);
    await wait(() => start.row.received !== undefined, label + ' result');
    assert.equal(steps.length, start.from + 1, 'One admitted input must produce one instrument command');
    return recordStep(start.row, before, start.input, expected);
  };
  const frameMemory = (label, item = null, scheduledAt = null) => {
    const pending = [...sidecars].reduce((count, engine) => count + engine.pending.size, 0);
    const pids = [...sidecars].map(engine => engine.child.pid).filter(Boolean);
    const rss = pids.length ? execFileSync('ps', ['-o', 'pid=,rss=', '-p', pids.join(',')], { encoding: 'utf8' })
      .trim().split('\n').map(row => row.trim().split(/\s+/).map(Number)) : [];
    return {
      label, at: now(), scheduledAt, acceptedCount: report.actionReceipts.length, pending, requestId: item?.requestId,
      hostLoadAvg: os.loadavg(),
      memory: app.getAppMetrics().map(row => ({ pid: row.pid, type: row.type, workingSetKiB: row.memory.workingSetSize }))
        .concat(rss.map(([pid, workingSetKiB]) => ({ pid, workingSetKiB, type: 'Engine' }))),
      cache: item?.draw.frame.cache, resources: item?.draw.frame.resources,
      allocation: item?.draw.frame.allocation, workers: item?.draw.frame.workers, programs: item?.draw.frame.programs,
      displayWork: item?.draw.frame.displayWork, fieldWork: item?.draw.frame.fieldWork,
    };
  };

  assert(!win.isVisible());
  wc.setBackgroundThrottling(false);
  wc.debugger.attach('1.3');
  await wc.debugger.sendCommand('Emulation.setFocusEmulationEnabled', { enabled: true });
  win.setContentSize(1440, 900);
  await until('innerWidth===1440 && innerHeight===900');
  await evaluate(installTrace);
  report.viewport = await js('({ width:innerWidth,height:innerHeight,pixelRatio:devicePixelRatio })');
  report.clock = { start: await calibrate() };
  report.gpu = app.getGPUFeatureStatus();
  assert(report.gpu.webgl === 'enabled' || report.gpu.webgl2 === 'enabled', 'No WebGL GPU');
  await until('!!document.querySelector(".rl-viewer canvas") && !document.querySelector(".viewer-loading")', 300000);
  await click('Rehearse');
  await until('!!document.querySelector(\'[aria-label="Surgical timeline"]\')', 300000);
  const verify = await evaluate(() => [...document.querySelectorAll('button')].find(button =>
    ['Verify saved replay', 'Verify replay for planning'].includes(button.textContent.trim()) && !button.disabled)?.textContent.trim());
  if (verify) await click(verify);
  await wait(() => latest?.caseHash === caseHash, 'the generated tissue source', 300000);
  await ready(latest);

  if (cold) {
    await phase('fresh-reopen');
    const reopened = latest;
    assert.notEqual(process.pid, previous.pid);
    assert([...sidecars].every(engine => !previous.sidecar.pids.includes(engine.child.pid)));
    assert.deepEqual(report.build.assets, previous.build.assets);
    assert.deepEqual(report.build.sourceFiles, previous.build.sourceFiles);
    assert.equal(report.build.nativeBinarySha256, previous.build.nativeBinarySha256);
    assert.deepEqual(stateWitness(reopened), previous.save.witness);
    assert.deepEqual(reopened.practiceEpisode, previous.episode);
    assert.deepEqual(reopened.branches, previous.save.branches);
    assert.deepEqual(history(reopened), previous.save.history);
    assert.equal((await visible(reopened)).view, 'surgeon');
    assert.deepEqual(await visible(reopened), previous.save.display);
    report.reopened = { ...identity(reopened), sha256: hashFile(savePath), historyLength: history(reopened).length,
      fence: (await completion(reopened, 0)).fence };
    const prior = await transition('seek', () => click('Previous recorded state'));
    assert.equal(prior.view.stateId, previous.save.history.at(-1).stateBefore);
    assert.equal(prior.readOnly, true);
    const restored = await transition('seek', () => click('Return to practice endpoint'));
    assert.deepEqual(stateWitness(restored), previous.save.witness);
    assert.notEqual(restored.readOnly, true);
    report.checks.independentFreshProcessReopenedExactLog = true;
    report.checks.reopenedReadOnlyBoundaryAndEditableEndpoint = true;
  } else {
    await phase('setup');
    if (await js('!!(' + control('Rehearse/edit stage') + ')')) await click('Rehearse/edit stage');
    if (latest.branchId !== parentBranch) await transition('selectBranch', () => select('Surgical branch', parentBranch));
    if (latest.position !== parentPosition) await transition('seek', () => click('Next recorded state'));
    assert.equal(latest.position, parentPosition);
    assert.equal(latest.branchId, parentBranch);
    const parent = stateWitness(latest);
    const undone = await transition('undo', () => click('Undo'));
    assert.equal(undone.position, parentPosition - 1);
    const redone = await transition('redo', () => click('Redo'));
    assert.deepEqual(stateWitness(redone), parent);
    const copied = await transition('branch', () => click('Branch'));
    assert.notEqual(copied.branchId, parentBranch);
    assert.deepEqual(copied.branches.find(branch => branch.id === copied.branchId).parent,
      { branchId: parentBranch, position: parentPosition, reset: false });
    assert.equal(copied.view.stateId, parent.stateId);
    report.originLifecycle = { undone: identity(undone), redone: identity(redone), forked: identity(copied), branches: copied.branches };
    const original = await transition('selectBranch', () => select('Surgical branch', parentBranch));
    assert.deepEqual(stateWitness(original), parent);
    report.checks.parentUndoRedoBranch = true;
    const practice = await transition('practice', () => click('Practice from this verified state'));
    assert.equal(practice.view.stateId, parent.stateId);
    assert.equal(practice.practiceEpisode.parent.stateId, parent.stateId);
    assert.equal(practice.practiceEpisode.parent.branchId, parentBranch);
    assert.equal(practice.practiceEpisode.parent.position, parentPosition);
    report.parent = parent;
    report.episode = practice.practiceEpisode;
    report.simulation = { physics: practice.view.physics, physicalClock: practice.view.physicalClock, toolMechanicsRevision: practice.view.toolMechanicsRevision };
    if (!await js('!!document.querySelector("[data-operative-view=surgeon][data-optical-camera]")')) {
      await click('Surgeon / Microscope');
    }
    await until('!!document.querySelector("[data-operative-view=surgeon][data-optical-camera]")');
    await click('left hand');

    await number('Action duration', 0.1);
    await perform('Practice instrument', 'accepted', () => select('Practice instrument', 'brain_retractor'), 'keydown');
    for (const [axis, value] of [['R', 12], ['A', 10], ['S', -5]]) await number('Tool position ' + axis, value);
    await number('Tip radius · mm', 1.5);
    await perform('Place tool');
    await number('Action duration', 1);
    await number('Tool position S', 11.8);
    await perform('Hold tissue');
    await number('Action duration', 0.4);
    await number('Tool position R', 12.4);
    const loaded = await perform('Move tool');
    assert(latest.view.holds?.left?.cells.length > 0 && loaded.witness.metrics.brainMaxDisplacementMm > 0,
      'The retractor must hold physically deformed tissue');
    report.checks.loadedTissue = true;

    await phase('camera-and-burst');
    const cameraRect = await evaluate(() => {
      const pane = document.querySelector('[data-accepted-tools]');
      const rect = pane.getBoundingClientRect();
      const x = Math.round(rect.left + rect.width * 0.55);
      const y = Math.round(rect.top + rect.height * 0.6);
      if (!pane.contains(document.elementFromPoint(x, y))) throw Error('Camera pane is obscured');
      return { x, y };
    });
    await number('Action duration', 30);
    const beforeLong = stateWitness(latest);
    const long = await beginStep('Observe interval', 'click', () => click('Observe interval'));
    await wait(() => events.some(event => event.id === long.row.requestId && event.event === 'started'), 'the long loaded action to reach the engine');
    const attempts = [];
    for (let i = 0; i < 12 && long.row.received === undefined; i++) {
      if (i < 6) {
        const target = await locate('Observe interval', true);
        if (!target.disabled || long.row.received !== undefined) break;
        const at = await js('performance.now()');
        await mouse(target.point);
        attempts.push({ at, point: target.point, disabled: target.disabled });
      }
      await send('dispatchMouseEvent', { type: 'mouseMoved', ...cameraRect });
      await pressMouse(cameraRect);
      for (let move = 1; move <= 6; move++) {
        await send('dispatchMouseEvent', {
          type: 'mouseMoved', x: cameraRect.x + (i % 2 ? -1 : 1) * move * 4, y: cameraRect.y + move, buttons: 1,
        });
        await delay(12);
      }
      await releaseMouse({ x: cameraRect.x + (i % 2 ? -1 : 1) * 24, y: cameraRect.y + 6 });
    }
    await wait(() => long.row.received !== undefined, 'long action result');
    assert.equal(subsetSteps('camera-and-burst').length, 1, 'Disabled burst inputs must not dispatch commands');
    assert(attempts.length >= 3, 'Require at least three disabled button attempts while pending');
    const completed = await recordStep(long.row, beforeLong, long.input, 'accepted');
    const cameraTrace = await evaluate(() => {
      const trace = window.practiceStress;
      const inputs = trace.inputs.filter(row => row.phase === 'camera-and-burst');
      const start = inputs[0].at;
      return {
        inputs, frames: trace.frames.filter(row => row.at >= start),
        fences: trace.fences.filter(row => row.at >= start),
      };
    });
    const bounds = await calibrate();
    const physics = stageProof(long.row.requestId, 'deformationTick');
    const beforeTerminal = at => at + bounds.high < long.row.received;
    const inside = at => physics && at + bounds.low > physics.envelope.start && at + bounds.high < physics.envelope.end;
    const cameraInputs = cameraTrace.inputs.filter(row => row.pane && row.type === 'pointermove');
    const changingFrames = cameraTrace.frames.filter((frame, i, all) => i > 0 && beforeTerminal(frame.at) &&
      frame.stateId === beforeLong.stateId && frame.source === beforeLong.caseHash && frame.sceneId === beforeLong.sceneId &&
      JSON.stringify(frame.camera) !== JSON.stringify(all[i - 1].camera));
    const completedDuringPhysics = changingFrames.flatMap(frame => {
      const fence = cameraTrace.fences.find(value => value.serial === frame.serial && value.completed);
      return inside(frame.at) && fence && inside(fence.at) ? [{ frame, fence }] : [];
    });
    const overlappingInputs = cameraInputs.filter(row => inside(row.at));
    assert(cameraInputs.length > 0 && changingFrames.length > 0, 'Camera input did not move the accepted view while the physics request was pending');
    if (!smoke) assert(overlappingInputs.length > 0 && completedDuringPhysics.length >= 2,
      'Require trusted orbit input and at least two matching completed camera frames inside the deformation phase');
    const disabledClickEvents = cameraTrace.inputs.filter(row => row.type === 'click' &&
      row.label === 'Observe interval' && row.seq !== long.input.seq && beforeTerminal(row.at));
    assert.equal(disabledClickEvents.length, 0, 'A disabled instrument button produced an admitted click');
    report.burst = { disabledAttempts: attempts, admitted: long.input, committed: completed.item.response,
      note: 'CDP delivered pointer attempts to disabled controls. The browser suppressed their clicks before command admission.' };
    report.camera = {
      clock: bounds, physics, trustedPointerMoves: cameraInputs.length, overlappingInputs,
      completedDuringPhysics, cadenceMs: distribution(completedDuringPhysics.slice(1)
        .map((entry, i) => entry.fence.at - completedDuringPhysics[i].fence.at)), trace: cameraTrace,
      measurement: 'WebGL fence poll within the first-to-last deformation tick envelope; cadence is GPU completion cadence',
    };
    report.checks.burstAdmissionAndSingleCommit = true;
    report.checks.cameraInputDuringDeformation = (overlappingInputs.length > 0 && completedDuringPhysics.length >= 2) ||
      'excluded: proof unavailable in smoke run';

    await phase('mid-flight-cancellation');
    const cancelBefore = stateWitness(latest);
    const cancelHistory = history(latest);
    const cancelBeforeCamera = await visible(latest);
    const pending = await beginStep('Observe interval', 'click', () => click('Observe interval'));
    const engineStarted = await wait(() => events.find(event => event.id === pending.row.requestId && event.event === 'started'),
      'request started before cancellation');
    // Let real, earlier ticks run. This delay is in the driver, with the live
    // production action continuing throughout it.
    await delay(140);
    assert.equal(pending.row.received, undefined, 'The cancellation action was no longer pending');
    const cancelSince = await js('performance.now()');
    await click('Cancel computation');
    const cancellation = await wait(() => ipcRows.find(row => row.channel === 'research:cancel' &&
      row.requestId === pending.row.requestId && row.received !== undefined), 'targeted cancellation acknowledgement');
    assert.equal(cancellation.cancelled, true);
    await wait(() => pending.row.received !== undefined, 'cancelled instrument terminal event');
    assert.match(pending.row.error, /Operation cancelled/);
    assert.deepEqual(stateWitness(latest), cancelBefore);
    assert.deepEqual(await visible(latest), cancelBeforeCamera);
    const cancelInput = await trustedInput('click', 'Cancel computation', cancelSince);
    assert(cancelInput?.seq > pending.input.seq && !cancelInput.disabled);
    const cancelBounds = await calibrate();
    await wait(() => smoke || timings.find(row => row.requestId === pending.row.requestId && row.process === 'python'),
      'cancelled worker to finish and emit its timings');
    const cancelPhysics = stageProof(pending.row.requestId, 'deformationTick');
    const provedMidFlight = cancelPhysics?.ticks > 0 && cancelPhysics.envelope.start < cancelInput.at + cancelBounds.low &&
      cancelPhysics.terminalEvent === 'cancelled';
    if (!smoke) assert(provedMidFlight, 'The cancelled request did not perform a verified deformation tick before user cancellation');
    // The real Save action synchronizes the backend with an inspect. Cancelling
    // the picker gives an independent post-cancellation receipt without writing.
    picker = 'cancel';
    const afterCancel = await transition('inspect', () => click('Save'));
    await wait(() => picker === null && ipcRows.some(row => row.channel === 'research:saveCase' &&
      row.phase === 'mid-flight-cancellation' && row.received !== undefined), 'cancelled save picker');
    assert(!fs.existsSync(savePath));
    assert.deepEqual(stateWitness(afterCancel), cancelBefore);
    assert.deepEqual(history(afterCancel), cancelHistory);
    assert.equal(events.filter(event => event.id === pending.row.requestId && event.event === 'cancelled').length, 1);
    report.cancellation = { requestId: pending.row.requestId, engineStarted, actionInput: pending.input,
      cancelInput, acknowledgement: cancellation, physics: cancelPhysics, clock: cancelBounds,
      before: cancelBefore.stateId, after: afterCancel.view.stateId, position: afterCancel.position };
    report.checks.cancellationPreservedCheckpointAndLog = true;
    report.checks.cancellationAfterDeformationStarted = provedMidFlight || 'excluded: proof unavailable in smoke run';

    await phase('rejection');
    const rejected = await perform('Practice instrument', 'rejected',
      () => select('Practice instrument', 'frazier_suction'), 'keydown');
    assert.equal(rejected.item.reasonCode, 'WITHDRAW_BEFORE_EXCHANGE');
    assert.equal(rejected.witness.stateId, cancelBefore.stateId);
    assert.deepEqual(history(latest), cancelHistory);
    report.checks.insertedExchangeRejectedWithoutCommit = true;

    await phase('sustained');
    await number('Action duration', 0.2);
    const period = smoke ? 5000 : 180000;
    const cadence = 500;
    const collectEvery = smoke ? 1000 : 15000;
    const sustainedStart = now();
    const sustained = report.sustained = { started: sustainedStart, targetDurationMs: period,
      targetCadenceMs: cadence, samples: [frameMemory('before', completed.item)] };
    let nextSample = sustainedStart + collectEvery;
    let count = 0;
    const sustainItems = [];
    while (now() < sustainedStart + period) {
      const turn = now();
      const x = count % 2 === 0 ? 12 : 12.4;
      await number('Tool position R', x);
      const done = await perform('Move tool');
      assert.equal(done.witness.tools.left.position[0], x);
      assert(latest.view.holds?.left?.cells.length > 0 && done.witness.metrics.brainMaxDisplacementMm > 0);
      sustainItems.push(done.item);
      count += 1;
      if (now() >= nextSample) {
        sustained.samples.push(frameMemory('interval', done.item, nextSample));
        if (!smoke && (nextSample - sustainedStart) % 60000 < 1)
          process.stdout.write('Sustained ' + Math.round((nextSample - sustainedStart) / 1000) + ' seconds: ' + count + ' accepted movements\n');
        nextSample += collectEvery;
      }
      await delay(Math.max(0, Math.min(turn + cadence, sustainedStart + period) - now()));
    }
    sustained.ended = now();
    sustained.samples.push(frameMemory('after', sustainItems.at(-1)));
    sustained.accepted = count;
    sustained.latency = distribution(sustainItems.map(row => row.commitLatencyMs));
    const thirds = [[], [], []];
    for (const item of sustainItems) {
      const index = Math.min(2, Math.floor(3 * (item.ipc.at - sustainedStart) / period));
      thirds[index].push(item);
    }
    sustained.windows = thirds.map((window, index) => ({
      startMs: index * period / 3, endMs: (index + 1) * period / 3,
      latencyMs: distribution(window.map(row => row.commitLatencyMs)),
      ipcMs: distribution(window.map(row => row.ipc.received - row.ipc.at)),
    }));
    if (!smoke) assert(count >= 120 && sustained.samples.length >= 12, 'Insufficient accepted sustained-action and residency samples');
    assert(sustained.samples.every(item => item.pending === 0), 'A quiescent frame retained IPC work');
    assert(sustainItems.every(({ draw }) => draw.frame.cache.retiredBytes <= draw.frame.cache.budgetBytes &&
      draw.frame.cache.retiredEntries <= 16), 'Surface retention exceeded its configured bounds');
    assert.equal(maxOutstandingSteps, 1);
    if (!smoke) {
      await wait(() => sustainItems.every(item => timings.some(row => row.requestId === item.requestId && row.process === 'python')),
        'timings for every sustained action');
      const tickCounts = sustainItems.map(item => timings.find(row => row.requestId === item.requestId && row.process === 'python').calls.deformationTick ?? 0);
      assert(tickCounts.every(count => count > 0), 'A sustained action did not exercise tissue deformation');
      sustained.deformationTicks = distribution(tickCounts);
      for (const [index, window] of sustained.windows.entries()) {
        const observations = thirds[index].map(item => timings.find(row => row.requestId === item.requestId && row.process === 'python'));
        window.workerQueueMs = distribution(observations.map(row => row.durationsNs.queue / 1e6));
        window.workerDispatchMs = distribution(observations.map(row => row.durationsNs.dispatch / 1e6));
        window.responseBytes = distribution(observations.map(row => row.responseBytes));
      }
    }
    report.checks.sustainedActionsAndSurfaceBudget = smoke ? 'smoke only' : true;

    await phase('held-input');
    const heldBefore = stateWitness(latest);
    await click('Hold controls');
    const holdControl = 'Hold to retract';
    const holdPoint = (await locate(holdControl)).point;
    const holdSince = await js('performance.now()');
    await send('dispatchMouseEvent', { type: 'mouseMoved', ...holdPoint });
    await pressMouse(holdPoint);
    const wanted = 4;
    await wait(() => subsetSteps('held-input').filter(row => row.received !== undefined && row.status === 'accepted').length >= wanted,
      'four committed slices from one held input');
    await releaseMouse(holdPoint);
    const down = await trustedInput('pointerdown', holdControl, holdSince);
    const up = await trustedInput('pointerup', holdControl, holdSince);
    assert(down && up && down.seq < up.seq);
    const heldRows = subsetSteps('held-input');
    await wait(() => heldRows.every(row => row.received !== undefined), 'held terminal receipts');
    await ready(latest);
    await delay(400);
    assert.equal(subsetSteps('held-input').length, wanted, 'Releasing the trusted pointer admitted another slice');
    const holdFrames = [];
    let priorHeld = heldBefore;
    let after = down.at;
    // Earlier held cursor states already have their own publication; the current
    // DOM will be at the final slice, so check earlier evidence by matching serial.
    for (const row of heldRows) {
      assert(row.raw && row.status === 'accepted');
      const result = row.raw;
      assert.equal(row.expectedStateId, priorHeld.stateId);
      assert.equal(row.expectedEpisodeId, priorHeld.branchId);
      assert.equal(row.action.operation, 'move');
      assert.equal(row.action.duration, 0.1);
      const next = stateWitness(result);
      checkAction(priorHeld, next, result.outcome, row.action);
      assert(Math.abs(next.tools.left.position[2] - priorHeld.tools.left.position[2] + 0.5) < 1e-6);
      const draw = await completion(result, after);
      const item = { requestId: row.requestId, phase: row.phase, action: row.action, before: priorHeld.stateId,
        status: row.status, response: row.response, input: down, draw, commitLatencyMs: null };
      report.actionReceipts.push(item);
      holdFrames.push(item);
      priorHeld = next;
      after = draw.publication.at;
      delete row.raw;
    }
    const holdBounds = await calibrate();
    assert(heldRows.every(row => row.at < up.at + holdBounds.low));
    assert(holdFrames.slice(1).every((item, index) =>
      holdFrames[index].draw.publication.at + holdBounds.high < heldRows[index + 1].at),
    'A held slice dispatched before the exact previous slice was published');
    report.hold = { down, up, clock: holdBounds, committed: holdFrames.map(row => row.response),
      firstInputToMatchingFenceMs: holdFrames[0].draw.fence.at - down.at,
      fenceCadenceMs: distribution(holdFrames.slice(1).map((item, i) => item.draw.fence.at - holdFrames[i].draw.fence.at)) };
    report.checks.heldSlicesMatchPreviousPublishedStateAndStop = true;

    await phase('lifecycle');
    await number('Action duration', 0.01);
    await perform('Release tool');
    await number('Action duration', 1);
    const withdrawn = await perform('Withdraw tool');
    assert(Object.values(withdrawn.witness.tools).every(tool => !tool.inserted));
    const final = latest;
    const finalHistory = history(final);
    const accepted = report.actionReceipts.filter(row => row.status === 'accepted');
    assert.deepEqual(finalHistory.slice(parentPosition), accepted.map(row => ({
      stateBefore: row.before, stateAfter: row.response.stateId, status: row.status, action: row.action,
    })), 'History must contain every accepted action in order and exclude disabled, rejected and cancelled input');
    const previousBoundary = await transition('seek', () => click('Previous recorded state'));
    assert.equal(previousBoundary.view.stateId, finalHistory.at(-1).stateBefore);
    assert.equal(previousBoundary.readOnly, true);
    assert.equal((await locate('Apply tool', true)).disabled, true);
    assert.equal(await js('(' + control('Undo') + ')===undefined && (' + control('Branch') + ')===undefined'), true);
    await transition('seek', () => click('Return to practice endpoint'));
    assert.deepEqual(stateWitness(latest), withdrawn.witness);
    report.checks.practiceBoundaryIsReadOnlyAndReturnsExactly = true;

    await phase('save');
    fs.copyFileSync(source, blockedSavePath, fs.constants.COPYFILE_EXCL);
    picker = 'protected';
    await click('Save');
    const failedSave = await wait(() => ipcRows.find(row => row.channel === 'research:saveCase' &&
      row.phase === 'save' && row.received !== undefined), 'protected-save failure');
    assert.match(failedSave.error, /PRACTICE_SAVE_AS/);
    await until('document.querySelector(".error-notice")?.textContent.includes("Choose a new filename")');
    assert.equal(hashFile(blockedSavePath), sourceSha256);
    assert.deepEqual(stateWitness(latest), withdrawn.witness);
    assert.deepEqual(history(latest), finalHistory);
    report.checks.protectedSaveRefusedWithoutOverwrite = true;

    picker = 'new';
    await click('Save');
    await until('document.body.innerText.includes("Workspace saved locally") && !document.querySelector(".viewer-loading")');
    assert(fs.statSync(savePath).size > 0);
    const saved = JSON.parse(execFileSync('python3', ['-c',
      'import json,sys,zipfile; a=json.loads(zipfile.ZipFile(sys.argv[1]).read("manifest.json"))["artifacts"]["surgical_toolbox"]; b=a["branchId"]; s=a["branches"][b]; print(json.dumps(dict(episode=a["practiceEpisodes"][b], position=a["position"], stateId=s["checkpoint"]["stateId"], actions=s["actions"])))',
      savePath], { encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 }));
    assert.deepEqual(saved.episode, report.episode);
    assert.equal(saved.position, latest.position);
    assert.equal(saved.stateId, latest.view.stateId);
    assert.deepEqual(saved.actions, finalHistory.map(row => row.action));
    assert.equal(hashFile(source), sourceSha256);
    report.save = { sha256: hashFile(savePath), file: savePath, witness: stateWitness(latest),
      display: await visible(latest), history: finalHistory, branches: latest.branches };
    report.checks.saveContainsExactCommittedLog = true;
  }
  report.clock.end = await calibrate();
  for (const [name, digest] of Object.entries(report.build.assets)) assert.equal(hashFile(path.join(assetDir, name)), digest);
  for (const [name, digest] of Object.entries(report.build.sourceFiles)) assert.equal(hashFile(path.join(buildRoot, name)), digest);
  if (report.build.nativeBinarySha256) assert.equal(hashFile(report.build.native.binaryPath), report.build.nativeBinarySha256);
  assert(report.clock.start.low <= report.clock.end.high && report.clock.end.low <= report.clock.start.high,
    'Renderer clock drifted out of its calibrated bounds');
}

app.on('browser-window-created', (_event, win) => {
  win.on('show', () => { report.appeared = true; win.hide(); });
  win.webContents.once('did-finish-load', async () => {
    try {
      await run(win);
      assert(!report.appeared && !win.isVisible());
      report.passed = true;
    } catch (error) {
      report.error = error.stack ?? String(error);
      report.latest = latest && identity(latest);
      report.rendererFailure = await win.webContents.executeJavaScript(
        '({ status: document.querySelector(".surgical-panel")?.innerText.slice(0,2200), alert: document.querySelector(".error-notice")?.innerText })',
        true).catch(() => null);
      process.stderr.write(report.error + '\n');
    } finally {
      report.finishedUtc = new Date().toISOString();
      report.maxOutstandingSteps = maxOutstandingSteps;
      report.maxSidecarPending = maxSidecarPending;
      report.ipc = ipcRows.map(({ raw, ...row }) => row);
      report.sidecar = { timings, events, pids: [...sidecars].map(engine => engine.child.pid) };
      report.dialogs = dialogs;
      fs.writeFileSync(receipt, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
      process.stdout.write(JSON.stringify({ passed: report.passed, receipt, checks: report.checks }) + '\n');
      await Promise.allSettled([...sidecars].map(engine => engine.stop()));
      fs.rmSync(scratch, { recursive: true, force: true });
      app.exit(report.passed ? 0 : 1);
    }
  });
});
require(main);
