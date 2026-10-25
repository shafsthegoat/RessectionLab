'use strict';
const test = require('node:test'), assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const { VascularPipelineProof } = require('./vascular-pipeline-proof.cjs');
const { MAX_RSS } = require('./prefix-resources.cjs');
const tick = async () => { for (let i = 0; i < 16; i++) await Promise.resolve(); };
const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; };
const D = 'sha256:' + 'a'.repeat(64), SHA = 'b'.repeat(64), MB = 1024 ** 2;
const BINDING = { schema: 'resectionlab.prepared-prefix-binding.v1', sceneId: D,
  originSnapshotHash: D, actionLogHash: D, protectionContractHash: D,
  implementation: 'c'.repeat(64), prefixImplementation: 'd'.repeat(64) };
const start = { command: 'seekPrefix', caseHash: D, branchId: 'branch', position: 0 };
const frame = (position, frameIndex) => ({ ...start, command: 'seekPrefixFrame', position, frameIndex });

function result(role, index = null, frameIndex = 0, counts = [2, 1]) {
  const verified = index === null ? 0 : index + 1, producer = role === 'producer';
  const cache = { mode: role === 'consumer' ? 'only' : 'prepare', loadedActions: role === 'consumer' ? verified : 0,
    publishedActions: producer ? verified : 0, misses: 0, lastSkip: null };
  const value = { caseHash: D, branchId: 'branch', position: index ?? 0,
    readOnly: true, changed: false, outcome: null, fullCase: null, budgetStopped: false,
    view: { scene: { sceneId: D }, stateId: D, fixture: role },
    verification: { ...BINDING, status: 'reconstructed-prefix', readOnly: true, verifiedThrough: verified,
      totalActions: counts.length, failure: null, preparedCache: { ...cache } } };
  if (role !== 'normal') value.prefixWorker = { schema: 'resectionlab.prefix-worker.v1', role, binding: { ...BINDING },
    sceneId: D, sourceArchiveSha256: SHA, verifiedThrough: verified, totalActions: counts.length, cache,
    maxStagedActionBytes: 64 * MB, actionFrameBytes: index === null ? 0 : 4096,
    store: { directory: '/cache', namespaceDirectory: '/cache/prepared-prefix-v1', maxDiskBytes: 512 * MB,
      maxActionBytes: 64 * MB, diskBytes: 8192, hits: role === 'consumer' ? verified : 0,
      publications: producer ? verified : 0, misses: 0, lastSkip: null } };
  if (index !== null) value.playback = { branchId: 'branch', position: index, actionIndex: index, frameIndex,
    frameCount: counts[index], readOnly: true, frameStateId: D };
  return value;
}

function harness(counts = [2, 1]) {
  const h = { clock: 1, failures: [], workers: [], kills: [], counts, serial: 0, customSample: null };
  h.sample = () => h.customSample ? h.customSample() : Promise.resolve([
    `${process.pid} 1 1`, ...h.workers.filter(worker => !worker.exited).map(worker => `${worker.child.pid} ${process.pid} 100`)
  ].join('\n'));
  h.proof = new VascularPipelineProof({ archivePath: '/original.saved', sourceArchiveSha256: SHA, branchId: 'branch',
    frameCounts: counts, cacheDirectory: '/cache', onFailure: error => h.failures.push(error), now: () => h.clock,
    sample: h.sample, _killPid: (pid, signal) => h.kills.push([pid, signal]) });
  h.worker = role => {
    const child = new EventEmitter(); child.pid = 20000 + h.workers.length;
    const worker = { child, stopping: false, exited: false };
    child.kill = signal => { h.kills.push([child.pid, signal]); worker.exited = true; child.emit('exit', null, signal); };
    worker.exit = (code = 0, signal = null, expected = true) => {
      worker.stopping = expected; worker.exited = true; child.emit('exit', code, signal);
    };
    h.workers.push(worker);
    h.proof.register(worker, role === 'normal' ? {} : { prefixWorker: { role, sourcePath: '/original.saved', sourceSha256: SHA, branchId: 'branch' } });
    return worker;
  };
  h.begin = (worker, op, args) => {
    const pending = deferred(), id = 'request-' + ++h.serial, cancel = () => { h.cancelled = id; };
    const lease = { id, promise: pending.promise, cancel }, began = h.clock++;
    const observed = h.proof.observeLease(worker, op, args, lease, began);
    return { ...pending, observed, lease, args };
  };
  h.finish = async (pending, value) => { h.clock++; pending.resolve(value); const actual = await pending.observed.promise; assert.equal(actual, value); return actual; };
  return h;
}

async function bootstrap(h) {
  const normal = h.worker('normal'); await h.proof.startResources();
  await h.finish(h.begin(normal, 'loadCase', { path: '/original.saved' }), { caseHash: D });
  await h.finish(h.begin(normal, 'surgicalRehearsal', { ...start }), result('normal', null, 0, h.counts));
  normal.exit();
  const consumer = h.worker('consumer'), producer = h.worker('producer');
  await h.proof.checkResources();
  for (const [role, worker] of [['consumer', consumer], ['producer', producer]])
    await h.finish(h.begin(worker, 'loadCase', { path: '/original.saved' }),
      { caseHash: D, prefixWorker: { schema: 'resectionlab.prefix-worker.v1', role, sourceArchiveSha256: SHA } });
  await h.finish(h.begin(producer, 'surgicalRehearsal', { ...start }), result('producer', null, 0, h.counts));
  await h.finish(h.begin(producer, 'surgicalRehearsal', frame(0, 0)), result('producer', 0, 0, h.counts));
  await h.finish(h.begin(consumer, 'surgicalRehearsal', { ...start, cacheOnly: true }), result('consumer', null, 0, h.counts));
  return { normal, consumer, producer };
}

function facade(counts) {
  return { enabled: true, mode: 'closed', activeWorkers: 0, workerCreations: 3, maxWorkers: 2,
    ordinaryInflight: 0, pendingTransitions: 0, failure: null, exitFailure: null,
    pipeline: { status: 'closed', preparedActions: counts.length, loadedActions: counts.length,
      deliveredFrames: counts.reduce((a, b) => a + b), repeatedFrames: 0, fatalCount: 0, pending: 0,
      maxPendingRequests: 2, maxPending: 2, maxLookahead: 1,
      maxPendingWindowBytes: 128 * MB, maxReservedActionBytes: 128 * MB } };
}
function report(h) {
  const ipc = [], receipts = [], shown = [];
  let serial = 0;
  for (let i = 0; i < h.counts.length; i++) for (let f = 0; f < h.counts[i]; f++) {
    const row = { command: 'seekPrefixFrame', branchId: 'branch', position: i, frameIndex: f, stateId: D };
    serial++; ipc.push({ ...row }); receipts.push({ ...row, notificationSerial: serial, receivedAt: 100 + serial });
    shown.push({ ...row, receiptNotificationSerial: serial, rendererReceivedAt: 100 + serial, firstSubmittedAt: 100 + serial + 0.5 });
  }
  return { qualificationMode: 'fresh-pipeline', playStarted: 1, playEnded: h.clock,
    playStartedRenderer: 100, playEndedRenderer: 101 + serial,
    allCompletedPlaybackIpc: ipc, rendererReceipts: receipts, submittedPlayback: shown };
}

async function completed() {
  const h = harness(), workers = await bootstrap(h);
  const { producer, consumer } = workers;
  await h.finish(h.begin(consumer, 'surgicalRehearsal', frame(0, 0)), result('consumer', 0));
  const next = h.begin(producer, 'surgicalRehearsal', frame(1, 0));
  await h.finish(h.begin(consumer, 'surgicalRehearsal', frame(0, 1)), result('consumer', 0, 1));
  await h.finish(next, result('producer', 1));
  await h.finish(h.begin(consumer, 'surgicalRehearsal', frame(1, 0)), result('consumer', 1));
  producer.exit(); consumer.exit(); await h.proof.closeResources(); return h;
}

test('complete bounded proof preserves result/args/cancel identity and accepted Python schema', async () => {
  const h = await completed(), proof = h.proof.finalize(report(h), facade(h.counts));
  assert.equal(proof.mainPid, process.pid); assert.equal(proof.workers.length, 3); assert.equal(proof.resources.maxAlive, 2);
  assert.equal(proof.leases.length, 4); assert.equal(proof.resources.closed, true); assert.equal(proof.leases[0].actionFrameBytes, 4096);
  assert.equal(proof.rendererTupleSha256.length, 64); assert.ok(Buffer.byteLength(JSON.stringify(proof, null, 2) + '\n') <= 512 * 1024);
  assert.equal(h.proof.statistics.maxLookahead, 1); assert.equal(h.proof.statistics.unresolved, 0);
  assert.equal(JSON.stringify(proof).includes('fixture'), false); assert.equal(JSON.stringify(proof).includes('view'), false);
  assert.equal(h.failures.length, 0);
});

test('lease observer retains original ID and cancel function and no full result graph', async () => {
  const h = harness(), normal = h.worker('normal');
  const args = { path: '/original.saved' }, pending = h.begin(normal, 'loadCase', args);
  assert.equal(pending.args, args); assert.equal(pending.observed.id, pending.lease.id);
  assert.equal(pending.observed.cancel, pending.lease.cancel); pending.observed.cancel(); assert.equal(h.cancelled, pending.lease.id);
  const value = { caseHash: D, largeFixture: { untouched: true } }; assert.equal(await h.finish(pending, value), value);
  await h.proof.killAll();
});

test('producer cannot begin next action before consumer first load finishes', async () => {
  const h = harness(), { producer, consumer } = await bootstrap(h);
  const load = h.begin(consumer, 'surgicalRehearsal', frame(0, 0));
  const bad = h.begin(producer, 'surgicalRehearsal', frame(1, 0));
  await assert.rejects(bad.observed.promise, /PRODUCER_LOOKAHEAD/);
  load.resolve(result('consumer', 0)); await assert.rejects(load.observed.promise, /PRODUCER_LOOKAHEAD/); await tick();
  assert.equal(h.failures.length, 1);
});

test('consumer cannot load while producer preparation is pending, nor repeat a frame', async () => {
  const h = harness(), { consumer } = await bootstrap(h);
  await h.finish(h.begin(consumer, 'surgicalRehearsal', frame(0, 0)), result('consumer', 0));
  const repeat = h.begin(consumer, 'surgicalRehearsal', frame(0, 0));
  await assert.rejects(repeat.observed.promise, /CONSUMER_SEQUENCE/); await tick(); assert.equal(h.failures.length, 1);
});

for (const mutate of [r => { r.prefixWorker.binding.actionLogHash = '0'.repeat(64); },
  r => { r.prefixWorker.store.publications = 0; }, r => { r.prefixWorker.cache.loadedActions = 1; },
  r => { r.prefixWorker.sourceArchiveSha256 = '0'.repeat(64); }, r => { r.prefixWorker.actionFrameBytes = 64 * MB + 1; }]) {
  test('wrong raw role binding/source/actual-new-write/counters/staging bytes fail before any proof', async () => {
    const h = harness(), { producer, consumer } = await bootstrap(h);
    await h.finish(h.begin(consumer, 'surgicalRehearsal', frame(0, 0)), result('consumer', 0));
    const pending = h.begin(producer, 'surgicalRehearsal', frame(1, 0)), value = result('producer', 1); mutate(value);
    pending.resolve(value); await assert.rejects(pending.observed.promise, /VASCULAR_PIPELINE_PROOF/); await tick();
    assert.equal(h.failures.length, 1);
  });
}

test('ordinary actual exit and exact role configuration are required; duplicate role is refused', async () => {
  let h = harness(); h.worker('normal'); assert.throws(() => h.worker('consumer'), /OLD_EXIT_REQUIRED/); await tick();
  h = harness(); await bootstrap(h); assert.throws(() => h.worker('producer'), /DUPLICATE_WORKER/); await tick();
});

test('exit failure or unexpected clean exit is terminal, even when membership leaves a pending ps sample', async () => {
  const h = harness(), normal = h.worker('normal'); const pending = deferred(); h.customSample = () => pending.promise;
  const checking = h.proof.checkResources(); normal.exit(0, null, false);
  pending.resolve(`${process.pid} 1 1`); await assert.rejects(checking, /UNEXPECTED_WORKER_EXIT/); await tick();
  assert.equal(h.failures.length, 1);
});

test('normal departure/new pair during pending sample ignores old PID reuse and samples new pair next', async () => {
  const h = harness(), normal = h.worker('normal'); const sample = deferred();
  await h.finish(h.begin(normal, 'loadCase', { path: '/original.saved' }), { caseHash: D });
  await h.finish(h.begin(normal, 'surgicalRehearsal', { ...start }), result('normal'));
  h.customSample = () => sample.promise; const checking = h.proof.checkResources(); normal.exit();
  const consumer = h.worker('consumer'), producer = h.worker('producer');
  sample.resolve(`${process.pid} 1 1\n${normal.child.pid} 999 999999999`); await checking;
  h.customSample = null; await h.proof.checkResources();
  assert.equal(h.proof.statistics.peakAggregatePythonTreeRssBytes, 200 * 1024);
  consumer.exit(); producer.exit(); await h.proof.killAll();
});

test('aggregate descendants and common constructor wall limit fail once and kill owned children only', async () => {
  const h = harness(), normal = h.worker('normal');
  h.customSample = async () => `${process.pid} 1 1\n${normal.child.pid} ${process.pid} 1\n30000 ${normal.child.pid} ${Math.ceil(MAX_RSS / 1024)}\n30001 999 99999999`;
  await assert.rejects(h.proof.checkResources(), /AGGREGATE_3_GIB_LIMIT/); await h.proof.killAll();
  assert.ok(h.kills.some(([pid]) => pid === 30000)); assert.ok(!h.kills.some(([pid]) => pid === 30001 || pid === process.pid));
  assert.equal(h.failures.length, 1);
  const wall = harness(); wall.worker('normal'); wall.clock = 600001;
  await assert.rejects(wall.proof.checkResources(), /COMMON_WALL_LIMIT/); await wall.proof.killAll();
});

test('early rejected/no-ID malformed lease is observed without unhandled rejection, and late success cannot revive failure', async () => {
  const h = harness(), normal = h.worker('normal');
  const early = Promise.reject(new Error('early original refusal'));
  const observed = h.proof.observeLease(normal, 'ping', {}, { id: null, promise: early }, h.clock);
  await assert.rejects(observed.promise, /LEASE_API/); await tick(); assert.equal(h.failures.length, 1);
  const h2 = harness(), worker = h2.worker('normal'), pending = h2.begin(worker, 'loadCase', { path: '/original.saved' });
  worker.exit(1); pending.resolve({ caseHash: D }); await assert.rejects(pending.observed.promise, /UNEXPECTED_WORKER_EXIT/); await tick();
});

test('finalize requires both exits, closed resources, all tuples, full click timing and exact closed facade counters', async () => {
  const h = harness(); await bootstrap(h);
  await assert.rejects(h.proof.closeResources(), /EXIT_BARRIER_REQUIRED/); await tick();
  for (const change of ['timing', 'tuple', 'receipt', 'facade']) {
    const current = await completed(), r = report(current), f = facade(current.counts);
    if (change === 'timing') r.playStarted = r.playEnded - 1;
    if (change === 'tuple') r.allCompletedPlaybackIpc[1].stateId = 'sha256:' + 'f'.repeat(64);
    if (change === 'receipt') r.rendererReceipts[1].receivedAt = 200;
    if (change === 'facade') f.pipeline.deliveredFrames--;
    assert.throws(() => current.proof.finalize(r, f), /VASCULAR_PIPELINE_PROOF/); await tick();
  }
});
