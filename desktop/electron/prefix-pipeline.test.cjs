'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { PrefixPipeline } = require('./prefix-pipeline.cjs');
const { Sidecar } = require('./sidecar.cjs');

const MB = 1024 ** 2, digest = 'sha256:' + 'a'.repeat(64);
const expectedBinding = { schema: 'resectionlab.prepared-prefix-binding.v1', sceneId: digest,
  originSnapshotHash: digest, actionLogHash: digest, protectionContractHash: digest,
  implementation: 'b'.repeat(64), prefixImplementation: 'c'.repeat(64) };
const tick = async () => { for (let i = 0; i < 12; i++) await Promise.resolve(); };

class Worker {
  constructor(role) { this.role = role; this.calls = []; this.stops = 0; }
  requestLease(op, args, timeout) {
    let resolve, reject;
    const promise = new Promise((a, b) => { resolve = a; reject = b; });
    const call = { op, args, timeout, resolve, reject, cancels: 0 };
    this.calls.push(call);
    // Deliberately let cancellation complete late, to exercise epoch ownership
    // rather than relying on the fake worker to suppress stale responses.
    return { id: this.role + this.calls.length, promise, cancel() { call.cancels++; } };
  }
  stop() { this.stops++; return Promise.resolve(); }
}

function response(role, position = null, frameIndex = 0, frameCount = 2, total = 3) {
  const verified = position === null ? 0 : position + 1;
  const loaded = role === 'consumer' ? verified : 0, published = role === 'producer' ? verified : 0;
  const cache = { mode: role === 'producer' ? 'prepare' : 'only', loadedActions: loaded,
    publishedActions: published, misses: 0, lastSkip: null };
  const result = { caseHash: 'case', branchId: 'branch', readOnly: true, budgetStopped: false,
    changed: false, outcome: null, position: position ?? 0,
    view: { stateId: digest, marker: role }, timelineToken: role + '-token',
    verification: { ...expectedBinding, readOnly: true, status: 'reconstructed-prefix',
      failure: null, verifiedThrough: verified, totalActions: total, preparedCache: { ...cache } },
    prefixWorker: { schema: 'resectionlab.prefix-worker.v1', role, binding: { ...expectedBinding },
      sceneId: digest, sourceArchiveSha256: 'd'.repeat(64), verifiedThrough: verified, totalActions: total,
      maxStagedActionBytes: 64 * MB, actionFrameBytes: position === null ? 0 : 4096, cache,
      store: { directory: '/shared', namespaceDirectory: '/shared/prepared-prefix-v1',
        maxDiskBytes: 512 * MB, maxActionBytes: 64 * MB, hits: loaded, misses: 0,
        publications: published, lastSkip: null, diskBytes: published * 8192 } } };
  if (position !== null) result.playback = { branchId: 'branch', readOnly: true, position,
    actionIndex: position, frameIndex, frameCount, frameStateId: digest };
  return result;
}

function harness() {
  const producer = new Worker('producer'), consumer = new Worker('consumer'), fatal = [];
  const pipeline = new PrefixPipeline({ producer, consumer, caseHash: 'case', branchId: 'branch',
    expectedBinding, onFatal: error => fatal.push(error) });
  return { producer, consumer, pipeline, fatal };
}
const args = (position, frameIndex, extra = {}) => ({ caseHash: 'case', branchId: 'branch',
  command: 'seekPrefixFrame', position, frameIndex, ...extra });

async function started(h, total = 3) {
  const starting = h.pipeline.start();
  assert.deepEqual(h.producer.calls[0].args, { caseHash: 'case', branchId: 'branch', command: 'seekPrefix', position: 0 });
  assert.equal(h.consumer.calls.length, 0);
  h.producer.calls[0].resolve(response('producer', null, 0, 2, total)); await tick();
  assert.equal(h.producer.calls.length, 2); assert.equal(h.consumer.calls.length, 0);
  assert.equal(h.producer.calls[1].args.position, 0);
  h.producer.calls[1].resolve(response('producer', 0, 0, 2, total)); await tick();
  assert.deepEqual(h.consumer.calls[0].args, { caseHash: 'case', branchId: 'branch', command: 'seekPrefix', position: 0, cacheOnly: true });
  h.consumer.calls[0].resolve(response('consumer', null, 0, 2, total));
  const initial = await starting;
  assert.equal(initial.prefixWorker, undefined); assert.equal(initial.view.marker, 'consumer');
  assert.equal(h.producer.calls.length, 2); // action1 awaits consumer load0
}

async function loadZero(h, total = 3) {
  const first = h.pipeline.seekFrame(args(0, 0));
  assert.equal(h.producer.calls.length, 2);
  h.consumer.calls[1].resolve(response('consumer', 0, 0, 2, total));
  const result = await first; await tick();
  assert.equal(result.view.marker, 'consumer'); assert.equal(result.prefixWorker, undefined);
  return result;
}

test('publication precedes load; load acknowledgment precedes next preparation; warm reads overlap only the next producer', async () => {
  const h = harness(); await started(h); await loadZero(h);
  assert.equal(h.producer.calls.length, 3); assert.equal(h.producer.calls[2].args.position, 1);
  const warm = h.pipeline.seekFrame(args(0, 1));
  h.consumer.calls[2].resolve(response('consumer', 0, 1)); await warm;
  const next = h.pipeline.seekFrame(args(1, 0)); await tick();
  assert.equal(h.consumer.calls.length, 3); // producer1 holds STORE, no load1 yet
  h.producer.calls[2].resolve(response('producer', 1)); await tick();
  assert.equal(h.consumer.calls.length, 4); assert.equal(h.producer.calls.length, 3);
  h.consumer.calls[3].resolve(response('consumer', 1)); await next; await tick();
  assert.equal(h.producer.calls.length, 4); assert.equal(h.producer.calls[3].args.position, 2);
  assert.equal(h.pipeline.statistics.maxLookahead, 1);
  assert.equal(h.pipeline.statistics.maxReservedActionBytes, 128 * MB);
  assert.equal(h.pipeline.statistics.loadedActions, 2);
  await h.pipeline.close(); h.producer.calls[3].resolve(response('producer', 2)); await tick();
  assert.deepEqual(h.fatal, []);
});

test('FIFO admits app plus one prefetch, rejects third without issuing work, and retains only bounded aggregate statistics', async () => {
  const h = harness(); await started(h);
  const first = h.pipeline.seekFrame(args(0, 0)), second = h.pipeline.seekFrame(args(0, 1));
  await assert.rejects(h.pipeline.seekFrame(args(1, 0)), /QUEUE_FULL/);
  assert.equal(h.consumer.calls.length, 2);
  h.consumer.calls[1].resolve(response('consumer', 0)); await first; await tick();
  assert.equal(h.consumer.calls[2].args.frameIndex, 1);
  h.consumer.calls[2].resolve(response('consumer', 0, 1)); await second;
  assert.equal(h.pipeline.statistics.maxPending, 2); assert.equal(h.pipeline.statistics.deliveredFrames, 2);
  assert.equal(Object.values(h.pipeline.statistics).some(Array.isArray), false);
  await h.pipeline.close();
});

test('only current resident frame may repeat; renderer acknowledgment tokens are copied only to consumer', async () => {
  const h = harness(); await started(h); await loadZero(h);
  const input = args(0, 0, { prefixToken: 'consumer-owner-token', prefixFieldTokens: { flapComponents: 'e'.repeat(64) } });
  const repeat = h.pipeline.seekFrame(input);
  input.prefixFieldTokens.flapComponents = 'f'.repeat(64);
  assert.equal(h.consumer.calls[2].args.prefixFieldTokens.flapComponents, 'e'.repeat(64));
  assert.equal(h.consumer.calls[2].args.prefixToken, 'consumer-owner-token');
  assert.equal(h.producer.calls.every(call => !('prefixToken' in call.args) && !('prefixFieldTokens' in call.args)), true);
  const returned = response('consumer', 0); returned.timelineOmitted = true; returned.viewFieldsOmitted = ['flapComponents'];
  returned.viewFieldTokens = { flapComponents: 'e'.repeat(64) };
  h.consumer.calls[2].resolve(returned);
  const actual = await repeat;
  assert.equal(actual.timelineToken, returned.timelineToken); assert.equal(actual.view, returned.view);
  assert.deepEqual(actual.viewFieldsOmitted, ['flapComponents']); assert.equal(actual.timelineOmitted, true);
  assert.equal(h.pipeline.statistics.repeatedFrames, 1);
  await h.pipeline.close();
});

for (const [label, mutate] of [
  ['foreign binding', r => { r.prefixWorker.binding.originSnapshotHash = '9'.repeat(64); }],
  ['foreign store', r => { r.prefixWorker.store.directory = '/other'; }],
  ['foreign archive', r => { r.prefixWorker.sourceArchiveSha256 = '9'.repeat(64); }],
  ['wrong role', r => { r.prefixWorker.role = 'producer'; }],
  ['consumer physics/publication', r => { r.prefixWorker.cache.publishedActions = 1; }],
  ['cache miss', r => { r.prefixWorker.cache.misses = 1; }],
  ['STORE miss', r => { r.prefixWorker.store.misses = 1; }],
  ['action byte overflow', r => { r.prefixWorker.actionFrameBytes = 64 * MB + 1; }],
  ['disk overflow', r => { r.prefixWorker.store.diskBytes = 512 * MB + 1; }],
  ['staging cap drift', r => { r.prefixWorker.maxStagedActionBytes++; }],
  ['verification drift', r => { r.verification.prefixImplementation = '0'.repeat(64); }],
  ['wrong frame', r => { r.playback.frameIndex = 1; }]
]) test(`consumer ${label} is terminal, never forwarded`, async () => {
  const h = harness(); await started(h);
  const pending = h.pipeline.seekFrame(args(0, 0));
  const bad = response('consumer', 0); mutate(bad); h.consumer.calls[1].resolve(bad);
  await assert.rejects(pending, /PREFIX_PIPELINE/); await tick();
  assert.equal(h.fatal.length, 1); assert.equal(h.pipeline.statistics.status, 'failed');
  assert.equal(h.producer.calls.length, 2); assert.equal(h.producer.stops, 1); assert.equal(h.consumer.stops, 1);
  await h.pipeline.close();
});

test('background committed action with skipped publication fails once and cancels active consumer; no fallback or retry', async () => {
  const h = harness(); await started(h); await loadZero(h);
  const pending = h.pipeline.seekFrame(args(0, 1));
  const failure = assert.rejects(pending, /CACHE_COUNTERS/);
  const committedButUnpublished = response('producer', 1);
  committedButUnpublished.prefixWorker.cache.publishedActions = 1;
  committedButUnpublished.prefixWorker.cache.lastSkip = 'disk quota';
  h.producer.calls[2].resolve(committedButUnpublished);
  await failure; await tick();
  assert.equal(h.consumer.calls[2].cancels, 1); assert.equal(h.fatal.length, 1);
  h.consumer.calls[2].resolve(response('consumer', 0, 1)); await tick();
  assert.equal(h.pipeline.statistics.deliveredFrames, 1); assert.equal(h.fatal.length, 1);
  assert.equal(h.producer.calls.length, 3); await h.pipeline.close();
});

test('rejected producer promise is observed even with no foreground request', async () => {
  const h = harness(); await started(h); await loadZero(h);
  h.producer.calls[2].reject(new Error('after committed action native identity unavailable')); await tick();
  assert.equal(h.pipeline.statistics.status, 'failed'); assert.equal(h.fatal.length, 1);
  await assert.rejects(h.pipeline.seekFrame(args(0, 1)), /FAILED/); await h.pipeline.close();
});

for (const request of [args(1, 0), args(0, 1), args(0, 0, { branchId: 'foreign' }),
  args(0, 0, { cacheOnly: true }), args(0, 0, { prefixFieldTokens: { flapComponents: 'invalid' } })]) {
  test('nonsequential/foreign/unsupported request cannot dispatch', async () => {
    const h = harness(); await started(h);
    await assert.rejects(h.pipeline.seekFrame(request), /PREFIX_PIPELINE/);
    assert.equal(h.consumer.calls.length, 1); assert.equal(h.fatal.length, 1); await h.pipeline.close();
  });
}

test('manual backwards seek cannot reload a previously consumed action', async () => {
  const h = harness(); await started(h); await loadZero(h);
  let call = h.pipeline.seekFrame(args(0, 1)); h.consumer.calls[2].resolve(response('consumer', 0, 1)); await call;
  h.producer.calls[2].resolve(response('producer', 1)); await tick();
  call = h.pipeline.seekFrame(args(1, 0)); h.consumer.calls[3].resolve(response('consumer', 1)); await call;
  await assert.rejects(h.pipeline.seekFrame(args(0, 1)), /SEQUENTIAL_ONLY/);
  assert.equal(h.consumer.calls.length, 4); await h.pipeline.close();
});

test('close rejects active and queued work, cancels both leases and ignores late results/rejections', async () => {
  const h = harness(); await started(h); await loadZero(h);
  const first = h.pipeline.seekFrame(args(0, 1)), second = h.pipeline.seekFrame(args(1, 0));
  const rejected = Promise.all([assert.rejects(first, /CLOSED/), assert.rejects(second, /CLOSED/)]);
  await h.pipeline.close(); await rejected;
  assert.equal(h.producer.calls[2].cancels, 1); assert.equal(h.consumer.calls[2].cancels, 1);
  h.producer.calls[2].reject(new Error('late producer rejection'));
  h.consumer.calls[2].resolve(response('consumer', 0, 1)); await tick();
  assert.equal(h.pipeline.statistics.deliveredFrames, 1); assert.equal(h.fatal.length, 0);
  await assert.rejects(h.pipeline.seekFrame(args(0, 1)), /CLOSED/);
  await assert.rejects(h.pipeline.start(), /CLOSED/);
  await h.pipeline.close(); assert.equal(h.producer.stops, 1); assert.equal(h.consumer.stops, 1);
});

test('close during startup prevents stale worker completion from starting consumer or publishing output', async () => {
  const h = harness(); const starting = h.pipeline.start(); const rejected = assert.rejects(starting, /CLOSED/);
  await h.pipeline.close(); await rejected;
  h.producer.calls[0].resolve(response('producer', null)); await tick();
  assert.equal(h.consumer.calls.length, 0); assert.equal(h.producer.calls.length, 1); assert.equal(h.fatal.length, 0);
});

test('final action does not launch a producer beyond the total action count', async () => {
  const h = harness(); await started(h, 1); await loadZero(h, 1);
  const final = h.pipeline.seekFrame(args(0, 1)); h.consumer.calls[2].resolve(response('consumer', 0, 1, 2, 1)); await final;
  assert.equal(h.producer.calls.length, 2); assert.equal(h.pipeline.statistics.maxLookahead, 1);
  await h.pipeline.close();
});

test('close preserves worker stop failure and cannot provide successful replacement-exit proof', async () => {
  const h = harness();
  const failure = new Error('actual producer exit not proven');
  h.producer.stop = () => { h.producer.stops++; return Promise.reject(failure); };
  const check = error => error instanceof AggregateError
    && error.message === 'PREFIX_PIPELINE:WORKER_STOP_FAILED'
    && error.errors.length === 1 && error.errors[0] === failure;
  await assert.rejects(h.pipeline.close(), check);
  assert.equal(h.producer.stops, 1); assert.equal(h.consumer.stops, 1);
  await assert.rejects(h.pipeline.close(), check);
  assert.equal(h.producer.stops, 1); assert.equal(h.consumer.stops, 1);
  await assert.rejects(h.pipeline.start(), /CLOSED/);
});

test('actual Sidecar early rejection without a request ID is observed and closes both workers', async () => {
  const h = harness();
  h.producer.closed = true;
  h.producer.request = Sidecar.prototype.request;
  h.producer.requestLease = Sidecar.prototype.requestLease;
  await assert.rejects(h.pipeline.start(), /INVALID_LEASE/);
  await h.pipeline.close();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(h.fatal.length, 1);
  assert.equal(h.producer.stops, 1); assert.equal(h.consumer.stops, 1);
});

test('close in the lease-finally continuation cannot resurrect startup or prepared metadata', async () => {
  for (const boundary of ['publication', 'consumer-start']) {
    const h = harness(); let armed = true;
    h.pipeline.now = () => {
      if (armed && (boundary === 'publication' ? h.producer.calls.length === 2 : h.consumer.calls.length === 1)) {
        armed = false; void h.pipeline.close();
      }
      return 0;
    };
    const starting = h.pipeline.start(); const rejection = assert.rejects(starting, /CLOSED/);
    h.producer.calls[0].resolve(response('producer', null)); await tick();
    h.producer.calls[1].resolve(response('producer', 0)); await tick();
    if (boundary === 'consumer-start') h.consumer.calls[0].resolve(response('consumer', null));
    await rejection; await h.pipeline.close();
    assert.equal(h.pipeline.status, 'closed'); assert.equal(h.pipeline.ready, null);
    if (boundary === 'publication') assert.equal(h.pipeline.statistics.preparedActions, 0);
  }
});

test('fresh actions may authenticate existing deterministic packages while actual new writes remain zero', async () => {
  const h = harness(); const start = h.pipeline.start();
  h.producer.calls[0].resolve(response('producer', null, 0, 2, 2)); await tick();
  const first = response('producer', 0, 0, 2, 2); first.prefixWorker.store.publications = 0;
  h.producer.calls[1].resolve(first); await tick();
  h.consumer.calls[0].resolve(response('consumer', null, 0, 2, 2)); await start;
  await loadZero(h, 2);
  const second = response('producer', 1, 0, 2, 2); second.prefixWorker.store.publications = 0;
  h.producer.calls[2].resolve(second); await tick();
  const warm = h.pipeline.seekFrame(args(0, 1)); h.consumer.calls[2].resolve(response('consumer',0,1,2,2)); await warm;
  const next = h.pipeline.seekFrame(args(1, 0)); h.consumer.calls[3].resolve(response('consumer',1,0,2,2)); await next;
  assert.equal(h.pipeline.statistics.preparedActions, 2);
  assert.deepEqual(h.pipeline.statistics.newPackagePublications, {producer:0,consumer:0});
  await h.pipeline.close();
});

test('actual store publication count cannot regress between committed producer actions', async () => {
  const h = harness(); await started(h); await loadZero(h);
  const regressed = response('producer',1); regressed.prefixWorker.store.publications = 0;
  h.producer.calls[2].resolve(regressed); await tick();
  assert.match(h.pipeline.failure.message, /STORE_PUBLICATION_DRIFT/);
  assert.equal(h.pipeline.status,'failed'); await h.pipeline.close();
});
