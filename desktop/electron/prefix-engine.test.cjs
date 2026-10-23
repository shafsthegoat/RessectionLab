'use strict';
const test = require('node:test'), assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const { PrefixEngine } = require('./prefix-engine.cjs');
const tick = async () => { for (let i = 0; i < 64; i++) await Promise.resolve(); };
const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; };
const digest = 'sha256:' + 'a'.repeat(64), sourceSha = 'b'.repeat(64), MB = 1024 ** 2;
const binding = { schema: 'resectionlab.prepared-prefix-binding.v1', sceneId: digest,
  originSnapshotHash: digest, actionLogHash: digest, protectionContractHash: digest,
  implementation: 'c'.repeat(64), prefixImplementation: 'd'.repeat(64) };
const startArgs = { command: 'seekPrefix', caseHash: 'case', branchId: 'branch', position: 0 };
const frameArgs = (frameIndex = 0) => ({ ...startArgs, command: 'seekPrefixFrame', frameIndex });

function prefixResult(role, action = null, frameIndex = 0) {
  const verified = action === null ? 0 : action + 1;
  const cache = { mode: role === 'consumer' ? 'only' : 'prepare', loadedActions: role === 'consumer' ? verified : 0,
    publishedActions: role === 'producer' ? verified : 0, misses: 0, lastSkip: null };
  const result = { caseHash: 'case', branchId: 'branch', position: action ?? 0, readOnly: true,
    changed: false, outcome: null, budgetStopped: false, timelineToken: role + '-token',
    view: { stateId: digest, scene: { sceneId: digest }, marker: role },
    verification: { ...binding, status: 'reconstructed-prefix', readOnly: true,
      verifiedThrough: verified, totalActions: 2, failure: null, preparedCache: { ...cache } } };
  if (role !== 'normal') result.prefixWorker = { schema: 'resectionlab.prefix-worker.v1', role,
    binding: { ...binding }, sceneId: digest, sourceArchiveSha256: sourceSha,
    verifiedThrough: verified, totalActions: 2, cache, maxStagedActionBytes: 64 * MB,
    actionFrameBytes: action === null ? 0 : 4096, store: { directory: '/cache', namespaceDirectory: '/cache/prepared-prefix-v1',
      maxDiskBytes: 512 * MB, maxActionBytes: 64 * MB, hits: cache.loadedActions, misses: 0,
      publications: cache.publishedActions, lastSkip: null, diskBytes: 8192 } };
  if (action !== null) result.playback = { branchId: 'branch', readOnly: true, position: action,
    actionIndex: action, frameIndex, frameCount: 2, frameStateId: digest };
  return result;
}

function registry(records = []) {
  return { records: new Map(records), clear() { this.records.clear(); },
    async read(id) { if (!this.records.has(id)) throw Error('Unknown asset'); return this.records.get(id); } };
}

class Worker extends EventEmitter {
  constructor(h, config) {
    super(); this.h = h; this.config = config; this.role = config.prefixWorker?.role || 'normal';
    this.child = { pid: 1000 + h.workers.length }; this.calls = []; this.stops = 0;
    this.assets = registry([[this.role === 'normal' ? 'old-image' : this.role + '-private-image', this.role + '-bytes']]);
    this.contactFamilyAvailability = { marker: this.role };
  }
  requestLease(op, args, timeout) {
    const pending = deferred(), call = { ...pending, op, args, timeout, cancels: 0, id: this.role + '-' + this.calls.length };
    this.calls.push(call);
    return { id: call.id, promise: pending.promise, cancel() { call.cancels++; return Promise.resolve(); } };
  }
  finish(index, result) {
    const call = this.calls[index];
    this.emit('event', { id: call.id, event: 'result', op: call.op, result });
    call.resolve(result);
  }
  stop() {
    if (!this.stopPromise) {
      this.stops++; this.emit('event', { event: 'engineStopped', error: { message: 'expected exit' } });
      this.stopPromise = (this.stopGate?.promise || Promise.resolve()).then(() => { this.h.alive.delete(this); });
    }
    return this.stopPromise;
  }
}

async function harness(enabled = true) {
  const h = { workers: [], alive: new Set(), peak: 0, held: [], resources: [], hashes: [], events: [], diagnostics: [] };
  h.createWorker = async config => {
    if (h.beforeCreate) await h.beforeCreate(config);
    const worker = new Worker(h, config); h.workers.push(worker); h.alive.add(worker);
    h.peak = Math.max(h.peak, h.alive.size); return worker;
  };
  const engine = await PrefixEngine.create({ enabled, createWorker: h.createWorker,
    sourceForHash: hash => hash === 'other' ? '/other.saved' : '/source.saved',
    hashSource: async filename => { h.hashes.push(filename); return { path: filename,
      sha256: filename === '/other.saved' ? 'e'.repeat(64) : sourceSha }; },
    preparedPrefixCacheDirectory: '/cache', maxWallMs: 10000,
    retainAssets: async original => {
      const held = { registry: registry(), bytes: 0, files: 0, closes: 0, appends: 0,
        async retain(next) { this.appends++; for (const [id, value] of next.records) this.registry.records.set(id, value); this.files = this.registry.records.size; this.bytes = this.files * 10; },
        async close() { this.closes++; this.registry.clear(); } };
      await held.retain(original); h.held.push(held); return held;
    }, resourcesFactory: options => {
      const resource = { options, starts: 0, checks: 0, closes: 0,
        statistics: { samples: 1, maxAggregatePythonTreeRssBytes: 3 * 1024 ** 3 },
        async start() { this.starts++; }, async check() { this.checks++; }, async close() { this.closes++; } };
      h.resources.push(resource); return resource;
    } });
  h.engine = engine; engine.on('event', event => h.events.push(event)); engine.on('diagnostic', event => h.diagnostics.push(event));
  return h;
}

const loadResult = role => ({ caseHash: 'case', prefixWorker: { schema: 'resectionlab.prefix-worker.v1', role, sourceArchiveSha256: sourceSha } });
async function activate(h) {
  const pending = h.engine.request('surgicalRehearsal', { ...startArgs }); await tick();
  h.workers[0].finish(0, prefixResult('normal')); await tick();
  assert.equal(h.workers.length, 3); assert.equal(h.workers[0].stops, 1); assert.equal(h.alive.has(h.workers[0]), false);
  const consumer = h.workers[1], producer = h.workers[2];
  assert.equal(consumer.calls[0].op, 'loadCase'); assert.equal(producer.calls.length, 0);
  consumer.finish(0, loadResult('consumer')); await tick();
  producer.finish(0, loadResult('producer')); await tick();
  producer.finish(1, prefixResult('producer')); await tick();
  producer.finish(2, prefixResult('producer', 0)); await tick();
  consumer.finish(1, prefixResult('consumer'));
  const result = await pending;
  return { consumer, producer, result };
}

test('default facade uses one ordinary worker and forwards original requests/results/events/availability/assets', async () => {
  const h = await harness(false), original = h.workers[0], args = { untouched: true };
  const pending = h.engine.request('ping', args, 1000), result = { pong: true };
  assert.equal(original.calls[0].args, args); original.finish(0, result);
  assert.equal(await pending, result); assert.equal(h.events[0].result, result);
  original.emit('diagnostic', 'ordinary diagnostic'); assert.deepEqual(h.diagnostics, ['ordinary diagnostic']);
  assert.equal(h.engine.contactFamilyAvailability, original.contactFamilyAvailability);
  assert.equal(await h.engine.assets.read('old-image'), 'normal-bytes');
  assert.equal(h.workers.length, 1); await h.engine.stop();
});

test('opt-in startup loads exact source sequentially after proven ordinary exit, keeps only one clone and at most two workers', async () => {
  const h = await harness(), { consumer, producer, result } = await activate(h);
  assert.equal(result.prefixWorker, undefined); assert.equal(result.view.marker, 'consumer');
  assert.equal(h.peak, 2); assert.equal(h.held.length, 1); assert.equal(h.resources[0].starts, 1);
  assert.equal(await h.engine.assets.read('old-image'), 'normal-bytes');
  for (const worker of [consumer, producer]) {
    assert.deepEqual(worker.calls[0].args, { path: '/source.saved' });
    assert.equal(worker.config.preparedPrefixCacheDirectory, '/cache');
    assert.deepEqual(worker.config.prefixWorker, { role: worker.role, sourcePath: '/source.saved', sourceSha256: sourceSha, branchId: 'branch' });
    assert.equal(worker.assets.records.size, 0);
  }
  assert.equal(h.events.some(event => event.event === 'engineStopped'), false);
  assert.equal(h.events.some(event => event.result?.prefixWorker), false);
  assert.equal(h.hashes.length >= 6, true); await h.engine.stop(); assert.equal(h.held[0].closes, 1);
});

test('ordinary work drains before initial prefix probe, and roles cannot spawn until old actual exit resolves', async () => {
  const h = await harness(), old = h.workers[0]; old.stopGate = deferred();
  const busy = h.engine.request('inspectEvidence', { caseHash: 'case' });
  const start = h.engine.request('surgicalRehearsal', { ...startArgs }); const rejection = assert.rejects(start, /CLOSED|STALE/);
  await tick(); assert.equal(old.calls.length, 1);
  old.finish(0, { complete: true }); await busy; await tick(); assert.equal(old.calls[1].args.command, 'seekPrefix');
  old.finish(1, prefixResult('normal')); await tick(); assert.equal(h.workers.length, 1);
  const stopping = h.engine.stop(); old.stopGate.resolve(); await stopping; await rejection;
  assert.equal(h.workers.length, 1);
});

test('only admitted consumer events forward; terminal metadata waits validation and ping cannot steal frame terminal slot', async () => {
  const h = await harness(), { consumer, producer } = await activate(h); h.events.length = 0;
  producer.emit('event', { event: 'progress', op: 'surgicalRehearsal', future: true }); producer.emit('diagnostic', 'future');
  consumer.emit('event', { event: 'progress', op: 'surgicalRehearsal', progress: 0.5 });
  assert.equal(h.events.length, 1); assert.equal(h.diagnostics.length, 0);
  const frame = h.engine.request('surgicalRehearsal', frameArgs());
  const ping = h.engine.request('ping', {}, 1000); consumer.finish(3, { pong: true }); await ping;
  const response = prefixResult('consumer', 0); consumer.finish(2, response);
  assert.equal(h.events.some(event => event.result?.view), false);
  const result = await frame;
  assert.equal(result.prefixWorker, undefined); assert.equal(h.events.at(-1).result, result);
  assert.equal(h.events.at(-1).result.timelineToken, 'consumer-token');
  assert.equal(h.events.some(event => event.result?.prefixWorker), false); await h.engine.stop();
});

test('wrong role-load source poisons startup and closes every created worker and retained snapshot', async () => {
  const h = await harness(); const start = h.engine.request('surgicalRehearsal', { ...startArgs });
  const rejection = assert.rejects(start, /ROLE_SOURCE_LOAD/); await tick(); h.workers[0].finish(0, prefixResult('normal')); await tick();
  const wrong = loadResult('consumer'); wrong.prefixWorker.sourceArchiveSha256 = '0'.repeat(64);
  h.workers[1].finish(0, wrong); await rejection;
  assert.equal(h.alive.size, 0); assert.equal(h.held[0].closes, 1); assert.equal(h.engine.statistics.mode, 'failed'); await h.engine.stop();
});

test('cancel closes both workers/resources without fallback; original cloned image remains readable until explicit new load', async () => {
  const h = await harness(), { consumer, producer } = await activate(h);
  const frame = h.engine.request('surgicalRehearsal', frameArgs()); const rejection = assert.rejects(frame, /CLOSED|STALE/);
  assert.deepEqual(await h.engine.request('cancel', {}), { cancelled: true }); await rejection;
  assert.equal(consumer.stops, 1); assert.equal(producer.stops, 1); assert.equal(h.workers.length, 3);
  assert.equal(await h.engine.assets.read('old-image'), 'normal-bytes');
  const loaded = h.engine.request('loadCase', { path: '/other.saved' }); await tick();
  const ordinary = h.workers[3]; ordinary.finish(0, { caseHash: 'other' }); await loaded;
  assert.equal(h.held[0].closes, 1); assert.equal(h.engine.statistics.mode, 'normal');
  await h.engine.stop();
});

test('resource failure rejects current fetch, never restarts physics and ignores stale late consumer completion', async () => {
  const h = await harness(), { consumer } = await activate(h);
  const frame = h.engine.request('surgicalRehearsal', frameArgs()); const rejected = assert.rejects(frame, /CLOSED|STALE/);
  h.resources[0].options.onFailure(new Error('aggregate RSS exceeded')); await rejected; await tick();
  consumer.finish(2, prefixResult('consumer', 0)); await tick();
  assert.equal(h.engine.statistics.mode, 'failed'); assert.equal(h.workers.length, 3);
  assert.equal(h.events.filter(event => event.event === 'engineStopped').length, 1);
  await assert.rejects(h.engine.request('surgicalRehearsal', frameArgs()), /aggregate RSS/);
  assert.equal(await h.engine.assets.read('old-image'), 'normal-bytes'); await h.engine.stop();
});

test('failed worker exit blocks replacement and stop propagates failure', async () => {
  const h = await harness(), { producer } = await activate(h);
  producer.stop = () => { producer.stops++; return Promise.reject(new Error('no actual exit proof')); };
  await assert.rejects(h.engine.request('loadCase', { path: '/other.saved' }), /EXIT_NOT_PROVEN/);
  assert.equal(h.workers.length, 3);
  await assert.rejects(h.engine.request('loadCase', { path: '/other.saved' }), /EXIT_NOT_PROVEN/);
  await assert.rejects(h.engine.stop(), /STOP_FAILED/); assert.equal(h.workers.length, 3);
});

test('background worker exit is terminal once, and producer diagnostics/future replies never reach renderer', async () => {
  const h = await harness(), { consumer, producer } = await activate(h);
  producer.emit('event', { event: 'engineStopped' }); consumer.emit('event', { event: 'engineStopped' }); await tick();
  assert.equal(h.events.filter(event => event.event === 'engineStopped').length, 1);
  assert.equal(h.engine.statistics.mode, 'failed'); assert.equal(h.workers.length, 3); await h.engine.stop();
});

test('same-source internal reload preserves original clone and next activation appends into that one bounded registry', async () => {
  const h = await harness(); await activate(h); h.events.length = 0;
  const operation = h.engine.request('inspectEvidence', { caseHash: 'case' }); await tick();
  const ordinary = h.workers[3]; assert.equal(ordinary.calls[0].op, 'loadCase');
  ordinary.finish(0, { caseHash: 'case' }); await tick();
  assert.equal(h.events.length, 0); // internal source result is not renderer data
  ordinary.finish(1, { evidence: true }); await operation;
  assert.equal(h.held[0].closes, 0); assert.equal(await h.engine.assets.read('old-image'), 'normal-bytes');
  ordinary.assets.records.set('new-visible-id', 'new-array');
  const next = h.engine.request('surgicalRehearsal', { ...startArgs }); await tick();
  ordinary.finish(2, prefixResult('normal')); await tick();
  const consumer = h.workers[4], producer = h.workers[5];
  consumer.finish(0, loadResult('consumer')); await tick(); producer.finish(0, loadResult('producer')); await tick();
  producer.finish(1, prefixResult('producer')); await tick(); producer.finish(2, prefixResult('producer', 0)); await tick();
  consumer.finish(1, prefixResult('consumer')); await next;
  assert.equal(h.held.length, 1); assert.equal(h.held[0].appends, 2);
  assert.equal(await h.engine.assets.read('new-visible-id'), 'new-array'); assert.equal(h.peak, 2);
  await h.engine.stop();
});

test('late asynchronous worker creation during stop is registered and exited without spawning its peer', async () => {
  const h = await harness(), gate = deferred();
  h.beforeCreate = config => config.prefixWorker?.role === 'consumer' ? gate.promise : Promise.resolve();
  const starting = h.engine.request('surgicalRehearsal', { ...startArgs }); const rejected = assert.rejects(starting, /CLOSED|STALE/);
  await tick(); h.workers[0].finish(0, prefixResult('normal')); await tick();
  assert.equal(h.workers.length, 1); const stop = h.engine.stop(); gate.resolve();
  await stop; await rejected;
  assert.equal(h.workers.length, 2); assert.equal(h.workers[1].stops, 1);
  assert.equal(h.alive.size, 0); assert.equal(h.held[0].closes, 1);
});

for (const outcome of ['cancel','role-failure']) test(`same-source restart ${outcome} preserves previously accepted image IDs`, async () => {
  const h = await harness(); await activate(h);
  const ordinaryRequest = h.engine.request('inspectEvidence',{caseHash:'case'}); await tick();
  const ordinary = h.workers[3]; ordinary.finish(0,{caseHash:'case'}); await tick();
  ordinary.finish(1,{evidence:true}); await ordinaryRequest;
  const restart = h.engine.request('surgicalRehearsal',{...startArgs});
  const rejected = assert.rejects(restart,outcome === 'cancel' ? /CLOSED|STALE/ : /ROLE_SOURCE_LOAD/);
  await tick(); ordinary.finish(2,prefixResult('normal')); await tick();
  if (outcome === 'cancel') await h.engine.request('cancel',{});
  else {const wrong=loadResult('consumer');wrong.caseHash='foreign';h.workers[4].finish(0,wrong);}
  await rejected;
  assert.equal(h.held.length,1); assert.equal(h.held[0].closes,0);
  assert.equal(await h.engine.assets.read('old-image'),'normal-bytes');
  assert.equal(h.alive.size,0); await h.engine.stop();
});

test('invalid early-rejected lease is observed and rejected without escaping the ordinary dispatch', async () => {
  const h = await harness(false), failure = new Error('early request refusal');
  h.workers[0].requestLease = () => ({ promise: Promise.reject(failure) });
  await assert.rejects(h.engine.request('ping', {}), /LEASE_API/);
  await tick(); await h.engine.stop();
});

test('trusted main origin is stable and detached on accepted consumer invokes/events, then clears on cancellation', async () => {
  const h = await harness(), {consumer,result} = await activate(h);
  const origin = structuredClone(result.prefixOrigin);
  assert.equal(origin.schema,'resectionlab.prefix-origin.v1');assert.equal(origin.kind,'fresh-reconstruction');
  assert.match(origin.activationId,/^[0-9a-f-]{36}$/);assert.equal(origin.caseHash,'case');
  assert.equal(origin.branchId,'branch');assert.equal(origin.sourceArchiveSha256,sourceSha);assert.equal(origin.sceneId,digest);
  result.prefixOrigin.caseHash = 'caller mutation';
  const pending = h.engine.request('surgicalRehearsal',frameArgs());
  const response = prefixResult('consumer',0);response.prefixOrigin={kind:'worker-provided'};
  consumer.finish(2,response);const frame = await pending;
  assert.deepEqual(frame.prefixOrigin,origin);assert.equal(frame.prefixWorker,undefined);
  assert.equal(frame.verification.preparedCache.mode,'only');
  assert.equal(h.events.at(-1).result,frame);
  assert.deepEqual(h.events.at(-1).result.prefixOrigin,origin);
  await h.engine.request('cancel',{});assert.equal(h.engine.prefixOrigin,null);
  await h.engine.stop();
});
