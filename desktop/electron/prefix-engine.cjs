'use strict';
const { EventEmitter } = require('node:events');
const { randomUUID } = require('node:crypto');
const fs = require('node:fs/promises');
const path = require('node:path');
const { PrefixPipeline } = require('./prefix-pipeline.cjs');
const { retainPrefixAssets, fileHash } = require('./prefix-assets.cjs');
const { PrefixResources } = require('./prefix-resources.cjs');

const need = (ok, message) => { if (!ok) throw new Error(`PREFIX_ENGINE:${message}`); };
const plain = value => value && typeof value === 'object' && !Array.isArray(value)
  && Object.getPrototypeOf(value) === Object.prototype;
const keys = ['sceneId', 'originSnapshotHash', 'actionLogHash', 'protectionContractHash', 'implementation', 'prefixImplementation'];
const digest = value => typeof value === 'string' && /^(?:sha256:)?[a-f0-9]{64}$/.test(value);
const freshStart = (op, args) => op === 'surgicalRehearsal' && args?.command === 'seekPrefix'
  && args.position === 0 && args.cacheOnly !== true;

async function hashSource(filename) {
  need(typeof filename === 'string' && path.isAbsolute(filename), 'SOURCE_PATH');
  const entry = await fs.lstat(filename);
  need(entry.isFile() && !entry.isSymbolicLink(), 'SOURCE_NOT_REGULAR');
  const real = await fs.realpath(filename), before = await fs.stat(real);
  const sha256 = await fileHash(real), after = await fs.stat(real);
  need(before.isFile() && before.size === after.size && before.mtimeMs === after.mtimeMs
    && before.ino === after.ino && before.dev === after.dev, 'SOURCE_CHANGED_DURING_HASH');
  return { path: real, sha256 };
}

class PrefixEngine extends EventEmitter {
  constructor({ createWorker, enabled = false, sourceForHash, maxWallMs = 7200000,
    preparedPrefixCacheDirectory, retainAssets = retainPrefixAssets,
    resourcesFactory = options => new PrefixResources(options), hashSource: hash = hashSource }) {
    super();
    need(typeof createWorker === 'function' && typeof enabled === 'boolean'
      && (!enabled || typeof sourceForHash === 'function'), 'CONFIGURATION');
    need(Number.isSafeInteger(maxWallMs) && maxWallMs >= 100 && maxWallMs <= 7200000, 'WALL_LIMIT');
    need(preparedPrefixCacheDirectory === undefined || typeof preparedPrefixCacheDirectory === 'string'
      && path.isAbsolute(preparedPrefixCacheDirectory), 'CACHE_DIRECTORY');
    this.createWorker = createWorker; this.enabled = enabled; this.sourceForHash = sourceForHash;
    this.maxWallMs = maxWallMs; this.preparedPrefixCacheDirectory = preparedPrefixCacheDirectory;
    this.retainAssets = retainAssets; this.resourcesFactory = resourcesFactory; this.hashSource = hash;
    this.mode = 'new'; this.epoch = 0; this.closed = false; this.failure = null; this.exitFailure = null;
    this.ordinary = null; this.consumer = null; this.producer = null; this.pipeline = null; this.resources = null;
    this.workers = new Set(); this.listeners = new Map(); this.calls = new Set(); this.stops = new WeakMap();
    this.retained = null; this.source = null; this.terminalEvent = null;
    this.prefixOrigin = null;
    this.tail = Promise.resolve(); this.transitions = 0; this.shutdownPromise = null; this.stopPromise = null;
    this.lastPipeline = null; this.lastResources = null; this.poisonNotified = false;
    this.counts = { transitions: 0, workerCreations: 0, maxWorkers: 0, maxInflight: 0, failures: 0 };
    this.assets = Object.freeze({ read: async assetId => {
      const epoch = this.epoch, active = this.consumer || this.ordinary;
      const registry = active?.assets;
      let result;
      if (registry?.records?.has(assetId)) result = await registry.read(assetId);
      else if (this.retained?.registry.records.has(assetId)) result = await this.retained.registry.read(assetId);
      else if (registry) result = await registry.read(assetId);
      else throw new Error('Unknown or expired transfer asset');
      this._live(epoch); return result;
    } });
  }

  static async create(options) {
    const owner = new PrefixEngine(options);
    try {
      owner.ordinary = await owner._create({}, owner.epoch);
      owner._attach(owner.ordinary, 'normal', owner.epoch); owner.mode = 'normal'; return owner;
    }
    catch (error) { await owner.stop().catch(() => {}); throw error; }
  }

  get contactFamilyAvailability() { return (this.ordinary || this.consumer)?.contactFamilyAvailability; }
  get statistics() {
    return { ...this.counts, enabled: this.enabled, mode: this.mode, activeWorkers: this.workers.size,
      ordinaryInflight: this.calls.size, pendingTransitions: this.transitions,
      pipeline: this.pipeline?.statistics || this.lastPipeline,
      resources: this.resources?.statistics || this.lastResources,
      retainedAssetBytes: this.retained?.bytes || 0, retainedAssetFiles: this.retained?.files || 0,
      failure: this.failure?.message || null, exitFailure: this.exitFailure?.message || null };
  }

  _live(epoch = this.epoch) {
    need(!this.closed && epoch === this.epoch, this.closed ? 'CLOSED' : 'STALE_SOURCE_EPOCH');
    if (this.exitFailure) throw this.exitFailure;
  }

  async _create(config, epoch) {
    this._live(epoch); need(this.workers.size < 2, 'WORKER_CAP');
    const worker = await this.createWorker({ ...config,
      ...(this.preparedPrefixCacheDirectory === undefined ? {} : { preparedPrefixCacheDirectory: this.preparedPrefixCacheDirectory }) });
    // Register even a stale asynchronous creation so cleanup must prove its exit.
    this.workers.add(worker); this.counts.workerCreations++;
    this.counts.maxWorkers = Math.max(this.counts.maxWorkers, this.workers.size);
    need(worker && typeof worker.requestLease === 'function' && typeof worker.stop === 'function'
      && typeof worker.on === 'function' && typeof worker.removeListener === 'function', 'WORKER_API');
    this._live(epoch); return worker;
  }

  _detach(worker) {
    const listeners = this.listeners.get(worker);
    if (listeners) { worker.removeListener('event', listeners.event); worker.removeListener('diagnostic', listeners.diagnostic); this.listeners.delete(worker); }
  }

  _attach(worker, role, epoch) {
    this._detach(worker);
    const event = payload => {
      if (this.closed || epoch !== this.epoch) return;
      if (role === 'normal') {
        if (worker === this.ordinary) this.emit('event', payload);
        return;
      }
      if (payload.event === 'engineStopped') {
        // Scheduler failures synchronously initiate both stops before onFatal.
        // Do not reenter that close or replace its original failure reason.
        if (this.pipeline && ['closed', 'failed'].includes(this.pipeline.statistics.status)) return;
        this._poison(new Error('PREFIX_ENGINE:WORKER_EXIT'), epoch); return;
      }
      if (role !== 'consumer' || this.mode !== 'pipeline') return;
      // Runtime ping is out of band and carries no prefix result/authority.
      if (payload.op === 'ping') { this.emit('event', payload); return; }
      if (payload.event === 'result') {
        // A raw Sidecar result event precedes promise validation. Hold just the
        // one active consumer terminal event until the scheduler admits it.
        if (this.terminalEvent) { this._poison(new Error('PREFIX_ENGINE:TERMINAL_EVENT_CAP'), epoch); return; }
        this.terminalEvent = payload;
      } else if (!['error', 'cancelled'].includes(payload.event)) this.emit('event', payload);
    };
    const diagnostic = message => {
      if (!this.closed && epoch === this.epoch && (role === 'normal' || role === 'consumer' && this.mode === 'pipeline')) this.emit('diagnostic', message);
    };
    worker.on('event', event); worker.on('diagnostic', diagnostic); this.listeners.set(worker, { event, diagnostic });
  }

  _call(worker, op, args, timeout, epoch) {
    this._live(epoch); need(this.calls.size < 64, 'INFLIGHT_CAP');
    const lease = worker.requestLease(op, args, timeout);
    // An early Sidecar refusal may already be a rejected promise. Observe it
    // before validating the lease so an invalid shape cannot leak a rejection.
    const response = lease?.promise;
    if (response && typeof response.then === 'function') Promise.resolve(response).catch(() => {});
    need(lease && lease.promise && typeof lease.promise.then === 'function' && typeof lease.cancel === 'function', 'LEASE_API');
    let reject;
    const cancelled = new Promise((_, no) => { reject = no; });
    const call = { cancel: error => {
      reject(error);
      try { Promise.resolve(lease.cancel()).catch(() => {}); } catch { /* shutdown still proves exit */ }
    } };
    this.calls.add(call); this.counts.maxInflight = Math.max(this.counts.maxInflight, this.calls.size);
    const promise = Promise.race([response, cancelled]).then(result => { this._live(epoch); return result; })
      .finally(() => this.calls.delete(call)).then(result => { this._live(epoch); return result; });
    call.promise = promise; promise.catch(() => {});
    return promise;
  }

  _interrupt() {
    this.epoch++; this.terminalEvent = null; this.prefixOrigin = null;
    for (const worker of this.workers) this._detach(worker);
    for (const call of this.calls) call.cancel(new Error('PREFIX_ENGINE:STALE_SOURCE_EPOCH'));
    if (this.pipeline) void this.pipeline.close().catch(() => {});
    if (this.resources) void this.resources.close().catch(() => {});
  }

  _poison(error, epoch) {
    if (this.closed || epoch !== this.epoch || this.mode === 'failed') return;
    this.failure = error; this.mode = 'failed'; this.counts.failures++;
    this._interrupt();
    void this._shutdown().catch(() => {});
    if (!this.poisonNotified) {
      this.poisonNotified = true;
      this.emit('event', { event: 'engineStopped', error: { code: 'PREFIX_PIPELINE_FAILED', message: error.message } });
    }
  }

  _stopWorker(worker) {
    if (!this.stops.has(worker)) this.stops.set(worker, Promise.resolve().then(() => worker.stop()));
    return this.stops.get(worker).then(() => { this.workers.delete(worker); this._detach(worker); });
  }

  _shutdown() {
    if (this.shutdownPromise) return this.shutdownPromise;
    this.shutdownPromise = (async () => {
      const errors = [];
      for (const worker of this.workers) this._detach(worker);
      this.terminalEvent = null;
      if (this.resources) {
        this.lastResources = this.resources.statistics;
        try { await this.resources.close(); } catch (error) { errors.push(error); }
        this.resources = null;
      }
      const pair = new Set([this.producer, this.consumer].filter(Boolean));
      if (this.pipeline) {
        this.lastPipeline = this.pipeline.statistics;
        try {
          await this.pipeline.close();
          for (const worker of pair) this.workers.delete(worker);
          this.pipeline = null; this.producer = null; this.consumer = null;
        } catch (error) { errors.push(error); }
      } else {
        // Partial startup has no scheduler yet; every created role still needs
        // an actual-exit proof before any replacement can be created.
        const results = await Promise.allSettled([...pair].map(worker => this._stopWorker(worker)));
        errors.push(...results.filter(row => row.status === 'rejected').map(row => row.reason));
        if (!errors.length) { this.producer = null; this.consumer = null; }
      }
      const others = [...this.workers].filter(worker => !pair.has(worker));
      const results = await Promise.allSettled(others.map(worker => this._stopWorker(worker)));
      errors.push(...results.filter(row => row.status === 'rejected').map(row => row.reason));
      if (this.ordinary && !this.workers.has(this.ordinary)) this.ordinary = null;
      if (errors.length) {
        this.exitFailure = new AggregateError(errors, 'PREFIX_ENGINE:EXIT_NOT_PROVEN');
        throw this.exitFailure;
      }
    })().finally(() => { this.shutdownPromise = null; });
    this.shutdownPromise.catch(() => {});
    return this.shutdownPromise;
  }

  _transition(task, interrupt = false) {
    try { this._live(); need(this.transitions < 2, 'TRANSITION_CAP'); }
    catch (error) { return Promise.reject(error); }
    if (interrupt) this._interrupt();
    const epoch = this.epoch;
    this.mode = 'transition'; this.transitions++; this.counts.transitions++;
    const result = this.tail.then(async () => {
      this._live(epoch);
      try { return await task(epoch); }
      catch (error) {
        try { await this._shutdown(); } catch (stopError) { throw stopError; }
        if (epoch === this.epoch && !this.closed) { this.failure = error; this.mode = 'failed'; }
        throw error;
      }
    }).finally(() => { this.transitions--; });
    this.tail = result.catch(() => {});
    return result;
  }

  async _source(hash) {
    const filename = await this.sourceForHash(hash), source = await this.hashSource(filename);
    need(plain(source) && typeof source.path === 'string' && path.isAbsolute(source.path)
      && /^[a-f0-9]{64}$/.test(source.sha256), 'SOURCE_IDENTITY');
    return { ...source, caseHash: hash };
  }

  async _sameSource(source) {
    const actual = await this.hashSource(source.path);
    need(actual.path === source.path && actual.sha256 === source.sha256, 'SOURCE_CHANGED');
  }

  _binding(result, args) {
    const v = result?.verification;
    need(plain(result) && result.caseHash === args.caseHash && result.branchId === args.branchId
      && result.position === 0 && result.readOnly === true && result.budgetStopped === false
      && result.changed === false && result.outcome === null && plain(v)
      && v.status === 'reconstructed-prefix' && v.readOnly === true && v.verifiedThrough === 0
      && v.failure === null && Number.isSafeInteger(v.totalActions) && v.totalActions > 0, 'NORMAL_PREFIX_ZERO');
    const binding = { schema: 'resectionlab.prepared-prefix-binding.v1', sceneId: result.view?.scene?.sceneId };
    for (const key of keys.slice(1)) binding[key] = v[key];
    need(keys.every(key => digest(binding[key])), 'NORMAL_BINDING');
    return binding;
  }

  async _discardRetained() {
    if (this.retained) { await this.retained.close(); this.retained = null; }
  }

  async _activate(args, timeout, epoch) {
    const priorRetained = this.retained;
    try {
      await Promise.allSettled([...this.calls].map(call => call.promise)); this._live(epoch);
      if (!this.ordinary) await this._restoreOrdinary(args, timeout, epoch);
      const source = await this._source(args.caseHash); this._live(epoch);
      const initial = await this._call(this.ordinary, 'surgicalRehearsal', args, timeout, epoch);
      const expectedBinding = this._binding(initial, args);
      await this._sameSource(source); this._live(epoch);
      if (this.retained) await this.retained.retain(this.ordinary.assets);
      else this.retained = await this.retainAssets(this.ordinary.assets);
      this._live(epoch);
      this._detach(this.ordinary); await this._stopWorker(this.ordinary); this.ordinary = null;
      this._live(epoch); this.source = source;
      for (const role of ['consumer', 'producer']) {
        const worker = await this._create({ prefixWorker: { role, sourcePath: source.path,
          sourceSha256: source.sha256, branchId: args.branchId } }, epoch);
        this[role] = worker; this._attach(worker, role, epoch);
      }
      this.resources = this.resourcesFactory({ pids: [this.producer.child.pid, this.consumer.child.pid],
        maxWallMs: this.maxWallMs, onFailure: error => this._poison(error, epoch) });
      await this.resources.start(); this._live(epoch);
      for (const role of ['consumer', 'producer']) {
        await this._sameSource(source); this._live(epoch);
        const loaded = await this._call(this[role], 'loadCase', { path: source.path }, timeout, epoch);
        const meta = loaded?.prefixWorker;
        need(loaded.caseHash === args.caseHash && plain(meta) && meta.schema === 'resectionlab.prefix-worker.v1'
          && meta.role === role && meta.sourceArchiveSha256 === source.sha256, 'ROLE_SOURCE_LOAD');
        await this._sameSource(source); this._live(epoch);
        // Role loads are private. Their new transfer IDs have never reached
        // the renderer, which retains only the bounded cloned original assets.
        this[role].assets.clear();
      }
      this.pipeline = new PrefixPipeline({ producer: this.producer, consumer: this.consumer,
        caseHash: args.caseHash, branchId: args.branchId, expectedBinding,
        onFatal: error => this._poison(error, epoch) });
      const result = await this.pipeline.start(timeout); this._live(epoch);
      await this.resources.check(); this._live(epoch);
      this.mode = 'pipeline'; this.failure = null; this.poisonNotified = false;
      this.prefixOrigin = Object.freeze({schema:'resectionlab.prefix-origin.v1',kind:'fresh-reconstruction',
        activationId:randomUUID(),caseHash:args.caseHash,branchId:args.branchId,
        sourceArchiveSha256:source.sha256,sceneId:expectedBinding.sceneId});
      return this._withOrigin(result, epoch);
    } catch (error) {
      await this._shutdown();
      // Cancellation of a same-source restart must preserve image IDs already
      // accepted by the renderer. A newly created snapshot is still disposable.
      if (this.retained !== priorRetained) await this._discardRetained();
      throw error;
    }
  }

  async _restoreOrdinary(args, timeout, epoch, load = true) {
    await this._shutdown(); this._live(epoch);
    this.ordinary = await this._create({}, epoch);
    if (!load) { this._attach(this.ordinary, 'normal', epoch); return; }
    const source = args.caseHash ? await this._source(args.caseHash) : this.source;
    need(source, 'EXPLICIT_SOURCE_REQUIRED'); await this._sameSource(source); this._live(epoch);
    if (this.source?.caseHash === source.caseHash) need(source.path === this.source.path
      && source.sha256 === this.source.sha256, 'ORIGINAL_SOURCE_CHANGED');
    const loaded = await this._call(this.ordinary, 'loadCase', { path: source.path }, timeout, epoch);
    need(loaded?.caseHash === source.caseHash, 'ORDINARY_SOURCE_LOAD');
    await this._sameSource(source); this._live(epoch);
    if (this.source && this.source.caseHash !== source.caseHash) await this._discardRetained();
    this._live(epoch); this.source = source;
    // Internal reloads do not deliver a second source-load result or its fresh
    // asset IDs to the renderer. Explicit subsequent requests forward normally.
    this._attach(this.ordinary, 'normal', epoch);
  }

  async _normal(op, args, timeout, epoch) {
    const before = this.enabled && op === 'loadCase' ? await this.hashSource(args.path) : null;
    this._live(epoch);
    const result = await this._call(this.ordinary, op, args, timeout, epoch);
    this._live(epoch);
    if (op === 'loadCase') {
      if (before) {
        await this._sameSource(before); this._live(epoch);
        need(typeof result?.caseHash === 'string' && result.caseHash.length > 0, 'ORDINARY_SOURCE_LOAD');
      }
      await this._discardRetained(); this._live(epoch); this.source = null;
    }
    return result;
  }

  request(op, args = {}, timeout = 120000) {
    try { this._live(); } catch (error) { return Promise.reject(error); }
    if (!this.enabled && this.mode === 'normal') return this._normal(op, args, timeout, this.epoch);
    if (op === 'cancel' && this.mode !== 'normal') {
      return this._transition(async epoch => { await this._shutdown(); this._live(epoch); this.mode = 'cancelled'; return { cancelled: true }; }, true);
    }
    if (this.mode === 'pipeline') {
      if (op === 'ping') {
        const epoch = this.epoch;
        return this._call(this.consumer, op, args, timeout, epoch)
          .catch(error => { this._poison(error, epoch); throw error; });
      }
      if (op === 'surgicalRehearsal' && args.command === 'seekPrefixFrame'
        && args.caseHash === this.source.caseHash && args.branchId === this.pipeline.branchId) {
        const epoch = this.epoch;
        return this.pipeline.seekFrame(args, timeout).then(result => {
          const publicResult = this._withOrigin(result, epoch);
          this._publishTerminal(publicResult, epoch); return publicResult;
        });
      }
    }
    if (this.mode === 'normal' && !freshStart(op, args)) return this._normal(op, args, timeout, this.epoch);
    if (this.mode === 'normal') return this._transition(epoch => this._activate(args, timeout, epoch));
    if (this.mode === 'transition' && op !== 'loadCase' && !freshStart(op, args)) return Promise.reject(new Error('PREFIX_ENGINE:TRANSITION_BUSY'));
    if (this.mode === 'failed' && op === 'surgicalRehearsal' && args.command === 'seekPrefixFrame') return Promise.reject(this.failure || new Error('PREFIX_ENGINE:RESTART_REQUIRED'));
    return this._transition(async epoch => {
      await this._restoreOrdinary(args, timeout, epoch, op !== 'loadCase');
      if (freshStart(op, args)) return this._activate(args, timeout, epoch);
      const result = await this._normal(op, args, timeout, epoch); this._live(epoch);
      this.mode = 'normal'; this.failure = null; this.poisonNotified = false; return result;
    }, true);
  }

  _publishTerminal(result, epoch) {
    this._live(epoch);
    const terminal = this.terminalEvent; this.terminalEvent = null;
    if (terminal) {
      try { need(terminal.result === result || terminal.result?.view === result.view, 'TERMINAL_RESULT_IDENTITY'); }
      catch (error) { this._poison(error, epoch); throw error; }
      this.emit('event', { ...terminal, result });
    }
  }

  _withOrigin(result, epoch) {
    this._live(epoch);
    need(this.mode === 'pipeline' && this.prefixOrigin, 'ORIGIN_UNAVAILABLE');
    // This describes preparation provenance, never future state or authority.
    // Worker-provided origin fields cannot replace the trusted main descriptor.
    return {...result, prefixOrigin:{...this.prefixOrigin}};
  }

  stop() {
    if (this.stopPromise) return this.stopPromise;
    this.closed = true; this.mode = 'closed'; this._interrupt();
    this.stopPromise = (async () => {
      await this.tail;
      const settled = await Promise.allSettled([this._shutdown(), this._discardRetained()]);
      const errors = settled.filter(row => row.status === 'rejected').map(row => row.reason);
      if (errors.length) throw new AggregateError(errors, 'PREFIX_ENGINE:STOP_FAILED');
    })();
    this.stopPromise.catch(() => {}); return this.stopPromise;
  }
}

module.exports = { PrefixEngine, hashSource };
