'use strict';
// Qualification observer only: never schedules work or supplies replay data.
const { execFile } = require('node:child_process');
const { promisify } = require('node:util');
const { createHash } = require('node:crypto');
const path = require('node:path');
const { treeRss, MAX_RSS } = require('./prefix-resources.cjs');
const run = promisify(execFile);
const WALL_MS = 600000, ACTION_BYTES = 64 * 1024 ** 2, DISK_BYTES = 512 * 1024 ** 2;
const PROOF_BYTES = 512 * 1024;
const KEYS = ['schema', 'sceneId', 'originSnapshotHash', 'actionLogHash', 'protectionContractHash', 'implementation', 'prefixImplementation'];
const plain = value => value !== null && typeof value === 'object' && !Array.isArray(value)
  && Object.getPrototypeOf(value) === Object.prototype;
const integer = (value, low, high) => Number.isSafeInteger(value) && value >= low && value <= high;
const digest = value => typeof value === 'string' && /^(?:sha256:)?[a-f0-9]{64}$/.test(value);
const need = (ok, message) => { if (!ok) throw new Error(`VASCULAR_PIPELINE_PROOF:${message}`); };
const cacheKeys = ['mode', 'loadedActions', 'publishedActions', 'misses', 'lastSkip'];
const storeKeys = ['hits', 'publications', 'misses', 'lastSkip'];
const project = (value, keys) => Object.fromEntries(keys.map(key => [key, value[key]]));

class VascularPipelineProof {
  constructor({ archivePath, sourceArchiveSha256, branchId, frameCounts, cacheDirectory, onFailure,
    now = () => performance.now(), sample, _killPid = (pid, signal) => process.kill(pid, signal) }) {
    need(typeof archivePath === 'string' && path.isAbsolute(archivePath)
      && typeof cacheDirectory === 'string' && path.isAbsolute(cacheDirectory)
      && /^[a-f0-9]{64}$/.test(sourceArchiveSha256) && typeof branchId === 'string'
      && branchId.length > 0 && branchId.length <= 256, 'INVALID_SOURCE');
    need(Array.isArray(frameCounts) && integer(frameCounts.length, 1, 73)
      && frameCounts.every(count => integer(count, 1, 1149))
      && frameCounts.reduce((a, b) => a + b, 0) <= 1149, 'INVALID_COUNTS');
    need(typeof onFailure === 'function' && typeof now === 'function' && typeof _killPid === 'function', 'INVALID_CALLBACK');
    this.archivePath = path.resolve(archivePath); this.sourceSha = sourceArchiveSha256;
    this.cacheDirectory = path.resolve(cacheDirectory); this.branchId = branchId;
    this.frameCounts = [...frameCounts]; this.total = frameCounts.length;
    this.frames = frameCounts.reduce((a, b) => a + b, 0);
    this.onFailure = onFailure; this.now = now; this.killPid = _killPid;
    this.sample = sample || (() => run('/bin/ps', ['-axo', 'pid=,ppid=,rss='],
      { timeout: 1500, maxBuffer: 1024 * 1024 }).then(result => result.stdout));
    this.began = now(); need(Number.isFinite(this.began) && this.began >= 0, 'INVALID_CLOCK');
    this.lastTime = this.began; this.failure = null; this.epoch = 0;
    this.records = []; this.workers = new Map(); this.roles = new Map(); this.pending = new Set(); this.ids = new Set();
    this.caseHash = null; this.binding = null; this.storeLimits = null; this.ledger = [];
    this.prepared = 0; this.loaded = 0; this.delivered = 0;
    this.producerPending = false; this.consumerPending = 0; this.maxConsumerPending = 0;
    this.requestCursor = [0, 0]; this.resultCursor = [0, 0]; this.lastLoadEnd = null; this.lastPrepareEnd = null;
    this.maxAlive = 0; this.maxLookahead = 0; this.peak = 0; this.samples = 0; this.pings = 0;
    this.resourceStarted = false; this.resourceClosed = false; this.resourcePending = null; this.timer = null;
    this.closedWall = null; this.killPromise = null; this.finalized = false;
    this.deadline = setTimeout(() => this._fail(new Error('VASCULAR_PIPELINE_PROOF:COMMON_WALL_LIMIT')), WALL_MS);
    this.deadline.unref?.();
  }

  get mainPid() { return process.pid; }
  get statistics() {
    return { failure: this.failure?.message || null, workers: this.records.length,
      alive: this.records.filter(row => !row.exited).length, maxAlive: this.maxAlive,
      preparedActions: this.prepared, loadedActions: this.loaded, deliveredFrames: this.delivered,
      ledgerRows: this.ledger.length, unresolved: this.pending.size,
      maxConsumerUnresolved: this.maxConsumerPending, maxLookahead: this.maxLookahead,
      samples: this.samples, peakAggregatePythonTreeRssBytes: this.peak,
      resourceClosed: this.resourceClosed };
  }

  _clock() {
    const value = this.now();
    need(Number.isFinite(value) && value >= this.lastTime, 'CLOCK_REVERSED');
    this.lastTime = value; return value;
  }
  _live(epoch = this.epoch) {
    if (this.failure) throw this.failure;
    need(!this.resourceClosed && !this.finalized && epoch === this.epoch, 'CLOSED_OR_STALE');
    need(this._clock() - this.began < WALL_MS, 'COMMON_WALL_LIMIT');
  }
  _fail(error) {
    if (!this.failure) {
      this.failure = error instanceof Error ? error : new Error(String(error)); this.epoch++;
      clearTimeout(this.timer); clearTimeout(this.deadline);
      void this.killAll().catch(() => {});
      try { Promise.resolve(this.onFailure(this.failure)).catch(() => {}); } catch { /* already terminal */ }
    }
    return this.failure;
  }

  register(worker, config = {}) {
    try {
      this._live();
      const child = worker?.child, role = config.prefixWorker?.role || 'normal';
      need(child && integer(child.pid, 1, Number.MAX_SAFE_INTEGER) && child.pid !== process.pid
        && typeof child.on === 'function' && typeof child.kill === 'function', 'INVALID_CHILD');
      need(!this.workers.has(worker) && !this.roles.has(role) && this.records.length < 3
        && !this.records.some(row => row.pid === child.pid), 'DUPLICATE_WORKER');
      need(['normal', 'consumer', 'producer'].includes(role), 'INVALID_ROLE');
      if (role === 'normal') need(this.records.length === 0 && config.prefixWorker === undefined, 'NORMAL_ORDER');
      else {
        const normal = this.roles.get('normal'), selected = config.prefixWorker;
        need(normal?.exited && normal.prefixDone && this.binding && normal.exitCode === 0 && normal.signal === null, 'OLD_EXIT_REQUIRED');
        need(selected.sourcePath === this.archivePath && selected.sourceSha256 === this.sourceSha
          && selected.branchId === this.branchId, 'FOREIGN_ROLE_SOURCE');
      }
      const row = { worker, child, role, pid: child.pid, exited: false, exitCode: null, signal: null,
        loaded: false, loadPending: false, prefixDone: false, prefixPending: false, normalCalls: 0 };
      this.records.push(row); this.workers.set(worker, row); this.roles.set(role, row);
      this.maxAlive = Math.max(this.maxAlive, this.records.filter(entry => !entry.exited).length);
      need(this.maxAlive <= 2, 'TWO_WORKER_LIMIT');
      child.on('error', error => this._fail(error));
      child.on('exit', (code, signal) => {
        if (row.exited) { this._fail(new Error('VASCULAR_PIPELINE_PROOF:DUPLICATE_EXIT')); return; }
        row.exited = true; row.exitCode = code; row.signal = signal ?? null;
        if (code !== 0 || row.signal !== null || worker.stopping !== true)
          this._fail(new Error('VASCULAR_PIPELINE_PROOF:UNEXPECTED_WORKER_EXIT'));
      });
      return worker;
    } catch (error) {
      // A caller can discover an invalid new child only after it was spawned.
      // Kill that handle as well, without ever signalling the Electron PID.
      if (worker?.child?.pid !== process.pid) try { worker?.child?.kill('SIGKILL'); } catch { /* failure remains terminal */ }
      throw this._fail(error);
    }
  }

  _begin(row, op, args, began) {
    need(!row.exited && plain(args) && Number.isFinite(began) && began >= this.began
      && began <= this._clock(), 'INVALID_LEASE_START');
    if (op === 'ping') { need(++this.pings <= 128, 'STATUS_LIMIT'); return { row, kind: 'ping', began }; }
    if (op === 'loadCase') {
      need(!row.loaded && !row.loadPending && Object.keys(args).length === 1 && args.path === this.archivePath, 'EXACT_SOURCE_LOAD_REQUIRED');
      row.loadPending = true; return { row, kind: 'load', began };
    }
    need(op === 'surgicalRehearsal' || row.role === 'normal' && op === 'listRuns', 'UNEXPECTED_OPERATION');
    need(row.loaded && args.caseHash === this.caseHash, 'CASE_HANDLE');
    if (row.role === 'normal') {
      need(++row.normalCalls <= 16 && !row.prefixPending && !row.prefixDone, 'NORMAL_BOOTSTRAP_ONLY');
      if (op === 'listRuns') {
        need(Object.keys(args).length === 1, 'NORMAL_INVENTORY'); return { row, kind: 'inventory', began };
      }
      if (args.command === 'inspectArchive') {
        need(args.branchId === undefined || args.branchId === this.branchId, 'NORMAL_BRANCH');
        return { row, kind: 'inspect', began };
      }
    }
    need(args.branchId === this.branchId, 'BRANCH_CHANGED');
    if (args.command === 'seekPrefix') {
      need(!row.prefixDone && !row.prefixPending && args.position === 0
        && Object.keys(args).every(key => ['command', 'caseHash', 'branchId', 'position', 'cacheOnly'].includes(key)), 'PREFIX_ZERO_REQUIRED');
      if (row.role === 'consumer') need(args.cacheOnly === true && this.prepared === 1, 'CONSUMER_START_AFTER_PUBLICATION');
      else need(!Object.hasOwn(args, 'cacheOnly'), 'FRESH_START_REQUIRED');
      row.prefixPending = true; return { row, kind: 'zero', began };
    }
    need(row.role !== 'normal' && row.prefixDone && args.command === 'seekPrefixFrame'
      && integer(args.position, 0, this.total - 1) && integer(args.frameIndex, 0, this.frameCounts[args.position] - 1), 'SEQUENTIAL_ONLY');
    const index = args.position, frame = args.frameIndex;
    need(Object.keys(args).every(key => ['command', 'caseHash', 'branchId', 'position', 'frameIndex', 'prefixToken', 'prefixFieldTokens'].includes(key)), 'FRAME_ARGUMENTS');
    if (row.role === 'producer') {
      need(!this.producerPending && index === this.prepared && frame === 0 && index <= this.loaded
        && !Object.hasOwn(args, 'prefixToken') && !Object.hasOwn(args, 'prefixFieldTokens'), 'PRODUCER_LOOKAHEAD');
      if (index > 0) need(this.lastLoadEnd !== null && this.lastLoadEnd <= began, 'LOAD_BEFORE_NEXT_PRODUCER');
      this.producerPending = true;
      this.maxLookahead = Math.max(this.maxLookahead, index + 1 - this.loaded);
    } else {
      need(index === this.requestCursor[0] && frame === this.requestCursor[1] && this.consumerPending < 2, 'CONSUMER_SEQUENCE');
      if (frame === 0) need(this.loaded === index && this.prepared === index + 1 && !this.producerPending
        && this.lastPrepareEnd !== null && this.lastPrepareEnd <= began, 'STORE_OVERLAP');
      else need(this.loaded === index + 1, 'CONSUMER_RESIDENCY');
      this.consumerPending++; this.maxConsumerPending = Math.max(this.maxConsumerPending, this.consumerPending);
      this.requestCursor = frame + 1 === this.frameCounts[index] ? [index + 1, 0] : [index, frame + 1];
    }
    return { row, kind: 'frame', index, frame, began };
  }

  _normalBinding(result) {
    const verification = result.verification;
    const value = { schema: 'resectionlab.prepared-prefix-binding.v1', sceneId: result.view?.scene?.sceneId };
    for (const key of KEYS.slice(2)) value[key] = verification[key];
    need(KEYS.slice(1).every(key => digest(value[key])), 'NORMAL_BINDING');
    this.binding = Object.freeze(value);
  }

  _control(result, row, verified, action) {
    const v = result?.verification, meta = result?.prefixWorker;
    need(plain(result) && result.caseHash === this.caseHash && result.branchId === this.branchId
      && result.readOnly === true && result.changed === false && result.outcome === null && result.fullCase === null
      && result.budgetStopped === false && plain(result.view) && digest(result.view.stateId)
      && plain(v) && v.status === 'reconstructed-prefix' && v.readOnly === true && v.failure === null
      && v.verifiedThrough === verified && Number.isSafeInteger(v.verifiedThrough) && v.totalActions === this.total, 'READ_ONLY_CONTROL');
    need(plain(meta) && meta.schema === 'resectionlab.prefix-worker.v1' && meta.role === row.role
      && meta.sceneId === this.binding.sceneId && meta.sourceArchiveSha256 === this.sourceSha
      && plain(meta.binding) && Object.keys(meta.binding).length === KEYS.length
      && KEYS.every(key => Object.hasOwn(meta.binding, key) && meta.binding[key] === this.binding[key])
      && KEYS.slice(2).every(key => v[key] === this.binding[key])
      && meta.verifiedThrough === verified && Number.isSafeInteger(meta.verifiedThrough)
      && meta.totalActions === this.total, 'FOREIGN_CONTROL_BINDING');
    const producer = row.role === 'producer';
    const expected = { mode: producer ? 'prepare' : 'only', loadedActions: producer ? 0 : verified,
      publishedActions: producer ? verified : 0, misses: 0, lastSkip: null };
    for (const cache of [meta.cache, v.preparedCache]) need(plain(cache) && cacheKeys.every(key => cache[key] === expected[key])
      && ['loadedActions', 'publishedActions', 'misses'].every(key => Number.isSafeInteger(cache[key])), 'CACHE_COUNTERS');
    const store = meta.store;
    need(plain(store) && store.directory === this.cacheDirectory
      && store.namespaceDirectory === path.join(this.cacheDirectory, 'prepared-prefix-v1')
      && integer(store.maxDiskBytes, 1, DISK_BYTES) && store.maxActionBytes === ACTION_BYTES
      && integer(store.diskBytes, 0, store.maxDiskBytes) && store.hits === expected.loadedActions
      && store.publications === expected.publishedActions && store.misses === 0 && store.lastSkip === null
      && ['hits', 'publications', 'misses'].every(key => Number.isSafeInteger(store[key])), 'ACTUAL_STORE_WRITES_OR_LIMIT');
    const limits = project(store, ['directory', 'namespaceDirectory', 'maxDiskBytes', 'maxActionBytes']);
    if (!this.storeLimits) this.storeLimits = limits;
    need(Object.keys(limits).every(key => limits[key] === this.storeLimits[key]), 'SHARED_STORE_CHANGED');
    need(meta.maxStagedActionBytes === ACTION_BYTES && integer(meta.actionFrameBytes, action ? 1 : 0, action ? ACTION_BYTES : 0), 'STAGING_BYTES');
    return meta;
  }

  _returned(context, result, requestId) {
    const { row, kind } = context;
    if (kind === 'ping') return;
    if (kind === 'load') {
      need(plain(result) && digest(result.caseHash), 'LOAD_CASE_IDENTITY');
      if (row.role === 'normal') {
        need(this.caseHash === null && !Object.hasOwn(result, 'prefixWorker'), 'NORMAL_LOAD_CONTROL'); this.caseHash = result.caseHash;
      } else need(result.caseHash === this.caseHash && result.prefixWorker?.schema === 'resectionlab.prefix-worker.v1'
        && result.prefixWorker.role === row.role && result.prefixWorker.sourceArchiveSha256 === this.sourceSha, 'FOREIGN_ROLE_LOAD');
      row.loaded = true; row.loadPending = false; return;
    }
    if (kind === 'inventory') return;
    if (kind === 'inspect') {
      need(result?.caseHash === this.caseHash && result.changed === false && result.verification?.readOnly === true
        && result.verification.status === 'unverified-import' && result.verification.canonical === false
        && result.verification.physicsActionsExecuted === 0, 'IMPORTED_AUTHORITY'); return;
    }
    if (kind === 'zero' && row.role === 'normal') {
      const v = result?.verification;
      need(plain(result) && result.caseHash === this.caseHash && result.branchId === this.branchId && result.position === 0
        && result.readOnly === true && result.changed === false && result.outcome === null && result.fullCase === null
        && result.budgetStopped === false && v?.status === 'reconstructed-prefix' && v.readOnly === true
        && v.verifiedThrough === 0 && v.totalActions === this.total && v.failure === null
        && !Object.hasOwn(result, 'prefixWorker'), 'NORMAL_PREFIX_ZERO');
      need(v.preparedCache?.mode === 'prepare' && v.preparedCache.loadedActions === 0
        && v.preparedCache.publishedActions === 0 && v.preparedCache.misses === 0 && v.preparedCache.lastSkip === null, 'NORMAL_ZERO_COUNTERS');
      this._normalBinding(result); row.prefixDone = true; row.prefixPending = false; return;
    }
    const verified = kind === 'zero' ? 0 : context.index + 1;
    const meta = this._control(result, row, verified, kind === 'frame');
    if (kind === 'zero') { need(result.position === 0 && !result.playback, 'ROLE_PREFIX_ZERO'); row.prefixDone = true; row.prefixPending = false; return; }
    const { index, frame, began } = context, playback = result.playback;
    need(plain(playback) && playback.branchId === this.branchId && playback.position === index && playback.actionIndex === index
      && playback.frameIndex === frame && playback.frameCount === this.frameCounts[index]
      && playback.readOnly === true && playback.frameStateId === result.view.stateId, 'FRAME_IDENTITY');
    const end = this._clock();
    if (row.role === 'producer') {
      need(this.producerPending && index === this.prepared, 'PRODUCER_COMPLETION_ORDER');
      this.producerPending = false; this.prepared++; this.lastPrepareEnd = end;
    } else {
      need(index === this.resultCursor[0] && frame === this.resultCursor[1], 'CONSUMER_COMPLETION_ORDER');
      this.resultCursor = frame + 1 === this.frameCounts[index] ? [index + 1, 0] : [index, frame + 1];
      this.consumerPending--; this.delivered++;
      if (frame === 0) { need(this.loaded === index, 'CONSUMER_LOAD_ORDER'); this.loaded++; this.lastLoadEnd = end; }
    }
    if (frame === 0) {
      need(this.ledger.length < 2 * this.total, 'LEDGER_CAP');
      this.ledger.push({ role: row.role, index, requestId, start: began, end, verifiedThrough: verified,
        cache: project(meta.cache, cacheKeys), store: project(meta.store, storeKeys), actionFrameBytes: meta.actionFrameBytes });
    }
  }

  observeLease(worker, op, args, lease, began = this.now()) {
    const response = Promise.resolve(lease?.promise); response.catch(() => {});
    let context, epoch, requestId;
    try {
      this._live(); epoch = this.epoch;
      const row = this.workers.get(worker);
      need(row && lease && typeof lease.id === 'string' && lease.id.length > 0 && lease.id.length <= 256
        && lease.promise && typeof lease.promise.then === 'function' && typeof lease.cancel === 'function', 'LEASE_API');
      requestId = lease.id;
      need(!this.ids.has(requestId) && this.ids.size < this.frames + this.total + 400, 'LEASE_ID_REUSE_OR_CAP');
      this.ids.add(requestId); context = this._begin(row, op, args, began);
      this.pending.add(context);
    } catch (error) {
      const failed = Promise.reject(this._fail(error)); failed.catch(() => {});
      return { get id() { return lease?.id; }, cancel: lease?.cancel, promise: failed };
    }
    const promise = response.then(result => {
      this._live(epoch); need(!context.row.exited, 'RESULT_AFTER_EXIT');
      this._returned(context, result, requestId); this._live(epoch); return result;
    }).catch(error => { throw this._fail(error); }).finally(() => this.pending.delete(context))
      .then(result => { this._live(epoch); return result; }).catch(error => { throw this._fail(error); });
    promise.catch(() => {});
    return { get id() { return lease.id; }, cancel: lease.cancel, promise };
  }

  _parse(text) {
    need(typeof text === 'string' && Buffer.byteLength(text) <= 1024 * 1024, 'RESOURCE_SAMPLE_SIZE');
    const rows = new Map();
    for (const line of text.trim().split('\n')) {
      const fields = line.trim().split(/\s+/).map(Number);
      need(fields.length === 3 && fields.every(value => Number.isSafeInteger(value) && value >= 0)
        && fields[0] > 0 && !rows.has(fields[0]), 'INVALID_RESOURCE_SAMPLE');
      rows.set(fields[0], { parent: fields[1], bytes: fields[2] * 1024 });
    }
    return rows;
  }

  checkResources() {
    if (this.resourcePending) return this.resourcePending;
    try { this._live(); } catch (error) { return Promise.reject(this._fail(error)); }
    const members = this.records.filter(row => !row.exited), epoch = this.epoch;
    this.resourcePending = (async () => {
      try {
        const text = await this.sample(), rows = this._parse(text);
        if (this.failure) throw this.failure;
        if (epoch !== this.epoch || this.resourceClosed) return;
        const roots = [];
        for (const row of members) {
          if (row.exited) continue; // membership changed while ps was in flight
          const sampled = rows.get(row.pid);
          if ((!sampled || sampled.parent !== process.pid) && row.worker.stopping === true) continue;
          need(sampled && sampled.parent === process.pid, 'LIVE_WORKER_MISSING_OR_FOREIGN_PARENT'); roots.push(row.pid);
        }
        const rss = roots.length ? treeRss(text, roots) : 0;
        this.samples++; this.peak = Math.max(this.peak, rss);
        need(rss <= MAX_RSS, 'AGGREGATE_3_GIB_LIMIT'); this._live(epoch);
      } catch (error) { throw this._fail(error); }
    })().finally(() => { this.resourcePending = null; });
    this.resourcePending.catch(() => {}); return this.resourcePending;
  }

  async startResources() {
    try { this._live(); need(!this.resourceStarted, 'RESOURCES_ALREADY_STARTED'); this.resourceStarted = true; await this.checkResources(); }
    catch (error) { throw this._fail(error); }
    const poll = () => {
      if (this.failure || this.resourceClosed) return;
      this.timer = setTimeout(() => this.checkResources().then(poll, () => {}), 250); this.timer.unref?.();
    };
    poll();
  }

  async closeResources() {
    if (this.resourceClosed) { if (this.failure) throw this.failure; return; }
    try {
      need(this.resourceStarted && this.records.length === 3 && this.records.every(row => row.exited), 'EXIT_BARRIER_REQUIRED');
      if (this.resourcePending) await this.resourcePending;
      this._live(); this.closedWall = this._clock() - this.began;
      this.resourceClosed = true; clearTimeout(this.timer); clearTimeout(this.deadline);
    } catch (error) { throw this._fail(error); }
  }

  killAll() {
    if (this.killPromise) return this.killPromise;
    // Enumerate descendants while their parent links still exist, then kill
    // leaves before roots. The production sample has a 1.5s timeout; even a
    // failed sample must still signal every known child handle in finally.
    const roots = this.records.filter(row => !row.exited);
    let sampling;
    try { sampling = Promise.resolve(this.sample()); } catch (error) { sampling = Promise.reject(error); }
    sampling.catch(() => {});
    let deadline;
    const bounded = Promise.race([sampling, new Promise((_, reject) => {
      deadline = setTimeout(() => reject(new Error('VASCULAR_PIPELINE_PROOF:KILL_SAMPLE_TIMEOUT')), 1500);
    })]);
    this.killPromise = bounded.then(text => {
      const rows = this._parse(text), owned = new Set();
      for (const row of roots) if (rows.get(row.pid)?.parent === process.pid) owned.add(row.pid);
      let changed = true;
      while (changed) {
        changed = false;
        for (const [pid, row] of rows) if (owned.has(row.parent) && !owned.has(pid)) { owned.add(pid); changed = true; }
      }
      for (const pid of [...owned].reverse()) if (pid !== process.pid && !roots.some(row => row.pid === pid)) {
        try { this.killPid(pid, 'SIGKILL'); } catch (error) { if (error.code !== 'ESRCH') throw error; }
      }
    }).finally(() => {
      clearTimeout(deadline);
      for (const row of roots) if (!row.exited) try { row.child.kill('SIGKILL'); } catch { /* failed qualification already terminal */ }
    });
    this.killPromise.catch(() => {}); return this.killPromise;
  }

  finalize(report, facade) {
    try {
      if (this.failure) throw this.failure;
      need(!this.finalized && this.resourceClosed && this.samples > 0 && this.peak > 0
        && this.records.length === 3 && ['normal', 'producer', 'consumer'].every(role => this.roles.has(role))
        && this.records.every(row => row.exited && row.exitCode === 0 && row.signal === null)
        && this.maxAlive === 2 && this.pending.size === 0, 'FINAL_EXIT_OR_RESOURCE_PROOF');
      need(this.prepared === this.total && this.loaded === this.total && this.delivered === this.frames
        && this.ledger.length === 2 * this.total && this.maxLookahead === 1 && !this.producerPending
        && this.consumerPending === 0 && this.maxConsumerPending <= 2, 'INCOMPLETE_PIPELINE');
      const p = facade?.pipeline;
      need(facade?.mode === 'closed' && facade.enabled === true && facade.activeWorkers === 0
        && facade.workerCreations === 3 && facade.maxWorkers === 2 && facade.ordinaryInflight === 0
        && facade.pendingTransitions === 0 && facade.failure === null && facade.exitFailure === null
        && p?.status === 'closed' && p.preparedActions === this.total && p.loadedActions === this.total
        && p.deliveredFrames === this.frames && p.repeatedFrames === 0 && p.fatalCount === 0 && p.pending === 0
        && p.maxPendingRequests === 2 && p.maxPending <= 2 && p.maxLookahead === 1
        && p.maxPendingWindowBytes === 2 * ACTION_BYTES && p.maxReservedActionBytes <= 2 * ACTION_BYTES,
      'FACADE_FINAL_STATISTICS');
      need(plain(report) && report.qualificationMode === 'fresh-pipeline'
        && ['playStarted', 'playEnded', 'playStartedRenderer', 'playEndedRenderer'].every(key => Number.isFinite(report[key]))
        && report.playStarted <= report.playEnded && report.playEnded - report.playStarted < WALL_MS
        && this.ledger.every(row => report.playStarted <= row.start && row.end <= report.playEnded), 'STARTUP_NOT_IN_CLICK_TIMING');
      const arrays = [report.allCompletedPlaybackIpc, report.rendererReceipts, report.submittedPlayback];
      need(arrays.every(rows => Array.isArray(rows) && rows.length === this.frames), 'RENDERER_TUPLE_COUNT');
      let ordinal = 0, lastReceived = -Infinity;
      for (let action = 0; action < this.total; action++) for (let frame = 0; frame < this.frameCounts[action]; frame++, ordinal++) {
        const [ipc, receipt, shown] = arrays.map(rows => rows[ordinal]);
        need([ipc, receipt, shown].every(row => plain(row) && row.branchId === this.branchId
          && row.position === action && row.frameIndex === frame && digest(row.stateId) && !row.unmatched)
          && receipt.stateId === ipc.stateId && shown.stateId === ipc.stateId
          && ipc.command === 'seekPrefixFrame' && !ipc.error && receipt.command === 'seekPrefixFrame'
          && receipt.notificationSerial === ordinal + 1 && shown.receiptNotificationSerial === ordinal + 1
          && Number.isFinite(receipt.receivedAt) && Number.isFinite(shown.firstSubmittedAt)
          && shown.rendererReceivedAt === receipt.receivedAt && receipt.receivedAt >= lastReceived
          && report.playStartedRenderer <= receipt.receivedAt && receipt.receivedAt <= shown.firstSubmittedAt
          && shown.firstSubmittedAt <= report.playEndedRenderer, 'EXACT_RENDERER_SEQUENCE_OR_CAUSALITY');
        lastReceived = receipt.receivedAt;
      }
      const tupleLists = arrays.map(rows => rows.map(row => [row.branchId, row.position, row.frameIndex, row.stateId]));
      const proof = { schema: 'resectionlab.vascular-pipeline-parent-proof.v1', passed: true,
        qualificationMode: 'fresh-pipeline', mainPid: process.pid, caseHash: this.caseHash,
        branchId: this.branchId, sourceArchiveSha256: this.sourceSha, binding: { ...this.binding },
        workers: this.records.map(row => project(row, ['role', 'pid', 'exited', 'exitCode', 'signal'])),
        resources: { maxAlive: this.maxAlive, peakAggregatePythonTreeRssBytes: this.peak,
          wallMs: this.closedWall, maxWallMs: WALL_MS, closed: true, failure: null },
        leases: this.ledger.map(row => ({ ...row, cache: { ...row.cache }, store: { ...row.store } })),
        rendererTupleSha256: createHash('sha256').update(JSON.stringify(tupleLists)).digest('hex') };
      need(Buffer.byteLength(JSON.stringify(proof, null, 2) + '\n') <= PROOF_BYTES, 'PARENT_PROOF_512_KIB_LIMIT');
      this.finalized = true; return proof;
    } catch (error) { throw this._fail(error); }
  }
}

module.exports = { VascularPipelineProof };
