'use strict';

// Opt-in scheduling only. Workers and their source/assets/resource watchdogs
// belong to the caller. Producer responses never become renderer responses.
const ACTION_BYTES = 64 * 1024 ** 2;
const DISK_BYTES = 512 * 1024 ** 2;
const BINDING_KEYS = ['schema', 'sceneId', 'originSnapshotHash', 'actionLogHash',
  'protectionContractHash', 'implementation', 'prefixImplementation'];
const DIGEST = /^(?:sha256:)?[0-9a-f]{64}$/;
const plain = value => value !== null && typeof value === 'object' && !Array.isArray(value)
  && Object.getPrototypeOf(value) === Object.prototype;
const integer = (value, min, max) => Number.isSafeInteger(value) && value >= min && value <= max;
const need = (ok, reason) => { if (!ok) throw new Error(`PREFIX_PIPELINE:${reason}`); };

function binding(value) {
  need(plain(value) && Object.keys(value).length === BINDING_KEYS.length
    && BINDING_KEYS.every(key => Object.hasOwn(value, key)), 'INVALID_BINDING');
  need(value.schema === 'resectionlab.prepared-prefix-binding.v1'
    && BINDING_KEYS.slice(1).every(key => typeof value[key] === 'string' && DIGEST.test(value[key])), 'INVALID_BINDING');
  return Object.freeze(Object.fromEntries(BINDING_KEYS.map(key => [key, value[key]])));
}

class PrefixPipeline {
  constructor({ producer, consumer, caseHash, branchId, expectedBinding, onFatal, now = () => Date.now() }) {
    need(producer !== consumer && [producer, consumer].every(worker => worker
      && typeof worker.requestLease === 'function' && typeof worker.stop === 'function'), 'INVALID_WORKERS');
    need(typeof caseHash === 'string' && caseHash.length > 0 && caseHash.length <= 1024
      && typeof branchId === 'string' && branchId.length > 0 && branchId.length <= 1024, 'INVALID_SOURCE');
    need(typeof onFatal === 'function' && typeof now === 'function', 'INVALID_CALLBACK');
    this.producer = producer; this.consumer = consumer;
    this.caseHash = caseHash; this.branchId = branchId;
    this.expectedBinding = binding(expectedBinding); this.onFatal = onFatal; this.now = now;
    this.status = 'new'; this.epoch = 0; this.leases = new Set(); this.queue = [];
    this.running = false; this.active = null; this.failure = null; this.stopPromise = null;
    this.total = null; this.sourceSha = null; this.store = null;
    this.storePublications = { producer: 0, consumer: 0 };
    this.storeFrontiers = { producer: 0, consumer: 0 };
    this.prepared = -1; this.loaded = -1; this.frame = -1; this.frameCount = 0;
    this.producing = null; this.ready = null; this.startPromise = null;
    this.counts = { producerRequests: 0, consumerRequests: 0, preparedActions: 0,
      loadedActions: 0, deliveredFrames: 0, repeatedFrames: 0, maxPending: 0,
      maxLookahead: 0, maxReservedActionBytes: 0, producerMilliseconds: 0,
      consumerMilliseconds: 0, queueMilliseconds: 0, fatalCount: 0 };
  }

  get statistics() {
    return { ...this.counts, status: this.status, preparedThrough: this.prepared + 1,
      loadedThrough: this.loaded + 1, pending: this.queue.length + Number(Boolean(this.active)),
      newPackagePublications: { ...this.storePublications },
      maxPendingRequests: 2, maxStagedActionBytes: ACTION_BYTES,
      maxPendingWindowBytes: 2 * ACTION_BYTES, maxDiskBytes: DISK_BYTES };
  }

  _live(epoch = this.epoch) {
    need(epoch === this.epoch && !['closed', 'failed'].includes(this.status),
      this.status === 'failed' ? 'FAILED' : 'CLOSED');
  }

  _timeout(timeout) { need(integer(timeout, 100, 300000), 'INVALID_TIMEOUT'); return timeout; }

  _stop() {
    if (!this.stopPromise) {
      this.stopPromise = Promise.allSettled([this.producer, this.consumer].map(worker => {
        try { return Promise.resolve(worker.stop()); } catch (error) { return Promise.reject(error); }
      })).then(results => {
        const failures = results.filter(result => result.status === 'rejected').map(result => result.reason);
        if (failures.length) throw new AggregateError(failures, 'PREFIX_PIPELINE:WORKER_STOP_FAILED');
      });
    }
    return this.stopPromise;
  }

  _invalidate(error, fatal) {
    if (['closed', 'failed'].includes(this.status)) return;
    this.failure = error; this.status = fatal ? 'failed' : 'closed'; this.epoch++;
    for (const lease of [...this.leases]) lease.cancel(error);
    for (const item of this.queue.splice(0)) { clearTimeout(item.timer); item.reject(error); }
    if (this.active) { clearTimeout(this.active.timer); this.active.reject(error); }
    this.ready = null;
    // Fatal/background shutdown has no awaiting caller. Keep its rejection
    // handled here, while close() still exposes the same failed exit proof.
    void this._stop().catch(() => {});
    if (fatal) {
      this.counts.fatalCount++;
      try { Promise.resolve(this.onFatal(error)).catch(() => {}); } catch { /* failure is already terminal */ }
    }
  }

  _fatal(error) {
    const actual = error instanceof Error ? error : new Error('PREFIX_PIPELINE:WORKER_FAILED');
    this._invalidate(actual, true);
    return actual;
  }

  close() {
    this._invalidate(new Error('PREFIX_PIPELINE:CLOSED'), false);
    return this._stop();
  }

  _lease(role, args, timeout) {
    const epoch = this.epoch; this._live(epoch);
    const worker = this[role], began = this.now();
    const lease = worker.requestLease('surgicalRehearsal', args, this._timeout(timeout));
    // Sidecar can reject before allocating an ID (for example after exit).
    // Observe that promise even if lease validation fails synchronously.
    const response = Promise.resolve(lease?.promise);
    response.catch(() => {});
    need(lease && (typeof lease.id === 'string' || Number.isSafeInteger(lease.id))
      && lease.promise && typeof lease.promise.then === 'function' && typeof lease.cancel === 'function', 'INVALID_LEASE');
    this.counts[`${role}Requests`]++;
    let rejectCancelled;
    const cancelled = new Promise((_, reject) => { rejectCancelled = reject; });
    const owned = { cancel: error => {
      rejectCancelled(error);
      try { Promise.resolve(lease.cancel()).catch(() => {}); } catch { /* still reject the owner */ }
    } };
    this.leases.add(owned);
    // Race attaches a rejection handler even when a cancelled worker resolves
    // late, so a dead epoch neither publishes output nor leaks a rejection.
    return Promise.race([response, cancelled]).then(result => {
      this._live(epoch); return result;
    }).finally(() => {
      this.leases.delete(owned);
      this.counts[`${role}Milliseconds`] += Math.max(0, this.now() - began);
    });
  }

  _result(role, result, verified, loaded, published, position, frameIndex) {
    need(plain(result) && result.caseHash === this.caseHash && result.branchId === this.branchId
      && result.readOnly === true && result.budgetStopped === false
      && result.changed === false && result.outcome === null && plain(result.view)
      && typeof result.view.stateId === 'string' && DIGEST.test(result.view.stateId), 'RESULT_SOURCE_OR_BUDGET');
    const meta = result.prefixWorker, verification = result.verification;
    need(plain(meta) && meta.schema === 'resectionlab.prefix-worker.v1' && meta.role === role, 'WORKER_ROLE');
    const actualBinding = binding(meta.binding);
    need(BINDING_KEYS.every(key => actualBinding[key] === this.expectedBinding[key]), 'FOREIGN_BINDING');
    need(meta.sceneId === this.expectedBinding.sceneId && typeof meta.sourceArchiveSha256 === 'string'
      && /^[0-9a-f]{64}$/.test(meta.sourceArchiveSha256), 'SOURCE_ARCHIVE');
    if (this.sourceSha === null) this.sourceSha = meta.sourceArchiveSha256;
    need(meta.sourceArchiveSha256 === this.sourceSha, 'SOURCE_ARCHIVE');
    need(integer(meta.totalActions, 1, 100000) && meta.verifiedThrough === verified, 'WORKER_FRONTIER');
    if (this.total === null) this.total = meta.totalActions;
    need(meta.totalActions === this.total, 'ACTION_COUNT_CHANGED');
    need(plain(verification) && verification.readOnly === true && verification.status === 'reconstructed-prefix'
      && verification.failure === null && verification.verifiedThrough === verified
      && verification.totalActions === this.total
      && BINDING_KEYS.slice(2).every(key => verification[key] === this.expectedBinding[key]), 'VERIFICATION');
    const cache = meta.cache, store = meta.store;
    need(plain(cache) && cache.mode === (role === 'producer' ? 'prepare' : 'only')
      && cache.loadedActions === loaded && cache.publishedActions === published
      && cache.misses === 0 && cache.lastSkip === null, 'CACHE_COUNTERS');
    need(plain(verification.preparedCache)
      && ['mode', 'loadedActions', 'publishedActions', 'misses', 'lastSkip']
        .every(key => verification.preparedCache[key] === cache[key]), 'CACHE_VERIFICATION');
    need(plain(store) && typeof store.directory === 'string' && store.directory.length > 0 && store.directory.length <= 4096
      && typeof store.namespaceDirectory === 'string' && store.namespaceDirectory.length > 0 && store.namespaceDirectory.length <= 4096
      && integer(store.maxDiskBytes, 1, DISK_BYTES) && store.maxActionBytes === ACTION_BYTES
      && integer(store.diskBytes, 0, store.maxDiskBytes) && store.hits === loaded
      && store.misses === 0 && integer(store.publications, 0, published) && store.lastSkip === null, 'STORE_COUNTERS_OR_CAPS');
    const advance = verified - this.storeFrontiers[role];
    const publicationDelta = store.publications - this.storePublications[role];
    need((advance === 0 || advance === 1) && (publicationDelta === 0
      || role === 'producer' && advance === 1 && publicationDelta === 1), 'STORE_PUBLICATION_DRIFT');
    const shared = [store.directory, store.namespaceDirectory, store.maxDiskBytes, store.maxActionBytes];
    if (this.store === null) this.store = shared;
    need(shared.every((value, index) => value === this.store[index]), 'FOREIGN_STORE');
    need(meta.maxStagedActionBytes === ACTION_BYTES
      && integer(meta.actionFrameBytes, 0, ACTION_BYTES)
      && (position === null ? meta.actionFrameBytes === 0 : meta.actionFrameBytes > 0), 'ACTION_BYTE_CAP');
    if (position !== null) {
      const playback = result.playback;
      need(plain(playback) && playback.branchId === this.branchId && playback.readOnly === true
        && playback.position === position && playback.actionIndex === position
        && playback.frameIndex === frameIndex && integer(playback.frameCount, 1, 100000)
        && frameIndex < playback.frameCount && plain(result.view)
        && typeof result.view.stateId === 'string' && DIGEST.test(result.view.stateId)
        && playback.frameStateId === result.view.stateId, 'FRAME_IDENTITY');
      this.storePublications[role] = store.publications; this.storeFrontiers[role] = verified;
      return playback.frameCount;
    }
    need(result.position === 0 && !result.playback, 'INITIAL_BOUNDARY');
    this.storePublications[role] = store.publications; this.storeFrontiers[role] = verified;
    return 0;
  }

  _args(command, extra = {}) {
    return { caseHash: this.caseHash, branchId: this.branchId, command, ...extra };
  }

  _prepare(index, timeout) {
    const epoch = this.epoch;
    need(!this.producing && !this.ready && index === this.prepared + 1 && index <= this.loaded + 1, 'LOOKAHEAD');
    this.counts.maxLookahead = Math.max(this.counts.maxLookahead, index - this.loaded);
    this.counts.maxReservedActionBytes = Math.max(this.counts.maxReservedActionBytes,
      ACTION_BYTES * (this.loaded >= 0 ? 2 : 1));
    const promise = this._lease('producer', this._args('seekPrefixFrame', { position: index, frameIndex: 0 }), timeout)
      .then(result => {
        this._live(epoch);
        const frameCount = this._result('producer', result, index + 1, 0, index + 1, index, 0);
        this.prepared = index; this.counts.preparedActions++;
        // Retain no future state, assets, timeline, tokens or producer result.
        this.ready = { index, frameCount };
      }).catch(error => { throw this._fatal(error); })
      .finally(() => { if (this.producing?.promise === promise) this.producing = null; });
    this.producing = { index, promise };
    promise.catch(() => {}); // background failures have a terminal owner
    return promise;
  }

  start(timeout = 120000) {
    try { this._live(); } catch (error) { return Promise.reject(error); }
    if (this.startPromise) return this.startPromise;
    try { this._live(); need(this.status === 'new', 'ALREADY_STARTED'); this._timeout(timeout); }
    catch (error) { return Promise.reject(error); }
    this.status = 'starting';
    const epoch = this.epoch;
    this.startPromise = (async () => {
      const initial = await this._lease('producer', this._args('seekPrefix', { position: 0 }), timeout);
      this._live(epoch);
      this._result('producer', initial, 0, 0, 0, null, null);
      await this._prepare(0, timeout);
      const result = await this._lease('consumer', this._args('seekPrefix', { position: 0, cacheOnly: true }), timeout);
      this._live(epoch);
      this._result('consumer', result, 0, 0, 0, null, null);
      this.status = 'ready';
      const { prefixWorker, ...publicResult } = result;
      return publicResult;
    })().catch(error => { throw this._fatal(error); });
    this.startPromise.catch(() => {});
    return this.startPromise;
  }

  _request(args) {
    need(plain(args) && args.command === 'seekPrefixFrame' && args.caseHash === this.caseHash
      && args.branchId === this.branchId && integer(args.position, 0, this.total - 1)
      && integer(args.frameIndex, 0, 99999), 'SEQUENTIAL_ONLY');
    need(Object.keys(args).every(key => ['caseHash', 'branchId', 'command', 'position', 'frameIndex',
      'prefixToken', 'prefixFieldTokens'].includes(key)), 'UNSUPPORTED_ARGUMENT');
    if (Object.hasOwn(args, 'prefixToken')) need(typeof args.prefixToken === 'string'
      && args.prefixToken.length >= 1 && args.prefixToken.length <= 1024, 'INVALID_ACK');
    if (Object.hasOwn(args, 'prefixFieldTokens')) need(plain(args.prefixFieldTokens)
      && Object.keys(args.prefixFieldTokens).every(key => key === 'flapComponents'
        && typeof args.prefixFieldTokens[key] === 'string' && /^[0-9a-f]{64}$/.test(args.prefixFieldTokens[key])), 'INVALID_ACK');
    return { ...args, ...(args.prefixFieldTokens ? { prefixFieldTokens: { ...args.prefixFieldTokens } } : {}) };
  }

  seekFrame(args, timeout = 120000) {
    try {
      this._live(); need(this.status === 'ready', 'NOT_STARTED'); this._timeout(timeout);
      // Two unresolved requests total: one active app request and one prefetch.
      need(this.queue.length + Number(Boolean(this.active)) < 2, 'QUEUE_FULL');
      args = this._request(args);
    } catch (error) {
      if (!String(error.message).endsWith(':QUEUE_FULL') && this.status === 'ready') this._fatal(error);
      return Promise.reject(error);
    }
    const promise = new Promise((resolve, reject) => {
      const item = { args, timeout, resolve, reject, began: this.now(), epoch: this.epoch };
      item.timer = setTimeout(() => this._fatal(new Error('PREFIX_PIPELINE:REQUEST_TIMEOUT')), timeout);
      this.queue.push(item);
      this.counts.maxPending = Math.max(this.counts.maxPending, this.queue.length + Number(Boolean(this.active)));
      void this._drain();
    });
    return promise;
  }

  async _drain() {
    if (this.running) return;
    this.running = true;
    try {
      while (this.queue.length && this.status === 'ready') {
        const item = this.queue.shift(); this.active = item;
        try {
          this._live(item.epoch);
          const { position, frameIndex } = item.args;
          const next = position === this.loaded + 1 && frameIndex === 0
            && (this.loaded === -1 || this.frame === this.frameCount - 1);
          const same = position === this.loaded
            && (frameIndex === this.frame || frameIndex === this.frame + 1) && frameIndex < this.frameCount;
          need(next || same, 'SEQUENTIAL_ONLY');
          if (next && this.producing) { need(this.producing.index === position, 'LOOKAHEAD'); await this.producing.promise; }
          this._live(item.epoch);
          if (next) need(this.ready?.index === position, 'PUBLICATION_REQUIRED');
          const remaining = Math.floor(item.timeout - Math.max(0, this.now() - item.began));
          need(remaining >= 100, 'REQUEST_TIMEOUT');
          this.counts.queueMilliseconds += Math.max(0, this.now() - item.began);
          const result = await this._lease('consumer', item.args, remaining);
          const count = this._result('consumer', result, (next ? position : this.loaded) + 1,
            (next ? position : this.loaded) + 1, 0, position, frameIndex);
          need(count === (next ? this.ready.frameCount : this.frameCount), 'FRAME_COUNT_CHANGED');
          this._live(item.epoch);
          if (next) {
            this.loaded = position; this.frameCount = count; this.ready = null; this.counts.loadedActions++;
          } else if (frameIndex === this.frame) this.counts.repeatedFrames++;
          this.frame = frameIndex; this.counts.deliveredFrames++;
          // STORE loading has finished before a new producer can take its lock.
          if (next && position + 1 < this.total) this._prepare(position + 1, item.timeout);
          const { prefixWorker, ...publicResult } = result;
          item.resolve(publicResult);
        } catch (error) { item.reject(this._fatal(error)); }
        finally { clearTimeout(item.timer); if (this.active === item) this.active = null; }
      }
    } finally { this.running = false; }
  }
}

module.exports = { PrefixPipeline };
