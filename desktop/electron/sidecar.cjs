'use strict';
const {familyAvailabilityRequest,familyRequest,checkedFamilyAvailability,validateFamilyResult}=require('./contact-family.cjs');
const {contactRequest,validateContactResult}=require('./public-contact.cjs');
const {comparisonRequest,validateComparisonResult}=require('./episode-comparison.cjs');
const {vascularRequest,validateVascularResult}=require('./episode-vascular.cjs');

const { spawn } = require('node:child_process');
const { randomUUID } = require('node:crypto');
const { EventEmitter } = require('node:events');
const path = require('node:path');
const fs = require('node:fs/promises');
const { episodeRequest, validateEpisodeResult } = require('./development-episode.cjs');
const { persistentPassiveRequest, validatePersistentPassiveV1Result } = require('./persistent-passive.cjs');
const { validateWorkspaceResult } = require('./workspace-session.cjs');
const { AssetRegistry } = require('./assets.cjs');
const { observedRequest, validateObservedEvent } = require('./observed-landmark-contract.cjs');
const {surgicalTransportLimitMs}=require('./surgical-timeout.cjs');

const {validateDiagnosticResult,diagnosticRequest}=require('./diagnostic-contract.cjs');
const MAX_RESPONSE_LINE_BYTES=32*1024*1024;
const REHEARSAL_TIMING_PREFIX = 'RESSECTIONLAB_REHEARSAL_TIMING ';
const REHEARSAL_TIMING_SCHEMA = 'resectionlab.rehearsal-timing.v1';
const TIMED_SIDE_OPERATIONS = new Set(['saveCase', 'loadCase', 'anatomySavePackage', 'anatomyOpen']);
const OPERATIONS = new Set(['compareSurgicalAnatomy','prepareSurgicalEnvironment','openBoundSurgicalReplay','compareBoundSurgicalReplays','inspectNativeWallSource','closeNativeWallSource','importSurgicalScene','surgicalRehearsal','createSurgicalFixture','createRehearsal','inspectRehearsal','stepRehearsal','compareRehearsal','anatomySavePackage','anatomySaveView','anatomyComponents','anatomyAttachedVesselReview','anatomyVesselGapReview','anatomyGraph','anatomyCreate','anatomyReconstruct','anatomyPlan','anatomyRemoval','anatomySection','anatomyLocalAlignment','anatomyCtaArterialCoverage','anatomyOpen','anatomyView','anatomyEdit','anatomySample','anatomyRegister','anatomyImport','anatomyInspectIntake','anatomyImportBatch','anatomyPrepare','anatomyExport','ping', 'publicContactFamilyAvailability', 'executePublicContactFamilyEpisode', 'executePublicSurfaceContactEpisode', 'inspectDevelopmentEpisodeComparison', 'executeDevelopmentEpisode', 'executePersistentPassiveEpisode', 'evaluateDevelopmentEpisodeVascular', 'createSyntheticCase', 'loadCase', 'importNifti', 'importDisplaySeries', 'importDiagnosticLayer', 'importStructuralEvidence', 'saveCase', 'generateRoutes', 'generateNativeRoutes', 'inspectRefinement', 'inspectAxisPlanning', 'inspectObservedLandmarkUpdate', 'cancel', 'inspectEvidence', 'operativeSourceSurfaces', 'trainPatient', 'nativeTraining', 'listRuns', 'replayTraining', 'exportCandidate', 'shutdown']);

function timingText(value) {
  return typeof value === 'string' && value.length <= 128 ? value : undefined;
}

class Sidecar extends EventEmitter {
  constructor({ python, cwd, sourcePath, transferDir, executable, runDir, observedSourceRoot, prefixWorker, preparedPrefixCacheDirectory }) {
    super();
    this.assets = new AssetRegistry(transferDir);
    this.pending = new Map();
    this.buffer = '';
    this.processing = Promise.resolve();
    this.transferDir = transferDir;
    this.closed = false;
    this.stopping = false;
    this.rehearsalTiming = process.argv.includes('--automation-hidden') && process.env.RESSECTIONLAB_REHEARSAL_TIMING === '1';
    const env = { ...process.env, PYTHONUNBUFFERED: '1' };
    delete env.PYTHONHOME;
    delete env.QT_PLUGIN_PATH;
    if (sourcePath) env.PYTHONPATH = sourcePath; else delete env.PYTHONPATH;
    if (preparedPrefixCacheDirectory !== undefined) {
      if (typeof preparedPrefixCacheDirectory !== 'string' || !path.isAbsolute(preparedPrefixCacheDirectory)) throw new Error('Prepared prefix directory must be absolute');
      env.RESSECTIONLAB_PREPARED_PREFIX_CACHE_DIRECTORY = preparedPrefixCacheDirectory;
    }
    const command = executable || python;
    const args = executable ? ['--transfer-dir', transferDir] : ['-u', '-m', 'resectionlab.desktop_bridge', '--transfer-dir', transferDir];
    if (this.rehearsalTiming) args.push('--rehearsal-timing');
    if (runDir) args.push('--run-dir', runDir);
    if (observedSourceRoot && !executable) args.push('--observed-source-root', observedSourceRoot);
    if (prefixWorker !== undefined) {
      if (executable || !prefixWorker || !['producer','consumer'].includes(prefixWorker.role)
        || typeof prefixWorker.sourcePath !== 'string' || !path.isAbsolute(prefixWorker.sourcePath)
        || !/^[a-f0-9]{64}$/.test(prefixWorker.sourceSha256)
        || typeof prefixWorker.branchId !== 'string' || !prefixWorker.branchId.trim() || prefixWorker.branchId.includes('\0') || prefixWorker.branchId.length > 128) throw new Error('Invalid trusted prefix worker configuration');
      args.push('--prefix-worker-role',prefixWorker.role,'--prefix-worker-source',prefixWorker.sourcePath,
        '--prefix-worker-sha256',prefixWorker.sourceSha256,'--prefix-worker-branch',prefixWorker.branchId);
    }
    this.child = spawn(command, args, { cwd, env, stdio: ['pipe', 'pipe', 'pipe'], shell: false });
    this.child.stdin.on('error', error => { if (!this.closed) this.failAll(error); });
    this.child.stdout.setEncoding('utf8');
    this.child.stdout.on('data', chunk => this.consume(chunk));
    if (this.rehearsalTiming) {
      this.diagnosticBuffer = '';
      this.diagnosticOverflow = false;
      this.child.stderr.setEncoding('utf8');
      this.child.stderr.on('data', chunk => this.consumeDiagnostics(chunk));
    } else {
      this.child.stderr.on('data', chunk => this.emit('diagnostic', String(chunk).slice(0, 16000)));
    }
    this.child.on('error', error => this.failAll(error));
    this.child.on('exit', (code, signal) => this.failAll(new Error(`Local research engine exited (${signal || code}); restart RessectionLab to retry.`)));
  }

  request(op, args = {}, timeoutMs = 120000, lease = null) {
    const requestStartedNs = this.rehearsalTiming ? process.hrtime.bigint() : null;
    if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 100 || timeoutMs > surgicalTransportLimitMs(op,args)) return Promise.reject(new Error('Invalid operation time budget'));
    if (!OPERATIONS.has(op)) return Promise.reject(new Error('Unsupported research operation'));
    if (this.closed || this.stopping) return Promise.reject(new Error('Local research engine is not running'));
    if (!args || typeof args !== 'object' || Array.isArray(args)) return Promise.reject(new Error('Operation arguments must be an object'));
    if (op === 'importDiagnosticLayer') {
      try { const {descriptorPath,...identity}=args; const checked=diagnosticRequest(identity);
        if(typeof descriptorPath!=='string'||!descriptorPath||descriptorPath.length>4096)throw Error('Choose a local diagnostic descriptor');
        args={...checked,descriptorPath};
      } catch(error) { return Promise.reject(error); }
    }
    if (op === 'inspectObservedLandmarkUpdate') {
      try { args = observedRequest(args); } catch (error) { return Promise.reject(error); }
    }
    if (op === 'publicContactFamilyAvailability' || op === 'executePublicContactFamilyEpisode') {
      try {args=op==='publicContactFamilyAvailability'?familyAvailabilityRequest(args):familyRequest(args,this.contactFamilyAvailability);}catch(error){return Promise.reject(error);}
    }
    if (op === 'executePublicSurfaceContactEpisode') {
      try { args=contactRequest(args); } catch(error) { return Promise.reject(error); }
    }
    if (op === 'inspectDevelopmentEpisodeComparison') {
      try { args=comparisonRequest(args); } catch(error) { return Promise.reject(error); }
    }
    if (op === 'executeDevelopmentEpisode') {
      try { args = episodeRequest(args); } catch (error) { return Promise.reject(error); }
    }
    if (op === 'executePersistentPassiveEpisode') {
      try { args = persistentPassiveRequest(args); } catch (error) { return Promise.reject(error); }
    }
    if (op === 'evaluateDevelopmentEpisodeVascular') {
      try { args=vascularRequest(args); } catch(error) { return Promise.reject(error); }
    }
    // Owned identities let the renderer cancel a specific run before its first
    // event. Transport metadata must not enter the Python operation arguments.
    const requestPrefix = op === 'surgicalRehearsal' ? 'surgical' :
      op === 'anatomyPrepare' && args.action === 'run' ? 'prepare' :
      op === 'inspectNativeWallSource' ? 'wall-source' :
      op === 'anatomyView' ? 'wall-source-view' : null;
    let id = randomUUID();
    if (requestPrefix && Object.hasOwn(args, 'requestId')) {
      const {requestId, ...payload} = args;
      const valid = typeof requestId === 'string' && requestId.startsWith(requestPrefix + '-') &&
        /^[a-f0-9-]{36}$/.test(requestId.slice(requestPrefix.length + 1)) && !this.pending.has(requestId);
      if (!valid) return Promise.reject(new Error('Invalid ' + requestPrefix + ' request identifier'));
      id = requestId; args = payload;
    }
    const command = requestStartedNs === null ? undefined : op === 'surgicalRehearsal'
      ? timingText(Object.hasOwn(args, 'command') ? args.command : 'inspect') : TIMED_SIDE_OPERATIONS.has(op) ? op : undefined;
    const encodingStartedNs = command === undefined ? null : process.hrtime.bigint();
    const message = JSON.stringify({ id, op, args, timeoutMs });
    const encodeRequest = encodingStartedNs === null ? undefined : Number(process.hrtime.bigint() - encodingStartedNs);
    if (Buffer.byteLength(message) > 1024 * 1024) return Promise.reject(new Error('Operation request is too large'));
    if (lease) lease.id = id;
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(id);
        this.child.stdin.write(JSON.stringify({ id: randomUUID(), op: 'cancel', args: { requestId: id } }) + '\n');
        reject(new Error('Operation exceeded its local time budget'));
      }, timeoutMs + 5000);
      const timing = command === undefined ? null : { command, requestStartedNs, encodeRequest,
        caseHash: timingText(args.caseHash), expectedStateId: timingText(args.expectedStateId),
        actionOperation: timingText(args.action?.operation), actionSlot: timingText(args.action?.slot) };
      this.pending.set(id, { resolve, reject, timeout, op, timing,
        ...(op === 'importDiagnosticLayer' ? {diagnosticArgs:{caseHash:args.caseHash,seriesId:args.seriesId}} : {}),
        ...(op === 'evaluateDevelopmentEpisodeVascular' ? { vascularArgs: args } : {}),
        ...(op === 'executePublicContactFamilyEpisode' ? {familyArgs:args,familyCatalog:structuredClone(this.contactFamilyAvailability)} : {}),
        ...(op === 'executePublicSurfaceContactEpisode' ? { contactArgs: args } : {}),
        ...(op === 'inspectDevelopmentEpisodeComparison' ? { comparisonArgs: args } : {}),
        ...(op === 'executeDevelopmentEpisode' ? { episodeArgs: args } : {}),
        ...(op === 'inspectObservedLandmarkUpdate' ? { observedArgs: args } : {}) });
      if (timing) timing.sentNs = process.hrtime.bigint();
      this.child.stdin.write(message + '\n', error => { if (error) this.failAll(error); });
    });
  }

  requestLease(op, args = {}, timeoutMs = 120000) {
    const state = {id:null};
    const promise = this.request(op,args,timeoutMs,state);
    return {get id(){return state.id;},promise,
      cancel:()=>state.id && !this.closed && !this.stopping
        ? this.request('cancel',{requestId:state.id},30000) : Promise.resolve({cancelled:false})};
  }

  consumeDiagnostics(chunk) {
    this.diagnosticBuffer += chunk;
    let newline;
    while ((newline = this.diagnosticBuffer.indexOf('\n')) >= 0) {
      const line = this.diagnosticBuffer.slice(0, newline);
      this.diagnosticBuffer = this.diagnosticBuffer.slice(newline + 1);
      if (!this.diagnosticOverflow && line) {
        let row;
        if (line.length <= REHEARSAL_TIMING_PREFIX.length + 4096 && line.startsWith(REHEARSAL_TIMING_PREFIX)) {
          try { row = JSON.parse(line.slice(REHEARSAL_TIMING_PREFIX.length)); } catch { /* ordinary diagnostic */ }
        }
        if (row?.schema === REHEARSAL_TIMING_SCHEMA && row.process === 'python') this.emit('rehearsalTiming', row);
        else this.emit('diagnostic', line.slice(0, 16000));
      }
      this.diagnosticOverflow = false;
    }
    if (this.diagnosticBuffer.length > 16000) {
      if (!this.diagnosticOverflow) this.emit('diagnostic', this.diagnosticBuffer.slice(0, 16000));
      this.diagnosticBuffer = '';
      this.diagnosticOverflow = true;
    }
  }

  consume(chunk) {
    if(this.closed)return;
    this.buffer += chunk;
    const exceeded=()=>{this.buffer='';this.child.kill('SIGTERM');this.failAll(new Error('Local engine response exceeded its 32 MiB protocol limit'));};
    let newline;
    while ((newline = this.buffer.indexOf('\n')) >= 0) {
      const line = this.buffer.slice(0, newline); this.buffer = this.buffer.slice(newline + 1);
      const byteCount = Buffer.byteLength(line, 'utf8');
      if(byteCount>MAX_RESPONSE_LINE_BYTES){exceeded();return;}
      if (!line.trim()) continue;
      const lineReadyNs = this.rehearsalTiming ? process.hrtime.bigint() : null;
      this.processing = this.processing.then(() => {
        if (lineReadyNs === null) return this.handle(JSON.parse(line));
        const parseStartedNs = process.hrtime.bigint();
        const message = JSON.parse(line);
        return this.handle(message, {lineReadyNs, parseStartedNs, parsedNs: process.hrtime.bigint(), responseBytes: byteCount + 1});
      }).catch(error => {
        this.failAll(error); this.child.kill('SIGTERM');
      });
    }
    if(Buffer.byteLength(this.buffer,'utf8')>MAX_RESPONSE_LINE_BYTES)exceeded();
  }

  async handle(message, transport = null) {
    const pending = this.pending.get(message.id);
    if (!pending) return;
    if (!['started', 'progress', 'result', 'error', 'cancelled'].includes(message.event)) throw new Error('Unknown local engine event');
    const terminal = ['result', 'error', 'cancelled'].includes(message.event);
    const timing = terminal && transport ? pending.timing : null;
    let exposeAssets;
    const observed = pending.op === 'inspectObservedLandmarkUpdate';
    // Validate the complete observation envelope and payload before any asset
    // traversal or event forwarding, not just after the request resolves.
    if (observed) validateObservedEvent(message, pending.observedArgs);
    if (message.event === 'result') {
      if (pending.op === 'importDiagnosticLayer') validateDiagnosticResult(message.result,pending.diagnosticArgs);
      if (pending.op === 'evaluateDevelopmentEpisodeVascular') validateVascularResult(message.result,pending.vascularArgs);
      if (pending.op === 'publicContactFamilyAvailability') this.contactFamilyAvailability=checkedFamilyAvailability(message.result);
      if (pending.op === 'executePublicContactFamilyEpisode') validateFamilyResult(message.result,pending.familyArgs,pending.familyCatalog);
      if (pending.op === 'executePublicSurfaceContactEpisode') validateContactResult(message.result,pending.contactArgs);
      if (pending.op === 'inspectDevelopmentEpisodeComparison') validateComparisonResult(message.result,pending.comparisonArgs);
      if (pending.op === 'executeDevelopmentEpisode') validateEpisodeResult(message.result, pending.episodeArgs);
      if (pending.op === 'executePersistentPassiveEpisode') validatePersistentPassiveV1Result(message.result);
      if (pending.op === 'loadCase') validateWorkspaceResult(message.result);
      if (['loadCase', 'importNifti', 'importStructuralEvidence', 'createSyntheticCase', 'executeDevelopmentEpisode', 'executePersistentPassiveEpisode', 'executePublicSurfaceContactEpisode', 'executePublicContactFamilyEpisode'].includes(pending.op)) this.assets.clear();
      // Source-surface arrays are re-read by the main process from the checked asset; keep digests, drop disk paths.
      if (pending.op === 'operativeSourceSurfaces' && message.result?.arrays) message.result = {...message.result,
        arrays: Object.fromEntries(Object.entries(message.result.arrays).map(([key, {path: _path, ...descriptor}]) => [key, descriptor]))};
      else if (!observed) {
        const startedNs = timing ? process.hrtime.bigint() : null;
        message.result = await this.assets.expose(message.result);
        if (timing) exposeAssets = Number(process.hrtime.bigint() - startedNs);
      }
    }
    const forwardStartedNs = timing ? process.hrtime.bigint() : null;
    this.emit('event', { ...message, op: pending.op });
    const forwardEvent = timing ? Number(process.hrtime.bigint() - forwardStartedNs) : undefined;
    if (terminal) {
      clearTimeout(pending.timeout); this.pending.delete(message.id);
      if (message.event === 'result') pending.resolve(message.result);
      else pending.reject(new Error(message.event === 'cancelled' ? 'Operation cancelled' : `${message.error?.code || 'ENGINE_ERROR'}: ${message.error?.message || 'Local research operation failed'}`));
    }
    if (timing) {
      const result = message.result;
      const outcome = timing.command === 'preview' ? result : result?.outcome;
      this.emit('rehearsalTiming', { schema: REHEARSAL_TIMING_SCHEMA, process: 'electron',
        requestId: message.id, command: timing.command, terminalEvent: message.event,
        caseHash: timing.caseHash, expectedStateId: timing.expectedStateId,
        actionOperation: timing.actionOperation, actionSlot: timing.actionSlot,
        branchId: timingText(result?.branchId), position: result?.position,
        responseStateId: timingText(result?.view?.stateId), status: timingText(outcome?.status),
        reasonCode: timingText(outcome?.reasonCode ?? message.error?.code), responseBytes: transport.responseBytes,
        requestSentMainMonotonicNs: timing.sentNs.toString(),
        responseLineReadyMainMonotonicNs: transport.lineReadyNs.toString(),
        durationsNs: { encodeRequest: timing.encodeRequest,
          requestToResponseLine: Number(transport.lineReadyNs - timing.sentNs),
          responseParseQueue: Number(transport.parseStartedNs - transport.lineReadyNs),
          responseParse: Number(transport.parsedNs - transport.parseStartedNs),
          exposeAssets, forwardEvent, requestToSidecarTerminal: Number(process.hrtime.bigint() - timing.requestStartedNs) },
      });
    }
  }

  failAll(error) {
    this.closed = true;
    for (const pending of this.pending.values()) { clearTimeout(pending.timeout); pending.reject(error); }
    this.pending.clear();
    this.emit('event', { event: 'engineStopped', error: { code: 'ENGINE_STOPPED', message: error.message } });
  }

  async stop() {
    if (this.stopPromise) {
      await this.stopPromise;
      return;
    }
    this.stopping = true;
    this.stopPromise = (async()=>{
      // A timeout/cancel reply can precede numerical worker completion. Do not
      // start its replacement or delete its files until the process has exited.
      if (this.child.pid && this.child.exitCode === null && this.child.signalCode === null) {
        await new Promise((resolve,reject)=>{
          let term,kill,deadline;
          const finished=()=>{clearTimeout(term);clearTimeout(kill);clearTimeout(deadline);resolve();};
          this.child.once('exit',finished);
          term=setTimeout(()=>this.child.kill('SIGTERM'),1500);
          kill=setTimeout(()=>this.child.kill('SIGKILL'),3000);
          deadline=setTimeout(()=>{this.child.removeListener('exit',finished);reject(new Error('Local research engine did not exit; replacement refused'));},5000);
          if (!this.closed) {
            this.child.stdin.write(JSON.stringify({id:randomUUID(),op:'shutdown',args:{}})+'\n',()=>{});
            this.child.stdin.end();
          }
        });
      }
      await fs.rm(this.transferDir,{recursive:true,force:true});
    })();
    return this.stopPromise;
  }
}

module.exports = { Sidecar, OPERATIONS, MAX_RESPONSE_LINE_BYTES };
