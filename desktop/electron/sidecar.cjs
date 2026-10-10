'use strict';
const {contactRequest,validateContactResult}=require('./public-contact.cjs');
const {comparisonRequest,validateComparisonResult}=require('./episode-comparison.cjs');
const {vascularRequest,validateVascularResult}=require('./episode-vascular.cjs');

const { spawn } = require('node:child_process');
const { randomUUID } = require('node:crypto');
const { EventEmitter } = require('node:events');
const path = require('node:path');
const fs = require('node:fs/promises');
const { episodeRequest, validateEpisodeResult } = require('./development-episode.cjs');
const { validateWorkspaceResult } = require('./workspace-session.cjs');
const { AssetRegistry } = require('./assets.cjs');
const { observedRequest, validateObservedEvent } = require('./observed-landmark-contract.cjs');

const OPERATIONS = new Set(['ping', 'executePublicSurfaceContactEpisode', 'inspectDevelopmentEpisodeComparison', 'executeDevelopmentEpisode', 'evaluateDevelopmentEpisodeVascular', 'createSyntheticCase', 'loadCase', 'importNifti', 'importDisplaySeries', 'importStructuralEvidence', 'saveCase', 'generateRoutes', 'generateNativeRoutes', 'inspectRefinement', 'inspectAxisPlanning', 'inspectObservedLandmarkUpdate', 'cancel', 'inspectEvidence', 'trainPatient', 'nativeTraining', 'listRuns', 'replayTraining', 'exportCandidate', 'shutdown']);

class Sidecar extends EventEmitter {
  constructor({ python, cwd, sourcePath, transferDir, executable, runDir, observedSourceRoot }) {
    super();
    this.assets = new AssetRegistry(transferDir);
    this.pending = new Map();
    this.buffer = '';
    this.processing = Promise.resolve();
    this.transferDir = transferDir;
    this.closed = false;
    const env = { ...process.env, PYTHONUNBUFFERED: '1' };
    delete env.PYTHONHOME;
    delete env.QT_PLUGIN_PATH;
    if (sourcePath) env.PYTHONPATH = sourcePath; else delete env.PYTHONPATH;
    const command = executable || python;
    const args = executable ? ['--transfer-dir', transferDir] : ['-u', '-m', 'resectionlab.desktop_bridge', '--transfer-dir', transferDir];
    if (runDir) args.push('--run-dir', runDir);
    if (observedSourceRoot && !executable) args.push('--observed-source-root', observedSourceRoot);
    this.child = spawn(command, args, { cwd, env, stdio: ['pipe', 'pipe', 'pipe'], shell: false });
    this.child.stdin.on('error', error => { if (!this.closed) this.failAll(error); });
    this.child.stdout.setEncoding('utf8');
    this.child.stdout.on('data', chunk => this.consume(chunk));
    this.child.stderr.on('data', chunk => this.emit('diagnostic', String(chunk).slice(0, 16000)));
    this.child.on('error', error => this.failAll(error));
    this.child.on('exit', (code, signal) => this.failAll(new Error(`Local research engine exited (${signal || code}); restart RessectionLab to retry.`)));
  }

  request(op, args = {}, timeoutMs = 120000) {
    if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 100 || timeoutMs > 300000) return Promise.reject(new Error('Invalid operation time budget'));
    if (!OPERATIONS.has(op)) return Promise.reject(new Error('Unsupported research operation'));
    if (this.closed) return Promise.reject(new Error('Local research engine is not running'));
    if (!args || typeof args !== 'object' || Array.isArray(args)) return Promise.reject(new Error('Operation arguments must be an object'));
    if (op === 'inspectObservedLandmarkUpdate') {
      try { args = observedRequest(args); } catch (error) { return Promise.reject(error); }
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
    if (op === 'evaluateDevelopmentEpisodeVascular') {
      try { args=vascularRequest(args); } catch(error) { return Promise.reject(error); }
    }
    const id = randomUUID();
    const message = JSON.stringify({ id, op, args, timeoutMs });
    if (Buffer.byteLength(message) > 1024 * 1024) return Promise.reject(new Error('Operation request is too large'));
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(id);
        this.child.stdin.write(JSON.stringify({ id: randomUUID(), op: 'cancel', args: { requestId: id } }) + '\n');
        reject(new Error('Operation exceeded its local time budget'));
      }, timeoutMs + 5000);
      this.pending.set(id, { resolve, reject, timeout, op,
        ...(op === 'evaluateDevelopmentEpisodeVascular' ? { vascularArgs: args } : {}),
        ...(op === 'executePublicSurfaceContactEpisode' ? { contactArgs: args } : {}),
        ...(op === 'inspectDevelopmentEpisodeComparison' ? { comparisonArgs: args } : {}),
        ...(op === 'executeDevelopmentEpisode' ? { episodeArgs: args } : {}),
        ...(op === 'inspectObservedLandmarkUpdate' ? { observedArgs: args } : {}) });
      this.child.stdin.write(message + '\n', error => { if (error) this.failAll(error); });
    });
  }

  consume(chunk) {
    this.buffer += chunk;
    if (this.buffer.length > 8 * 1024 * 1024) { this.child.kill('SIGTERM'); this.failAll(new Error('Local engine response exceeded its protocol limit')); return; }
    let newline;
    while ((newline = this.buffer.indexOf('\n')) >= 0) {
      const line = this.buffer.slice(0, newline); this.buffer = this.buffer.slice(newline + 1);
      if (!line.trim()) continue;
      this.processing = this.processing.then(() => this.handle(JSON.parse(line))).catch(error => {
        this.failAll(error); this.child.kill('SIGTERM');
      });
    }
  }

  async handle(message) {
    const pending = this.pending.get(message.id);
    if (!pending) return;
    if (!['started', 'progress', 'result', 'error', 'cancelled'].includes(message.event)) throw new Error('Unknown local engine event');
    const observed = pending.op === 'inspectObservedLandmarkUpdate';
    // Validate the complete observation envelope and payload before any asset
    // traversal or event forwarding, not just after the request resolves.
    if (observed) validateObservedEvent(message, pending.observedArgs);
    if (message.event === 'result') {
      if (pending.op === 'evaluateDevelopmentEpisodeVascular') validateVascularResult(message.result,pending.vascularArgs);
      if (pending.op === 'executePublicSurfaceContactEpisode') validateContactResult(message.result,pending.contactArgs);
      if (pending.op === 'inspectDevelopmentEpisodeComparison') validateComparisonResult(message.result,pending.comparisonArgs);
      if (pending.op === 'executeDevelopmentEpisode') validateEpisodeResult(message.result, pending.episodeArgs);
      if (pending.op === 'loadCase') validateWorkspaceResult(message.result);
      if (['loadCase', 'importNifti', 'importStructuralEvidence', 'createSyntheticCase', 'executeDevelopmentEpisode', 'executePublicSurfaceContactEpisode'].includes(pending.op)) this.assets.clear();
      if (!observed) message.result = await this.assets.expose(message.result);
    }
    this.emit('event', { ...message, op: pending.op });
    if (['result', 'error', 'cancelled'].includes(message.event)) {
      clearTimeout(pending.timeout); this.pending.delete(message.id);
      if (message.event === 'result') pending.resolve(message.result);
      else pending.reject(new Error(message.event === 'cancelled' ? 'Operation cancelled' : `${message.error?.code || 'ENGINE_ERROR'}: ${message.error?.message || 'Local research operation failed'}`));
    }
  }

  failAll(error) {
    this.closed = true;
    for (const pending of this.pending.values()) { clearTimeout(pending.timeout); pending.reject(error); }
    this.pending.clear();
    this.emit('event', { event: 'engineStopped', error: { code: 'ENGINE_STOPPED', message: error.message } });
  }

  async stop() {
    if (!this.closed) {
      this.child.stdin.write(JSON.stringify({ id: randomUUID(), op: 'shutdown', args: {} }) + '\n');
      this.child.stdin.end();
      await new Promise(resolve => {
        const timeout = setTimeout(() => { this.child.kill('SIGTERM'); resolve(); }, 1500);
        this.child.once('exit', () => { clearTimeout(timeout); resolve(); });
      });
    }
    await fs.rm(this.transferDir, { recursive: true, force: true });
  }
}

module.exports = { Sidecar, OPERATIONS };
