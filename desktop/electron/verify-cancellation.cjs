'use strict';

// Exercise real policy updates and cancellation across the frozen JSONL bridge.
// Only optimization and selection are requested; no final evaluation is run.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { createHash } = require('node:crypto');
const { Sidecar } = require('./sidecar.cjs');
const { verificationDirectory } = require('./verification-output.cjs');

async function main() {
  const repo = path.resolve(__dirname, '../..');
  const reportDir = verificationDirectory(repo);
  const packaged = process.argv.includes('--bundle');
  const suffix = packaged ? 'packaged' : 'frozen';
  const temporary = await fs.mkdtemp(path.join(os.tmpdir(), 'ressectionlab-cancellation-'));
  const runDir = path.join(temporary, 'runs');
  const resources = path.join(repo, 'desktop/release/RessectionLab-darwin-arm64/RessectionLab.app/Contents/Resources');
  const executable = packaged
    ? path.join(resources, 'research-engine/ressectionlab-engine')
    : path.join(repo, 'desktop/sidecar/ressectionlab-engine/ressectionlab-engine');
  const report = {
    schemaVersion: 1, frozen: true, packaged, passed: false, checks: {},
    verificationStartedUtc: new Date().toISOString(), executable,
    scope: 'synthetic_native_training_cancellation_restart_resume_selection', finalEvaluation: false,
    shape: [24, 24, 24], budgetSeconds: 10, seed: 11,
  };
  const started = performance.now();
  let engine;
  let phase = 'cancel';
  let cancelPromise;
  let cancelledRunId;
  let cancelledRequestId;
  let cancellationStarted;
  let cancelledTerminalEvents = 0;
  let observedGradientSteps;
  let diagnostics = '';

  async function fileIdentity(filename) {
    const contents = await fs.readFile(filename);
    return { path: await fs.realpath(filename), sha256: createHash('sha256').update(contents).digest('hex') };
  }

  async function buildIdentity() {
    const identity = { executable: await fileIdentity(executable), engineSource: null, renderer: null };
    // Read provenance from the exact target engine, never the current checkout
    // or a build-report.json that might describe a different build attempt.
    const manifestPath = path.join(path.dirname(executable), '_internal/build_info/BUILD_INPUT_MANIFEST.json');
    try {
      const contents = await fs.readFile(manifestPath);
      const manifest = JSON.parse(contents.toString('utf8'));
      identity.engineSource = {
        path: await fs.realpath(manifestPath),
        manifestSha256: createHash('sha256').update(contents).digest('hex'),
        sourceDigest: manifest.source_digest ?? manifest.sourceDigest ?? null,
        revision: manifest.revision ?? null,
        createdUtc: manifest.created_utc ?? manifest.createdUtc ?? null,
      };
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
    // The archive identifies the actual bundled renderer and Electron main
    // code. A standalone sidecar test makes no claim about a renderer build.
    if (packaged) identity.renderer = await fileIdentity(path.join(resources, 'app.asar'));
    return identity;
  }

  function observeEvent(event) {
    if (event.id === cancelledRequestId && event.event === 'cancelled') cancelledTerminalEvents += 1;
    if (phase !== 'cancel' || cancelPromise || event.op !== 'trainPatient' || event.event !== 'progress'
      || !(event.progress?.metrics?.gradient_steps >= 2)) return;
    observedGradientSteps = event.progress.metrics.gradient_steps;
    cancelledRunId = event.progress.runId;
    cancelledRequestId = event.id;
    cancellationStarted = performance.now();
    // Attach both handlers immediately: the cancellation acknowledgement and
    // the original training rejection are separate JSONL terminal messages.
    cancelPromise = engine.request('cancel', { requestId: event.id }).then(
      result => ({ result }), error => ({ error }),
    );
  }

  async function start() {
    const transferDir = await fs.mkdtemp(path.join(temporary, 'transfers-'));
    engine = new Sidecar({ executable, transferDir, runDir, cwd: os.tmpdir() });
    engine.on('event', observeEvent);
    engine.on('diagnostic', message => {
      diagnostics = (diagnostics + message).slice(-4000);
      process.stderr.write(message);
    });
    return engine.request('createSyntheticCase', { shape: report.shape });
  }

  async function readRun(runId) {
    const directory = path.join(runDir, runId);
    const manifest = JSON.parse(await fs.readFile(path.join(directory, 'bridge-run.json'), 'utf8'));
    assert.match(manifest.integrity, /^[a-f0-9]{64}$/, 'run must carry its local integrity signature');
    for (const [filename, field] of [['checkpoint.pt', 'checkpointSha256'], ['contract.json', 'contractSha256']]) {
      const bytes = await fs.readFile(path.join(directory, filename));
      assert.equal(createHash('sha256').update(bytes).digest('hex'), manifest[field], `${filename} must match its signed manifest`);
    }
    const training = JSON.parse(await fs.readFile(path.join(directory, 'result.json'), 'utf8'));
    return { manifest, training };
  }

  try {
    report.buildIdentity = await buildIdentity();
    report.executableSha256 = report.buildIdentity.executable.sha256;
    const source = await start();
    const trainingStarted = performance.now();
    await assert.rejects(
      engine.request('trainPatient', { caseHash: source.caseHash, budgetSeconds: report.budgetSeconds, seed: report.seed }, 120000),
      /Operation cancelled/,
    );
    assert.ok(cancelPromise, 'training must reach a saved update before cancellation');
    const acknowledgement = await cancelPromise;
    if (acknowledgement.error) throw acknowledgement.error;
    assert.equal(acknowledgement.result.cancelled, true);
    assert.equal(acknowledgement.result.requestId, cancelledRequestId);
    assert.equal(cancelledTerminalEvents, 1);
    report.cancellationAcknowledgementSeconds = (performance.now() - cancellationStarted) / 1000;
    report.cancelledTrainingSeconds = (performance.now() - trainingStarted) / 1000;
    report.observedGradientStepsBeforeCancel = observedGradientSteps;
    report.checks.trainingCancelledAfterRealUpdates = true;

    // The bridge serializes numerical work, so this request also waits for the
    // cancelled learner to finish persisting its signed checkpoint.
    const cancelledList = await engine.request('listRuns', { caseHash: source.caseHash });
    const cancelledRun = cancelledList.runs.find(run => run.runId === cancelledRunId);
    assert.ok(cancelledRun, 'cancelled run must remain visible after signature validation');
    assert.equal(cancelledRun.status, 'cancelled');
    assert.equal(cancelledRun.hasCheckpoint, true);
    assert.equal(cancelledRun.hasAcceptedReplay, false);
    const before = await readRun(cancelledRunId);
    assert.equal(before.manifest.status, 'cancelled');
    // The checkpoint snapshot can retain "running" if interruption occurred
    // after its atomic save. The signed bridge manifest owns terminal status.
    report.cancelledCheckpointSnapshotStatus = before.training.status;
    assert.ok(before.training.gradient_steps >= observedGradientSteps);
    assert.equal(before.training.actor_parameters_changed, true);
    report.cancelledGradientSteps = before.training.gradient_steps;
    report.cancelledOptimizationEnvironmentSteps = before.training.optimization_environment_steps;
    report.cancelledSelectionEnvironmentSteps = before.training.selection_environment_steps;
    report.checks.cancelledCheckpointSignedAndListed = true;

    phase = 'resume';
    const originalPid = engine.child.pid;
    await engine.stop();
    assert.deepEqual(await buildIdentity(), report.buildIdentity, 'target build changed before engine restart');
    const restored = await start();
    assert.notEqual(engine.child.pid, originalPid);
    assert.equal(restored.caseHash, source.caseHash);
    const persisted = await engine.request('listRuns', { caseHash: restored.caseHash });
    assert.ok(persisted.runs.some(run => run.runId === cancelledRunId && run.status === 'cancelled' && run.hasCheckpoint));
    report.checks.cancelledRunRestoredAfterEngineRestart = true;

    const resumeStarted = performance.now();
    const result = await engine.request('trainPatient', { caseHash: restored.caseHash, resumeRunId: cancelledRunId }, 120000);
    report.resumeSeconds = (performance.now() - resumeStarted) / 1000;
    assert.equal(result.runId, cancelledRunId);
    assert.equal(result.config.budgetSeconds, report.budgetSeconds);
    assert.equal(result.config.seed, report.seed);
    assert.ok(result.training.gradient_steps > before.training.gradient_steps, 'resume must perform additional gradient updates');
    assert.equal(result.training.actor_parameters_changed, true);
    assert.notEqual(result.training.latest_actor_hash, before.training.latest_actor_hash, 'resume must change actor weights again');
    assert.equal(result.training.initial_checkpoint_hash, before.training.initial_checkpoint_hash);
    assert.equal(result.training.replay_status, 'accepted_independent_geometry');
    assert.equal(result.training.role, 'selection');
    assert.equal(result.training.final_evaluation, false);
    assert.equal(result.training.replay.role, 'selection');
    assert.equal(result.training.replay.final_evaluation, false);
    assert.equal(result.training.replay.native_certificate.complete_tool_checked, true);
    assert.equal(result.clinicalDeficitProbability, null);
    report.checks.resumeContinuedActorUpdatesUnderOriginalContract = true;
    report.checks.independentSelectionAcceptedWithoutFinalEvaluation = true;

    const after = await readRun(cancelledRunId);
    assert.equal(after.manifest.config.budgetSeconds, before.manifest.config.budgetSeconds);
    assert.equal(after.manifest.contractSha256, before.manifest.contractSha256);
    assert.notEqual(after.manifest.checkpointSha256, before.manifest.checkpointSha256);
    const finalList = await engine.request('listRuns', { caseHash: restored.caseHash });
    const resumedRun = finalList.runs.find(run => run.runId === cancelledRunId);
    assert.equal(resumedRun.hasCheckpoint, true);
    assert.equal(resumedRun.hasAcceptedReplay, true);
    report.checks.resumedCheckpointSignedAndListed = true;
    assert.deepEqual(await buildIdentity(), report.buildIdentity, 'target build changed during verification');
    report.checks.runtimeBuildIdentityUnchanged = true;
    Object.assign(report, {
      passed: true, resumedGradientSteps: result.training.gradient_steps,
      additionalGradientSteps: result.training.gradient_steps - before.training.gradient_steps,
      resumedOptimizationEnvironmentSteps: result.training.optimization_environment_steps,
      resumedSelectionEnvironmentSteps: result.training.selection_environment_steps,
      finalStatus: result.status, selectionReplayStatus: result.training.replay_status,
      cancelledTerminalEvents,
    });
  } catch (error) {
    report.error = String(error.stack || error).slice(0, 4000);
    if (diagnostics) report.diagnostics = diagnostics;
    process.exitCode = 1;
  } finally {
    if (engine) await engine.stop();
    await fs.rm(temporary, { recursive: true, force: true });
    report.elapsedSeconds = (performance.now() - started) / 1000;
    const artifact = path.join(reportDir, `electron-cancellation-${suffix}.json`);
    await fs.mkdir(path.dirname(artifact), { recursive: true });
    await fs.writeFile(artifact, JSON.stringify(report, null, 2) + '\n');
    process.stdout.write(JSON.stringify(report) + '\n');
  }
}

main().catch(error => { process.stderr.write(error.stack + '\n'); process.exitCode = 1; });
