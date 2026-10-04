'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { Sidecar } = require('./sidecar.cjs');

async function main() {
  const repo = path.resolve(__dirname, '../..');
  const packaged = process.argv.includes('--bundle');
  const frozen = process.argv.includes('--frozen') || packaged;
  const suffix = packaged ? 'packaged' : frozen ? 'frozen' : 'source';
  const temporary = await fs.mkdtemp(path.join(os.tmpdir(), 'ressectionlab-learning-'));
  const runDir = path.join(temporary, 'runs');
  const checks = {};
  const report = { frozen, packaged, checks, passed: false, scope: 'synthetic_native_selection_replay', finalEvaluation: false };
  const started = performance.now();
  let engine;
  async function start() {
    const transferDir = await fs.mkdtemp(path.join(temporary, 'transfers-'));
    const options = frozen ? { executable: path.join(repo, packaged
      ? 'desktop/release/RessectionLab-darwin-arm64/RessectionLab.app/Contents/Resources/research-engine/ressectionlab-engine'
      : 'desktop/sidecar/ressectionlab-engine/ressectionlab-engine') }
      : { python: path.join(repo, '.venv/bin/python'), sourcePath: path.join(repo, 'src') };
    engine = new Sidecar({ ...options, transferDir, runDir, cwd: os.tmpdir() });
    engine.on('diagnostic', message => process.stderr.write(message));
    return engine.request('createSyntheticCase', { shape: [24, 24, 24] });
  }
  try {
    const source = await start();
    const result = await engine.request('trainPatient', { caseHash: source.caseHash, budgetSeconds: 5, seed: 11 }, 120000);
    const training = result.training;
    assert.ok(training.gradient_steps > 0); assert.equal(training.actor_parameters_changed, true); checks.actualPolicyUpdate = true;
    assert.equal(training.replay_status, 'accepted_independent_geometry');
    assert.equal(training.replay.final_evaluation, false);
    assert.equal(training.replay.native_certificate.complete_tool_checked, true); checks.independentSelectionGeometry = true;
    const selected = { caseHash: source.caseHash, runId: result.runId };
    const initial = await engine.request('replayTraining', { ...selected, step: 0 });
    const initialMask = await engine.assets.read(initial.removedMask.assetId);
    assert.ok(initialMask.every(value => value === 0)); checks.initialReplayEmpty = true;
    const final = await engine.request('replayTraining', selected);
    const mask = await engine.assets.read(final.removedMask.assetId);
    assert.ok(mask.every(value => value === 0 || value === 1));
    assert.equal(final.clinicalDeficitProbability, null); assert.equal(final.finalEvaluation, false); checks.nativeReplayMask = true;
    const runs = await engine.request('listRuns', { caseHash: source.caseHash });
    assert.ok(runs.runs.some(run => run.runId === result.runId && run.hasCheckpoint && run.hasAcceptedReplay)); checks.checkpointListed = true;
    const exportedPath = path.join(temporary, 'candidate.json');
    await engine.request('exportCandidate', { ...selected, path: exportedPath });
    const exported = JSON.parse(await fs.readFile(exportedPath, 'utf8'));
    assert.equal(exported.candidate.artifact_hash, training.replay.artifact_hash);
    assert.equal(exported.final_evaluation, false); checks.candidateExport = true;
    await assert.rejects(engine.request('trainPatient', { caseHash: source.caseHash, resumeRunId: result.runId, budgetSeconds: 6 }), /RESUME_CONTRACT_CHANGED/);
    checks.alteredResumeContractRejected = true;
    await engine.stop();
    const restored = await start(); assert.equal(restored.caseHash, source.caseHash);
    const replay = await engine.request('replayTraining', selected);
    assert.equal(replay.simulatedRemovedTargetVolumeMm3, final.simulatedRemovedTargetVolumeMm3); checks.restartIndependentRecheck = true;
    const reportFile = path.join(runDir, result.runId, 'native-refinement.json');
    const tampered = JSON.parse(await fs.readFile(reportFile, 'utf8')); tampered.replay.metrics.clinical_deficit_probability = .9;
    await fs.writeFile(reportFile, JSON.stringify(tampered));
    await assert.rejects(engine.request('replayTraining', selected), /RUN_INTEGRITY_FAILED/); checks.modifiedReportRejected = true;
    Object.assign(report, { passed: true, gradientSteps: training.gradient_steps, stepCount: final.stepCount,
      simulatedRemovedTargetVolumeMm3: final.simulatedRemovedTargetVolumeMm3, simulatedRemovedNormalVolumeMm3: final.simulatedRemovedNormalVolumeMm3 });
  } catch (error) { report.error = String(error.stack).slice(0, 3000); process.exitCode = 1; }
  finally {
    if (engine) await engine.stop(); await fs.rm(temporary, { recursive: true, force: true });
    report.elapsedSeconds = (performance.now() - started) / 1000;
    await fs.mkdir(path.join(repo, 'artifacts'), { recursive: true });
    await fs.writeFile(path.join(repo, 'artifacts', `electron-training-${suffix}.json`), JSON.stringify(report, null, 2));
    process.stdout.write(JSON.stringify(report) + '\n');
  }
}
main().catch(error => { process.stderr.write(error.stack + '\n'); process.exitCode = 1; });
