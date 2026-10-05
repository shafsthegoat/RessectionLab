'use strict';

// Read actual frozen prior arrays and optionally save view-position fixtures.
// No policy updates, final evaluation, source edits, or evidence approval.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { createHash } = require('node:crypto');
const { Sidecar } = require('./sidecar.cjs');
const { verificationDirectory } = require('./verification-output.cjs');

const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const flat = (p, shape) => (p[0] * shape[1] + p[1]) * shape[2] + p[2];

async function main() {
  const repo = path.resolve(__dirname, '../..');
  const packaged = process.argv.includes('--bundle');
  const executable = path.join(repo, packaged
    ? 'desktop/release/RessectionLab-darwin-arm64/RessectionLab.app/Contents/Resources/research-engine/ressectionlab-engine'
    : 'desktop/sidecar/ressectionlab-engine/ressectionlab-engine');
  const output = verificationDirectory(repo);
  const transferDir = await fs.mkdtemp(path.join(os.tmpdir(), 'ressectionlab-priors-'));
  const engine = new Sidecar({ executable, transferDir, cwd: os.tmpdir() });
  const report = { passed: false, packaged, executableSha256: sha(await fs.readFile(executable)),
    trainingRequested: false, finalEvaluationRequested: false, layers: [], fixtures: [] };
  const started = performance.now();
  engine.on('diagnostic', message => process.stderr.write(message));
  try {
    report.protocol = await engine.request('ping');
    const originalPath = path.join(repo, 'outputs/cases/UCSF-PDGM-0004-prior-proposals.ressectionlab');
    const originalBundleHash = sha(await fs.readFile(originalPath));
    const data = await engine.request('loadCase', { path: originalPath });
    report.caseHash = data.caseHash;
    report.planningHash = data.planningHash;
    assert.equal(data.priorProposals.length, 7);
    const expected = JSON.parse(await fs.readFile(path.join(repo, 'artifacts/desktop-renderer/ucsf-seven-prior-hydration.json'), 'utf8'));
    const signatures = data.priorProposals.map(p => ({ mapId: p.mapId, data: p.data.sha256, coverage: p.samplingCoverage.sha256 }));
    for (const proposal of data.priorProposals) {
      assert.equal(proposal.viewOnly, true);
      assert.equal(proposal.planningEligible, false);
      assert.equal(proposal.patientSpecificFunction, false);
      assert.equal(proposal.clinicalDeficitProbability, null);
      assert.equal(proposal.reviewStatus, 'alignment_review_required');
      const valuesBytes = await engine.assets.read(proposal.data.assetId);
      const coverage = await engine.assets.read(proposal.samplingCoverage.assetId);
      const values = new Float32Array(valuesBytes.buffer, valuesBytes.byteOffset, valuesBytes.byteLength / 4);
      const reference = expected.results.find(p => p.mapId === proposal.mapId);
      assert.ok(reference);
      const samples = {};
      for (const [name, example] of Object.entries(reference.examples)) {
        const index = flat(example.sample.voxel, proposal.shape);
        assert.equal(coverage[index] === 1, example.sample.covered);
        if (example.sample.covered) assert.ok(Math.abs(values[index] - example.sample.value) < 1e-10);
        samples[name] = { world: example.world, covered: coverage[index] === 1, value: coverage[index] ? values[index] : null };
      }
      report.layers.push({ mapId: proposal.mapId, mapKind: proposal.mapKind,
        valueSha256: proposal.data.sha256, coverageSha256: proposal.samplingCoverage.sha256, samples });
    }
    if (process.argv.includes('--prepare-cursors')) {
      const fixtures = {
        'covered-nonzero': [-54, 106, 77],
        'covered-zero': [-32, 24, 18],
        'outside-fov': [0, 239, 0],
        'incomplete-support': [-39.75, 106, 77],
        'binary-included': [-64, 92, 79],
      };
      const directory = path.join(repo, 'outputs/cases/electron-prior-qa');
      await fs.mkdir(directory, { recursive: true });
      for (const [name, cursor] of Object.entries(fixtures)) {
        const fixture = path.join(directory, `${name}.ressectionlab`);
        await engine.request('saveCase', { caseHash: data.caseHash, path: fixture, overwrite: true, workspace: { cursor } });
        const reopened = await engine.request('loadCase', { path: fixture });
        assert.equal(reopened.caseHash, data.caseHash);
        assert.equal(reopened.planningHash, data.planningHash);
        assert.equal(reopened.mri.sha256, data.mri.sha256);
        assert.deepEqual(reopened.compartments.map(p => [p.name, p.sourceArray?.sha256, p.volumeMm3]), data.compartments.map(p => [p.name, p.sourceArray?.sha256, p.volumeMm3]));
        assert.deepEqual(reopened.priorProposals.map(p => ({ mapId: p.mapId, data: p.data.sha256, coverage: p.samplingCoverage.sha256 })), signatures);
        assert.deepEqual(reopened.artifacts.workspace.cursor, cursor);
        report.fixtures.push({ name, path: path.relative(repo, fixture), cursor, bundleSha256: sha(await fs.readFile(fixture)),
          sourceArraysAndPriorsUnchanged: true, caseHashUnchanged: true });
      }
    }
    assert.equal(sha(await fs.readFile(originalPath)), originalBundleHash);
    report.originalBundleSha256 = originalBundleHash;
    report.originalBundleUnchanged = true;
    report.passed = true;
  } catch (error) {
    report.error = String(error.stack).slice(0, 2400);
    process.exitCode = 1;
  } finally {
    await engine.stop();
    report.elapsedSeconds = (performance.now() - started) / 1000;
    await fs.mkdir(output, { recursive: true });
    await fs.writeFile(path.join(output, 'prior-protocol-validation.json'), JSON.stringify(report, null, 2) + '\n');
    process.stdout.write(JSON.stringify({ passed: report.passed, layers: report.layers.length, fixtures: report.fixtures.length, elapsedSeconds: report.elapsedSeconds, error: report.error }) + '\n');
  }
}
main().catch(error => { process.stderr.write(error.stack + '\n'); process.exitCode = 1; });
