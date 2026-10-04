'use strict';

// Runs the exact main-process JSONL and binary path without a browser or source imports.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { Sidecar } = require('./sidecar.cjs');

async function main() {
  const repo = path.resolve(__dirname, '../..');
  const transferDir = await fs.mkdtemp(path.join(os.tmpdir(), 'ressectionlab-verify-'));
  const packaged = process.argv.includes('--bundle');
  const frozen = process.argv.includes('--frozen') || packaged;
  const output = path.join(repo, 'artifacts', packaged ? 'electron-sidecar-packaged.json' : frozen ? 'electron-sidecar-frozen.json' : 'electron-sidecar-source.json');
  const checks = {};
  const start = performance.now();
  const sidecar = new Sidecar({ transferDir, cwd: os.tmpdir(), ...(frozen
    ? { executable: path.join(repo, packaged ? 'desktop/release/RessectionLab-darwin-arm64/RessectionLab.app/Contents/Resources/research-engine/ressectionlab-engine' : 'desktop/sidecar/ressectionlab-engine/ressectionlab-engine') }
    : { python: path.join(repo, '.venv/bin/python'), sourcePath: path.join(repo, 'src') }) });
  sidecar.on('diagnostic', message => process.stderr.write(message));
  const report = { frozen, checks, passed: false };
  try {
    const ping = await sidecar.request('ping');
    assert.equal(ping.protocolVersion, 1); checks.protocol = true;
    const value = await sidecar.request('loadCase', { path: path.join(repo, 'outputs/cases/UCSF-PDGM-0004.ressectionlab') });
    assert.equal(value.mri.path, undefined);
    const bytes = await sidecar.assets.read(value.mri.assetId);
    assert.equal(bytes.byteLength, value.shape.reduce((a, b) => a * b, 4));
    assert.ok(new Float32Array(bytes.buffer).some(number => number > 0)); checks.realMriTransfer = true;
    for (const compartment of value.compartments) await sidecar.assets.read(compartment.array.assetId);
    checks.allCompartmentTransfers = true;
    await assert.rejects(sidecar.request('generateRoutes', { caseHash: 'stale-case', allowEstimatedSupport: true })); checks.staleCaseRejected = true;
    const routes = await sidecar.request('generateRoutes', { caseHash: value.caseHash, allowEstimatedSupport: true, config: { targetsPerCompartment: 1, maxWindows: 1 } });
    assert.ok(routes.candidates.length >= 2);
    assert.ok(routes.candidates.every(route => route.clinical_deficit_probability === null && route.simulated_removed_target_volume_mm3 === null));
    checks.searchAndHonestOutputs = true;
    const workspace = { selectedRouteIds: routes.candidates.slice(0, 2).map(route => route.route_id), cursorMm: [0, 0, 0] };
    const savePath = path.join(transferDir, 'roundtrip.ressectionlab');
    await sidecar.request('saveCase', { caseHash: value.caseHash, path: savePath, workspace });
    const reopened = await sidecar.request('loadCase', { path: savePath });
    assert.equal(reopened.caseHash, value.caseHash);
    assert.deepEqual(reopened.artifacts.workspace.selectedRouteIds, workspace.selectedRouteIds);
    assert.deepEqual(reopened.artifacts.workspace.cursorMm, workspace.cursorMm);
    assert.deepEqual(reopened.artifacts.workspace.routes, routes.candidates);
    assert.equal(reopened.artifacts.workspace.case_hash, value.caseHash); checks.workspaceRoundtrip = true;
    await assert.rejects(sidecar.assets.read(value.mri.assetId)); checks.previousAssetInvalidated = true;
    const cancelResult = await sidecar.request('cancel', { requestId: 'no-such-request' });
    assert.ok(cancelResult); checks.cancellationControl = true;
    report.caseId = value.caseId; report.shape = value.shape; report.candidateCount = routes.candidates.length;
    report.passed = true;
  } catch (error) { report.error = String(error.stack).slice(0, 2000); process.exitCode = 1; }
  finally {
    await sidecar.stop(); report.elapsedSeconds = (performance.now() - start) / 1000;
    await fs.mkdir(path.dirname(output), { recursive: true }); await fs.writeFile(output, JSON.stringify(report, null, 2));
    process.stdout.write(JSON.stringify(report) + '\n');
  }
}
main().catch(error => { process.stderr.write(error.stack + '\n'); process.exitCode = 1; });
