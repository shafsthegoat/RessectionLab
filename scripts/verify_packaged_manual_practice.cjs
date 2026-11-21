'use strict';

// node scripts/verify_packaged_manual_practice.cjs APP CASE ACTIONS EMPTY_OUTPUT [REFERENCE_REPORT]
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {execFileSync} = require('node:child_process');
const {launchWithHarness} = require('./verify_packaged_workflow.cjs');

const [, , appArg, caseArg, actionsArg, outputArg, reference] = process.argv;
const root = path.resolve(__dirname, '..');
const testRoot = path.join(root, 'testcases');
const app = fs.realpathSync(appArg);
const source = fs.realpathSync(caseArg);
const actions = fs.realpathSync(actionsArg);
const output = path.resolve(outputArg);
for (const input of [source, actions]) assert(input.startsWith(testRoot + path.sep), 'Use a verified copy of each fixture in testcases');
assert(output.startsWith(path.join(testRoot, 'outputs') + path.sep), 'Use an ignored output');
assert(!fs.existsSync(output), 'Preserve previous evidence');
const harness = path.join(root, 'desktop/electron/verify-packaged-manual-practice.cjs');
const sha = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const read = name => JSON.parse(fs.readFileSync(path.join(output, name), 'utf8'));
const signedFiles = ['Contents/MacOS/RessectionLab', 'Contents/Resources/app.asar', 'Contents/Resources/research-engine/ressectionlab-engine'];
function packageIdentity() {
  execFileSync('/usr/bin/codesign', ['--verify', '--deep', '--strict', app], {stdio: ['ignore', 'pipe', 'pipe']});
  return Object.fromEntries(signedFiles.map(name => [name, sha(path.join(app, name))]));
}
function stateSequence(report) {
  return report.actions.map(({result}) => {
    const {stateId, physicalDigest, metricDigest, position, tools} = result.witness;
    return {stateId, physicalDigest, metricDigest, position, tools, requested: result.requested, status: result.status};
  });
}

async function main() {
  fs.mkdirSync(output, {recursive: true});
  const receipt = {schema: 'resectionlab.actual-signed-manual-practice.v1', app, source, actions, passed: false};
  try {
    receipt.package = packageIdentity();
    receipt.sourceSha256 = sha(source);
    receipt.actionsSha256 = sha(actions);
    for (const phase of ['warm', 'cold']) {
      const launcher = path.join(output, phase + '.launcher');
      const result = await launchWithHarness({
        app, harness, output, launcher, schema: 'resectionlab.actual-packaged-practice-launch.v1',
        env: {PACKAGED_PRACTICE_APP: app, PACKAGED_PRACTICE_CASE: source, PACKAGED_PRACTICE_ACTIONS: actions,
          PACKAGED_PRACTICE_OUTPUT: output, PACKAGED_PRACTICE_COLD: phase === 'cold' ? '1' : '0'},
        verify({executable}) {
          const proof = read('packaged-' + phase + '.json');
          assert(proof.passed, proof.error);
          assert.equal(proof.appExecutable, executable);
        }
      });
      assert(result.passed, result.error);
    }
    const warm = read('packaged-warm.json'), cold = read('packaged-cold.json');
    assert.deepEqual(packageIdentity(), receipt.package, 'Signed bundle changed during testing');
    assert.equal(sha(source), receipt.sourceSha256);
    assert.equal(sha(actions), receipt.actionsSha256);
    if (reference) {
      const expected = JSON.parse(fs.readFileSync(reference, 'utf8'));
      assert(expected.passed);
      assert.deepEqual(stateSequence(read('benchmark.json')), stateSequence(expected));
      receipt.referenceReportSha256 = sha(reference);
    }
    receipt.renderer = warm.loadedRenderer;
    receipt.nativeBinarySha256 = warm.engineAfter.numericalBackend.binarySha256;
    receipt.manualKernels = warm.manualKernels;
    receipt.stateId = warm.finalState;
    receipt.restart = {warmPid: warm.pid, coldPid: cold.pid, stateId: cold.finalState, history: cold.reopenedHistory};
    receipt.savedSha256 = warm.savedSha256;
    receipt.passed = true;
  } catch (error) {
    receipt.error = String(error.stack ?? error);
    console.error(error);
  }
  fs.writeFileSync(path.join(output, 'signed-package-proof.json'), JSON.stringify(receipt, null, 2) + '\n');
  console.log(JSON.stringify({passed: receipt.passed, output}));
  process.exitCode = receipt.passed ? 0 : 1;
}
main().catch(error => { console.error(error); process.exitCode = 1; });
