'use strict';
const test = require('node:test'), assert = require('node:assert/strict');
const path = require('node:path');
const { verificationDirectory } = require('./verification-output.cjs');
test('default evidence directories are unique ignored runs and explicit report paths still work', () => {
  const original = process.argv;
  try {
    process.argv = ['node', 'generated-verifier'];
    const root = path.resolve(__dirname, '../..'), first = verificationDirectory(root), second = verificationDirectory(root);
    assert(first.startsWith(path.join(root, 'testcases/outputs/desktop-verification') + path.sep));
    assert.notEqual(first, second);
    process.argv.push('--report-dir', 'testcases/explicit-proof');
    assert.equal(verificationDirectory(root), path.join(root, 'testcases/explicit-proof'));
    process.argv.pop();
    assert.throws(() => verificationDirectory(root), /requires/);
  } finally { process.argv = original; }
});
