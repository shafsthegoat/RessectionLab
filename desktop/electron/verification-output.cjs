'use strict';

const path = require('node:path');

// Development-only verifier option, so evidence from successive builds can coexist.
function verificationDirectory(repo) {
  const index = process.argv.indexOf('--report-dir');
  if (index < 0) return path.join(repo, 'artifacts');
  const value = process.argv[index + 1];
  if (!value || value.startsWith('--')) throw new Error('--report-dir requires a directory');
  return path.resolve(repo, value);
}

module.exports = { verificationDirectory };
