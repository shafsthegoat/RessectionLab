'use strict';

const { plainArgs } = require('./security.cjs');

// Only the main process selects paths. A cancelled dialog never starts an import.
async function importStructuralEvidence(args, { pick, request }) {
  plainArgs(args, ['caseHash', 'variant']);
  if (!['nocsf', 'main'].includes(args.variant)) throw new Error('Choose a supported brain-envelope variant');
  const sourceImagePath = await pick('Select the source MRI used to estimate the envelope', ['nii', 'gz']);
  if (!sourceImagePath) return null;
  const maskPath = await pick('Select the proposed brain-envelope mask', ['nii', 'gz']);
  if (!maskPath) return null;
  const reportPath = await pick('Select the envelope provenance report', ['json']);
  if (!reportPath) return null;
  return request('importStructuralEvidence', { ...args, sourceImagePath, maskPath, reportPath });
}

module.exports = { importStructuralEvidence };
