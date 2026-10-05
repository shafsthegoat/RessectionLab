'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { importStructuralEvidence } = require('./structural-import.cjs');

test('structural import uses only native selected paths after all dialogs complete', async () => {
  const selected = ['/research/source.nii.gz', '/research/estimate.nii.gz', '/research/report.json'];
  const dialogs = [];
  const requests = [];
  const result = { caseHash: 'new-identity', structuralEvidence: [] };
  assert.equal(await importStructuralEvidence({ caseHash: 'case-identity', variant: 'nocsf' }, {
    pick: async (title, extensions) => { dialogs.push({ title, extensions }); return selected[dialogs.length - 1]; },
    request: async (...args) => { requests.push(args); return result; },
  }), result);
  assert.deepEqual(dialogs.map(dialog => dialog.extensions), [['nii', 'gz'], ['nii', 'gz'], ['json']]);
  assert.deepEqual(requests, [['importStructuralEvidence', { caseHash: 'case-identity', variant: 'nocsf', sourceImagePath: selected[0], maskPath: selected[1], reportPath: selected[2] }]]);
});

test('cancelling any structural dialog leaves the case untouched', async () => {
  for (let cancelled = 0; cancelled < 3; cancelled += 1) {
    let picks = 0;
    let requests = 0;
    assert.equal(await importStructuralEvidence({ caseHash: 'case-identity', variant: 'main' }, {
      pick: async () => picks++ === cancelled ? null : '/research/chosen-file',
      request: async () => { requests += 1; },
    }), null);
    assert.equal(picks, cancelled + 1);
    assert.equal(requests, 0);
  }
});

test('renderer paths and unsupported variants are rejected before opening dialogs', async () => {
  for (const args of [
    { caseHash: 'case-identity', variant: 'main', sourceImagePath: '/private/injected.nii' },
    { caseHash: 'case-identity', variant: 'main', maskPath: '/private/injected.nii' },
    { caseHash: 'case-identity', variant: 'main', reportPath: '/private/injected.json' },
    { caseHash: 'case-identity', variant: 'unknown' },
  ]) {
    let touched = false;
    await assert.rejects(importStructuralEvidence(args, {
      pick: async () => { touched = true; }, request: async () => { touched = true; },
    }));
    assert.equal(touched, false);
  }
});
