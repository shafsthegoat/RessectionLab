'use strict';

const assert = require('node:assert/strict');
const { randomUUID } = require('node:crypto');
const { existsSync } = require('node:fs');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const { setTimeout: delay } = require('node:timers/promises');
const { Sidecar } = require('./sidecar.cjs');

const repo = path.resolve(__dirname, '../..');
const python = process.env.RESSECTIONLAB_TEST_PYTHON || path.join(repo, '.venv/bin/python');

async function exercise(hidden) {
  const transferDir = await fs.mkdtemp(path.join(os.tmpdir(), 'rehearsal-timing-test-'));
  const argumentCount = process.argv.length;
  if (hidden) process.argv.push('--automation-hidden');
  const sidecar = new Sidecar({ python, cwd: repo, sourcePath: path.join(repo, 'src'), transferDir,
    runDir: path.join(transferDir, 'runs') });
  const traces = [];
  sidecar.on('rehearsalTiming', row => traces.push(row));
  try {
    const { caseHash } = await sidecar.request('createSurgicalFixture');
    const initial = await sidecar.request('surgicalRehearsal', { caseHash, command: 'create' });
    const step = { caseHash, command: 'step', expectedStateId: initial.view.stateId,
      action: { operation: 'inspect', slot: 'left', duration: .2 } };
    const requestId = 'surgical-' + randomUUID();
    const accepted = await sidecar.request('surgicalRehearsal', { ...step, requestId });
    const rejected = await sidecar.request('surgicalRehearsal', { ...step, expectedStateId: accepted.view.stateId,
      action: { ...step.action, operation: 'invalid-operation' } });
    await assert.rejects(() => sidecar.request('surgicalRehearsal', step), /SURGICAL_STATE_STALE/);
    if (hidden) {
      for (let attempt = 0; attempt < 100 && traces.filter(row => row.requestId === requestId).length < 2; attempt++) {
        await delay(20);
      }
      const pythonRow = traces.find(row => row.process === 'python' && row.requestId === requestId);
      const electronRow = traces.find(row => row.process === 'electron' && row.requestId === requestId);
      assert(pythonRow && electronRow, 'Both sides report the same request');
      assert.equal(pythonRow.responseStateId, accepted.view.stateId);
      assert.equal(electronRow.responseStateId, accepted.view.stateId);
      assert.equal(pythonRow.caseHash, caseHash);
      assert.equal(pythonRow.sourceBranchId, initial.branchId);
      assert.equal(pythonRow.sourcePosition, initial.position);
      assert.equal(pythonRow.position, accepted.position);
      assert.equal(pythonRow.responseBytes, electronRow.responseBytes);
      for (const name of ['queue', 'dispatch', 'simulationStep', 'snapshot', 'view', 'metrics', 'stateIdentity',
        'jsonBudget', 'serialize', 'writeFlush']) {
        assert(Number.isSafeInteger(pythonRow.durationsNs[name]) && pythonRow.durationsNs[name] >= 0, name);
      }
      const simulation = pythonRow.stageBoundsNs.simulationStep;
      assert(simulation.firstStart >= pythonRow.durationsNs.queue && simulation.lastEnd > simulation.firstStart);
      const latestOrigin = BigInt(electronRow.responseLineReadyMainMonotonicNs) - BigInt(pythonRow.responseWriteStartSinceQueueNs);
      assert(latestOrigin >= BigInt(electronRow.requestSentMainMonotonicNs));
    } else {
      assert.equal(traces.length, 0, 'The environment variable requires hidden automation');
    }
    return { accepted, rejected };
  } finally {
    process.argv.length = argumentCount;
    await sidecar.stop();
  }
}

test('hidden rehearsal timings pair real sidecar responses and preserve accepted and rejected results',
  { skip: !existsSync(python), timeout: 30000 }, async () => {
    const variables = { RESSECTIONLAB_REHEARSAL_TIMING: '1', RESSECTIONLAB_NUMERICAL_BACKEND: 'reference' };
    const original = Object.fromEntries(Object.keys(variables).map(key => [key, process.env[key]]));
    Object.assign(process.env, variables);
    try {
      const ordinary = await exercise(false);
      const traced = await exercise(true);
      assert.deepEqual(traced, ordinary);
      assert.equal(traced.accepted.outcome.status, 'accepted');
      assert.equal(traced.rejected.outcome.status, 'rejected');
    } finally {
      for (const [key, value] of Object.entries(original)) {
        if (value === undefined) delete process.env[key];
        else process.env[key] = value;
      }
    }
  });
