'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { EventEmitter } = require('node:events');
const { axisInspectionArgs, inspectAxisPlanning, MAX_INSPECTION_BYTES } = require('./axis-inspection.cjs');
const { Sidecar, OPERATIONS } = require('./sidecar.cjs');

function validArgs() {
  return { caseHash: 'sha256:' + 'a'.repeat(64), planningHash: 'sha256:' + 'b'.repeat(64),
    routeId: 'search-0123456789abcdef', routePlanningModelHash: 'c'.repeat(64),
    toolIds: ['native-wide-aspiration', 'native-fine-aspiration'],
    acknowledgeNeighboringColumns: true, acknowledgeEstimatedSupport: false,
    expectedBindingHash: 'sha256:' + 'd'.repeat(64) };
}

function response(args = validArgs()) {
  return { schemaVersion: 1, caseHash: args.caseHash, planningHash: args.planningHash,
    routeId: args.routeId, routePlanningModelHash: args.routePlanningModelHash,
    accessSource: 'selected_route_window_only', requestedToolIds: [...args.toolIds].sort(),
    anchorWindowRas: { center_mm: [-5, 10, 25], normal_inward: [0, 0, -1], radius_mm: 6, window_id: 'window-1' },
    anchorWindowNormalization: { normalAbsoluteTolerance: 4 * Number.EPSILON, normalMaximumDifference: 0 },
    inspection: { binding_hash: args.expectedBindingHash, fixture: true } };
}

test('read-only inspection forwards the exact bound request and accepts canonical tool order', async () => {
  const args = validArgs();
  const expected = response(args);
  const calls = [];
  const result = await inspectAxisPlanning(args, { request: async (...values) => { calls.push(values); return expected; } });
  assert.equal(result, expected);
  assert.deepEqual(calls, [['inspectAxisPlanning', args]]);
  assert.notEqual(calls[0][1], args);
  assert.notEqual(calls[0][1].toolIds, args.toolIds);
});

test('inspection refuses renderer paths, geometry, execution and learner controls before dispatch', async () => {
  let calls = 0;
  for (const key of ['path', 'outputDir', 'config', 'entryMm', 'targetMm', 'window', 'tools', 'code',
    'op', 'reward', 'worlds', 'budgetSeconds', 'seed', 'resumeRunId']) {
    await assert.rejects(inspectAxisPlanning({ ...validArgs(), [key]: 'injected' }, {
      request: async () => { calls++; },
    }), /Unsupported operation argument/);
  }
  assert.equal(calls, 0);
});

test('inspection requires exact hash formats, bounded route IDs and explicit acknowledgments', () => {
  const coercible = { toString: () => validArgs().caseHash };
  const changes = [
    { caseHash: 'a'.repeat(64) }, { caseHash: coercible }, { planningHash: null },
    { planningHash: 'sha256:' + 'A'.repeat(64) },
    { routePlanningModelHash: 'sha256:' + 'c'.repeat(64) },
    { routePlanningModelHash: 'c'.repeat(63) }, { expectedBindingHash: null },
    { expectedBindingHash: 'd'.repeat(64) }, { routeId: '' }, { routeId: ' '.repeat(10) },
    { routeId: 'r'.repeat(129) }, { routeId: 'route\n1' }, { routeId: 1 },
    { acknowledgeNeighboringColumns: false }, { acknowledgeNeighboringColumns: 1 },
    { acknowledgeEstimatedSupport: 'true' }, { acknowledgeEstimatedSupport: undefined },
  ];
  for (const change of changes) assert.throws(() => axisInspectionArgs({ ...validArgs(), ...change }));
  const args = validArgs(); delete args.expectedBindingHash;
  assert.equal(Object.hasOwn(axisInspectionArgs(args), 'expectedBindingHash'), false);
  assert.equal(axisInspectionArgs({ ...args, routeId: 'r'.repeat(128) }).routeId.length, 128);
});

test('inspection restricts tools to one or two distinct named native profiles', () => {
  for (const toolIds of [[], [undefined], Array(1), 'native-fine-aspiration',
    ['native-fine-aspiration', 'native-fine-aspiration'],
    ['native-fine-aspiration', 'native-wide-aspiration', 'third'],
    ['aspirator-a'], [{ tool_id: 'native-fine-aspiration' }]]) {
    assert.throws(() => axisInspectionArgs({ ...validArgs(), toolIds }), /native research tools/);
  }
  assert.deepEqual(axisInspectionArgs({ ...validArgs(), toolIds: ['native-fine-aspiration'] }).toolIds,
    ['native-fine-aspiration']);
});

test('stale or substituted result identities cannot cross the main-process boundary', async () => {
  const changes = [null, [], { schemaVersion: 2 }, { caseHash: 'sha256:' + 'e'.repeat(64) },
    { planningHash: 'sha256:' + 'e'.repeat(64) }, { routeId: 'different-route' },
    { routePlanningModelHash: 'e'.repeat(64) }, { accessSource: 'renderer_window' },
    { anchorWindowRas: null }, { anchorWindowRas: { ...response().anchorWindowRas, center_mm: [0, 0, NaN] } },
    { anchorWindowRas: { ...response().anchorWindowRas, radius_mm: 0 } },
    { anchorWindowNormalization: { normalAbsoluteTolerance: 1e-3, normalMaximumDifference: 0 } },
    { anchorWindowNormalization: { normalAbsoluteTolerance: 0, normalMaximumDifference: Number.EPSILON } },
    { requestedToolIds: ['native-fine-aspiration'] },
    { requestedToolIds: ['native-fine-aspiration', 'native-fine-aspiration'] },
    { requestedToolIds: ['native-fine-aspiration', 'unknown'] }, { requestedToolIds: Array(2) }];
  for (const change of changes) {
    await assert.rejects(inspectAxisPlanning(validArgs(), {
      request: async () => change === null || Array.isArray(change) ? change : { ...response(), ...change },
    }), /does not match/);
  }
  const missing = response(); delete missing.inspection;
  await assert.rejects(inspectAxisPlanning(validArgs(), { request: async () => missing }), /does not match/);
});

test('the bound request survives caller mutation and engine rejection is propagated without retry', async () => {
  const args = validArgs();
  const result = await inspectAxisPlanning(args, { request: async (_op, sent) => {
    args.caseHash = 'changed'; args.toolIds.pop();
    return response(sent);
  } });
  assert.equal(result.caseHash, validArgs().caseHash);
  assert.equal(result.requestedToolIds.length, 2);
  let calls = 0;
  const cancelled = new Error('Operation cancelled');
  await assert.rejects(inspectAxisPlanning(validArgs(), { request: async () => { calls++; throw cancelled; } }),
    error => error === cancelled);
  assert.equal(calls, 1);
});

test('oversized inspection data is refused before reaching the renderer', async () => {
  await assert.rejects(inspectAxisPlanning(validArgs(), {
    request: async () => ({ ...response(), inspection: { padding: 'x'.repeat(MAX_INSPECTION_BYTES) } }),
  }), /size limit/);
});

function fakeEngine(t) {
  const engine = Object.create(Sidecar.prototype);
  EventEmitter.call(engine);
  engine.pending = new Map(); engine.closed = false;
  const writes = [], events = [];
  engine.child = { stdin: { write: line => { writes.push(JSON.parse(line)); } } };
  engine.assets = { clear: () => assert.fail('Read-only inspection must not invalidate case assets'),
    expose: async value => value };
  engine.on('event', event => events.push(event));
  t.after(() => { for (const pending of engine.pending.values()) clearTimeout(pending.timeout); });
  return { engine, writes, events };
}

test('sidecar retains normal request IDs, progress and default budget without changing fixed-route APIs', async t => {
  const { engine, writes, events } = fakeEngine(t);
  for (const op of ['inspectAxisPlanning', 'inspectRefinement', 'generateNativeRoutes', 'trainPatient', 'replayTraining'])
    assert.equal(OPERATIONS.has(op), true);
  await assert.rejects(engine.request('executePython', {}), /Unsupported/);
  const pending = engine.request('inspectAxisPlanning', validArgs());
  assert.equal(writes[0].timeoutMs, 120000);
  assert.equal(writes[0].op, 'inspectAxisPlanning');
  const id = writes[0].id;
  await engine.handle({ id, event: 'started' });
  await engine.handle({ id, event: 'progress', progress: { fraction: 0.5, message: 'Inspecting declared geometry' } });
  await engine.handle({ id, event: 'result', result: response() });
  assert.deepEqual(await pending, response());
  assert.deepEqual(events.map(value => [value.event, value.op]),
    [['started', 'inspectAxisPlanning'], ['progress', 'inspectAxisPlanning'], ['result', 'inspectAxisPlanning']]);
  assert.equal(engine.pending.size, 0);
});

test('cancelled inspection withholds late results while other request IDs remain live', async t => {
  const { engine, writes, events } = fakeEngine(t);
  const pending = engine.request('inspectAxisPlanning', validArgs());
  const rejected = assert.rejects(pending, /Operation cancelled/);
  const other = engine.request('inspectRefinement', { caseHash: validArgs().caseHash, routeId: 'search-other' });
  await engine.handle({ id: writes[0].id, event: 'cancelled' });
  await rejected;
  await engine.handle({ id: writes[0].id, event: 'result', result: response() });
  assert.equal(engine.pending.size, 1);
  assert.deepEqual(events.map(value => value.event), ['cancelled']);
  await engine.handle({ id: writes[1].id, event: 'result', result: { status: 'no_actionable_moves' } });
  assert.deepEqual(await other, { status: 'no_actionable_moves' });
});

test('preload exposes a named inspection method, not arbitrary operation execution', async () => {
  let api;
  const calls = [];
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, 'preload.cjs'), 'utf8'), {
    require: name => {
      assert.equal(name, 'electron');
      return { contextBridge: { exposeInMainWorld: (_name, value) => { api = value; } },
        ipcRenderer: { invoke: async (...values) => { calls.push(values); return response(); } } };
    },
  });
  assert.equal(Object.isFrozen(api), true);
  assert.equal(api.request, undefined);
  await api.inspectAxisPlanning(validArgs());
  assert.deepEqual(calls, [['research:inspectAxisPlanning', validArgs()]]);
  assert.equal(typeof api.inspectRefinement, 'function');
  assert.equal(typeof api.trainPatient, 'function');
});
