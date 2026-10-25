'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const source = fs.readFileSync(path.join(__dirname, 'preload.cjs'), 'utf8');
const request = () => ({ command: 'seekPrefixFrame', caseHash: 'case-a', branchId: 'main', position: 7, frameIndex: 3 });
const result = () => ({ branchId: 'boundary-branch', position: 8,
  playback: { branchId: 'main', position: 7, frameIndex: 3 },
  view: { stateId: 'sha256:exact-state', fluid: { 13: 2 } } });
function load({ flag, noProcess = false, invoke = () => Promise.resolve(result()) } = {}) {
  let api;
  const calls = [], requires = [], environment = {};
  if (flag !== undefined) environment.RESSECTIONLAB_PLAYBACK_RECEIPT_OBSERVER = flag;
  vm.runInNewContext(source, {
    ...(noProcess ? {} : { process: { env: environment } }),
    require(name) {
      requires.push(name); assert.equal(name, 'electron');
      return { contextBridge: { exposeInMainWorld(name, value) { assert.equal(name, 'resectionApi'); api = value; } },
        ipcRenderer: { invoke(...args) { calls.push(args); return invoke(...args); } } };
    },
  });
  return { api, calls, requires, environment };
}
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

test('default path has no observer/dependency globals and returns original invoke promise', async () => {
  const pending = deferred(), input = request(), expected = result();
  const h = load({ noProcess: true, invoke: () => pending.promise });
  assert.equal(h.api.onSurgicalReceipt, undefined);
  assert.equal(Object.isFrozen(h.api), true);
  assert.deepEqual(h.requires, ['electron']);
  const returned = h.api.surgicalRehearsal(input);
  assert.equal(returned, pending.promise);
  assert.deepEqual(h.calls, [['research:surgicalRehearsal', input]]);
  pending.resolve(expected);
  assert.equal(await returned, expected);
});

test('absent flag remains off after startup environment changes', async () => {
  const original = Promise.resolve(result()), h = load({ invoke: () => original });
  h.environment.RESSECTIONLAB_PLAYBACK_RECEIPT_OBSERVER = '1';
  assert.equal(h.api.onSurgicalReceipt, undefined);
  assert.equal(h.api.surgicalRehearsal(request()), original);
  await original;
});

for (const flag of ['0', '', 'true', ' 1', '1 ', 1, null, false]) {
  test(`invalid explicit receipt flag fails startup (${JSON.stringify(flag)})`, () => {
    assert.throws(() => load({ flag }), /SURGICAL_RECEIPT_OBSERVER:INVALID_STARTUP_FLAG/);
  });
}

test('opt-in observer synchronously stamps detached exact tuple before result consumer', async () => {
  const pending = deferred(), expected = result(), before = JSON.stringify(expected), events = [];
  const h = load({ flag: '1', invoke: () => pending.promise });
  h.environment.RESSECTIONLAB_PLAYBACK_RECEIPT_OBSERVER = undefined;
  let receipt;
  const cleanup = h.api.onSurgicalReceipt(value => { events.push('receipt'); receipt = value; });
  const input = request(), returned = h.api.surgicalRehearsal(input);
  assert.notEqual(returned, pending.promise);
  input.position = 99; input.caseHash = 'changed-after-invoke';
  const consumer = returned.then(value => { events.push('consumer'); return value; });
  pending.resolve(expected);
  assert.equal(await consumer, expected);
  assert.deepEqual(events, ['receipt', 'consumer']);
  assert.deepEqual(JSON.parse(JSON.stringify(receipt)), {
    command: 'seekPrefixFrame', branchId: 'main', position: 7, frameIndex: 3,
    stateId: 'sha256:exact-state', notificationSerial: 1,
    request: { branchId: 'main', position: 7, frameIndex: 3, caseHash: 'case-a' },
  });
  assert.equal(Object.isFrozen(receipt), true);
  assert.equal(Object.isFrozen(receipt.request), true);
  assert.equal(receipt.view, undefined);
  assert.equal(JSON.stringify(expected), before);
  cleanup(); cleanup();
});

test('projection falls back to boundary branch/position but preserves frame zero exactly', async () => {
  const expected = { branchId: 'main', position: 0, playback: { frameIndex: 0 }, view: { stateId: 'zero' } };
  const h = load({ flag: '1', invoke: () => Promise.resolve(expected) });
  let receipt;
  h.api.onSurgicalReceipt(value => { receipt = value; });
  assert.equal(await h.api.surgicalRehearsal({ ...request(), position: 0, frameIndex: 0 }), expected);
  assert.equal(receipt.branchId, 'main'); assert.equal(receipt.position, 0); assert.equal(receipt.frameIndex, 0);
});

test('non-frame commands keep original invoke promise even in observer mode without subscription', async () => {
  const original = Promise.resolve(result()), h = load({ flag: '1', invoke: () => original });
  for (const command of ['inspect', 'seekPrefix', 'seekFrame', 'step']) {
    assert.equal(h.api.surgicalRehearsal({ ...request(), command }), original);
  }
  await original;
});

test('invoke rejection reaches consumer unchanged and never notifies or poisons next receipt', async () => {
  const error = new Error('original invoke rejection');
  let reject = true, notifications = 0;
  const h = load({ flag: '1', invoke: () => reject ? Promise.reject(error) : Promise.resolve(result()) });
  h.api.onSurgicalReceipt(() => { notifications++; });
  await assert.rejects(h.api.surgicalRehearsal(request()), value => value === error);
  assert.equal(notifications, 0);
  reject = false;
  await h.api.surgicalRehearsal(request());
  assert.equal(notifications, 1);
});

test('default rejected promise also passes through by identity', async () => {
  const error = new Error('unchanged'), original = Promise.reject(error), h = load({ invoke: () => original });
  assert.equal(h.api.surgicalRehearsal(request()), original);
  await assert.rejects(original, value => value === error);
});

test('subscription is required, unique for the window, and cleanup cannot silently disable observation', async () => {
  let h = load({ flag: '1' });
  assert.throws(() => h.api.surgicalRehearsal(request()), /NO_ACTIVE_SUBSCRIPTION/);
  assert.equal(h.calls.length, 0);
  h = load({ flag: '1' });
  assert.throws(() => h.api.onSurgicalReceipt(null), /CALLBACK_REQUIRED/);
  h = load({ flag: '1' });
  const cleanup = h.api.onSurgicalReceipt(() => {});
  assert.throws(() => h.api.onSurgicalReceipt(() => {}), /ALREADY_REGISTERED/);
  assert.throws(() => h.api.surgicalRehearsal(request()), /ALREADY_REGISTERED/);
  cleanup();
  h = load({ flag: '1' });
  const close = h.api.onSurgicalReceipt(() => {});
  close(); close();
  assert.throws(() => h.api.surgicalRehearsal(request()), /NO_ACTIVE_SUBSCRIPTION/);
  assert.throws(() => h.api.onSurgicalReceipt(() => {}), /NO_ACTIVE_SUBSCRIPTION/);
});

test('cleanup of in-flight subscription rejects the observed delivery without invoking callback', async () => {
  const pending = deferred(), h = load({ flag: '1', invoke: () => pending.promise });
  let calls = 0;
  const cleanup = h.api.onSurgicalReceipt(() => { calls++; });
  const observed = h.api.surgicalRehearsal(request());
  cleanup(); pending.resolve(result());
  await assert.rejects(observed, /NO_ACTIVE_SUBSCRIPTION/);
  assert.equal(calls, 0);
});

for (const invalid of [
  { branchId: '' }, { position: -1 }, { position: 7.5 }, { frameIndex: '3' }, { frameIndex: NaN },
]) {
  test(`invalid request identity declines before IPC (${JSON.stringify(invalid)})`, () => {
    const h = load({ flag: '1' }); h.api.onSurgicalReceipt(() => {});
    assert.throws(() => h.api.surgicalRehearsal({ ...request(), ...invalid }), /INVALID_REQUEST_IDENTITY/);
    assert.equal(h.calls.length, 0);
  });
}

for (const [label, mutate] of [
  ['branch', value => { value.playback.branchId = 'other'; }],
  ['position', value => { value.playback.position = 8; }],
  ['frame', value => { value.playback.frameIndex = 4; }],
  ['missing-frame', value => { delete value.playback.frameIndex; }],
  ['missing-state', value => { delete value.view.stateId; }],
  ['nonstring-state', value => { value.view.stateId = 3; }],
]) {
  test(`invalid result identity fails before callback/consumer (${label})`, async () => {
    const invalid = result(); mutate(invalid);
    const h = load({ flag: '1', invoke: () => Promise.resolve(invalid) });
    let callback = false, consumer = false;
    h.api.onSurgicalReceipt(() => { callback = true; });
    await assert.rejects(h.api.surgicalRehearsal(request()).then(() => { consumer = true; }), /INVALID_RESULT_IDENTITY/);
    assert.equal(callback, false); assert.equal(consumer, false);
    assert.throws(() => h.api.surgicalRehearsal(request()), /INVALID_RESULT_IDENTITY/);
  });
}

test('callback throw fails observer mode and prevents result publication or later callbacks', async () => {
  const h = load({ flag: '1' }); let calls = 0, consumer = false;
  h.api.onSurgicalReceipt(() => { calls++; throw new Error('renderer callback failed'); });
  await assert.rejects(h.api.surgicalRehearsal(request()).then(() => { consumer = true; }), /CALLBACK_FAILED/);
  assert.equal(calls, 1); assert.equal(consumer, false);
  assert.throws(() => h.api.surgicalRehearsal(request()), /CALLBACK_FAILED/);
});

test('async callback is explicitly refused without an unhandled rejected callback promise', async () => {
  const h = load({ flag: '1' });
  h.api.onSurgicalReceipt(async () => { throw new Error('async callback'); });
  await assert.rejects(h.api.surgicalRehearsal(request()), /ASYNC_CALLBACK/);
});

test('window notification serial is monotonic and hard-capped at 20000', async () => {
  const h = load({ flag: '1' }); let notifications = 0;
  h.api.onSurgicalReceipt(value => { assert.equal(value.notificationSerial, ++notifications); });
  for (let index = 0; index < 20000; index++) await h.api.surgicalRehearsal(request());
  assert.equal(notifications, 20000); assert.equal(h.calls.length, 20000);
  assert.throws(() => h.api.surgicalRehearsal(request()), /NOTIFICATION_CAP/);
  assert.equal(notifications, 20000); assert.equal(h.calls.length, 20000);
});
