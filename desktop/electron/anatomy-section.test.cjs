'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const { createRequire } = require('node:module');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function sectionBoundary() {
  const main = path.join(__dirname, 'main.cjs');
  const localRequire = createRequire(main);
  const handlers = new Map();
  const calls = [];
  const frame = { url: 'file:///trusted-app.html' };
  const window = { webContents: { mainFrame: frame }, isDestroyed: () => false };
  const engine = { request: async (...args) => { calls.push(args); } };
  const electron = {
    app: { requestSingleInstanceLock: () => false, quit() {}, on() {} },
    ipcMain: { handle: (channel, handler) => handlers.set(channel, handler) },
  };
  const context = vm.createContext({
    __dirname, process, Buffer, console, fixture: { window, engine, url: frame.url },
    require: name => name === 'electron' ? electron : localRequire(name),
  });
  vm.runInContext(fs.readFileSync(main, 'utf8'), context);
  vm.runInContext('window = fixture.window; engine = fixture.engine; allowedUrl = fixture.url; bindOperations();', context);
  return {
    calls,
    invoke: args => handlers.get('research:anatomySection')({ sender: window.webContents, senderFrame: frame }, args),
  };
}

const sectionArgs = {
  collectionId: 'collection', snapshotId: 'snapshot', seriesId: 't1',
  center: [12, 12, 12], normal: [0, 0, 1], spacingMm: 0.5, layers: ['skull'],
};

test('section IPC forwards original false for ordinary sampling', async () => {
  const boundary = sectionBoundary();
  const args = { ...sectionArgs, original: false };
  await boundary.invoke(structuredClone(args));
  assert.deepEqual(boundary.calls, [['anatomySection', args, 300000]]);
});

test('section IPC forwards original true for checkerboard inspection', async () => {
  const boundary = sectionBoundary();
  const args = { ...sectionArgs, movingId: 'registered-ct', original: true };
  await boundary.invoke(structuredClone(args));
  assert.deepEqual(boundary.calls, [['anatomySection', args, 300000]]);
});

test('section IPC still rejects renderer-supplied paths', async () => {
  const boundary = sectionBoundary();
  await assert.rejects(boundary.invoke({ ...sectionArgs, original: true, path: '/renderer/scan' }), /Unsupported operation argument/);
  assert.equal(boundary.calls.length, 0);
});
