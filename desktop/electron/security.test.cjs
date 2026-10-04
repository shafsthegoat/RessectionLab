'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { plainArgs, assertSender } = require('./security.cjs');

const MAX_ARGUMENT_BYTES = 512 * 1024;

test('plainArgs preserves allowed argument records, including null-prototype records', () => {
  const args = { caseHash: 'case-1', config: { count: 3 }, toolIds: ['tool-1'] };
  assert.equal(plainArgs(args, ['caseHash', 'config', 'toolIds']), args);
  const nullPrototype = Object.assign(Object.create(null), { caseHash: 'case-1' });
  assert.equal(plainArgs(nullPrototype, ['caseHash']), nullPrototype);
  assert.deepEqual(plainArgs({}, []), {});
});

test('plainArgs rejects primitives, arrays and non-record objects accepted by structured cloning', () => {
  for (const value of [null, undefined, false, 3, 'case-1', [], new Date(), new Map(), new Set(), /case/,
    new Number(3), new String(''), new Uint8Array(), Object.create({ caseHash: 'inherited' })]) {
    assert.throws(() => plainArgs(value, ['caseHash']), undefined, `accepted ${Object.prototype.toString.call(value)}`);
  }
});

test('plainArgs rejects undeclared operations and prototype-like own keys', () => {
  for (const value of [{ caseHash: 'case-1', path: '/private/file' },
    JSON.parse('{"__proto__":{"path":"/private/file"}}'), { constructor: {} }, { prototype: {} }]) {
    assert.throws(() => plainArgs(value, ['caseHash']), /Unsupported operation argument/);
  }
});

test('plainArgs enforces its serialized byte limit, including multibyte text', () => {
  const overhead = Buffer.byteLength(JSON.stringify({ caseHash: '' }));
  const boundary = { caseHash: 'a'.repeat(MAX_ARGUMENT_BYTES - overhead) };
  assert.equal(plainArgs(boundary, ['caseHash']), boundary);
  assert.throws(() => plainArgs({ caseHash: `${boundary.caseHash}a` }, ['caseHash']), /limit/);
  const multibyte = { caseHash: '🧠'.repeat(MAX_ARGUMENT_BYTES / 4) };
  assert.ok(JSON.stringify(multibyte).length < MAX_ARGUMENT_BYTES);
  assert.throws(() => plainArgs(multibyte, ['caseHash']), /limit/);
});

test('plainArgs rejects arguments that cannot be serialized', () => {
  const circular = {};
  circular.config = circular;
  assert.throws(() => plainArgs(circular, ['config']));
  assert.throws(() => plainArgs({ config: 1n }, ['config']));
});

function trustedSender() {
  const allowedUrl = 'file:///Applications/RessectionLab.app/Contents/Resources/app/dist/index.html';
  const mainFrame = { url: allowedUrl };
  const webContents = { mainFrame };
  const window = { isDestroyed: () => false, webContents };
  const event = { sender: webContents, senderFrame: mainFrame };
  return { event, window, allowedUrl };
}

test('assertSender accepts only the trusted window main frame at its exact URL', () => {
  const { event, window, allowedUrl } = trustedSender();
  assert.doesNotThrow(() => assertSender(event, window, allowedUrl));
});

test('assertSender rejects another window, same-origin child frame, and destroyed window', () => {
  const { event, window, allowedUrl } = trustedSender();
  assert.throws(() => assertSender({ ...event, sender: {} }, window, allowedUrl), /Untrusted/);
  assert.throws(() => assertSender({ ...event, senderFrame: { url: allowedUrl } }, window, allowedUrl), /Untrusted/);
  assert.throws(() => assertSender(event, { ...window, isDestroyed: () => true }, allowedUrl), /Untrusted/);
  assert.throws(() => assertSender(event, undefined, allowedUrl), /Untrusted/);
  assert.throws(() => assertSender({ ...event, senderFrame: null }, window, allowedUrl), /Untrusted/);
});

test('assertSender rejects navigation, URL suffixes and lookalike origins', () => {
  for (const url of ['https://example.org/', 'file:///private/other.html',
    'file:///Applications/RessectionLab.app/Contents/Resources/app/dist/index.html?redirect=1',
    'file:///Applications/RessectionLab.app/Contents/Resources/app/dist/index.html#frame',
    'http://127.0.0.1:5173.evil.example/', 'http://localhost:5173/']) {
    const { event, window, allowedUrl } = trustedSender();
    event.senderFrame.url = url;
    assert.throws(() => assertSender(event, window, allowedUrl), /Untrusted/);
  }
});

test('assertSender accepts the configured development URL and rejects a different port', () => {
  const { event, window } = trustedSender();
  event.senderFrame.url = 'http://127.0.0.1:5173/';
  assert.doesNotThrow(() => assertSender(event, window, event.senderFrame.url));
  assert.throws(() => assertSender(event, window, 'http://127.0.0.1:5174/'), /Untrusted/);
});
