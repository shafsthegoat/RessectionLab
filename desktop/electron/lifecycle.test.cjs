'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { publishBundle } = require('./publish.cjs');
const { createLogger } = require('./logging.cjs');

async function workspace(t) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'ressectionlab-lifecycle-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const current = path.join(root, 'current'); const staged = path.join(root, 'staged');
  await fs.mkdir(current); await fs.mkdir(staged);
  await fs.writeFile(path.join(current, 'marker'), 'prior working app');
  await fs.writeFile(path.join(staged, 'marker'), 'verified replacement');
  return { root, current, staged };
}

test('failed publication restores the previous app at its stable path', async t => {
  const { current, staged } = await workspace(t);
  const injected = { ...fs, rename: async (from, to) => {
    if (from === staged) throw Object.assign(new Error('injected publication failure'), { code: 'EACCES' });
    return fs.rename(from, to);
  } };
  await assert.rejects(publishBundle(staged, current, injected), /injected publication failure/);
  assert.equal(await fs.readFile(path.join(current, 'marker'), 'utf8'), 'prior working app');
  assert.equal(await fs.readFile(path.join(staged, 'marker'), 'utf8'), 'verified replacement');
});

test('successful publication retains the prior app and installs the verified replacement', async t => {
  const { current, staged } = await workspace(t);
  const { previousDirectory } = await publishBundle(staged, current);
  assert.equal(await fs.readFile(path.join(previousDirectory, 'marker'), 'utf8'), 'prior working app');
  assert.equal(await fs.readFile(path.join(current, 'marker'), 'utf8'), 'verified replacement');
});

test('diagnostics rotate during a long-running app session', async t => {
  const { root } = await workspace(t);
  const directory = path.join(root, 'logs');
  const log = createLogger(directory, { maxBytes: 120, stderr: { write() {} } });
  log('first record'); log('second record'); log('third record'); log('fourth record');
  const active = await fs.readFile(path.join(directory, 'desktop.log'), 'utf8');
  const previous = await fs.readFile(path.join(directory, 'desktop.previous.log'), 'utf8');
  assert.ok(Buffer.byteLength(active) <= 120);
  assert.ok(Buffer.byteLength(previous) <= 120);
  assert.match(active, /fourth record/);
  assert.match(previous, /first record|second record|third record/);
});
