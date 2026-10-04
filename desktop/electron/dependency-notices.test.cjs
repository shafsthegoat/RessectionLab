'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { createHash } = require('node:crypto');
const { captureRendererNotices, capturePythonNotices, captureElectronNotices } = require('./dependency-notices.cjs');
const sha = bytes => createHash('sha256').update(bytes).digest('hex');

async function temporary(t) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'ressection-notices-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  return root;
}
async function write(root, name, value) {
  const target = path.join(root, name);
  await fs.mkdir(path.dirname(target), { recursive: true });
  await fs.writeFile(target, typeof value === 'object' ? JSON.stringify(value) : value);
}
async function packageAt(root, name, extra = {}, license = true) {
  const prefix = `node_modules/${name}`;
  await write(root, `${prefix}/package.json`, { name, version: '1.0.0', main: 'index.js', license: 'MIT', ...extra });
  await write(root, `${prefix}/index.js`, 'module.exports = {};');
  if (license) await write(root, `${prefix}/LICENSE`, `Exact upstream fixture notice for ${name}\n`);
}

test('captures runtime transitive closure, omits dev tools and reports absent upstream text', async t => {
  const root = await temporary(t);
  await write(root, 'package.json', { dependencies: { first: '1.0.0' }, devDependencies: { buildtool: '1.0.0' } });
  await write(root, 'pnpm-lock.yaml', 'packages:\n  first@1.0.0:\n  second@1.0.0:\n');
  await packageAt(root, 'first', { dependencies: { second: '1.0.0' }, peerDependencies: { '@types/first': '*' } });
  await packageAt(root, 'second', {}, false);
  await packageAt(root, 'buildtool');
  const output = path.join(root, 'notices'); await fs.mkdir(output);
  const result = await captureRendererNotices(root, output);
  assert.deepEqual(result.packages.map(p => p.name), ['first', 'second']);
  assert.equal(result.unresolved[0].component, 'second@1.0.0');
  assert.equal(result.omitted_type_peers.length, 1);
  assert.equal(await fs.readFile(path.join(output, result.packages[0].license_files[0]), 'utf8'), 'Exact upstream fixture notice for first\n');
});

test('refuses installed renderer versions missing from captured lock', async t => {
  const root = await temporary(t);
  await write(root, 'package.json', { dependencies: { first: '*' } });
  await write(root, 'pnpm-lock.yaml', 'packages:\n  first@2.0.0:\n');
  await packageAt(root, 'first');
  const output = path.join(root, 'notices'); await fs.mkdir(output);
  await assert.rejects(captureRendererNotices(root, output), /absent from captured pnpm lock/);
});

async function engineFixture(root) {
  const bytes = Buffer.from('frozen executable fixture');
  const payload = Buffer.from('frozen native library fixture');
  const text = Buffer.from('Exact upstream notice\n');
  await write(root, 'engine/engine', bytes.toString());
  await write(root, 'engine/_internal/library.dylib', payload.toString());
  await write(root, 'source/python/lib/LICENSE', text.toString());
  const manifest = { schema_version: 1, engine_sha256: sha(bytes),
    files: { 'python/lib/LICENSE': { sha256: sha(text), bytes: text.length } },
    collected_payload: { '_internal/library.dylib': { sha256: sha(payload), bytes: payload.length } } };
  await write(root, 'source/inventory.json', manifest);
  const output = path.join(root, 'out'); await fs.mkdir(output);
  return { source: path.join(root, 'source'), output, executable: path.join(root, 'engine/engine'), manifest };
}

test('binds notices to numerical executable and actual native payload', async t => {
  const root = await temporary(t); const f = await engineFixture(root);
  await capturePythonNotices(f.source, f.output, f.executable);
  assert.equal(await fs.readFile(path.join(f.output, 'numerical-engine/python/lib/LICENSE'), 'utf8'), 'Exact upstream notice\n');
  await write(root, 'engine/_internal/library.dylib', 'different native library');
  await assert.rejects(capturePythonNotices(f.source, path.join(root, 'another'), f.executable), /Notice hash mismatch/);
});

test('rejects stale engine notice inventories and altered upstream notice bytes', async t => {
  const root = await temporary(t); const f = await engineFixture(root);
  await write(root, 'engine/engine', 'replacement executable');
  await assert.rejects(capturePythonNotices(f.source, f.output, f.executable), /exact numerical engine/);
  await write(root, 'engine/engine', 'frozen executable fixture');
  await write(root, 'source/python/lib/LICENSE', 'tampered text');
  await assert.rejects(capturePythonNotices(f.source, f.output, f.executable), /Notice hash mismatch/);
});

test('rejects changed internal payload links even when the target bytes match', async t => {
  const root = await temporary(t); const f = await engineFixture(root);
  await fs.copyFile(path.join(root, 'engine/_internal/library.dylib'), path.join(root, 'engine/_internal/other.dylib'));
  await fs.symlink('other.dylib', path.join(root, 'engine/_internal/alias.dylib'));
  f.manifest.collected_payload['_internal/alias.dylib'] = {
    ...f.manifest.collected_payload['_internal/library.dylib'], symlink: 'library.dylib' };
  await write(root, 'source/inventory.json', f.manifest);
  await assert.rejects(capturePythonNotices(f.source, f.output, f.executable), /payload link changed/);
});

test('rejects notice inventory traversal and external symlinks', async t => {
  const root = await temporary(t); const f = await engineFixture(root);
  f.manifest.files['../outside'] = { sha256: sha(Buffer.from('x')), bytes: 1 };
  await write(root, 'source/inventory.json', f.manifest);
  await assert.rejects(capturePythonNotices(f.source, f.output, f.executable), /Unsafe notice inventory path/);
  delete f.manifest.files['../outside'];
  await write(root, 'source/inventory.json', f.manifest);
  await write(root, 'outside', 'Exact upstream notice\n');
  await fs.unlink(path.join(root, 'source/python/lib/LICENSE'));
  await fs.symlink(path.join(root, 'outside'), path.join(root, 'source/python/lib/LICENSE'));
  await assert.rejects(capturePythonNotices(f.source, f.output, f.executable), /escapes its capture/);
});

test('requires both Electron and Chromium upstream notices', async t => {
  const root = await temporary(t); const output = path.join(root, 'out'); await fs.mkdir(output);
  await write(root, 'electron/dist/LICENSE', 'Electron upstream license');
  await assert.rejects(captureElectronNotices(path.join(root, 'electron'), '1.0.0', output), /ENOENT/);
});
