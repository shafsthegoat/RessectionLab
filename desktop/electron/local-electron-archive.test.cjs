'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { localElectronArchive, verifyLocalElectronArchive, MAX_ARCHIVE_BYTES } = require('./local-electron-archive.cjs');

const version = '44.5.1';
const fileName = `electron-v${version}-darwin-arm64.zip`;
const bytes = Buffer.from('PK\x03\x04generated archive bytes; never extracted or executed');
const checksum = createHash('sha256').update(bytes).digest('hex');
const checksums = { [fileName]: checksum };
async function fixture(t) {
  const directory = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'electron-archive-control-')));
  t.after(() => fs.rm(directory, { recursive: true, force: true }));
  const filename = path.join(directory, fileName);
  await fs.writeFile(filename, bytes);
  return { directory, filename };
}

test('absent option preserves default download/cache behavior without file access', async () => {
  assert.equal(await localElectronArchive(undefined, null, null), null);
  await verifyLocalElectronArchive(null);
});

test('exact named arm64 bytes bind to the upstream checksum and survive postcheck', async t => {
  const f = await fixture(t);
  const record = await localElectronArchive(f.directory, version, checksums);
  assert.equal(record.directory, f.directory);
  assert.equal(record.fileName, fileName);
  assert.equal(record.bytes, bytes.length);
  assert.equal(record.sha256, checksum);
  assert.equal(record.version, version);
  assert.ok(Object.isFrozen(record) && Object.isFrozen(record.identity));
  await verifyLocalElectronArchive(record);
  assert.deepEqual(await fs.readFile(f.filename), bytes);
});

test('present empty, relative and malformed directories do not become default mode', async () => {
  for (const directory of ['', '.', '../cache', null, 123]) {
    await assert.rejects(localElectronArchive(directory, version, checksums), /absolute directory/);
  }
});

test('missing directory or exact standard filename fails without creating it', async t => {
  const f = await fixture(t);
  const missing = path.join(f.directory, 'missing');
  await assert.rejects(localElectronArchive(missing, version, checksums), /ENOENT/);
  await assert.rejects(fs.stat(missing), /ENOENT/);
  await fs.rename(f.filename, path.join(f.directory, 'foreign-runtime.zip'));
  await assert.rejects(localElectronArchive(f.directory, version, checksums), /ENOENT/);
  await assert.rejects(fs.stat(f.filename), /ENOENT/);
});

test('exact version and its arm64 checksum are mandatory', async t => {
  const f = await fixture(t);
  for (const invalid of ['^44.5.1', '../44.5.1', '044.5.1', '44.5', '', null]) {
    await assert.rejects(localElectronArchive(f.directory, invalid, checksums), /exact Electron version/);
  }
  for (const invalid of [{}, { [fileName]: checksum.toUpperCase() }, { [fileName]: 'bad' }, { 'electron-v44.5.1-darwin-x64.zip': checksum }]) {
    await assert.rejects(localElectronArchive(f.directory, version, invalid), /installed arm64 checksum/);
  }
  await assert.rejects(localElectronArchive(f.directory, '44.5.2', { 'electron-v44.5.2-darwin-arm64.zip': checksum }), /ENOENT/);
});

test('incorrect bytes and empty archives fail before use', async t => {
  const f = await fixture(t);
  await fs.writeFile(f.filename, Buffer.alloc(bytes.length, 7));
  await assert.rejects(localElectronArchive(f.directory, version, checksums), /checksum differs/);
  await fs.writeFile(f.filename, Buffer.alloc(0));
  await assert.rejects(localElectronArchive(f.directory, version, checksums), /size.*bounded/);
});

test('regular file and bounded size are checked before hashing', async t => {
  const f = await fixture(t);
  await fs.unlink(f.filename); await fs.mkdir(f.filename);
  await assert.rejects(localElectronArchive(f.directory, version, checksums), /regular archive/);
  await fs.rmdir(f.filename);
  const handle = await fs.open(f.filename, 'w');
  try { await handle.truncate(MAX_ARCHIVE_BYTES + 1); } finally { await handle.close(); }
  await assert.rejects(localElectronArchive(f.directory, version, checksums), /size.*bounded/);
});

test('archive symlink is rejected even with exact expected target bytes', async t => {
  const f = await fixture(t), target = path.join(f.directory, 'original.zip');
  await fs.rename(f.filename, target); await fs.symlink(target, f.filename);
  await assert.rejects(localElectronArchive(f.directory, version, checksums), /not a symlink/);
});

test('directory and ancestor symlinks are rejected', async t => {
  const f = await fixture(t), inner = path.join(f.directory, 'inner'), link = path.join(f.directory, 'redirect');
  await fs.mkdir(inner); await fs.writeFile(path.join(inner, fileName), bytes);
  await fs.symlink(inner, link);
  await assert.rejects(localElectronArchive(link, version, checksums), /directory symlinks/);
  await fs.mkdir(path.join(inner, 'child')); await fs.writeFile(path.join(inner, 'child', fileName), bytes);
  await assert.rejects(localElectronArchive(path.join(link, 'child'), version, checksums), /directory symlinks/);
});

test('postcheck rejects changed bytes and does not repair the archive', async t => {
  const f = await fixture(t), record = await localElectronArchive(f.directory, version, checksums);
  const changed = Buffer.alloc(bytes.length, 9); await fs.writeFile(f.filename, changed);
  await assert.rejects(verifyLocalElectronArchive(record), /checksum differs/);
  assert.deepEqual(await fs.readFile(f.filename), changed);
});

test('postcheck rejects replaced identical bytes and missing or redirected archive', async t => {
  const f = await fixture(t), record = await localElectronArchive(f.directory, version, checksums);
  const replacement = path.join(f.directory, 'replacement.zip');
  await fs.writeFile(replacement, bytes); await fs.rename(replacement, f.filename);
  await assert.rejects(verifyLocalElectronArchive(record), /replaced or modified/);
  await fs.unlink(f.filename);
  await assert.rejects(verifyLocalElectronArchive(record), /ENOENT/);
  await fs.writeFile(replacement, bytes); await fs.symlink(replacement, f.filename);
  await assert.rejects(verifyLocalElectronArchive(record), /not a symlink/);
});
