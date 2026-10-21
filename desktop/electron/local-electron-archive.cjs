'use strict';

const fs = require('node:fs/promises');
const { constants } = require('node:fs');
const path = require('node:path');
const { createHash } = require('node:crypto');

const VERSION = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?$/;
const MAX_ARCHIVE_BYTES = 512 * 1024 * 1024;
const need = (condition, message) => { if (!condition) throw new Error(`Local Electron archive: ${message}`); };
const identity = stat => Object.fromEntries(['dev', 'ino', 'size', 'mtimeNs', 'ctimeNs'].map(key => [key, String(stat[key])]));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

/** Optional host build input. The caller first binds version to its captured lock. */
async function localElectronArchive(directory, version, checksums) {
  if (directory === undefined) return null;
  need(typeof directory === 'string' && directory.length > 0 && path.isAbsolute(directory), 'an absolute directory is required');
  need(typeof version === 'string' && VERSION.test(version), 'an exact Electron version is required');
  const fileName = `electron-v${version}-darwin-arm64.zip`;
  const expected = checksums?.[fileName];
  need(typeof expected === 'string' && /^[a-f0-9]{64}$/.test(expected), 'the installed arm64 checksum is required');
  const canonical = path.resolve(directory);
  need(await fs.realpath(canonical) === canonical, 'directory symlinks are not permitted');
  need((await fs.lstat(canonical)).isDirectory(), 'expected a directory');
  const filename = path.join(canonical, fileName);
  const namedBefore = await fs.lstat(filename, { bigint: true });
  need(namedBefore.isFile() && !namedBefore.isSymbolicLink(), 'expected a regular archive, not a symlink');
  need(namedBefore.size > 0n && namedBefore.size <= BigInt(MAX_ARCHIVE_BYTES), 'archive size is outside the bounded read limit');
  const handle = await fs.open(filename, constants.O_RDONLY | constants.O_NOFOLLOW);
  let digest;
  try {
    const before = await handle.stat({ bigint: true });
    need(before.isFile() && same(identity(namedBefore), identity(before)), 'archive changed before reading');
    const hash = createHash('sha256'), buffer = Buffer.alloc(1024 * 1024);
    let offset = 0;
    while (offset < Number(before.size)) {
      const { bytesRead } = await handle.read(buffer, 0, Math.min(buffer.length, Number(before.size) - offset), offset);
      need(bytesRead > 0, 'archive became incomplete');
      hash.update(buffer.subarray(0, bytesRead)); offset += bytesRead;
    }
    const after = await handle.stat({ bigint: true });
    const namedAfter = await fs.lstat(filename, { bigint: true });
    need(namedAfter.isFile() && same(identity(before), identity(after)) && same(identity(after), identity(namedAfter)), 'archive changed while reading');
    need(await fs.realpath(canonical) === canonical, 'directory changed while reading');
    digest = hash.digest('hex');
  } finally { await handle.close(); }
  need(digest === expected, 'checksum differs from the installed upstream checksum');
  return Object.freeze({ directory: canonical, fileName, version, bytes: Number(namedBefore.size), sha256: digest, identity: Object.freeze(identity(namedBefore)) });
}

/** Local ZIP mode bypasses the downloader; recheck before signing/publication. */
async function verifyLocalElectronArchive(record) {
  if (record === null) return;
  const current = await localElectronArchive(record.directory, record.version, { [record.fileName]: record.sha256 });
  need(current.bytes === record.bytes && same(current.identity, record.identity), 'archive was replaced or modified during packaging');
}

module.exports = { localElectronArchive, verifyLocalElectronArchive, MAX_ARCHIVE_BYTES };
