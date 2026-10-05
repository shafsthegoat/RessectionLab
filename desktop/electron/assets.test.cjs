'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { AssetRegistry, MAX_ASSET_BYTES } = require('./assets.cjs');

async function fixture(t) {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'ressectionlab-assets-test-'));
  t.after(() => fs.rm(directory, { recursive: true, force: true }));
  const root = path.join(directory, 'session');
  await fs.mkdir(root, { mode: 0o700 });
  const bytes = Buffer.from([0, 1, 2, 3, 4, 5, 6, 7]);
  const diskPath = path.join(root, 'array.bin');
  await fs.writeFile(diskPath, bytes);
  const descriptor = { path: diskPath, dtype: 'uint16', shape: [2, 2], byteLength: bytes.length,
    byteOrder: 'little', order: 'C', sha256: createHash('sha256').update(bytes).digest('hex') };
  return { directory, root, bytes, diskPath, descriptor, registry: new AssetRegistry(root) };
}

test('registered arrays expose opaque, unique IDs and preserve verified bytes', async t => {
  const { descriptor, registry, bytes, diskPath } = await fixture(t);
  const first = await registry.register(descriptor);
  const second = await registry.register({ ...descriptor, sha256: `sha256:${descriptor.sha256}`, assetId: 'untrusted-id' });
  for (const exposed of [first, second]) {
    assert.equal(Object.hasOwn(exposed, 'path'), false);
    assert.match(exposed.assetId, /^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/);
    assert.equal(JSON.stringify(exposed).includes(diskPath), false);
    const result = await registry.read(exposed.assetId);
    assert.ok(result instanceof Uint8Array);
    assert.deepEqual(Buffer.from(result), bytes);
  }
  assert.notEqual(first.assetId, second.assetId);
  assert.notEqual(second.assetId, 'untrusted-id');
  assert.equal(descriptor.path, diskPath, 'registration must not mutate the engine descriptor');
});

test('recursive exposure handles nested arrays while stripping transfer paths', async t => {
  const { descriptor, registry } = await fixture(t);
  const exposed = await registry.expose({ caseHash: 'case-1', nested: [null, 5, { image: descriptor }] });
  assert.equal(exposed.caseHash, 'case-1');
  assert.deepEqual(exposed.nested.slice(0, 2), [null, 5]);
  assert.equal(Object.hasOwn(exposed.nested[2].image, 'path'), false);
  assert.deepEqual(exposed.nested[2].image.shape, [2, 2]);
  assert.equal((await registry.read(exposed.nested[2].image.assetId)).length, 8);
});

test('unknown IDs, filesystem paths, and expired session IDs are rejected', async t => {
  const { descriptor, registry, diskPath } = await fixture(t);
  const exposed = await registry.register(descriptor);
  for (const id of [undefined, null, 0, {}, [], diskPath, '../array.bin', 'unknown', new String(exposed.assetId)]) {
    await assert.rejects(registry.read(id), /Unknown or expired/);
  }
  registry.clear();
  await assert.rejects(registry.read(exposed.assetId), /Unknown or expired/);
});

test('registration rejects traversal, sibling-prefix directories, and escaped symlinks', async t => {
  const { directory, root, descriptor, registry, bytes } = await fixture(t);
  const outside = path.join(directory, 'outside.bin');
  await fs.writeFile(outside, bytes);
  const sibling = path.join(directory, 'session-other');
  await fs.mkdir(sibling);
  await fs.writeFile(path.join(sibling, 'array.bin'), bytes);
  const escapedLink = path.join(root, 'escape.bin');
  await fs.symlink(outside, escapedLink);
  for (const diskPath of [path.join(root, '..', 'outside.bin'), path.join(sibling, 'array.bin'), escapedLink, root]) {
    await assert.rejects(registry.register({ ...descriptor, path: diskPath }), /escapes this session/);
  }
});

test('registration permits a symlink resolving to an asset inside the same session', async t => {
  const { root, descriptor, registry, bytes, diskPath } = await fixture(t);
  const link = path.join(root, 'internal-link.bin');
  await fs.symlink(diskPath, link);
  const exposed = await registry.register({ ...descriptor, path: link });
  assert.deepEqual(Buffer.from(await registry.read(exposed.assetId)), bytes);
});

test('registered file replacement with an escaped symlink is rejected on read', async t => {
  const { directory, descriptor, registry, bytes, diskPath } = await fixture(t);
  const exposed = await registry.register(descriptor);
  const outside = path.join(directory, 'outside.bin');
  await fs.writeFile(outside, bytes);
  await fs.unlink(diskPath);
  await fs.symlink(outside, diskPath);
  await assert.rejects(registry.read(exposed.assetId), /location changed/);
});

test('changed file size, same-size corruption, and deletion are detected', async t => {
  const { descriptor, registry, bytes, diskPath } = await fixture(t);
  const exposed = await registry.register(descriptor);
  await fs.writeFile(diskPath, Buffer.concat([bytes, Buffer.from([8])]));
  await assert.rejects(registry.read(exposed.assetId), /size changed/);
  await fs.writeFile(diskPath, Buffer.alloc(bytes.length, 255));
  await assert.rejects(registry.read(exposed.assetId), /checksum mismatch/);
  await fs.unlink(diskPath);
  await assert.rejects(registry.read(exposed.assetId), { code: 'ENOENT' });
});

test('directories cannot be read as transfer assets', async t => {
  const { root, descriptor, registry } = await fixture(t);
  const folder = path.join(root, 'folder');
  await fs.mkdir(folder);
  const exposed = await registry.register({ ...descriptor, path: folder });
  await assert.rejects(registry.read(exposed.assetId), /size changed/);
});

test('invalid dtype, shape, dimensions and overflowing layouts are rejected', async t => {
  const { descriptor, registry } = await fixture(t);
  for (const patch of [{ dtype: 'complex64' }, { dtype: 'constructor' }, { dtype: null },
    { shape: [] }, { shape: [1, 1, 1, 1, 4] }, { shape: '2,2' }, { shape: [0, 4] },
    { shape: [-1, 4] }, { shape: [0.5, 8] }, { shape: [NaN] }, { shape: [Infinity] },
    { shape: ['2', 2] }, { shape: [Number.MAX_SAFE_INTEGER + 1] },
    { shape: [Number.MAX_SAFE_INTEGER, 2], byteLength: Number.MAX_SAFE_INTEGER * 4 },
    { shape: [MAX_ASSET_BYTES / 2 + 1], byteLength: MAX_ASSET_BYTES + 2 }]) {
    await assert.rejects(registry.register({ ...descriptor, ...patch }), /Invalid transfer array/);
  }
});

test('declared byte lengths, endianness and memory order must match the array contract', async t => {
  const { descriptor, registry } = await fixture(t);
  for (const patch of [{ byteLength: 7 }, { byteLength: 9 }, { byteLength: '8' }, { byteLength: -8 },
    { byteLength: NaN }, { byteOrder: 'big' }, { byteOrder: undefined }, { order: 'F' }, { order: undefined }]) {
    await assert.rejects(registry.register({ ...descriptor, ...patch }), /Invalid transfer array layout/);
  }
});

test('invalid checksum syntax is rejected and a valid but wrong checksum fails on read', async t => {
  const { descriptor, registry } = await fixture(t);
  for (const sha256 of [undefined, null, '', 'a'.repeat(63), 'a'.repeat(65), 'g'.repeat(64),
    'A'.repeat(64), `md5:${descriptor.sha256}`, `sha256:sha256:${descriptor.sha256}`]) {
    await assert.rejects(registry.register({ ...descriptor, sha256 }), /Invalid asset checksum/);
  }
  const exposed = await registry.register({ ...descriptor, sha256: '0'.repeat(64) });
  await assert.rejects(registry.read(exposed.assetId), /checksum mismatch/);
});
