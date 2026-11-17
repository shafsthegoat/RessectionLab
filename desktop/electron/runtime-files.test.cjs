'use strict';
const test = require('node:test'), assert = require('node:assert/strict');
const fs = require('node:fs/promises'), path = require('node:path');
const { createHash } = require('node:crypto');
const { runtimeFiles, stageElectronRuntime } = require('./runtime-files.cjs');
const sha = bytes => createHash('sha256').update(bytes).digest('hex');

test('runtime staging retains exact dependency closure and all repository development tools', async () => {
  const root = path.resolve(__dirname, '../..');
  const base = await fs.mkdtemp(path.join(root, 'testcases/runtime-stage-'));
  try {
    const source = path.join(base, 'source');
    await fs.cp(__dirname, source, { recursive: true });
    const hashes = {};
    for (const name of await fs.readdir(source)) {
      if ((await fs.lstat(path.join(source, name))).isFile()) hashes['electron/' + name] = sha(await fs.readFile(path.join(source, name)));
    }
    const staged = path.join(base, 'runtime');
    const receipt = await stageElectronRuntime(source, staged, hashes);
    assert.equal(receipt.fileCount, 34);
    assert.deepEqual((await fs.readdir(staged)).sort(), [...runtimeFiles].sort());
    for (const name of runtimeFiles) assert.equal(sha(await fs.readFile(path.join(staged, name))), hashes['electron/' + name]);
    for (const name of ['main.cjs', 'preload.cjs', 'sidecar.cjs', 'prefix-engine.cjs', 'frozen-cusa.cjs']) assert(receipt.files[name]);
    for (const name of ['package-macos.cjs', 'runtime-files.test.cjs', 'benchmark-surgical-playback.cjs', 'vascular_playback_monitor.py']) {
      assert(receipt.excludedSourcePaths.includes('electron/' + name));
      assert((await fs.stat(path.join(source, name))).isFile());
      await assert.rejects(fs.stat(path.join(staged, name)), { code: 'ENOENT' });
    }
    await assert.rejects(stageElectronRuntime(source, staged, hashes), { code: 'EEXIST' });
    const original = await fs.readFile(path.join(source, 'main.cjs'));
    await fs.appendFile(path.join(source, 'main.cjs'), '\nrequire("./new-runtime.cjs");\n');
    await assert.rejects(stageElectronRuntime(source, path.join(base, 'changed'), hashes), /changed/);
    hashes['electron/main.cjs'] = sha(await fs.readFile(path.join(source, 'main.cjs')));
    await assert.rejects(stageElectronRuntime(source, path.join(base, 'new-dependency'), hashes), /not allowlisted/);
    await fs.writeFile(path.join(source, 'main.cjs'), Buffer.concat([original, Buffer.from('\nrequire(selectedModule);\n')]));
    hashes['electron/main.cjs'] = sha(await fs.readFile(path.join(source, 'main.cjs')));
    await assert.rejects(stageElectronRuntime(source, path.join(base, 'computed'), hashes), /Computed/);
    await fs.writeFile(path.join(source, 'main.cjs'), original);
    hashes['electron/main.cjs'] = sha(original);
    await fs.unlink(path.join(source, 'preload.cjs'));
    await assert.rejects(stageElectronRuntime(source, path.join(base, 'missing'), hashes), { code: 'ENOENT' });
  } finally { await fs.rm(base, { recursive: true, force: true }); }
});
