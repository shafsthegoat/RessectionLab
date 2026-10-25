'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const {spawn} = require('node:child_process');
const {once} = require('node:events');
const {Sidecar} = require('./sidecar.cjs');

test('stop waits for actual child exit even when shutdown and SIGTERM are ignored', async () => {
  const transferDir = await fs.mkdtemp(path.join(os.tmpdir(), 'sidecar-exit-test-'));
  const child = spawn(process.execPath, ['-e',
    "process.on('SIGTERM',()=>{});process.stdin.resume();setInterval(()=>{},1000);process.stdout.write('ready');"],
    {stdio:['pipe','pipe','pipe']});
  const engine = Object.assign(Object.create(Sidecar.prototype), {child,transferDir,closed:false,stopping:false});
  try {
    await once(child.stdout, 'data');
    const began = performance.now();
    const stopping = engine.stop();
    assert.equal(engine.stopping, true);
    await assert.rejects(engine.request('ping'), /not running/);
    await Promise.all([stopping,engine.stop()]);
    assert.equal(child.signalCode, 'SIGKILL');
    assert.ok(performance.now()-began >= 2900);
    await assert.rejects(fs.stat(transferDir), {code:'ENOENT'});
  } finally {
    if (child.exitCode === null && child.signalCode === null) {child.kill('SIGKILL');await once(child,'exit');}
    await fs.rm(transferDir,{recursive:true,force:true});
  }
});

test('stop cleans transfer files after a child that never spawned without waiting for an impossible exit event', async () => {
  const transferDir = await fs.mkdtemp(path.join(os.tmpdir(), 'sidecar-exit-test-'));
  const engine = Object.assign(Object.create(Sidecar.prototype), {child:{pid:undefined},transferDir,closed:true});
  await engine.stop();
  await assert.rejects(fs.stat(transferDir), {code:'ENOENT'});
});
