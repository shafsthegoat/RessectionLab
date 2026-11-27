'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const vm = require('node:vm');
const { createHash } = require('node:crypto');
const { createRequire } = require('node:module');
const { OPERATIONS } = require('./sidecar.cjs');
const { AssetRegistry } = require('./assets.cjs');

function mainHandlers(engine, casePath) {
  const handlers = new Map();
  const mainFile = path.join(__dirname, 'main.cjs');
  const localRequire = createRequire(mainFile);
  const frame = { url: 'file:///trusted-local-app.html' };
  const window = { isDestroyed: () => false, webContents: { mainFrame: frame } };
  const context = vm.createContext({
    __dirname, process, Buffer, console, __engine: engine, __window: window, __url: frame.url, __case: casePath,
    require: name => name === 'electron' ? {
      app: { requestSingleInstanceLock: () => false, quit() {}, on() {} },
      ipcMain: { handle: (name, callback) => handlers.set(name, callback) },
    } : localRequire(name),
  });
  vm.runInContext(fs.readFileSync(mainFile, 'utf8'), context);
  vm.runInContext('engine = __engine; window = __window; allowedUrl = __url; currentCasePath = __case; bindOperations();', context);
  const event = { sender: window.webContents, senderFrame: frame };
  return args => handlers.get('research:operativeSourceSurfaces')(event, args);
}

test('source surfaces come from the engine reply as checked transfer assets', async () => {
  assert.equal(OPERATIONS.has('operativeSourceSurfaces'), true);
  const folder = fs.mkdtempSync(path.join(os.tmpdir(), 'source-surfaces-'));
  try {
    const casePath = path.join(folder, 'case.ressectionlab');
    const transfers = path.join(folder, 'transfers');
    fs.mkdirSync(transfers);
    const positions = Buffer.from(new Float32Array([1, 2, 3, 4, 5, 6]).buffer), cells = Buffer.from(new Int32Array([7, -1]).buffer);
    fs.writeFileSync(path.join(transfers, 'positions.bin'), positions);
    fs.writeFileSync(path.join(transfers, 'cells.bin'), cells);
    const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
    const descriptor = (file, bytes, dtype, shape) => ({ path: path.join(transfers, file), dtype, shape, byteOrder: 'little', order: 'C', byteLength: bytes.length, sha256: digest(bytes) });
    let request;
    const reply = { caseHash: 'h', status: 'available', assetName: 'operative-source-surfaces.npz', header: {},
      arrays: { 'surface.pial.positions': descriptor('positions.bin', positions, 'float32', [2, 3]), 'surface.pial.cells': descriptor('cells.bin', cells, 'int32', [2]) } };
    const engine = { request: async (op, args) => { request = { op, args }; return structuredClone(reply); } };
    // The main process forwards the case path and returns the engine reply (the sidecar exposes its transfers).
    const result = await mainHandlers(engine, casePath)({ caseHash: 'h' });
    assert.deepEqual(JSON.parse(JSON.stringify(request)), { op: 'operativeSourceSurfaces', args: { caseHash: 'h', casePath } });
    assert.deepEqual(JSON.parse(JSON.stringify(result)), JSON.parse(JSON.stringify(reply)));
    // The sidecar exposes each array as a transfer asset without its disk path; reads are checked against the digest.
    const registry = new AssetRegistry(transfers);
    const exposed = await registry.expose(structuredClone(reply));
    const id = exposed.arrays['surface.pial.positions'].assetId;
    assert.equal(typeof id, 'string');
    assert.equal(exposed.arrays['surface.pial.positions'].path, undefined);
    assert.deepEqual([...new Float32Array((await registry.read(id)).slice().buffer)], [1, 2, 3, 4, 5, 6]);
    assert.deepEqual([...new Int32Array((await registry.read(exposed.arrays['surface.pial.cells'].assetId)).slice().buffer)], [7, -1]);
    // A changed file is refused; an unavailable reply passes through untouched.
    fs.writeFileSync(path.join(transfers, 'cells.bin'), Buffer.from(new Int32Array([8, -1]).buffer));
    await assert.rejects(registry.read(exposed.arrays['surface.pial.cells'].assetId), /checksum mismatch/);
    const unavailable = { caseHash: 'h', status: 'unavailable', code: 'ASSET_NOT_FOUND', reason: 'none' };
    assert.deepEqual(JSON.parse(JSON.stringify(await mainHandlers({ request: async () => unavailable }, casePath)({ caseHash: 'h' }))), unavailable);
  } finally {
    fs.rmSync(folder, { recursive: true, force: true });
  }
});
