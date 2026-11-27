'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const vm = require('node:vm');
const zlib = require('node:zlib');
const { createHash } = require('node:crypto');
const { createRequire } = require('node:module');
const { OPERATIONS } = require('./sidecar.cjs');

function npy(descr, shape, bytes) {
  let header = `{'descr': '${descr}', 'fortran_order': False, 'shape': (${shape.join(', ')}${shape.length === 1 ? ',' : ''}), }`;
  while ((10 + header.length + 1) % 64) header += ' ';
  header += '\n';
  const head = Buffer.alloc(10);
  head.write('\x93NUMPY', 0, 'latin1');
  head[6] = 1; head[7] = 0; head.writeUInt16LE(header.length, 8);
  return Buffer.concat([head, Buffer.from(header, 'latin1'), Buffer.from(bytes)]);
}

/** A minimal .npz: one deflated and one stored member. */
function npz(members) {
  const locals = [], central = [];
  let offset = 0;
  for (const [index, [name, data]] of members.entries()) {
    const deflate = index === 0, body = deflate ? zlib.deflateRawSync(data) : data;
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0); local.writeUInt16LE(deflate ? 8 : 0, 8);
    local.writeUInt32LE(body.length, 18); local.writeUInt32LE(data.length, 22); local.writeUInt16LE(name.length, 26);
    const header = Buffer.alloc(46);
    header.writeUInt32LE(0x02014b50, 0); header.writeUInt16LE(deflate ? 8 : 0, 10);
    header.writeUInt32LE(body.length, 20); header.writeUInt32LE(data.length, 24); header.writeUInt16LE(name.length, 28); header.writeUInt32LE(offset, 42);
    locals.push(local, Buffer.from(name), body);
    central.push(header, Buffer.from(name));
    offset += 30 + name.length + body.length;
  }
  const directory = Buffer.concat(central), end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(members.length, 8); end.writeUInt16LE(members.length, 10);
  end.writeUInt32LE(directory.length, 12); end.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, directory, end]);
}

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

test('source surfaces are read from the checked asset beside the case and must match the engine digests', async () => {
  assert.equal(OPERATIONS.has('operativeSourceSurfaces'), true);
  const folder = fs.mkdtempSync(path.join(os.tmpdir(), 'source-surfaces-'));
  try {
    const casePath = path.join(folder, 'case.ressectionlab');
    const positions = Buffer.from(new Float32Array([1, 2, 3, 4, 5, 6]).buffer), cells = Buffer.from(new Int32Array([7, -1]).buffer);
    fs.writeFileSync(path.join(folder, 'operative-source-surfaces.npz'),
      npz([['surface.pial.positions.npy', npy('<f4', [2, 3], positions)], ['surface.pial.cells.npy', npy('<i4', [2], cells)]]));
    const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
    const descriptor = (bytes, dtype, shape) => ({ path: '/never/forwarded', dtype, shape, byteOrder: 'little', order: 'C', byteLength: bytes.length, sha256: digest(bytes) });
    let request;
    const reply = { caseHash: 'h', status: 'available', assetName: 'operative-source-surfaces.npz', header: {},
      arrays: { 'surface.pial.positions': descriptor(positions, 'float32', [2, 3]), 'surface.pial.cells': descriptor(cells, 'int32', [2]) } };
    const engine = { request: async (op, args) => { request = { op, args }; return structuredClone(reply); } };
    const result = await mainHandlers(engine, casePath)({ caseHash: 'h' });
    assert.deepEqual(JSON.parse(JSON.stringify(request)), { op: 'operativeSourceSurfaces', args: { caseHash: 'h', casePath } });
    assert.deepEqual([...new Float32Array(result.arrays['surface.pial.positions'].data.slice().buffer)], [1, 2, 3, 4, 5, 6]);
    assert.deepEqual([...new Int32Array(result.arrays['surface.pial.cells'].data.slice().buffer)], [7, -1]);
    // A changed array is refused; an unavailable reply passes through untouched.
    reply.arrays['surface.pial.cells'].sha256 = 'f'.repeat(64);
    await assert.rejects(mainHandlers(engine, casePath)({ caseHash: 'h' }), /differs from the checked asset/);
    const unavailable = { caseHash: 'h', status: 'unavailable', code: 'ASSET_NOT_FOUND', reason: 'none' };
    assert.deepEqual(JSON.parse(JSON.stringify(await mainHandlers({ request: async () => unavailable }, casePath)({ caseHash: 'h' }))), unavailable);
  } finally {
    fs.rmSync(folder, { recursive: true, force: true });
  }
});
