'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const { createHash, randomUUID } = require('node:crypto');

const DTYPE_BYTES = Object.freeze({ uint8: 1, int8: 1, uint16: 2, int16: 2, uint32: 4, int32: 4, float32: 4, float64: 8 });
const MAX_ASSET_BYTES = 512 * 1024 * 1024;

class AssetRegistry {
  constructor(root) { this.root = path.resolve(root); this.records = new Map(); }
  clear() { this.records.clear(); }

  async register(descriptor) {
    const resolved = await fs.realpath(descriptor.path);
    const root = await fs.realpath(this.root);
    if (!resolved.startsWith(root + path.sep)) throw new Error('Transfer asset escapes this session');
    const bytesPerItem = DTYPE_BYTES[descriptor.dtype];
    const shape = descriptor.shape;
    if (!bytesPerItem || !Array.isArray(shape) || shape.length < 1 || shape.length > 4
      || shape.some(value => !Number.isSafeInteger(value) || value < 1)) throw new Error('Invalid transfer array shape or dtype');
    const expectedBytes = shape.reduce((product, dimension) => product * dimension, bytesPerItem);
    if (!Number.isSafeInteger(expectedBytes) || expectedBytes > MAX_ASSET_BYTES || expectedBytes !== descriptor.byteLength
      || descriptor.byteOrder !== 'little' || descriptor.order !== 'C') throw new Error('Invalid transfer array layout');
    const sha256 = String(descriptor.sha256).replace(/^sha256:/, '');
    if (!/^[a-f0-9]{64}$/.test(sha256)) throw new Error('Invalid asset checksum');
    const assetId = randomUUID();
    this.records.set(assetId, { path: resolved, byteLength: expectedBytes, sha256 });
    const { path: _diskPath, ...publicDescriptor } = descriptor;
    return { ...publicDescriptor, assetId };
  }

  async expose(value) {
    if (Array.isArray(value)) return Promise.all(value.map(item => this.expose(item)));
    if (value && typeof value === 'object') {
      if (typeof value.path === 'string' && 'dtype' in value && 'byteLength' in value) return this.register(value);
      return Object.fromEntries(await Promise.all(Object.entries(value).map(async ([key, item]) => [key, await this.expose(item)])));
    }
    return value;
  }

  async read(assetId) {
    if (typeof assetId !== 'string' || !this.records.has(assetId)) throw new Error('Unknown or expired transfer asset');
    const record = this.records.get(assetId);
    const resolved = await fs.realpath(record.path);
    if (resolved !== record.path || !resolved.startsWith(await fs.realpath(this.root) + path.sep)) throw new Error('Transfer asset location changed');
    const stat = await fs.stat(resolved);
    if (!stat.isFile() || stat.size !== record.byteLength) throw new Error('Transfer asset size changed');
    const bytes = await fs.readFile(resolved);
    if (createHash('sha256').update(bytes).digest('hex') !== record.sha256) throw new Error('Transfer asset checksum mismatch');
    return new Uint8Array(bytes);
  }
}

module.exports = { AssetRegistry, MAX_ASSET_BYTES };
