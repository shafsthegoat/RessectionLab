'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { createRequire } = require('node:module');

const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const isNotice = name => /^(licen[cs]e|copying|copyright|notice|authors)([._-]|$)/i.test(name);
const safeName = name => name.replaceAll('@', '').replaceAll('/', '__');

function safeRelative(name) {
  if (!name || path.isAbsolute(name) || name.split(/[\\/]/).some(x => x === '..' || !x)) {
    throw new Error(`Unsafe notice inventory path: ${name}`);
  }
  return name;
}

async function copySource(source, output, relative, origin, files) {
  safeRelative(relative);
  const bytes = await fs.readFile(source);
  const target = path.join(output, relative);
  await fs.mkdir(path.dirname(target), { recursive: true });
  await fs.writeFile(target, bytes, { flag: 'wx' });
  if (sha(await fs.readFile(source)) !== sha(bytes)) throw new Error(`Notice changed during capture: ${origin}`);
  files[relative] = { source: origin, sha256: sha(bytes), bytes: bytes.length };
}

async function resolvePackage(name, parentRoot) {
  // package.json exports are not universal. Resolve the runtime entry, then
  // find its owning package instead of reading unrelated node_modules entries.
  const request = createRequire(path.join(parentRoot, 'package.json'));
  let entry;
  try { entry = request.resolve(`${name}/package.json`); }
  catch { entry = request.resolve(name); }
  let current = path.dirname(await fs.realpath(entry));
  for (;;) {
    try {
      const manifest = JSON.parse(await fs.readFile(path.join(current, 'package.json'), 'utf8'));
      if (manifest.name === name) return { root: current, manifest };
    } catch (error) { if (error.code !== 'ENOENT') throw error; }
    const parent = path.dirname(current);
    if (parent === current) throw new Error(`Cannot locate upstream package: ${name}`);
    current = parent;
  }
}

async function captureRendererNotices(root, output) {
  const project = JSON.parse(await fs.readFile(path.join(root, 'package.json'), 'utf8'));
  const lock = await fs.readFile(path.join(root, 'pnpm-lock.yaml'), 'utf8');
  const pending = Object.keys(project.dependencies || {}).sort().map(name => ({ name, from: root, via: 'direct dependency' }));
  const seen = new Map();
  const files = {}, unresolved = [], omittedOptional = [], omittedTypePeers = [];
  while (pending.length) {
    const item = pending.shift();
    let upstream;
    try { upstream = await resolvePackage(item.name, item.from); }
    catch (error) {
      if (item.optional && error.code === 'MODULE_NOT_FOUND') { omittedOptional.push(item.name); continue; }
      throw error;
    }
    const { root: packageRoot, manifest } = upstream;
    const id = `${manifest.name}@${manifest.version}`;
    if (seen.has(id)) {
      if (seen.get(id).package_json_sha256 !== sha(await fs.readFile(path.join(packageRoot, 'package.json')))) {
        throw new Error(`Same package identity has different metadata: ${id}`);
      }
      continue;
    }
    if (!lock.split('\n').some(line => line.trim() === `${id}:` || line.trim() === `'${id}':`)) {
      throw new Error(`Installed runtime package is absent from captured pnpm lock: ${id}`);
    }
    const prefix = `renderer/${safeName(manifest.name)}-${manifest.version}`;
    const metadataPath = `${prefix}/package.json`;
    await copySource(path.join(packageRoot, 'package.json'), output, metadataPath, `${id}/package.json`, files);
    const entry = { name: manifest.name, version: manifest.version, declared_license: manifest.license ?? null,
      package_json_sha256: files[metadataPath].sha256, reached_via: item.via, license_files: [] };
    seen.set(id, entry);
    // Root notices cover the package distribution. Do not import unrelated
    // example-directory licenses (Three's optional examples have other terms).
    for (const filename of (await fs.readdir(packageRoot)).sort()) {
      if (!isNotice(filename)) continue;
      const source = path.join(packageRoot, filename);
      if (!(await fs.stat(source)).isFile()) continue;
      const relative = `${prefix}/${filename}`;
      await copySource(source, output, relative, `${id}/${filename}`, files);
      entry.license_files.push(relative);
    }
    if (!entry.license_files.length) {
      unresolved.push({ component: id, reason: 'Installed runtime package declares a license but contains no upstream root license/notice text. Its package.json is retained; no substitute license is invented.' });
    }
    for (const [kind, deps] of Object.entries({ dependency: manifest.dependencies, optional: manifest.optionalDependencies, peer: manifest.peerDependencies })) {
      for (const name of Object.keys(deps || {}).sort()) {
        if (kind === 'peer' && name.startsWith('@types/')) { omittedTypePeers.push(`${id}: ${name}`); continue; }
        pending.push({ name, from: packageRoot, via: `${id} ${kind}`, optional: kind === 'optional' || (kind === 'peer' && manifest.peerDependenciesMeta?.[name]?.optional) });
      }
    }
  }
  const result = { schema_version: 1,
    collector_source_sha256: sha(await fs.readFile(__filename)),
    scope: 'Conservative installed production dependency and runtime peer closure, not a claim that every file in each package was emitted by the renderer bundler. Dev-only tools and type peers are excluded.',
    lock_sha256: sha(Buffer.from(lock)), packages: [...seen.values()].sort((a, b) => a.name.localeCompare(b.name)),
    files, unresolved, omitted_optional_packages: omittedOptional.sort(), omitted_type_peers: omittedTypePeers.sort() };
  await fs.writeFile(path.join(output, 'renderer-inventory.json'), JSON.stringify(result, null, 2) + '\n', { flag: 'wx' });
  return result;
}

async function verifyFiles(root, files) {
  const capturedRoot = await fs.realpath(root);
  for (const [relative, expected] of Object.entries(files)) {
    safeRelative(relative);
    const filename = path.join(root, relative);
    const actual = await fs.realpath(filename);
    if (!actual.startsWith(capturedRoot + path.sep)) throw new Error('Notice path escapes its capture');
    if (expected.symlink !== undefined && await fs.readlink(filename) !== expected.symlink) throw new Error(`Collected payload link changed: ${relative}`);
    const bytes = await fs.readFile(actual);
    if (sha(bytes) !== expected.sha256 || bytes.length !== expected.bytes) throw new Error(`Notice hash mismatch: ${relative}`);
  }
}

async function capturePythonNotices(source, output, engineExecutable) {
  const manifestBytes = await fs.readFile(path.join(source, 'inventory.json'));
  const manifest = JSON.parse(manifestBytes);
  if (manifest.schema_version !== 1 || !manifest.files || !manifest.collected_payload || manifest.engine_sha256 !== sha(await fs.readFile(engineExecutable))) {
    throw new Error('Python notice inventory does not belong to this exact numerical engine');
  }
  await verifyFiles(source, manifest.files);
  await verifyFiles(path.dirname(engineExecutable), manifest.collected_payload);
  const destination = path.join(output, 'numerical-engine');
  await fs.mkdir(destination);
  for (const relative of Object.keys(manifest.files)) {
    safeRelative(relative);
    await fs.mkdir(path.dirname(path.join(destination, relative)), { recursive: true });
    await fs.copyFile(path.join(source, relative), path.join(destination, relative));
  }
  await fs.writeFile(path.join(destination, 'inventory.json'), manifestBytes);
  await verifyFiles(destination, manifest.files);
  return manifest;
}

async function captureElectronNotices(electronRoot, version, output) {
  const files = {};
  for (const filename of ['LICENSE', 'LICENSES.chromium.html']) {
    await copySource(path.join(electronRoot, 'dist', filename), output, `electron/${filename}`, `electron-v${version}-darwin-arm64/${filename}`, files);
  }
  return { version, files };
}

module.exports = { captureRendererNotices, capturePythonNotices, captureElectronNotices, verifyFiles, safeRelative };
