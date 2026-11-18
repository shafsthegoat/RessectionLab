'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const { createHash } = require('node:crypto');

// Installed app code only. Development verifiers, fixtures and benchmarks stay
// in the repository and in the complete packaging source snapshot.
const runtimeFiles = Object.freeze([
  'assets.cjs', 'axis-inspection.cjs', 'case-library-selection.cjs', 'case-library.cjs',
  'contact-family-authority.cjs', 'contact-family.cjs', 'development-episode.cjs',
  'diagnostic-contract.cjs', 'diagnostic-import.cjs', 'display-series-import.cjs',
  'episode-authority.cjs', 'episode-comparison.cjs', 'episode-vascular.cjs',
  'external-case-catalog.cjs', 'frozen-cusa.cjs', 'logging.cjs', 'main.cjs',
  'observed-landmark-contract.cjs', 'patient-catalog.cjs', 'persistent-passive.cjs',
  'prefix-assets.cjs', 'prefix-engine.cjs', 'prefix-pipeline.cjs', 'prefix-resources.cjs',
  'preload.cjs', 'public-contact-authority.cjs', 'public-contact.cjs',
  'rehearsal-unsaved.cjs', 'secondary-display.cjs', 'security.cjs', 'sidecar.cjs',
  'structural-import.cjs', 'study-engine-recording.cjs', 'surgical-timeout.cjs', 'workspace-session.cjs',
]);
const digest = bytes => createHash('sha256').update(bytes).digest('hex');

async function stageElectronRuntime(source, destination, sourceHashes) {
  const rows = {}, dependencies = {};
  for (const name of runtimeFiles) {
    const filename = path.join(source, name);
    if (!(await fs.lstat(filename)).isFile()) throw Error('Runtime member is not a regular file: ' + name);
    const bytes = await fs.readFile(filename), sha256 = digest(bytes);
    if (sha256 !== sourceHashes['electron/' + name]) throw Error('Captured runtime member changed: ' + name);
    rows[name] = { bytes: bytes.length, sha256 };
    dependencies[name] = [];
    // This deliberately fails closed on computed requires. New loading patterns
    // need an explicit packaging review, including any path-based resources.
    for (const match of bytes.toString('utf8').matchAll(/\brequire\s*\(([^)]*)\)/g)) {
      const literal = match[1].trim().match(/^(['"])([^'"]+)\1$/);
      if (!literal) throw Error('Computed runtime require needs review: ' + name);
      const specifier = literal[2];
      if (specifier.startsWith('.')) {
        const target = path.posix.normalize(path.posix.join(path.posix.dirname(name), specifier));
        if (!runtimeFiles.includes(target)) throw Error('Runtime dependency is not allowlisted: ' + target);
        dependencies[name].push(target);
      } else if (specifier !== 'electron' && !specifier.startsWith('node:')) {
        throw Error('External runtime dependency needs review: ' + specifier);
      }
    }
    if (/\bimport\s*\(/.test(bytes.toString('utf8'))) throw Error('Dynamic runtime import needs review: ' + name);
  }
  const reachable = new Set();
  function visit(name) { if (!reachable.has(name)) { reachable.add(name); dependencies[name].forEach(visit); } }
  ['main.cjs', 'preload.cjs', 'sidecar.cjs'].forEach(visit);
  if (reachable.size !== runtimeFiles.length) throw Error('Unused runtime allowlist member needs review');
  await fs.mkdir(destination); // Never merge with an older or contaminated stage.
  for (const name of runtimeFiles) {
    await fs.copyFile(path.join(source, name), path.join(destination, name));
    if (digest(await fs.readFile(path.join(destination, name))) !== rows[name].sha256)
      throw Error('Staged runtime member changed: ' + name);
  }
  const actual = (await fs.readdir(destination)).sort();
  if (JSON.stringify(actual) !== JSON.stringify([...runtimeFiles].sort())) throw Error('Unexpected staged runtime files');
  return { schema: 'resectionlab.electron-runtime.v1', files: rows,
    fileCount: runtimeFiles.length, contentBytes: Object.values(rows).reduce((sum, row) => sum + row.bytes, 0),
    resources: ['../dist/', 'preload.cjs', 'Contents/Resources/research-engine/'],
    excludedSourcePaths: Object.keys(sourceHashes).filter(name => name.startsWith('electron/') && !rows[name.slice(9)]).sort() };
}

module.exports = { runtimeFiles, stageElectronRuntime };
