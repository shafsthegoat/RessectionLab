'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const { spawnSync } = require('node:child_process');
const { packager } = require('@electron/packager');
const { publishBundle } = require('./publish.cjs');
const { assertLockedElectronVersion } = require('./electron-version.cjs');
const { captureRendererNotices, capturePythonNotices, captureElectronNotices } = require('./dependency-notices.cjs');

async function hashes(root, relative = '') {
  const result = {};
  const entries = await fs.readdir(path.join(root, relative), { withFileTypes: true });
  entries.sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0);
  for (const entry of entries) {
    const name = path.join(relative, entry.name);
    if (entry.isSymbolicLink()) result[name] = 'symlink:' + await fs.readlink(path.join(root, name));
    else if (entry.isDirectory()) Object.assign(result, await hashes(root, name));
    else if (entry.isFile()) result[name] = crypto.createHash('sha256').update(await fs.readFile(path.join(root, name))).digest('hex');
  }
  return result;
}
async function verifyLinks(root, current = root) {
  let count = 0;
  for (const entry of await fs.readdir(current, {withFileTypes: true})) {
    const filename = path.join(current, entry.name);
    if (entry.isSymbolicLink()) {
      const target = await fs.realpath(filename);
      if (!target.startsWith(root + path.sep)) throw new Error(`Bundle link escapes app: ${filename}`);
      count += 1;
    } else if (entry.isDirectory()) count += await verifyLinks(root, filename);
  }
  return count;
}
function run(command, args, cwd) {
  const result = spawnSync(command, args, { cwd, encoding: 'utf8', stdio: 'pipe', maxBuffer: 16 * 1024 * 1024 });
  if (result.status !== 0) throw new Error(`${command} failed: ${result.stdout}\n${result.stderr}`);
  return result.stdout;
}

async function main() {
  const root = path.resolve(__dirname, '..');
  const repo = path.dirname(root);
  const release = path.join(root, 'release');
  await fs.mkdir(release, { recursive: true });
  const reportPath = path.join(release, 'build-report.json');
  await fs.rm(reportPath, { force: true });
  const input = await fs.mkdtemp(path.join(release, 'inputs-'));
  const started = Date.now();
  try {
    const copied = ['src', 'electron', 'index.html', 'vite.config.ts', 'tsconfig.json', 'package.json', 'pnpm-lock.yaml'];
    for (const item of copied) await fs.cp(path.join(root, item), path.join(input, item), { recursive: true });
    const sourceHashes = await hashes(input);
    for (const [relative, expected] of Object.entries(sourceHashes)) {
      const actual = crypto.createHash('sha256').update(await fs.readFile(path.join(root, relative))).digest('hex');
      if (actual !== expected) throw new Error('Desktop source changed during snapshot capture; retry');
      await fs.chmod(path.join(input, relative), 0o444);
    }
    const iconInput = path.join(input, 'RessectionLab.icns');
    const iconSource = path.join(repo, 'packaging/RessectionLab.icns');
    await fs.copyFile(iconSource, iconInput);
    const iconSha256 = crypto.createHash('sha256').update(await fs.readFile(iconInput)).digest('hex');
    if (iconSha256 !== crypto.createHash('sha256').update(await fs.readFile(iconSource)).digest('hex')) throw new Error('App icon changed during capture');
    await fs.chmod(iconInput, 0o444);
    await fs.symlink(path.join(root, 'node_modules'), path.join(input, 'node_modules'), 'dir');
    // Both checking and bundling run against the same captured renderer sources.
    run(process.execPath, [path.join(root, 'node_modules/typescript/bin/tsc'), '--noEmit'], input);
    run(process.execPath, [path.join(root, 'node_modules/vite/bin/vite.js'), 'build'], input);
    for (const [relative, expected] of Object.entries(sourceHashes)) {
      const actual = crypto.createHash('sha256').update(await fs.readFile(path.join(input, relative))).digest('hex');
      if (actual !== expected) throw new Error('Captured desktop input changed during build');
    }
    const sourcePackage = JSON.parse(await fs.readFile(path.join(input, 'package.json'), 'utf8'));
    const electronVersion = assertLockedElectronVersion(require('electron/package.json').version, await fs.readFile(path.join(input, 'pnpm-lock.yaml'), 'utf8'));
    const electronChecksums = require('electron/checksums.json');
    const electronArchiveSha256 = electronChecksums[`electron-v${electronVersion}-darwin-arm64.zip`];
    if (!/^[a-f0-9]{64}$/.test(electronArchiveSha256 || '')) throw new Error('Installed Electron package lacks the arm64 archive checksum');
    const stage = path.join(input, 'runtime');
    await fs.mkdir(stage);
    for (const item of ['dist', 'electron']) await fs.cp(path.join(input, item), path.join(stage, item), { recursive: true });
    await fs.writeFile(path.join(stage, 'package.json'), JSON.stringify({ name: sourcePackage.name, version: sourcePackage.version, main: sourcePackage.main }));
    const engineInput = path.join(input, 'research-engine');
    const originalEngine = path.join(root, 'sidecar/ressectionlab-engine');
    const expectedEngineHashes = await hashes(originalEngine);
    await fs.cp(originalEngine, engineInput, { recursive: true, verbatimSymlinks: true });
    const engineHashes = await hashes(engineInput);
    if (JSON.stringify(expectedEngineHashes) !== JSON.stringify(engineHashes) || JSON.stringify(engineHashes) !== JSON.stringify(await hashes(originalEngine))) throw new Error('Numerical engine changed during snapshot capture');
    // Notices live outside app.asar so a copied .app includes readable upstream
    // texts. Python inventory must belong to these exact frozen payload bytes.
    const noticeInput = path.join(input, 'ThirdPartyNotices');
    await fs.mkdir(noticeInput);
    const rendererNotices = await captureRendererNotices(input, noticeInput);
    const pythonNotices = await capturePythonNotices(path.join(root, 'sidecar/notices'), noticeInput, path.join(engineInput, 'ressectionlab-engine'));
    const electronNotices = await captureElectronNotices(path.dirname(require.resolve('electron/package.json')), electronVersion, noticeInput);
    const unresolvedNotices = [...rendererNotices.unresolved, ...pythonNotices.unresolved];
    const noticeIndex = { schema_version: 1, electron: electronNotices, renderer_inventory: 'renderer-inventory.json', numerical_inventory: 'numerical-engine/inventory.json',
      source_completeness: unresolvedNotices.length ? 'incomplete' : 'complete_for_detected_components', unresolved: unresolvedNotices,
      project_license: 'No project license is selected by this inventory. Upstream terms apply to their respective components.' };
    await fs.writeFile(path.join(noticeInput, 'inventory.json'), JSON.stringify(noticeIndex, null, 2) + '\n');
    await fs.writeFile(path.join(noticeInput, 'README.txt'), 'Third-party dependency notices\n\nUpstream texts are copied without alteration. Consult inventory.json for exact component versions, source hashes and unresolved source gaps. Renderer packages form a conservative runtime dependency closure. Numerical entries identify the actual frozen modules and collected payload. These files do not select a license for RessectionLab or establish clinical or distribution approval.\n\nElectron and Chromium: electron/\nRenderer library notices: renderer/ and renderer-inventory.json\nFrozen Python and native wheel notices: numerical-engine/\n');
    const noticeHashes = await hashes(noticeInput);
    const manifest = { sourceHashes, engineHashes, noticeHashes, noticeSourceCompleteness: noticeIndex.source_completeness, appIconSha256: iconSha256, electronVersion, electronArchiveSha256, sourceDigest: crypto.createHash('sha256').update(JSON.stringify(sourceHashes)).digest('hex'),
      revision: run('git', ['rev-parse', 'HEAD'], repo).trim(), gitStatus: run('git', ['status', '--porcelain'], repo).split('\n').filter(Boolean), createdUtc: new Date().toISOString() };
    await fs.writeFile(path.join(stage, 'BUILD_INPUT_MANIFEST.json'), JSON.stringify(manifest));
    const paths = await packager({ dir: stage, name: 'RessectionLab', platform: 'darwin', arch: 'arm64', out: path.join(input, 'package-output'), overwrite: true, asar: true,
      electronVersion, download: { checksums: electronChecksums }, appBundleId: 'org.ressectionlab.electron', appCategoryType: 'public.app-category.medical',
      icon: iconInput, prune: false,
      extendInfo: { NSHumanReadableCopyright: 'RessectionLab research software. Research use only.' },
    });
    let appPath = path.join(paths[0], 'RessectionLab.app');
    // Compare against the notices delivered by the checksum-verified Electron
    // archive used for this package, not merely an installed version label.
    for (const filename of ['LICENSE', 'LICENSES.chromium.html']) {
      const emittedHash = crypto.createHash('sha256').update(await fs.readFile(path.join(paths[0], filename))).digest('hex');
      if (emittedHash !== electronNotices.files[`electron/${filename}`].sha256) throw new Error('Electron archive and captured notice text differ');
    }
    await fs.cp(engineInput, path.join(appPath, 'Contents/Resources/research-engine'), { recursive: true, verbatimSymlinks: true });
    const noticeDestination = path.join(appPath, 'Contents/Resources/ThirdPartyNotices');
    await fs.cp(noticeInput, noticeDestination, { recursive: true });
    if (JSON.stringify(await hashes(noticeDestination)) !== JSON.stringify(noticeHashes)) throw new Error('Bundled dependency notice bytes changed');
    const internalSymlinks = await verifyLinks(appPath);
    run('/usr/bin/codesign', ['--force', '--deep', '--sign', '-', appPath], repo);
    run('/usr/bin/codesign', ['--verify', '--deep', '--strict', appPath], repo);
    const finalDirectory = path.join(release, 'RessectionLab-darwin-arm64');
    const { previousDirectory } = await publishBundle(paths[0], finalDirectory);
    appPath = path.join(finalDirectory, 'RessectionLab.app');
    await fs.writeFile(reportPath, JSON.stringify({ status: 'built_unverified', appPath, architecture: 'arm64', signing: 'local_ad_hoc', notarized: false, internalSymlinks, previousDirectory, sourceDigest: manifest.sourceDigest, noticeSourceCompleteness: noticeIndex.source_completeness, unresolvedNotices, noticeLocation: 'Contents/Resources/ThirdPartyNotices', inputSnapshot: input, elapsedSeconds: (Date.now()-started)/1000 }, null, 2));
    process.stdout.write(appPath + '\n');
  } catch (error) { await fs.writeFile(reportPath, JSON.stringify({ status: 'failed', error: error.message })); throw error; }
}
main().catch(error => { process.stderr.write(error.stack+'\n'); process.exitCode = 1; });
