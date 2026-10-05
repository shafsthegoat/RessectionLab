'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { assertLockedElectronVersion } = require('./electron-version.cjs');

const lock = `lockfileVersion: '9.0'

settings:
  autoInstallPeers: true

importers:
  .:
    dependencies:
      react:
        specifier: ^19.3.0
        version: 19.3.0
    devDependencies:
      electron:
        specifier: ^44.5.0
        version: 44.5.1

packages:
  electron@44.5.1:
    resolution: {integrity: sha512-fixture}

snapshots:
  electron@44.5.1: {}
`;

test('Electron runtime matches the resolved importer version, not its semver specifier', () => {
  assert.equal(assertLockedElectronVersion('44.5.1', lock), '44.5.1');
  assert.equal(assertLockedElectronVersion('44.5.1', lock.replace(/\n/g, '\r\n')), '44.5.1');
  assert.equal(assertLockedElectronVersion('44.5.1', lock.replace('      electron:', "      'electron':")), '44.5.1');
  assert.throws(() => assertLockedElectronVersion('44.5.0', lock), /differs from locked/);
  assert.throws(() => assertLockedElectronVersion('44.6.0', lock), /differs from locked/);
});

test('missing Electron or its resolved version fails without using packages, snapshots or specifier', () => {
  for (const changed of [
    lock.replace('      electron:', '      other:'),
    lock.replace('        version: 44.5.1\n', ''),
    lock.replace('        specifier: ^44.5.0\n', ''),
    lock.replace(/importers:[\s\S]*?(?=packages:)/, ''),
  ]) assert.throws(() => assertLockedElectronVersion('44.5.1', changed), /Cannot bind Electron/);
});

test('duplicate or ambiguous importer resolutions are rejected', () => {
  const electron = '      electron:\n        specifier: ^44.5.0\n        version: 44.5.1\n';
  for (const changed of [
    lock.replace(electron, electron + electron),
    lock.replace(electron, electron + electron.replace('electron:', "'electron':")),
    lock.replace('        version: 44.5.1\n', '        version: 44.5.1\n        version: 44.5.1\n'),
    lock.replace(electron, electron + `    optionalDependencies:\n${electron}`),
    lock.replace('packages:', '  other-project:\n    devDependencies:\n' + electron + '\npackages:'),
    lock.replace('packages:', '  .:\n    devDependencies:\n' + electron + '\npackages:'),
    lock.replace('packages:', 'importers:\n  .:\n    devDependencies:\n' + electron + '\npackages:'),
    lock.replace(electron, `    devDependencies:\n${electron}`),
  ]) assert.throws(() => assertLockedElectronVersion('44.5.1', changed), /Cannot bind Electron/);
});

test('unsupported YAML aliases, merges, inline maps and schemas fail closed', () => {
  for (const changed of [
    lock.replace("lockfileVersion: '9.0'", "lockfileVersion: '6.0'"),
    lock.replace('  .:', '  .: &root'),
    lock.replace('      electron:', '      electron: *other'),
    lock.replace('    devDependencies:', '    devDependencies: {electron: {version: 44.5.1}}'),
    lock.replace('        version: 44.5.1', '        <<: *resolution'),
    lock.replace('      electron:', '\telectron:'),
    lock.replace('importers:', "'importers':"),
    lock.replace('    devDependencies:', '    devDependencies:\n    <<: *other'),
  ]) assert.throws(() => assertLockedElectronVersion('44.5.1', changed), /Cannot bind Electron/);
});

test('local paths, semver ranges, peer suffixes and malformed installed versions are rejected', () => {
  for (const version of ['^44.5.1', 'file:../electron', 'link:../electron', '44.5.1(peer@1.0.0)', '044.5.1', '44.5.1 # ignored']) {
    assert.throws(() => assertLockedElectronVersion('44.5.1', lock.replace('        version: 44.5.1', `        version: ${version}`)), /Cannot bind Electron/);
  }
  for (const installed of [undefined, null, 44, '', '^44.5.1', 'v44.5.1']) {
    assert.throws(() => assertLockedElectronVersion(installed, lock), /invalid installed version/);
  }
});
