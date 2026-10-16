'use strict';
// Opt-in, independent JSON/ZIP-manifest forensics. No application/model imports,
// engine execution, array archive reads, or replay of patient actions.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {execFileSync} = require('node:child_process');
const ROOT = path.resolve(__dirname, '../..');
const RUN = 'outputs/first-release-frozen-sidecar-run-v1';
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const rawHash = value => sha(JSON.stringify(value));
const ordered = value => Array.isArray(value) ? value.map(ordered) : value !== null && typeof value === 'object'
  ? Object.fromEntries(Object.keys(value).sort().map(key => [key, ordered(value[key])])) : value;
const canonicalHash = value => rawHash(ordered(value));
const PINS = {
  [`${RUN}/report.json`]: 'a7251122a901c384f6c12228888da05945ba35d20300c9ee39ae57244580713a',
  [`${RUN}/compact-summary.json`]: '2192f6df5273229434d74d3f16adc6ecd3c4e044fdcdc91a25d3d811443a233b',
  [`${RUN}/stopped-rehearsal.json`]: '7b5423cd916bdc4ab1be05148c4a44fd04f4e37606ef002ef164c4ef2b8fa285',
  [`${RUN}/UCSF-frozen-reviewed.ressectionlab`]: '4605b84b79b5d643ceae9108267b77ed33ed6a036354eeb53983038b27285a3d',
  'artifacts/first-release-frozen-sidecar-preparation-v2/frozen-auditor.mjs': 'e177a8f1499609b30db84627f758b0553282a4e251ab8c0b0c3163f389c70109',
  'artifacts/first-release-frozen-sidecar-preparation-v2/preparation.json': '40c53acf6b203614a374c423fdf5b3d0594fdc3125bd64e20a9199cfd7d85547',
  'src/resectionlab/imaging.py': '45df035c47f9c509fef91ebc02f1c3bef40266c052dbd1de6d133815a5af6afa',
  'src/resectionlab/workspace_bundle.py': '9a1bf1d0f749bb67857d65803aeeaf8f07927ebce8a03fb4364c59e8b151bf1e',
  'src/resectionlab/rehearsal_bridge.py': '6adb1f9b6f8d9d13b8991c7a7cad1aa9375aa2049ce3ce35c3cc5d7af791597e',
  'src/resectionlab/rehearsal_episode.py': '4c5b0b8b7d941069b54266904383f1a3113d96830847bf9cd3ba02f28807e6fe',
  'src/resectionlab/core.py': '094bb902552117ee0be1ee4ed79ae0b09a7005e48d24ca4a9a4674ff54af077b',
};
function fileHash(relative) {
  const filename = path.join(ROOT, relative), fd = fs.openSync(filename, 'r');
  const hash = crypto.createHash('sha256'), buffer = Buffer.alloc(1024 * 1024);
  try { for (let count; (count = fs.readSync(fd, buffer)) > 0;) hash.update(buffer.subarray(0, count)); }
  finally { fs.closeSync(fd); }
  return hash.digest('hex');
}
function verifyPins() {
  return Object.fromEntries(Object.entries(PINS).map(([name, expected]) => {
    const actual = fileHash(name); assert.equal(actual, expected, `Immutable input changed: ${name}`);
    return [name, {sha256: actual, bytes: fs.statSync(path.join(ROOT, name)).size}];
  }));
}
function compare(left, right) {
  const result = {nodes: 0, objects: 0, arrays: 0, scalars: 0, numbers: 0,
    differences: [], objectKeyOrderPaths: [], signedZeroRepresentationPaths: []};
  const kind = value => value === null ? 'null' : Array.isArray(value) ? 'array' : typeof value;
  function visit(a, b, location) {
    result.nodes++;
    const type = kind(a);
    if (type !== kind(b)) { result.differences.push({path: location, kind: 'type'}); return; }
    if (type === 'array') {
      result.arrays++;
      if (a.length !== b.length) result.differences.push({path: location, kind: 'array-length'});
      for (let i = 0; i < Math.min(a.length, b.length); i++) visit(a[i], b[i], `${location}[${i}]`);
    } else if (type === 'object') {
      result.objects++;
      const ak = Object.keys(a), bk = Object.keys(b);
      if (JSON.stringify(ak.slice().sort()) !== JSON.stringify(bk.slice().sort()))
        result.differences.push({path: location, kind: 'object-keys'});
      else if (JSON.stringify(ak) !== JSON.stringify(bk)) result.objectKeyOrderPaths.push(location);
      for (const key of ak) if (Object.hasOwn(b, key)) visit(a[key], b[key], `${location}.${key}`);
    } else {
      result.scalars++;
      if (type === 'number') {
        result.numbers++;
        assert.ok(Number.isFinite(a) && Number.isFinite(b), 'Nonfinite JSON scalar');
        if (a === 0 && b === 0 && !Object.is(a, b)) result.signedZeroRepresentationPaths.push(location);
      }
      if (a !== b) result.differences.push({path: location, kind: 'scalar-value'});
    }
  }
  visit(left, right, '$');
  return result;
}
function exactValues(left, right) {
  const result = compare(left, right);
  assert.equal(result.differences.length, 0, 'JSON value/type/ordered-array difference');
  assert.equal(canonicalHash(left), canonicalHash(right), 'Canonical JSON differs');
  return {...result, objectKeyOrderChanges: result.objectKeyOrderPaths.length,
    signedZeroRepresentationChanges: result.signedZeroRepresentationPaths.length,
    canonicalSha256: canonicalHash(left), leftRawJsonSha256: rawHash(left), rightRawJsonSha256: rawHash(right)};
}
function controls() {
  const good = {z: [{b: 2, a: 1}], n: 0, yes: true};
  const reordered = {yes: true, n: 0, z: [{a: 1, b: 2}]};
  assert.equal(exactValues(good, reordered).objectKeyOrderChanges, 2);
  const mutations = [
    {z: [{b: 3, a: 1}], n: 0, yes: true},
    {z: [{b: '2', a: 1}], n: 0, yes: true},
    {z: [{b: 2, a: 1}], n: 0, yes: 1},
    {z: [{b: 2, a: 1}, null], n: 0, yes: true},
    {z: [{b: 2, a: 1}], n: null, yes: true},
    {z: [{b: 2, a: 1}], n: 0},
    {z: [{b: 2, a: 1}], n: 0, yes: true, extra: false},
  ];
  for (const changed of mutations) assert.throws(() => exactValues(good, changed));
  assert.throws(() => exactValues([1, 2], [2, 1]));
  assert.equal(exactValues({n: 0}, {n: -0}).signedZeroRepresentationChanges, 1);
  assert.notEqual(rawHash(good), rawHash(reordered));
  return {passed: 11, scope: 'Generated key-order, value, type, ordered-array, key-set and signed-zero controls'};
}
function readJson(relative) { return JSON.parse(fs.readFileSync(path.join(ROOT, relative), 'utf8')); }
function reorderOnlyActions(episode, stored) {
  exactValues(episode.savedEpisode, stored);
  assert.equal(episode.history.length, stored.actions.length);
  episode.savedEpisode.actions = structuredClone(stored.actions);
  for (const row of episode.history) {
    assert.ok(Number.isInteger(row.actionIndex) && row.actionIndex >= 0 && row.actionIndex < stored.actions.length);
    exactValues(row.action, stored.actions[row.actionIndex]);
    row.action = structuredClone(stored.actions[row.actionIndex]);
  }
}
function main() {
  const generatedControls = controls(), before = verifyPins();
  const report = readJson(`${RUN}/report.json`), summary = readJson(`${RUN}/compact-summary.json`);
  const stopped = readJson(`${RUN}/stopped-rehearsal.json`);
  assert.equal(report.status, 'failed'); assert.equal(summary.status, 'failed');
  assert.equal(report.failures.length, 1); assert.equal(report.journal.length, 21);
  assert.equal(report.runtime.auditorSha256, PINS['artifacts/first-release-frozen-sidecar-preparation-v2/frozen-auditor.mjs']);
  const bundle = path.join(ROOT, RUN, 'UCSF-frozen-reviewed.ressectionlab');
  const manifestBytes = execFileSync(path.join(ROOT, '.venv/bin/python'), ['-I', '-c',
    'import sys,zipfile\nwith zipfile.ZipFile(sys.argv[1]) as z:\n assert z.namelist().count("manifest.json")==1\n assert z.getinfo("manifest.json").file_size<=1024*1024\n sys.stdout.buffer.write(z.read("manifest.json"))', bundle],
    {cwd: ROOT, maxBuffer: 1024 * 1024, timeout: 10000});
  assert.equal(sha(manifestBytes), 'cca109ec15e7f10e0d2cf497a04b408d7d5b2d8e7c927ffec716e8edcc1a50df');
  const manifest = JSON.parse(manifestBytes), stored = manifest.artifacts.patient_rehearsal;
  const liveSaved = exactValues(stopped.episode.savedEpisode, stored.live);
  const reconstructed = structuredClone(stopped);
  reorderOnlyActions(reconstructed.episode, stored.live);
  const liveExport = exactValues(stopped.episode, reconstructed.episode);
  assert.equal(liveExport.nodes, 52426); assert.equal(liveExport.objectKeyOrderChanges, 6);
  assert.equal(liveExport.signedZeroRepresentationChanges, 0);
  assert.equal(liveExport.leftRawJsonSha256, '8bbdd7f0bc9d67485adb2013f69146c0b151e1b399f3bc107a160198d3f7b90c');
  assert.equal(liveExport.rightRawJsonSha256, '4e644bb998e66cfe7d6e4b4567baddb6c2cf14bed3db403dfcfc5fb7e186bf49');
  assert.ok(report.failures[0].message.includes(`+ '${liveExport.rightRawJsonSha256}'`));
  assert.ok(report.failures[0].message.includes(`- '${liveExport.leftRawJsonSha256}'`));
  assert.equal(reconstructed.comparison.alternatives.length, 2);
  assert.equal(stored.comparison.alternatives.length, 2);
  const branches = reconstructed.comparison.alternatives.map((row, index) => {
    const original = stopped.comparison.alternatives[index], saved = stored.comparison.alternatives[index];
    assert.equal(row.id, saved.id); assert.equal(row.label, saved.label);
    const savedRecord = exactValues(original.episode.savedEpisode, saved.record);
    reorderOnlyActions(row.episode, saved.record);
    return {id: row.id, actions: saved.record.actions.length, historyEntries: row.episode.history.length,
      replayFrames: row.episode.replayFrames.length, savedRecord, exportedEpisode: exactValues(original.episode, row.episode)};
  });
  const comparison = exactValues(stopped.comparison, reconstructed.comparison);
  const wholeReplyBeforeSavedView = exactValues(stopped, reconstructed);
  reconstructed.savedView = structuredClone(manifest.artifacts.workspace.rehearsalView);
  const last = report.journal.at(-1);
  assert.equal(last.op, 'inspectRehearsal'); assert.equal(last.status, 'result');
  assert.equal(rawHash(reconstructed), last.resultSha256);
  assert.equal(last.resultSha256, '452641d0f3757b3a09991118d64405d2cd1e84f935ea8acc92d155d16a39c1af');
  assert.equal(report.packageBinding.engineByteDifferences.length, 0);
  assert.equal(report.sessions.length, 2);
  for (const session of report.sessions) { assert.equal(session.cleanup.exitCode, 0); assert.equal(session.cleanup.signalCode, null); }
  const localDetails = path.join(ROOT, RUN, 'independent-order-forensics');
  fs.mkdirSync(localDetails, {recursive: false});
  const freshFile = path.join(localDetails, 'reconstructed-fresh-reply.json');
  fs.writeFileSync(freshFile, JSON.stringify(reconstructed), {flag: 'wx'});
  assert.equal(fileHash(path.relative(ROOT, freshFile)), last.resultSha256);
  const after = verifyPins(); assert.deepEqual(after, before);
  const result = {
    schema: 'independent-frozen-sidecar-json-order-forensics-v1', status: 'cause_confirmed',
    createdUtc: new Date().toISOString(), auditorSha256: sha(fs.readFileSync(__filename)), generatedControls,
    scope: 'Existing JSON, compressed bundle byte hashes and only its manifest.json member; no array decoding, backend execution or clinical acceptance.',
    inputFiles: before, inputFilesUnchangedAfterAudit: true,
    archiveRead: {member: 'manifest.json', bytes: manifestBytes.length, sha256: sha(manifestBytes), otherMembersDecoded: 0},
    originalFailure: {status: report.status, name: report.failures[0].name, assertion: 'Fresh engine episode hash mismatch',
      actualSha256: liveExport.rightRawJsonSha256, expectedSha256: liveExport.leftRawJsonSha256,
      requests: report.journal.length, engineByteDifferences: 0, childExitCodes: report.sessions.map(s => s.cleanup.exitCode),
      remainingCaseChecksReached: false},
    sourceMechanism: [
      {file: 'artifacts/first-release-frozen-sidecar-preparation-v2/frozen-auditor.mjs', detail: 'jsonSha hashes JSON.stringify without canonical object-key order; fresh episode assertion used this encoding fingerprint.'},
      {file: 'src/resectionlab/imaging.py', detail: 'save_case encodes the manifest with sort_keys=True.'},
      {file: 'src/resectionlab/workspace_bundle.py', detail: 'encoded sorts JSON keys; save_workspace writes the manifest through encoded.'},
      {file: 'src/resectionlab/core.py', detail: 'freeze/thaw preserve mapping iteration order; semantic_digest sorts keys.'},
      {file: 'src/resectionlab/rehearsal_episode.py', detail: 'restore executes saved action dictionaries; history and serialize retain their insertion order, while source and final-state checks use semantic identity.'},
      {file: 'src/resectionlab/rehearsal_bridge.py', detail: 'Payload export rebuilds the live and alternative episodes and adds the saved view as a final response field.'},
    ],
    live: {actions: stored.live.actions.length, historyEntries: stopped.episode.history.length,
      replayFrames: stopped.episode.replayFrames.length, savedRecord: liveSaved, exportedEpisode: liveExport},
    branches, comparison, wholeReplyBeforeSavedView,
    freshReply: {journalSha256: last.resultSha256, independentReconstructionSha256: rawHash(reconstructed),
      exactMatch: true, savedViewAddition: {keys: Object.keys(reconstructed.savedView), rawJsonSha256: rawHash(reconstructed.savedView)},
      reconstructionFile: path.relative(ROOT, freshFile)},
    conclusion: 'Object insertion order explains the complete observed fresh reply and the original failed episode fingerprint. The live export has no numeric, JSON-type, scalar-value or ordered-array change. Eight signed-zero spelling differences in each directly archived source-binding record normalize under JSON.stringify and are not causal.',
    limits: ['This is an independent reconstruction from retained records, not a new engine run or independently captured fresh reply.',
      'The exact full-reply journal hash supplies the link to the actual fresh-process response.',
      'The original failed protocol run remains failed; later case checks were not reached.',
      'No anatomy, physics, surgical or clinical acceptance is inferred.'],
  };
  const destination = path.join(__dirname, 'independent-json-order-forensics.json');
  fs.writeFileSync(destination, JSON.stringify(result, null, 2) + '\n', {flag: 'wx'});
  fs.chmodSync(destination, 0o444); fs.chmodSync(freshFile, 0o444); fs.chmodSync(__filename, 0o444);
  console.log(JSON.stringify({status: result.status, report: path.relative(ROOT, destination), reportSha256: fileHash(path.relative(ROOT, destination)),
    auditorSha256: result.auditorSha256, fullFreshReplySha256: last.resultSha256, liveNodes: liveExport.nodes, liveKeyOrderChanges: 6}));
}
if (process.argv.length === 3 && process.argv[2] === '--self-test') console.log(JSON.stringify(controls()));
else if (process.argv.length === 3 && process.argv[2] === '--execute-metadata-forensics') main();
else { console.error('Use --self-test or the explicitly authorized --execute-metadata-forensics'); process.exitCode = 2; }
