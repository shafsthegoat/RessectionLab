'use strict';

// node scripts/benchmark_acquired_replay.cjs MODE BASELINE_ROOT OUTPUT PAIRS TRIALS [--timing] [--resume]
// Use --prepare alone to verify and stage the existing immutable fixtures.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');
const {spawn, execFileSync} = require('node:child_process');

const root = path.resolve(__dirname, '..');
const fixtures = path.join(root, 'testcases', 'outputs');
const media = '/Volumes/shafrir';
const storage = path.join(media, 'old coding projects/RessectionLabData');
const evidenceRoot = path.join(storage, 'qc/rehearsal-performance-20261121');
const sha = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const json = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const write = (filename, value) => fs.writeFileSync(filename, JSON.stringify(value, null, 2) + '\n');

function verifyDrive() {
  const info = execFileSync('/usr/sbin/diskutil', ['info', '-plist', media]);
  const field = key => execFileSync('/usr/bin/plutil', ['-extract', key, 'raw', '-o', '-', '-'], {input: info, encoding: 'utf8'}).trim();
  const uuid = '9CC3623C-E38B-3EEE-B6BB-B9A06C193700';
  assert.equal(field('VolumeUUID'), uuid);
  assert.equal(field('MountPoint'), media);
  assert.equal(fs.realpathSync(storage), storage);
  const marker = json(path.join(storage, '.ressectionlab-data-root.json'));
  assert.equal(marker.mount_point, media);
  assert.equal(marker.volume_uuid, uuid);
  assert.equal(marker.root_id, '756c4d3b-aabf-4837-a354-52aa500af3f9');
}

function prepare() {
  const sourceRoot = path.join(storage, 'qc/interactive-operative-practice-20261121');
  const index = path.join(sourceRoot, 'acceptance-sources.json');
  const {sources} = json(index);
  const data = {
    ucsfStudy: sources['UCSF-0004 matched visual comparison study'],
    ucsfEpisode: sources['UCSF-0004 matched visual comparison recording'],
    biteStudy: sources['BITE-09 planning'],
    pairedCase: {path: path.join(sourceRoot, 'ready/bipolar-thermal-source.ressectionlab'),
      sha256: sources['Generated tissue and wound control'].sha256},
    pairedActions: {path: path.join(sourceRoot, 'mechanics/paired-actions.json'),
      sha256: '64ed965204b26c53d636a3118c5d548ecbfb987d02d9006fc7965dca594342c4'}
  };
  const inputDir = path.join(fixtures, 'rehearsal-performance-acquired-inputs');
  fs.mkdirSync(inputDir, {recursive: true});
  const prepared = {sourceIndex: index, sourceIndexSha256: sha(index)};
  for (const [key, item] of Object.entries(data)) {
    const source = fs.realpathSync(item.path);
    assert(source.startsWith(storage + path.sep));
    const filename = path.join(inputDir, path.basename(source));
    if (!fs.existsSync(filename)) fs.copyFileSync(source, filename, fs.constants.COPYFILE_EXCL);
    assert.equal(sha(filename), item.sha256, key + ' source digest mismatch');
    prepared[key] = {filename, source, sha256: item.sha256};
  }
  const inputs = path.join(inputDir, 'inputs.json');
  write(inputs, prepared);
  return inputs;
}

function machineLoad() {
  const groups = {};
  const processes = execFileSync('/bin/ps', ['-axo', '%cpu,rss,comm'], {encoding: 'utf8'}).split('\n');
  for (const row of processes) {
    const match = row.match(/^\s*([\d.]+)\s+(\d+)\s+(.+)$/);
    if (!match) continue;
    const [, cpu, rss, command] = match;
    const kind = /Electron|RessectionLab/.test(command) ? 'desktop' :
      /python|ressectionlab-engine/i.test(command) ? 'pythonOrEngine' :
        /node|clang|vite|pytest/.test(command) ? 'buildOrTest' : 'other';
    const group = groups[kind] ??= {processes: 0, cpuPercent: 0, workingSetKiB: 0};
    group.processes++;
    group.cpuPercent = Math.round((group.cpuPercent + Number(cpu)) * 10) / 10;
    group.workingSetKiB += Number(rss);
  }
  return {utc: new Date().toISOString(), uptime: execFileSync('/usr/bin/uptime', {encoding: 'utf8'}).trim(), groups};
}

function distribution(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  const percentile = fraction => sorted[Math.ceil(sorted.length * fraction) - 1] ?? null;
  return {samples: sorted.length, p50: percentile(.5), p95: percentile(.95), max: sorted.at(-1) ?? null};
}

function measuredStages(report) {
  const byId = new Map(report.samples.map(sample => [sample.requestId, sample]));
  const aggregate = new Map();
  const matched = {electron: new Set(), python: new Set()};
  for (const stage of report.stageTiming) {
    if (stage.command !== 'seek' || stage.terminalEvent !== 'result' || !byId.has(stage.requestId)) continue;
    const sample = byId.get(stage.requestId);
    assert.equal(stage.responseStateId, sample.witness.stateId);
    assert.equal(stage.branchId, sample.witness.branchId);
    assert.equal(stage.position, sample.position);
    matched[stage.process].add(stage.requestId);
    for (const [name, ns] of Object.entries(stage.durationsNs)) {
      const key = stage.process + '.' + name;
      if (!aggregate.has(key)) aggregate.set(key, []);
      aggregate.get(key).push(ns / 1e6);
    }
  }
  assert.equal(matched.electron.size, report.samples.length, 'Match every measured seek to Electron transport');
  assert.equal(matched.python.size, report.samples.length, 'Match every measured seek to Python service');
  return Object.fromEntries(Array.from(aggregate, ([name, values]) => [name, distribution(values)]));
}

function proof(report) {
  // Each cold import gets a fresh workspace handle. The verified scan digest,
  // selected series, and source snapshot bind the same archived source.
  const {collectionId, ...display} = report.display;
  return {
    sourceFiles: report.sourceFiles, study: report.study, viewport: report.viewport, display,
    field: report.field, replay: report.replay, practiceAdmission: report.practiceAdmission,
    contextMeshes: report.contextMeshes, visibleMeshes: report.visibleMeshes,
    samples: report.samples.map(sample => report.mode === 'ucsf'
      ? {position: sample.position, witness: sample.witness, display: sample.display,
        source: sample.fence.sourceCaseHash, triangles: sample.frame.drawStatistics.triangles}
      : {position: sample.position, removed: sample.removed, surface: sample.surface, pattern: sample.pattern,
        canvasPixels: sample.canvasPixels, camera: sample.camera, triangles: sample.triangles, targetMesh: sample.targetMesh})
  };
}

async function run({appRoot, inputs, output, mode, trials, timing}) {
  const electron = path.join(root, 'desktop/node_modules/.bin/electron');
  const harness = path.join(root, 'desktop/electron/benchmark-acquired-replay.cjs');
  const log = fs.createWriteStream(output + '.log', {flags: 'wx'});
  const child = spawn(electron, [harness, appRoot, inputs, output, mode, String(trials)], {
    cwd: root, stdio: ['ignore', 'pipe', 'pipe'],
    env: {...process.env, RESSECTIONLAB_REHEARSAL_TIMING: timing ? '1' : '0'}
  });
  child.stdout.on('data', data => { log.write(data); process.stdout.write(data); });
  child.stderr.on('data', data => { log.write(data); process.stderr.write(data); });
  try {
    const status = await new Promise((resolve, reject) => {
      child.once('error', reject);
      child.once('close', resolve);
    });
    assert.equal(status, 0, 'Hidden Electron failed; see ' + output + '.log');
  } finally {
    await new Promise(resolve => log.end(resolve));
  }
  const report = json(path.join(output, 'report.json'));
  assert(report.passed, report.error);
  return report;
}

async function main() {
  verifyDrive();
  const [mode, baselineArg, outputArg, pairArg, trialArg, ...options] = process.argv.slice(2);
  const inputs = prepare();
  if (mode === '--prepare') {
    console.log(inputs);
    return;
  }
  assert(['ucsf', 'bite'].includes(mode));
  assert(options.every(option => ['--timing', '--resume'].includes(option)));
  const timing = options.includes('--timing'), resume = options.includes('--resume');
  const pairs = Number(pairArg), trials = Number(trialArg);
  assert(Number.isInteger(pairs) && pairs >= 1 && pairs <= 10);
  assert(Number.isInteger(trials) && trials >= 1 && trials <= 10);
  const output = path.resolve(outputArg);
  assert(output.startsWith(fixtures + path.sep), 'Use an output under testcases/outputs');
  const parent = fs.realpathSync(path.dirname(output));
  assert(parent.startsWith(fixtures) || parent.startsWith(evidenceRoot + path.sep) || parent === evidenceRoot);
  const baseline = fs.realpathSync(baselineArg);
  assert.notEqual(baseline, root);
  const summaryPath = path.join(output, 'summary.json');
  let prior = null;
  if (resume) {
    prior = json(summaryPath);
    assert(!prior.passed && prior.mode === mode && prior.timing === timing && prior.baseline === baseline && prior.candidate === root);
    assert.deepEqual(prior.inputs, json(inputs));
    fs.copyFileSync(summaryPath, path.join(output, 'summary-before-resume.json'), fs.constants.COPYFILE_EXCL);
  } else fs.mkdirSync(output);
  const summary = {schema: 'resectionlab.matched-acquired-desktop-performance.v1',
    mode, timing, inputs: json(inputs), baseline, candidate: root, builds: {}, runs: [], passed: false};
  const metrics = {baseline: [], candidate: []};
  let expected;
  try {
    for (let pair = 0; pair < pairs; pair++) {
      for (const arm of (pair % 2 === 0 ? ['baseline', 'candidate'] : ['candidate', 'baseline'])) {
        verifyDrive();
        const attempt = path.join(output, String(pair + 1).padStart(2, '0') + '-' + arm);
        const previous = prior?.runs.find(row => row.pair === pair + 1 && row.arm === arm);
        const row = previous ?? {pair: pair + 1, arm, attempt, loadBefore: machineLoad()};
        summary.runs.push(row);
        const appRoot = arm === 'baseline' ? baseline : root;
        let report;
        if (previous) {
          assert.equal(previous.attempt, attempt);
          report = json(path.join(attempt, 'report.json'));
          assert(report.passed && report.appRoot === appRoot && report.mode === mode && report.trials === trials);
          console.log(mode, 'retained', pair + 1, '/', pairs, arm);
        } else {
          console.log(mode, 'pair', pair + 1, '/', pairs, arm);
          report = await run({appRoot, inputs, output: attempt, mode, trials, timing});
          row.loadAfter = machineLoad();
        }
        row.timings = Object.fromEntries(['inputToGpuMs', 'ipcMs', 'responseToGpuMs', 'queueAndDispatchMs', 'inputToCpuSubmissionMs']
          .map(name => [name, distribution(report.samples.map(sample => sample[name]))]));
        row.nativeReadyMs = report.nativeReadyMs;
        row.bindingReadyMs = report.bindingReadyMs;
        row.nativeBinarySha256 = report.backendAfter.numericalBackend.binarySha256;
        if (timing && mode === 'ucsf') row.stageMs = measuredStages(report);
        const build = {files: report.buildFiles, renderer: report.rendererAssets, nativeBinarySha256: row.nativeBinarySha256};
        if (summary.builds[arm]) assert.deepEqual(build, summary.builds[arm], arm + ' executable inputs changed between runs');
        else summary.builds[arm] = build;
        const current = proof(report);
        if (expected) assert.deepEqual(current, expected, 'Acquired case, visible meshes or accepted states differ');
        else expected = current;
        metrics[arm].push(...report.samples);
        write(summaryPath, summary);
      }
    }
    summary.counts = Object.fromEntries(Object.entries(metrics).map(([arm, samples]) => [arm, {
      total: samples.length,
      inputToGpuMs: distribution(samples.map(sample => sample.inputToGpuMs)),
      triangleCount: distribution(samples.map(sample => sample.triangles ?? sample.frame.drawStatistics.triangles)),
      ipcMs: distribution(samples.map(sample => sample.ipcMs)),
      responseToGpuMs: distribution(samples.map(sample => sample.responseToGpuMs)),
      queueAndDispatchMs: distribution(samples.map(sample => sample.queueAndDispatchMs)),
      inputToCpuSubmissionMs: distribution(samples.map(sample => sample.inputToCpuSubmissionMs)),
      requestToMeshWorkerResultMs: distribution(samples.flatMap(sample => sample.work ?? []).map(work => work.requestToResultMs)),
      cpuSubmissionMs: distribution(samples.map(sample => sample.frame?.cpuMs ?? sample.cpuSubmissionMs)),
      gpuFencePollMs: distribution(samples.map(sample => sample.fence?.completionPollMs ?? sample.submissionToGpuMs))
    }]));
    summary.passed = true;
  } catch (error) {
    summary.error = String(error.stack ?? error);
    console.error(error);
  }
  write(summaryPath, summary);
  process.exitCode = summary.passed ? 0 : 1;
}

main().catch(error => { console.error(error); process.exitCode = 1; });
