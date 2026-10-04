'use strict';

// A development verifier, never invoked by the desktop. It requires explicit
// local inputs and retains full responses; its summary grants no clinical or
// replay authority. Only the inspection/search operations below are exercised.
// Positive input: declared skull-stripped MRI with source targets and no brain
// mask, so the separate estimated-support acknowledgment is actually required.
// Blocked input: an explicitly full-head case without reviewed usable support.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { createHash } = require('node:crypto');

const NATIVE_TOOLS = ['native-fine-aspiration', 'native-wide-aspiration'];
const HASH = /^sha256:[a-f0-9]{64}$/;
const differentHash = value => `${value.slice(0, -1)}${value.endsWith('0') ? '1' : '0'}`;
const count = value => assert.ok(Number.isSafeInteger(value) && value >= 0, 'invalid inventory count');
const unsignedZeros = value => Array.isArray(value) ? value.map(unsignedZeros) : value === 0 ? 0 : value;

function sameWindow(actual, expected, tolerance) {
  for (const key of ['center_mm', 'radius_mm', 'window_id']) assert.deepEqual(unsignedZeros(actual[key]), unsignedZeros(expected[key]));
  assert.equal(actual.normal_inward.length, 3);
  const delta = Math.max(...actual.normal_inward.map((x, i) => Math.abs(x - expected.normal_inward[i])));
  assert.ok(Number.isFinite(delta) && delta <= tolerance, 'access normal changed beyond declared roundoff');
  return delta;
}

function validateInspection(result, request, route, source, toolCatalog) {
  assert.equal(result.schemaVersion, 1);
  for (const key of ['caseHash', 'planningHash', 'routeId', 'routePlanningModelHash']) {
    assert.equal(result[key], request[key], `stale ${key}`);
  }
  assert.equal(result.accessSource, 'selected_route_window_only');
  const toolIds = NATIVE_TOOLS.filter(id => request.toolIds.includes(id));
  assert.deepEqual(result.requestedToolIds, toolIds);
  const anchor = structuredClone(route.window);
  if (source.frame === 'LPS+') {
    for (const key of ['center_mm', 'normal_inward']) anchor[key] = anchor[key].map((x, i) => i < 2 ? -x : x);
  } else assert.equal(source.frame, 'RAS+');
  sameWindow(result.anchorWindowRas, anchor, 0);
  const report = result.inspection;
  assert.equal(report.version, 'native-axis-inspection-v1');
  assert.equal(report.role, 'inspection');
  assert.equal(report.inventory_complete, true);
  assert.equal(report.candidate_eligible, false);
  assert.equal(report.removal_authorized, false);
  assert.equal(report.clinical_deficit_probability, null);
  assert.deepEqual(report.accounting, { gradient_steps: 0, executed_transitions: 0,
    native_commits: 0, simulated_removed_volume_mm3: 0 });
  const binding = report.binding;
  assert.equal(binding.case_hash, request.caseHash);
  assert.equal(binding.planning_hash, request.planningHash);
  assert.equal(binding.case_id, source.caseId);
  assert.equal(binding.geometry_frame, 'RAS+');
  assert.equal(binding.source_frame, source.frame);
  assert.deepEqual(binding.source_shape, source.shape);
  const affine = source.affine.map((row, index) => row.map(value => source.frame === 'LPS+' && index < 2 ? -value : value));
  assert.deepEqual(unsignedZeros(binding.native_affine_ras_mm), unsignedZeros(affine), 'native grid differs from the source grid in RAS');
  const normalization = result.anchorWindowNormalization;
  assert.equal(normalization.normalAbsoluteTolerance, 4 * Number.EPSILON);
  const requested = binding.requested_access_ras;
  assert.equal(sameWindow(requested, anchor, 4 * Number.EPSILON), normalization.normalMaximumDifference);
  const conversionTolerance = source.frame === 'LPS+' ? 4 * Number.EPSILON : 0;
  assert.equal(binding.access_conversion.source_frame, source.frame);
  assert.equal(binding.access_conversion.normal_absolute_tolerance, conversionTolerance);
  assert.equal(sameWindow(binding.access, requested, conversionTolerance), binding.access_conversion.normal_maximum_absolute_difference);
  assert.deepEqual(binding.tools, toolIds.map(id => toolCatalog.find(tool => tool.tool_id === id)));
  assert.equal(binding.neighboring_columns_acknowledged, true);
  assert.equal(binding.source_support.estimated_support_acknowledged, request.acknowledgeEstimatedSupport);
  assert.equal(binding.input_profile, 'RAW');
  assert.equal(binding.world_role, null);
  assert.equal(binding.world_partitions_created, false);
  assert.equal(binding.population_priors_used, false);
  assert.deepEqual(binding.functional_evidence_available, { motor: false, language: false });
  for (const hash of [binding.binding_hash, report.inspection_hash]) assert.match(hash, HASH);
  if (request.expectedBindingHash) assert.equal(binding.binding_hash, request.expectedBindingHash);
  const inventory = report.inventory, batch = inventory.batch;
  assert.equal(inventory.status, 'complete');
  assert.equal(batch.unsupported_reason, null);
  assert.equal(batch.geometry_certified, false);
  assert.equal(batch.removal_authorized, false);
  assert.equal(binding.proposal_rule.offsets_source_voxels.length, 13, 'unexpected desktop proposal rule');
  assert.equal(batch.slot_count, 13 * toolIds.length);
  assert.equal(batch.ledger.length, batch.slot_count);
  assert.equal(batch.cavity_state_hash, report.initial_cavity_state_hash);
  assert.equal(batch.engine_model_hash, binding.native_config_hash);
  assert.equal(batch.proposal_model_hash, binding.proposal_model_hash);
  assert.equal(batch.rule_hash, binding.proposal_rule_hash);
  const reasons = {}, proposed = new Set(), slots = new Set();
  for (const row of batch.ledger) {
    assert.ok(Number.isInteger(row.column_index) && row.column_index >= 0 && row.column_index < 13, 'out-of-range column slot');
    const key = `${row.column_index}:${row.tool_id}`;
    assert.ok(!slots.has(key), 'duplicate column/tool slot'); slots.add(key);
    assert.deepEqual(row.offset_source_voxels, binding.proposal_rule.offsets_source_voxels[row.column_index]);
    assert.ok(toolIds.includes(row.tool_id));
    reasons[row.reason] = (reasons[row.reason] || 0) + 1;
    if (row.proposal_id !== null) {
      assert.equal(row.reason, 'PROPOSED_UNCERTIFIED');
      assert.ok(!proposed.has(row.proposal_id), 'duplicate proposal'); proposed.add(row.proposal_id);
    } else assert.notEqual(row.reason, 'PROPOSED_UNCERTIFIED');
  }
  assert.deepEqual(slots, new Set(binding.proposal_rule.offsets_source_voxels.flatMap((_, index) => toolIds.map(id => `${index}:${id}`))), 'incomplete column/tool Cartesian inventory');
  assert.deepEqual(batch.counts, reasons);
  assert.deepEqual(new Set(batch.proposals.map(p => p.proposal_id)), proposed);
  assert.equal(batch.proposals.length, proposed.size);
  const accepted = [];
  for (const proposal of batch.proposals) {
    const slot = batch.ledger.find(row => row.proposal_id === proposal.proposal_id);
    for (const key of ['column_index', 'offset_source_voxels', 'tool_id']) assert.deepEqual(proposal[key], slot[key]);
    const attempts = inventory.attempts.filter(a => a.proposal_id === proposal.proposal_id);
    assert.ok(attempts.length >= 1 && attempts.length <= 2, 'missing or duplicate preview');
    assert.equal(attempts[0].phase, 'primary');
    assert.deepEqual(attempts[0].tip_mm, proposal.primary_target_mm);
    if (attempts.length === 2) {
      assert.equal(attempts[0].feasible, false, 'fallback after an accepted primary');
      assert.equal(attempts[1].phase, 'fallback');
      assert.deepEqual(attempts[1].tip_mm, proposal.fallback_target_mm);
    } else if (!attempts[0].feasible) assert.equal(proposal.fallback_target_mm, null, 'missing fallback attempt');
    for (const attempt of attempts) {
      assert.equal(attempt.status, 'complete');
      assert.equal(typeof attempt.feasible, 'boolean');
      assert.equal(attempt.tool_id, proposal.tool_id);
      assert.deepEqual(attempt.entry_mm, proposal.entry_mm);
      if (attempt.feasible) accepted.push(attempt);
    }
  }
  assert.ok(inventory.attempts.every(a => proposed.has(a.proposal_id)), 'unaccounted preview attempt');
  count(report.legal_non_stop_actions);
  assert.equal(report.legal_non_stop_actions, accepted.length);
  assert.equal(report.actions.length, accepted.length + 1);
  assert.ok(report.actions.every(action => typeof action.action_id === 'string' && action.action_id.length > 0));
  assert.equal(new Set(report.actions.map(action => action.action_id)).size, report.actions.length, 'duplicate or reserved action ID');
  assert.deepEqual(report.actions[0], { action_id: 'STOP', kind: 'stop', geometry: null, native_preview: null });
  assert.equal(new Set(inventory.certified_action_ids).size, accepted.length);
  assert.deepEqual(report.actions.slice(1).map(a => a.action_id), inventory.certified_action_ids);
  for (const [i, action] of report.actions.slice(1).entries()) {
    const attempt = accepted[i];
    assert.equal(action.kind, 'native_stroke');
    assert.equal(action.geometry.frame, 'RAS+');
    for (const key of ['tool_id', 'entry_mm', 'tip_mm']) assert.deepEqual(action.geometry[key], attempt[key]);
    const preview = action.native_preview;
    assert.equal(preview.scope, 'native_engine_preview_only');
    assert.equal(preview.independent_history_checked, false);
    assert.equal(preview.proposal_id, attempt.proposal_id);
    assert.equal(preview.phase, attempt.phase);
    assert.equal(preview.source_hash, batch.source_hash);
    assert.equal(preview.source_state_hash, batch.cavity_state_hash);
    assert.equal(preview.native_config_hash, binding.native_config_hash);
    count(preview.unexecuted_contained_cell_count); count(preview.unexecuted_contact_cell_count);
    assert.ok(preview.unexecuted_contained_cell_count > 0, 'accepted native preview has no contained cells');
  }
  assert.equal(report.status, accepted.length ? 'ready' : 'no_actionable_moves');
  return { slots: batch.slot_count, proposed: proposed.size, previewAttempts: inventory.attempts.length,
    legalNonStopActions: accepted.length, status: report.status, bindingHash: binding.binding_hash };
}

async function cancellationProbe(engine, args) {
  const events = [];
  let requestId, cancellation;
  const observe = event => {
    if (event.op !== 'inspectAxisPlanning') return;
    events.push(structuredClone(event));
    if (event.event === 'started' && !requestId) {
      requestId = event.id;
      cancellation = engine.request('cancel', { requestId }).then(result => ({ result }), error => ({ error }));
    }
  };
  engine.on('event', observe);
  try {
    const terminal = await engine.request('inspectAxisPlanning', args).then(result => ({ result }), error => ({ error }));
    assert.ok(requestId && cancellation, 'inspection must report its started request ID');
    const acknowledgement = await cancellation;
    if (acknowledgement.error) throw acknowledgement.error;
    assert.equal(acknowledgement.result.requestId, requestId);
    assert.equal(typeof acknowledgement.result.cancelled, 'boolean');
    const terminals = events.filter(e => e.id === requestId && ['result', 'error', 'cancelled'].includes(e.event));
    assert.equal(terminals.length, 1, 'exactly one client-delivered terminal event per request');
    if (acknowledgement.result.cancelled) {
      assert.match(String(terminal.error), /Operation cancelled/);
      assert.equal(terminals[0].event, 'cancelled');
      assert.equal(terminal.result, undefined);
      return { outcome: 'cancelled_before_completion', requestId, acknowledgement: acknowledgement.result, events };
    }
    assert.ok(terminal.result, 'false cancellation is valid only if complete result won the race');
    assert.equal(terminals[0].event, 'result');
    return { outcome: 'completed_before_cancellation', requestId, acknowledgement: acknowledgement.result, events, result: terminal.result };
  } finally { engine.off('event', observe); }
}

function validateRawCancellation(wire, requestId, outcome) {
  const messages = wire.split('\n').filter(line => line.trim()).map(line => JSON.parse(line));
  const terminals = messages.filter(event => event.id === requestId && ['result', 'error', 'cancelled'].includes(event.event));
  assert.equal(terminals.length, 1, 'raw engine emitted duplicate or missing cancellation terminal');
  assert.equal(terminals[0].event, outcome === 'cancelled_before_completion' ? 'cancelled' : 'result');
  return { rawTerminalCount: terminals.length, terminal: terminals[0].event };
}

async function verifyWorkflow(engine, { casePath, blockedCasePath }, record) {
  const ping = await engine.request('ping');
  assert.equal(ping.protocolVersion, 1);
  assert.ok(ping.operations.includes('inspectAxisPlanning'), 'frozen engine lacks the inspection operation');
  const source = await engine.request('loadCase', { path: casePath });
  await record('loaded-case', source);
  assert.equal(source.brainMask, null, 'acknowledgment probe requires an explicit skull-stripped case without reviewed brain support');
  assert.notEqual(source.metadata?.structural_coverage, 'full_head', 'positive fixture cannot silently be full-head');
  const before = await engine.request('inspectEvidence', { caseHash: source.caseHash });
  const routes = await engine.request('generateNativeRoutes', { caseHash: source.caseHash });
  await record('native-routes', routes);
  const route = routes.candidates.find(r => r.geometry.feasible === true && r.tool.tool_id === 'native-wide-aspiration');
  assert.ok(route, 'explicit input case must yield a feasible named native-wide approach; no substitute case is created');
  const args = { caseHash: source.caseHash, planningHash: source.planningHash, routeId: route.route_id,
    routePlanningModelHash: route.planning_model_hash, toolIds: [...NATIVE_TOOLS],
    acknowledgeNeighboringColumns: true, acknowledgeEstimatedSupport: true };
  await record('inspection-request', args);
  const rejected = [];
  for (const [name, patch, pattern] of [
    ['neighbor_ack', { acknowledgeNeighboringColumns: false }, /acknowledg|neighbor/i],
    ['support_ack', { acknowledgeEstimatedSupport: false }, /acknowledg|support/i],
    ['case_hash', { caseHash: differentHash(args.caseHash) }, /case|stale/i],
    ['planning_hash', { planningHash: differentHash(args.planningHash) }, /planning|stale/i],
    ['route_id', { routeId: 'unavailable-verification-route' }, /route|stale/i],
    ['route_model', { routePlanningModelHash: differentHash(args.routePlanningModelHash) }, /model|stale/i],
  ]) {
    let message;
    await assert.rejects(engine.request('inspectAxisPlanning', { ...args, ...patch }), error => {
      message = String(error); return pattern.test(message);
    }, name);
    rejected.push({ name, message });
  }
  await record('rejections', rejected);
  const result = await engine.request('inspectAxisPlanning', args);
  await record('inspection-result', result); // Full result precedes summary validation.
  const summary = validateInspection(result, args, route, source, ping.nativeResearchTools);
  await assert.rejects(engine.request('inspectAxisPlanning', { ...args,
    expectedBindingHash: differentHash(result.inspection.binding.binding_hash) }), /binding|stale/i);
  const cancellation = await cancellationProbe(engine, args);
  await record('cancellation', cancellation);
  if (cancellation.result) validateInspection(cancellation.result, args, route, source, ping.nativeResearchTools);
  assert.deepEqual(await engine.request('inspectEvidence', { caseHash: source.caseHash }), before, 'inspection changed source evidence');
  const blocked = await engine.request('loadCase', { path: blockedCasePath });
  await record('blocked-case', blocked);
  assert.notEqual(blocked.caseHash, source.caseHash, 'blocked input must be a different actual case');
  assert.equal(blocked.metadata?.structural_coverage, 'full_head', 'blocked fixture must explicitly be full-head');
  await assert.rejects(engine.request('generateNativeRoutes', { caseHash: blocked.caseHash }), /support|envelope|brain|full.head/i);
  await assert.rejects(engine.request('inspectAxisPlanning', { ...args, caseHash: blocked.caseHash,
    planningHash: blocked.planningHash }), /route|stale|support|envelope/i);
  await assert.rejects(engine.assets.read(source.mri.assetId), /Unknown or expired/);
  return { ...summary, checks: { twoSeparateAcknowledgments: true, staleCasePlanningRouteModel: true,
    completeInitialInventory: true, zeroTransitionsGradientsCommitsRemoval: true, sourceEvidenceUnchanged: true,
    blockedFullHead: true, previousSourceAssetsInvalidated: true }, cancellationOutcome: cancellation.outcome,
    cancellationRequestId: cancellation.requestId,
    scopeLimits: ['Read-only protocol validation; no training, independent history certificate or clinical approval.',
      'Engine cases are cached by source identity; UI late-response suppression is checked separately in the renderer.',
      'Inspection/binding hashes are preserved from the authoritative raw receipt; this verifier does not reimplement Python JSON hashing.'] };
}

function parseArguments(argv) {
  const allowed = new Set(['--engine', '--case', '--blocked-case', '--report-dir']);
  const values = {};
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i], value = argv[i + 1];
    if (!allowed.has(key) || values[key] || !value || !path.isAbsolute(value)) throw new Error('Require unique explicit absolute --engine, --case, --blocked-case and --report-dir paths');
    values[key] = value;
  }
  if (Object.keys(values).length !== allowed.size) throw new Error('Missing explicit engine, case, blocked-case or report-dir path');
  return values;
}

async function fileIdentity(filename) {
  const bytes = await fs.readFile(filename);
  return { path: await fs.realpath(filename), bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') };
}

async function main(argv = process.argv.slice(2)) {
  const options = parseArguments(argv);
  await fs.mkdir(options['--report-dir']); // Refuse to overwrite a historical verification.
  const report = { schemaVersion: 1, passed: false, startedUtc: new Date().toISOString(), scope: 'frozen_read_only_axis_inspection', trainingRequested: false };
  const events = [], rawChunks = [];
  const transferDir = await fs.mkdtemp(path.join(os.tmpdir(), 'ressectionlab-axis-verification-'));
  let engine, engineClosed;
  const record = (name, value) => fs.writeFile(path.join(options['--report-dir'], `${name}.json`), JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });
  try {
    const manifest = path.join(path.dirname(options['--engine']), '_internal/build_info/BUILD_INPUT_MANIFEST.json');
    report.requestedPaths = { executable: options['--engine'], engineManifest: manifest,
      caseFile: options['--case'], blockedCaseFile: options['--blocked-case'] };
    report.identities = Object.fromEntries(await Promise.all(Object.entries(report.requestedPaths).map(async ([name, filename]) => [name, await fileIdentity(filename)])));
    await record('engine-manifest', JSON.parse(await fs.readFile(manifest, 'utf8')));
    const { Sidecar } = require('./sidecar.cjs');
    engine = new Sidecar({ executable: options['--engine'], transferDir, cwd: os.tmpdir() });
    engineClosed = new Promise(resolve => engine.child.once('close', resolve));
    // Listen to the raw pipe as well as delivered events. Sidecar intentionally
    // drops late messages after a terminal event, so its events alone cannot
    // prove the engine withheld a late or duplicate result.
    engine.child.stdout.on('data', chunk => rawChunks.push(String(chunk)));
    engine.on('event', event => events.push(structuredClone(event)));
    engine.on('diagnostic', text => { report.diagnostics = (report.diagnostics || '') + text; });
    report.summary = await verifyWorkflow(engine, { casePath: options['--case'], blockedCasePath: options['--blocked-case'] }, record);
    // Reopen each original argument, not only its previously resolved target;
    // retargeting a supplied symlink must also invalidate the input identity.
    for (const [name, identity] of Object.entries(report.identities)) assert.deepEqual(await fileIdentity(report.requestedPaths[name]), identity, 'verification input changed');
    report.passed = true;
  } catch (error) { report.error = String(error.stack || error); process.exitCode = 1; }
  finally {
    if (engine) await engine.stop(); else await fs.rm(transferDir, { recursive: true, force: true });
    if (engineClosed) {
      let timeout;
      try {
        await Promise.race([engineClosed, new Promise((_, reject) => {
          timeout = setTimeout(() => reject(new Error('Engine pipes did not close; raw JSONL capture is incomplete')), 5000);
        })]);
      } catch (error) { report.passed = false; report.error = String(error.stack || error); process.exitCode = 1; }
      finally { clearTimeout(timeout); }
    }
    const rawWire = rawChunks.join('');
    await fs.writeFile(path.join(options['--report-dir'], 'engine-jsonl.txt'), rawWire, { flag: 'wx' });
    if (report.passed) {
      try { report.rawCancellation = validateRawCancellation(rawWire, report.summary.cancellationRequestId, report.summary.cancellationOutcome); }
      catch (error) { report.passed = false; report.error = String(error.stack || error); process.exitCode = 1; }
    }
    await record('events', events); await record('verification', report);
    process.stdout.write(JSON.stringify({ passed: report.passed, output: options['--report-dir'], error: report.error }) + '\n');
  }
}

module.exports = { validateInspection, cancellationProbe, validateRawCancellation, verifyWorkflow, parseArguments };
if (require.main === module) main().catch(error => { process.stderr.write(String(error.stack) + '\n'); process.exitCode = 1; });
