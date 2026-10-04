'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const { validateInspection, cancellationProbe, validateRawCancellation, verifyWorkflow, parseArguments } = require('./verify-axis-inspection.cjs');

const hash = character => `sha256:${character.repeat(64)}`;
function fixture() {
  const tools = [{ tool_id: 'native-fine-aspiration', tip_radius_mm: 1.25 },
    { tool_id: 'native-wide-aspiration', tip_radius_mm: 2.25 }];
  const window = { window_id: 'test-window', center_mm: [0, 1, 2], normal_inward: [1, 0, 0], radius_mm: 6 };
  const source = { caseId: 'synthetic-protocol-only', caseHash: hash('a'), planningHash: hash('b'), frame: 'RAS+', shape: [4, 4, 4],
    affine: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], brainMask: null, mri: { assetId: 'old' } };
  const route = { route_id: 'native-axis-fixture', planning_model_hash: 'c'.repeat(64), window,
    tool: tools[1], geometry: { feasible: true } };
  const request = { caseHash: source.caseHash, planningHash: source.planningHash,
    routeId: route.route_id, routePlanningModelHash: route.planning_model_hash,
    toolIds: tools.map(t => t.tool_id), acknowledgeNeighboringColumns: true, acknowledgeEstimatedSupport: true };
  const offsets = Array.from({ length: 13 }, (_, index) => [index, 0]);
  const ledger = offsets.flatMap((offset, column_index) => tools.map((tool, toolIndex) => ({ column_index,
    offset_source_voxels: offset, tool_id: tool.tool_id, reason: column_index === 0 && toolIndex === 0 ? 'PROPOSED_UNCERTIFIED' : 'NO_REMAINING_TARGET_IN_COLUMN',
    proposal_id: column_index === 0 && toolIndex === 0 ? 'proposal-one' : null })));
  const proposal = { proposal_id: 'proposal-one', column_index: 0, offset_source_voxels: [0, 0],
    tool_id: tools[0].tool_id, entry_mm: [0, 1, 2], primary_target_mm: [3, 1, 2], fallback_target_mm: [2, 1, 2] };
  const attempt = { proposal_id: proposal.proposal_id, phase: 'primary', tool_id: proposal.tool_id,
    entry_mm: proposal.entry_mm, tip_mm: proposal.primary_target_mm, feasible: true, reason: 'accepted', status: 'complete' };
  const batch = { source_hash: hash('d'), cavity_state_hash: hash('e'), engine_model_hash: hash('f'),
    proposal_model_hash: hash('1'), rule_hash: hash('2'), unsupported_reason: null, geometry_certified: false,
    removal_authorized: false, slot_count: 26, ledger, counts: { PROPOSED_UNCERTIFIED: 1, NO_REMAINING_TARGET_IN_COLUMN: 25 }, proposals: [proposal] };
  const binding = { case_hash: source.caseHash, planning_hash: source.planningHash, case_id: source.caseId,
    geometry_frame: 'RAS+', source_frame: source.frame, source_shape: source.shape, native_affine_ras_mm: source.affine,
    access: window, requested_access_ras: window,
    access_conversion: { source_frame: 'RAS+', normal_absolute_tolerance: 0, normal_maximum_absolute_difference: 0 },
    tools, neighboring_columns_acknowledged: true, source_support: { estimated_support_acknowledged: true },
    input_profile: 'RAW', world_role: null, world_partitions_created: false, population_priors_used: false,
    functional_evidence_available: { motor: false, language: false }, binding_hash: hash('3'),
    proposal_rule: { offsets_source_voxels: offsets }, native_config_hash: batch.engine_model_hash,
    proposal_model_hash: batch.proposal_model_hash, proposal_rule_hash: batch.rule_hash };
  const inspection = { version: 'native-axis-inspection-v1', role: 'inspection', inventory_complete: true,
    candidate_eligible: false, removal_authorized: false, clinical_deficit_probability: null,
    accounting: { gradient_steps: 0, executed_transitions: 0, native_commits: 0, simulated_removed_volume_mm3: 0 }, binding,
    inspection_hash: hash('4'), initial_cavity_state_hash: batch.cavity_state_hash,
    inventory: { status: 'complete', batch, attempts: [attempt], certified_action_ids: ['action-one'] },
    legal_non_stop_actions: 1, status: 'ready', actions: [
      { action_id: 'STOP', kind: 'stop', geometry: null, native_preview: null },
      { action_id: 'action-one', kind: 'native_stroke', geometry: { frame: 'RAS+', tool_id: attempt.tool_id,
        entry_mm: attempt.entry_mm, tip_mm: attempt.tip_mm }, native_preview: { scope: 'native_engine_preview_only',
        independent_history_checked: false, proposal_id: proposal.proposal_id, phase: 'primary',
        source_hash: batch.source_hash, source_state_hash: batch.cavity_state_hash, native_config_hash: batch.engine_model_hash,
        unexecuted_contained_cell_count: 3, unexecuted_contact_cell_count: 5 } },
    ] };
  const result = { schemaVersion: 1, caseHash: request.caseHash, planningHash: request.planningHash,
    routeId: request.routeId, routePlanningModelHash: request.routePlanningModelHash, accessSource: 'selected_route_window_only',
    anchorWindowRas: window, anchorWindowNormalization: { normalAbsoluteTolerance: 4 * Number.EPSILON, normalMaximumDifference: 0 },
    requestedToolIds: request.toolIds, inspection };
  return structuredClone({ source, route, request, tools, result });
}

test('complete synthetic protocol inventory is counted without treating previews as removal', () => {
  const f = fixture();
  assert.deepEqual(validateInspection(f.result, f.request, f.route, f.source, f.tools), {
    slots: 26, proposed: 1, previewAttempts: 1, legalNonStopActions: 1, status: 'ready', bindingHash: hash('3') });
  assert.equal(f.result.inspection.accounting.simulated_removed_volume_mm3, 0);
});

test('LPS source grids and cached windows must retain their exact physical RAS geometry', () => {
  const f = fixture();
  f.source.frame = 'LPS+';
  f.source.affine = [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  f.route.window = { ...f.route.window, center_mm: [0, -1, 2], normal_inward: [-1, 0, 0] };
  f.result.inspection.binding.source_frame = 'LPS+';
  Object.assign(f.result.inspection.binding.access_conversion, { source_frame: 'LPS+', normal_absolute_tolerance: 4 * Number.EPSILON });
  assert.equal(validateInspection(f.result, f.request, f.route, f.source, f.tools).slots, 26);
});

test('complete supported STOP-only inventory remains an honest no-action result', () => {
  const f = fixture(), report = f.result.inspection, batch = report.inventory.batch;
  Object.assign(batch.ledger[0], { reason: 'NO_REMAINING_TARGET_IN_COLUMN', proposal_id: null });
  batch.counts = { NO_REMAINING_TARGET_IN_COLUMN: 26 }; batch.proposals = [];
  report.inventory.attempts = []; report.inventory.certified_action_ids = [];
  report.actions = report.actions.slice(0, 1); report.legal_non_stop_actions = 0; report.status = 'no_actionable_moves';
  assert.equal(validateInspection(f.result, f.request, f.route, f.source, f.tools).legalNonStopActions, 0);
});

test('receipt checker rejects stale identity, partial inventories, forged counts and clinical promotion', () => {
  for (const alter of [
    f => { f.result.caseHash = hash('f'); },
    f => { f.result.inspection.binding.native_affine_ras_mm = [[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]; },
    f => { f.result.inspection.binding.access_conversion.source_frame = 'LPS+'; },
    f => { f.result.inspection.inventory_complete = false; },
    f => { f.result.inspection.inventory.batch.ledger.pop(); },
    f => { const slot = f.result.inspection.inventory.batch.ledger.at(-1); slot.column_index = 99; delete slot.offset_source_voxels; },
    f => { f.result.inspection.inventory.batch.counts.NO_REMAINING_TARGET_IN_COLUMN = 24; },
    f => { f.result.inspection.inventory.attempts = []; },
    f => { f.result.inspection.actions[1].geometry.tool_id = 'unrequested'; },
    f => { f.result.inspection.actions[1].action_id = 'STOP'; f.result.inspection.inventory.certified_action_ids = ['STOP']; },
    f => { f.result.inspection.actions[1].native_preview.source_state_hash = hash('a'); },
    f => { f.result.inspection.accounting.executed_transitions = 1; },
    f => { f.result.inspection.candidate_eligible = true; },
    f => { f.result.inspection.clinical_deficit_probability = 0.01; },
  ]) {
    const f = fixture(); alter(f);
    assert.throws(() => validateInspection(f.result, f.request, f.route, f.source, f.tools));
  }
});

test('a missing required fallback and a fallback after primary acceptance are rejected', () => {
  const f = fixture(); f.result.inspection.inventory.attempts[0].feasible = false;
  assert.throws(() => validateInspection(f.result, f.request, f.route, f.source, f.tools), /missing fallback/);
  const g = fixture();
  g.result.inspection.inventory.attempts.push({ ...g.result.inspection.inventory.attempts[0], phase: 'fallback', tip_mm: [2, 1, 2] });
  assert.throws(() => validateInspection(g.result, g.request, g.route, g.source, g.tools), /fallback after/);
});

test('explicit path parser refuses defaults, duplicates, relative paths and unknown flags', () => {
  const args = ['--engine', '/tmp/engine', '--case', '/tmp/one.ressectionlab', '--blocked-case', '/tmp/full.ressectionlab', '--report-dir', '/tmp/new'];
  assert.equal(parseArguments(args)['--engine'], '/tmp/engine');
  for (const invalid of [[], args.slice(0, 6), [...args, '--engine', '/tmp/other'], [...args, '--train', '/tmp/no'],
    ['--engine', 'relative', ...args.slice(2)]]) assert.throws(() => parseArguments(invalid));
});

class MockEngine extends EventEmitter {
  constructor(mode = 'cancel') { super(); this.mode = mode; this.calls = []; this.f = fixture(); this.nextId = 0; this.cancelResult = null;
    this.assets = { read: async () => { throw new Error('Unknown or expired transfer asset'); } }; }
  async request(op, args = {}) {
    this.calls.push({ op, args: structuredClone(args) });
    if (op === 'ping') return { protocolVersion: 1, operations: ['inspectAxisPlanning'], nativeResearchTools: this.f.tools };
    if (op === 'loadCase') { this.blocked = args.path.includes('blocked'); return this.blocked
      ? { ...this.f.source, caseHash: hash('7'), planningHash: hash('8'), metadata: { structural_coverage: 'full_head' } } : this.f.source; }
    if (op === 'inspectEvidence') return { caseHash: args.caseHash, unknowns: ['unassessed'] };
    if (op === 'generateNativeRoutes') {
      if (this.blocked) throw new Error('BRAIN_ENVELOPE_REQUIRED: reviewed support needed');
      return { candidates: [this.f.route] };
    }
    if (op === 'cancel') return { requestId: args.requestId, cancelled: this.mode !== 'complete' };
    assert.equal(op, 'inspectAxisPlanning', 'verifier requested an unexpected operation');
    for (const key of ['caseHash', 'planningHash', 'routeId', 'routePlanningModelHash']) {
      if (args[key] !== this.f.request[key]) throw new Error(`stale ${key}`);
    }
    if (!args.acknowledgeNeighboringColumns) throw new Error('neighbor acknowledgment required');
    if (!args.acknowledgeEstimatedSupport) throw new Error('support acknowledgment required');
    if (args.expectedBindingHash && args.expectedBindingHash !== hash('3')) throw new Error('stale binding');
    const id = `inspection-${++this.nextId}`;
    this.emit('event', { id, op, event: 'started' });
    const cancellationPending = this.calls.some(call => call.op === 'cancel' && call.args.requestId === id);
    if (cancellationPending && this.mode === 'cancel') {
      this.emit('event', { id, op, event: 'cancelled' }); throw new Error('Operation cancelled');
    }
    const result = structuredClone(this.f.result);
    this.emit('event', { id, op, event: 'result', result }); return result;
  }
}

test('cancellation before completion has one cancelled terminal and no published inventory', async () => {
  const engine = new MockEngine();
  const result = await cancellationProbe(engine, engine.f.request);
  assert.equal(result.outcome, 'cancelled_before_completion'); assert.equal(result.result, undefined);
});

test('completion winning cancellation is recorded honestly instead of failing on cancelled false', async () => {
  const engine = new MockEngine('complete');
  const result = await cancellationProbe(engine, engine.f.request);
  assert.equal(result.outcome, 'completed_before_cancellation');
  assert.equal(result.acknowledgement.cancelled, false); assert.equal(result.result.inspection.inventory_complete, true);
});

test('a true cancellation acknowledgement followed by a successful result is inconsistent', async () => {
  const engine = new MockEngine('inconsistent');
  await assert.rejects(cancellationProbe(engine, engine.f.request));
});

test('raw wire verification catches a late result that a completed Sidecar promise would hide', () => {
  const cancelled = { id: 'one', event: 'cancelled' }, late = { id: 'one', event: 'result', result: {} };
  assert.deepEqual(validateRawCancellation(JSON.stringify(cancelled) + '\n', 'one', 'cancelled_before_completion'), { rawTerminalCount: 1, terminal: 'cancelled' });
  for (const events of [[cancelled, late], [cancelled, cancelled], [late], []]) {
    assert.throws(() => validateRawCancellation(events.map(x => JSON.stringify(x)).join('\n'), 'one', 'cancelled_before_completion'));
  }
  assert.equal(validateRawCancellation(JSON.stringify(late), 'one', 'completed_before_cancellation').terminal, 'result');
});

test('mock workflow preserves raw result, rejects both acknowledgments and identities, never trains', async () => {
  const engine = new MockEngine(), records = new Map();
  const report = await verifyWorkflow(engine, { casePath: '/tmp/case.ressectionlab', blockedCasePath: '/tmp/blocked.ressectionlab' }, async (name, value) => records.set(name, structuredClone(value)));
  assert.equal(report.checks.completeInitialInventory, true);
  assert.equal(records.get('inspection-result').inspection.inventory.batch.ledger.length, 26);
  assert.equal(records.get('rejections').length, 6);
  assert.ok(engine.calls.every(call => !/train|save|export|replay/i.test(call.op)));
});

test('raw inspection result is retained before a failed summary validation', async () => {
  const engine = new MockEngine(), records = new Map();
  engine.f.result.inspection.accounting.native_commits = 1;
  await assert.rejects(verifyWorkflow(engine, { casePath: '/tmp/case.ressectionlab', blockedCasePath: '/tmp/blocked.ressectionlab' }, async (name, value) => records.set(name, structuredClone(value))));
  assert.equal(records.get('inspection-result').inspection.accounting.native_commits, 1);
});
