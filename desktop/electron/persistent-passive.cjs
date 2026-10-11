'use strict';

// This is a generated software-control envelope. It is intentionally separate
// from the legacy stroke episode and admits no observed or patient replay.
const { createHash } = require('node:crypto');
const { isDeepStrictEqual } = require('node:util');

const HASH = /^sha256:[a-f0-9]{64}$/;
const SCHEMA = 'resectionlab.generated-persistent-passive-episode.v1';
const ACTIONS = ['PASSIVE_MOVE', 'ACQUIRE_GENERATED_STATUS', 'PASSIVE_WITHDRAW',
  'PASSIVE_WITHDRAW_STATIONARY', 'STOP'];
const PHASES = ['ready', 'held', 'inspected', 'withdrawn', 'both_withdrawn', 'stopped'];
const TIME_SOURCE = 'fixed_generated_schedule_not_measured_surgical_time';
const SCOPE = 'generated passive geometry and declared time only; no surgical or sensor validation';
const FRAME_MEANING = 'operative state after each committed passive action; not legacy native stroke frames';
const MAX_EPISODE_BYTES = 2 * 1024 * 1024;
const FIXTURE = 'generated-persistent-passive-v1';
const PARAMETER_SOURCE = 'generic research geometry; not a verified commercial device';

function object(value) { return value !== null && typeof value === 'object' && !Array.isArray(value); }
function exact(value, keys) {
  return object(value) && Object.keys(value).length === keys.length &&
    keys.every(key => Object.hasOwn(value, key));
}
function need(condition, reason) {
  if (!condition) throw new Error(`Generated passive replay withheld: ${reason}`);
}
function digest(value) { return typeof value === 'string' && HASH.test(value); }
function sha(value) { return `sha256:${createHash('sha256').update(value).digest('hex')}`; }
function same(a, b) { return isDeepStrictEqual(a, b); }
function finite(value) { return typeof value === 'number' && Number.isFinite(value); }
function close(a, b) { return finite(a) && finite(b) && Math.abs(a - b) < 1e-8; }
function pose(value) {
  return exact(value, ['tip_mm', 'axis_unit']) &&
    [value.tip_mm, value.axis_unit].every(row => Array.isArray(row) && row.length === 3 && row.every(finite)) &&
    close(value.axis_unit.reduce((sum, component) => sum + component * component, 0), 1);
}
function jsonBound(value, limit, label) {
  need(typeof value === 'string' && Buffer.byteLength(value) <= limit, `${label} size changed`);
  try { return JSON.parse(value); } catch { throw new Error(`Generated passive replay withheld: invalid ${label}`); }
}
function persistentPassiveRequest(value) {
  need(exact(value, ['fixture', 'paired']) && value.fixture === FIXTURE && value.paired === true,
    'choose only the fixed paired generated passive fixture');
  return {fixture: FIXTURE, paired: true};
}
function toolRecord(tool, id) {
  return exact(tool, ['tool_id', 'tip_radius_mm', 'shaft_radius_mm', 'working_length_mm',
    'max_access_angle_deg', 'tip_length_mm', 'parameter_source']) &&
    tool.tool_id === id && tool.tip_radius_mm === .15 && tool.shaft_radius_mm === .1 &&
    tool.working_length_mm === 6 && tool.max_access_angle_deg === 30 &&
    tool.tip_length_mm === .4 && tool.parameter_source === PARAMETER_SOURCE;
}
// The generated ToolGeometry dataclass owns floating-point dimensions. Python
// canonical JSON writes integer-valued floats as 6.0/30.0, unlike JSON.stringify.
// This serializer is deliberately limited to the frozen binding shape.
function bindingDigest(binding) {
  const quoted = JSON.stringify;
  const tool = item => `{"max_access_angle_deg":30.0,"parameter_source":${quoted(item.parameter_source)},` +
    `"shaft_radius_mm":0.1,"tip_length_mm":0.4,"tip_radius_mm":0.15,` +
    `"tool_id":${quoted(item.tool_id)},"working_length_mm":6.0}`;
  return sha(`{"affineHash":${quoted(binding.affineHash)},"caseHash":${quoted(binding.caseHash)},` +
    `"decisionModelHash":${quoted(binding.decisionModelHash)},` +
    `"nominalTargetMaskHash":${quoted(binding.nominalTargetMaskHash)},` +
    `"sourceFrameHash":${quoted(binding.sourceFrameHash)},` +
    `"sourceHash":${quoted(binding.sourceHash)},` +
    `"structuralIntensityHash":${quoted(binding.structuralIntensityHash)},` +
    `"supportHash":${quoted(binding.supportHash)},"tools":[${binding.tools.map(tool).join(',')}]}`);
}

/** Validate only the fixed, paired, six-frame generated passive v1 envelope.
 * This is a transport check, not proof of surgical outcomes or saved-file
 * authorship. Reopen is admitted only after the backend's fresh native replay
 * and a distinct workspace validation marker have been checked. */
function validatePersistentPassiveV1Result(result, { origin = 'live', revalidation } = {}) {
  need(origin === 'live' || (origin === 'reopened' &&
    revalidation === 'authoritative_generated_passive_replay'),
  'saved passive replay lacks backend revalidation');
  need(exact(result, ['case', 'episode', 'episodeCanonicalJson', 'sourceBinding']),
    'unexpected response members');
  const e = result.episode, c = result.case, b = result.sourceBinding;
  need(exact(b, ['caseHash', 'sourceHash', 'decisionModelHash', 'structuralIntensityHash',
    'supportHash', 'nominalTargetMaskHash', 'affineHash', 'sourceFrameHash', 'tools', 'bindingHash']) &&
    [b.caseHash, b.sourceHash, b.decisionModelHash, b.structuralIntensityHash,
      b.supportHash, b.nominalTargetMaskHash, b.affineHash, b.sourceFrameHash,
      b.bindingHash].every(digest) &&
    Array.isArray(b.tools) && b.tools.length === 2 &&
    toolRecord(b.tools[0], e?.protocol?.stationary_tool_id) &&
    toolRecord(b.tools[1], e?.protocol?.moving_tool_id) &&
    bindingDigest(b) === b.bindingHash, 'invalid fixed source/tool binding');
  need(exact(e, ['schema', 'sourceHash', 'decisionModelHash', 'protocol', 'actions', 'history',
    'physicalHistoryCanonicalJson', 'physicalHistoryHash', 'frames', 'frameMeaning',
    'finalStateHash', 'scope', 'episodeId']), 'unexpected episode members');
  need(e.schema === SCHEMA && e.scope === SCOPE && e.frameMeaning === FRAME_MEANING &&
    [e.sourceHash, e.decisionModelHash, e.physicalHistoryHash, e.finalStateHash, e.episodeId].every(digest) &&
    e.sourceHash === b.sourceHash && e.decisionModelHash === b.decisionModelHash,
    'episode source, scope or identity changed');
  need(object(c) && c.caseHash === b.caseHash && c.frame === 'RAS+' && c.physicalUnits === 'mm' &&
    same(c.shape, [13, 13, 12]) && c.metadata?.is_synthetic === true &&
    c.metadata?.evidence_kind === 'generated_software_fixture' &&
    c.metadata?.patient_admission === false && c.metadata?.source_task_hash === e.sourceHash,
    'display CaseData is not the bound generated source');

  const canonical = jsonBound(result.episodeCanonicalJson, MAX_EPISODE_BYTES, 'episode serialization');
  const { episodeId, ...body } = e;
  need(sha(result.episodeCanonicalJson) === e.episodeId && same(canonical, body),
    'episode differs from its canonical serialization');
  need(Buffer.byteLength(JSON.stringify(e)) <= MAX_EPISODE_BYTES,
    'episode exceeds response budget');
  const history = jsonBound(e.physicalHistoryCanonicalJson, MAX_EPISODE_BYTES, 'physical history');
  need(sha(e.physicalHistoryCanonicalJson) === e.physicalHistoryHash && same(history, e.history),
    'physical history serialization changed');

  const p = e.protocol;
  need(exact(p, ['version', 'moving_tool_id', 'start_pose', 'hold_pose',
    'stationary_tool_id', 'stationary_pose', 'stationary_exit_pose', 'generated_seconds',
    'time_meaning', 'movement_scope']) && p.version === 'generated-persistent-passive-v1' &&
    typeof p.moving_tool_id === 'string' && typeof p.stationary_tool_id === 'string' &&
    p.moving_tool_id !== p.stationary_tool_id &&
    [p.start_pose, p.hold_pose, p.stationary_pose, p.stationary_exit_pose].every(pose) &&
    Array.isArray(p.generated_seconds) && p.generated_seconds.length === 5 &&
    p.generated_seconds.every(v => finite(v) && v >= 0) &&
    p.generated_seconds.reduce((sum, v) => sum + v, 0) <= 60 &&
    p.time_meaning === 'declared generated schedule; not measured surgical duration' &&
    p.movement_scope === 'one passive rigid tool, other tool stationary; no active cutting',
    'paired tool protocol or declared time changed');
  need(same(e.actions, ACTIONS) && Array.isArray(e.history) && e.history.length === 5 &&
    Array.isArray(e.frames) && e.frames.length === 6,
    'the paired episode must contain five actions and six frames');

  let elapsed = 0;
  let nativeState = null;
  for (let i = 0; i < e.frames.length; i++) {
    const frame = e.frames[i], state = frame?.operative;
    need(exact(frame, ['frameIndex', 'operative']) && frame.frameIndex === i && object(state) &&
      state.phase === PHASES[i] && close(state.elapsed_generated_seconds, elapsed) &&
      object(state.moving_tool) && object(state.other_tool) &&
      state.moving_tool.tool_id === p.moving_tool_id && state.other_tool.tool_id === p.stationary_tool_id &&
      pose(state.moving_tool.pose) && pose(state.other_tool.pose) &&
      state.moving_tool.physical_absence_claim === false &&
      state.other_tool.physical_absence_claim === false,
    `frame ${i} changed its phase, time or operative tool identity`);
    const heldMoving = i === 1 || i === 2;
    const heldOther = i <= 3;
    need(same(state.moving_tool.pose, heldMoving ? p.hold_pose : p.start_pose) &&
      same(state.other_tool.pose, heldOther ? p.stationary_pose : p.stationary_exit_pose) &&
      state.moving_tool.retained_at_depth === heldMoving &&
      state.other_tool.retained_at_depth === heldOther &&
      state.moving_tool.activation === (heldMoving ? 'passive_only' : 'off') &&
      state.other_tool.activation === 'off' &&
      state.moving_tool.pose_status === (heldMoving ? 'held_in_declared_void' : 'access_pose_recorded') &&
      state.other_tool.pose_status === (heldOther ? 'held_in_declared_void' : 'access_pose_recorded'),
    `frame ${i} changed a retained or withdrawn pose`);
    need(state.inspection_status === (i < 2 ? 'not_acquired' : e.history[1]?.information_event?.status),
      `frame ${i} changed the generated information reveal boundary`);
    if (i === e.history.length) break;
    const row = e.history[i], action = ACTIONS[i];
    const duration = p.generated_seconds[i];
    elapsed += duration;
    need(object(row) && row.action_id === action && row.time_source === TIME_SOURCE &&
      close(row.duration_generated_seconds, duration) && close(row.elapsed_generated_seconds, elapsed) &&
      digest(row.source_state_hash) && row.source_state_hash === row.result_state_hash &&
      (nativeState === null || row.source_state_hash === nativeState) &&
      digest(row.operative_before) && digest(row.operative_after) &&
      (i === 0 || row.operative_before === e.history[i - 1].operative_after) &&
      Array.isArray(row.removed_indices_native) && row.removed_indices_native.length === 0 &&
      Array.isArray(row.contact_indices_native) && row.contact_indices_native.length === 0 &&
      row.clinical_deficit_probability === null && row.physiological_response === null && row.reward === 0,
    `history ${i} changed native state, declared time or simulated outcome scope`);
    nativeState = row.result_state_hash;
    const movingMotion = i === 0 || i === 2;
    const otherMotion = i === 3;
    const motion = movingMotion || otherMotion;
    need(row.tool_id === (i === 4 ? null : otherMotion ? p.stationary_tool_id : p.moving_tool_id) &&
      row.interaction_mode === (motion ? 'passive_motion' : i === 1 ? 'generated_information' : 'stop') &&
      same(row.pose_before, otherMotion ? p.stationary_pose : movingMotion && i === 2 ? p.hold_pose :
        i === 1 ? p.hold_pose : p.start_pose) &&
      same(row.pose_after, otherMotion ? p.stationary_exit_pose : movingMotion && i === 0 ? p.hold_pose :
        i === 1 ? p.hold_pose : p.start_pose),
    `history ${i} changed the physical action or pose`);
    need(motion ? row.geometry?.feasible === true && same(row.geometry.failures, []) &&
      row.geometry.exposure_volume_mm3 === 0 && row.independent_tissue_geometry?.feasible === true &&
      row.independent_tissue_geometry.stationary_tool_pair_independently_checked === false &&
      row.information_event === null : row.geometry === null && row.independent_tissue_geometry === null,
    `history ${i} changed the bounded geometric certificate`);
    need(i === 1 ? row.information_event?.version === 'generated-registration-status-v1' &&
      ['verified', 'unresolved'].includes(row.information_event.status) &&
      digest(row.information_event.input_fingerprint) &&
      row.information_event.meaning === 'generated protocol status; no anatomy, sensor or clinical inference' &&
      row.stop_reason === null :
      row.information_event === null && (i !== 4 ? row.stop_reason === null : row.stop_reason ===
        (e.history[1]?.information_event?.status === 'unresolved' ?
          'generated_registration_unresolved' : 'declared_stop_without_injury_claim')),
    `history ${i} changed the generated information or stop reason`);
  }
  return result;
}

module.exports = { persistentPassiveRequest, validatePersistentPassiveV1Result, bindingDigest };
