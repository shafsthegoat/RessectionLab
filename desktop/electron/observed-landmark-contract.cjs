'use strict';

const { createHash } = require('node:crypto');
const STUDY = 'resect-case4-sparse-update-v1';
const FIELD = '5e3944a40a0240af99e8c82360fb02241ea4ab3a7148c60950e2829e64e79858';
const IDS = [1, 14, 8, 17, 19, 7];
// Independent transport digests of the 12 valid states from the pinned six-B
// artifact. No V/raw/image inputs. Serialization is shared with Python's
// transport_digest; these are NOT recomputations of core.semantic_digest.
const PINS = Object.freeze({
  'before:1': 'fe1bf57f35b54832c592f712271a65f5cfcca6819821e296dfaa18b20476e259',
  'before:14': '2e9a3667b889bb26f21a6f0599ebd183be3ae97a5f3f8add304f5ab45aa40641',
  'before:8': 'd8bbe0126a4a7b3a6ce480ca441ae32a6a3f21ffceeed32cb72f629c11ba956e',
  'before:17': '5047b5860231c9bfb21806e327e70b2e6c839d364198b646e65682bec4368061',
  'before:19': '1d879bfe108fb4be16717ed0b1a59522f9f4385e5a6c08b51301f3b5dcedbe53',
  'before:7': '5a252e8990b45894b89520855431e6085b93e4fbc90638e3d4919d4d4d3981b2',
  'during:1': 'c5b25c242fc678aed1b8e8d4c6dfa979bdeb163cc592377c881c58c430f70ede',
  'during:14': 'f96fee5828354c1d4c8545ac35b46bac8c2a34666633171c6114f897da110918',
  'during:8': '86a22a37f5ed3d7091967d2496c05dec7b5ed9e4013feff9f46d9aa6c8a6e5f2',
  'during:17': '9bd9221d8bfe7d736baac9afa9a3df17de864ae4b7f655f597a47d437e6f984f',
  'during:19': '7379e7bc97dfeaa9b27f2ddbcadae11aaf11bf3a4a685f0bd058f558b45b34a0',
  'during:7': '174d516c6af48aff2a939feb5636a19838393cb7ae04919be4eab01e8334c661',
});

function requireValue(condition) {
  if (!condition) throw new Error('Observation payload does not match the pinned source/state contract');
}
function record(value, keys) {
  requireValue(value && typeof value === 'object' && !Array.isArray(value)
    && [Object.prototype, null].includes(Object.getPrototypeOf(value)));
  requireValue(Reflect.ownKeys(value).length === keys.length && keys.every(key => Object.hasOwn(value, key)));
}
const hash = value => typeof value === 'string' && /^sha256:[a-f0-9]{64}$/.test(value);
const vector = value => Array.isArray(value) && value.length === 3
  && [0, 1, 2].every(i => typeof value[i] === 'number' && Number.isFinite(value[i]));
const matrix = value => Array.isArray(value) && value.length === 4
  && [0, 1, 2, 3].every(i => Array.isArray(value[i]) && value[i].length === 4
    && [0, 1, 2, 3].every(j => typeof value[i][j] === 'number' && Number.isFinite(value[i][j])));

function transportDigest(value) {
  function node(item) {
    if (item === null) return ['null'];
    if (typeof item === 'boolean') return ['boolean', item];
    if (typeof item === 'number') {
      requireValue(Number.isFinite(item));
      const bytes = Buffer.alloc(8); bytes.writeDoubleBE(item === 0 ? 0 : item);
      return ['number', bytes.toString('hex')];
    }
    if (typeof item === 'string') return ['string', item];
    if (Array.isArray(item)) return ['array', Array.from(item, node)];
    requireValue(item && typeof item === 'object' && [Object.prototype, null].includes(Object.getPrototypeOf(item)));
    const keys = Reflect.ownKeys(item);
    requireValue(keys.every(key => typeof key === 'string' && /^[\x00-\x7f]*$/.test(key)));
    return ['object', keys.sort().map(key => [key, node(item[key])])];
  }
  return createHash('sha256').update(JSON.stringify(node(value)), 'utf8').digest('hex');
}

function binding(value) {
  record(value, ['studyId', 'patientGroup', 'dataRole', 'fieldSha256', 'partitionHash', 'protocolSha256',
    'sourceImageSha256', 'destinationImageSha256', 'sourcePair', 'sourceDoi', 'sourceLicense']);
  requireValue(value.studyId === STUDY && value.patientGroup === 'RESECT:Case4' && value.dataRole === 'DEVELOPMENT'
    && value.fieldSha256 === FIELD && hash(value.partitionHash) && hash(value.sourceImageSha256)
    && hash(value.destinationImageSha256) && /^[a-f0-9]{64}$/.test(value.protocolSha256)
    && ['sourcePair', 'sourceDoi', 'sourceLicense'].every(key => typeof value[key] === 'string'));
}
function snapshot(value) {
  record(value, ['schemaVersion', 'kind', 'binding', 'phase', 'landmarkId', 'stateHash', 'resultHash']);
  binding(value.binding);
  requireValue(value.schemaVersion === 1 && value.kind === 'observed_landmark_snapshot'
    && ['before', 'during'].includes(value.phase) && IDS.includes(value.landmarkId)
    && hash(value.stateHash) && hash(value.resultHash));
}

function observedRequest(value) {
  const fields = { open: ['action', 'studyId'],
    advance: ['action', 'studyId', 'expectedStateHash', 'phase', 'landmarkId'],
    reopen: ['action', 'studyId', 'expectedStateHash', 'snapshot'] };
  requireValue(value && typeof value.action === 'string' && Object.hasOwn(fields, value.action));
  record(value, fields[value.action]);
  requireValue(value.studyId === STUDY);
  if (value.action !== 'open') requireValue(hash(value.expectedStateHash)
    || (value.action === 'reopen' && value.expectedStateHash === null));
  if (value.action === 'advance') requireValue(['before', 'during'].includes(value.phase) && IDS.includes(value.landmarkId));
  if (value.action === 'reopen') snapshot(value.snapshot);
  requireValue(Buffer.byteLength(JSON.stringify(value)) <= 16 * 1024);
  return JSON.parse(JSON.stringify(value));
}

function validateObservedResult(result, request) {
  request = observedRequest(request);
  record(result, ['schemaVersion', 'binding', 'state', 'stateHash', 'prediction', 'resultHash', 'snapshot']);
  requireValue(Buffer.byteLength(JSON.stringify(result)) <= 32 * 1024);
  binding(result.binding); snapshot(result.snapshot);
  const state = result.state;
  const during = state?.phase === 'during';
  record(state, ['schemaVersion', 'studyId', 'patientGroup', 'dataRole', 'mode', 'phase', 'phaseOrdinal',
    'phaseOrderEvidence', 'acquisitionTimestamp', 'elapsedSeconds', 'measurementAvailabilityTimestamp', 'timeSemantics',
    'frame', 'units', 'partitionHash', 'observationRole', 'boundaryIds', 'sourceImageSha256', 'sourceWorldToRasMm',
    'observations', 'selectedInspectionPoint', 'anatomicalTarget', 'toolPose', 'brainSupport', 'cavitySupport',
    'physicalClearanceMm', 'totalRegistrationUncertaintyMm', 'selectedAction', 'decisionReason',
    ...(during ? ['destinationImageSha256', 'destinationWorldToRasMm'] : [])]);
  requireValue(result.schemaVersion === 1 && state.schemaVersion === 1 && state.studyId === STUDY
    && state.patientGroup === 'RESECT:Case4' && state.dataRole === 'DEVELOPMENT'
    && state.mode === 'observed_correspondence_replay' && ['before', 'during'].includes(state.phase)
    && state.phaseOrdinal === (during ? 1 : 0) && state.frame === 'RAS+' && state.units === 'mm'
    && state.selectedAction === 'ABSTAIN_UNSUPPORTED_CLEARANCE' && matrix(state.sourceWorldToRasMm)
    && (!during || matrix(state.destinationWorldToRasMm))
    && ['acquisitionTimestamp', 'elapsedSeconds', 'measurementAvailabilityTimestamp', 'anatomicalTarget', 'toolPose',
      'brainSupport', 'cavitySupport', 'physicalClearanceMm', 'totalRegistrationUncertaintyMm'].every(key => state[key] === null));
  requireValue(Array.isArray(state.boundaryIds) && state.boundaryIds.length === 6
    && IDS.every((id, i) => state.boundaryIds[i] === id) && Array.isArray(state.observations) && state.observations.length === 6);
  state.observations.forEach((point, i) => {
    record(point, ['landmarkId', 'sourceRasMm', ...(during ? ['observedRasMm', 'measuredDisplacementMm'] : [])]);
    requireValue(point.landmarkId === IDS[i] && vector(point.sourceRasMm)
      && (!during || (vector(point.observedRasMm) && vector(point.measuredDisplacementMm))));
  });
  record(state.selectedInspectionPoint, ['landmarkId', 'kind', 'positionRasMm']);
  const row = state.selectedInspectionPoint.landmarkId;
  requireValue(IDS.includes(row) && vector(state.selectedInspectionPoint.positionRasMm)
    && state.selectedInspectionPoint.kind === 'source_linked_correspondence_not_anatomical_target');
  if (during) {
    record(result.prediction, ['method', 'fieldSha256', 'positionRasMm', 'scope']);
    requireValue(result.prediction.method === 'proper_rigid' && result.prediction.fieldSha256 === FIELD
      && vector(result.prediction.positionRasMm) && typeof result.prediction.scope === 'string');
  } else requireValue(result.prediction === null);
  requireValue(hash(result.stateHash) && hash(result.resultHash) && result.snapshot.stateHash === result.stateHash
    && result.snapshot.resultHash === result.resultHash && result.snapshot.phase === state.phase
    && result.snapshot.landmarkId === row && transportDigest(result.snapshot.binding) === transportDigest(result.binding));
  // Pinned entire payload: changes to measurements, predictions, labels, audit
  // bindings, snapshots or their claimed hashes cannot authorize themselves.
  requireValue(transportDigest(result) === PINS[`${state.phase}:${row}`]);
  if (request.action === 'advance') requireValue(state.phase === request.phase && row === request.landmarkId);
  if (request.action === 'reopen') requireValue(transportDigest(result.snapshot) === transportDigest(request.snapshot));
  function freeze(value) {
    if (value && typeof value === 'object') { Object.values(value).forEach(freeze); Object.freeze(value); }
    return value;
  }
  return freeze(result);
}

function validateObservedEvent(message, request) {
  if (message.event === 'result') {
    record(message, ['id', 'event', 'result']);
    validateObservedResult(message.result, request);
  } else if (message.event === 'error') {
    record(message, ['id', 'event', 'error']); record(message.error, ['code', 'message']);
    requireValue(typeof message.error.code === 'string' && message.error.code.length <= 128
      && typeof message.error.message === 'string' && message.error.message.length <= 4096);
  } else {
    // This small inspection operation does not produce progress payloads.
    requireValue(['started', 'cancelled'].includes(message.event)); record(message, ['id', 'event']);
  }
}

module.exports = { observedRequest, validateObservedResult, validateObservedEvent, transportDigest, PINS };
