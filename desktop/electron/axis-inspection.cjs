'use strict';

const { plainArgs } = require('./security.cjs');

const HASH = /^sha256:[a-f0-9]{64}$/;
const MODEL_HASH = /^[a-f0-9]{64}$/;
const MAX_INSPECTION_BYTES = 2 * 1024 * 1024;
const NATIVE_TOOLS = new Set(['native-fine-aspiration', 'native-wide-aspiration']);
const KEYS = ['caseHash', 'planningHash', 'routeId', 'routePlanningModelHash', 'toolIds',
  'acknowledgeNeighboringColumns', 'acknowledgeEstimatedSupport', 'expectedBindingHash'];

function validAnchorWindow(window) {
  const vector = value => Array.isArray(value) && value.length === 3 && Array.from(value).every(Number.isFinite);
  return window && typeof window === 'object' && !Array.isArray(window)
    && vector(window.center_mm) && vector(window.normal_inward)
    && Number.isFinite(window.radius_mm) && window.radius_mm > 0
    && typeof window.window_id === 'string' && window.window_id.length > 0;
}

function validNormalization(value) {
  return value && typeof value === 'object' && !Array.isArray(value)
    && Number.isFinite(value.normalAbsoluteTolerance) && value.normalAbsoluteTolerance >= 0
    && value.normalAbsoluteTolerance <= 4 * Number.EPSILON
    && Number.isFinite(value.normalMaximumDifference) && value.normalMaximumDifference >= 0
    && value.normalMaximumDifference <= value.normalAbsoluteTolerance;
}

function axisInspectionArgs(value) {
  plainArgs(value, KEYS);
  if (typeof value.caseHash !== 'string' || !HASH.test(value.caseHash)
      || typeof value.planningHash !== 'string' || !HASH.test(value.planningHash)
      || typeof value.routePlanningModelHash !== 'string' || !MODEL_HASH.test(value.routePlanningModelHash)
      || (value.expectedBindingHash !== undefined
        && (typeof value.expectedBindingHash !== 'string' || !HASH.test(value.expectedBindingHash))))
    throw new Error('Axis inspection requires exact source and model identities');
  if (typeof value.routeId !== 'string' || !value.routeId.trim() || value.routeId.length > 128
      || /[\u0000-\u001f\u007f]/.test(value.routeId))
    throw new Error('Axis inspection requires a selected route identifier');
  if (!Array.isArray(value.toolIds) || value.toolIds.length < 1 || value.toolIds.length > 2
      || Array.from(value.toolIds).some(id => typeof id !== 'string' || !NATIVE_TOOLS.has(id))
      || new Set(value.toolIds).size !== value.toolIds.length)
    throw new Error('Choose one or two distinct native research tools');
  if (value.acknowledgeNeighboringColumns !== true
      || typeof value.acknowledgeEstimatedSupport !== 'boolean')
    throw new Error('Axis inspection requires explicit research-model acknowledgments');
  // Retain a request snapshot while the engine prepares the read-only inspection.
  return { caseHash: value.caseHash, planningHash: value.planningHash,
    routeId: value.routeId, routePlanningModelHash: value.routePlanningModelHash,
    toolIds: [...value.toolIds], acknowledgeNeighboringColumns: true,
    acknowledgeEstimatedSupport: value.acknowledgeEstimatedSupport,
    ...(value.expectedBindingHash === undefined ? {} : { expectedBindingHash: value.expectedBindingHash }) };
}

async function inspectAxisPlanning(value, { request }) {
  const args = axisInspectionArgs(value);
  const result = await request('inspectAxisPlanning', args);
  if (result && Buffer.byteLength(JSON.stringify(result)) > MAX_INSPECTION_BYTES)
    throw new Error('Axis inspection response exceeds its local size limit');
  if (!result || typeof result !== 'object' || Array.isArray(result)
      || result.schemaVersion !== 1 || result.accessSource !== 'selected_route_window_only'
      || !validAnchorWindow(result.anchorWindowRas)
      || !validNormalization(result.anchorWindowNormalization)
      || ['caseHash', 'planningHash', 'routeId', 'routePlanningModelHash'].some(key => result[key] !== args[key])
      || !Array.isArray(result.requestedToolIds)
      || result.requestedToolIds.length !== args.toolIds.length
      || new Set(result.requestedToolIds).size !== args.toolIds.length
      || Array.from(result.requestedToolIds).some(id => !args.toolIds.includes(id))
      || !Object.hasOwn(result, 'inspection'))
    throw new Error('Axis inspection response does not match the requested source and model');
  // Numerical inspection contents are validated by the renderer's dedicated boundary.
  return result;
}

module.exports = { axisInspectionArgs, inspectAxisPlanning, MAX_INSPECTION_BYTES };
