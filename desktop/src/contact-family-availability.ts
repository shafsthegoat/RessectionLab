import type {ContactFamilyAvailability, ContactFamilyRequest} from './contact-family-types.ts';
import {contactNeed, contactSame} from './public-contact-authority.ts';
import {checkedContactFamilyRequest} from './contact-family-request.ts';
const HASH = /^sha256:[a-f0-9]{64}$/;
const digest = (value: unknown) => typeof value === 'string' && HASH.test(value);
const keys = (value: object) => Object.keys(value).sort().join(',');

export function checkedFamilyAvailability(raw: ContactFamilyAvailability): ContactFamilyAvailability {
  const value = structuredClone(raw);
  contactNeed(value && keys(value) === 'experimentHash,familyHash,fixture,layouts,methods,releaseHash,version' &&
    ['generated-public-contact-learning-availability-v1', 'generated-public-contact-learning-availability-v2'].includes(value.version) &&
    value.fixture === 'generated-public-contact-family-v2' && digest(value.familyHash) &&
    (value.experimentHash === null || digest(value.experimentHash)) &&
    (value.releaseHash === null || digest(value.releaseHash)) &&
    (value.releaseHash === null) === (value.experimentHash === null),
    'family availability identity changed.');
  contactNeed(Array.isArray(value.layouts) && value.layouts.length === 24 &&
    new Set(value.layouts.map(row => row.layoutId)).size === 24, 'incomplete family layout inventory.');
  const counts = {TRAIN: 0, SELECT: 0, MEASUREMENT_EVAL: 0};
  for (const row of value.layouts) {
    contactNeed(keys(row) === 'goals,interactive,layoutId,role' &&
      typeof row.layoutId === 'string' && /^pcf-(?:0[0-9]|1[0-9]|2[0-3])$/.test(row.layoutId) &&
      Object.hasOwn(counts, row.role) && contactSame(row.goals, ['surface', 'deep']) &&
      row.interactive === (row.role === 'TRAIN' || row.role === 'SELECT'), 'layout role or interactive admission changed.');
    counts[row.role]++;
  }
  contactNeed(contactSame(counts, {TRAIN: 12, SELECT: 4, MEASUREMENT_EVAL: 8}), 'layout role denominator changed.');
  contactNeed(value.methods && keys(value.methods) === (value.version === 'generated-public-contact-learning-availability-v2' ? 'IL,IL_TRAIN_REFIT,RL,SEARCH,STOP' : 'IL,RL,SEARCH,STOP'), 'missing fixed method slot.');
  for (const method of ['STOP', 'SEARCH', 'IL', 'RL'] as const) {
    const row = value.methods[method];
    contactNeed(row && keys(row) === 'available,reason' && typeof row.available === 'boolean' &&
      (row.reason === null || typeof row.reason === 'string' && row.reason.length > 0 && row.reason.length <= 1000) &&
      (row.available ? row.reason === null : row.reason !== null), 'method availability is ambiguous.');
  }
  contactNeed(value.methods.IL.available === value.methods.RL.available &&
    (value.releaseHash !== null || !value.methods.IL.available), 'unreleased learned method cannot be enabled.');
  if (value.version === 'generated-public-contact-learning-availability-v2') {
    const row = value.methods.IL_TRAIN_REFIT;
    contactNeed(row && keys(row) === 'allowedRoles,available,checkpointFileSha256,evidence,experimentHash,knownTRAINOutcome,parameterHash,reason,releaseHash,trainingBudget' &&
      typeof row.available === 'boolean' && contactSame(row.allowedRoles, ['TRAIN']) &&
      (row.reason === null || typeof row.reason === 'string' && row.reason.length > 0 && row.reason.length <= 1000) &&
      (row.available ? row.reason === null : row.reason !== null), 'TRAIN refit availability is ambiguous.');
    if (row.releaseHash === null) {
      contactNeed(!row.available && row.experimentHash === null && row.parameterHash === null && row.checkpointFileSha256 === null &&
        row.evidence === null && row.trainingBudget === null && row.knownTRAINOutcome === null, 'unpublished refit carries artifact identity.');
    } else {
      contactNeed(row.releaseHash === 'sha256:68091a27acf63715f23e5a649de1f06dee56b2391e6bb6919fc8bbb36a2d8887' &&
        row.experimentHash === 'sha256:fd211e00dbe6dd1c9ed50a58d6c3b9f9f8896a0acd3a7815b2c52e4a8dc1014f' &&
        row.parameterHash === 'sha256:0bdd6713358937ac2c665ff700210b24eb6a88433f6f410fc87bb62f23f7fe20' &&
        row.checkpointFileSha256 === '5d7151397141c92ff82d8684814b9a0caed111f1809268bd448b8c1ea26d6bf7' &&
        row.evidence && keys(row.evidence) === 'fitResultSha256,independentAuditSha256,rolloutResultSha256' &&
        Object.values(row.evidence).every(hash => typeof hash === 'string' && /^[a-f0-9]{64}$/.test(hash)) &&
        contactSame(row.trainingBudget, {updates: 32, statesPerUpdate: 40, lossForwards: 1280, fixedReadoutForwards: 80}),
        'TRAIN refit artifact or extra training budget changed.');
      const outcome = row.knownTRAINOutcome;
      contactNeed(outcome && keys(outcome) === 'STOPOnly,goalContacts,meanReturn,savedSEARCHContacts,scope,tasks' &&
        outcome.tasks === 24 && outcome.goalContacts === 6 && outcome.savedSEARCHContacts === 16 && outcome.STOPOnly === 18 &&
        typeof outcome.meanReturn === 'number' && Math.abs(outcome.meanReturn - 0.05633333333333332) < 1e-12 &&
        outcome.scope === 'generated_TRAIN_native_results_no_heldout_claim', 'fixed TRAIN refit negative outcomes changed.');
    }
  }
  return value;
}

export function requireInteractiveFamilyRequest(raw: unknown, catalog: ContactFamilyAvailability) {
  const request = checkedContactFamilyRequest(raw);
  const value = checkedFamilyAvailability(catalog);
  const layout = value.layouts.find(row => row.layoutId === request.layoutId);
  contactNeed(layout?.interactive === true && layout.role !== 'MEASUREMENT_EVAL', 'held-out layouts are unavailable for interactive execution.');
  if (request.selector === 'IL_TRAIN_REFIT') contactNeed(layout.role === 'TRAIN', 'Full-teacher imitation is available only on TRAIN layouts.');
  const method = value.methods[request.selector];
  contactNeed(method?.available, method?.reason ?? 'method unavailable.');
  return {request, layout, catalog: value};
}
