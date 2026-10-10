import type {ContactFamilyAvailability, ContactFamilyRequest} from './contact-family-types.ts';
import {contactNeed, contactSame} from './public-contact-authority.ts';
import {checkedContactFamilyRequest} from './contact-family-request.ts';
const HASH = /^sha256:[a-f0-9]{64}$/;
const digest = (value: unknown) => typeof value === 'string' && HASH.test(value);
const keys = (value: object) => Object.keys(value).sort().join(',');

export function checkedFamilyAvailability(raw: ContactFamilyAvailability): ContactFamilyAvailability {
  const value = structuredClone(raw);
  contactNeed(value && keys(value) === 'experimentHash,familyHash,fixture,layouts,methods,releaseHash,version' &&
    value.version === 'generated-public-contact-learning-availability-v1' &&
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
  contactNeed(value.methods && keys(value.methods) === 'IL,RL,SEARCH,STOP', 'missing fixed method slot.');
  for (const method of ['STOP', 'SEARCH', 'IL', 'RL'] as const) {
    const row = value.methods[method];
    contactNeed(row && keys(row) === 'available,reason' && typeof row.available === 'boolean' &&
      (row.reason === null || typeof row.reason === 'string' && row.reason.length > 0 && row.reason.length <= 1000) &&
      (row.available ? row.reason === null : row.reason !== null), 'method availability is ambiguous.');
  }
  contactNeed(value.methods.IL.available === value.methods.RL.available &&
    (value.releaseHash !== null || !value.methods.IL.available), 'unreleased learned method cannot be enabled.');
  return value;
}

export function requireInteractiveFamilyRequest(raw: unknown, catalog: ContactFamilyAvailability) {
  const request = checkedContactFamilyRequest(raw);
  const value = checkedFamilyAvailability(catalog);
  const layout = value.layouts.find(row => row.layoutId === request.layoutId);
  contactNeed(layout?.interactive === true && layout.role !== 'MEASUREMENT_EVAL', 'held-out layouts are unavailable for interactive execution.');
  contactNeed(value.methods[request.selector].available, value.methods[request.selector].reason ?? 'method unavailable.');
  return {request, layout, catalog: value};
}
