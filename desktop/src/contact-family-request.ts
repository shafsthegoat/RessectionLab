import type {ContactFamilyRequest} from './contact-family-types.ts';

export function checkedContactFamilyRequest(value: unknown): ContactFamilyRequest {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Choose a generated layout, goal and method.');
  }
  const input = value as Record<string, unknown>;
  if (Object.keys(input).sort().join(',') !== 'fixture,goalId,layoutId,selector' ||
      input.fixture !== 'generated-public-contact-family-v2' ||
      typeof input.layoutId !== 'string' || !/^pcf-(?:0[0-9]|1[0-9]|2[0-3])$/.test(input.layoutId) ||
      typeof input.goalId !== 'string' || !['surface', 'deep'].includes(input.goalId) ||
      typeof input.selector !== 'string' || !['STOP', 'SEARCH', 'IL', 'RL', 'IL_TRAIN_REFIT'].includes(input.selector)) {
    throw new Error('Only the fixed generated family and named methods are supported.');
  }
  // Canonical role and released method availability belong to the owned backend.
  // A syntactically valid ID does not authorize held-out execution or a checkpoint.
  return {fixture: input.fixture, layoutId: input.layoutId,
    goalId: input.goalId as ContactFamilyRequest['goalId'],
    selector: input.selector as ContactFamilyRequest['selector']};
}
