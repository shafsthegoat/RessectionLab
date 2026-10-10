import type {ContactFamilyAvailability, ContactFamilyRequest, ContactFamilyResult} from './contact-family-types.ts';
import type {ResectionApi} from './types.ts';
import {checkedPublicContactAssets, contactNeed, metadataJson} from './public-contact-authority.ts';
import {checkedFamilyAuthority, FAMILY_SHAPE} from './contact-family-authority.ts';
import {checkedFamilyObservation} from './contact-family-observation.ts';
import {hydrateNativeEpisodeReplay} from './native-episode-replay.ts';
import {checkedNativeContactOutcome} from './native-contact-outcome.ts';
import {validateCaseDescriptor} from './case-data.ts';
import {sha256Bytes} from './source-integrity.ts';

export function checkedFamilyExecution(result: ContactFamilyResult, catalog: ContactFamilyAvailability, request: ContactFamilyRequest) {
  const e = result.episode, p = result.executionProvenance, author = e.learnedAuthorship;
  const refit = request.selector === 'IL_TRAIN_REFIT';
  if (refit) {
    const method = catalog.methods.IL_TRAIN_REFIT;
    contactNeed(Object.keys(result).sort().join(',') === 'case,episode,episodeCanonicalJson,executionProvenance,policyVariant' &&
      result.policyVariant === 'IL_TRAIN_REFIT' && e.selector === 'IL' && e.splitRole === 'TRAIN' && method?.available &&
      p?.version === 'generated-contact-train-refit-execution-v1', 'refit response is not the requested distinct TRAIN method.');
    const expectedKeys = ['version', 'variant', 'algorithm', 'layoutId', 'goalId', 'splitRole', 'experimentHash', 'familyHash',
      'releaseManifestSha256', 'fitResultSha256', 'rolloutResultSha256', 'independentAuditSha256', 'checkpointFileSha256',
      'architectureHash', 'parameterHash', 'trainingLineageHash', 'completedUpdates', 'statesPerUpdate',
      'inferenceOptimizerUpdates', 'ownedResultSha256', 'ownedSupervisionSha256'];
    contactNeed(author && Object.keys(p).sort().join(',') === expectedKeys.sort().join(',') && p.variant === 'IL_TRAIN_REFIT' &&
      p.algorithm === 'IL' && p.layoutId === e.layoutId && p.goalId === e.publicGoal.goalId && p.splitRole === 'TRAIN' &&
      p.familyHash === e.familyHash && p.experimentHash === method.experimentHash && p.experimentHash === author.experimentHash &&
      p.architectureHash === author.architectureHash && p.parameterHash === method.parameterHash && p.parameterHash === author.parameterHash &&
      p.trainingLineageHash === author.trainingLineageHash && p.checkpointFileSha256 === method.checkpointFileSha256 &&
      p.checkpointFileSha256 === author.checkpointFileSha256 && p.completedUpdates === 32 && p.statesPerUpdate === 40 &&
      p.inferenceOptimizerUpdates === 0 && method.releaseHash === 'sha256:' + p.releaseManifestSha256 &&
      p.fitResultSha256 === method.evidence?.fitResultSha256 && p.rolloutResultSha256 === method.evidence?.rolloutResultSha256 &&
      p.independentAuditSha256 === method.evidence?.independentAuditSha256 &&
      [p.releaseManifestSha256, p.fitResultSha256, p.rolloutResultSha256, p.independentAuditSha256, p.checkpointFileSha256,
        p.ownedResultSha256, p.ownedSupervisionSha256].every(hash => typeof hash === 'string' && /^[a-f0-9]{64}$/.test(hash)),
      'live TRAIN refit differs from its separate audited publication.');
    return;
  }
  contactNeed(result.policyVariant === undefined, 'original method cannot carry refit identity.');
  const learned = e.selector === 'IL' || e.selector === 'RL';
  contactNeed(Object.keys(result).sort().join(',') === (learned ?
    'case,episode,episodeCanonicalJson,executionProvenance' : 'case,episode,episodeCanonicalJson'), 'unexpected family response fields.');
  if (!learned) { contactNeed(p === undefined && author === null, 'nonlearned method claims live actor provenance.'); return; }
  contactNeed(p?.version === 'generated-contact-family-execution-v1', 'original method requires its paired release.');
  const expectedKeys = ['version', 'selector', 'layoutId', 'goalId', 'splitRole', 'experimentHash', 'familyHash',
    'releaseManifestSha256', 'pilotResultSha256', 'finalFreezeSha256', 'checkpointFileSha256', 'architectureHash',
    'parameterHash', 'trainingLineageHash', 'completedUpdates', 'inferenceOptimizerUpdates', 'ownedResultSha256', 'ownedSupervisionSha256'];
  contactNeed(p && author && Object.keys(p).sort().join(',') === expectedKeys.sort().join(',') &&
    p.version === 'generated-contact-family-execution-v1' && p.selector === e.selector &&
    p.layoutId === e.layoutId && p.goalId === e.publicGoal.goalId && p.splitRole === e.splitRole &&
    p.familyHash === e.familyHash && p.experimentHash === catalog.experimentHash &&
    p.experimentHash === author.experimentHash && p.architectureHash === author.architectureHash &&
    p.parameterHash === author.parameterHash && p.trainingLineageHash === author.trainingLineageHash &&
    p.checkpointFileSha256 === author.checkpointFileSha256 &&
    p.completedUpdates === 32 && p.inferenceOptimizerUpdates === 0 &&
    catalog.releaseHash === 'sha256:' + p.releaseManifestSha256 &&
    [p.releaseManifestSha256, p.pilotResultSha256, p.finalFreezeSha256, p.checkpointFileSha256,
      p.ownedResultSha256, p.ownedSupervisionSha256].every(hash => typeof hash === 'string' && /^[a-f0-9]{64}$/.test(hash)),
    'live learned response differs from released artifacts or owned worker.');
}

export async function hydrateContactFamilyEpisode(
  raw: ContactFamilyResult, api: ResectionApi, requested: ContactFamilyRequest, available: ContactFamilyAvailability,
) {
  const input = structuredClone(raw), request = structuredClone(requested), catalog = structuredClone(available);
  checkedPublicContactAssets(input);
  const e = input.episode, expected = checkedFamilyAuthority(e, request, catalog);
  checkedFamilyExecution(input, catalog, request);
  const digest = async (text: string) => 'sha256:' + await sha256Bytes(new TextEncoder().encode(text));
  contactNeed(await digest(expected.objectiveJson) === e.publicGoal.objectiveHash &&
    await digest(expected.declarationJson) === e.taskContract.detachedObservationBinding.declaration_hash,
    'family public objective/declaration digest changed.');
  if (e.learnedAuthorship) contactNeed(await digest(metadataJson(e.taskContract.checkpoint)) === e.learnedAuthorship.trainingLineageHash,
    'published training lineage does not match its digest.');
  contactNeed(input.case.metadata.family_hash === e.familyHash && input.case.metadata.layout_id === e.layoutId &&
    input.case.metadata.scope === 'public_generated_contact_learning_no_patient_transfer' &&
    input.case.workspaceSession === undefined, 'family source metadata or transient admission changed.');
  const geometry = await hydrateNativeEpisodeReplay(input, api, 2, FAMILY_SHAPE);
  const support = new Uint8Array(await api.readAsset(input.case.brainMask!.assetId)).slice();
  const observation = await checkedFamilyObservation(e, geometry.volume, support);
  const {voxelVolumeMm3} = validateCaseDescriptor(input.case);
  return {...geometry, episode: e, source: input.case, observation,
    executionProvenance: input.executionProvenance ?? null,
    policyVariant: input.policyVariant ?? null,
    ...checkedNativeContactOutcome(e, geometry.frames, voxelVolumeMm3)};
}
export type ContactFamilyView = Awaited<ReturnType<typeof hydrateContactFamilyEpisode>>;
